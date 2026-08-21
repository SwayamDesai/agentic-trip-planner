"""Points of interest via Overpass (OpenStreetMap) and Wikivoyage. Keyless.

Raw Overpass output is too noisy to plan from: an unfiltered tourism query on
Lisbon returns unnamed viewpoints and a zoo train alongside the national
museums. Two filters fix the signal:

    [name]      - drop unnamed nodes
    [wikidata]  - require a Wikidata entry, which is a decent proxy for
                  "notable enough that a visitor might care"

Overpass is a shared volunteer service, so results are cached aggressively and
the query asks for a bounded result set.
"""

import time

import requests
from langchain_core.tools import tool

from providers.cache import TTL_STATIC, cached
from tools.schemas import Place, validate_rows

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
WIKIVOYAGE_URL = "https://en.wikivoyage.org/w/api.php"
UA = {"User-Agent": "trip-planner/0.1 (github.com/local; dev use)"}

# Tag values that are technically tourism/historic but are not places you plan
# a day around — plaques, statues, street art.
_SKIP = {"memorial", "artwork", "yes", "monument"}

CATEGORY_FILTERS = {
    "sights": '[tourism~"attraction|viewpoint|museum|gallery|zoo|theme_park"]',
    "museums": '[tourism~"museum|gallery"]',
    "historic": '[historic~"castle|monastery|church|ruins|fort|city_gate|palace"]',
    "food": '[amenity~"restaurant|cafe"]',
}


def _bbox(lat: float, lon: float, km: float) -> tuple[float, float, float, float]:
    """Rough degree box around a point. Fine at city scale."""
    dlat = km / 111.0
    dlon = km / 85.0  # conservative for mid latitudes
    return (lat - dlat, lon - dlon, lat + dlat, lon + dlon)


# Overpass is a shared volunteer service that sheds load with 429s and 504s
# under pressure — observed live during development. A single attempt turns a
# transient outage into "this city has no attractions", so retry briefly before
# giving up. Deliberately few, short attempts: the agent is already slow, and
# hammering a free community service to succeed would be the wrong fix.
_OVERPASS_ATTEMPTS = 3
_OVERPASS_BACKOFF = (1.0, 3.0)


def _overpass(query: str):
    """POST to Overpass, retrying transient server-side failures."""
    last: Exception | None = None
    for attempt in range(_OVERPASS_ATTEMPTS):
        try:
            resp = requests.post(
                OVERPASS_URL, data={"data": query}, headers=UA, timeout=90
            )
            # 429 = rate limited, 5xx = overloaded. Both are worth another try;
            # a 4xx from a malformed query never is.
            if resp.status_code == 429 or resp.status_code >= 500:
                resp.raise_for_status()
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            last = exc
            status = getattr(getattr(exc, "response", None), "status_code", None)
            retryable = status is None or status == 429 or status >= 500
            if not retryable or attempt == _OVERPASS_ATTEMPTS - 1:
                raise
            time.sleep(_OVERPASS_BACKOFF[min(attempt, len(_OVERPASS_BACKOFF) - 1)])
    raise last  # pragma: no cover - loop always returns or raises


@tool
def find_places(
    lat: float,
    lon: float,
    category: str = "sights",
    radius_km: float = 6.0,
    limit: int = 25,
) -> dict:
    """Find real, notable points of interest near a location from OpenStreetMap.

    USE WHEN: you need actual places to put in an itinerary. Call once per
    category you need — a plan mixing sights and food needs two calls.

    DO NOT USE WHEN: you want opening hours, entry prices, ratings, or
    booking — it returns NONE of these, so any cost you write is your own
    estimate. It also has no notion of whether a place is worth a whole day.

    Results are filtered to named places with a Wikidata entry, so they are
    real attractions rather than street furniture. The "food" category skips
    that filter and is therefore noisier.

    Args:
        lat: Latitude of the city centre.
        lon: Longitude of the city centre.
        category: One of "sights", "museums", "historic", "food".
        radius_km: Search radius in kilometres.
        limit: Maximum number of places to return.

    Returns a `places` list of {name, kind, lat, lon}. Schedule only these —
    never add an attraction from memory, however famous.
    """
    filt = CATEGORY_FILTERS.get(category)
    if filt is None:
        return {
            "error": f"unknown category {category!r}",
            "valid": sorted(CATEGORY_FILTERS),
        }

    s, w, n, e = _bbox(lat, lon, radius_km)
    box = f"{s:.4f},{w:.4f},{n:.4f},{e:.4f}"
    # 'food' has few wikidata entries, so relax that filter for it only
    notability = "" if category == "food" else "[wikidata]"
    query = (
        f"[out:json][timeout:40];"
        f"nwr{filt}[name]{notability}({box});"
        f"out center {min(limit * 3, 120)};"
    )

    def fetch():
        resp = _overpass(query)
        out = []
        for el in resp.json().get("elements", []):
            tags = el.get("tags", {})
            kind = tags.get("tourism") or tags.get("historic") or tags.get("amenity")
            if kind in _SKIP:
                continue
            centre = el.get("center") or el
            if centre.get("lat") is None:
                continue
            # OSM's `name` is in the local language, which for Kyoto yields
            # "京都タワー". Prefer the English name when the data has one, and
            # keep the local name alongside it so it stays usable on signage.
            local = tags["name"]
            english = tags.get("name:en")
            out.append(
                {
                    "name": english or local,
                    **({"local_name": local} if english and english != local else {}),
                    "kind": kind,
                    "lat": round(centre["lat"], 5),
                    "lon": round(centre["lon"], 5),
                }
            )
        validated, dropped = validate_rows(out, Place)
        return {
            "places": validated[:limit],
            "source": "OpenStreetMap via Overpass",
            **({"omitted_malformed": dropped} if dropped else {}),
        }

    return cached(
        "places",
        f"{lat:.3f},{lon:.3f},{category},{radius_km},{limit}",
        fetch,
        ttl=TTL_STATIC,
    )


@tool
def city_guide(city: str) -> dict:
    """Get a real travel-guide summary of a city from Wikivoyage.

    USE WHEN: you need orientation — which districts exist, how the city is
    laid out — before grouping activities geographically.

    DO NOT USE WHEN: you need specific venues (use find_places), prices, or
    anything time-sensitive. The text is prose, not structured data, and may
    be out of date.

    Args:
        city: City name, e.g. "Lisbon".
    """

    def fetch():
        resp = requests.get(
            WIKIVOYAGE_URL,
            params={
                "action": "query",
                "prop": "extracts",
                "titles": city,
                "explaintext": 1,
                "exintro": 1,
                "format": "json",
                "redirects": 1,
            },
            headers=UA,
            timeout=30,
        )
        resp.raise_for_status()
        pages = resp.json().get("query", {}).get("pages", {})
        page = next(iter(pages.values()), {})
        extract = (page.get("extract") or "").strip()
        if not extract:
            return {"error": f"no Wikivoyage entry for {city!r}"}
        return {"city": page.get("title", city), "summary": extract[:1500]}

    return cached("cityguide", city.lower().strip(), fetch, ttl=TTL_STATIC)
