"""Airport codes from a bundled dataset, not from model recall.

A wrong IATA code is the worst kind of error this system can make: it prices a
real route between the wrong places, and the fare looks entirely plausible.
Nothing downstream can detect it. "Granada" mistyped as SVQ returns Seville
fares that are perfectly valid and completely useless.

OSM was tried first and rejected: an Overpass query around Chicago returned
Muskegon and Grissom Air Reserve Base ahead of O'Hare, and a second call came
back non-JSON under load. Ranking noise plus flakiness, for something that is
fundamentally a static lookup.

So the data is bundled (`airportsdata`, ~7,900 IATA airports, offline) and used
two ways:

    find_airports()  a tool, so the model LOOKS UP a code instead of recalling
                     one. This is the primary fix.
    validate_route() a deterministic check inside search_flights, for when the
                     model skips the lookup and guesses anyway.
"""

from functools import lru_cache
from math import asin, cos, radians, sin, sqrt
from typing import Optional

from langchain_core.tools import tool

from tools.geo import geocode
from tools.schemas import tool_error

# A code may legitimately sit well outside its city: Frankfurt-Hahn is ~110km
# from Frankfurt, Stansted ~48km from London, Beauvais ~75km from Paris. Set
# the bar where a genuinely different city is implied, not merely a distant one.
PLAUSIBLE_KM = 150.0
SEARCH_RADIUS_KM = 150.0


@lru_cache(maxsize=1)
def _dataset() -> dict:
    import airportsdata

    return airportsdata.load("IATA")


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres."""
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = (
        sin(dlat / 2) ** 2
        + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    )
    return 2 * 6371.0 * asin(sqrt(a))


def lookup(iata: str) -> Optional[dict]:
    """Resolve an IATA code, or None if no such code exists."""
    if not iata or len(iata.strip()) != 3:
        return None
    return _dataset().get(iata.strip().upper())


# The dataset has no field for "carries scheduled passenger service", so a
# distance-only ranking offered RAF Northolt and Biggin Hill above Heathrow for
# London. Name patterns are the honest available filter.
_NOT_COMMERCIAL = (
    "raf ", "raf-", "air base", "air force", " afb", "afb ", "naval",
    "army", "military", "aerodrome club", "gliding", "airstrip",
)


def is_commercial(name: str) -> bool:
    lowered = f" {(name or '').lower()} "
    return not any(marker in lowered for marker in _NOT_COMMERCIAL)


def airports_near(lat: float, lon: float, radius_km: float, limit: int = 6) -> list[dict]:
    """IATA airports within `radius_km`, nearest first.

    Military and general-aviation fields are excluded: they hold IATA codes but
    no traveller can fly to them, and offering one as an option is worse than
    offering nothing.
    """
    found = []
    for code, record in _dataset().items():
        if not is_commercial(record.get("name", "")):
            continue
        distance = haversine_km(lat, lon, record["lat"], record["lon"])
        if distance <= radius_km:
            found.append(
                {
                    "iata": code,
                    "name": record["name"],
                    "city": record["city"],
                    "distance_km": round(distance, 1),
                }
            )
    found.sort(key=lambda r: r["distance_km"])
    return found[:limit]


@tool
def find_airports(city: str) -> dict:
    """Find the real IATA airport codes serving a city.

    USE WHEN: you need an airport code. Always call this before search_flights
    rather than recalling a code from memory — a wrong code silently prices the
    wrong route.

    DO NOT USE WHEN: you already looked up this city in this conversation.
    It returns no fares, schedules or airline information.

    Args:
        city: City name, e.g. "Granada" or "Chicago".

    Returns `airports`, nearest first, each with iata, name and distance_km.
    Large cities have several — prefer the nearest major one unless the
    traveller asked otherwise.
    """
    location = geocode(city)
    if "error" in location or location.get("lat") is None:
        return tool_error(
            f"Could not locate {city!r}, so its airports cannot be found.",
            "not_found",
        )

    airports = airports_near(location["lat"], location["lon"], SEARCH_RADIUS_KM)
    if not airports:
        return tool_error(
            f"No airport with an IATA code within {SEARCH_RADIUS_KM:.0f}km of "
            f"{city!r}. It may need to be reached via a larger nearby city.",
            "no_results",
        )
    return {"city": location["name"], "airports": airports}


def validate_route(
    origin_iata: str,
    destination_iata: str,
    origin_city: str = "",
    destination_city: str = "",
) -> list[str]:
    """Check that both codes exist and, if cities are given, that they match.

    Returns human-readable problems, each naming the correct codes, so a model
    reading the error can fix its own call.
    """
    problems: list[str] = []

    for label, code, city in (
        ("origin", origin_iata, origin_city),
        ("destination", destination_iata, destination_city),
    ):
        record = lookup(code)
        if record is None:
            suggestion = ""
            if city:
                location = geocode(city)
                if location.get("lat") is not None:
                    near = airports_near(location["lat"], location["lon"], SEARCH_RADIUS_KM)
                    if near:
                        suggestion = (
                            f" Airports near {city}: "
                            + ", ".join(f"{a['iata']} ({a['name']})" for a in near[:3])
                        )
            problems.append(f"{label} {code!r} is not a real IATA code.{suggestion}")
            continue

        if not city:
            continue

        location = geocode(city)
        if location.get("lat") is None:
            continue

        distance = haversine_km(
            location["lat"], location["lon"], record["lat"], record["lon"]
        )
        if distance > PLAUSIBLE_KM:
            near = airports_near(location["lat"], location["lon"], SEARCH_RADIUS_KM)
            options = ", ".join(f"{a['iata']} ({a['name']})" for a in near[:3])
            problems.append(
                f"{label} {code} is {record['name']} in {record['city']}, "
                f"{distance:.0f}km from {city} — that is a different place."
                + (f" Use one of: {options}" if options else "")
            )

    return problems
