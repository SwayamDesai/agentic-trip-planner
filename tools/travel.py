"""Real flight and hotel prices.

Flights use two backends in order:

    1. Travelpayouts /aviasales/v3/prices_for_dates - free, unmetered, ~1s.
       It serves a cache of fares real Aviasales users searched for, so
       coverage is uneven: JFK-LON returns several fares, ORD-LIS returned
       exactly one, and an exact-date query on a quiet route returns nothing
       at all. Tried first precisely because it costs nothing.
    2. SerpApi google_flights - live scrape, ~8s, metered at 250/month.
       Only called when Travelpayouts comes back empty, which keeps the
       metered quota for hotels.

Hotels are SerpApi google_hotels only. Travelpayouts' hotel API (Hotellook)
is gone — engine.hotellook.com 404s on every documented path, so there is no
free specialised hotel source left to fall back to.

Every result carries a `source` field. The agents surface it, because "live
Google Flights" and "a cached fare from someone else's search last week" are
different claims and should not be presented identically.
"""

import os

import requests
from dotenv import load_dotenv
from langchain_core.tools import tool

from providers.cache import TTL_FLIGHTS, TTL_HOTELS, cached
from tools.airports import validate_route
from tools.schemas import (
    FlightOffer,
    HotelOffer,
    empty_result,
    parse_money,
    tool_error,
    validate_rows,
)

# This module reads credentials directly, so it loads .env itself rather than
# relying on another import having done so. Without this the tools silently
# behave as if every source were empty when used outside the full graph.
load_dotenv()

TP_URL = "https://api.travelpayouts.com/aviasales/v3/prices_for_dates"
SERPAPI_URL = "https://serpapi.com/search"


def _tp_flights(origin: str, dest: str, depart: str, ret: str | None) -> list[dict]:
    token = os.getenv("TRAVELPAYOUTS_TOKEN")
    if not token:
        return []

    params = {
        "origin": origin,
        "destination": dest,
        "departure_at": depart,
        "currency": "usd",
        "sorting": "price",
        "limit": 10,
        "one_way": "false" if ret else "true",
    }
    if ret:
        params["return_at"] = ret

    resp = requests.get(
        TP_URL, params=params, headers={"X-Access-Token": token}, timeout=40
    )
    resp.raise_for_status()
    rows = resp.json().get("data") or []

    return [
        {
            "airline": r.get("airline"),
            "flight_number": r.get("flight_number"),
            "price_usd": r.get("price"),
            "departure_at": r.get("departure_at"),
            "return_at": r.get("return_at"),
            "stops": r.get("transfers"),
            "duration_minutes": r.get("duration"),
        }
        for r in rows
    ]


def _fast_flights_raw(
    origin: str, dest: str, depart: str, ret: str | None, adults: int
):
    """The network half of the fast-flights call, split out so the parsing half
    is testable offline. Returns the library's raw result list."""
    from fast_flights import FlightQuery, Passengers, create_query, get_flights

    legs = [FlightQuery(date=depart, from_airport=origin, to_airport=dest)]
    if ret:
        legs.append(FlightQuery(date=ret, from_airport=dest, to_airport=origin))

    return get_flights(
        create_query(
            flights=legs,
            trip="round-trip" if ret else "one-way",
            seat="economy",
            passengers=Passengers(adults=max(adults, 1)),
            currency="USD",
        )
    )


def _fast_flights(
    origin: str, dest: str, depart: str, ret: str | None, adults: int
) -> list[dict]:
    """Google Flights via its own protobuf endpoint. Keyless and unmetered.

    Returns per-leg detail, so connections are real rather than inferred from a
    stop count. Prices come back as the TOTAL for the whole party, unlike
    SerpApi which quotes per person — normalised here so both backends feed the
    agent the same unit.
    """
    result = _fast_flights_raw(origin, dest, depart, ret, adults)

    def _stamp(dt) -> str:
        y, m, d = dt.date
        hh, mm = dt.time
        return f"{y:04d}-{m:02d}-{d:02d} {hh:02d}:{mm:02d}"

    out = []
    for entry in list(result):
        hops = entry.flights or []
        if not hops:
            continue
        total = entry.price
        out.append(
            {
                "airline": ", ".join(entry.airlines or []) or entry.type,
                "price_usd": round(total / max(adults, 1)),
                "price_total_usd": total,
                "departure_at": _stamp(hops[0].departure),
                "departure_airport": hops[0].from_airport.code,
                "arrival_at": _stamp(hops[-1].arrival),
                "arrival_airport": hops[-1].to_airport.code,
                "stops": len(hops) - 1,
                "duration_minutes": sum(h.duration or 0 for h in hops),
                "connections": [
                    f"{h.from_airport.code}->{h.to_airport.code} "
                    f"{_stamp(h.departure)} to {_stamp(h.arrival)}"
                    for h in hops
                ],
                "price_covers": "round trip" if ret else "one way",
            }
        )
    validated, _ = validate_rows(out, FlightOffer)
    validated.sort(key=lambda r: r["price_usd"])
    return validated[:10]


def _serpapi(engine: str, extra: dict) -> dict:
    key = os.getenv("SERPAPI_KEY")
    if not key:
        return {"error": "SERPAPI_KEY not set in .env"}

    resp = requests.get(
        SERPAPI_URL,
        params={"engine": engine, "api_key": key, "hl": "en", "gl": "us", **extra},
        timeout=90,
    )
    resp.raise_for_status()
    return resp.json()


def _serp_flights(
    origin: str, dest: str, depart: str, ret: str | None, adults: int = 1
) -> list[dict]:
    """SerpApi returns a PER-PERSON fare (we search a single adult), whereas
    fast-flights returns a party total. Both are normalised here so the agent
    never compares mismatched units."""
    payload = {
        "departure_id": origin,
        "arrival_id": dest,
        "outbound_date": depart,
        "currency": "USD",
        "type": "1" if ret else "2",  # 1 = round trip, 2 = one way
    }
    if ret:
        payload["return_date"] = ret

    data = _serpapi("google_flights", payload)
    out = []
    # best_flights is Google's own shortlist; other_flights is the long tail
    for group in ("best_flights", "other_flights"):
        for entry in data.get(group) or []:
            legs = entry.get("flights") or []
            if not legs:
                continue
            # legs[0] is the first hop, legs[-1] the last, so the itinerary's
            # true arrival is the final leg's arrival, not the first leg's.
            dep = legs[0].get("departure_airport") or {}
            arr = legs[-1].get("arrival_airport") or {}
            out.append(
                {
                    "airline": legs[0].get("airline"),
                    "flight_number": legs[0].get("flight_number"),
                    "price_usd": entry.get("price"),
                    "price_total_usd": (entry.get("price") or 0) * max(adults, 1),
                    "departure_at": dep.get("time"),
                    "departure_airport": dep.get("id"),
                    "arrival_at": arr.get("time"),
                    "arrival_airport": arr.get("id"),
                    "stops": len(legs) - 1,
                    "duration_minutes": entry.get("total_duration"),
                    # Round-trip searches price the whole trip, but Google only
                    # returns outbound itineraries in this response; the return
                    # leg's times need a second call keyed on departure_token,
                    # which would spend another metered search.
                    "price_covers": "round trip" if ret else "one way",
                }
            )
    validated, _ = validate_rows(out, FlightOffer)
    return validated[:10]


@tool
def search_flights(
    origin_iata: str,
    destination_iata: str,
    departure_date: str,
    return_date: str = "",
    # Defaults are required by the signature (they follow `return_date`), but
    # empty cities are rejected at runtime — see the check below.
    travelers: int = 1,
    origin_city: str = "",
    destination_city: str = "",
) -> dict:
    """Get real flight prices for a route.

    USE WHEN: you need fares for the trip. One call covers the whole round
    trip — do not call it separately for the outbound and return.

    DO NOT USE WHEN: you want return-leg departure/arrival times (not
    returned), baggage fees, seat availability, refund rules, or booking
    links. Never fill those in from your own knowledge.

    Args:
        origin_iata: 3-letter IATA code, e.g. "ORD" for Chicago. You must
            convert city names to codes yourself.
        destination_iata: 3-letter IATA code, e.g. "LIS" for Lisbon.
        departure_date: YYYY-MM-DD.
        return_date: YYYY-MM-DD for a round trip, or "" for one way.
        travelers: Number of travellers, used to price per person correctly.
        origin_city: The origin city name. REQUIRED — the codes are checked
            against it, so a wrong-city code is caught before a search is spent.
        destination_city: The destination city name. REQUIRED, same reason.

    Returns `options` cheapest-first, each with price_usd (PER PERSON),
    price_total_usd (whole party), outbound departure/arrival, connections,
    and `price_covers`. Also returns `source`:
      google_flights_direct / google_flights_serpapi  live, requested dates.
      travelpayouts_cache_month  real fares for OTHER dates that month —
        say the requested date had no live fare and treat as indicative only.

    An empty `options` means no real fare was found. Report that plainly
    rather than substituting a remembered price.
    """
    # City names are required, not optional: when they were optional the model
    # could omit them and the wrong-city check silently did nothing, which is
    # exactly the failure this guard exists to prevent.
    if not origin_city.strip() or not destination_city.strip():
        return tool_error(
            "origin_city and destination_city are required so the airport "
            "codes can be checked against them. Call again with both.",
            "missing_argument",
        )

    # Validate before spending a request. A wrong-but-real code returns valid
    # fares for the wrong route, which nothing downstream can detect.
    route_problems = validate_route(
        origin_iata, destination_iata, origin_city, destination_city
    )
    if route_problems:
        return tool_error(
            " ".join(route_problems), "bad_airport_code", retryable=False
        ) | {"guidance": "Fix the airport code and call again with the correct one."}

    ret = return_date or None
    key = f"{origin_iata},{destination_iata},{departure_date},{ret},{travelers}"

    def fetch():
        # 1. keyless and unmetered, and the richest data of the three
        try:
            rows = _fast_flights(
                origin_iata, destination_iata, departure_date, ret, travelers
            )
        except Exception:  # noqa: BLE001 - undocumented endpoint; fall through
            rows = []
        if rows:
            return {
                "options": rows,
                "source": "google_flights_direct",
                "note": "Live Google Flights prices for the requested dates.",
            }

        # 2. metered (250/month), so only when the scraper breaks
        try:
            rows = _serp_flights(
                origin_iata, destination_iata, departure_date, ret, travelers
            )
        except requests.RequestException:
            rows = []
        if rows:
            return {
                "options": rows,
                "source": "google_flights_serpapi",
                "note": "Live Google Flights prices for the requested dates.",
            }

        # 3. last resort: real fares, but for other dates in the same month
        if len(departure_date) == 10:
            try:
                rows = _tp_flights(
                    origin_iata, destination_iata, departure_date[:7], None
                )
            except requests.RequestException:
                rows = []
            if rows:
                return {
                    "options": rows,
                    "source": "travelpayouts_cache_month",
                    "note": (
                        "No fare found for the requested date from any live "
                        "source. These are real cached fares for OTHER dates "
                        "in the same month — a price indication only."
                    ),
                }

        # A genuine absence, distinguished from a failure: no backend errored,
        # there simply are no fares. Said explicitly so the agent can report it
        # instead of guessing, and cached briefly rather than re-queried.
        return empty_result(
            "flights",
            f"No fares were found for {origin_iata}-{destination_iata} on "
            f"{departure_date}. The route may not be served on these dates.",
        )

    return cached("flights", key, fetch, ttl=TTL_FLIGHTS)


@tool
def search_hotels(
    city: str, check_in: str, check_out: str, adults: int = 2
) -> dict:
    """Get real hotel prices and ratings for a city and date range.

    USE WHEN: you need places to stay for the trip. One call covers the whole
    stay and returns every tier, so call it once.

    DO NOT USE WHEN: you need the neighbourhood or address (NOT returned — do
    not guess a district), room availability, cancellation terms, or booking
    links. Do not use it to look up one specific named hotel.

    Args:
        city: City name, e.g. "Lisbon".
        check_in: YYYY-MM-DD.
        check_out: YYYY-MM-DD.
        adults: Number of adults.

    Returns `options` with name, price_per_night (WHOLE PARTY, not per
    person), hotel_class, rating, type and amenities. These are live rates —
    report them as returned, and do not describe them as estimates. An empty
    `options` means no live rates were found; say so rather than estimating.
    """
    key = f"{city.lower()},{check_in},{check_out},{adults}"

    def fetch():
        try:
            data = _serpapi(
                "google_hotels",
                {
                    "q": city,
                    "check_in_date": check_in,
                    "check_out_date": check_out,
                    "adults": adults,
                    "currency": "USD",
                },
            )
        except requests.RequestException as exc:
            return {"options": [], "source": "none", "error": str(exc)}

        if "error" in data:
            return {"options": [], "source": "none", "error": data["error"]}

        rows = []
        for p in (data.get("properties") or [])[:12]:
            rows.append(
                {
                    "name": p.get("name"),
                    # arrives as a string like "$148"; parsed here rather than
                    # left for the model to interpret
                    "price_per_night": parse_money(
                        (p.get("rate_per_night") or {}).get("lowest")
                    ),
                    "hotel_class": p.get("hotel_class"),
                    "rating": p.get("overall_rating"),
                    "type": p.get("type"),
                    "amenities": (p.get("amenities") or [])[:5],
                }
            )

        out, dropped = validate_rows(rows, HotelOffer)
        if not out:
            return empty_result(
                "hotels",
                f"No bookable rooms were returned for {city} on those dates.",
            )

        note = "Live Google Hotels rates for the requested dates."
        if dropped:
            # surfaced rather than hidden: silently dropping listings would
            # read as "this city has little availability"
            note += (
                f" {dropped} listing(s) omitted for having no usable nightly "
                f"rate."
            )
        return {"options": out, "source": "google_hotels_live", "note": note}

    return cached("hotels", key, fetch, ttl=TTL_HOTELS)
