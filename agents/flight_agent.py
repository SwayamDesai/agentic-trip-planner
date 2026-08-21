from agents.base import describe_request, run_tool_agent
from verify import verify_flights
from models import FlightResult, TripState
from tools.airports import find_airports
from tools.travel import search_flights

SYSTEM = """You research real flight options. Return 3-5, cheapest first.

TOOLS
1. find_airports(city) -> real IATA codes serving that city, nearest first.
   Call this for BOTH cities first. Do NOT recall codes from memory: a wrong
   code prices a real route between the wrong places and the fare looks
   perfectly plausible.
2. search_flights(origin_iata, destination_iata, departure_date, return_date,
   travelers, origin_city, destination_city) — always pass the city names too,
   so the codes are checked against them.
It returns, per option: airline, price_usd (PER PERSON), price_total_usd (whole
party), departure_at/arrival_at, departure_airport/arrival_airport, stops,
duration_minutes, connections (each hop with times), price_covers, source.

THE TOOL CANNOT tell you:
- return-leg times. A round-trip `price_usd` covers the return, but the times
  shown are OUTBOUND only. Never state return times.
- baggage fees, seat availability, refund rules, or booking links.
Do not supply any of these from your own knowledge.

RULES:
- Report only options the tool returned. If `options` is empty, return an empty
  list and say so in `reasoning`. Never substitute a remembered price.
- `price_usd` in your output is per person. Say "round trip" or "one way" in
  notes, matching `price_covers`.
- Render duration_minutes as "8h 10m". Put the connection airports in notes
  when stops > 0.
- `source` tells you what the numbers are — state it in notes:
    google_flights_direct / google_flights_serpapi -> live, requested dates
    travelpayouts_cache_month -> real fares for OTHER dates that month; say
      the requested date had no live fare and treat these as indicative only."""


def flight_agent(state: TripState) -> TripState:
    req = state["request"]
    return run_tool_agent(
        name="flight",
        state=state,
        schema=FlightResult,
        system=SYSTEM,
        user=(
            f"{describe_request(req)}\n\n"
            f"Find real flights from {req.origin} to {req.destination}, "
            f"departing {req.start_date}, returning {req.end_date}, "
            f"for {req.travelers} traveler(s)."
        ),
        tools=[find_airports, search_flights],
        max_rounds=5,
        temperature=0.1,
        verify=verify_flights,
    )
