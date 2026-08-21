from agents.base import describe_request, run_tool_agent
from verify import verify_hotels
from models import HotelResult, TripState
from tools.travel import search_hotels

SYSTEM = """You research real places to stay. Return up to 4 spanning price
tiers: cheapest, mid-range, splurge, plus a best-value pick.

TOOL — search_hotels(city, check_in, check_out, adults)
Returns, per property: name, price_per_night (for the WHOLE PARTY, not per
person), hotel_class, rating, type (hotel / vacation rental / etc), amenities.

THE TOOL CANNOT tell you:
- the neighbourhood or address. Only set `area` if the property NAME makes it
  unambiguous; otherwise use the city name. Never guess a district.
- room availability, cancellation terms, or booking links.
- whether breakfast/parking is included beyond what `amenities` lists.
Do not fill these in from your own knowledge of the city.

RULES:
- Report only properties the tool returned, at the rates it returned. If
  `options` is empty, return an empty list and say so in `reasoning`.
- `price_per_night_usd` is for the whole party.
- In notes: the property type, the listed amenities, and "live Google Hotels
  rate" — these are real current prices, not estimates."""


def hotel_agent(state: TripState) -> TripState:
    req = state["request"]
    return run_tool_agent(
        name="hotels",
        state=state,
        schema=HotelResult,
        system=SYSTEM,
        user=(
            f"{describe_request(req)}\n\n"
            f"Find real lodging in {req.destination} from "
            f"{req.start_date} to {req.end_date} for {req.travelers} guest(s)."
        ),
        tools=[search_hotels],
        temperature=0.1,
        verify=verify_hotels,
    )
