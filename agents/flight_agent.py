from agents.base import describe_request, run_tool_agent
from verify import verify_flights
from models import FlightResult, TripState
from tools.airports import find_airports
from tools.travel import search_flights
from providers import prompts


def flight_agent(state: TripState) -> TripState:
    req = state["request"]
    return run_tool_agent(
        name="flight",
        state=state,
        schema=FlightResult,
        system=prompts.get("flight").text,
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
