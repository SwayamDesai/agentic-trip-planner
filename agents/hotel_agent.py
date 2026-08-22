from agents.base import describe_request, run_tool_agent
from verify import verify_hotels
from models import HotelResult, TripState
from tools.travel import search_hotels
from providers import prompts


def hotel_agent(state: TripState) -> TripState:
    req = state["request"]
    return run_tool_agent(
        name="hotels",
        state=state,
        schema=HotelResult,
        system=prompts.get("hotels").text,
        user=(
            f"{describe_request(req)}\n\n"
            f"Find real lodging in {req.destination} from "
            f"{req.start_date} to {req.end_date} for {req.travelers} guest(s)."
        ),
        tools=[search_hotels],
        temperature=0.1,
        verify=verify_hotels,
    )
