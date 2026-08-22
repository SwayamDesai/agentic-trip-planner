from agents.base import describe_request, run_tool_agent
from verify import verify_weather
from models import TripState, WeatherResult
from tools.geo import geocode_place
from tools.weather import get_weather
from providers import prompts


def weather_agent(state: TripState) -> TripState:
    req = state["request"]
    return run_tool_agent(
        name="weather",
        state=state,
        schema=WeatherResult,
        system=prompts.get("weather").text,
        user=(
            f"{describe_request(req)}\n\n"
            f"Get the real daily outlook for {req.destination} "
            f"from {req.start_date} to {req.end_date}."
        ),
        tools=[geocode_place, get_weather],
        temperature=0.1,
        verify=verify_weather,
    )
