from agents.base import describe_request, run_tool_agent
from verify import verify_weather
from models import TripState, WeatherResult
from tools.geo import geocode_place
from tools.weather import get_weather

SYSTEM = """You report the expected weather for each day of the trip.

TOOLS
1. geocode_place(place) -> lat/lon. Call this first.
2. get_weather(lat, lon, start_date, end_date) -> one entry per day with
   condition, high_c, low_c, precipitation_chance, plus a `source` field.

`source` is the critical field:
  "forecast"        -> a real forecast. You may call it a forecast.
  "climate_normals" -> multi-year averages for those calendar dates, used
                       because the trip is beyond the ~16-day forecast window.
                       This is NOT a forecast. Say so explicitly in
                       packing_advice. precipitation_chance here means "share of
                       past years with rain on that date", not today's odds.

THE TOOLS CANNOT give you: hourly detail, severe-weather warnings, a real
forecast beyond ~16 days, or sea/UV conditions. Do not supply these yourself.

RULES:
- Report the tool's numbers as returned. Do not round heavily, and do not add
  or drop days.
- Temperatures are already Celsius.
- End with short packing advice grounded in the actual numbers."""


def weather_agent(state: TripState) -> TripState:
    req = state["request"]
    return run_tool_agent(
        name="weather",
        state=state,
        schema=WeatherResult,
        system=SYSTEM,
        user=(
            f"{describe_request(req)}\n\n"
            f"Get the real daily outlook for {req.destination} "
            f"from {req.start_date} to {req.end_date}."
        ),
        tools=[geocode_place, get_weather],
        temperature=0.1,
        verify=verify_weather,
    )
