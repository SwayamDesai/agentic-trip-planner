"""Weather via Open-Meteo. Keyless, no signup.

Open-Meteo's forecast only reaches ~16 days ahead. Trips are usually planned
further out than that, so this module has two modes and always reports which
one produced the numbers:

    forecast  - real forecast, trip starts inside the forecast window
    normals   - the same calendar dates averaged over the previous N years,
                pulled from the historical archive

Labelling matters: presenting a multi-year average as a "forecast" would be a
plausible-looking lie, and the agent prompt relies on this field to phrase its
output honestly.
"""

from datetime import date, timedelta

import requests
from langchain_core.tools import tool

from providers.cache import TTL_FORECAST, TTL_STATIC, cached
from tools.schemas import DayWeather, validate_rows

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

DAILY_VARS = "temperature_2m_max,temperature_2m_min,precipitation_sum,weather_code"
NORMALS_YEARS = 5
# Open-Meteo's forecast reaches ~16 days; past that only normals exist.
FORECAST_HORIZON_DAYS = 16

# WMO weather codes -> short description.
WMO = {
    0: "clear",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "freezing fog",
    51: "light drizzle",
    53: "drizzle",
    55: "heavy drizzle",
    61: "light rain",
    63: "rain",
    65: "heavy rain",
    71: "light snow",
    73: "snow",
    75: "heavy snow",
    80: "rain showers",
    81: "rain showers",
    82: "violent rain showers",
    95: "thunderstorm",
    96: "thunderstorm with hail",
    99: "thunderstorm with hail",
}


def _get(url: str, params: dict) -> dict:
    resp = requests.get(url, params=params, timeout=40)
    resp.raise_for_status()
    return resp.json()


def _daterange(start: str, end: str) -> list[str]:
    d0 = date.fromisoformat(start)
    d1 = date.fromisoformat(end)
    return [(d0 + timedelta(days=i)).isoformat() for i in range((d1 - d0).days + 1)]


def _try_forecast(lat: float, lon: float, start: str, end: str) -> dict | None:
    """Real forecast, or None if the dates fall outside the supported window."""
    try:
        data = _get(
            FORECAST_URL,
            {
                "latitude": lat,
                "longitude": lon,
                "daily": DAILY_VARS + ",precipitation_probability_max",
                "timezone": "auto",
                "start_date": start,
                "end_date": end,
            },
        )
    except requests.HTTPError:
        return None  # out of range; caller falls back to normals

    daily = data.get("daily")
    if not daily or not daily.get("time"):
        return None

    days = [
            {
                "date": t,
                "condition": WMO.get(code, "unknown"),
                "high_c": hi,
                "low_c": lo,
                "precipitation_chance": prob if prob is not None else 0,
            }
        for t, hi, lo, code, prob in zip(
            daily["time"],
            daily["temperature_2m_max"],
            daily["temperature_2m_min"],
            daily["weather_code"],
            daily.get("precipitation_probability_max") or [None] * len(daily["time"]),
        )
    ]
    validated, dropped = validate_rows(days, DayWeather)
    if not validated:
        return None  # nothing usable; caller falls back to normals
    return {
        "source": "forecast",
        "note": "Real forecast from Open-Meteo."
        + (f" {dropped} day(s) omitted as malformed." if dropped else ""),
        "days": validated,
    }


def _normals(lat: float, lon: float, start: str, end: str) -> dict:
    """Average the same calendar dates over the previous NORMALS_YEARS years."""
    wanted = _daterange(start, end)
    this_year = date.fromisoformat(start).year

    # month-day -> list of (high, low, precip_mm) across sampled years
    buckets: dict[str, list[tuple[float, float, float]]] = {d[5:]: [] for d in wanted}

    for offset in range(1, NORMALS_YEARS + 1):
        year = this_year - offset
        try:
            data = _get(
                ARCHIVE_URL,
                {
                    "latitude": lat,
                    "longitude": lon,
                    "start_date": f"{year}-{start[5:]}",
                    "end_date": f"{year}-{end[5:]}",
                    "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
                    "timezone": "auto",
                },
            )
        except requests.HTTPError:
            continue

        daily = data.get("daily") or {}
        for t, hi, lo, pr in zip(
            daily.get("time", []),
            daily.get("temperature_2m_max", []),
            daily.get("temperature_2m_min", []),
            daily.get("precipitation_sum", []),
        ):
            if t[5:] in buckets and hi is not None and lo is not None:
                buckets[t[5:]].append((hi, lo, pr or 0.0))

    days = []
    for d in wanted:
        samples = buckets.get(d[5:], [])
        if not samples:
            continue
        n = len(samples)
        high = sum(s[0] for s in samples) / n
        low = sum(s[1] for s in samples) / n
        wet = sum(1 for s in samples if s[2] >= 1.0)
        days.append(
            {
                "date": d,
                "condition": "often wet" if wet * 2 > n else "usually dry",
                "high_c": round(high, 1),
                "low_c": round(low, 1),
                # share of sampled years with >=1mm rain on this date
                "precipitation_chance": round(100 * wet / n),
                "years_sampled": n,
            }
        )

    days, dropped = validate_rows(days, DayWeather)
    return {
        "source": "climate_normals",
        "note": (
            f"NOT a forecast. Averaged from the same calendar dates over the last "
            f"{NORMALS_YEARS} years, because the trip is beyond Open-Meteo's "
            f"~16-day forecast window. precipitation_chance is the share of those "
            f"years with measurable rain on that date."
        ),
        "days": days,
    }


@tool
def get_weather(lat: float, lon: float, start_date: str, end_date: str) -> dict:
    """Get real day-by-day weather for a location over a date range.

    USE WHEN: you need the outlook for the trip dates. One call covers the
    whole range — do not call it per day.

    DO NOT USE WHEN: you want hourly conditions, storm warnings, or sea and UV
    data; it returns none of these. Do not call it for dates outside the trip.

    IMPORTANT — check the `source` field before describing the result:
      "forecast"        real forecast, safe to call a forecast.
      "climate_normals" multi-year averages for those calendar dates, returned
                        because the trip is past the ~16-day forecast window.
                        This is NOT a forecast and must not be called one.

    Args:
        lat: Latitude.
        lon: Longitude.
        start_date: First day, YYYY-MM-DD.
        end_date: Last day, YYYY-MM-DD.
    """
    key = f"{lat:.3f},{lon:.3f},{start_date},{end_date}"

    # Two namespaces, because these are different kinds of data sharing one
    # function. A forecast is perishable; climate normals are averages of years
    # already past and never change. Caching both for 6h meant re-fetching five
    # years of archive data every six hours for nothing.
    within_forecast_window = (
        date.fromisoformat(start_date) - date.today()
    ).days <= FORECAST_HORIZON_DAYS

    if within_forecast_window:
        def fetch_forecast():
            return _try_forecast(lat, lon, start_date, end_date) or _normals(
                lat, lon, start_date, end_date
            )

        result = cached("weather_forecast", key, fetch_forecast, ttl=TTL_FORECAST)
        if result.get("source") == "forecast":
            return result
        # fell back to normals: re-serve from the long-lived namespace instead
        # of holding immutable data on a 6h lifetime

    return cached(
        "weather_normals",
        key,
        lambda: _normals(lat, lon, start_date, end_date),
        ttl=TTL_STATIC,
    )
