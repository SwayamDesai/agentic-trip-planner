"""Weather tool: forecast vs climate-normals, and the honesty of the labelling.

The `source` field is load-bearing. Presenting a multi-year average as a
forecast would be a confident, plausible lie, so the distinction is tested
rather than assumed.
"""

import requests

from tools import weather


def _forecast_payload():
    return {
        "daily": {
            "time": ["2026-09-10", "2026-09-11"],
            "temperature_2m_max": [25.0, 26.1],
            "temperature_2m_min": [17.9, 17.7],
            "weather_code": [0, 95],
            "precipitation_probability_max": [10, 80],
        }
    }


def test_forecast_is_labelled_forecast(monkeypatch):
    monkeypatch.setattr(weather, "_get", lambda url, params: _forecast_payload())
    out = weather._try_forecast(38.7, -9.1, "2026-09-10", "2026-09-11")
    assert out["source"] == "forecast"
    assert out["days"][0]["condition"] == "clear"
    assert out["days"][1]["condition"] == "thunderstorm"
    assert out["days"][1]["precipitation_chance"] == 80


def test_unknown_wmo_code_does_not_crash(monkeypatch):
    payload = _forecast_payload()
    payload["daily"]["weather_code"] = [0, 12345]
    monkeypatch.setattr(weather, "_get", lambda url, params: payload)
    out = weather._try_forecast(38.7, -9.1, "2026-09-10", "2026-09-11")
    assert out["days"][1]["condition"] == "unknown"


def test_out_of_range_forecast_returns_none(monkeypatch):
    """Open-Meteo 400s past ~16 days; that must signal a fallback, not an error."""

    def boom(url, params):
        raise requests.HTTPError("400 out of allowed range")

    monkeypatch.setattr(weather, "_get", boom)
    assert weather._try_forecast(38.7, -9.1, "2027-09-10", "2027-09-11") is None


def test_missing_precipitation_field_defaults_to_zero(monkeypatch):
    payload = _forecast_payload()
    del payload["daily"]["precipitation_probability_max"]
    monkeypatch.setattr(weather, "_get", lambda url, params: payload)
    out = weather._try_forecast(38.7, -9.1, "2026-09-10", "2026-09-11")
    assert out["days"][0]["precipitation_chance"] == 0


def test_normals_average_across_years(monkeypatch):
    """Two sampled years, one wet: precipitation_chance is the SHARE of years."""
    calls = []

    def fake_get(url, params):
        calls.append(params["start_date"])
        year = params["start_date"][:4]
        rain = 5.0 if year == "2025" else 0.0
        return {
            "daily": {
                "time": [f"{year}-09-10"],
                "temperature_2m_max": [20.0],
                "temperature_2m_min": [10.0],
                "precipitation_sum": [rain],
            }
        }

    monkeypatch.setattr(weather, "_get", fake_get)
    monkeypatch.setattr(weather, "NORMALS_YEARS", 2)
    out = weather._normals(38.7, -9.1, "2026-09-10", "2026-09-10")

    assert out["source"] == "climate_normals"
    assert "NOT a forecast" in out["note"]
    day = out["days"][0]
    assert day["date"] == "2026-09-10", "reported under the TRIP year, not the sample year"
    assert day["high_c"] == 20.0
    assert day["precipitation_chance"] == 50, "1 of 2 sampled years was wet"
    assert day["years_sampled"] == 2
    assert len(calls) == 2


def test_normals_condition_flips_on_majority_wet(monkeypatch):
    def fake_get(url, params):
        year = params["start_date"][:4]
        return {
            "daily": {
                "time": [f"{year}-09-10"],
                "temperature_2m_max": [20.0],
                "temperature_2m_min": [10.0],
                "precipitation_sum": [5.0],
            }
        }

    monkeypatch.setattr(weather, "_get", fake_get)
    monkeypatch.setattr(weather, "NORMALS_YEARS", 2)
    assert weather._normals(38.7, -9.1, "2026-09-10", "2026-09-10")["days"][0][
        "condition"
    ] == "often wet"


def test_a_failed_year_is_skipped_not_fatal(monkeypatch):
    def fake_get(url, params):
        if params["start_date"].startswith("2025"):
            raise requests.HTTPError("no data")
        year = params["start_date"][:4]
        return {
            "daily": {
                "time": [f"{year}-09-10"],
                "temperature_2m_max": [20.0],
                "temperature_2m_min": [10.0],
                "precipitation_sum": [0.0],
            }
        }

    monkeypatch.setattr(weather, "_get", fake_get)
    monkeypatch.setattr(weather, "NORMALS_YEARS", 2)
    out = weather._normals(38.7, -9.1, "2026-09-10", "2026-09-10")
    assert out["days"][0]["years_sampled"] == 1


def test_tool_routes_near_dates_to_the_forecast_namespace(monkeypatch):
    """Dates inside the ~16-day horizon may have a real forecast, so they go to
    the short-lived cache."""
    from datetime import date, timedelta

    monkeypatch.setattr(
        weather, "_try_forecast", lambda *a: {"source": "forecast", "days": [1]}
    )
    monkeypatch.setattr(
        weather, "_normals", lambda *a: {"source": "climate_normals", "days": [2]}
    )
    soon = (date.today() + timedelta(days=4)).isoformat()
    out = weather.get_weather.invoke(
        {"lat": 1.0, "lon": 2.0, "start_date": soon, "end_date": soon}
    )
    assert out["source"] == "forecast"


def test_far_dates_skip_the_forecast_entirely(monkeypatch):
    """Beyond the horizon a forecast cannot exist, so do not even ask — and
    cache the result as immutable rather than perishable."""
    from datetime import date, timedelta

    asked = []
    monkeypatch.setattr(
        weather, "_try_forecast", lambda *a: asked.append(1) or {"source": "forecast"}
    )
    monkeypatch.setattr(
        weather, "_normals", lambda *a: {"source": "climate_normals", "days": [2]}
    )
    far = (date.today() + timedelta(days=90)).isoformat()
    out = weather.get_weather.invoke(
        {"lat": 3.0, "lon": 4.0, "start_date": far, "end_date": far}
    )
    assert out["source"] == "climate_normals"
    assert asked == [], "no point asking for a forecast that cannot exist"


def test_near_date_falling_back_lands_in_the_normals_namespace(monkeypatch):
    """A near date with no forecast available must not pin immutable normals to
    a 6-hour lifetime."""
    from datetime import date, timedelta

    monkeypatch.setattr(weather, "_try_forecast", lambda *a: None)
    monkeypatch.setattr(
        weather, "_normals", lambda *a: {"source": "climate_normals", "days": [2]}
    )
    soon = (date.today() + timedelta(days=5)).isoformat()
    out = weather.get_weather.invoke(
        {"lat": 5.0, "lon": 6.0, "start_date": soon, "end_date": soon}
    )
    assert out["source"] == "climate_normals"


def test_daterange_inclusive():
    assert weather._daterange("2026-09-10", "2026-09-12") == [
        "2026-09-10",
        "2026-09-11",
        "2026-09-12",
    ]
