"""Flight/hotel tool parsing and backend fallback ordering.

These carry the highest-stakes regressions in the project: a price-unit slip
produces a number that looks entirely plausible ($1104 vs $2208) while being
wrong by 2x, and nothing downstream can detect it.
"""

import pytest
import requests

from tools import travel


@pytest.fixture(autouse=True)
def offline_geocode(monkeypatch):
    """`search_flights` now geocodes the cities to validate the airport codes,
    so these tests need coordinates without a network call."""
    cities = {
        "chicago": (41.88, -87.63),
        "lisbon": (38.72, -9.14),
        "granada": (37.18, -3.60),
        "seville": (37.39, -5.98),
    }

    def fake(place):
        hit = cities.get((place or "").strip().lower())
        if hit is None:
            return {"error": f"could not geocode {place!r}"}
        return {"name": place, "lat": hit[0], "lon": hit[1]}

    monkeypatch.setattr("tools.airports.geocode", fake)


# --- fast-flights parsing ---


def test_price_normalised_to_per_person(monkeypatch, fast_flights_result):
    """Regression: fast-flights prices the whole party, SerpApi prices per head.

    Mixing the two units silently doubles or halves every fare.
    """
    monkeypatch.setattr(travel, "_fast_flights_raw", lambda *a, **k: fast_flights_result)
    rows = travel._fast_flights("ORD", "LIS", "2026-09-10", "2026-09-13", adults=2)
    assert rows[0]["price_usd"] == 1104
    assert rows[0]["price_total_usd"] == 2208


def test_single_traveller_price_unchanged(monkeypatch, fast_flights_result):
    monkeypatch.setattr(travel, "_fast_flights_raw", lambda *a, **k: fast_flights_result)
    rows = travel._fast_flights("ORD", "LIS", "2026-09-10", None, adults=1)
    assert rows[0]["price_usd"] == rows[0]["price_total_usd"] == 2208


def test_zero_travellers_does_not_divide_by_zero(monkeypatch, fast_flights_result):
    monkeypatch.setattr(travel, "_fast_flights_raw", lambda *a, **k: fast_flights_result)
    rows = travel._fast_flights("ORD", "LIS", "2026-09-10", None, adults=0)
    assert rows[0]["price_usd"] == 2208


def test_arrival_taken_from_final_leg(monkeypatch, fast_flights_result):
    """Regression: reading legs[0].arrival gives the layover, not the destination."""
    monkeypatch.setattr(travel, "_fast_flights_raw", lambda *a, **k: fast_flights_result)
    row = travel._fast_flights("ORD", "LIS", "2026-09-10", "2026-09-13", adults=2)[0]
    assert row["departure_airport"] == "ORD"
    assert row["arrival_airport"] == "LIS", "must be LIS, not the DUB layover"
    assert row["departure_at"] == "2026-09-10 21:00"
    assert row["arrival_at"] == "2026-09-11 21:35"


def test_stops_and_duration_span_all_legs(monkeypatch, fast_flights_result):
    monkeypatch.setattr(travel, "_fast_flights_raw", lambda *a, **k: fast_flights_result)
    row = travel._fast_flights("ORD", "LIS", "2026-09-10", "2026-09-13", adults=2)[0]
    assert row["stops"] == 1, "2 hops == 1 stop"
    assert row["duration_minutes"] == 630, "sum of both legs"
    assert len(row["connections"]) == 2


def test_round_trip_flag_reflects_return_date(monkeypatch, fast_flights_result):
    monkeypatch.setattr(travel, "_fast_flights_raw", lambda *a, **k: fast_flights_result)
    rt = travel._fast_flights("ORD", "LIS", "2026-09-10", "2026-09-13", 2)[0]
    ow = travel._fast_flights("ORD", "LIS", "2026-09-10", None, 2)[0]
    assert rt["price_covers"] == "round trip"
    assert ow["price_covers"] == "one way"


def test_entries_without_legs_are_skipped(monkeypatch):
    import types

    monkeypatch.setattr(
        travel,
        "_fast_flights_raw",
        lambda *a, **k: [types.SimpleNamespace(type="XX", price=1, airlines=[], flights=[])],
    )
    assert travel._fast_flights("ORD", "LIS", "2026-09-10", None, 1) == []


# --- serpapi parsing ---


def test_serp_arrival_from_final_leg(monkeypatch, serp_flights_payload):
    monkeypatch.setattr(travel, "_serpapi", lambda engine, extra: serp_flights_payload)
    row = travel._serp_flights("ORD", "LIS", "2026-09-10", "2026-09-13")[0]
    assert row["arrival_airport"] == "LIS"
    assert row["stops"] == 1
    assert row["price_usd"] == 1104


def test_serp_and_fastflights_agree_on_units(monkeypatch, serp_flights_payload):
    """Regression: SerpApi rows omitted price_total_usd entirely, so the two
    backends handed the agent different shapes. Validation caught it."""
    monkeypatch.setattr(travel, "_serpapi", lambda engine, extra: serp_flights_payload)
    row = travel._serp_flights("ORD", "LIS", "2026-09-10", "2026-09-13", adults=2)[0]
    assert row["price_usd"] == 1104, "per person"
    assert row["price_total_usd"] == 2208, "whole party"


def test_serp_missing_key_returns_error_not_silence(monkeypatch):
    """Regression: an unset key once looked identical to 'no flights found'."""
    monkeypatch.delenv("SERPAPI_KEY", raising=False)
    out = travel._serpapi("google_flights", {})
    assert "error" in out and "SERPAPI_KEY" in out["error"]


# --- backend fallback ordering ---


def _stub(monkeypatch, *, fast=None, serp=None, tp=None):
    monkeypatch.setattr(travel, "_fast_flights", lambda *a, **k: fast or [])
    monkeypatch.setattr(travel, "_serp_flights", lambda *a, **k: serp or [])
    monkeypatch.setattr(travel, "_tp_flights", lambda *a, **k: tp or [])


ROW = [{"airline": "X", "price_usd": 100}]


def _search(**kw):
    # cities are mandatory now: without them the route check cannot run, so the
    # tool refuses rather than searching an unverified route
    args = {
        "origin_iata": "ORD",
        "destination_iata": "LIS",
        "departure_date": "2026-09-10",
        "return_date": "2026-09-13",
        "travelers": 2,
        "origin_city": "Chicago",
        "destination_city": "Lisbon",
    }
    args.update(kw)
    return travel.search_flights.invoke(args)


def test_keyless_backend_preferred(monkeypatch):
    """fast-flights is unmetered, so it must win before quota is spent."""
    _stub(monkeypatch, fast=ROW, serp=ROW, tp=ROW)
    assert _search()["source"] == "google_flights_direct"


def test_falls_back_to_serpapi(monkeypatch):
    _stub(monkeypatch, fast=[], serp=ROW, tp=ROW)
    assert _search()["source"] == "google_flights_serpapi"


def test_month_cache_is_last_resort(monkeypatch):
    """Real fares for the WRONG dates rank below live fares for the right ones."""
    _stub(monkeypatch, fast=[], serp=[], tp=ROW)
    out = _search()
    assert out["source"] == "travelpayouts_cache_month"
    assert "OTHER dates" in out["note"]


def test_all_backends_empty_reports_none(monkeypatch):
    _stub(monkeypatch)
    out = _search()
    assert out["source"] == "none" and out["options"] == []


def test_scraper_exception_does_not_abort(monkeypatch):
    """The scraper hits an undocumented endpoint; a crash must fall through."""

    def boom(*a, **k):
        raise RuntimeError("google changed the protobuf")

    monkeypatch.setattr(travel, "_fast_flights", boom)
    monkeypatch.setattr(travel, "_serp_flights", lambda *a, **k: ROW)
    monkeypatch.setattr(travel, "_tp_flights", lambda *a, **k: [])
    assert _search()["source"] == "google_flights_serpapi"


def test_serp_network_error_falls_through(monkeypatch):
    def boom(*a, **k):
        raise requests.ConnectionError("down")

    monkeypatch.setattr(travel, "_fast_flights", lambda *a, **k: [])
    monkeypatch.setattr(travel, "_serp_flights", boom)
    monkeypatch.setattr(travel, "_tp_flights", lambda *a, **k: ROW)
    assert _search()["source"] == "travelpayouts_cache_month"


def test_traveller_count_is_part_of_cache_key(monkeypatch):
    """Per-person prices differ by party size, so the key must include it."""
    seen = []

    def spy(o, d, dep, ret, adults):
        seen.append(adults)
        return [{"airline": "X", "price_usd": 100 * adults}]

    monkeypatch.setattr(travel, "_fast_flights", spy)
    _search(travelers=1)
    _search(travelers=3)
    assert seen == [1, 3], "a shared key would have served run 1's result to run 2"


# --- hotels ---


def test_hotel_parsing(monkeypatch):
    monkeypatch.setattr(
        travel,
        "_serpapi",
        lambda engine, extra: {
            "properties": [
                {
                    "name": "Hotel A",
                    "rate_per_night": {"lowest": "$67"},
                    "hotel_class": "3-star hotel",
                    "overall_rating": 4.3,
                    "type": "hotel",
                    "amenities": ["Wi-Fi", "Pool", "Gym", "Bar", "Spa", "Extra"],
                }
            ]
        },
    )
    out = travel.search_hotels.invoke(
        {"city": "Lisbon", "check_in": "2026-09-10", "check_out": "2026-09-13"}
    )
    assert out["source"] == "google_hotels_live"
    row = out["options"][0]
    assert row["name"] == "Hotel A" and row["rating"] == 4.3
    assert len(row["amenities"]) == 5, "amenities are capped to keep tokens down"


def test_hotel_api_error_surfaces(monkeypatch):
    monkeypatch.setattr(travel, "_serpapi", lambda e, x: {"error": "quota exceeded"})
    out = travel.search_hotels.invoke(
        {"city": "Lisbon", "check_in": "2026-09-10", "check_out": "2026-09-13"}
    )
    assert out["options"] == [] and "quota" in out["error"]


def test_missing_city_names_are_refused(monkeypatch):
    """Regression: the city args were optional, so the wrong-city airport check
    silently did nothing whenever the model omitted them."""
    monkeypatch.setattr(
        travel, "_fast_flights", lambda *a, **k: pytest.fail("should not search")
    )
    out = travel.search_flights.invoke({
        "origin_iata": "ORD", "destination_iata": "LIS",
        "departure_date": "2026-12-01", "travelers": 2,
    })
    assert out["error_kind"] == "missing_argument"
