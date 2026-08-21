"""Geocoding, including the rate limit Nominatim's usage policy requires."""

import pytest

from tools import geo


class _Resp:
    def __init__(self, payload):
        self._p = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._p


HIT = [{"display_name": "Lisboa, Portugal", "lat": "38.7077507", "lon": "-9.1365919"}]


def test_geocode_parses_and_casts(monkeypatch):
    monkeypatch.setattr(geo.requests, "get", lambda *a, **k: _Resp(HIT))
    out = geo.geocode_place.invoke({"place": "Lisbon"})
    assert out["name"] == "Lisboa, Portugal"
    assert isinstance(out["lat"], float) and out["lat"] == pytest.approx(38.7077507)


def test_no_result_returns_error(monkeypatch):
    monkeypatch.setattr(geo.requests, "get", lambda *a, **k: _Resp([]))
    assert "error" in geo.geocode_place.invoke({"place": "Nowhereton"})


def test_repeat_lookups_are_cached(monkeypatch):
    calls = []

    def spy(*a, **k):
        calls.append(1)
        return _Resp(HIT)

    monkeypatch.setattr(geo.requests, "get", spy)
    geo.geocode_place.invoke({"place": "Lisbon"})
    geo.geocode_place.invoke({"place": "  LISBON "})
    assert len(calls) == 1, "policy asks clients to cache; key is normalised"


def test_calls_are_throttled_to_one_per_second(monkeypatch):
    """Nominatim's usage policy caps clients at 1 request/sec.

    The fan-out runs agents in threads, so the throttle has to serialise
    rather than merely sleep.
    """
    slept = []
    monkeypatch.setattr(geo.time, "sleep", lambda s: slept.append(s))
    monkeypatch.setattr(geo.requests, "get", lambda *a, **k: _Resp(HIT))
    monkeypatch.setattr(geo, "_last_call", geo.time.perf_counter())

    geo._throttled_get("http://x", {})
    assert slept and 0 < slept[0] <= 1.05, f"expected a sub-second wait, got {slept}"


def test_no_wait_when_last_call_is_old(monkeypatch):
    slept = []
    monkeypatch.setattr(geo.time, "sleep", lambda s: slept.append(s))
    monkeypatch.setattr(geo.requests, "get", lambda *a, **k: _Resp(HIT))
    monkeypatch.setattr(geo, "_last_call", geo.time.perf_counter() - 10)

    geo._throttled_get("http://x", {})
    assert slept == [], "no need to wait if the last call was long ago"


def test_user_agent_is_sent(monkeypatch):
    """Nominatim rejects clients that do not identify themselves."""
    seen = {}

    def spy(url, params=None, headers=None, timeout=None):
        seen["headers"] = headers
        return _Resp(HIT)

    monkeypatch.setattr(geo.requests, "get", spy)
    monkeypatch.setattr(geo, "_last_call", 0.0)
    geo._throttled_get("http://x", {})
    assert "User-Agent" in seen["headers"]
