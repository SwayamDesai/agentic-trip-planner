"""Shared fixtures.

Every test here runs offline. Nothing calls an LLM or a travel API: those are
metered on free tiers, and a suite you cannot afford to run is a suite that
does not get run.
"""

import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Make real network access impossible.

    A test that quietly reaches a live API burns metered free-tier quota and
    turns the suite into something too expensive to run often. This turns any
    such slip into a loud failure instead of a slow pass — which is exactly
    how the checkpointer test was caught doing 75s of real LLM calls.

    Tests that need HTTP stub the specific module attribute they use
    (e.g. `places.requests.post`), which replaces the attribute and so is
    unaffected by this.
    """
    import requests.sessions

    def blocked(self, method, url, *a, **k):
        raise AssertionError(
            f"test attempted real network access: {method} {url}. "
            "Stub the call instead."
        )

    monkeypatch.setattr(requests.sessions.Session, "request", blocked)


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    """Point the on-disk cache at a temp dir for every test.

    Without this, tests would read the developer's real cache and pass or fail
    depending on what was run manually beforehand.
    """
    from providers import cache

    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path / "cache")
    return tmp_path / "cache"


@pytest.fixture
def trip():
    from models import TripRequest

    return TripRequest(
        origin="Chicago",
        destination="Lisbon",
        start_date="2026-09-10",
        end_date="2026-09-13",
        travelers=2,
        budget_usd=3000,
        preferences=["food", "history"],
    )


# --- fakes shaped like the fast_flights return value ---


def _dt(y, m, d, hh, mm):
    return types.SimpleNamespace(date=(y, m, d), time=(hh, mm))


def _airport(code):
    return types.SimpleNamespace(code=code, name=f"{code} airport")


def hop(frm, to, dep, arr, duration):
    return types.SimpleNamespace(
        from_airport=_airport(frm),
        to_airport=_airport(to),
        departure=_dt(*dep),
        arrival=_dt(*arr),
        duration=duration,
    )


@pytest.fixture
def fast_flights_result():
    """One 2-hop itinerary priced for the whole party.

    Mirrors the real shape: `price` is the total for all passengers, which is
    the unit mismatch that makes normalisation worth testing.
    """
    entry = types.SimpleNamespace(
        type="EI",
        price=2208,  # party total for 2 adults => 1104 per person
        airlines=["Aer Lingus"],
        flights=[
            hop("ORD", "DUB", (2026, 9, 10, 21, 0), (2026, 9, 11, 10, 35), 455),
            hop("DUB", "LIS", (2026, 9, 11, 18, 40), (2026, 9, 11, 21, 35), 175),
        ],
    )
    return [entry]


@pytest.fixture
def serp_flights_payload():
    """Shape of a SerpApi google_flights response, trimmed to what we parse."""
    return {
        "best_flights": [
            {
                "price": 1104,
                "total_duration": 630,
                "flights": [
                    {
                        "airline": "Aer Lingus",
                        "flight_number": "EI 124",
                        "departure_airport": {"id": "ORD", "time": "2026-09-10 21:00"},
                        "arrival_airport": {"id": "DUB", "time": "2026-09-11 10:35"},
                    },
                    {
                        "airline": "Aer Lingus",
                        "flight_number": "EI columns",
                        "departure_airport": {"id": "DUB", "time": "2026-09-11 18:40"},
                        "arrival_airport": {"id": "LIS", "time": "2026-09-11 21:35"},
                    },
                ],
            }
        ],
        "other_flights": [],
    }
