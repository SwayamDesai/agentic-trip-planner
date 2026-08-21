"""Trip identity and persistence."""

import pytest

from models import TripRequest
from providers import memory


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "DB_PATH", tmp_path / "trips.sqlite")


def _req(**kw):
    base = dict(
        origin="Chicago", destination="Lisbon",
        start_date="2026-09-10", end_date="2026-09-13",
        travelers=2, budget_usd=3000, preferences=["food"],
    )
    base.update(kw)
    return TripRequest(**base)


def test_same_trip_same_id():
    assert memory.thread_id(_req()) == memory.thread_id(_req())


def test_id_is_case_and_whitespace_insensitive():
    a = memory.thread_id(_req(origin="Chicago"))
    b = memory.thread_id(_req(origin="  chicago "))
    assert a == b, "trivial input differences must resume, not fork"


def test_preference_order_does_not_change_id():
    a = memory.thread_id(_req(preferences=["food", "history"]))
    b = memory.thread_id(_req(preferences=["history", "food"]))
    assert a == b


@pytest.mark.parametrize(
    "field,value",
    [
        ("destination", "Porto"),
        ("start_date", "2026-09-11"),
        ("end_date", "2026-09-14"),
        ("travelers", 3),
        ("budget_usd", 4000),
        ("preferences", ["nightlife"]),
    ],
)
def test_any_meaningful_change_forks_the_trip(field, value):
    """A different trip must not resume another trip's results."""
    assert memory.thread_id(_req()) != memory.thread_id(_req(**{field: value}))


def test_forget_on_missing_db_is_safe():
    assert memory.forget(_req()) == 0


def test_list_trips_on_missing_db_is_empty():
    assert memory.list_trips() == []


def test_checkpointer_roundtrip_and_forget():
    """A saved trip is listed, then removed.

    Every agent key is pre-populated so the skip guards in `agents.base` fire
    and no model is invoked — that is the whole point of resume.
    """
    from models import (
        FlightResult, HotelResult, ItineraryResult, PlacesResult, WeatherResult,
    )
    from orchestrator import build_graph

    saver = memory.get_checkpointer()
    config = {"configurable": {"thread_id": memory.thread_id(_req())}}

    graph = build_graph(saver)
    out = graph.invoke(
        {
            "request": _req(),
            "errors": [],
            "flight": FlightResult(options=[]),
            "hotels": HotelResult(options=[]),
            "weather": WeatherResult(daily=[], packing_advice="x"),
            "places": PlacesResult(city="Lisbon", candidates=[]),
            "itinerary": ItineraryResult(days=[]),
        },
        config,
    )
    assert out["plan"], "synthesize should still render"

    trips = memory.list_trips()
    assert len(trips) == 1
    assert trips[0]["thread_id"] == memory.thread_id(_req())
    assert memory.forget(_req()) > 0
    assert memory.list_trips() == []


def test_resume_carries_prior_results_and_resets_errors(monkeypatch):
    """plan_trip seeds the graph with what earlier runs produced.

    Stale error text must NOT carry over: an agent that has since succeeded
    would otherwise still be reported as failing.
    """
    import orchestrator
    from models import FlightResult, WeatherResult

    prior = {
        "flight": FlightResult(options=[]),
        "weather": None,                       # failed last time
        "errors": ["weather: RateLimit (stale)"],
    }

    class FakeGraph:
        def get_state(self, config):
            return type("S", (), {"values": prior})()

        def invoke(self, seed, config):
            captured.update(seed)
            return {"plan": "ok"}

    captured = {}
    monkeypatch.setattr(orchestrator, "get_checkpointer", lambda: object())
    monkeypatch.setattr(orchestrator, "build_graph", lambda cp=None: FakeGraph())

    orchestrator.plan_trip(_req())

    assert captured["flight"] is prior["flight"], "success is carried forward"
    assert "weather" not in captured, "a failed agent is left unset so it re-runs"
    # None, not []: the channel is append-reduced, so `[]` appends nothing and
    # would leave the checkpointed value in place. See models.accumulate.
    assert captured["errors"] is None, "stale errors are cleared"


def test_fresh_mode_ignores_saved_state(monkeypatch):
    import orchestrator

    used = {}

    class FakeGraph:
        def invoke(self, seed, config=None):
            used["seed"] = seed
            used["config"] = config
            return {"plan": "ok"}

    monkeypatch.setattr(orchestrator, "build_graph", lambda cp=None: FakeGraph())
    monkeypatch.setattr(
        orchestrator, "get_checkpointer",
        lambda: pytest.fail("--fresh must not touch the checkpointer"),
    )

    orchestrator.plan_trip(_req(), remember=False)
    assert used["config"] is None
    # run_id identifies this run's metrics collector and trace
    assert set(used["seed"]) == {
        "run_id", "request", "errors", "warnings", "evidence",
    }
