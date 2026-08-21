"""Checkpointed resume: what carries over, and what must not.

Three channels are append-reduced so concurrent agents can all write to them.
That makes them accumulate across resumes unless explicitly cleared, and all
three describe ONE RUN rather than the trip:

    errors    a failure the last run had may have succeeded since
    warnings  otherwise the same finding is reported once per resume
    evidence  stale place data could validate an itinerary the current run's
              places node never returned

`operator.add` cannot express a reset — seeding `[]` appends nothing and leaves
the checkpointed value in place — which is why there is a custom reducer.
"""

import pytest

from models import TripRequest, accumulate


REQ = TripRequest(
    origin="Chicago", destination="Seville",
    start_date="2026-11-10", end_date="2026-11-12", travelers=2, budget_usd=6000,
)


# --- the reducer ---


def test_appends_across_writers():
    """Concurrent agents must all land, which is the reason for a reducer."""
    assert accumulate(["a"], ["b"]) == ["a", "b"]


def test_none_resets():
    assert accumulate(["a", "b", "c"], None) == []


def test_empty_list_appends_nothing_but_does_not_clear():
    """The exact behaviour of operator.add that caused the bug, preserved so the
    difference between 'nothing to add' and 'clear this' stays explicit."""
    assert accumulate(["a"], []) == ["a"]


def test_handles_an_absent_channel():
    assert accumulate(None, ["a"]) == ["a"]
    assert accumulate(None, None) == []


# --- what plan_trip seeds ---


def _captured_seed(monkeypatch, prior):
    import orchestrator

    captured = {}

    class FakeGraph:
        def get_state(self, config):
            return type("S", (), {"values": prior})()

        def invoke(self, seed, config=None):
            captured.update(seed)
            return {"plan": "ok"}

    monkeypatch.setattr(orchestrator, "get_checkpointer", lambda: object())
    monkeypatch.setattr(orchestrator, "build_graph", lambda cp=None: FakeGraph())
    orchestrator.plan_trip(REQ)
    return captured


@pytest.mark.parametrize("channel", ["errors", "warnings", "evidence"])
def test_per_run_channels_are_reset_on_resume(monkeypatch, channel):
    """Regression: after three resumes of one trip, a single warning was
    reported three times, and a stale error survived the agent succeeding."""
    prior = {
        "errors": ["stale error"],
        "warnings": ["stale warning"],
        "evidence": [{"agent": "places", "kind": "places", "names": ["Old Place"]}],
    }
    seed = _captured_seed(monkeypatch, prior)
    assert seed[channel] is None, "None is the reset signal; [] would not clear"


def test_agent_results_are_carried_forward(monkeypatch):
    """The whole point of resume: do not re-pay for work that succeeded."""
    from models import FlightResult, PlacesResult

    flight = FlightResult(options=[])
    places = PlacesResult(city="Seville")
    seed = _captured_seed(monkeypatch, {"flight": flight, "places": places})
    assert seed["flight"] is flight
    assert seed["places"] is places


def test_failed_agents_are_left_unset_so_they_rerun(monkeypatch):
    seed = _captured_seed(monkeypatch, {"flight": None, "hotels": object()})
    assert "flight" not in seed, "a failed agent must run again"
    assert "hotels" in seed


def test_places_is_carried_forward(monkeypatch):
    """Added with the architecture change; omitting it would silently re-run the
    place research on every resume."""
    from models import PlacesResult

    seed = _captured_seed(monkeypatch, {"places": PlacesResult(city="X")})
    assert "places" in seed


# --- serialisation ---


def test_every_state_model_is_registered_for_deserialisation():
    """LangGraph warns on unregistered types and will block them.

    Enumerated from the module rather than hand-listed: a hand-list drifts, and
    the failure mode is a checkpoint that loads as plain dicts and then throws
    on the first attribute access.
    """
    from providers.memory import _allowed_models

    registered = {name for _, name in _allowed_models()}
    for required in (
        "TripRequest", "FlightResult", "HotelResult", "WeatherResult",
        "PlacesResult", "ItineraryResult", "BudgetResult", "CostBreakdown",
    ):
        assert required in registered, f"{required} would deserialise as a dict"


def test_allowlist_is_specific_not_blanket():
    """A checkpoint file is untrusted input once it moves between machines."""
    from providers.memory import _allowed_models

    allowed = _allowed_models()
    assert allowed is not True
    assert all(isinstance(entry, tuple) and len(entry) == 2 for entry in allowed)
    assert all(module == "models" for module, _ in allowed)
