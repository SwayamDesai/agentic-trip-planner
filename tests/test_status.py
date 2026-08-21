"""Agent criticality: which failures degrade a plan and which invalidate it."""

import pytest

from models import (
    BudgetResult,
    PlacesResult,
    FlightResult,
    HotelResult,
    ItineraryResult,
    TripRequest,
    WeatherResult,
)
from status import OPTIONAL, REQUIRED, missing_agents, plan_status

REQ = TripRequest(
    origin="Chicago", destination="Seville",
    start_date="2026-10-05", end_date="2026-10-08", travelers=2, budget_usd=2600,
)


def _state(**overrides):
    from costs import compute_breakdown

    state = {
        "request": REQ,
        "flight": FlightResult(options=[]),
        "hotels": HotelResult(options=[]),
        # `places` is required: the itinerary is composed from its candidates
        "places": PlacesResult(city="Seville", candidates=[]),
        "itinerary": ItineraryResult(days=[]),
        "weather": WeatherResult(daily=[]),
        "errors": [],
    }
    state["budget"] = BudgetResult(breakdown=compute_breakdown(state))
    state.update(overrides)
    return state


# --- classification ---


def test_everything_present_is_ok():
    assert plan_status(_state())[0] == "ok"


@pytest.mark.parametrize("agent", REQUIRED)
def test_any_required_agent_missing_fails_the_run(agent):
    status, notes = plan_status(_state(**{agent: None}))
    assert status == "failed"
    assert any(agent in n for n in notes)


@pytest.mark.parametrize("agent", OPTIONAL)
def test_optional_agent_missing_only_degrades(agent):
    status, notes = plan_status(_state(**{agent: None}))
    assert status == "degraded"
    assert any(agent in n for n in notes)


def test_weather_is_optional_not_required():
    """The explicit requirement: weather enriches, it does not gate."""
    assert "weather" in OPTIONAL and "weather" not in REQUIRED


def test_flight_and_hotels_are_required():
    assert "flight" in REQUIRED and "hotels" in REQUIRED


def test_required_failure_outranks_optional():
    status, _ = plan_status(_state(flight=None, weather=None))
    assert status == "failed", "a missing requirement is not softened by a degradation"


def test_degraded_note_says_what_was_lost():
    """'weather unavailable' is not actionable; the consequence is."""
    _, notes = plan_status(_state(weather=None))
    joined = " ".join(notes)
    assert "indoor activities" in joined and "packing advice" in joined


def test_missing_agents_helper():
    state = _state(flight=None, weather=None)
    assert missing_agents(state, REQUIRED) == ["flight"]
    assert missing_agents(state, OPTIONAL) == ["weather"]


# --- rendering ---


def test_degraded_plan_is_banner_flagged():
    from orchestrator import synthesize

    out = synthesize(_state(weather=None))
    assert out["status"] == "degraded"
    assert "Partial plan" in out["plan"]


def test_failed_plan_warns_against_relying_on_it():
    from orchestrator import synthesize

    out = synthesize(_state(flight=None))
    plan = out["plan"]
    assert out["status"] == "failed"
    assert "INCOMPLETE PLAN" in plan
    assert "Do not rely on it" in plan


def test_ok_plan_has_no_banner():
    from orchestrator import synthesize

    plan = synthesize(_state())["plan"]
    assert "INCOMPLETE" not in plan and "Partial plan" not in plan


def test_banner_appears_before_the_content():
    """A caveat below the itinerary would be missed by a skim-reader."""
    from orchestrator import synthesize

    plan = synthesize(_state(flight=None))["plan"]
    assert "## Cost" in plan, "budget section should still render"
    assert plan.index("INCOMPLETE PLAN") < plan.index("## Cost")


def test_weatherless_itinerary_is_told_not_to_assume_sunshine(monkeypatch):
    import agents.itinerary_agent as mod
    from models import FlightResult, HotelResult, PlaceCandidate

    captured = {}
    monkeypatch.setattr(
        mod, "run_agent", lambda **kw: captured.update(kw) or {"itinerary": None}
    )
    mod.itinerary_agent({
        "request": REQ,
        "errors": [],
        "flight": FlightResult(options=[]),
        "hotels": HotelResult(options=[]),
        "places": PlacesResult(city="Seville", candidates=[
            PlaceCandidate(name="Torre del Oro", kind="attraction",
                           category="sights", lat=37.38, lon=-5.99)]),
    })
    assert "Do not assume good weather" in captured["user"]


# --- fail-fast (guardrail 1) ---


def test_places_is_required():
    """The itinerary is composed from its candidates, so it gates the plan."""
    assert "places" in REQUIRED


def test_blocked_by_failure_names_the_cause():
    from status import blocked_by_failure

    out = blocked_by_failure({"flight": None, "hotels": object(), "places": object()},
                             "itinerary")
    assert out["itinerary"] is None
    assert "flight failed" in out["errors"][0]
    assert "Re-run" in out["errors"][0], "the fix should be actionable"


def test_blocked_lists_every_dead_dependency():
    from status import blocked_by_failure

    out = blocked_by_failure({"flight": None, "hotels": None, "places": None},
                             "itinerary")
    for name in ("flight", "hotels", "places"):
        assert name in out["errors"][0]


def test_not_blocked_when_dependencies_are_present():
    from status import blocked_by_failure

    state = {"flight": object(), "hotels": object(), "places": object()}
    assert blocked_by_failure(state, "itinerary") is None


def test_budget_is_never_blocked():
    """Its arithmetic is useful even from partial data, and costs nothing."""
    from status import blocked_by_failure

    assert blocked_by_failure({"flight": None, "hotels": None}, "budget") is None
