"""Resolving an under-specified request.

Stated values are honoured exactly; only genuinely absent ones are filled in.
"""

import pytest

import scope
from models import TripScope


@pytest.fixture
def no_llm(monkeypatch):
    """Fail loudly if resolution calls a model when it should not need to."""
    monkeypatch.setattr(
        scope, "invoke_structured",
        lambda *a, **k: pytest.fail("should not have asked the model"),
    )


def _resolve(**kw):
    base = dict(origin="Chicago", destination="Lisbon", start_date="2026-09-10")
    base.update(kw)
    return scope.resolve_request(**base)


# --- travellers ---


def test_travellers_default_to_two(no_llm):
    req, _ = _resolve(end_date="2026-09-13")
    assert req.travelers == 2


def test_stated_travellers_are_honoured(no_llm):
    assert _resolve(end_date="2026-09-13", travelers=5)[0].travelers == 5


def test_one_traveller_is_not_overridden(no_llm):
    """A solo trip must not be silently bumped to the default of 2."""
    assert _resolve(end_date="2026-09-13", travelers=1)[0].travelers == 1


# --- duration: stated ---


def test_end_date_is_honoured_exactly(no_llm):
    req, reason = _resolve(end_date="2026-09-20")
    assert req.end_date == "2026-09-20"
    assert reason is None
    assert req.nights_chosen_by_system is False


def test_night_count_is_honoured(no_llm):
    req, reason = _resolve(nights=5)
    assert req.end_date == "2026-09-15", "start + 5 nights"
    assert reason is None
    assert req.nights_chosen_by_system is False


def test_end_date_wins_over_nights(no_llm):
    """An explicit end date is the most specific instruction available."""
    req, _ = _resolve(end_date="2026-09-12", nights=9)
    assert req.end_date == "2026-09-12"


def test_stated_nights_are_clamped(no_llm):
    assert _resolve(nights=99)[0].end_date == "2026-09-24", "clamped to 14"
    assert _resolve(nights=0)[0].end_date == "2026-09-11", "clamped to 1"


# --- duration: inferred ---


def test_absent_duration_is_asked_of_the_model(monkeypatch):
    monkeypatch.setattr(
        scope, "invoke_structured",
        lambda *a, **k: TripScope(nights=4, reasoning="a capital worth four nights"),
    )
    req, reason = _resolve()
    assert req.end_date == "2026-09-14"
    assert reason == "a capital worth four nights"
    assert req.nights_chosen_by_system is True, "must be disclosed as assumed"


def test_model_answer_is_clamped(monkeypatch):
    monkeypatch.setattr(
        scope, "invoke_structured",
        lambda *a, **k: TripScope(nights=90, reasoning="too long"),
    )
    assert _resolve()[0].end_date == "2026-09-24", "clamped to 14 nights"


def test_model_failure_falls_back_without_blocking(monkeypatch):
    """A bad recommendation must not stop the trip being planned."""

    def boom(*a, **k):
        raise RuntimeError("rate limited")

    monkeypatch.setattr(scope, "invoke_structured", boom)
    req, reason = _resolve()
    assert req.end_date == "2026-09-13", "fallback of 3 nights"
    assert "RuntimeError" in reason
    assert req.nights_chosen_by_system is True


def test_preferences_are_passed_to_the_model(monkeypatch):
    seen = {}

    def spy(agent, schema, messages, temperature, **kw):
        seen["body"] = messages[1]["content"]
        return TripScope(nights=3, reasoning="ok")

    monkeypatch.setattr(scope, "invoke_structured", spy)
    _resolve(preferences=["hiking", "food"])
    assert "hiking" in seen["body"] and "Lisbon" in seen["body"]


# --- other fields pass through ---


def test_budget_and_preferences_survive(no_llm):
    req, _ = _resolve(end_date="2026-09-13", budget_usd=2500, preferences=["food"])
    assert req.budget_usd == 2500 and req.preferences == ["food"]


def test_absent_budget_stays_none(no_llm):
    assert _resolve(end_date="2026-09-13")[0].budget_usd is None


def test_assumed_duration_is_disclosed_in_the_plan(monkeypatch):
    """The reader must never mistake an assumed length for a stated one."""
    from orchestrator import synthesize

    monkeypatch.setattr(
        scope, "invoke_structured",
        lambda *a, **k: TripScope(nights=4, reasoning="four nights suits it"),
    )
    req, _ = _resolve()
    plan = synthesize({"request": req, "errors": []})["plan"]
    assert "system chose it" in plan


def test_stated_duration_is_not_disclosed(no_llm):
    from orchestrator import synthesize

    req, _ = _resolve(end_date="2026-09-13")
    plan = synthesize({"request": req, "errors": []})["plan"]
    assert "system chose it" not in plan


def test_trip_identity_differs_by_resolved_dates(no_llm):
    """Two different inferred lengths are different trips, not one resumed."""
    from providers.memory import thread_id

    a, _ = _resolve(nights=3)
    b, _ = _resolve(nights=5)
    assert thread_id(a) != thread_id(b)
