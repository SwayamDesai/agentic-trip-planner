"""Shared cost arithmetic, including the activity allowance."""

import pytest

from costs import activity_allowance, nights
from models import (
    FlightOption,
    FlightResult,
    HotelOption,
    HotelResult,
    TripRequest,
)


def _req(**kw):
    base = dict(
        origin="Chicago", destination="Seville",
        start_date="2026-10-05", end_date="2026-10-08",   # 3 nights
        travelers=2, budget_usd=2600,
    )
    base.update(kw)
    return TripRequest(**base)


def _state(fare=None, rate=None, **kw):
    from models import PlaceCandidate, PlacesResult

    state = {
        "request": _req(**kw),
        "errors": [],
        # the itinerary is composed from candidates now, so a places result is
        # part of a minimally valid state
        "places": PlacesResult(city="Seville", candidates=[
            PlaceCandidate(name="Torre del Oro", kind="attraction",
                           category="sights", lat=37.38, lon=-5.99)]),
    }
    if fare is not None:
        state["flight"] = FlightResult(
            options=[FlightOption(airline="X", departure="d", arrival="a",
                                  duration="1h", stops=0, price_usd=fare)]
        )
    if rate is not None:
        state["hotels"] = HotelResult(
            options=[HotelOption(name="H", area="C", rating=4.0,
                                 price_per_night_usd=rate)]
        )
    return state


def test_nights_from_dates():
    assert nights(_req()) == 3


def test_allowance_is_budget_minus_committed():
    """1413 x 2 travellers + 117 x 3 nights = 3177, over a 2600 budget."""
    a = activity_allowance(_state(fare=1413.0, rate=117.0))
    assert a["committed_usd"] == 3177.0
    assert a["remaining_usd"] == -577.0


def test_negative_remaining_is_not_clamped():
    """'You are $577 over before any activities' is the useful signal.

    Clamping to 0 would let the agent believe it merely has nothing to spend,
    rather than that the trip is already unaffordable.
    """
    assert activity_allowance(_state(fare=1413.0, rate=117.0))["remaining_usd"] < 0


def test_healthy_budget_leaves_room():
    a = activity_allowance(_state(fare=400.0, rate=80.0))
    assert a["committed_usd"] == 1040.0
    assert a["remaining_usd"] == 1560.0
    # 1560 / 2 travellers / 3 nights
    assert a["per_person_per_day"] == 260.0


def test_no_budget_means_no_allowance():
    assert activity_allowance(_state(fare=400.0, budget_usd=None)) is None


def test_missing_agents_are_flagged_as_understated():
    a = activity_allowance(_state(fare=400.0))
    assert a["unknown"] == ["lodging"]
    assert a["lodging_usd"] is None
    assert a["committed_usd"] == 800.0, "absent costs count as 0, not a guess"


def test_nothing_committed_yet():
    a = activity_allowance(_state())
    assert a["committed_usd"] == 0.0
    assert a["remaining_usd"] == 2600.0
    assert set(a["unknown"]) == {"flights", "lodging"}


def test_zero_nights_does_not_divide_by_zero():
    a = activity_allowance(_state(fare=100.0, end_date="2026-10-05"))
    assert a["per_person_per_day"] == pytest.approx(1200.0)


def test_infeasible_budget_is_stated_plainly(monkeypatch):
    """When cheapest travel alone busts the budget, the agent must be told so
    rather than left to infer it from a negative number."""
    import agents.itinerary_agent as mod

    captured = {}
    monkeypatch.setattr(
        mod, "run_agent", lambda **kw: captured.update(kw) or {"itinerary": None}
    )
    mod.itinerary_agent(_state(fare=1413.0, rate=117.0))
    body = captured["user"]
    assert "NOT ACHIEVABLE" in body
    assert "[free]" in body, "must be told to use only free-marked candidates"
    assert "$577" in body, "the size of the gap should be explicit"


def test_affordable_budget_is_a_hard_cap(monkeypatch):
    import agents.itinerary_agent as mod

    captured = {}
    monkeypatch.setattr(
        mod, "run_agent", lambda **kw: captured.update(kw) or {"itinerary": None}
    )
    mod.itinerary_agent(_state(fare=400.0, rate=80.0))
    body = captured["user"]
    assert "hard cap" in body and "$1560" in body
    assert "NOT ACHIEVABLE" not in body


def test_no_budget_asks_for_a_middle_trip(monkeypatch):
    """Rule 3: with no budget, neither bargain-hunting nor luxury."""
    import agents.itinerary_agent as mod

    captured = {}
    monkeypatch.setattr(
        mod, "run_agent", lambda **kw: captured.update(kw) or {"itinerary": None}
    )
    # both travel costs must be present or the itinerary is skipped as blocked
    mod.itinerary_agent(_state(fare=400.0, rate=80.0, budget_usd=None))
    body = captured["user"]
    assert "balanced middle trip" in body
    assert "hard cap" not in body and "ALLOWANCE" not in body


def test_meals_are_not_priced_in_the_itinerary():
    """Regression: pricing meals double-counted food, which the budget declared
    excluded. Now enforced from the OSM tag rather than instructed — the two
    prompt rules that governed it contradicted each other."""
    from costs import is_free_kind, is_meal

    assert is_meal("restaurant") and is_meal("cafe")
    assert is_free_kind("restaurant"), "meals carry no cost in the itinerary"
    assert not is_free_kind("castle"), "paid monuments still cost"


# --- tier selection ---


def test_budget_given_costs_the_cheapest():
    from costs import compute_breakdown

    b = compute_breakdown(_state(fare=400.0, rate=80.0))
    assert b.tier == "cheapest"


def test_no_budget_costs_the_middle_option():
    from costs import compute_breakdown
    from models import FlightOption, FlightResult

    state = _state(budget_usd=None)
    state["flight"] = FlightResult(
        options=[
            FlightOption(airline=f"A{p}", departure="d", arrival="a", duration="1h",
                         stops=0, price_usd=p)
            for p in (400.0, 900.0, 2000.0)
        ]
    )
    b = compute_breakdown(state)
    assert b.tier == "mid"
    assert b.flights_usd == 1800.0, "middle fare 900 x 2, not the 400 floor"


def test_feasibility_uses_the_floor_not_the_tier():
    """A mid-tier plan can be over budget while the trip is still feasible."""
    from costs import compute_breakdown
    from models import FlightOption, FlightResult

    state = _state(rate=80.0)
    state["flight"] = FlightResult(
        options=[
            FlightOption(airline=f"A{p}", departure="d", arrival="a", duration="1h",
                         stops=0, price_usd=p)
            for p in (400.0, 1200.0)
        ]
    )
    b = compute_breakdown(state)
    assert b.feasible is True, "cheapest 400x2 + 240 lodging fits 2600"


def test_infeasible_when_floor_exceeds_budget():
    from costs import compute_breakdown

    b = compute_breakdown(_state(fare=1413.0, rate=117.0))
    assert b.feasible is False
    assert b.travel_only_usd == 3177.0


def test_feasibility_unknown_without_both_costs():
    """Cannot judge feasibility while a floor cost is still missing."""
    from costs import compute_breakdown

    assert compute_breakdown(_state(fare=400.0)).feasible is None


def test_no_budget_means_no_feasibility_verdict():
    from costs import compute_breakdown

    assert compute_breakdown(_state(fare=400.0, rate=80.0, budget_usd=None)).feasible is None


def test_itinerary_is_skipped_when_an_upstream_requirement_failed():
    """Guardrail: the largest agent must not spend tokens on a doomed run."""
    import agents.itinerary_agent as mod

    state = _state(rate=80.0)          # no flight result
    state["flight"] = None
    out = mod.itinerary_agent(state)
    assert out["itinerary"] is None
    assert "skipped because flight failed" in out["errors"][0]


# --- places: category selection is a judgement, with a safe fallback ---


def test_categories_are_chosen_by_the_agent(monkeypatch):
    """The decision depends on the destination, not just stated interests —
    travellers usually state none at all."""
    import agents.places_agent as mod
    from models import PlaceSearchPlan

    seen = {}

    def fake(agent, schema, messages, temperature, **kw):
        seen["prompt"] = messages[1]["content"]
        return PlaceSearchPlan(
            categories=["historic", "sights"],
            reasoning="Seville is known for Moorish architecture",
        )

    monkeypatch.setattr(mod, "invoke_structured", fake)
    categories, why = mod._choose_categories(
        "Seville", [], "Capital of Andalusia", "system prompt"
    )
    assert categories == ["historic", "sights"]
    assert "Moorish" in why
    assert "stated no particular interests" in seen["prompt"]
    assert "Capital of Andalusia" in seen["prompt"], "guide informs the choice"


def test_stated_interests_reach_the_agent(monkeypatch):
    import agents.places_agent as mod
    from models import PlaceSearchPlan

    seen = {}

    def fake(agent, schema, messages, temperature, **kw):
        seen["prompt"] = messages[1]["content"]
        return PlaceSearchPlan(categories=["food"], reasoning="ok")

    monkeypatch.setattr(mod, "invoke_structured", fake)
    mod._choose_categories("Lyon", ["food", "wine"], "", "system prompt")
    assert "food, wine" in seen["prompt"]


def test_unknown_category_from_the_model_is_dropped(monkeypatch):
    """The schema restricts the vocabulary; this is the belt-and-braces check."""
    import agents.places_agent as mod

    class Loose:
        categories = ["historic", "nightlife", "historic"]
        reasoning = "x"

    monkeypatch.setattr(mod, "invoke_structured", lambda *a, **k: Loose())
    categories, _ = mod._choose_categories("Berlin", [], "", "system prompt")
    assert categories == ["historic"], "unknown dropped, duplicate collapsed"


def test_model_failure_falls_back_deterministically(monkeypatch):
    """A rate limit must degrade the research, not end the plan."""
    import agents.places_agent as mod

    def boom(*a, **k):
        raise RuntimeError("rate limited")

    monkeypatch.setattr(mod, "invoke_structured", boom)
    categories, why = mod._choose_categories(
        "Seville", ["history"], "", "system prompt"
    )
    assert categories == ["sights", "historic"]
    assert "fell back" in why


def test_fallback_with_no_interests_still_returns_something():
    from agents.places_agent import categories_for

    assert categories_for([]) == ["sights"]


def test_empty_category_list_falls_back(monkeypatch):
    import agents.places_agent as mod

    class Empty:
        categories = []
        reasoning = "x"

    monkeypatch.setattr(mod, "invoke_structured", lambda *a, **k: Empty())
    categories, why = mod._choose_categories(
        "Seville", ["food"], "", "system prompt"
    )
    assert categories == ["sights", "food"] and "fell back" in why
