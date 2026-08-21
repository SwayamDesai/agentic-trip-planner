"""Graph wiring, state merging and the synthesize renderer."""

import pytest

from models import (
    Activity,
    DayForecast,
    DayPlan,
    FlightOption,
    FlightResult,
    HotelOption,
    HotelResult,
    ItineraryResult,
    WeatherResult,
)
from orchestrator import build_graph, synthesize


def _full_state(trip):
    return {
        "request": trip,
        "flight": FlightResult(
            options=[
                FlightOption(
                    airline="Aer Lingus", departure="2026-09-10 21:00 (ORD)",
                    arrival="2026-09-11 21:35 (LIS)", duration="18h 35m",
                    stops=1, price_usd=1104.0, notes="round trip",
                )
            ]
        ),
        "hotels": HotelResult(
            options=[
                HotelOption(name="Hotel A", area="Lisbon", rating=4.3,
                            price_per_night_usd=67.0, notes="live rate")
            ]
        ),
        "weather": WeatherResult(
            daily=[
                DayForecast(date="2026-09-10", condition="dry", high_c=25, low_c=18),
                DayForecast(date="2026-09-11", condition="dry", high_c=26, low_c=18),
            ],
            packing_advice="light layers",
        ),
        "itinerary": ItineraryResult(
            days=[
                DayPlan(date="2026-09-10", activities=[
                    Activity(name="Alcazar", time_of_day="morning",
                             duration_hours=2, cost_usd=12, notes="estimated"),
                    Activity(name="Plaza", time_of_day="evening", duration_hours=1),
                ])
            ]
        ),
        "errors": [],
    }


def test_renders_every_section(trip):
    plan = synthesize(_full_state(trip))["plan"]
    for heading in ("## Flights", "## Where to stay", "## Weather", "## Itinerary"):
        assert heading in plan
    assert "Aer Lingus" in plan and "Hotel A" in plan and "Alcazar" in plan
    assert "Agents that failed" not in plan


def test_cost_section_renders_budget_breakdown(trip):
    """The Cost section comes from the budget agent's computed figures.

    flights 1104 x 2 + lodging 67 x 3 nights + activities 12 x 2 = 2433
    """
    from agents.budget_agent import compute_breakdown
    from models import BudgetAdvice, BudgetResult

    state = _full_state(trip)
    state["budget"] = BudgetResult(
        breakdown=compute_breakdown(state),
        advice=BudgetAdvice(
            assessment="Comfortably affordable.",
            suggestions=["swap to Hotel A at $67/night"],
            unbudgeted=["food", "local transport"],
        ),
    )
    plan = synthesize(state)["plan"]
    assert "## Cost" in plan
    assert "$2433" in plan, plan[plan.index("## Cost"):]
    assert "3 night(s)" in plan, "nights come from trip dates, not weather days"
    assert "within budget" in plan and "$567 to spare" in plan
    assert "Comfortably affordable." in plan
    assert "swap to Hotel A" in plan
    assert "food" in plan


def test_over_budget_is_called_out(trip):
    from agents.budget_agent import compute_breakdown
    from models import BudgetResult

    state = _full_state(trip)
    state["request"] = trip.model_copy(update={"budget_usd": 1000})
    state["budget"] = BudgetResult(breakdown=compute_breakdown(state))
    assert "over budget by $1433" in synthesize(state)["plan"]


def test_incomplete_subtotal_is_flagged(trip):
    """A subtotal missing an agent's costs must not look authoritative."""
    from agents.budget_agent import compute_breakdown
    from models import BudgetResult

    state = _full_state(trip)
    state["hotels"] = None
    state["budget"] = BudgetResult(breakdown=compute_breakdown(state))
    plan = synthesize(state)["plan"]
    assert "Incomplete" in plan and "lodging" in plan


def test_partial_state_still_renders(trip):
    """A failed agent omits its section rather than blocking the plan."""
    state = _full_state(trip)
    state["flight"] = None
    state["itinerary"] = None
    plan = synthesize(state)["plan"]
    assert "## Flights" not in plan
    assert "## Where to stay" in plan and "## Weather" in plan


def test_errors_are_surfaced(trip):
    state = _full_state(trip)
    state["errors"] = ["weather: RateLimit", "flight: AuthError"]
    plan = synthesize(state)["plan"]
    assert "## Agents that failed" in plan
    assert "weather: RateLimit" in plan and "flight: AuthError" in plan


def test_empty_state_does_not_crash(trip):
    plan = synthesize({"request": trip, "errors": []})["plan"]
    assert "Chicago" in plan and "Lisbon" in plan


def test_free_activity_shows_free(trip):
    plan = synthesize(_full_state(trip))["plan"]
    assert "free" in plan


# --- graph topology ---


def test_graph_compiles():
    assert build_graph() is not None


def test_fanout_agents_are_parallel_and_itinerary_is_downstream():
    """The fan-out three must not depend on each other; itinerary must depend
    on weather, since it reads the outlook to place indoor activities."""
    graph = build_graph()
    edges = graph.get_graph().edges
    pairs = {(e.source, e.target) for e in edges}

    for node in ("flight", "weather", "hotels"):
        assert ("__start__", node) in pairs, f"{node} should start immediately"
        assert (node, "itinerary") in pairs, f"itinerary must wait for {node}"

    assert ("weather", "flight") not in pairs, "fan-out must not be chained"
    assert ("itinerary", "budget") in pairs, "budget needs itinerary's costs"
    assert ("budget", "synthesize") in pairs
    assert ("itinerary", "synthesize") not in pairs, "budget sits between them"


def test_errors_channel_accumulates_across_agents():
    """Regression: without an append reducer, concurrent failures overwrite
    each other and only one error survives."""
    from models import TripState
    from typing import get_type_hints

    hints = get_type_hints(TripState, include_extras=True)
    assert hasattr(hints["errors"], "__metadata__"), "errors needs a reducer"


def test_infeasible_budget_is_rendered_plainly(trip):
    """Rule 3: if the budget cannot be met, the plan must say so outright."""
    from costs import compute_breakdown
    from models import BudgetResult

    state = _full_state(trip)
    state["request"] = trip.model_copy(update={"budget_usd": 500})
    state["budget"] = BudgetResult(breakdown=compute_breakdown(state))
    plan = synthesize(state)["plan"]
    assert "not achievable within $500" in plan
    assert "cheapest flights and lodging alone" in plan


def test_mid_tier_is_disclosed_when_no_budget(trip):
    """Rule 2: a mid-range costing should not be mistaken for the cheapest."""
    from costs import compute_breakdown
    from models import BudgetResult

    state = _full_state(trip)
    state["request"] = trip.model_copy(update={"budget_usd": None})
    state["budget"] = BudgetResult(breakdown=compute_breakdown(state))
    plan = synthesize(state)["plan"]
    assert "mid-range options" in plan
    assert "over budget" not in plan


def test_cost_labels_track_the_tier(trip):
    """Regression: labels said 'cheapest' while showing mid-range figures."""
    from costs import compute_breakdown
    from models import BudgetResult

    state = _full_state(trip)
    state["request"] = trip.model_copy(update={"budget_usd": None})
    state["budget"] = BudgetResult(breakdown=compute_breakdown(state))
    plan = synthesize(state)["plan"]
    assert "mid-range x 2 traveler(s)" in plan
    assert "cheapest x" not in plan

    state["request"] = trip
    state["budget"] = BudgetResult(breakdown=compute_breakdown(state))
    plan = synthesize(state)["plan"]
    assert "cheapest x 2 traveler(s)" in plan
    assert "mid-range x" not in plan
