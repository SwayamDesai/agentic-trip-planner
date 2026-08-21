"""Budget arithmetic. Pure and deterministic, so it is fully testable.

The unit conventions are the trap here: flight fares are PER PERSON, hotel
rates are for the WHOLE PARTY, activity costs are PER PERSON. Mixing any two
produces a total that looks entirely reasonable and is wrong.
"""

import pytest

from agents.budget_agent import budget_agent, compute_breakdown
from models import (
    Activity,
    BudgetAdvice,
    BudgetResult,
    DayPlan,
    FlightOption,
    FlightResult,
    HotelOption,
    HotelResult,
    ItineraryResult,
    TripRequest,
)


def _flight(*prices):
    return FlightResult(
        options=[
            FlightOption(
                airline="X", departure="d", arrival="a", duration="1h",
                stops=0, price_usd=p,
            )
            for p in prices
        ]
    )


def _hotels(*rates):
    return HotelResult(
        options=[
            HotelOption(name=f"H{r}", area="City", rating=4.0, price_per_night_usd=r)
            for r in rates
        ]
    )


def _itinerary(*costs):
    return ItineraryResult(
        days=[
            DayPlan(
                date="2026-09-10",
                activities=[
                    Activity(name=f"A{c}", time_of_day="morning",
                             duration_hours=1, cost_usd=c)
                    for c in costs
                ],
            )
        ]
    )


def _state(**kw):
    req = TripRequest(
        origin="Chicago", destination="Lisbon",
        start_date="2026-09-10", end_date="2026-09-13",   # 3 nights
        travelers=2, budget_usd=3000,
    )
    if "request" in kw:
        req = kw.pop("request")
    state = {"request": req, "errors": []}
    state.update(kw)
    return state


# --- unit conventions ---


def test_flights_scale_by_travellers():
    """Fares are per person."""
    b = compute_breakdown(_state(flight=_flight(1104.0, 1316.0)))
    assert b.flights_usd == 2208.0, "cheapest 1104 x 2 travellers"


def test_lodging_does_not_scale_by_travellers():
    """Nightly rates already cover the party — scaling would double-count."""
    b = compute_breakdown(_state(hotels=_hotels(67.0, 364.0)))
    assert b.lodging_usd == 201.0, "cheapest 67 x 3 nights, NOT x travellers"


def test_activities_scale_by_travellers():
    b = compute_breakdown(_state(itinerary=_itinerary(12.0, 0.0, 10.0)))
    assert b.activities_usd == 44.0, "(12 + 0 + 10) x 2 travellers"


def test_subtotal_is_the_sum():
    b = compute_breakdown(
        _state(flight=_flight(1104.0), hotels=_hotels(67.0), itinerary=_itinerary(12.0))
    )
    assert b.flights_usd == 2208.0
    assert b.lodging_usd == 201.0
    assert b.activities_usd == 24.0
    assert b.subtotal_usd == 2433.0


def test_cheapest_option_is_used_not_first():
    b = compute_breakdown(_state(flight=_flight(2000.0, 900.0, 1500.0)))
    assert b.flights_usd == 1800.0, "900 x 2"


# --- budget comparison ---


def test_under_budget():
    b = compute_breakdown(_state(flight=_flight(1000.0)))
    assert b.subtotal_usd == 2000.0
    assert b.over_under_usd == -1000.0
    assert b.within_budget is True


def test_over_budget():
    b = compute_breakdown(_state(flight=_flight(1800.0)))
    assert b.over_under_usd == 600.0
    assert b.within_budget is False


def test_exactly_on_budget_counts_as_within():
    b = compute_breakdown(_state(flight=_flight(1500.0)))
    assert b.over_under_usd == 0.0 and b.within_budget is True


def test_no_budget_leaves_comparison_unset():
    req = TripRequest(
        origin="A", destination="B", start_date="2026-09-10",
        end_date="2026-09-13", travelers=2, budget_usd=None,
    )
    b = compute_breakdown(_state(request=req, flight=_flight(500.0)))
    assert b.over_under_usd is None and b.within_budget is None


# --- missing data honesty ---


def test_missing_agents_are_named():
    """A subtotal built from partial data must not look authoritative."""
    b = compute_breakdown(_state(flight=_flight(1000.0)))
    assert set(b.missing) == {"lodging", "activities"}


def test_failed_agent_contributes_zero_not_a_guess():
    b = compute_breakdown(_state(flight=None, hotels=_hotels(50.0)))
    assert b.flights_usd == 0.0 and "flights" in b.missing


def test_empty_options_treated_as_missing():
    b = compute_breakdown(_state(flight=FlightResult(options=[])))
    assert "flights" in b.missing


def test_nothing_at_all_is_still_valid():
    b = compute_breakdown(_state())
    assert b.subtotal_usd == 0.0
    assert len(b.missing) == 3


# --- edge cases ---


def test_nights_from_dates_not_weather():
    req = TripRequest(
        origin="A", destination="B", start_date="2026-09-10",
        end_date="2026-09-20", travelers=1, budget_usd=None,
    )
    assert compute_breakdown(_state(request=req)).nights == 10


def test_same_day_trip_has_zero_nights():
    req = TripRequest(
        origin="A", destination="B", start_date="2026-09-10",
        end_date="2026-09-10", travelers=1, budget_usd=None,
    )
    b = compute_breakdown(_state(request=req, hotels=_hotels(100.0)))
    assert b.nights == 0 and b.lodging_usd == 0.0


def test_malformed_dates_do_not_crash():
    req = TripRequest(
        origin="A", destination="B", start_date="not-a-date",
        end_date="also-bad", travelers=1, budget_usd=None,
    )
    assert compute_breakdown(_state(request=req)).nights == 0


def test_zero_travellers_floored_to_one():
    req = TripRequest(
        origin="A", destination="B", start_date="2026-09-10",
        end_date="2026-09-11", travelers=0, budget_usd=None,
    )
    b = compute_breakdown(_state(request=req, flight=_flight(500.0)))
    assert b.travelers == 1 and b.flights_usd == 500.0


# --- node behaviour ---


def test_arithmetic_survives_advice_failure(monkeypatch):
    """If the model dies, the numbers must still reach the plan."""
    import agents.budget_agent as mod

    def boom(*a, **k):
        raise RuntimeError("model down")

    monkeypatch.setattr(mod, "invoke_structured", boom)
    out = budget_agent(_state(flight=_flight(1000.0)))
    assert out["budget"].breakdown.subtotal_usd == 2000.0
    assert out["budget"].advice is None
    assert "model down" in out["errors"][0]


def test_advice_is_attached(monkeypatch):
    import agents.budget_agent as mod

    monkeypatch.setattr(
        mod, "invoke_structured",
        lambda *a, **k: BudgetAdvice(assessment="fine", suggestions=["x"]),
    )
    out = budget_agent(_state(flight=_flight(1000.0)))
    assert out["budget"].advice.assessment == "fine"


def test_skips_when_breakdown_unchanged(monkeypatch):
    """Resume must stay free when nothing upstream moved."""
    import agents.budget_agent as mod

    monkeypatch.setattr(
        mod, "invoke_structured",
        lambda *a, **k: pytest.fail("should not have called the model"),
    )
    state = _state(flight=_flight(1000.0))
    prior = BudgetResult(
        breakdown=compute_breakdown(state),
        advice=BudgetAdvice(assessment="cached"),
    )
    state["budget"] = prior
    assert budget_agent(state) == {}


def test_recomputes_when_upstream_changed(monkeypatch):
    """Stale advice must not survive an upstream agent re-running."""
    import agents.budget_agent as mod

    calls = []

    def spy(*a, **k):
        calls.append(1)
        return BudgetAdvice(assessment="fresh")

    monkeypatch.setattr(mod, "invoke_structured", spy)

    old_state = _state(flight=_flight(1000.0))
    prior = BudgetResult(
        breakdown=compute_breakdown(old_state),
        advice=BudgetAdvice(assessment="stale"),
    )
    new_state = _state(flight=_flight(1800.0), budget=prior)   # flight re-ran
    out = budget_agent(new_state)
    assert calls, "breakdown moved, so advice must be regenerated"
    assert out["budget"].advice.assessment == "fresh"


def test_no_advice_cached_still_calls_model(monkeypatch):
    import agents.budget_agent as mod

    monkeypatch.setattr(
        mod, "invoke_structured", lambda *a, **k: BudgetAdvice(assessment="new")
    )
    state = _state(flight=_flight(1000.0))
    state["budget"] = BudgetResult(breakdown=compute_breakdown(state), advice=None)
    assert budget_agent(state)["budget"].advice is not None


def test_advice_failure_message_is_summarised(monkeypatch):
    """Regression: budget_agent had its own handler that bypassed the shared
    summariser, so a ~700-char 429 payload reached the plan verbatim."""
    import agents.budget_agent as mod

    raw = (
        "Error code: 429 - {'error': {'message': 'Rate limit reached for model "
        "`openai/gpt-oss-120b` in organization `org_01k` on tokens per day "
        "(TPD): Limit 200000, Used 199445, Requested 833. Please try again in "
        "2m0.09s. Need more tokens? Upgrade at https://console.groq.com/billing'}}"
    )

    def boom(*a, **k):
        raise RuntimeError(raw)

    monkeypatch.setattr(mod, "invoke_structured", boom)
    out = budget_agent(_state(flight=_flight(1000.0)))
    message = out["errors"][0]
    assert len(message) < 160, f"still a firehose: {message}"
    assert "daily token quota" in message
    assert "console.groq" not in message
    assert out["budget"].breakdown.subtotal_usd == 2000.0, "arithmetic survives"
