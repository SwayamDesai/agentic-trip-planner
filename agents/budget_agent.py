"""Budget reconciliation.

Split deliberately down the middle:

    compute_breakdown()  pure Python. Every figure is arithmetic over numbers
                         the other agents already returned.
    LLM                  interpretation only — is this affordable, what would
                         you change, what is not counted.

The model never sees a blank total to fill in, so it cannot invent one. This
mirrors `synthesize`: anything derivable is derived, not generated.
"""

from agents.base import (
    _trace,
    agent_span,
    bind,
    deadline_for,
    describe_request,
    timeout_for,
)
from costs import activity_allowance, compute_breakdown
from tools.schemas import summarize_exception
from verify import verify_breakdown_arithmetic, verify_budget_cap
from models import (
    BudgetAdvice,
    BudgetResult,
    CostBreakdown,
    TripState,
)
from providers import metrics, prompts
from providers.llm import DeadlineExceeded, invoke_structured

import time


def _describe(state: TripState, breakdown: CostBreakdown) -> str:
    """Give the model the computed figures plus the options it may reference."""
    req = state["request"]
    lines = [
        describe_request(req),
        "",
        "COMPUTED BREAKDOWN (do not recalculate):",
        f"  flights    ${breakdown.flights_usd:.0f}  "
        f"(cheapest fare x {breakdown.travelers} travellers)",
        f"  lodging    ${breakdown.lodging_usd:.0f}  "
        f"(cheapest rate x {breakdown.nights} nights, whole party)",
        f"  activities ${breakdown.activities_usd:.0f}  "
        f"(entry costs x {breakdown.travelers} travellers)",
        f"  SUBTOTAL   ${breakdown.subtotal_usd:.0f}",
        f"  tier={breakdown.tier}"
        + (
            "  (a floor: cheapest flight and lodging already chosen)"
            if breakdown.tier == "cheapest"
            else "  (a middle option; cheaper choices exist)"
        ),
    ]
    if breakdown.budget_usd is not None:
        verdict = "OVER" if not breakdown.within_budget else "under"
        lines.append(
            f"  budget     ${breakdown.budget_usd} -> {verdict} by "
            f"${abs(breakdown.over_under_usd or 0):.0f}"
        )
    if breakdown.feasible is False:
        lines.append(
            f"  NOT ACHIEVABLE: cheapest travel + lodging alone is "
            f"${breakdown.travel_only_usd:.0f}, already over the budget."
        )
    if breakdown.missing:
        lines.append(f"  INCOMPLETE: no data for {', '.join(breakdown.missing)}")

    flight = state.get("flight")
    if flight and flight.options:
        lines.append("\nFlight options (price per person):")
        lines += [
            f"  {o.airline} ${o.price_usd:.0f}, {o.stops} stop(s), {o.duration}"
            for o in flight.options[:6]
        ]

    hotels = state.get("hotels")
    if hotels and hotels.options:
        lines.append("\nLodging options (per night, whole party):")
        lines += [
            f"  {h.name} ${h.price_per_night_usd:.0f}, rated {h.rating}"
            for h in hotels.options[:6]
        ]

    itinerary = state.get("itinerary")
    if itinerary and itinerary.days:
        paid = [
            f"{a.name} ${a.cost_usd:.0f}"
            for d in itinerary.days
            for a in d.activities
            if a.cost_usd
        ]
        if paid:
            lines.append("\nPaid activities (per person): " + ", ".join(paid[:8]))

    return "\n".join(lines)


def budget_agent(state: TripState) -> TripState:
    """Reconcile costs against the budget.

    Runs downstream of every other agent. The skip guard is conditional rather
    than the usual "already has a result": a blanket skip would keep stale
    advice after an upstream agent re-ran on resume, while no skip at all would
    re-pay for advice on every resume. So it recomputes the breakdown — cheap,
    pure arithmetic — and only calls the model when the numbers actually moved.
    """
    bind(state)
    breakdown = compute_breakdown(state)

    prior = state.get("budget")
    if prior is not None and prior.advice is not None and prior.breakdown == breakdown:
        _trace("budget", "skip (breakdown unchanged)", time.perf_counter())
        metrics.record_outcome("budget", "skipped", 0.0)
        return {}

    # The span wraps only the part that can fail. Both failure paths below used
    # to leave it open, which lost it from the trace.
    # Resolved before the span opens, so the span records the version used.
    prompt = prompts.get("budget")
    with agent_span("budget", prompt=prompt.label):
        return _advise(state, breakdown, prompt.text)


def _advise(state: TripState, breakdown: CostBreakdown, system: str) -> TripState:
    """Ask the model to interpret figures it cannot change."""
    t0 = time.perf_counter()
    _trace("budget", "start", t0)

    try:
        advice = invoke_structured(
            "budget",
            BudgetAdvice,
            [
                {"role": "system", "content": system},
                {"role": "user", "content": _describe(state, breakdown)},
            ],
            0.2,
            deadline=deadline_for("budget"),
        )
        _trace("budget", "done", t0)
        metrics.record_outcome("budget", "done", time.perf_counter() - t0)
        update = {"budget": BudgetResult(breakdown=breakdown, advice=advice)}
        checks = verify_breakdown_arithmetic(breakdown) + verify_budget_cap(
            breakdown, activity_allowance(state)
        )
        if checks:
            update["warnings"] = [f"budget: {w}" for w in checks]
        return update
    except DeadlineExceeded:
        # the arithmetic is already done and is the valuable half
        _trace("budget", "TIMEOUT", t0)
        metrics.record_outcome("budget", "timeout", time.perf_counter() - t0)
        return {
            "budget": BudgetResult(breakdown=breakdown, advice=None),
            "errors": [
                f"budget advice: timed out after {timeout_for('budget'):.0f}s; "
                f"the cost breakdown below is unaffected."
            ],
        }
    except Exception as exc:  # noqa: BLE001 - keep the arithmetic even if advice fails
        _trace("budget", f"FAILED {type(exc).__name__}", t0)
        metrics.record_outcome("budget", "failed", time.perf_counter() - t0)
        # summarised like every other agent error: the raw provider payload is
        # ~700 characters of JSON and tells the reader nothing useful
        return {
            "budget": BudgetResult(breakdown=breakdown, advice=None),
            "errors": [summarize_exception(exc, "budget advice")["error"]],
        }
