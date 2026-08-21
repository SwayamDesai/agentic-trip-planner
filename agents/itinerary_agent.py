"""Itinerary composition. One structured call, no tool loop.

The place research moved to `places_agent`, which runs in the fan-out and costs
no tokens. What remains here is the part that genuinely needs a model:
arranging real places into days, against a weather outlook and a budget.

Meal costs and the paid/free distinction are no longer asked for in the prompt.
The tool already knows which candidates are restaurants and cafes, so that is
decided deterministically in `verify`/`costs` rather than instructed — the two
prompt rules that governed it contradicted each other ("never record 0 for a
place that charges" against "meals: set cost to 0"), and a contradiction is not
something a better wording fixes.
"""

from agents.base import describe_request, run_agent
from costs import activity_allowance
from models import ItineraryResult, TripState
from status import blocked_by_failure
from verify import verify_itinerary

# Candidate list sent to the model. Enough to choose from, small enough that
# the prompt stays inside a single-call token budget.
MAX_CANDIDATES = 40

# Activities per day. One number, used by the prompt AND the density scorer, so
# the instruction and the grader cannot drift apart.
MIN_PER_DAY = 2
MAX_PER_DAY = 5

SYSTEM = f"""You arrange real places into a day-by-day itinerary.

You are given a numbered list of CANDIDATE PLACES. It is the only source of
places you may use.

RULES:
- Use ONLY candidates from the list, by their exact name. If a famous
  attraction is not on the list, it does not go in the plan.
- One entry per trip day, in date order, covering every day.
- {MIN_PER_DAY} to {MAX_PER_DAY} activities per day. Meals count as activities.
- Group places that are geographically close on the same day, using the
  coordinates given.
- Arrival day light, departure day short.
- Use each place at most once across the whole trip.
- If a weather outlook is given, put indoor activities on wet days and set
  `indoor: true`.

COSTS: you have no price data — no tool provides entry fees. Every non-zero
cost is YOUR ESTIMATE and its notes must say "estimated". Keep estimates
conservative. Places marked `[free]` in the list cost 0; do not price them.

You cannot know opening hours or closing days. Never claim a place is open."""


def _candidate_lines(places, allowance) -> str:
    """Render candidates for the prompt, marking the ones known to be free.

    Marking is derived from the OpenStreetMap tag, not left to the model: it
    knows a plaza from a palace far more reliably than a language model
    reasoning about admission policy.
    """
    from costs import is_free_kind

    lines = []
    for i, c in enumerate(places.candidates[:MAX_CANDIDATES], 1):
        marker = " [free]" if is_free_kind(c.kind, c.category) else ""
        lines.append(
            f"{i:2}. {c.name}{marker} — {c.kind or c.category} "
            f"({c.lat:.4f}, {c.lon:.4f})"
        )
    return "\n".join(lines)


def itinerary_agent(state: TripState) -> TripState:
    """Compose the itinerary from researched places.

    Skipped when an upstream requirement already failed: the run cannot produce
    a usable plan, so spending the largest agent's tokens on it is waste.
    """
    blocked = blocked_by_failure(state, "itinerary")
    if blocked:
        return blocked

    req = state["request"]
    places = state["places"]

    prompt = [
        describe_request(req),
        f"CANDIDATE PLACES in {places.city} "
        f"(categories: {', '.join(places.categories)}):\n"
        + _candidate_lines(places, None),
    ]

    if places.guide:
        prompt.append(f"City orientation:\n{places.guide[:700]}")

    weather = state.get("weather")
    if weather and weather.daily:
        outlook = "\n".join(
            f"  {d.date}: {d.condition}, {d.low_c}-{d.high_c}C, "
            f"{d.precipitation_chance}% rain"
            for d in weather.daily
        )
        prompt.append(f"Weather outlook:\n{outlook}")
    else:
        prompt.append(
            "No weather data is available. Do not assume good weather: mix "
            "indoor and outdoor options each day, and say in `reasoning` that "
            "the plan was built without a weather outlook."
        )

    # flight and hotels feed this node, so their costs are known here. Without
    # this the agent knows the headline budget but not that travel and lodging
    # may already have consumed it.
    allowance = activity_allowance(state)
    if allowance is None:
        prompt.append(
            "No budget was given. Plan a balanced middle trip: mix free places "
            "with a few worthwhile paid ones. Do not optimise for cheapest."
        )
    elif allowance["feasible"] is False:
        prompt.append(
            f"THIS BUDGET IS NOT ACHIEVABLE: cheapest travel and lodging alone "
            f"exceed ${allowance['budget_usd']} by "
            f"${abs(allowance['remaining_usd']):.0f}. The activity allowance is "
            f"$0, so use only places marked [free], and name in `reasoning` the "
            f"paid highlights you left out."
        )
    else:
        prompt.append(
            f"ACTIVITY ALLOWANCE (hard cap): ${allowance['remaining_usd']:.0f} "
            f"for the whole party, about "
            f"${allowance['per_person_per_day']:.0f} per person per day. Your "
            f"estimated activity costs must not exceed this."
            + (
                f" No data for {', '.join(allowance['unknown'])}, so that figure "
                f"is optimistic — be conservative."
                if allowance["unknown"]
                else ""
            )
        )

    prompt.append(
        f"Build the itinerary for {req.start_date} to {req.end_date}."
    )

    return run_agent(
        name="itinerary",
        state=state,
        schema=ItineraryResult,
        system=SYSTEM,
        user="\n\n".join(prompt),
        temperature=0.4,
        verify=lambda result: verify_itinerary(result, req, state),
    )
