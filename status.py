"""Whether a run succeeded, degraded, or failed.

Not every agent matters equally, so treating all failures alike is wrong in
both directions: it either fails a usable plan because the weather lookup
broke, or it quietly ships a "plan" with no flights in it.

    REQUIRED  flight, hotels, itinerary
              A trip plan without these is not a trip plan. If one is missing
              the run FAILED — reported as such, with a non-zero exit code, so
              a caller or script cannot mistake it for success.

    OPTIONAL  weather, budget
              These enrich the plan. If one is missing the run is DEGRADED:
              everything else is published, and the gap is stated plainly.

`itinerary` is classed required by judgement, not instruction: flights and beds
with nothing to do is not an answer to "plan my trip". Move it to OPTIONAL here
if you disagree — that is the only change needed.
"""

from typing import Literal

from models import TripState

REQUIRED = ("flight", "hotels", "places", "itinerary")
OPTIONAL = ("weather", "budget")

# What each downstream agent cannot work without. Checked before the agent
# runs, because a plan that is already unusable should not cost the tokens of
# its most expensive node — on a free tier that is most of a run's budget.
UPSTREAM = {
    "itinerary": ("flight", "hotels", "places"),
    "budget": (),  # its arithmetic is useful even from partial data
}

Status = Literal["ok", "degraded", "failed"]

# What is lost when an optional agent is missing, so the caveat is concrete
# rather than "weather unavailable".
def blocked_by_failure(state: TripState, agent: str) -> dict | None:
    """A state update that skips `agent`, or None if it should run.

    Returns the same shape as a failure so the graph and the status logic need
    no special case: the key stays None and the reason is explicit.
    """
    dead = [
        name for name in UPSTREAM.get(agent, ()) if state.get(name) is None
    ]
    if not dead:
        return None
    return {
        agent: None,
        "errors": [
            f"{agent}: skipped because {', '.join(dead)} failed, so this plan "
            f"cannot be completed. Re-run to retry the failed agents."
        ],
    }


_DEGRADED_IMPACT = {
    "places": "no attraction data, so no itinerary could be built",
    "weather": (
        "no weather data, so indoor activities were not placed on wet days and "
        "there is no packing advice"
    ),
    "budget": "no cost breakdown, so the trip was not checked against a budget",
}


def missing_agents(state: TripState, names: tuple[str, ...]) -> list[str]:
    """Agents that produced no result. An error-isolated failure writes None."""
    return [name for name in names if state.get(name) is None]


def plan_status(state: TripState) -> tuple[Status, list[str]]:
    """Classify the run and explain any shortfall.

    Returns the status and a list of human-readable notes: what is missing and
    what that costs the reader.
    """
    missing_required = missing_agents(state, REQUIRED)
    missing_optional = missing_agents(state, OPTIONAL)

    notes: list[str] = []
    for name in missing_required:
        notes.append(f"{name} data is missing, which this plan cannot do without")
    for name in missing_optional:
        notes.append(f"{name}: {_DEGRADED_IMPACT.get(name, 'unavailable')}")

    if missing_required:
        return "failed", notes
    if missing_optional:
        return "degraded", notes
    return "ok", notes
