"""The one shape a finished plan takes.

There are two paths to a plan — the synchronous `/api/plan` stream and the
async job the worker runs — and they used to build their own result dicts. The
worker nested the agent output under `sections` while the API returned it at the
top level, so the browser could render one and not the other. Moving the UI onto
the job API surfaced that immediately, which is the good version of this bug;
the bad version is a field quietly added to one path and missing from the other
six months later.

So both call this, and the browser has one contract regardless of which side of
the queue produced the answer.
"""

from models import TripRequest, TripState
from status import OPTIONAL, REQUIRED

# The keys an agent writes, and the order the plan reads in.
SECTIONS = ("flight", "hotels", "weather", "itinerary", "budget")


def plan_payload(state: TripState, request: TripRequest) -> dict:
    """Flatten graph state into what the browser renders."""

    def dump(key: str):
        value = state.get(key)
        return value.model_dump() if value is not None else None

    return {
        "request": request.model_dump(),
        "status": state.get("status"),
        **{key: dump(key) for key in SECTIONS},
        "warnings": state.get("warnings") or [],
        "errors": state.get("errors") or [],
        "markdown": state.get("plan"),
        "metrics": state.get("metrics"),
        "cache": state.get("cache"),
        "required": list(REQUIRED),
        "optional": list(OPTIONAL),
    }
