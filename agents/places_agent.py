"""Destination research: decide what to look for, then look for it.

Split out of the itinerary agent, which was spending ~5 LLM round trips on
this — one per tool call, because it held the tools and had to drive them.

The decision itself genuinely needs a model. Which kinds of place matter
depends on the destination, and travellers usually state no interests at all:
Seville rewards historic architecture, Lyon food, Bergen the outdoors. A fixed
mapping from stated preferences cannot know that, and with no preferences given
it has nothing to map.

So the shape is one judgement call plus deterministic execution:

    geocode      -> deterministic
    city guide   -> deterministic, and gives the model context to judge with
    categories   -> ONE structured call, from a closed vocabulary
    find_places  -> deterministic, once per chosen category

One LLM call instead of five, and it runs in the fan-out because none of it
depends on weather, fares or lodging — the reason the itinerary agent used to
start its research last.
"""

import time

from agents.base import _trace, deadline_for, timeout_for
from models import PlaceCandidate, PlaceSearchPlan, PlacesResult, TripState
from providers import metrics, tracing
from providers.llm import DeadlineExceeded, invoke_structured
from tools.geo import geocode
from tools.places import CATEGORY_FILTERS, city_guide, find_places

SYSTEM = """You decide what kinds of place to research for a trip.

Choose 2-3 categories from exactly these, most important first:
  sights   — landmarks, viewpoints, notable attractions
  museums  — museums and galleries
  historic — castles, palaces, monasteries, city walls, old quarters
  food     — restaurants and cafes

Choose on what the DESTINATION is actually known for, not on a generic
template. Seville rewards historic architecture; Lyon food; Bergen viewpoints
and the outdoors; Florence museums.

If the traveller stated interests, honour them — but they usually state none,
and "no interests given" is not a reason to default blindly. Use the city
description to judge.

Always include at least one category that covers general sightseeing unless the
destination is genuinely specialised."""


# Fallback only, for when the model call fails. Interests map to Overpass
# categories; "sights" is always included so a trip still has somewhere to go.
_INTEREST_CATEGORIES = {
    "food": "food",
    "restaurant": "food",
    "restaurants": "food",
    "eating": "food",
    "cuisine": "food",
    "history": "historic",
    "historic": "historic",
    "historical": "historic",
    "heritage": "historic",
    "architecture": "historic",
    "culture": "museums",
    "museum": "museums",
    "museums": "museums",
    "art": "museums",
    "galleries": "museums",
}

PER_CATEGORY_LIMIT = 12


def categories_for(preferences: list[str]) -> list[str]:
    """Deterministic fallback when the model cannot be reached.

    Weaker than the agent — it cannot know what a destination is known for, and
    with no stated interests it returns only "sights" — but it always returns
    something usable, so a rate limit degrades the plan instead of ending it.
    """
    chosen = ["sights"]
    for preference in preferences or []:
        category = _INTEREST_CATEGORIES.get(preference.strip().lower())
        if category and category not in chosen:
            chosen.append(category)
    return chosen


def _choose_categories(destination: str, preferences: list[str], guide: str):
    """Ask which categories suit this destination. Falls back if unavailable."""
    stated = (
        f"The traveller mentioned: {', '.join(preferences)}."
        if preferences
        else "The traveller stated no particular interests."
    )
    try:
        plan = invoke_structured(
            "places",
            PlaceSearchPlan,
            [
                {"role": "system", "content": SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"Destination: {destination}\n{stated}\n\n"
                        + (f"About the city:\n{guide[:900]}\n\n" if guide else "")
                        + "Which categories should be researched?"
                    ),
                },
            ],
            0.2,
            deadline=deadline_for("places"),
        )
        # the schema restricts the vocabulary, but de-duplicate and bound it
        chosen: list[str] = []
        for category in plan.categories:
            if category in CATEGORY_FILTERS and category not in chosen:
                chosen.append(category)
        if chosen:
            return chosen[:3], plan.reasoning
    except Exception as exc:  # noqa: BLE001 - degrade, do not fail the node
        return (
            categories_for(preferences),
            f"fell back to default categories ({type(exc).__name__})",
        )
    return categories_for(preferences), "fell back to default categories"


def places_agent(state: TripState) -> TripState:
    """Gather real candidate places for the destination."""
    if state.get("places") is not None:
        _trace("places", "skip (cached from earlier run)", time.perf_counter())
        metrics.record_outcome("places", "skipped", 0.0)
        return {}

    t0 = time.perf_counter()
    _trace("places", "start", t0)
    req = state["request"]

    location = geocode(req.destination)
    if location.get("lat") is None:
        _trace("places", "FAILED geocode", t0)
        metrics.record_outcome("places", "failed", time.perf_counter() - t0)
        return {
            "places": None,
            "errors": [
                f"places: could not locate {req.destination!r}, so no attractions "
                f"could be found."
            ],
        }

    # the guide is fetched first so the category decision is informed by what
    # the city is actually known for, not just its name
    _trace("places", "tool city_guide", time.perf_counter())
    guide_result = city_guide.invoke({"city": req.destination})
    guide = "" if guide_result.get("error") else guide_result.get("summary", "")

    categories, rationale = _choose_categories(
        req.destination, req.preferences, guide
    )
    _trace("places", f"categories {','.join(categories)}", time.perf_counter())

    candidates: list[PlaceCandidate] = []
    failures: list[str] = []

    for category in categories:
        _trace("places", f"tool find_places:{category}", time.perf_counter())
        result = find_places.invoke(
            {
                "lat": location["lat"],
                "lon": location["lon"],
                "category": category,
                "limit": PER_CATEGORY_LIMIT,
            }
        )
        if result.get("error"):
            failures.append(f"{category}: {result['error']}")
            continue
        for place in result.get("places", []):
            candidates.append(
                PlaceCandidate(
                    name=place["name"],
                    kind=place.get("kind"),
                    category=category,
                    lat=place["lat"],
                    lon=place["lon"],
                )
            )

    if not candidates:
        _trace("places", "FAILED no candidates", t0)
        metrics.record_outcome("places", "failed", time.perf_counter() - t0)
        return {
            "places": None,
            "errors": [
                f"places: no attractions found near {req.destination}"
                + (f" ({'; '.join(failures[:2])})" if failures else "")
            ],
        }

    result = PlacesResult(
        city=location["name"],
        guide=guide,
        candidates=candidates,
        categories=categories,
        rationale=rationale,
    )
    metrics.record_outcome("places", "done", time.perf_counter() - t0)
    _trace("places", f"done ({len(candidates)} candidates)", t0)

    return {
        "places": result,
        # evidence for the groundedness scorer: these are the only names the
        # itinerary is permitted to use
        "evidence": [
            {
                "agent": "places",
                "kind": "places",
                "names": [c.name for c in candidates],
                "coords": [[c.lat, c.lon] for c in candidates],
            }
        ],
    }
