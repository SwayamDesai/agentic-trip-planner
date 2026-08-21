"""Shared state + agent output schemas.

Design rule: every agent writes to exactly ONE top-level key of TripState.
No two agents touch the same key, so when we move to parallel execution in
phase 2 LangGraph can merge the partial updates without custom reducers.
"""

import operator
from typing import Annotated, Literal, Optional, TypedDict

from pydantic import BaseModel, Field

# --- trip request (user input) ---


class TripRequest(BaseModel):
    origin: str = Field(description="departure city or IATA code")
    destination: str
    start_date: str = Field(description="YYYY-MM-DD")
    end_date: str = Field(description="YYYY-MM-DD")
    travelers: int = 2
    budget_usd: Optional[int] = None
    preferences: list[str] = Field(default_factory=list)
    nights_chosen_by_system: bool = Field(
        default=False,
        description=(
            "True when the traveller gave no end date or night count and the "
            "system picked a trip length. Surfaced in the plan so an assumed "
            "duration is never mistaken for a stated one."
        ),
    )


# --- agent outputs ---


class FlightOption(BaseModel):
    airline: str
    departure: str = Field(
        description="Outbound departure, 'YYYY-MM-DD HH:MM (IATA)'. Never the return leg."
    )
    arrival: str = Field(
        description="Outbound arrival at the destination, 'YYYY-MM-DD HH:MM (IATA)'."
    )
    duration: str = Field(description="Total outbound duration, e.g. '8h 10m'.")
    stops: int
    price_usd: float = Field(
        description="Fare PER PERSON in USD, exactly as the tool returned it."
    )
    notes: str = Field(
        default="",
        description=(
            "Must state 'round trip' or 'one way', and the data source. If the "
            "fare came from cached data for a different date, say so here."
        ),
    )


class FlightResult(BaseModel):
    options: list[FlightOption] = Field(default_factory=list)
    reasoning: str = Field(
        default="",
        description=(
            "If options is empty, state that no real fare was found for this "
            "route and date. Never explain it away with a remembered price."
        ),
    )


class DayForecast(BaseModel):
    date: str
    condition: str
    high_c: float
    low_c: float
    precipitation_chance: int = Field(
        default=0,
        description=(
            "As returned by the tool. For climate_normals this is the share of "
            "past years with rain on this date, NOT a forecast probability."
        ),
    )


class WeatherResult(BaseModel):
    daily: list[DayForecast] = Field(default_factory=list)
    packing_advice: str = Field(
        default="",
        description=(
            "Grounded in the actual numbers. If the tool's source was "
            "climate_normals, state plainly that these are multi-year averages "
            "and not a forecast."
        ),
    )


class PlaceCandidate(BaseModel):
    """A real place returned by OpenStreetMap. Never model-generated."""

    name: str
    kind: Optional[str] = None
    category: str = Field(description="the search category that found it")
    lat: float
    lon: float


class PlaceSearchPlan(BaseModel):
    """Which kinds of place to look for at a destination.

    A judgement call, not a lookup: it depends on what the destination is known
    for, and travellers usually state no interests at all. Seville rewards
    historic architecture, Lyon food, Bergen the outdoors — a fixed mapping from
    stated preferences cannot know any of that.

    The vocabulary is closed, so the agent chooses among real search categories
    rather than inventing one the tool cannot serve.
    """

    categories: list[Literal["sights", "museums", "historic", "food"]] = Field(
        description=(
            "2 to 3 categories, most important first. Always include the ones "
            "the destination is actually known for."
        )
    )
    reasoning: str = Field(
        description="One sentence on why these suit this destination."
    )


class PlacesResult(BaseModel):
    """Destination research: candidates the itinerary must be built from.

    Produced by deterministic tool calls, not by a model — which is why it can
    run in the fan-out instead of waiting behind the join.
    """

    city: str
    guide: str = ""
    candidates: list[PlaceCandidate] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)
    rationale: str = Field(
        default="", description="why these categories were searched"
    )


class Activity(BaseModel):
    name: str = Field(
        description="Exactly as returned by find_places. Never a place from memory."
    )
    time_of_day: Literal["morning", "midday", "afternoon", "evening"] = Field(
        description=(
            "One of exactly: morning, midday, afternoon, evening. A closed set: "
            "free text drifted to 'lunch', 'brunch' and 'late afternoon'."
        )
    )
    duration_hours: float
    cost_usd: float = Field(
        default=0.0,
        description=(
            "YOUR ESTIMATE, per person — no tool provides entry prices. Use 0 "
            "for free. Whenever this is above 0 the notes MUST contain the word "
            "'estimated'."
        ),
    )
    indoor: bool = Field(
        default=False,
        description="True for indoor venues. Prefer these on days with rain.",
    )
    notes: str = Field(
        default="",
        description=(
            "Must contain 'estimated' if cost_usd > 0. Do not state opening "
            "hours or ticket prices as fact — no tool provides them."
        ),
    )


class DayPlan(BaseModel):
    date: str
    activities: list[Activity] = Field(default_factory=list)


class ItineraryResult(BaseModel):
    days: list[DayPlan] = Field(default_factory=list)
    reasoning: str = ""


class HotelOption(BaseModel):
    name: str
    area: str = Field(
        description=(
            "Neighbourhood ONLY if the property name makes it unambiguous; "
            "otherwise the city name. The tool returns no location field, so "
            "never guess a district."
        )
    )
    rating: float
    price_per_night_usd: float = Field(
        description="Per night for the WHOLE PARTY, as returned. Not per person."
    )
    notes: str = Field(
        default="",
        description=(
            "Property type and the amenities the tool listed. These are live "
            "rates, so do not call them estimates. Do not add amenities, "
            "cancellation terms or availability the tool did not return."
        ),
    )


class HotelResult(BaseModel):
    options: list[HotelOption] = Field(default_factory=list)
    reasoning: str = ""


class TripScope(BaseModel):
    """How long a trip should be, when the traveller did not say."""

    nights: int = Field(
        description="Recommended nights for this destination, between 1 and 14."
    )
    reasoning: str = Field(
        description="One sentence on why this length suits the destination."
    )


class CostBreakdown(BaseModel):
    """Computed in Python from the other agents' figures — never by a model.

    A budget that quietly mis-adds is worse than no budget at all, so nothing
    here is generated: every field is arithmetic over numbers the flight,
    hotel and itinerary agents already returned.
    """

    travelers: int
    nights: int
    tier: str = Field(
        default="cheapest",
        description=(
            "Which option was costed. 'cheapest' when a budget was given, since "
            "a cap is best met from the floor; 'mid' when none was, so the plan "
            "is neither bargain-hunting nor luxury."
        ),
    )
    travel_only_usd: float = Field(
        default=0.0,
        description="Flights + lodging. The unavoidable floor before activities.",
    )
    feasible: Optional[bool] = Field(
        default=None,
        description=(
            "False when the CHEAPEST flights and lodging alone exceed the "
            "budget, so no itinerary can bring the trip inside it. None when no "
            "budget was given."
        ),
    )
    flights_usd: float = Field(description="cheapest fare x travelers")
    lodging_usd: float = Field(description="cheapest nightly rate x nights, whole party")
    activities_usd: float = Field(description="itinerary entry costs x travelers")
    subtotal_usd: float
    budget_usd: Optional[int] = None
    over_under_usd: Optional[float] = Field(
        default=None, description="positive = over budget, negative = headroom"
    )
    within_budget: Optional[bool] = None
    missing: list[str] = Field(
        default_factory=list,
        description="Agents that produced nothing, so their cost is absent.",
    )


class BudgetAdvice(BaseModel):
    """The only part a model writes: interpretation, not numbers."""

    assessment: str = Field(
        description=(
            "Two sentences at most on whether this trip fits the budget. Quote "
            "only figures given to you; never compute a new total."
        )
    )
    suggestions: list[str] = Field(
        default_factory=list,
        description=(
            "Up to 4 concrete, actionable changes referencing real options that "
            "were provided — e.g. naming a cheaper hotel from the list. No "
            "generic advice like 'travel off-season'."
        ),
    )
    unbudgeted: list[str] = Field(
        default_factory=list,
        description=(
            "Costs this trip will incur that the subtotal does NOT include, "
            "e.g. food, local transport, baggage fees."
        ),
    )


class BudgetResult(BaseModel):
    breakdown: CostBreakdown
    advice: Optional[BudgetAdvice] = None


# --- graph state ---


class TripState(TypedDict, total=False):
    """Shared memory passed between nodes.

    `request` is read-only for agents. Each agent fills its own result key.
    `errors` is the one key every agent may write, so it carries an append
    reducer — otherwise concurrent writes in phase 2 would overwrite instead
    of accumulate.
    """

    request: TripRequest
    flight: Optional[FlightResult]
    weather: Optional[WeatherResult]
    places: Optional[PlacesResult]
    itinerary: Optional[ItineraryResult]
    budget: Optional[BudgetResult]
    hotels: Optional[HotelResult]
    errors: Annotated[list[str], operator.add]
    # deterministic checks that found something suspect but not fatal; same
    # append reducer, since several agents can report concurrently
    warnings: Annotated[list[str], operator.add]
    # What the tools actually returned this run, projected down to the fields
    # needed to check the answer against its sources. Append-reduced like
    # `errors`, since several agents write it concurrently.
    #
    # A projection rather than the raw payloads: these are checkpointed, and
    # storing full POI lists would bloat the store for no benefit.
    evidence: Annotated[list[dict], operator.add]
    plan: Optional[str]
    status: Optional[str]  # "ok" | "degraded" | "failed"
    # attached after the graph completes, not written by any node
    metrics: Optional[dict]
    cache: Optional[dict]
