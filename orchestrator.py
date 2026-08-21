"""Trip planner graph.

Phase 2 topology — fan out the independent agents, then join:

            +-> flight  -+
    START --+-> weather --+-> itinerary -> synthesize -> END
            +-> hotels  -+

flight, weather and hotels share no inputs, so they run concurrently. Only
`itinerary` has a real dependency: it reads the weather outlook to place indoor
activities on wet days, so it must sit downstream of the fan-out.

The join is safe without custom merge logic because of two choices made in
phase 1: every agent writes exactly one, distinct top-level state key, and the
one shared key (`errors`) carries an append reducer. Three agents can fail
concurrently without overwriting each other's messages.

Wall clock is now the slowest fan-out agent plus itinerary, not the sum of all
four.
"""

from langgraph.graph import END, START, StateGraph

from agents.budget_agent import budget_agent
from agents.flight_agent import flight_agent
from agents.hotel_agent import hotel_agent
from agents.itinerary_agent import itinerary_agent
from agents.places_agent import places_agent
from agents.weather_agent import weather_agent
from models import TripRequest, TripState
from providers import metrics, tracing
from providers.cache import stats as cache_stats
from providers.memory import get_checkpointer, thread_id
from status import plan_status


def synthesize(state: TripState) -> TripState:
    """Render the collected agent output as markdown.

    Deliberately not an LLM call: this is presentation over data the agents
    already produced, so a formatter cannot invent prices the way a model can.
    """
    req = state["request"]
    status, notes = plan_status(state)

    out: list[str] = [
        f"# {req.origin} to {req.destination}",
        f"{req.start_date} to {req.end_date} | "
        f"{req.travelers} traveler(s)"
        + (f" | budget ${req.budget_usd}" if req.budget_usd else "")
        + (
            "\n\n_Trip length was not specified; the system chose it._"
            if req.nights_chosen_by_system
            else ""
        ),
    ]

    flight = state.get("flight")
    if flight and flight.options:
        out.append("\n## Flights")
        for o in flight.options:
            out.append(
                f"- **{o.airline}** ${o.price_usd:.0f} — {o.departure} to "
                f"{o.arrival}, {o.duration}, "
                f"{'nonstop' if o.stops == 0 else f'{o.stops} stop(s)'}"
                + (f". {o.notes}" if o.notes else "")
            )

    hotels = state.get("hotels")
    if hotels and hotels.options:
        out.append("\n## Where to stay")
        for h in hotels.options:
            out.append(
                f"- **{h.name}** ({h.area}) ${h.price_per_night_usd:.0f}/night, "
                f"rated {h.rating}"
                + (f". {h.notes}" if h.notes else "")
            )

    weather = state.get("weather")
    if weather and weather.daily:
        out.append("\n## Weather")
        for d in weather.daily:
            out.append(
                f"- {d.date}: {d.condition}, {d.low_c:.0f}-{d.high_c:.0f}C, "
                f"{d.precipitation_chance}% rain"
            )
        if weather.packing_advice:
            out.append(f"\n_Packing:_ {weather.packing_advice}")

    itinerary = state.get("itinerary")
    if itinerary and itinerary.days:
        out.append("\n## Itinerary")
        for day in itinerary.days:
            out.append(f"\n**{day.date}**")
            for a in day.activities:
                cost = f"${a.cost_usd:.0f}" if a.cost_usd else "free"
                out.append(
                    f"- _{a.time_of_day}_ — {a.name} "
                    f"({a.duration_hours:g}h, {cost}"
                    f"{', indoor' if a.indoor else ''})"
                    + (f". {a.notes}" if a.notes else "")
                )

    budget = state.get("budget")
    if budget:
        b = budget.breakdown
        out.append("\n## Cost")
        # label must track the tier actually costed, or a mid-range figure
        # gets presented as the cheapest available
        pick = "cheapest" if b.tier == "cheapest" else "mid-range"
        out.append(
            f"- Flights: ${b.flights_usd:.0f} "
            f"({pick} x {b.travelers} traveler(s))"
        )
        out.append(
            f"- Lodging: ${b.lodging_usd:.0f} "
            f"({pick} x {b.nights} night(s), whole party)"
        )
        out.append(f"- Activities: ${b.activities_usd:.0f}")
        out.append(f"- **Subtotal: ${b.subtotal_usd:.0f}**")
        if b.budget_usd is not None:
            delta = abs(b.over_under_usd or 0)
            verdict = (
                f"within budget, ${delta:.0f} to spare"
                if b.within_budget
                else f"**over budget by ${delta:.0f}**"
            )
            out.append(f"- Budget ${b.budget_usd} — {verdict}")
            if b.feasible is False:
                out.append(
                    f"- **This trip is not achievable within ${b.budget_usd}.** "
                    f"The cheapest flights and lodging alone come to "
                    f"${b.travel_only_usd:.0f}, before any activities."
                )
        elif b.tier == "mid":
            out.append(
                "- _No budget given, so this is costed at mid-range options "
                "rather than the cheapest available._"
            )
        if b.missing:
            out.append(
                f"- _Incomplete: no data for {', '.join(b.missing)}, "
                f"so the subtotal is understated._"
            )

        advice = budget.advice
        if advice:
            out.append(f"\n{advice.assessment}")
            if advice.suggestions:
                out.append("\n**To save money**")
                out.extend(f"- {s}" for s in advice.suggestions)
            if advice.unbudgeted:
                out.append(f"\n_Not included: {', '.join(advice.unbudgeted)}._")

    if status == "failed":
        out.insert(
            1,
            "\n> **INCOMPLETE PLAN — required data is missing.**\n"
            + "\n".join(f"> - {n}" for n in notes)
            + "\n>\n> What follows is partial. Do not rely on it as a full plan.",
        )
    elif status == "degraded":
        out.insert(
            1,
            "\n> **Partial plan.** "
            + " ".join(f"Missing {n}." for n in notes),
        )

    warnings = state.get("warnings") or []
    if warnings:
        out.append("\n## Checks that flagged something")
        out.extend(f"- {w}" for w in warnings)

    errors = state.get("errors") or []
    if errors:
        out.append("\n## Agents that failed")
        out.extend(f"- {e}" for e in errors)

    return {"plan": "\n".join(out), "status": status}


def build_graph(checkpointer=None):
    g = StateGraph(TripState)

    g.add_node("weather", weather_agent)
    g.add_node("places", places_agent)
    g.add_node("itinerary", itinerary_agent)
    g.add_node("flight", flight_agent)
    g.add_node("hotels", hotel_agent)
    g.add_node("budget", budget_agent)
    g.add_node("synthesize", synthesize)

    # fan out: no shared inputs, so these three run concurrently
    # `places` joins the fan-out: it needs only the destination, and makes no
    # model calls at all, so it costs nothing to run alongside the others.
    for node in ("flight", "weather", "hotels", "places"):
        g.add_edge(START, node)

    # join: itinerary waits for all three. It only needs `weather`, but edges
    # from the other two keep it as a single barrier rather than a second
    # fan-out stage, and they cost no extra wall clock.
    for node in ("flight", "weather", "hotels", "places"):
        g.add_edge(node, "itinerary")

    # budget needs every other agent's figures, so it sits last
    g.add_edge("itinerary", "budget")
    g.add_edge("budget", "synthesize")
    g.add_edge("synthesize", END)

    return g.compile(checkpointer=checkpointer)


def plan_trip(request: TripRequest, remember: bool = True) -> TripState:
    """Run the graph, resuming any saved state for this same trip.

    With `remember` on, results are checkpointed per trip. Rerunning the same
    request replays the graph but the node guards skip agents that already
    succeeded, so only previously-failed agents spend tokens again.

    The whole run is wrapped in one trace and one metrics collector, so a plan
    can be answered for after the fact: which agent spent what, where it failed,
    and how much of it came from cache.
    """
    metrics.reset()

    with tracing.trace_run(
        "plan_trip",
        origin=request.origin,
        destination=request.destination,
        start_date=request.start_date,
        end_date=request.end_date,
        travelers=request.travelers,
        budget_usd=request.budget_usd,
    ):
        if not remember:
            state = build_graph().invoke(
                {"request": request, "errors": None, "warnings": None,
                 "evidence": None}
            )
        else:
            checkpointer = get_checkpointer()
            graph = build_graph(checkpointer)
            config = {"configurable": {"thread_id": thread_id(request)}}

            saved = graph.get_state(config)
            prior = saved.values if saved and saved.values else {}

            # Carry forward what earlier runs produced, and CLEAR the per-run
            # channels. All three describe one run, not the trip:
            #   errors    a failure the last run had may have since succeeded
            #   warnings  otherwise the same finding is reported once per resume
            #   evidence  stale place data could validate an itinerary the
            #             current run's places node never returned — the exact
            #             imprecision that moving off the cache was meant to fix
            #
            # None is the reset signal; `[]` would append nothing and leave the
            # checkpointed value in place.
            seed: TripState = {
                "request": request,
                "errors": None,
                "warnings": None,
                "evidence": None,
            }
            for key in ("flight", "weather", "hotels", "places", "itinerary"):
                if prior.get(key) is not None:
                    seed[key] = prior[key]

            state = graph.invoke(seed, config)

    run = metrics.finish()
    state = dict(state)
    state["metrics"] = run.as_dict()
    state["cache"] = cache_stats()
    return state
