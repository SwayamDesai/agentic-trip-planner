"""Deterministic checks on agent output.

Every prompt rule that matters is also checked here. A rule that exists only in
a prompt is a request; a rule checked against the tool payloads is enforced.

Findings are returned as warnings, never corrections. Two reasons:

  - a wrong auto-fix is worse than a visible caveat. Name matching is fuzzy,
    so silently deleting an "unverified" activity could remove a real place.
  - the user asked that essential items not be trimmed to satisfy a budget.

So the plan is published with its caveats attached and the reader decides.
"""

import unicodedata
from datetime import date, timedelta
from typing import Iterable, Optional

# Tool payload keys that carry the source rows an agent was supposed to use.
_PLACE_KEYS = ("places",)
_OPTION_KEYS = ("options",)

# A day of sightseeing should be coherent on foot or a short hop. Generous, so
# only a genuinely scattered day trips it.
MAX_DAY_SPREAD_KM = 15.0


def _km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    from tools.airports import haversine_km

    return haversine_km(lat1, lon1, lat2, lon2)


def _norm(text: str) -> str:
    """Loose name key for comparing a reported name against a source name.

    Deliberately forgiving — a false "unverified" warning on a real place is
    noise, so only a clearly different name should fail to match. Two cases the
    naive version got wrong:

      "Alcázar"    stripping non-ASCII gave "alczar", which never matched the
                   accent-free spelling a model tends to produce. NFKD folds
                   the accent instead of deleting the letter.
      "京都タワー"   stripping non-ASCII emptied it entirely, so every
                   non-Latin place looked unverified.
    """
    decomposed = unicodedata.normalize("NFKD", text or "")
    return "".join(
        ch.lower()
        for ch in decomposed
        if ch.isalnum() and not unicodedata.combining(ch)
    )


def _rows(payloads: Iterable[dict], keys: tuple[str, ...]) -> list[dict]:
    out = []
    for payload in payloads or []:
        if not isinstance(payload, dict):
            continue
        for key in keys:
            value = payload.get(key)
            if isinstance(value, list):
                out.extend(r for r in value if isinstance(r, dict))
    return out


# --- itinerary ------------------------------------------------------------


def verify_itinerary(itinerary, req, state: dict) -> list[str]:
    """Check dates, duplicates, provenance, meal pricing and day coherence.

    Takes the whole state rather than raw payloads: the candidate list now
    comes from the `places` node, which also carries the coordinates and OSM
    kinds needed for the geographic and meal-cost checks.
    """
    problems: list[str] = []
    days = list(itinerary.days or [])
    if not days:
        return ["itinerary is empty"]

    # dates must sit inside the trip, in order, without repeats
    try:
        start = date.fromisoformat(req.start_date)
        end = date.fromisoformat(req.end_date)
    except ValueError:
        start = end = None

    seen_dates: set[str] = set()
    parsed: list[Optional[date]] = []
    for day in days:
        try:
            parsed.append(date.fromisoformat(day.date))
        except ValueError:
            problems.append(f"unparseable date {day.date!r}")
            parsed.append(None)
            continue
        if day.date in seen_dates:
            problems.append(f"duplicate day {day.date}")
        seen_dates.add(day.date)

    if start and end:
        for value in filter(None, parsed):
            if not (start <= value <= end):
                problems.append(
                    f"day {value.isoformat()} falls outside the trip "
                    f"({req.start_date} to {req.end_date})"
                )
        expected = {
            (start + timedelta(days=i)).isoformat()
            for i in range((end - start).days + 1)
        }
        missing = sorted(expected - seen_dates)
        if missing:
            problems.append(
                f"{len(missing)} trip day(s) have no plan: {', '.join(missing[:4])}"
                + (" ..." if len(missing) > 4 else "")
            )

    ordered = [v for v in parsed if v]
    if ordered != sorted(ordered):
        problems.append("days are not in chronological order")

    # activities must not repeat across the trip
    counts: dict[str, list[str]] = {}
    for day in days:
        for activity in day.activities or []:
            counts.setdefault(_norm(activity.name), []).append(activity.name)
    for repeats in counts.values():
        if len(repeats) > 1:
            problems.append(
                f"{repeats[0]!r} is scheduled {len(repeats)} times"
            )

    places = state.get("places")
    catalogue = {_norm(c.name): c for c in (places.candidates if places else [])}
    known = set(catalogue) - {""}

    # meals must not be priced: food is reported as excluded from the budget,
    # so pricing a restaurant here would double-count it
    from costs import is_meal

    for day in days:
        for activity in day.activities or []:
            candidate = catalogue.get(_norm(activity.name))
            if candidate is None:
                continue
            if activity.cost_usd and is_meal(candidate.kind, candidate.category):
                problems.append(
                    f"{activity.name!r} is a meal priced at "
                    f"${activity.cost_usd:.0f}, but food is excluded from the "
                    f"budget and must not be costed here"
                )

    # a day should be walkable: stops spread across a region are not a plan
    for day in days:
        coords = [
            (catalogue[_norm(a.name)].lat, catalogue[_norm(a.name)].lon)
            for a in (day.activities or [])
            if _norm(a.name) in catalogue
        ]
        if len(coords) < 2:
            continue
        spread = max(
            _km(a[0], a[1], b[0], b[1]) for a in coords for b in coords
        )
        if spread > MAX_DAY_SPREAD_KM:
            problems.append(
                f"{day.date}: stops span {spread:.0f}km, more than the "
                f"{MAX_DAY_SPREAD_KM:.0f}km a single day should cover"
            )

    if known:
        unverified = [
            a.name
            for d in days
            for a in (d.activities or [])
            if _norm(a.name) not in known
        ]
        if unverified:
            problems.append(
                f"{len(unverified)} activity name(s) not found in tool results: "
                + ", ".join(unverified[:4])
                + (" ..." if len(unverified) > 4 else "")
            )

    return problems


# --- flights and lodging --------------------------------------------------


def verify_flights(result, payloads: list[dict]) -> list[str]:
    """Every reported fare must exist in what the tool returned."""
    rows = _rows(payloads, _OPTION_KEYS)
    if not rows:
        return []
    known_prices = {round(float(r["price_usd"])) for r in rows if r.get("price_usd")}
    known_airlines = {_norm(str(r.get("airline", ""))) for r in rows}
    known_airlines.discard("")

    problems = []
    for option in result.options or []:
        if known_prices and round(option.price_usd) not in known_prices:
            problems.append(
                f"fare ${option.price_usd:.0f} for {option.airline} does not "
                f"match any price the tool returned"
            )
        # Airlines arrive concatenated for codeshares ("Delta, Air France"), so
        # an exact match is too strict. But a bare substring test let a
        # two-character invented name match almost anything, so the reported
        # name must be a decent-length prefix-or-whole of a known one.
        reported = _norm(option.airline)
        if known_airlines and not any(
            reported == a or (len(reported) >= 4 and reported in a)
            for a in known_airlines
        ):
            problems.append(f"airline {option.airline!r} was not in tool results")
    return problems


def verify_hotels(result, payloads: list[dict]) -> list[str]:
    """Every reported property and rate must exist in what the tool returned."""
    rows = _rows(payloads, _OPTION_KEYS)
    if not rows:
        return []
    by_name = {
        _norm(str(r.get("name", ""))): r.get("price_per_night")
        for r in rows
        if r.get("name")
    }

    problems = []
    for option in result.options or []:
        key = _norm(option.name)
        match = next((v for k, v in by_name.items() if k == key), None)
        if match is None:
            # allow a containment match: names are long and get truncated
            match = next(
                (v for k, v in by_name.items() if key and (key in k or k in key)),
                None,
            )
            if match is None:
                problems.append(f"hotel {option.name!r} was not in tool results")
                continue
        if match and round(float(match)) != round(option.price_per_night_usd):
            problems.append(
                f"{option.name}: reported ${option.price_per_night_usd:.0f}/night "
                f"but the tool returned ${float(match):.0f}"
            )
    return problems


# --- weather --------------------------------------------------------------


def verify_weather(result, payloads: list[dict]) -> list[str]:
    """The agent must not add, drop or alter the days the tool returned."""
    rows = []
    for payload in payloads or []:
        if isinstance(payload, dict) and isinstance(payload.get("days"), list):
            rows = payload["days"]
    if not rows:
        return []

    tool_days = {r["date"]: r for r in rows if isinstance(r, dict) and r.get("date")}
    reported = {d.date for d in result.daily or []}

    problems = []
    invented = sorted(reported - set(tool_days))
    if invented:
        problems.append(f"day(s) not returned by the tool: {', '.join(invented[:4])}")
    dropped = sorted(set(tool_days) - reported)
    if dropped:
        problems.append(f"day(s) omitted from the tool result: {', '.join(dropped[:4])}")

    for day in result.daily or []:
        source = tool_days.get(day.date)
        if source and abs(float(source.get("high_c", day.high_c)) - day.high_c) > 1.0:
            problems.append(
                f"{day.date}: reported high {day.high_c}C but the tool said "
                f"{source['high_c']}C"
            )
    return problems


# --- budget ---------------------------------------------------------------


def verify_breakdown_arithmetic(breakdown) -> list[str]:
    """The subtotal must equal its parts.

    Cheap, and it guards the one number a traveller actually acts on. A silent
    mis-add here would be invisible: every component looks plausible.
    """
    parts = breakdown.flights_usd + breakdown.lodging_usd + breakdown.activities_usd
    if abs(parts - breakdown.subtotal_usd) > 0.01:
        return [
            f"subtotal ${breakdown.subtotal_usd:.2f} does not equal its parts "
            f"(${parts:.2f})"
        ]
    if breakdown.budget_usd is not None and breakdown.over_under_usd is not None:
        expected = breakdown.subtotal_usd - breakdown.budget_usd
        if abs(expected - breakdown.over_under_usd) > 0.01:
            return [
                f"over/under ${breakdown.over_under_usd:.2f} does not match "
                f"subtotal minus budget (${expected:.2f})"
            ]
    return []


def verify_budget_cap(breakdown, allowance: Optional[dict]) -> list[str]:
    """Flag a blown activity allowance. Never trims: the user asked that
    essential items not be removed to satisfy a number."""
    if not allowance or breakdown.activities_usd is None:
        return []

    remaining = allowance["remaining_usd"]
    if breakdown.activities_usd <= 0:
        # spending nothing cannot breach a cap, even a negative one
        return []
    if remaining <= 0:
        return [
            f"activities total ${breakdown.activities_usd:.0f} although travel "
            f"and lodging already exceeded the budget by ${abs(remaining):.0f}; "
            f"nothing was removed"
        ]
    if remaining > 0 and breakdown.activities_usd > remaining:
        return [
            f"activities total ${breakdown.activities_usd:.0f}, over the "
            f"${remaining:.0f} left after travel and lodging; nothing was removed"
        ]
    return []
