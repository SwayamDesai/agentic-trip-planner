"""Tier 1 scorers: deterministic quality checks on a finished plan.

These are not unit tests. Unit tests ask "is the code correct"; these ask "is
the OUTPUT good" — a question with no fixed answer, so it is scored on
properties rather than compared to an expected plan.

Almost none of it needs a judge. `verify.py` already compares the model's
answer against the tool payloads that produced it; this module turns those
comparisons into numbers that can be tracked across prompt changes.

That is the point: six prompts changed during development, and one of them made
the model relabel paid attractions as free to satisfy a budget cap. It was
caught by eye. These scorers catch it mechanically.
"""

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

from costs import activity_allowance


def evidence_of(state: dict, kind: str) -> list[dict]:
    """Evidence records of one kind, from the run being scored.

    Sourced from `state["evidence"]` — what the tools returned during THIS run
    — rather than from the on-disk cache. Reading the cache was approximate:
    it could contain payloads from other runs, other dates, or nothing at all
    once a TTL expired, so a groundedness score could silently drift from what
    actually happened.
    """
    return [r for r in (state.get("evidence") or []) if r.get("kind") == kind]


def known_place_names(state: dict) -> set[str]:
    return {
        _norm(name)
        for record in evidence_of(state, "places")
        for name in record.get("names", [])
    } - {""}


def weather_source(state: dict) -> Optional[str]:
    """The source the weather tool actually reported.

    Previously inferred by string-matching condition text for "usually dry" /
    "often wet", which guessed at the answer instead of reading it.
    """
    records = evidence_of(state, "weather")
    return records[-1].get("source") if records else None


@dataclass
class Score:
    name: str
    passed: bool
    value: Optional[float] = None      # 0..1 where a ratio makes sense
    detail: str = ""
    critical: bool = False             # a failure here invalidates the plan


@dataclass
class Scorecard:
    scores: list[Score] = field(default_factory=list)

    def add(self, *scores: Score) -> None:
        self.scores.extend(scores)

    @property
    def critical_failures(self) -> list[Score]:
        return [s for s in self.scores if s.critical and not s.passed]

    @property
    def failures(self) -> list[Score]:
        return [s for s in self.scores if not s.passed]

    def as_dict(self) -> dict:
        graded = [s for s in self.scores if s.value is not None]
        return {
            "passed": not self.failures,
            "critical_passed": not self.critical_failures,
            "mean_value": (
                round(sum(s.value for s in graded) / len(graded), 3) if graded else None
            ),
            "scores": [
                {
                    "name": s.name,
                    "passed": s.passed,
                    "value": s.value,
                    "critical": s.critical,
                    "detail": s.detail,
                }
                for s in self.scores
            ],
        }


def _norm(text: str) -> str:
    from verify import _norm as shared

    return shared(text)


# --- groundedness: the central honesty property ---------------------------


def score_groundedness(state: dict) -> Score:
    """Fraction of scheduled activities that appear in the tool results.

    The prompt says "only schedule places the tools returned". This measures
    whether that held, rather than trusting it.
    """
    itinerary = state.get("itinerary")
    known = known_place_names(state)

    if not itinerary or not itinerary.days:
        return Score("groundedness", False, 0.0, "no itinerary", critical=True)
    if not known:
        # the itinerary agent never got place data, so there is nothing to
        # check against — not evidence of good behaviour, so say so plainly
        return Score(
            "groundedness", True, None,
            "not assessable: the run recorded no place evidence",
        )

    names = [a.name for d in itinerary.days for a in (d.activities or [])]
    if not names:
        return Score("groundedness", False, 0.0, "no activities", critical=True)

    grounded = [n for n in names if _norm(n) in known]
    ratio = len(grounded) / len(names)
    unverified = [n for n in names if _norm(n) not in known][:3]
    return Score(
        "groundedness",
        ratio == 1.0,
        round(ratio, 3),
        "all activities traced to tool output"
        if ratio == 1.0
        else f"{len(names) - len(grounded)} unverified, e.g. {', '.join(unverified)}",
        critical=True,
    )


def score_price_fidelity(state: dict) -> Score:
    """Every reported fare and rate must exist in the tool payloads."""
    problems = []
    checked = False

    flight = state.get("flight")
    fares = {
        round(float(row["price_usd"]))
        for record in evidence_of(state, "flights")
        for row in record.get("rows", [])
        if row.get("price_usd")
    }
    if flight and flight.options and fares:
        checked = True
        for o in flight.options:
            if round(o.price_usd) not in fares:
                problems.append(f"fare ${o.price_usd:.0f} not in tool output")

    hotels = state.get("hotels")
    rates = {
        round(float(row["price_per_night"]))
        for record in evidence_of(state, "hotels")
        for row in record.get("rows", [])
        if row.get("price_per_night")
    }
    if hotels and hotels.options and rates:
        checked = True
        for h in hotels.options:
            if round(h.price_per_night_usd) not in rates:
                problems.append(f"{h.name}: ${h.price_per_night_usd:.0f} not in tool output")

    if not checked:
        return Score(
            "price_fidelity", True, None,
            "not assessable: the run recorded no price evidence",
        )
    return Score(
        "price_fidelity",
        not problems,
        0.0 if problems else 1.0,
        "; ".join(problems[:3]) or "all prices traced to tool output",
        critical=True,
    )


# --- constraint adherence -------------------------------------------------


def score_day_coverage(state: dict) -> Score:
    """Every trip day planned, in order, exactly once."""
    req, itinerary = state.get("request"), state.get("itinerary")
    if not itinerary or not itinerary.days:
        return Score("day_coverage", False, 0.0, "no itinerary")

    start = date.fromisoformat(req.start_date)
    end = date.fromisoformat(req.end_date)
    expected = {
        (start + timedelta(days=i)).isoformat() for i in range((end - start).days + 1)
    }
    got = [d.date for d in itinerary.days]

    covered = expected & set(got)
    ratio = len(covered) / len(expected)
    issues = []
    if len(got) != len(set(got)):
        issues.append("duplicate days")
    if got != sorted(got):
        issues.append("out of order")
    if set(got) - expected:
        issues.append("days outside the trip")
    missing = sorted(expected - set(got))
    if missing:
        issues.append(f"{len(missing)} unplanned")

    return Score(
        "day_coverage", not issues and ratio == 1.0, round(ratio, 3),
        "; ".join(issues) or "every day planned once, in order",
    )


def score_no_duplicate_activities(state: dict) -> Score:
    itinerary = state.get("itinerary")
    if not itinerary or not itinerary.days:
        return Score("no_duplicates", False, 0.0, "no itinerary")
    names = [_norm(a.name) for d in itinerary.days for a in (d.activities or [])]
    dupes = {n for n in names if names.count(n) > 1}
    return Score(
        "no_duplicates", not dupes,
        round(1 - len(dupes) / max(len(set(names)), 1), 3),
        f"{len(dupes)} repeated" if dupes else "no repeats",
    )


def score_allowance_respected(state: dict) -> Score:
    """A stated budget is a cap, so activity spend must fit what is left."""
    allowance = activity_allowance(state)
    budget = state.get("budget")
    if allowance is None:
        return Score("allowance", True, None, "no budget given")
    if budget is None:
        return Score("allowance", False, 0.0, "no breakdown to check")

    spend = budget.breakdown.activities_usd
    remaining = allowance["remaining_usd"]

    if allowance["feasible"] is False:
        ok = spend == 0
        return Score(
            "allowance", ok, 1.0 if ok else 0.0,
            "free-only plan as required" if ok
            else f"spent ${spend:.0f} although the budget was already exceeded",
        )
    ok = spend <= max(remaining, 0)
    return Score(
        "allowance", ok, 1.0 if ok else 0.0,
        f"${spend:.0f} of ${remaining:.0f}" + ("" if ok else " — over cap"),
    )


def score_activity_density(state: dict) -> Score:
    """2-4 activities per day. More is over-packed; zero is a hole."""
    itinerary = state.get("itinerary")
    if not itinerary or not itinerary.days:
        return Score("density", False, 0.0, "no itinerary")
    # bounds imported from the prompt that states them, so the instruction and
    # the grader cannot drift apart — they did: 2-4 instructed, 1-5 graded
    from agents.itinerary_agent import MAX_PER_DAY, MIN_PER_DAY

    counts = [len(d.activities or []) for d in itinerary.days]
    ok_days = [c for c in counts if MIN_PER_DAY <= c <= MAX_PER_DAY]
    ratio = len(ok_days) / len(counts)
    return Score(
        "density", ratio == 1.0, round(ratio, 3),
        f"per-day counts {counts}",
    )


def score_weather_awareness(state: dict) -> Score:
    """On wet days, prefer indoor activities."""
    weather, itinerary = state.get("weather"), state.get("itinerary")
    if not (weather and weather.daily and itinerary and itinerary.days):
        return Score("weather_aware", True, None, "no weather to reason about")

    wet = {d.date for d in weather.daily if d.precipitation_chance >= 40}
    if not wet:
        return Score("weather_aware", True, None, "no wet days in range")

    checked = indoor = 0
    for day in itinerary.days:
        if day.date not in wet:
            continue
        for a in day.activities or []:
            checked += 1
            indoor += 1 if a.indoor else 0
    if not checked:
        return Score("weather_aware", True, None, "no activities on wet days")

    ratio = indoor / checked
    return Score(
        "weather_aware", ratio >= 0.5, round(ratio, 3),
        f"{indoor}/{checked} indoor on wet days",
    )


# --- honesty of labelling -------------------------------------------------


def score_honest_weather_label(state: dict) -> Score:
    """Climate normals must never be presented as a forecast."""
    weather = state.get("weather")
    if not weather:
        return Score("weather_label", True, None, "no weather")
    source = weather_source(state)
    if source != "climate_normals":
        return Score("weather_label", True, None, f"source={source}")

    advice = (weather.packing_advice or "").lower()
    disclosed = any(
        p in advice for p in ("not a forecast", "normal", "average", "historical")
    )
    return Score(
        "weather_label", disclosed, 1.0 if disclosed else 0.0,
        "normals disclosed" if disclosed
        else "multi-year averages presented without qualification",
        critical=True,
    )


def score_honest_activity_costs(state: dict) -> Score:
    """A non-zero activity cost is the model's estimate and must say so."""
    itinerary = state.get("itinerary")
    if not itinerary or not itinerary.days:
        return Score("cost_labels", True, None, "no itinerary")
    paid = [a for d in itinerary.days for a in (d.activities or []) if a.cost_usd]
    if not paid:
        return Score("cost_labels", True, None, "no paid activities")
    labelled = [a for a in paid if "estimat" in (a.notes or "").lower()]
    ratio = len(labelled) / len(paid)
    return Score(
        "cost_labels", ratio == 1.0, round(ratio, 3),
        f"{len(labelled)}/{len(paid)} priced activities marked estimated",
    )


def score_honest_fare_basis(state: dict) -> Score:
    """A round-trip fare shown as one-way understates the trip by half."""
    flight = state.get("flight")
    if not flight or not flight.options:
        return Score("fare_basis", True, None, "no flights")
    stated = [
        o for o in flight.options
        if any(p in (o.notes or "").lower() for p in ("round", "one way", "one-way"))
    ]
    ratio = len(stated) / len(flight.options)
    return Score(
        "fare_basis", ratio == 1.0, round(ratio, 3),
        f"{len(stated)}/{len(flight.options)} state the fare basis",
    )


def score_budget_verdict(state: dict) -> Score:
    """If a budget was given, the plan must say whether it fits."""
    budget = state.get("budget")
    req = state.get("request")
    if not req or req.budget_usd is None:
        return Score("budget_verdict", True, None, "no budget given")
    if budget is None:
        return Score("budget_verdict", False, 0.0, "no breakdown produced")
    b = budget.breakdown
    stated = b.within_budget is not None
    return Score(
        "budget_verdict", stated, 1.0 if stated else 0.0,
        f"within_budget={b.within_budget}, feasible={b.feasible}",
    )


# --- schema discipline ----------------------------------------------------


VALID_SLOTS = {"morning", "midday", "afternoon", "evening"}


def score_schema_discipline(state: dict) -> Score:
    itinerary = state.get("itinerary")
    if not itinerary or not itinerary.days:
        return Score("schema", False, 0.0, "no itinerary")
    slots = [a.time_of_day for d in itinerary.days for a in (d.activities or [])]
    bad = [s for s in slots if s not in VALID_SLOTS]
    return Score(
        "schema", not bad, round(1 - len(bad) / max(len(slots), 1), 3),
        f"invalid slots: {set(bad)}" if bad else "all time slots in the closed set",
    )


def score_plan(state: dict) -> Scorecard:
    """Run every tier-1 scorer over one finished plan.

    Everything needed is in `state`, including the evidence the run collected,
    so scoring is a pure function of that run — reproducible and independent of
    cache contents.
    """
    card = Scorecard()
    card.add(
        score_groundedness(state),
        score_price_fidelity(state),
        score_day_coverage(state),
        score_no_duplicate_activities(state),
        score_allowance_respected(state),
        score_activity_density(state),
        score_weather_awareness(state),
        score_honest_weather_label(state),
        score_honest_activity_costs(state),
        score_honest_fare_basis(state),
        score_budget_verdict(state),
        score_schema_discipline(state),
    )
    return card


# --- case expectations -----------------------------------------------------


def _actual(state: dict, key: str):
    """Resolve one expectation key against a finished run."""
    req = state.get("request")
    budget = state.get("budget")
    breakdown = budget.breakdown if budget else None

    if key == "status":
        return state.get("status")
    if key == "travelers":
        return req.travelers if req else None
    if key == "nights_chosen_by_system":
        return req.nights_chosen_by_system if req else None
    if key == "nights":
        return breakdown.nights if breakdown else None
    if key == "tier":
        return breakdown.tier if breakdown else None
    if key == "feasible":
        return breakdown.feasible if breakdown else None
    if key == "activities_usd":
        return breakdown.activities_usd if breakdown else None
    if key == "weather_source":
        return weather_source(state)
    raise KeyError(f"no resolver for expectation {key!r}")


def score_expectations(state: dict, expect: dict) -> list[Score]:
    """Check a case's declared properties against what the run produced.

    These are the case-specific assertions — "an infeasible budget must report
    feasible=False and spend nothing on activities" — that the generic scorers
    cannot express. Without them a golden set of edge cases only exercises the
    code; it does not check that each edge behaved as intended.
    """
    scores: list[Score] = []
    for key, wanted in (expect or {}).items():
        try:
            got = _actual(state, key)
        except KeyError as exc:
            scores.append(Score(f"expect:{key}", False, 0.0, str(exc)))
            continue
        ok = got == wanted
        scores.append(
            Score(
                f"expect:{key}",
                ok,
                1.0 if ok else 0.0,
                f"expected {wanted!r}, got {got!r}",
                critical=not ok,
            )
        )
    return scores
