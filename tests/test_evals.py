"""The scorers must themselves be correct, or the eval lies."""

import pytest

from evals.scorers import score_plan
from models import (
    Activity, BudgetResult, DayForecast, DayPlan, FlightOption, FlightResult,
    HotelOption, HotelResult, ItineraryResult, TripRequest, WeatherResult,
)
from costs import compute_breakdown

REQ = TripRequest(
    origin="Chicago", destination="Seville", start_date="2026-11-10",
    end_date="2026-11-12", travelers=2, budget_usd=6000,
)

# Evidence records as `agents.base._evidence` produces them — what the tools
# returned during the run, not whatever is in the cache afterwards.
EVIDENCE = [
    {"agent": "itinerary", "kind": "places",
     "names": ["Real Alcázar de Sevilla", "Torre del Oro", "Plaza del Triunfo"]},
    {"agent": "flight", "kind": "flights",
     "rows": [{"airline": "X", "price_usd": 1000}]},
    {"agent": "hotels", "kind": "hotels",
     "rows": [{"name": "Corral del Conde", "price_per_night": 100}]},
    {"agent": "weather", "kind": "weather", "source": "climate_normals",
     "rows": [{"date": "2026-11-10", "high_c": 18}]},
]


def _activity(name, cost=0.0, indoor=False, notes="", slot="morning"):
    return Activity(name=name, time_of_day=slot, duration_hours=1.0,
                    cost_usd=cost, indoor=indoor, notes=notes)


def _state(**over):
    from models import PlaceCandidate, PlacesResult

    state = {
        "request": REQ,
        "evidence": list(EVIDENCE),
        "places": PlacesResult(city="Seville", candidates=[
            PlaceCandidate(name=n, kind="attraction", category="sights",
                           lat=37.38 + i * 0.001, lon=-5.99)
            for i, n in enumerate(
                ["Real Alcázar de Sevilla", "Torre del Oro", "Plaza del Triunfo"]
            )
        ]),
        "flight": FlightResult(options=[FlightOption(
            airline="X", departure="d", arrival="a", duration="1h",
            stops=0, price_usd=1000.0, notes="round trip fare")]),
        "hotels": HotelResult(options=[HotelOption(
            name="Corral del Conde", area="Seville", rating=4.5,
            price_per_night_usd=100.0, notes="live rate")]),
        "weather": WeatherResult(
            daily=[DayForecast(date=f"2026-11-{d}", condition="usually dry",
                               high_c=18, low_c=9, precipitation_chance=0)
                   for d in (10, 11, 12)],
            packing_advice="These are climate normals, not a forecast."),
        # two activities per day: the density bound now comes from the prompt
        # that states it, and the prompt asks for at least two
        "itinerary": ItineraryResult(days=[
            DayPlan(date="2026-11-10", activities=[
                _activity("Plaza del Triunfo"), _activity("Torre del Oro")]),
            DayPlan(date="2026-11-11", activities=[
                _activity("Real Alcázar de Sevilla", 13.0, True, "estimated entry fee"),
                _activity("Plaza del Triunfo")]),
            DayPlan(date="2026-11-12", activities=[
                _activity("Torre del Oro"), _activity("Plaza del Triunfo")]),
        ]),
        "errors": [],
    }
    state.update(over)
    state["budget"] = BudgetResult(breakdown=compute_breakdown(state))
    return state


def _score(state, name):
    card = score_plan(state)
    return next(s for s in card.scores if s.name == name)


# --- a good plan passes everything ---


def test_clean_plan_has_no_failures():
    """A well-formed plan trips nothing except the deliberate repeats in this
    fixture, which exist only to satisfy the minimum activities per day."""
    card = score_plan(_state())
    unexpected = [s for s in card.failures if s.name != "no_duplicates"]
    assert unexpected == [], [f"{s.name}: {s.detail}" for s in unexpected]
    assert not card.critical_failures


# --- groundedness ---


def test_invented_place_is_caught():
    state = _state(itinerary=ItineraryResult(days=[
        DayPlan(date="2026-11-10", activities=[_activity("Plaza del Triunfo")]),
        DayPlan(date="2026-11-11", activities=[_activity("Palace of Nowhere")]),
        DayPlan(date="2026-11-12", activities=[_activity("Torre del Oro")]),
    ]))
    s = _score(state, "groundedness")
    assert not s.passed and s.critical
    assert s.value == 0.667, "scores are rounded to 3 places"
    assert "Palace of Nowhere" in s.detail


def test_groundedness_is_not_assessable_without_evidence():
    """No recorded place data means the score is unknown, not good.

    Reported as not-assessable rather than passing silently, so a run that
    lost its evidence cannot masquerade as a grounded one.
    """
    s = _score(_state(evidence=[]), "groundedness")
    assert s.value is None
    assert "not assessable" in s.detail


# --- price fidelity ---


def test_invented_fare_is_caught():
    state = _state(flight=FlightResult(options=[FlightOption(
        airline="X", departure="d", arrival="a", duration="1h",
        stops=0, price_usd=777.0, notes="round trip")]))
    s = _score(state, "price_fidelity")
    assert not s.passed and s.critical


def test_altered_hotel_rate_is_caught():
    state = _state(hotels=HotelResult(options=[HotelOption(
        name="Corral del Conde", area="Seville", rating=4.5,
        price_per_night_usd=55.0, notes="live rate")]))
    assert not _score(state, "price_fidelity").passed


# --- constraints ---


def test_missing_day_is_caught():
    state = _state(itinerary=ItineraryResult(days=[
        DayPlan(date="2026-11-10", activities=[_activity("Torre del Oro")])]))
    s = _score(state, "day_coverage")
    assert not s.passed and s.value == 0.333


def test_day_outside_trip_is_caught():
    state = _state(itinerary=ItineraryResult(days=[
        DayPlan(date="2026-12-25", activities=[_activity("Torre del Oro")])]))
    assert "outside the trip" in _score(state, "day_coverage").detail


def test_repeated_activity_is_caught():
    state = _state(itinerary=ItineraryResult(days=[
        DayPlan(date="2026-11-10", activities=[_activity("Torre del Oro")]),
        DayPlan(date="2026-11-11", activities=[_activity("Torre del Oro")]),
        DayPlan(date="2026-11-12", activities=[_activity("Plaza del Triunfo")]),
    ]))
    assert not _score(state, "no_duplicates").passed


def test_overspend_against_an_infeasible_budget_is_caught():
    """The exact regression: spending when travel already busted the budget."""
    req = REQ.model_copy(update={"budget_usd": 200})
    state = _state(request=req)
    s = _score(state, "allowance")
    assert not s.passed
    assert "already exceeded" in s.detail


def test_no_budget_means_allowance_is_not_graded():
    state = _state(request=REQ.model_copy(update={"budget_usd": None}))
    assert _score(state, "allowance").value is None


# --- honesty ---


def test_normals_presented_as_forecast_is_caught():
    """The critical honesty property: averages must not be called a forecast."""
    state = _state(weather=WeatherResult(
        daily=[DayForecast(date="2026-11-10", condition="usually dry",
                           high_c=18, low_c=9)],
        packing_advice="Expect sunshine all week."))
    s = _score(state, "weather_label")
    assert not s.passed and s.critical


def test_forecast_source_is_not_required_to_disclaim():
    """A real forecast may be called a forecast.

    The source is read from the evidence record, not guessed from the
    condition text — "clear" vs "usually dry" was never a reliable signal.
    """
    evidence = [r for r in EVIDENCE if r["kind"] != "weather"] + [
        {"agent": "weather", "kind": "weather", "source": "forecast", "rows": []}
    ]
    state = _state(
        evidence=evidence,
        weather=WeatherResult(
            daily=[DayForecast(date="2026-11-10", condition="clear", high_c=18, low_c=9)],
            packing_advice="Pack light layers."),
    )
    assert _score(state, "weather_label").passed


def test_unlabelled_estimate_is_caught():
    """The Alcázar regression in reverse: a price with no 'estimated' marker."""
    state = _state(itinerary=ItineraryResult(days=[
        DayPlan(date="2026-11-10", activities=[
            _activity("Real Alcázar de Sevilla", 13.0, True, notes="entry fee")]),
        DayPlan(date="2026-11-11", activities=[_activity("Torre del Oro")]),
        DayPlan(date="2026-11-12", activities=[_activity("Plaza del Triunfo")]),
    ]))
    assert not _score(state, "cost_labels").passed


def test_fare_basis_must_be_stated():
    state = _state(flight=FlightResult(options=[FlightOption(
        airline="X", departure="d", arrival="a", duration="1h",
        stops=0, price_usd=1000.0, notes="live price")]))
    assert not _score(state, "fare_basis").passed


# --- weather awareness ---


def test_outdoor_activities_on_wet_days_are_caught():
    state = _state(
        weather=WeatherResult(
            daily=[DayForecast(date="2026-11-10", condition="often wet",
                               high_c=14, low_c=8, precipitation_chance=80)],
            packing_advice="Climate normals, not a forecast."),
        itinerary=ItineraryResult(days=[DayPlan(
            date="2026-11-10",
            activities=[_activity("Plaza del Triunfo"), _activity("Torre del Oro")])]),
    )
    s = _score(state, "weather_aware")
    assert not s.passed and s.value == 0.0


# --- schema ---


def test_invalid_time_slot_is_caught():
    bad = Activity.model_construct(
        name="Torre del Oro", time_of_day="brunch", duration_hours=1.0,
        cost_usd=0.0, indoor=False, notes="")
    state = _state(itinerary=ItineraryResult(days=[
        DayPlan(date="2026-11-10", activities=[bad])]))
    assert not _score(state, "schema").passed


# --- scorecard shape ---


def test_scorecard_reports_critical_separately():
    state = _state(flight=FlightResult(options=[FlightOption(
        airline="X", departure="d", arrival="a", duration="1h",
        stops=0, price_usd=999.0, notes="round trip")]))
    card = score_plan(state)
    assert card.critical_failures
    assert card.as_dict()["critical_passed"] is False


# --- evidence projection ---


def test_evidence_projection_captures_what_scoring_needs():
    """`_evidence` must record enough to check an answer, and no more."""
    from agents.base import _evidence

    payloads = [
        {"places": [{"name": "Kyoto Tower", "local_name": "京都タワー"}]},
        {"options": [{"airline": "Iberia", "price_usd": 1413, "stops": 1}],
         "source": "google_flights_direct"},
        {"options": [{"name": "Corral del Conde", "price_per_night": 118.0}],
         "source": "google_hotels_live"},
        {"days": [{"date": "2026-11-10", "high_c": 18.0, "condition": "usually dry"}],
         "source": "climate_normals"},
    ]
    records = _evidence("itinerary", payloads)
    kinds = {r["kind"] for r in records}
    assert kinds == {"places", "flights", "hotels", "weather"}

    places = next(r for r in records if r["kind"] == "places")
    assert "Kyoto Tower" in places["names"]
    assert "京都タワー" in places["names"], "local names count as verified too"

    weather = next(r for r in records if r["kind"] == "weather")
    assert weather["source"] == "climate_normals", "source recorded, not inferred"


def test_evidence_skips_failed_tool_calls():
    """An error payload is not evidence of anything."""
    from agents.base import _evidence

    assert _evidence("flight", [{"error": "quota spent", "error_kind": "rate_limited"}]) == []


def test_flights_and_hotels_are_distinguished():
    """Both use an `options` key; the rate field tells them apart."""
    from agents.base import _evidence

    hotels = _evidence("hotels", [{"options": [{"name": "H", "price_per_night": 100}]}])
    flights = _evidence("flight", [{"options": [{"airline": "A", "price_usd": 500}]}])
    assert hotels[0]["kind"] == "hotels"
    assert flights[0]["kind"] == "flights"


# --- case expectations (previously never checked) ---


def test_expectations_are_actually_asserted():
    """Regression: `Case.expect` was declared on all 12 cases and read by
    nothing, so the golden set exercised code without checking behaviour."""
    from evals.scorers import score_expectations

    state = _state()
    passing = score_expectations(state, {"travelers": 2})
    assert passing[0].passed

    failing = score_expectations(state, {"travelers": 99})
    assert not failing[0].passed
    assert failing[0].critical, "a violated expectation invalidates the case"
    assert "expected 99, got 2" in failing[0].detail


def test_unknown_expectation_key_fails_loudly():
    """A typo in a case must not silently pass."""
    from evals.scorers import score_expectations

    scores = score_expectations(_state(), {"nonexistent_property": 1})
    assert not scores[0].passed


def test_every_case_expectation_has_a_resolver():
    """Guards against a case declaring a property the runner cannot check."""
    from evals.cases import CASES
    from evals.scorers import score_expectations

    for case in CASES:
        for score in score_expectations(_state(), case.expect):
            assert "no resolver" not in score.detail, (
                f"{case.id} expects {score.name} but nothing resolves it"
            )
