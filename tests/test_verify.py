"""Deterministic checks on agent output.

Each test corresponds to a rule that previously existed only in a prompt.
"""

import pytest

from models import (
    Activity,
    DayPlan,
    FlightOption,
    FlightResult,
    HotelOption,
    HotelResult,
    ItineraryResult,
    TripRequest,
    WeatherResult,
    DayForecast,
)
from verify import (
    verify_budget_cap,
    verify_flights,
    verify_hotels,
    verify_itinerary,
    verify_weather,
)

REQ = TripRequest(
    origin="Chicago", destination="Seville",
    start_date="2026-10-05", end_date="2026-10-07",   # 3 days
    travelers=2, budget_usd=2600,
)

def _places(*specs):
    """Build the `places` state the itinerary verifier reads.

    Coordinates are deliberately close together: the verifier now also checks
    that a day's stops are geographically coherent, so distant fixtures would
    trip an unrelated warning.
    """
    from models import PlaceCandidate, PlacesResult

    return PlacesResult(
        city="Seville",
        candidates=[
            PlaceCandidate(
                name=name, kind=kind, category="sights",
                lat=37.38 + i * 0.001, lon=-5.99 + i * 0.001,
            )
            for i, (name, kind) in enumerate(specs)
        ],
    )


PLACES_STATE = {
    "places": _places(
        ("Real Alcázar de Sevilla", "castle"),
        ("Torre del Oro", "attraction"),
        ("Kyoto Tower", "attraction"),
    )
}


def _day(date, *names):
    return DayPlan(
        date=date,
        activities=[
            Activity(name=n, time_of_day="morning", duration_hours=1) for n in names
        ],
    )


def _itin(*days):
    return ItineraryResult(days=list(days))


# --- itinerary dates ---


def test_clean_itinerary_has_no_warnings():
    it = _itin(
        _day("2026-10-05", "Real Alcázar de Sevilla"),
        _day("2026-10-06", "Torre del Oro"),
        _day("2026-10-07", "Kyoto Tower"),
    )
    assert verify_itinerary(it, REQ, PLACES_STATE) == []


def test_date_outside_the_trip_is_flagged():
    it = _itin(_day("2026-11-30", "Torre del Oro"))
    assert any("outside the trip" in w for w in verify_itinerary(it, REQ, PLACES_STATE))


def test_duplicate_day_is_flagged():
    it = _itin(_day("2026-10-05", "Torre del Oro"), _day("2026-10-05", "Kyoto Tower"))
    assert any("duplicate day" in w for w in verify_itinerary(it, REQ, PLACES_STATE))


def test_missing_days_are_flagged():
    it = _itin(_day("2026-10-05", "Torre del Oro"))
    warnings = verify_itinerary(it, REQ, PLACES_STATE)
    assert any("no plan" in w for w in warnings)


def test_out_of_order_days_flagged():
    it = _itin(
        _day("2026-10-07", "Torre del Oro"),
        _day("2026-10-06", "Kyoto Tower"),
        _day("2026-10-05", "Real Alcázar de Sevilla"),
    )
    assert any("chronological" in w for w in verify_itinerary(it, REQ, PLACES_STATE))


def test_unparseable_date_flagged():
    it = _itin(_day("not-a-date", "Torre del Oro"))
    assert any("unparseable" in w for w in verify_itinerary(it, REQ, PLACES_STATE))


def test_empty_itinerary_flagged():
    assert verify_itinerary(_itin(), REQ, PLACES_STATE) == ["itinerary is empty"]


# --- duplicates ---


def test_repeated_activity_is_flagged():
    """Observed live: Arco del Postigo appeared on day 1 and day 4."""
    it = _itin(
        _day("2026-10-05", "Torre del Oro"),
        _day("2026-10-06", "Torre del Oro"),
        _day("2026-10-07", "Kyoto Tower"),
    )
    assert any("scheduled 2 times" in w for w in verify_itinerary(it, REQ, PLACES_STATE))


# --- provenance: the core anti-hallucination check ---


def test_place_not_in_tool_results_is_flagged():
    """Previously a prompt promise with nothing behind it."""
    it = _itin(
        _day("2026-10-05", "Real Alcázar de Sevilla"),
        _day("2026-10-06", "Invented Palace of Nowhere"),
        _day("2026-10-07", "Torre del Oro"),
    )
    warnings = verify_itinerary(it, REQ, PLACES_STATE)
    assert any("not found in tool results" in w for w in warnings)
    assert any("Invented Palace" in w for w in warnings)


def test_name_matching_tolerates_punctuation_and_case():
    """A false 'unverified' on a real place is noise, so matching is loose."""
    it = _itin(
        _day("2026-10-05", "real alcazar de sevilla"),
        _day("2026-10-06", "Torre  del   Oro!"),
        _day("2026-10-07", "Kyoto Tower"),
    )
    warnings = verify_itinerary(it, REQ, PLACES_STATE)
    assert not any("not found" in w for w in warnings)


def test_candidate_name_matching_is_exact_after_normalising():
    """Candidates carry one canonical name (English preferred at the tool
    boundary), so provenance is a normalised exact match."""
    it = _itin(
        _day("2026-10-05", "Kyoto Tower"),
        _day("2026-10-06", "Torre del Oro"),
        _day("2026-10-07", "Real Alcázar de Sevilla"),
    )
    assert not any("not found" in w for w in verify_itinerary(it, REQ, PLACES_STATE))


def test_no_places_means_no_provenance_claim():
    """With no candidate list to compare against, do not accuse the model."""
    it = _itin(
        _day("2026-10-05", "Anything"),
        _day("2026-10-06", "At All"),
        _day("2026-10-07", "Really"),
    )
    assert not any("not found" in w for w in verify_itinerary(it, REQ, {}))


# --- flights ---


FLIGHT_ROWS = [{"options": [
    {"airline": "Aer Lingus", "price_usd": 1104},
    {"airline": "Delta, Air France", "price_usd": 1416},
]}]


def _flight(airline, price):
    return FlightResult(options=[FlightOption(
        airline=airline, departure="d", arrival="a", duration="1h",
        stops=0, price_usd=price,
    )])


def test_matching_fare_passes():
    assert verify_flights(_flight("Aer Lingus", 1104.0), FLIGHT_ROWS) == []


def test_invented_fare_is_flagged():
    assert any("does not match" in w
               for w in verify_flights(_flight("Aer Lingus", 799.0), FLIGHT_ROWS))


def test_invented_airline_is_flagged():
    assert any("not in tool results" in w
               for w in verify_flights(_flight("Fictional Air", 1104.0), FLIGHT_ROWS))


def test_concatenated_airline_names_match():
    """Tools return 'Delta, Air France' for codeshares."""
    assert verify_flights(_flight("Delta, Air France", 1416.0), FLIGHT_ROWS) == []


def test_no_flight_payloads_means_no_claim():
    assert verify_flights(_flight("Anything", 1.0), []) == []


# --- hotels ---


HOTEL_ROWS = [{"options": [
    {"name": "Corral del Conde", "price_per_night": 118.0},
    {"name": "Hotel Patio de las Cruces", "price_per_night": 117.0},
]}]


def _hotel(name, rate):
    return HotelResult(options=[HotelOption(
        name=name, area="Seville", rating=4.2, price_per_night_usd=rate
    )])


def test_matching_hotel_passes():
    assert verify_hotels(_hotel("Corral del Conde", 118.0), HOTEL_ROWS) == []


def test_altered_rate_is_flagged():
    warnings = verify_hotels(_hotel("Corral del Conde", 89.0), HOTEL_ROWS)
    assert any("but the tool returned $118" in w for w in warnings)


def test_invented_hotel_is_flagged():
    assert any("not in tool results" in w
               for w in verify_hotels(_hotel("Hotel Imaginary", 100.0), HOTEL_ROWS))


def test_truncated_hotel_name_still_matches():
    assert verify_hotels(_hotel("Corral del Conde", 118.0), HOTEL_ROWS) == []


# --- weather ---


WEATHER_ROWS = [{"days": [
    {"date": "2026-10-05", "condition": "dry", "high_c": 30.0, "low_c": 18.0,
     "precipitation_chance": 0},
    {"date": "2026-10-06", "condition": "dry", "high_c": 31.0, "low_c": 19.0,
     "precipitation_chance": 0},
]}]


def _weather(*days):
    return WeatherResult(daily=[
        DayForecast(date=d, condition="dry", high_c=h, low_c=18.0) for d, h in days
    ])


def test_faithful_weather_passes():
    assert verify_weather(_weather(("2026-10-05", 30.0), ("2026-10-06", 31.0)),
                          WEATHER_ROWS) == []


def test_dropped_day_is_flagged():
    assert any("omitted" in w
               for w in verify_weather(_weather(("2026-10-05", 30.0)), WEATHER_ROWS))


def test_invented_day_is_flagged():
    warnings = verify_weather(
        _weather(("2026-10-05", 30.0), ("2026-10-06", 31.0), ("2026-10-09", 25.0)),
        WEATHER_ROWS,
    )
    assert any("not returned by the tool" in w for w in warnings)


def test_altered_temperature_is_flagged():
    warnings = verify_weather(
        _weather(("2026-10-05", 42.0), ("2026-10-06", 31.0)), WEATHER_ROWS
    )
    assert any("but the tool said" in w for w in warnings)


def test_small_rounding_difference_tolerated():
    assert verify_weather(_weather(("2026-10-05", 30.4), ("2026-10-06", 31.0)),
                          WEATHER_ROWS) == []


# --- budget cap ---


class _B:
    def __init__(self, activities):
        self.activities_usd = activities


def test_cap_respected_passes():
    assert verify_budget_cap(_B(100.0), {"remaining_usd": 500.0}) == []


def test_exceeded_cap_is_flagged_not_trimmed():
    warnings = verify_budget_cap(_B(700.0), {"remaining_usd": 500.0})
    assert any("over the $500 left" in w for w in warnings)
    assert any("nothing was removed" in w for w in warnings), (
        "essential items must not be trimmed to satisfy a number"
    )


def test_spending_when_already_over_is_flagged():
    warnings = verify_budget_cap(_B(92.0), {"remaining_usd": -577.0})
    assert any("already exceeded" in w for w in warnings)


def test_no_budget_means_no_cap_check():
    assert verify_budget_cap(_B(9999.0), None) == []


def test_zero_spend_never_breaches_a_cap():
    """Regression: a live run reported 'activities total $0, over the $-355
    left'. Spending nothing cannot overspend, even against a negative
    allowance."""
    assert verify_budget_cap(_B(0.0), {"remaining_usd": -355.0}) == []
    assert verify_budget_cap(_B(0.0), {"remaining_usd": 500.0}) == []


# --- meal pricing and day coherence ---


def test_priced_meal_is_flagged():
    """Food is reported as excluded from the budget, so pricing a restaurant in
    the itinerary would double-count it."""
    from models import PlaceCandidate, PlacesResult

    state = {"places": PlacesResult(city="Seville", candidates=[
        PlaceCandidate(name="Bar San Lorenzo", kind="restaurant",
                       category="food", lat=37.38, lon=-5.99)])}
    it = ItineraryResult(days=[DayPlan(date="2026-10-05", activities=[
        Activity(name="Bar San Lorenzo", time_of_day="evening",
                 duration_hours=2.0, cost_usd=30.0)])])
    warnings = verify_itinerary(it, REQ, state)
    assert any("must not be costed here" in w for w in warnings)


def test_free_meal_is_accepted():
    from models import PlaceCandidate, PlacesResult

    state = {"places": PlacesResult(city="Seville", candidates=[
        PlaceCandidate(name="Bar San Lorenzo", kind="restaurant",
                       category="food", lat=37.38, lon=-5.99)])}
    it = ItineraryResult(days=[DayPlan(date="2026-10-05", activities=[
        Activity(name="Bar San Lorenzo", time_of_day="evening",
                 duration_hours=2.0, cost_usd=0.0)])])
    assert not any("costed here" in w for w in verify_itinerary(it, REQ, state))


def test_scattered_day_is_flagged():
    """A day whose stops span a region is not a plan."""
    from models import PlaceCandidate, PlacesResult

    state = {"places": PlacesResult(city="Seville", candidates=[
        PlaceCandidate(name="Near", kind="attraction", category="sights",
                       lat=37.38, lon=-5.99),
        PlaceCandidate(name="Far", kind="attraction", category="sights",
                       lat=37.90, lon=-5.99)])}
    it = ItineraryResult(days=[DayPlan(date="2026-10-05", activities=[
        Activity(name="Near", time_of_day="morning", duration_hours=1.0),
        Activity(name="Far", time_of_day="afternoon", duration_hours=1.0)])])
    assert any("span" in w and "km" in w for w in verify_itinerary(it, REQ, state))


# --- breakdown arithmetic ---


def test_breakdown_arithmetic_is_checked():
    from verify import verify_breakdown_arithmetic
    from models import CostBreakdown

    good = CostBreakdown(travelers=2, nights=3, flights_usd=1000.0,
                         lodging_usd=300.0, activities_usd=50.0,
                         subtotal_usd=1350.0, travel_only_usd=1300.0)
    assert verify_breakdown_arithmetic(good) == []

    bad = good.model_copy(update={"subtotal_usd": 9999.0})
    assert any("does not equal its parts" in w
               for w in verify_breakdown_arithmetic(bad))


def test_over_under_arithmetic_is_checked():
    from verify import verify_breakdown_arithmetic
    from models import CostBreakdown

    bad = CostBreakdown(travelers=2, nights=3, flights_usd=1000.0,
                        lodging_usd=300.0, activities_usd=0.0,
                        subtotal_usd=1300.0, travel_only_usd=1300.0,
                        budget_usd=1000, over_under_usd=77.0)
    assert any("does not match" in w for w in verify_breakdown_arithmetic(bad))


# --- tightened airline matching ---


def test_short_invented_airline_no_longer_slips_through():
    """A bare substring test let a two-character name match almost anything."""
    rows = [{"options": [{"airline": "Aer Lingus", "price_usd": 1104}]}]
    result = _flight("Ae", 1104.0)
    assert any("not in tool results" in w for w in verify_flights(result, rows))


# --- closed-day check ---


def test_activity_on_a_closed_day_is_flagged():
    """The most concrete way a plan fails on the ground: a locked door."""
    from models import PlaceCandidate, PlacesResult

    state = {"places": PlacesResult(city="Seville", candidates=[
        PlaceCandidate(name="Museo Arqueológico", kind="museum", category="museums",
                       lat=37.38, lon=-5.99,
                       opening_hours="Tu-Sa 10:00-17:00")])}
    # 2026-10-05 is a Monday
    it = ItineraryResult(days=[DayPlan(date="2026-10-05", activities=[
        Activity(name="Museo Arqueológico", time_of_day="morning",
                 duration_hours=2.0)])])
    warnings = verify_itinerary(it, REQ, state)
    assert any("Monday" in w and "posted hours" in w for w in warnings)


def test_activity_on_an_open_day_is_not_flagged():
    from models import PlaceCandidate, PlacesResult

    state = {"places": PlacesResult(city="Seville", candidates=[
        PlaceCandidate(name="Museo Arqueológico", kind="museum", category="museums",
                       lat=37.38, lon=-5.99,
                       opening_hours="Tu-Sa 10:00-17:00")])}
    # 2026-10-06 is a Tuesday
    it = ItineraryResult(days=[DayPlan(date="2026-10-06", activities=[
        Activity(name="Museo Arqueológico", time_of_day="morning",
                 duration_hours=2.0)])])
    assert not any("posted hours" in w for w in verify_itinerary(it, REQ, state))


def test_untagged_place_is_never_flagged():
    """Most POIs carry no hours; absence must not become a false accusation."""
    from models import PlaceCandidate, PlacesResult

    state = {"places": PlacesResult(city="Seville", candidates=[
        PlaceCandidate(name="Torre del Oro", kind="attraction", category="sights",
                       lat=37.38, lon=-5.99)])}
    it = ItineraryResult(days=[DayPlan(date="2026-10-05", activities=[
        Activity(name="Torre del Oro", time_of_day="morning", duration_hours=1.0)])])
    assert not any("posted hours" in w for w in verify_itinerary(it, REQ, state))


def test_seasonal_hours_are_not_flagged():
    """An unparseable spec must not produce a confident 'closed'."""
    from models import PlaceCandidate, PlacesResult

    state = {"places": PlacesResult(city="Seville", candidates=[
        PlaceCandidate(name="Real Alcázar", kind="castle", category="historic",
                       lat=37.38, lon=-5.99,
                       opening_hours="Oct-Mar: 09:30-17:00; Apr-Sep: 09:30-19:00")])}
    it = ItineraryResult(days=[DayPlan(date="2026-10-05", activities=[
        Activity(name="Real Alcázar", time_of_day="morning", duration_hours=2.0)])])
    assert not any("posted hours" in w for w in verify_itinerary(it, REQ, state))
