"""Validation at the tool boundary.

The point is not that valid data passes — it is that malformed data is REJECTED
loudly instead of flowing into the cost arithmetic as a plausible number.
"""

import pytest

from tools.schemas import (
    DayWeather,
    FlightOffer,
    HotelOffer,
    Place,
    parse_money,
    validate_rows,
)


# --- money parsing (was previously the model's job) ---


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("$148", 148.0),
        ("$1,234", 1234.0),          # the case that could become 1.234
        ("$1,234.50", 1234.5),
        ("1234", 1234.0),
        (148, 148.0),
        (148.5, 148.5),
        ("€1 234", 1234.0),
        ("USD 99", 99.0),
    ],
)
def test_money_parsed_in_python(raw, expected):
    assert parse_money(raw) == expected


@pytest.mark.parametrize("raw", [None, "", "N/A", "n/a", "-", "free", "sold out", True, [], {}])
def test_non_prices_are_none_never_zero(raw):
    """0.0 would read as 'free' and quietly understate the trip."""
    assert parse_money(raw) is None


# --- hotels ---


def _hotel(**kw):
    base = {"name": "H", "price_per_night": "$100", "rating": 4.2}
    base.update(kw)
    return base


def test_string_rate_becomes_a_float():
    assert HotelOffer.model_validate(_hotel()).price_per_night == 100.0


def test_null_rate_is_rejected():
    """Observed live (Hotel Alfonso XIII). Previously the model had to invent
    a number for a required float field."""
    kept, dropped = validate_rows([_hotel(price_per_night=None)], HotelOffer)
    assert kept == [] and dropped == 1


def test_zero_rate_is_rejected():
    assert validate_rows([_hotel(price_per_night="$0")], HotelOffer)[1] == 1


def test_missing_hotel_class_is_tolerated():
    """14 of 15 live rows had a null hotel_class — apartments and hostels."""
    kept, dropped = validate_rows([_hotel(hotel_class=None)], HotelOffer)
    assert len(kept) == 1 and dropped == 0
    assert "hotel_class" not in kept[0], "absent stays absent"


def test_impossible_rating_is_rejected():
    assert validate_rows([_hotel(rating=47.0)], HotelOffer)[1] == 1


def test_one_bad_row_does_not_discard_the_good_ones():
    """A single odd listing must not turn a working search into 'no results'."""
    kept, dropped = validate_rows(
        [_hotel(name="Good"), _hotel(name="Bad", price_per_night=None), _hotel(name="Also good")],
        HotelOffer,
    )
    assert [k["name"] for k in kept] == ["Good", "Also good"]
    assert dropped == 1


# --- flights ---


def _flight(**kw):
    base = {
        "airline": "Aer Lingus", "price_usd": 1104, "price_total_usd": 2208,
        "departure_at": "2026-09-10 21:00", "stops": 1, "duration_minutes": 630,
    }
    base.update(kw)
    return base


def test_valid_flight_passes():
    assert FlightOffer.model_validate(_flight()).price_usd == 1104.0


def test_missing_total_is_rejected():
    """Regression: SerpApi rows omitted price_total_usd, so the two flight
    backends handed the agent different shapes."""
    row = _flight()
    del row["price_total_usd"]
    assert validate_rows([row], FlightOffer)[1] == 1


def test_zero_fare_is_rejected():
    assert validate_rows([_flight(price_usd=0)], FlightOffer)[1] == 1


def test_negative_stops_rejected():
    assert validate_rows([_flight(stops=-1)], FlightOffer)[1] == 1


def test_unparseable_fare_rejected():
    assert validate_rows([_flight(price_usd="call us")], FlightOffer)[1] == 1


# --- weather and places ---


def test_precipitation_out_of_range_rejected():
    row = {"date": "2026-09-10", "condition": "dry", "high_c": 25.0,
           "low_c": 18.0, "precipitation_chance": 5000}
    assert validate_rows([row], DayWeather)[1] == 1


def test_valid_weather_row_passes():
    row = {"date": "2026-09-10", "condition": "dry", "high_c": 25.0,
           "low_c": 18.0, "precipitation_chance": 20}
    assert validate_rows([row], DayWeather)[0][0]["high_c"] == 25.0


def test_impossible_coordinates_rejected():
    assert validate_rows(
        [{"name": "Nowhere", "lat": 999.0, "lon": 0.0}], Place
    )[1] == 1


def test_valid_place_passes():
    kept, dropped = validate_rows(
        [{"name": "Alcázar", "kind": "castle", "lat": 37.38, "lon": -5.99}], Place
    )
    assert dropped == 0 and kept[0]["name"] == "Alcázar"


def test_extra_fields_are_preserved():
    """Tools add context (connections, amenities) beyond the validated core."""
    kept, _ = validate_rows(
        [_flight(connections=["ORD->DUB", "DUB->LIS"])], FlightOffer
    )
    assert kept[0]["connections"] == ["ORD->DUB", "DUB->LIS"]
