"""Airport code resolution and route validation.

A wrong-but-real code is the failure this guards: it returns valid fares for
the wrong cities, and nothing downstream can tell.
"""

import pytest

from tools.airports import (
    PLAUSIBLE_KM,
    airports_near,
    haversine_km,
    lookup,
    validate_route,
)


@pytest.fixture(autouse=True)
def stub_geocode(monkeypatch):
    """Offline city coordinates, so no network and no Nominatim throttle."""
    cities = {
        "chicago": (41.88, -87.63),
        "seville": (37.39, -5.98),
        "granada": (37.18, -3.60),
        "lisbon": (38.72, -9.14),
    }

    def fake(place):
        hit = cities.get(place.strip().lower())
        if hit is None:
            return {"error": f"could not geocode {place!r}"}
        return {"name": place, "lat": hit[0], "lon": hit[1]}

    monkeypatch.setattr("tools.airports.geocode", fake)


def test_distance_is_sane():
    # Chicago to Lisbon is roughly 6,400 km
    assert 6200 < haversine_km(41.88, -87.63, 38.72, -9.14) < 6700


def test_real_code_resolves():
    assert "Hare" in lookup("ORD")["name"]


@pytest.mark.parametrize("code", ["ZZZ", "QQQ", "", "O", "ORDD", None])
def test_bad_codes_do_not_resolve(code):
    assert lookup(code) is None


def test_lookup_is_case_insensitive():
    assert lookup("ord") == lookup("ORD")


def test_correct_route_passes():
    assert validate_route("ORD", "SVQ", "Chicago", "Seville") == []


def test_wrong_city_code_is_caught():
    """The real failure mode: SVQ is a valid code, just not for Granada."""
    problems = validate_route("ORD", "SVQ", "Chicago", "Granada")
    assert len(problems) == 1
    assert "different place" in problems[0]
    assert "GRX" in problems[0], "must name the correct code so it can self-fix"


def test_nonexistent_code_is_caught_with_suggestions():
    problems = validate_route("ORD", "QQQ", "Chicago", "Granada")
    assert "not a real IATA code" in problems[0]
    assert "GRX" in problems[0]


def test_both_ends_can_fail_independently():
    assert len(validate_route("QQQ", "ZZZ", "Chicago", "Granada")) == 2


def test_distant_but_legitimate_airport_is_allowed():
    """Secondary airports are often far out; only a different CITY is an error."""
    assert PLAUSIBLE_KM >= 100, "must tolerate e.g. Frankfurt-Hahn at ~110km"


def test_validation_skipped_without_city_names():
    """Codes alone can still be checked for existence, not for correctness."""
    assert validate_route("ORD", "SVQ") == []
    assert validate_route("ORD", "QQQ") != []


def test_unknown_city_does_not_produce_a_false_alarm():
    """If the city cannot be geocoded, we cannot judge the code — stay quiet."""
    assert validate_route("ORD", "SVQ", "Chicago", "Atlantis") == []


def test_airports_near_is_sorted_by_distance():
    found = airports_near(41.88, -87.63, 60.0)
    assert found == sorted(found, key=lambda a: a["distance_km"])
    assert "ORD" in {a["iata"] for a in found}


def test_search_flights_refuses_a_bad_code(monkeypatch):
    """The check runs BEFORE any request is spent."""
    from tools import travel

    monkeypatch.setattr(
        travel, "_fast_flights", lambda *a, **k: pytest.fail("should not have searched")
    )
    out = travel.search_flights.invoke(
        {
            "origin_iata": "ORD",
            "destination_iata": "SVQ",
            "departure_date": "2026-10-05",
            "origin_city": "Chicago",
            "destination_city": "Granada",
        }
    )
    assert out["error_kind"] == "bad_airport_code"
    assert "GRX" in out["error"]
