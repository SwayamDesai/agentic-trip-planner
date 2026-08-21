import pytest

"""POI filtering. Raw Overpass output is too noisy to plan from."""

from tools import places


class _Resp:
    """A successful response. `status_code` is read by the retry logic."""

    status_code = 200

    def __init__(self, payload):
        self._p = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._p


def _elements(*items):
    return {"elements": list(items)}


def test_low_value_tags_are_dropped(monkeypatch):
    """Plaques, statues and generic 'yes' are not places to plan a day around."""
    payload = _elements(
        {"tags": {"name": "Real Museum", "tourism": "museum"}, "lat": 1.0, "lon": 2.0},
        {"tags": {"name": "A Plaque", "historic": "memorial"}, "lat": 1.0, "lon": 2.0},
        {"tags": {"name": "Street Art", "tourism": "artwork"}, "lat": 1.0, "lon": 2.0},
        {"tags": {"name": "Vague", "historic": "yes"}, "lat": 1.0, "lon": 2.0},
    )
    monkeypatch.setattr(places.requests, "post", lambda *a, **k: _Resp(payload))
    out = places.find_places.invoke({"lat": 1.0, "lon": 2.0, "category": "sights"})
    assert [p["name"] for p in out["places"]] == ["Real Museum"]


def test_entries_without_coordinates_are_dropped(monkeypatch):
    payload = _elements(
        {"tags": {"name": "No Coords", "tourism": "museum"}},
        {"tags": {"name": "Has Coords", "tourism": "museum"}, "lat": 1.0, "lon": 2.0},
    )
    monkeypatch.setattr(places.requests, "post", lambda *a, **k: _Resp(payload))
    out = places.find_places.invoke({"lat": 1.0, "lon": 2.0, "category": "sights"})
    assert [p["name"] for p in out["places"]] == ["Has Coords"]


def test_way_centre_is_used(monkeypatch):
    """Ways carry their position under `center`, not at the top level."""
    payload = _elements(
        {"tags": {"name": "Big Palace", "historic": "palace"},
         "center": {"lat": 38.7, "lon": -9.1}}
    )
    monkeypatch.setattr(places.requests, "post", lambda *a, **k: _Resp(payload))
    out = places.find_places.invoke({"lat": 1.0, "lon": 2.0, "category": "historic"})
    assert out["places"][0]["lat"] == 38.7


def test_limit_is_respected(monkeypatch):
    payload = _elements(*[
        {"tags": {"name": f"M{i}", "tourism": "museum"}, "lat": 1.0, "lon": 2.0}
        for i in range(30)
    ])
    monkeypatch.setattr(places.requests, "post", lambda *a, **k: _Resp(payload))
    out = places.find_places.invoke(
        {"lat": 1.0, "lon": 2.0, "category": "museums", "limit": 5}
    )
    assert len(out["places"]) == 5


def test_bad_category_lists_valid_ones():
    out = places.find_places.invoke({"lat": 1.0, "lon": 2.0, "category": "nightclubs"})
    assert "error" in out and "sights" in out["valid"]


def test_food_relaxes_the_notability_filter(monkeypatch):
    """Restaurants rarely have Wikidata entries, so requiring one returns nothing."""
    seen = {}

    def spy(url, data=None, headers=None, timeout=None):
        seen["q"] = data["data"]
        return _Resp(_elements())

    monkeypatch.setattr(places.requests, "post", spy)
    places.find_places.invoke({"lat": 1.0, "lon": 2.0, "category": "food"})
    assert "[wikidata]" not in seen["q"]

    places.find_places.invoke({"lat": 1.0, "lon": 2.0, "category": "sights"})
    assert "[wikidata]" in seen["q"], "notability filter required elsewhere"


def test_bbox_grows_with_radius():
    small = places._bbox(38.7, -9.1, 1.0)
    large = places._bbox(38.7, -9.1, 10.0)
    assert (large[2] - large[0]) > (small[2] - small[0])


def test_city_guide_missing_page_reports_error(monkeypatch):
    class R(_Resp):
        pass

    monkeypatch.setattr(
        places.requests, "get",
        lambda *a, **k: R({"query": {"pages": {"-1": {"missing": ""}}}}),
    )
    out = places.city_guide.invoke({"city": "Atlantis"})
    assert "error" in out


def test_english_name_preferred_when_available(monkeypatch):
    """OSM's `name` is local-language: Kyoto returns 京都タワー, not Kyoto Tower."""
    payload = _elements(
        {
            "tags": {
                "name": "京都タワー",
                "name:en": "Kyoto Tower",
                "tourism": "attraction",
            },
            "lat": 34.98,
            "lon": 135.75,
        }
    )
    monkeypatch.setattr(places.requests, "post", lambda *a, **k: _Resp(payload))
    place = places.find_places.invoke(
        {"lat": 34.98, "lon": 135.75, "category": "sights"}
    )["places"][0]
    assert place["name"] == "Kyoto Tower"
    assert place["local_name"] == "京都タワー", "local name kept for signage"


def test_local_name_used_when_no_translation(monkeypatch):
    payload = _elements(
        {"tags": {"name": "Alcázar", "tourism": "attraction"}, "lat": 1.0, "lon": 2.0}
    )
    monkeypatch.setattr(places.requests, "post", lambda *a, **k: _Resp(payload))
    place = places.find_places.invoke({"lat": 1.0, "lon": 2.0, "category": "sights"})["places"][0]
    assert place["name"] == "Alcázar"
    assert "local_name" not in place, "no duplicate when there is nothing to translate"


# --- Overpass resilience ---


class _Err(_Resp):
    """A response that fails with a given status."""

    def __init__(self, status):
        super().__init__({})
        self.status_code = status

    def raise_for_status(self):
        import requests

        exc = requests.HTTPError(f"{self.status} error")
        exc.response = self
        raise exc

    @property
    def status(self):
        return self.status_code


def _ok():
    r = _Resp(_elements(
        {"tags": {"name": "Museum", "tourism": "museum"}, "lat": 1.0, "lon": 2.0}
    ))
    r.status_code = 200
    return r


def test_transient_504_is_retried(monkeypatch):
    """Observed live: Overpass 504s under load. One attempt would report the
    city as having no attractions."""
    monkeypatch.setattr(places.time, "sleep", lambda s: None)
    responses = [_Err(504), _Err(504), _ok()]
    monkeypatch.setattr(places.requests, "post", lambda *a, **k: responses.pop(0))
    out = places.find_places.invoke({"lat": 1.0, "lon": 2.0, "category": "sights"})
    assert [p["name"] for p in out["places"]] == ["Museum"]


def test_rate_limit_is_retried(monkeypatch):
    monkeypatch.setattr(places.time, "sleep", lambda s: None)
    responses = [_Err(429), _ok()]
    monkeypatch.setattr(places.requests, "post", lambda *a, **k: responses.pop(0))
    assert places.find_places.invoke(
        {"lat": 1.0, "lon": 2.0, "category": "sights"}
    )["places"]


def test_client_error_is_not_retried(monkeypatch):
    """A 400 means the query is malformed; retrying cannot help."""
    import requests as rq

    calls = []

    def spy(*a, **k):
        calls.append(1)
        return _Err(400)

    monkeypatch.setattr(places.time, "sleep", lambda s: None)
    monkeypatch.setattr(places.requests, "post", spy)
    with pytest.raises(rq.HTTPError):
        places.find_places.invoke({"lat": 1.0, "lon": 2.0, "category": "sights"})
    assert len(calls) == 1


def test_persistent_failure_gives_up(monkeypatch):
    """Bounded attempts: do not hammer a free community service."""
    import requests as rq

    calls = []

    def spy(*a, **k):
        calls.append(1)
        return _Err(504)

    monkeypatch.setattr(places.time, "sleep", lambda s: None)
    monkeypatch.setattr(places.requests, "post", spy)
    with pytest.raises(rq.HTTPError):
        places.find_places.invoke({"lat": 1.0, "lon": 2.0, "category": "sights"})
    assert len(calls) == places._OVERPASS_ATTEMPTS
