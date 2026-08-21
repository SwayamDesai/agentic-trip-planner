"""Geocoding via Nominatim (OpenStreetMap). Keyless.

Nominatim's usage policy requires an identifying User-Agent and at most 1
request per second. Results are cached, and calls are serialised through a
lock because the agent fan-out is concurrent.
"""

import threading
import time

import requests
from langchain_core.tools import tool

from providers.cache import TTL_IMMUTABLE, cached

UA = {"User-Agent": "trip-planner/0.1 (github.com/local; dev use)"}
_NOMINATIM_LOCK = threading.Lock()
_last_call = 0.0


def _throttled_get(url: str, params: dict) -> dict | list:
    """Serialise and rate-limit calls to honour Nominatim's 1 req/sec policy."""
    global _last_call
    with _NOMINATIM_LOCK:
        wait = 1.05 - (time.perf_counter() - _last_call)
        if wait > 0:
            time.sleep(wait)
        resp = requests.get(url, params=params, headers=UA, timeout=30)
        _last_call = time.perf_counter()
    resp.raise_for_status()
    return resp.json()


def geocode(place: str) -> dict:
    """Resolve a place name to coordinates. Cached; safe to call repeatedly."""

    def fetch():
        data = _throttled_get(
            "https://nominatim.openstreetmap.org/search",
            {"q": place, "format": "json", "limit": 1},
        )
        if not data:
            return {"error": f"could not geocode {place!r}"}
        hit = data[0]
        return {
            "name": hit["display_name"],
            "lat": float(hit["lat"]),
            "lon": float(hit["lon"]),
        }

    return cached("geocode", place.lower().strip(), fetch, ttl=TTL_IMMUTABLE)


@tool
def geocode_place(place: str) -> dict:
    """Resolve a place name to latitude and longitude.

    USE WHEN: you need coordinates before calling a tool that takes lat/lon
    (get_weather, find_places). Always call this first for a new place.

    DO NOT USE WHEN: you already have coordinates for this place from an
    earlier call in this conversation — reuse them instead of re-resolving.
    Do not use it to check whether a place exists, to get its population, or
    to pick an airport code; it returns only a name and coordinates.

    Args:
        place: Place name, e.g. "Lisbon" or "Lisbon, Portugal".

    Returns the resolved display name plus `lat` and `lon`, or an `error` key.
    """
    return geocode(place)
