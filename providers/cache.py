"""On-disk JSON cache for external API calls.

Two reasons this exists rather than being an optimisation:

1. Nominatim's usage policy caps requests at 1/sec and asks that clients cache
   results. Overpass is a shared volunteer service with similar etiquette.
2. Free-tier LLM latency already makes runs slow; re-fetching identical POI and
   geocode data on every run during development wastes minutes.
"""

import hashlib
import json
import os
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Optional

CACHE_DIR = Path(os.getenv("TRIP_CACHE_DIR", Path(__file__).parent.parent / ".cache"))

_HOUR = 3600
_DAY = 24 * _HOUR

# Lifetimes are set by how fast the DATA changes and how expensive a refresh
# is — not by one global default. The second half matters: flight lookups are
# unmetered (fast-flights is keyless), while every hotel lookup spends one of
# 250 SerpApi searches a month. So hotels tolerate more staleness than flights,
# even though both are "prices".
TTL_IMMUTABLE = int(os.getenv("TTL_IMMUTABLE", 365 * _DAY))   # city coordinates
TTL_STATIC = int(os.getenv("TTL_STATIC", 30 * _DAY))          # POIs, guides, normals
TTL_ADVISORY = int(os.getenv("TTL_ADVISORY", 30 * _DAY))      # trip-length advice
TTL_FORECAST = int(os.getenv("TTL_FORECAST", 6 * _HOUR))      # real forecasts move
TTL_FLIGHTS = int(os.getenv("TTL_FLIGHTS", 4 * _HOUR))        # cheap to refresh
TTL_HOTELS = int(os.getenv("TTL_HOTELS", 12 * _HOUR))         # metered, so reused harder
TTL_EMPTY = int(os.getenv("TTL_EMPTY", 1 * _HOUR))            # genuine "no results"

DEFAULT_TTL = TTL_STATIC


def _path_for(namespace: str, key: str) -> Path:
    digest = hashlib.sha256(key.encode()).hexdigest()[:20]
    return CACHE_DIR / namespace / f"{digest}.json"


# One lock per key, so a cold cache hit by several agents at once produces ONE
# upstream fetch. Without this, the parallel fan-out double-fetches on every
# cold start — and with real traffic it becomes a thundering herd on services
# like Overpass, which has already shed load on us once.
_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()

# Observability: the cache is read-through, so it fills as the app is used.
# These counters make that visible instead of a matter of faith.
STATS: Counter = Counter()


def _lock_for(path: Path) -> threading.Lock:
    key = str(path)
    with _LOCKS_GUARD:
        lock = _LOCKS.get(key)
        if lock is None:
            lock = _LOCKS[key] = threading.Lock()
        return lock


def _read(path: Path, ttl: int) -> Optional[tuple[Any, float]]:
    """Return (value, age_seconds) if a live entry exists."""
    try:
        if not path.exists():
            return None
        age = time.time() - path.stat().st_mtime
        if age >= ttl:
            return None
        return json.loads(path.read_text()), age
    except (OSError, json.JSONDecodeError):
        return None  # an unreadable entry is a miss


def cached(
    namespace: str,
    key: str,
    producer: Callable[[], Any],
    ttl: Optional[int] = None,
) -> Any:
    """Read-through cache: serve from disk, else produce and store.

    The cache builds itself as the app is used — nothing is precomputed. A
    served entry carries `_cache` metadata (age in seconds) so callers can
    disclose how old a price is rather than implying it is live.

    A failure is never stored. An empty-but-genuine result (`error_kind` of
    "no_results") IS stored, briefly: "no flights on this route" is a real
    answer, while "the API is down" is not.
    """
    ttl = DEFAULT_TTL if ttl is None else ttl
    path = _path_for(namespace, key)

    hit = _read(path, ttl)
    if hit is not None:
        STATS[f"{namespace}.hit"] += 1
        value, age = hit
        if isinstance(value, dict):
            value = {**value, "_cache": {"age_seconds": round(age)}}
        return value

    with _lock_for(path):
        # another thread may have filled it while we waited
        hit = _read(path, ttl)
        if hit is not None:
            STATS[f"{namespace}.hit_after_wait"] += 1
            value, age = hit
            if isinstance(value, dict):
                value = {**value, "_cache": {"age_seconds": round(age)}}
            return value

        STATS[f"{namespace}.miss"] += 1
        value = producer()

        if isinstance(value, dict):
            kind = value.get("error_kind")
            if kind == "no_results":
                ttl = min(ttl, TTL_EMPTY)   # real absence, but recheck sooner
            elif value.get("error") or value.get("source") == "none":
                STATS[f"{namespace}.not_stored"] += 1
                return value                # a failure must not be persisted

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value))
        except (OSError, TypeError):
            pass  # unwritable or non-serialisable: still return the live value

        return value


def stats() -> dict:
    """Hit/miss counts per namespace, plus an overall hit rate."""
    hits = sum(v for k, v in STATS.items() if ".hit" in k)
    misses = sum(v for k, v in STATS.items() if k.endswith(".miss"))
    total = hits + misses
    return {
        "by_namespace": dict(sorted(STATS.items())),
        "hits": hits,
        "misses": misses,
        "hit_rate": round(hits / total, 3) if total else None,
    }
