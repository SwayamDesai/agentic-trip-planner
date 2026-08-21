"""Tiered TTLs, single-flight fetching, and read-through behaviour."""

import threading
import time

import pytest

from providers import cache


@pytest.fixture(autouse=True)
def clean_stats():
    cache.STATS.clear()
    yield
    cache.STATS.clear()


# --- lifetimes are ordered by how fast the data actually changes ---


def test_ttl_tiers_are_ordered():
    assert cache.TTL_IMMUTABLE > cache.TTL_STATIC > cache.TTL_FORECAST
    assert cache.TTL_FORECAST > cache.TTL_EMPTY


def test_metered_data_is_held_longer_than_unmetered():
    """Hotels cost one of 250 SerpApi searches a month; flights are keyless.

    So hotels tolerate more staleness — TTL follows the cost of refreshing,
    not just how fast the number moves.
    """
    assert cache.TTL_HOTELS > cache.TTL_FLIGHTS


def test_ttls_are_env_overridable(monkeypatch):
    monkeypatch.setenv("TTL_HOTELS", "99")
    import importlib

    reloaded = importlib.reload(cache)
    try:
        assert reloaded.TTL_HOTELS == 99
    finally:
        monkeypatch.delenv("TTL_HOTELS", raising=False)
        importlib.reload(cache)


# --- read-through: the cache fills as the app is used ---


def test_first_call_is_a_miss_second_is_a_hit():
    calls = []
    produce = lambda: calls.append(1) or {"v": 1}

    cache.cached("ns", "k", produce, ttl=60)
    cache.cached("ns", "k", produce, ttl=60)
    assert len(calls) == 1
    assert cache.STATS["ns.miss"] == 1
    assert cache.STATS["ns.hit"] == 1


def test_stats_report_a_hit_rate():
    cache.cached("ns", "a", lambda: {"v": 1}, ttl=60)
    cache.cached("ns", "a", lambda: {"v": 1}, ttl=60)
    report = cache.stats()
    assert report["hits"] == 1 and report["misses"] == 1
    assert report["hit_rate"] == 0.5


def test_served_entry_reports_its_age():
    """Callers must be able to say how old a price is rather than imply it is
    live."""
    cache.cached("ns", "k", lambda: {"v": 1}, ttl=60)
    served = cache.cached("ns", "k", lambda: {"v": 2}, ttl=60)
    assert served["v"] == 1
    assert "_cache" in served and served["_cache"]["age_seconds"] >= 0


def test_fresh_production_has_no_age_marker():
    assert "_cache" not in cache.cached("ns", "k", lambda: {"v": 1}, ttl=60)


# --- what is and is not persisted ---


def test_failures_are_not_persisted():
    cache.cached("ns", "k", lambda: {"error": "upstream down"}, ttl=60)
    assert cache.STATS["ns.not_stored"] == 1
    assert cache.cached("ns", "k", lambda: {"v": "live"}, ttl=60)["v"] == "live"


def test_source_none_is_not_persisted():
    cache.cached("ns", "k", lambda: {"options": [], "source": "none"}, ttl=60)
    assert cache.cached("ns", "k", lambda: {"source": "live"}, ttl=60)["source"] == "live"


def test_genuine_emptiness_is_cached_briefly():
    """'No flights on this route' is a real answer; 'the API is down' is not."""
    empty = {"error": "No flights found.", "error_kind": "no_results"}
    cache.cached("ns", "route", lambda: empty, ttl=cache.TTL_STATIC)
    calls = []
    again = cache.cached(
        "ns", "route", lambda: calls.append(1) or {"v": "fresh"}, ttl=cache.TTL_STATIC
    )
    assert calls == [], "a genuine absence should be reused"
    assert again["error_kind"] == "no_results"


def test_empty_result_ttl_is_shortened():
    """Cached, but rechecked sooner than the namespace's normal lifetime."""
    assert cache.TTL_EMPTY < cache.TTL_STATIC


# --- single-flight ---


def test_concurrent_cold_reads_produce_one_fetch():
    """The parallel fan-out hits a cold cache from several agents at once.

    Without single-flight this double-fetches every cold start, and under real
    traffic becomes a thundering herd on shared services like Overpass.
    """
    calls = []
    started = threading.Barrier(6)

    def produce():
        calls.append(1)
        time.sleep(0.05)
        return {"v": "once"}

    def worker():
        started.wait()
        cache.cached("ns", "hot", produce, ttl=60)

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(calls) == 1, f"expected one upstream fetch, got {len(calls)}"
    assert cache.STATS["ns.hit_after_wait"] >= 1, "waiters served from the fill"


def test_different_keys_do_not_block_each_other():
    calls = []

    def produce():
        calls.append(1)
        time.sleep(0.02)
        return {"v": 1}

    threads = [
        threading.Thread(target=lambda i=i: cache.cached("ns", f"k{i}", produce, ttl=60))
        for i in range(4)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(calls) == 4, "distinct keys are independent"
