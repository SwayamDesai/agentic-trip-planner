"""Cache semantics — including the failure-persistence bug."""

from providers.cache import cached


def test_miss_then_hit():
    calls = []

    def producer():
        calls.append(1)
        return {"value": 42}

    assert cached("ns", "k", producer)["value"] == 42
    assert cached("ns", "k", producer)["value"] == 42
    assert len(calls) == 1, "second call should have been served from disk"


def test_distinct_keys_do_not_collide():
    assert cached("ns", "a", lambda: {"v": "a"})["v"] == "a"
    assert cached("ns", "b", lambda: {"v": "b"})["v"] == "b"


def test_expired_entry_refetches():
    cached("ns", "k", lambda: {"v": 1}, ttl=0)
    assert cached("ns", "k", lambda: {"v": 2}, ttl=0)["v"] == 2


def test_failures_are_never_persisted():
    """Regression: a 12h TTL on a transient outage locked the failure in.

    A `source: none` result means a backend was down or a quota was spent, not
    that the answer is genuinely empty.
    """
    assert cached("f", "k", lambda: {"options": [], "source": "none"})["source"] == "none"
    hit = cached("f", "k", lambda: {"options": [{"a": 1}], "source": "live"})
    assert hit["source"] == "live", "the miss should not have been cached"


def test_error_results_are_never_persisted():
    cached("f", "k2", lambda: {"error": "quota exceeded"})
    assert "error" not in cached("f", "k2", lambda: {"options": [1], "source": "live"})


def test_successful_result_is_persisted():
    cached("f", "k3", lambda: {"options": [{"a": 1}], "source": "live"})
    survived = cached("f", "k3", lambda: {"options": [], "source": "none"})
    assert survived["source"] == "live"


def test_unserialisable_value_still_returned():
    """A value we cannot write to disk must still reach the caller."""

    class NotJson:
        pass

    out = cached("ns", "weird", lambda: {"obj": NotJson(), "source": "live"})
    assert "obj" in out
