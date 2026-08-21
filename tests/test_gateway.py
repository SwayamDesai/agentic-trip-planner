"""The gateway: buckets, identity, concurrency, policy and ordering.

Written against the behaviours a gateway is actually judged on — does a refusal
deduct, is the decision atomic under concurrency, is a stream refused before it
opens — rather than just "does it return 429 sometimes".
"""

import asyncio
import threading

import pytest

from gateway.buckets import Bucket, BucketStore, Decision
from gateway.concurrency import ConcurrencyLimiter
from gateway.identity import KeyStore, anonymous, extract_key
from gateway.policy import policy_for


@pytest.fixture
def store(tmp_path):
    return BucketStore(tmp_path / "g.sqlite")


# --- the algorithm ---------------------------------------------------------


def test_spending_within_capacity_is_allowed(store):
    b = Bucket(capacity=10, refill_per_second=1)
    d = store.take("k", 4, b)
    assert d.allowed and d.remaining == 6


def test_exhausting_the_bucket_refuses(store):
    b = Bucket(capacity=10, refill_per_second=1)
    store.take("k", 10, b)
    assert store.take("k", 1, b).allowed is False


def test_a_refusal_does_not_deduct(store):
    """Otherwise a client retrying in a loop starves itself: each attempt would
    push the next one further away."""
    b = Bucket(capacity=10, refill_per_second=1)
    store.take("k", 9, b)
    before = store.peek("k", b)
    store.take("k", 5, b)          # refused
    store.take("k", 5, b)          # refused again
    assert store.peek("k", b) == pytest.approx(before, abs=0.1)


def test_retry_after_reflects_the_shortfall(store):
    b = Bucket(capacity=10, refill_per_second=2)   # 2 tokens/sec
    store.take("k", 10, b)
    d = store.take("k", 6, b)
    assert d.retry_after == pytest.approx(3.0, abs=0.2), "6 tokens at 2/sec"


def test_refill_is_lazy_and_time_based(store, monkeypatch):
    """No timer, no background task — elapsed time is computed on read."""
    b = Bucket(capacity=10, refill_per_second=1)
    store.take("k", 10, b)
    assert store.peek("k", b) < 1

    real = __import__("time").time
    monkeypatch.setattr("gateway.buckets.time.time", lambda: real() + 5)
    assert store.peek("k", b) == pytest.approx(5, abs=0.2)


def test_refill_is_capped_at_capacity(store, monkeypatch):
    b = Bucket(capacity=10, refill_per_second=1)
    store.take("k", 1, b)
    real = __import__("time").time
    monkeypatch.setattr("gateway.buckets.time.time", lambda: real() + 10_000)
    assert store.peek("k", b) == 10, "an idle bucket does not overflow"


def test_buckets_are_independent(store):
    b = Bucket(capacity=5, refill_per_second=1)
    store.take("alice", 5, b)
    assert store.take("bob", 5, b).allowed, "one caller cannot exhaust another"


def test_per_day_helper():
    b = Bucket.per_day(86_400)
    assert b.refill_per_second == pytest.approx(1.0)


def test_burst_can_differ_from_the_daily_rate():
    """Capacity is the burst allowance; the rate is sustained throughput."""
    b = Bucket.per_day(200, burst=80)
    assert b.capacity == 80
    assert b.refill_per_second == pytest.approx(200 / 86_400)


# --- weighted cost, the reason this is not request-counting ---------------


def test_expensive_and_cheap_routes_share_one_budget(store):
    """A plan is ~40x a chat turn. Counting requests would let a caller spend
    40x their share while looking compliant."""
    b = Bucket(capacity=100, refill_per_second=0)
    assert store.take("k", 40, b).allowed        # one plan
    assert store.take("k", 40, b).allowed        # two
    assert store.take("k", 40, b).allowed is False
    for _ in range(20):
        assert store.take("k", 1, b).allowed     # chat turns still fit


# --- atomicity ------------------------------------------------------------


def test_concurrent_takes_do_not_oversell(store, tmp_path):
    """Read-modify-write on a shared counter is a race: without a write lock,
    two requests can both see the last token and both spend it."""
    b = Bucket(capacity=20, refill_per_second=0)
    results = []
    lock = threading.Lock()

    def worker():
        s = BucketStore(tmp_path / "g.sqlite")   # separate connection per thread
        d = s.take("k", 1, b)
        with lock:
            results.append(d.allowed)

    threads = [threading.Thread(target=worker) for _ in range(30)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sum(results) == 20, f"exactly capacity should succeed, got {sum(results)}"


# --- refunds --------------------------------------------------------------


def test_refund_returns_tokens(store):
    b = Bucket(capacity=10, refill_per_second=0)
    store.take("k", 6, b)
    store.refund("k", 6, b)
    assert store.peek("k", b) == pytest.approx(10, abs=0.1)


def test_refund_cannot_mint_tokens(store):
    """A double refund must not exceed capacity, or a bug becomes free credit."""
    b = Bucket(capacity=10, refill_per_second=0)
    store.take("k", 2, b)
    for _ in range(5):
        store.refund("k", 2, b)
    assert store.peek("k", b) == 10


# --- headers --------------------------------------------------------------


def test_headers_use_conventional_names():
    d = Decision(allowed=True, remaining=7.4, retry_after=0, limit=10)
    h = d.headers()
    assert h["X-RateLimit-Limit"] == "10"
    assert h["X-RateLimit-Remaining"] == "7"
    assert "Retry-After" not in h, "only meaningful on a refusal"


def test_refusal_includes_retry_after():
    d = Decision(allowed=False, remaining=0, retry_after=12.3, limit=10)
    assert d.headers()["Retry-After"] == "12"


def test_remaining_never_goes_negative_in_headers():
    d = Decision(allowed=False, remaining=-3, retry_after=1, limit=10)
    assert d.headers()["X-RateLimit-Remaining"] == "0"


# --- identity -------------------------------------------------------------


def test_issued_key_resolves_to_its_tier(tmp_path):
    ks = KeyStore(tmp_path / "k.sqlite")
    raw = ks.issue(tier="pro", label="demo")
    p = ks.resolve(raw)
    assert p.tier == "pro" and p.authenticated
    assert p.limits["max_concurrent"] == 3


def test_raw_key_is_not_stored(tmp_path):
    """A leaked store should not be a leaked credential list."""
    path = tmp_path / "k.sqlite"
    ks = KeyStore(path)
    raw = ks.issue()
    assert raw.encode() not in path.read_bytes()


def test_principal_id_does_not_contain_the_secret(tmp_path):
    ks = KeyStore(tmp_path / "k.sqlite")
    raw = ks.issue()
    assert raw not in ks.resolve(raw).id, "ids reach logs and metric labels"


def test_unknown_and_revoked_keys_do_not_resolve(tmp_path):
    ks = KeyStore(tmp_path / "k.sqlite")
    raw = ks.issue()
    assert ks.resolve("atl_wrong") is None
    ks.revoke(raw)
    assert ks.resolve(raw) is None


def test_unknown_tier_is_rejected(tmp_path):
    ks = KeyStore(tmp_path / "k.sqlite")
    with pytest.raises(ValueError, match="unknown tier"):
        ks.issue(tier="enterprise")


@pytest.mark.parametrize("headers,expected", [
    ({"authorization": "Bearer atl_abc"}, "atl_abc"),
    ({"authorization": "bearer atl_abc"}, "atl_abc"),
    ({"x-api-key": "atl_xyz"}, "atl_xyz"),
    ({"authorization": "Basic dXNlcg=="}, None),
    ({}, None),
])
def test_key_extraction(headers, expected):
    assert extract_key(headers) == expected


def test_anonymous_is_scoped_to_the_ip():
    """Unauthenticated traffic still needs a scope, or one script drains the
    shared quota for everyone."""
    p = anonymous("203.0.113.7")
    assert p.id == "ip:203.0.113.7" and not p.authenticated
    assert p.limits["daily_credits"] < 100


# --- concurrency ----------------------------------------------------------


def test_concurrency_admits_up_to_the_limit():
    async def run():
        lim = ConcurrencyLimiter()
        admitted = []

        async def hold(_):
            async with lim.slot("u", 2) as ok:
                admitted.append(ok)
                if ok:
                    await asyncio.sleep(0.05)

        await asyncio.gather(*[hold(i) for i in range(5)])
        return admitted, await lim.active("u")

    admitted, active = asyncio.run(run())
    assert sum(admitted) == 2
    assert active == 0, "every slot released"


def test_slot_released_even_when_the_handler_raises():
    """A client disconnecting mid-stream must not leak a slot — a leaked slot
    is a permanent lockout."""
    async def run():
        lim = ConcurrencyLimiter()
        with pytest.raises(RuntimeError):
            async with lim.slot("u", 1) as ok:
                assert ok
                raise RuntimeError("client vanished")
        return await lim.active("u")

    assert asyncio.run(run()) == 0


def test_limiter_does_not_deadlock_when_the_handler_inspects_it():
    """Regression: the admission lock was held across the yield, so any handler
    that queried the limiter deadlocked against it."""
    async def run():
        lim = ConcurrencyLimiter()
        async with lim.slot("u", 1) as ok:
            assert ok
            return await lim.active("u")

    assert asyncio.run(asyncio.wait_for(run(), timeout=5)) == 1


def test_principals_have_separate_slots():
    async def run():
        lim = ConcurrencyLimiter()
        async with lim.slot("a", 1) as first, lim.slot("b", 1) as second:
            return first, second

    assert asyncio.run(run()) == (True, True)


# --- policy ---------------------------------------------------------------


def test_plan_costs_far_more_than_chat():
    plan = policy_for("GET", "/api/plan")
    chat = policy_for("POST", "/api/chat")
    assert plan.cost > chat.cost * 20, "cost should track real work"
    assert plan.concurrency, "a two-minute stream must hold a slot"
    assert plan.refundable, "a resumed plan does almost no work"


@pytest.mark.parametrize("path", ["/", "/styles.css", "/app.js", "/health", "/gateway/status"])
def test_static_and_admin_routes_are_unmetered(path):
    """A gateway that rate-limits its own status page cannot be debugged."""
    assert policy_for("GET", path) is None


def test_unknown_api_route_is_unmetered():
    assert policy_for("GET", "/api/nonexistent") is None


def test_method_is_part_of_the_policy_key():
    assert policy_for("POST", "/api/plan") is None
    assert policy_for("GET", "/api/plan") is not None


# --- the limits-library backend -------------------------------------------


def test_limits_backend_supports_weighted_cost():
    """The decisive capability: without cost=N, a library can only count
    requests, and a plan would cost the same as a chat turn."""
    from gateway.limiters import LimitsBackend

    lb = LimitsBackend("memory://")
    b = Bucket.per_day(100, burst=100)
    assert lb.take("u", 40, b).allowed
    assert lb.take("u", 40, b).allowed
    assert lb.take("u", 40, b).allowed is False, "120 > 100"
    assert lb.take("u", 1, b).allowed, "a cheap request still fits"


def test_limits_backend_reports_remaining_and_retry_after():
    from gateway.limiters import LimitsBackend

    lb = LimitsBackend("memory://")
    b = Bucket.per_day(10, burst=10)
    lb.take("u", 10, b)
    d = lb.take("u", 5, b)
    assert not d.allowed and d.retry_after > 0
    assert d.headers()["Retry-After"]


def test_limits_backend_keys_are_independent():
    from gateway.limiters import LimitsBackend

    lb = LimitsBackend("memory://")
    b = Bucket.per_day(10, burst=10)
    lb.take("alice", 10, b)
    assert lb.take("bob", 5, b).allowed


def test_backend_is_selectable(monkeypatch, tmp_path):
    """Both implementations satisfy the same interface, so the middleware does
    not know or care which is in use."""
    from gateway.buckets import BucketStore
    from gateway.limiters import LimitsBackend, build_limiter

    monkeypatch.setenv("GATEWAY_LIMITER", "bucket")
    assert isinstance(build_limiter(tmp_path / "a.sqlite"), BucketStore)

    monkeypatch.setenv("GATEWAY_LIMITER", "limits")
    assert isinstance(build_limiter(tmp_path / "a.sqlite"), LimitsBackend)


@pytest.mark.parametrize("backend", ["bucket", "limits"])
def test_both_backends_meter_weighted_cost_the_same_way(backend, tmp_path, monkeypatch):
    """Different algorithms, same observable contract for the case that matters."""
    monkeypatch.setenv("GATEWAY_LIMITER", backend)
    from gateway.limiters import build_limiter

    limiter = build_limiter(tmp_path / f"{backend}.sqlite")
    b = Bucket.per_day(100, burst=100)
    assert limiter.take("u", 60, b).allowed
    assert limiter.take("u", 60, b).allowed is False
