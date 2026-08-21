"""Token bucket rate limiting.

The classic algorithm, and worth understanding before reaching for a library.

A bucket holds up to `capacity` tokens and refills at `refill_per_second`. A
request costs some number of tokens; if the bucket holds enough, they are
removed and the request proceeds. Otherwise it is refused, and the caller is
told how long until enough tokens exist.

Two properties make it the usual choice:

  * capacity is a BURST allowance. A user idle for an hour can spend the whole
    bucket at once, which is what people actually do — arrive, do several
    things, leave.
  * refill rate is the SUSTAINED throughput. Over a long window, throughput
    converges on the refill rate no matter how the requests clump.

Compare the alternatives:

  fixed window     "100 per hour", reset on the hour. Simple, but allows 200
                   requests across a window boundary — the classic stampede at
                   :59 and :00.
  sliding window   accurate, but needs a log of timestamps per user: more
                   storage and more work per request.
  leaky bucket     shapes output at a constant rate (nginx `limit_req`), which
                   is what you want for smoothing traffic to a fragile upstream,
                   but it queues rather than refuses, so it is a poor fit for a
                   synchronous HTTP answer.

Refill is LAZY: computed from elapsed time when the bucket is read. No timer,
no background task, and a process restart loses nothing but the clock tick.
"""

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class Bucket:
    """A rate limit: burst size plus sustained rate."""

    capacity: float
    refill_per_second: float

    @classmethod
    def per_day(cls, amount: float, burst: Optional[float] = None) -> "Bucket":
        return cls(capacity=burst if burst is not None else amount,
                   refill_per_second=amount / 86_400.0)

    @classmethod
    def per_minute(cls, amount: float, burst: Optional[float] = None) -> "Bucket":
        return cls(capacity=burst if burst is not None else amount,
                   refill_per_second=amount / 60.0)


@dataclass
class Decision:
    """The outcome of asking a bucket for tokens."""

    allowed: bool
    remaining: float
    retry_after: float          # seconds until the request would succeed
    limit: float
    scope: str = ""             # which bucket decided, for the error message

    def headers(self) -> dict[str, str]:
        """Conventional rate-limit headers.

        Not an RFC, but near-universal (GitHub, Stripe, AWS): clients and
        libraries look for these names, so a custom scheme buys nothing.
        """
        out = {
            "X-RateLimit-Limit": str(int(self.limit)),
            "X-RateLimit-Remaining": str(max(0, int(self.remaining))),
            "X-RateLimit-Reset": str(int(time.time() + self.retry_after)),
        }
        if not self.allowed:
            # Retry-After is what makes a 429 actionable rather than a wall
            out["Retry-After"] = str(max(1, int(self.retry_after + 0.5)))
        return out


class BucketStore:
    """Persistent buckets, one row per key.

    SQLite rather than Redis because this is a single process and the state is
    tiny. The interface is deliberately narrow so swapping in Redis later means
    reimplementing one method.

    Correctness note: read-modify-write on a shared counter is a race. Two
    concurrent requests could both read 1 token and both spend it. So the whole
    operation runs inside `BEGIN IMMEDIATE`, which takes SQLite's write lock up
    front instead of upgrading a read lock and risking SQLITE_BUSY.
    """

    def __init__(self, path: Path | str):
        self.path = str(path)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=5.0, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL")   # readers do not block writers
        return conn

    def _init_schema(self) -> None:
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS buckets (
                    key         TEXT PRIMARY KEY,
                    tokens      REAL NOT NULL,
                    last_refill REAL NOT NULL
                )
                """
            )

    def take(self, key: str, cost: float, bucket: Bucket, scope: str = "") -> Decision:
        """Attempt to spend `cost` tokens from `key`.

        On refusal nothing is deducted: a refused request must not make the
        next one wait longer, or a client retrying in a loop would starve
        itself indefinitely.
        """
        now = time.time()
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")

            row = conn.execute(
                "SELECT tokens, last_refill FROM buckets WHERE key = ?", (key,)
            ).fetchone()

            if row is None:
                tokens, last = bucket.capacity, now
            else:
                tokens, last = row
                # lazy refill: the whole reason no timer is needed
                elapsed = max(0.0, now - last)
                tokens = min(
                    bucket.capacity, tokens + elapsed * bucket.refill_per_second
                )

            if tokens >= cost:
                tokens -= cost
                allowed, retry_after = True, 0.0
            else:
                allowed = False
                shortfall = cost - tokens
                retry_after = (
                    shortfall / bucket.refill_per_second
                    if bucket.refill_per_second > 0
                    else float("inf")
                )

            conn.execute(
                "INSERT INTO buckets(key, tokens, last_refill) VALUES(?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET tokens = ?, last_refill = ?",
                (key, tokens, now, tokens, now),
            )
            conn.execute("COMMIT")
        finally:
            conn.close()

        return Decision(
            allowed=allowed,
            remaining=tokens,
            retry_after=retry_after,
            limit=bucket.capacity,
            scope=scope,
        )

    def refund(self, key: str, cost: float, bucket: Bucket) -> None:
        """Return tokens to a bucket.

        Needed because cost is charged BEFORE the work is known to be
        chargeable: a request that turns out to be free (served from cache) or
        that failed upstream should not be billed. Capped at capacity so a
        double refund cannot mint tokens.
        """
        now = time.time()
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT tokens, last_refill FROM buckets WHERE key = ?", (key,)
            ).fetchone()
            if row is not None:
                tokens, last = row
                elapsed = max(0.0, now - last)
                tokens = min(
                    bucket.capacity,
                    tokens + elapsed * bucket.refill_per_second + cost,
                )
                conn.execute(
                    "UPDATE buckets SET tokens = ?, last_refill = ? WHERE key = ?",
                    (tokens, now, key),
                )
            conn.execute("COMMIT")
        finally:
            conn.close()

    def peek(self, key: str, bucket: Bucket) -> float:
        """Current tokens without spending any."""
        now = time.time()
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT tokens, last_refill FROM buckets WHERE key = ?", (key,)
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            return bucket.capacity
        tokens, last = row
        return min(bucket.capacity, tokens + max(0.0, now - last) * bucket.refill_per_second)

    def reset(self, key: str) -> None:
        conn = self._connect()
        try:
            conn.execute("DELETE FROM buckets WHERE key = ?", (key,))
        finally:
            conn.close()
