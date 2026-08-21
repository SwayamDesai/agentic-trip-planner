"""Two limiter backends behind one interface, and why both exist.

`gateway.buckets.BucketStore` is a hand-rolled token bucket. This module wraps
the `limits` library instead. Same interface, different algorithm, and the
tradeoff is worth understanding rather than hiding:

                    hand-rolled token bucket        limits (moving window)
  algorithm         continuous lazy refill          rolling window of hits
  burst vs rate     SEPARATE: capacity is burst,    ONE number: 200/day means
                    refill is sustained rate        200 in any 24h window
  storage           SQLite, survives restart        memory / Redis / Memcached
                                                    / MongoDB / etcd — no SQLite
  refunds           yes                             not in the API
  battle-tested     no                              yes, widely deployed

`limits` is the better choice wherever it fits: maintained, audited, and with
real distributed backends. But note the storage row, because it decides which
default makes sense where.

**Deployment gotcha, found the hard way.** `limits` has no file-backed store, so
without Redis it runs `memory://` — and a restart resets every counter. Mounting
a volume does not help: there is nothing on disk to persist. A container restart
therefore hands every caller a fresh quota, which is precisely what the limiter
exists to prevent. It was verified by restarting the container and watching
global credits go from 594 back to 600.

So:
    single process, no Redis   -> GATEWAY_LIMITER=bucket   (persists to SQLite)
    Redis available            -> GATEWAY_LIMITER=limits   with
                                  GATEWAY_STORAGE_URI=redis://...
    multiple instances         -> limits + Redis is the ONLY correct option;
                                  the SQLite bucket is per-instance

The default here is `limits` because that is right once there is a Redis, and
the Dockerfile overrides it to `bucket` because the deployed shape has none.
"""

import os
from pathlib import Path
from typing import Optional, Protocol

from gateway.buckets import Bucket, BucketStore, Decision


class Limiter(Protocol):
    """What the middleware needs. Neither backend exposes more than this."""

    def take(self, key: str, cost: float, bucket: Bucket, scope: str = "") -> Decision: ...
    def refund(self, key: str, cost: float, bucket: Bucket) -> None: ...
    def peek(self, key: str, bucket: Bucket) -> float: ...


class LimitsBackend:
    """`limits`-backed limiter.

    Uses the moving-window strategy: more accurate than a fixed window, which
    permits a double burst across the boundary (200 requests at 23:59 and 200
    more at 00:01). The cost is per-key storage of hit timestamps rather than a
    single counter — acceptable at this scale, and the correctness is worth it.

    Refunds are approximated by clearing the key, because the library has no
    concept of returning a charge. That is coarse and the docstring says so:
    a refund gives the caller their whole window back rather than one request's
    worth. Acceptable for the case it exists to serve — a resumed plan that did
    no work — and not something to rely on for accounting.
    """

    def __init__(self, uri: Optional[str] = None):
        from limits import storage, strategies

        # memory:// does NOT survive a restart. Fine for tests and for a
        # deployment that has Redis; wrong as a persistent default, which is
        # why the Dockerfile selects the SQLite-backed bucket instead.
        uri = uri or os.getenv("GATEWAY_STORAGE_URI", "memory://")
        self._storage = storage.storage_from_string(uri)
        self._limiter = strategies.MovingWindowRateLimiter(self._storage)
        self.uri = uri

    @staticmethod
    def _item(bucket: Bucket):
        """Translate a Bucket into a limits RateLimitItem.

        A token bucket has two numbers and a window has one, so the sustained
        rate is what carries over: capacity is the amount, and the window is
        however long that amount takes to refill.
        """
        from limits import RateLimitItemPerSecond

        seconds = max(1, int(round(bucket.capacity / max(bucket.refill_per_second, 1e-9))))
        return RateLimitItemPerSecond(int(bucket.capacity), seconds)

    def take(self, key: str, cost: float, bucket: Bucket, scope: str = "") -> Decision:
        item = self._item(bucket)
        allowed = self._limiter.hit(item, key, cost=int(cost))
        window = self._limiter.get_window_stats(item, key)
        import time as _time

        return Decision(
            allowed=allowed,
            remaining=float(window.remaining),
            retry_after=max(0.0, window.reset_time - _time.time()) if not allowed else 0.0,
            limit=float(bucket.capacity),
            scope=scope,
        )

    def refund(self, key: str, cost: float, bucket: Bucket) -> None:
        # coarse: the library cannot return a single charge, so the window is
        # cleared. Documented rather than disguised.
        self._storage.clear(self._item(bucket).key_for(key))

    def peek(self, key: str, bucket: Bucket) -> float:
        return float(self._limiter.get_window_stats(self._item(bucket), key).remaining)


def build_limiter(db_path: Path | str) -> Limiter:
    """The configured limiter."""
    choice = os.getenv("GATEWAY_LIMITER", "limits").strip().lower()
    if choice == "bucket":
        return BucketStore(db_path)
    return LimitsBackend()
