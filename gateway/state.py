"""Shared gateway stores.

The middleware and the admin routes need the same buckets, keys and concurrency
counter. Reaching into Starlette's built middleware stack to find the live
instance works until it doesn't — the stack is an implementation detail and it
is constructed lazily.

So the stores live here, created once by `install()`, and both consumers read
them from a single place. Explicit shared state beats introspection.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from gateway.buckets import Bucket
from gateway.concurrency import ConcurrencyLimiter
from gateway.identity import KeyStore
from gateway.limiters import Limiter, build_limiter

# The service's own upstream budget, shared by every caller. Anchored to
# reality: three Groq keys at 200k tokens/day, and a plan costs ~40k, so about
# 15 fresh plans a day exist to share out. At 40 credits per plan, 600.
GLOBAL_DAILY_CREDITS = 600
GLOBAL_KEY = "global"


@dataclass
class Gateway:
    buckets: Limiter
    keys: KeyStore
    concurrency: ConcurrencyLimiter

    @property
    def global_bucket(self) -> Bucket:
        return Bucket.per_day(GLOBAL_DAILY_CREDITS)


_gateway: Optional[Gateway] = None


def build(db_path: Path | str) -> Gateway:
    global _gateway
    _gateway = Gateway(
        # `limits` by default; the hand-rolled token bucket via
        # GATEWAY_LIMITER=bucket. See gateway/limiters.py for the tradeoff.
        buckets=build_limiter(db_path),
        keys=KeyStore(db_path),
        concurrency=ConcurrencyLimiter(),
    )
    return _gateway


def current() -> Optional[Gateway]:
    return _gateway
