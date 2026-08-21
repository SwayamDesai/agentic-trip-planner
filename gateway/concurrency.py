"""Limiting requests in flight, which rate limiting alone does not do.

A token bucket caps how OFTEN a caller may start work. It says nothing about
how much work runs at once. Both matter, and they fail differently:

  rate    a caller who sends 100 requests over an hour is within any sane rate
          limit, and fine.
  in-flight
          a caller who opens 10 planning streams simultaneously is also within
          the rate limit, and very much not fine: each stream spawns six agents,
          so that is 60 concurrent model calls against a per-minute token
          ceiling. Every one of them then fails, including everyone else's.

Long-lived responses are where this bites. A normal request is over in
milliseconds and self-limits; an SSE stream runs for two minutes.

Deliberately in-process: an asyncio-safe counter, no persistence. A restart
clears it, which is correct — nothing is actually in flight after a restart, so
persisted slots would be leaked forever.
"""

import asyncio
from collections import defaultdict
from contextlib import asynccontextmanager


class ConcurrencyLimiter:
    """Per-principal cap on simultaneous in-flight requests."""

    def __init__(self) -> None:
        self._active: dict[str, int] = defaultdict(int)
        self._lock = asyncio.Lock()

    async def active(self, key: str) -> int:
        async with self._lock:
            return self._active[key]

    async def snapshot(self) -> dict[str, int]:
        async with self._lock:
            return {k: v for k, v in self._active.items() if v}

    @asynccontextmanager
    async def slot(self, key: str, limit: int):
        """Hold a slot for the duration of a request.

        Yields True when admitted and False when at the limit — the caller
        decides what to do, because a gateway that raised here would make every
        call site handle an exception for an ordinary outcome.

        The release is in a `finally` so a client that disconnects mid-stream
        does not leak its slot. That is the failure that turns a working limiter
        into a permanent lockout.
        """
        # Admission is decided under the lock, but the lock must NOT be held
        # across the yield: the caller runs its whole request inside this block,
        # and anything it does that touches the limiter would deadlock against
        # us. Decide, release, then yield.
        async with self._lock:
            admitted = self._active[key] < limit
            if admitted:
                self._active[key] += 1

        if not admitted:
            yield False
            return

        try:
            yield True
        finally:
            # released in `finally` so a client disconnecting mid-stream cannot
            # leak its slot — a leaked slot is a permanent lockout
            async with self._lock:
                self._active[key] = max(0, self._active[key] - 1)
