"""The gateway itself: the order in which checks run, and why.

Ordering is the design. Each check is cheap relative to the one after it, and
each can refuse, so the sequence is arranged so the cheapest refusal happens
first and nothing expensive is spent on a request that was never going to run:

    1. route policy   free routes exit immediately — no identity lookup, no DB
    2. identity       one hashed DB read; a bad key is refused before any spend
    3. concurrency    an in-memory counter; refuses without touching a bucket,
                      so a burst of parallel requests does not drain the quota
                      it was never allowed to use
    4. global bucket  protects the SHARED upstream budget. Checked BEFORE the
                      per-user bucket so a user is not charged for capacity the
                      service does not have.
    5. user bucket    fairness between callers
    6. upstream       the actual application
    7. refund         return the charge if the work turned out to be free

Two decisions worth defending:

**Charge before, refund after.** The cost of a request is not knowable until it
runs — a resumed plan reuses cached agents and does almost nothing. Charging
optimistically and refunding is safe; charging afterwards is not, because a
caller could start unlimited concurrent work before any of it is billed.

**Refuse a stream before it opens.** A 429 emitted mid-SSE looks to
`EventSource` like a network fault, and the browser reconnects — turning one
refusal into a retry loop. So metering happens before the response begins.
"""

import time
import uuid
from pathlib import Path

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from gateway import errors, state
from gateway.buckets import Bucket
from gateway.identity import anonymous, extract_key
from gateway.policy import policy_for
from gateway.state import GLOBAL_KEY

class GatewayMiddleware(BaseHTTPMiddleware):
    """Stores come from `gateway.state`, shared with the admin routes."""

    def __init__(self, app, db_path: Path | str = ".gateway.sqlite"):
        super().__init__(app)
        self.gw = state.current() or state.build(db_path)

    @property
    def buckets(self):
        return self.gw.buckets

    @property
    def keys(self):
        return self.gw.keys

    @property
    def concurrency(self):
        return self.gw.concurrency

    @property
    def global_bucket(self):
        return self.gw.global_bucket

    async def dispatch(self, request: Request, call_next):
        started = time.perf_counter()
        request_id = uuid.uuid4().hex[:12]

        # 1. unmetered routes leave immediately: no identity, no database
        policy = policy_for(request.method, request.url.path)
        if policy is None:
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            return response

        # 2. identity. A supplied-but-invalid key is an error rather than a
        # silent downgrade to anonymous — quietly giving someone a smaller
        # limit than they think they have is a bad way to spend their day.
        raw_key = extract_key(request.headers)
        if raw_key:
            principal = self.keys.resolve(raw_key)
            if principal is None:
                return errors.unauthorized()
        else:
            client = request.client.host if request.client else "unknown"
            principal = anonymous(client)

        limits = principal.limits
        user_bucket = Bucket.per_day(limits["daily_credits"], burst=limits["burst"])

        # 3. concurrency, before any spend
        if policy.concurrency:
            async with self.concurrency.slot(
                principal.id, limits["max_concurrent"]
            ) as admitted:
                if not admitted:
                    active = await self.concurrency.active(principal.id)
                    return errors.too_many_concurrent(
                        limits["max_concurrent"], active
                    )
                return await self._metered(
                    request, call_next, principal, policy, user_bucket,
                    request_id, started,
                )

        return await self._metered(
            request, call_next, principal, policy, user_bucket, request_id, started,
        )

    async def _metered(
        self, request, call_next, principal, policy, user_bucket,
        request_id, started,
    ):
        # 4. the shared budget first: no point charging a user for capacity the
        # service does not have
        globally = self.buckets.take(
            GLOBAL_KEY, policy.cost, self.global_bucket, scope="global"
        )
        if not globally.allowed:
            return errors.rate_limited(globally, "global")

        # 5. the caller's own budget
        personal = self.buckets.take(
            principal.id, policy.cost, user_bucket, scope="user"
        )
        if not personal.allowed:
            # the global charge succeeded, so give it back — the request is not
            # going to consume that capacity
            self.buckets.refund(GLOBAL_KEY, policy.cost, self.global_bucket)
            return errors.rate_limited(personal, "user")

        # 6. upstream
        try:
            response = await call_next(request)
        except Exception:
            # an upstream crash did no billable work
            self.buckets.refund(principal.id, policy.cost, user_bucket)
            self.buckets.refund(GLOBAL_KEY, policy.cost, self.global_bucket)
            raise

        # 7. refund work that turned out to be free. The application signals
        # this with a header, so the gateway needs no knowledge of what a plan
        # is — it stays a metering layer rather than becoming coupled to the app.
        if policy.refundable and response.headers.get("X-Work-Performed") == "none":
            self.buckets.refund(principal.id, policy.cost, user_bucket)
            self.buckets.refund(GLOBAL_KEY, policy.cost, self.global_bucket)
            personal.remaining += policy.cost

        for name, value in personal.headers().items():
            response.headers[name] = value
        response.headers["X-Request-ID"] = request_id
        response.headers["X-RateLimit-Cost"] = str(policy.cost)
        response.headers["X-Principal-Tier"] = principal.tier
        response.headers["X-Response-Time-Ms"] = (
            f"{(time.perf_counter() - started) * 1000:.0f}"
        )
        return response
