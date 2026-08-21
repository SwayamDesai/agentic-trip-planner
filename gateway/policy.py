"""What each route costs, and which limits apply to it.

The central idea of a gateway that meters *business* work rather than raw
requests: **not all requests are equal**. In this app a plan run is roughly 40
LLM calls and ~40k tokens; a chat turn is one call and ~1k. Charging both "one
request" would let a user spend 40x their fair share while looking compliant.

So a route declares a COST in credits, and buckets are denominated in credits
rather than requests. That is exactly what GitHub does for its GraphQL API,
where a query's cost is computed from the nodes it touches — a request's cost
is not knowable from its URL alone, so the limit cannot live at the edge.

Costs here are anchored to measured token usage, so a credit means something.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class RoutePolicy:
    cost: int                    # credits
    concurrency: bool = False    # does it hold a concurrency slot?
    refundable: bool = False     # can the charge be returned if it did no work?
    description: str = ""


# Measured: a fresh plan is ~13 LLM calls / ~40k tokens; a chat turn is ~1k.
# 40:1 reflects that. A resumed plan does almost no work, hence refundable.
ROUTES: dict[tuple[str, str], RoutePolicy] = {
    ("GET", "/api/plan"): RoutePolicy(
        cost=40, concurrency=True, refundable=True,
        description="full multi-agent plan",
    ),
    ("POST", "/api/chat"): RoutePolicy(
        cost=1, description="one conversational turn",
    ),
}

# Anything not listed is free: health checks, the SPA, static assets. Metering
# static files would be noise, and a gateway that rate-limits its own dashboard
# is a gateway nobody can debug.
FREE_PREFIXES = ("/health", "/gateway", "/docs", "/openapi.json")


def policy_for(method: str, path: str) -> Optional[RoutePolicy]:
    """The policy for a request, or None when the route is unmetered."""
    if any(path.startswith(prefix) for prefix in FREE_PREFIXES):
        return None
    if path == "/" or "." in path.rsplit("/", 1)[-1]:
        return None  # the SPA and its assets
    return ROUTES.get((method.upper(), path))
