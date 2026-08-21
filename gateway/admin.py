"""Gateway introspection and key management.

Deliberately unmetered (see `policy.FREE_PREFIXES`): a gateway whose own status
endpoint is rate-limited is a gateway you cannot debug at exactly the moment you
need to.

In a real deployment these would sit behind admin auth. They are open here
because this is a learning build running on localhost, and pretending otherwise
would be security theatre.
"""

from fastapi import APIRouter
from pydantic import BaseModel

from gateway import state
from gateway.buckets import Bucket
from gateway.identity import TIERS
from gateway.policy import ROUTES
from gateway.state import GLOBAL_DAILY_CREDITS, GLOBAL_KEY

router = APIRouter(prefix="/gateway", tags=["gateway"])


class IssueRequest(BaseModel):
    tier: str = "free"
    label: str = ""


@router.get("/status")
async def status() -> dict:
    """Current budgets and in-flight work."""
    _gateway = state.current()
    if _gateway is None:
        return {"error": "gateway not installed"}

    global_bucket = Bucket.per_day(GLOBAL_DAILY_CREDITS)
    return {
        "global": {
            "credits_remaining": round(
                _gateway.buckets.peek(GLOBAL_KEY, global_bucket), 1
            ),
            "credits_per_day": GLOBAL_DAILY_CREDITS,
            "note": "shared by every caller; protects the upstream LLM budget",
        },
        "in_flight": await _gateway.concurrency.snapshot(),
        "tiers": TIERS,
        "route_costs": {
            f"{method} {path}": {
                "credits": p.cost,
                "holds_concurrency_slot": p.concurrency,
                "refundable": p.refundable,
            }
            for (method, path), p in ROUTES.items()
        },
    }


@router.get("/me")
async def me(api_key: str = "") -> dict:
    """What a given key is worth, without spending any of it."""
    _gateway = state.current()
    if _gateway is None:
        return {"error": "gateway not installed"}

    principal = _gateway.keys.resolve(api_key) if api_key else None
    if api_key and principal is None:
        return {"error": "unknown or revoked key"}
    if principal is None:
        return {"tier": "anonymous", "note": "no key supplied", "limits": TIERS["anonymous"]}

    limits = principal.limits
    bucket = Bucket.per_day(limits["daily_credits"], burst=limits["burst"])
    return {
        "tier": principal.tier,
        "label": principal.label,
        "limits": limits,
        "credits_remaining": round(_gateway.buckets.peek(principal.id, bucket), 1),
        "in_flight": await _gateway.concurrency.active(principal.id),
    }


@router.post("/keys")
async def issue_key(body: IssueRequest) -> dict:
    """Mint a key. Shown once — only its hash is stored."""
    _gateway = state.current()
    if _gateway is None:
        return {"error": "gateway not installed"}
    try:
        raw = _gateway.keys.issue(tier=body.tier, label=body.label)
    except ValueError as exc:
        return {"error": str(exc)}
    return {
        "api_key": raw,
        "tier": body.tier,
        "warning": "Store this now; it cannot be retrieved again.",
        "usage": "Authorization: Bearer <key>  or  X-API-Key: <key>",
    }


@router.get("/keys")
async def list_keys() -> dict:
    _gateway = state.current()
    if _gateway is None:
        return {"error": "gateway not installed"}
    return {"keys": _gateway.keys.list_keys()}
