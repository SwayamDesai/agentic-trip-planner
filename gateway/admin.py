"""Gateway introspection and key management.

Deliberately unmetered (see `policy.FREE_PREFIXES`): a gateway whose own status
endpoint is rate-limited is a gateway you cannot debug at exactly the moment you
need to.

In a real deployment these would sit behind admin auth. They are open here
because this is a learning build running on localhost, and pretending otherwise
would be security theatre.
"""

from fastapi import APIRouter, Request
from pydantic import BaseModel

from gateway import state
from gateway.buckets import Bucket
from gateway.identity import TIERS, anonymous, client_ip
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
async def me(request: Request, api_key: str = "") -> dict:
    """What the caller is worth, without spending any of it.

    Reports the bucket the caller lands in, not just the tier. Behind a proxy
    that is the only way to answer "why did I get a 429" — the difference
    between one shared bucket and one per visitor is invisible from the outside,
    and was in fact wrong here until someone looked.
    """
    _gateway = state.current()
    if _gateway is None:
        return {"error": "gateway not installed"}

    principal = _gateway.keys.resolve(api_key) if api_key else None
    if api_key and principal is None:
        return {"error": "unknown or revoked key"}
    if principal is None:
        anon = anonymous(
            client_ip(
                request.client.host if request.client else None,
                request.headers.get("x-forwarded-for"),
            )
        )
        limits = anon.limits
        bucket = Bucket.per_day(limits["daily_credits"], burst=limits["burst"])
        return {
            "tier": anon.tier,
            "note": "no key supplied",
            "identity": anon.id,
            "limits": limits,
            "credits_remaining": round(
                _gateway.buckets.peek(anon.id, bucket), 1
            ),
        }

    limits = principal.limits
    bucket = Bucket.per_day(limits["daily_credits"], burst=limits["burst"])
    return {
        "tier": principal.tier,
        "label": principal.label,
        "identity": principal.id,
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
