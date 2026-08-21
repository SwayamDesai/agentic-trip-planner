"""One error shape for everything the gateway refuses.

A gateway that invents its own error format per failure mode forces every
client to special-case it. This reuses the envelope the project's tools already
use — `error`, `error_kind`, `retryable`, `guidance` — so the frontend renders a
gateway refusal exactly like any other failure, and the field that matters most
is explicit: whether retrying could possibly help.
"""

from typing import Optional

from fastapi.responses import JSONResponse


def refusal(
    status: int,
    kind: str,
    message: str,
    guidance: str,
    retryable: bool,
    headers: Optional[dict] = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={
            "error": message,
            "error_kind": kind,
            "retryable": retryable,
            "guidance": guidance,
        },
        headers=headers or {},
    )


def rate_limited(decision, scope: str) -> JSONResponse:
    wait = int(decision.retry_after + 0.5)
    if scope == "global":
        message = (
            f"The shared daily quota is spent. It refills gradually; about "
            f"{wait}s until this request would fit."
        )
        guidance = (
            "This is not your limit — the whole service shares one upstream "
            "budget. Retry later; nothing you change will help sooner."
        )
    else:
        message = (
            f"Rate limit reached. About {wait}s until this request would fit."
        )
        guidance = "Retry after the interval in Retry-After."
    return refusal(429, "rate_limited", message, guidance, True, decision.headers())


def too_many_concurrent(limit: int, active: int) -> JSONResponse:
    return refusal(
        429,
        "too_many_concurrent",
        f"{active} request(s) already running and the limit is {limit}.",
        "Wait for the current request to finish before starting another.",
        True,
        {"Retry-After": "10"},
    )


def unauthorized() -> JSONResponse:
    return refusal(
        401,
        "invalid_api_key",
        "The API key was not recognised.",
        "Check the key, or omit it entirely to use the anonymous tier.",
        False,
    )
