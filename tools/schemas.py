"""Validation at the tool boundary.

Agent *output* was already validated by Pydantic; tool *output* was not. That
asymmetry is dangerous in the opposite direction: an unchecked read turns
upstream API drift into silently wrong numbers instead of a loud failure.

Two concrete problems this exists to stop, both observed in live data:

  1. SerpApi returns hotel rates as strings — "$148", not 148. The agent schema
     wants a float, so the MODEL was doing the currency parsing, and its output
     fed straight into the budget arithmetic. "$1,234" could plausibly become
     1234, 1.234 or 1234000 with nothing to catch it.
  2. Some properties come back with a null rate (observed: Hotel Alfonso XIII).
     The agent schema requires a float, so the model had to invent one or
     silently drop the row. Neither was instructed.

So money is parsed in Python, and rows that cannot be validated are dropped
with a counted reason rather than passed through half-formed.
"""

import re
from typing import Optional

from pydantic import BaseModel, Field, ValidationError, field_validator

# Matches the numeric part of "$148", "1,234.50", "€1 234". Thousands
# separators are assumed to be commas or spaces: every request we make pins
# currency=USD and gl=us, so European decimal-comma format is not expected.
_MONEY_RE = re.compile(r"[-+]?\d[\d,\s]*(?:\.\d+)?")
_NON_PRICE = {"", "n/a", "na", "none", "null", "-", "unavailable"}


def parse_money(value: object) -> Optional[float]:
    """Coerce a price of unknown shape to a float, or None if there is none.

    Accepts numbers as-is and strings with currency symbols and separators.
    Returns None for absent or non-numeric values so the caller can decide
    whether to drop the row — never 0.0, which would read as "free".
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None

    text = value.strip().lower()
    if text in _NON_PRICE:
        return None

    match = _MONEY_RE.search(text)
    if not match:
        return None
    cleaned = match.group(0).replace(",", "").replace(" ", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


class _Money(BaseModel):
    """Base for rows whose money fields arrive as arbitrary strings."""

    model_config = {"extra": "allow"}


class FlightOffer(_Money):
    airline: str
    price_usd: float = Field(gt=0, description="per person")
    price_total_usd: float = Field(gt=0, description="whole party")
    departure_at: str
    arrival_at: Optional[str] = None
    stops: int = Field(ge=0)
    duration_minutes: Optional[int] = Field(default=None, ge=0)

    @field_validator("price_usd", "price_total_usd", mode="before")
    @classmethod
    def _coerce_price(cls, v):
        parsed = parse_money(v)
        if parsed is None:
            raise ValueError(f"unparseable price {v!r}")
        return parsed


class HotelOffer(_Money):
    name: str
    # required: a property with no rate cannot be costed, and letting the model
    # supply one is the fabrication this class exists to prevent
    price_per_night: float = Field(gt=0)
    rating: Optional[float] = Field(default=None, ge=0, le=5)
    hotel_class: Optional[str] = None

    @field_validator("price_per_night", mode="before")
    @classmethod
    def _coerce_rate(cls, v):
        parsed = parse_money(v)
        if parsed is None:
            raise ValueError(f"unparseable rate {v!r}")
        return parsed


class DayWeather(_Money):
    date: str
    condition: str
    high_c: float
    low_c: float
    precipitation_chance: int = Field(ge=0, le=100)


class Place(_Money):
    name: str
    kind: Optional[str] = None
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)


def validate_rows(rows: list[dict], model: type[BaseModel]) -> tuple[list[dict], int]:
    """Validate each row independently.

    One malformed row must not discard the good ones — a single odd listing
    should not turn a working search into "no results". Returns the surviving
    rows and how many were dropped, so the tool can report the loss instead of
    hiding it.
    """
    kept: list[dict] = []
    dropped = 0
    for row in rows:
        try:
            kept.append(model.model_validate(row).model_dump(exclude_none=True))
        except ValidationError:
            dropped += 1
    return kept, dropped


# --- errors ---------------------------------------------------------------
#
# A tool failure must reach the agent as a short, actionable sentence. Two
# failure shapes were making that impossible:
#
#   empty      {"options": []} says nothing about WHY, so the agent cannot tell
#              "no flights on this route" from "the API is down".
#   firehose   the raw provider payload. A Groq 429 body is ~700 characters of
#              JSON; pasted into the context it crowds out real data and, on a
#              tool-result cap of 2500 chars, can evict it entirely.
#
# So errors are summarised to a kind, one sentence, and whether retrying could
# possibly help.

RETRYABLE = "retryable"
PERMANENT = "permanent"

_ERROR_SIGNATURES = [
    # (marker in the raw text, kind, agent-facing message, retryable)
    ("tokens per day", "quota_exhausted",
     "The model's daily token quota is spent; this will not recover soon.", False),
    ("rate_limit", "rate_limited",
     "The service is rate limiting requests.", True),
    ("429", "rate_limited",
     "The service is rate limiting requests.", True),
    ("invalid_api_key", "auth_failed",
     "The API key was rejected.", False),
    ("401", "auth_failed", "The API key was rejected.", False),
    ("403", "auth_failed", "Access to this service was refused.", False),
    ("404", "not_found", "The service has no data at that address.", False),
    ("timeout", "upstream_timeout",
     "The service did not respond in time.", True),
    ("timed out", "upstream_timeout",
     "The service did not respond in time.", True),
    ("504", "upstream_down", "The service is overloaded.", True),
    ("503", "upstream_down", "The service is unavailable.", True),
    ("502", "upstream_down", "The service is unavailable.", True),
    ("500", "upstream_down", "The service reported an internal error.", True),
    ("connection", "network_error",
     "Could not reach the service.", True),
]


def tool_error(message: str, kind: str = "error", retryable: bool = False) -> dict:
    """Build the standard error payload a tool returns."""
    return {
        "error": message,
        "error_kind": kind,
        "retryable": retryable,
        "guidance": (
            "Retrying may help." if retryable
            else "Do NOT retry this call. Work around it or report it honestly."
        ),
    }


def summarize_exception(exc: BaseException, context: str = "") -> dict:
    """Turn an exception into a short, actionable tool error.

    Deliberately does not include the raw payload: the agent needs to know what
    happened and whether to retry, and a wall of provider JSON tells it neither
    while consuming the context budget.
    """
    text = f"{type(exc).__name__} {exc}".lower()
    for marker, kind, message, retryable in _ERROR_SIGNATURES:
        if marker in text:
            return tool_error(
                f"{context + ': ' if context else ''}{message}", kind, retryable
            )

    # Unrecognised: keep the exception type and a hard-truncated detail, never
    # the whole payload.
    detail = " ".join(str(exc).split())[:160]
    return tool_error(
        f"{context + ': ' if context else ''}{type(exc).__name__}: {detail}",
        "unexpected",
        False,
    )


def empty_result(what: str, reason: str = "") -> dict:
    """An honest empty result: says it is empty AND why."""
    return {
        "error": (
            f"No {what} found. {reason}".strip()
            if reason
            else f"No {what} found for the requested parameters."
        ),
        "error_kind": "no_results",
        "retryable": False,
        "guidance": (
            "This is a genuine absence of data, not a failure. Report it "
            "honestly; do not retry and do not substitute remembered values."
        ),
    }
