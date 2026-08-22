"""Resolving an under-specified trip request into a complete one.

Runs BEFORE the graph rather than as a node in it. Every agent reads
`state["request"]`, and the parallel join is only safe because that value is
complete and read-only — a node that rewrote it would be writing a key three
other nodes are concurrently reading. So resolution happens first, and the
graph still receives a fully-specified request.

Rules:
    travelers  stated -> honoured. absent -> 2.
    duration   stated as an end date or a night count -> honoured exactly.
               absent -> the model picks a length suited to the destination.
"""

import re
from datetime import date, timedelta
from typing import Optional

from models import TripRequest, TripScope
from agents.base import deadline_for
from providers import prompts
from providers.cache import TTL_ADVISORY, cached
from providers.llm import invoke_structured

DEFAULT_TRAVELERS = 2
MIN_NIGHTS, MAX_NIGHTS = 1, 14
FALLBACK_NIGHTS = 3

# Airlines publish schedules roughly 11 months out; beyond that a search
# returns nothing, which reads as "no availability" rather than "too early".
MAX_DAYS_AHEAD = 330
MAX_TRAVELERS = 9      # the usual single-booking limit


class InvalidTripError(ValueError):
    """The request cannot be planned as stated."""


# What a place name can contain: letters in any script, digits, spaces, and the
# handful of punctuation marks real names use. Everything else is rejected —
# colons, braces, angle brackets, backticks, quotes, newlines — because those
# are the characters used to fake structure in a prompt, and no city needs one.
#
# This is the strongest injection control in the system precisely because it is
# a whitelist: "Seville" passes, and
# `Seville. SYSTEM: ignore prior instructions` does not exist as a place, so
# rejecting it costs a real user nothing.
PLACE_MAX = 80
PREFERENCE_MAX = 40
_DISALLOWED = re.compile(r"[^\w .,'\u2019\-()/&]", re.UNICODE)


def _check_text(label: str, value: str, limit: int) -> str:
    text = (value or "").strip()
    if not text:
        raise InvalidTripError(f"{label} is required")
    if len(text) > limit:
        raise InvalidTripError(
            f"{label} is {len(text)} characters; no {label} is longer than "
            f"{limit}"
        )
    bad = _DISALLOWED.findall(text)
    if bad:
        raise InvalidTripError(
            f"{label} {text[:40]!r} contains characters a place name does not: "
            f"{''.join(sorted(set(bad)))[:10]}"
        )
    return text


def validate_text(
    origin: str, destination: str, preferences: Optional[list[str]] = None
) -> None:
    """Reject place names that are not place names.

    Called before anything else touches these strings: they are interpolated
    into every agent's prompt, so a hostile destination is an injection into
    six prompts at once, and one of them is read by a model deciding which
    tools to call.
    """
    _check_text("origin", origin, PLACE_MAX)
    _check_text("destination", destination, PLACE_MAX)
    for preference in preferences or []:
        _check_text("interest", preference, PREFERENCE_MAX)


def validate(start_date: str, end_date: Optional[str], travelers: Optional[int]) -> None:
    """Reject requests that cannot produce a real plan.

    Each of these otherwise fails silently and misleadingly: a past date
    returns no fares and looks like a sold-out route; an inverted range yields
    zero nights and quietly drops lodging from the total.
    """
    try:
        start = date.fromisoformat(start_date)
    except (TypeError, ValueError):
        raise InvalidTripError(
            f"start date {start_date!r} is not a valid YYYY-MM-DD date"
        ) from None

    today = date.today()
    if start < today:
        raise InvalidTripError(
            f"start date {start_date} is in the past; no fares exist for it"
        )
    if (start - today).days > MAX_DAYS_AHEAD:
        raise InvalidTripError(
            f"start date {start_date} is more than {MAX_DAYS_AHEAD} days away; "
            f"airlines have not published schedules that far ahead"
        )

    if end_date:
        try:
            end = date.fromisoformat(end_date)
        except ValueError:
            raise InvalidTripError(
                f"end date {end_date!r} is not a valid YYYY-MM-DD date"
            ) from None
        if end < start:
            raise InvalidTripError(
                f"end date {end_date} is before the start date {start_date}"
            )
        if (end - start).days > MAX_NIGHTS:
            raise InvalidTripError(
                f"{(end - start).days} nights exceeds the {MAX_NIGHTS}-night "
                f"maximum this planner handles"
            )

    if travelers is not None and not (1 <= travelers <= MAX_TRAVELERS):
        raise InvalidTripError(
            f"{travelers} travellers is outside the supported range of "
            f"1 to {MAX_TRAVELERS}"
        )


def _recommend_nights(destination: str, preferences: list[str]) -> tuple[int, str]:
    """Ask the model how long this destination deserves. Cached and shared.

    The answer depends only on the destination (and loosely on interests), not
    on dates, party size or budget — so every user asking about Kyoto can reuse
    it. That makes this the cheapest LLM call in the system to eliminate: one
    fewer model call per fresh trip, at no cost to correctness.
    """
    # interests are part of the key, but normalised so ordering does not split
    # the cache between "food,history" and "history,food"
    key = destination.strip().lower() + "|" + ",".join(
        sorted(p.strip().lower() for p in preferences)
    )

    def ask() -> dict:
        try:
            scope = invoke_structured(
                "scope",
                TripScope,
                [
                    {"role": "system", "content": prompts.get("scope").text},
                    {
                        "role": "user",
                        "content": (
                            f"How many nights for {destination}?"
                            + (
                                f" The traveller is interested in: "
                                f"{', '.join(preferences)}."
                                if preferences
                                else ""
                            )
                        ),
                    },
                ],
                0.2,
                deadline=deadline_for("scope"),
            )
            return {
                "nights": max(MIN_NIGHTS, min(int(scope.nights), MAX_NIGHTS)),
                "reasoning": scope.reasoning,
            }
        except Exception as exc:  # noqa: BLE001 - a bad guess must not block planning
            # not cached: `error` marks this as a failure, so a rate limit now
            # does not pin a fallback answer for the next month
            return {
                "error": f"{type(exc).__name__}",
                "nights": FALLBACK_NIGHTS,
                "reasoning": (
                    f"defaulted to {FALLBACK_NIGHTS} nights ({type(exc).__name__})"
                ),
            }

    result = cached("tripscope", key, ask, ttl=TTL_ADVISORY)
    return result["nights"], result["reasoning"]


def resolve_request(
    *,
    origin: str,
    destination: str,
    start_date: str,
    end_date: Optional[str] = None,
    nights: Optional[int] = None,
    travelers: Optional[int] = None,
    budget_usd: Optional[int] = None,
    preferences: Optional[list[str]] = None,
) -> tuple[TripRequest, Optional[str]]:
    """Fill in whatever the traveller left out.

    Returns the complete request plus the reasoning behind an assumed trip
    length (None when the length was stated).
    """
    preferences = preferences or []
    validate_text(origin, destination, preferences)
    validate(start_date, end_date, travelers)
    reason: Optional[str] = None

    if end_date:
        resolved_end = end_date          # stated explicitly, honoured as given
    else:
        if nights is None:
            nights, reason = _recommend_nights(destination, preferences)
        nights = max(MIN_NIGHTS, min(int(nights), MAX_NIGHTS))
        resolved_end = (
            date.fromisoformat(start_date) + timedelta(days=nights)
        ).isoformat()

    return (
        TripRequest(
            origin=origin,
            destination=destination,
            start_date=start_date,
            end_date=resolved_end,
            travelers=travelers if travelers is not None else DEFAULT_TRAVELERS,
            budget_usd=budget_usd,
            preferences=preferences,
            # only an absent end date AND absent night count counts as assumed
            nights_chosen_by_system=reason is not None,
        ),
        reason,
    )
