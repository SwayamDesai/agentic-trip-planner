"""Turning conversation into a trip request.

The planner needs a structured request; a person types "long weekend in Lisbon
in October, two of us, around $3k". This module bridges the two, and asks for
what is genuinely missing rather than guessing.

Only three things cannot be defaulted: where from, where to, and when. Everything
else has a defined fallback (2 travellers, a system-chosen length, a mid-range
plan with no budget cap), so the conversation stays short.
"""

from datetime import date
from typing import Optional

from pydantic import BaseModel, Field

from agents.base import deadline_for
from providers.llm import invoke_structured

REQUIRED_FIELDS = ("origin", "destination", "start_date")


class TripExtraction(BaseModel):
    """What the conversation so far establishes about the trip."""

    origin: Optional[str] = Field(default=None, description="departure city")
    destination: Optional[str] = Field(default=None, description="destination city")
    start_date: Optional[str] = Field(
        default=None, description="YYYY-MM-DD, resolved from relative phrasing"
    )
    end_date: Optional[str] = Field(default=None, description="YYYY-MM-DD if stated")
    nights: Optional[int] = Field(default=None, description="only if explicitly stated")
    travelers: Optional[int] = Field(default=None, description="only if stated")
    budget_usd: Optional[int] = Field(
        default=None, description="total trip budget in USD, only if stated"
    )
    preferences: list[str] = Field(
        default_factory=list, description="interests such as food, history, hiking"
    )
    reply: str = Field(
        description=(
            "One or two sentences to the traveller. If something required is "
            "missing, ask for exactly that. Otherwise confirm what you have."
        )
    )


SYSTEM = """You collect trip details from a conversation. Today is {today}.

Extract only what the traveller has actually said or clearly implied. Leave a
field null rather than guessing it.

Resolve relative dates against today: "next month" -> the 1st of next month,
"first week of October" -> that month's 1st. Always output YYYY-MM-DD.

Set `nights` ONLY if a length was stated ("4 nights", "long weekend" = 3).
Set `travelers` ONLY if a number was stated; "my wife and I" is 2, "solo" is 1.
Set `budget_usd` ONLY if an amount was stated; "around 3k" is 3000. A budget is
a hard cap, so never invent one.

You need origin, destination and a start date. If any is missing, ask for the
missing ones in `reply` — briefly, and all at once rather than one at a time.
If you have all three, confirm the trip in one sentence and say you are
planning it. Do not list what you defaulted.

Never invent prices, airlines or hotels; you only gather requirements."""


def extract(history: list[dict], message: str) -> TripExtraction:
    """Merge a new message into what earlier turns established."""
    transcript = "\n".join(
        f"{turn['role']}: {turn['content']}" for turn in history[-8:]
    )
    return invoke_structured(
        "chat",
        TripExtraction,
        [
            {"role": "system", "content": SYSTEM.format(today=date.today().isoformat())},
            {
                "role": "user",
                "content": (
                    f"Conversation so far:\n{transcript or '(nothing yet)'}\n\n"
                    f"New message: {message}\n\n"
                    "Extract the trip details known at this point."
                ),
            },
        ],
        0.1,
        deadline=deadline_for("chat"),
    )


def missing_fields(extraction: TripExtraction) -> list[str]:
    """Required fields still unknown."""
    return [f for f in REQUIRED_FIELDS if not getattr(extraction, f, None)]


def is_ready(extraction: TripExtraction) -> bool:
    return not missing_fields(extraction)
