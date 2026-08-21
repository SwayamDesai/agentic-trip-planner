"""Reading OpenStreetMap `opening_hours` well enough to avoid a closed door.

The full specification is large and ambiguous, and a partial parser that
pretends otherwise is worse than none: scheduling a visit for a day the place
is shut is bad, but *claiming* a place is shut when it is not is worse.

So this answers a deliberately narrow question — "is this place closed on this
weekday?" — with three possible answers:

    True    definitely closed
    False   definitely open
    None    the spec is beyond this parser; say nothing

Anything it does not confidently understand returns None. Roughly a third of
notable POIs carry the tag at all, so partial coverage is the realistic ceiling
and unknown must be a first-class result rather than a guess.
"""

import re
from datetime import date
from typing import Optional

DAYS = ["Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"]
_INDEX = {day: i for i, day in enumerate(DAYS)}

# Constructs this parser does not attempt. Seeing any of them means "unknown",
# because guessing at a seasonal or week-numbered rule invites a false
# "closed" — the one error worse than no information.
_UNSUPPORTED = re.compile(
    r"""
    \b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b   # month ranges
    | week\s*\d                                              # week numbers
    | \[\s*-?\d                                              # nth-weekday
    | sunrise | sunset | dawn | dusk
    | \bopen\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

_DAY_TOKEN = re.compile(r"\b(Mo|Tu|We|Th|Fr|Sa|Su)\b", re.IGNORECASE)
_DAY_RANGE = re.compile(
    r"\b(Mo|Tu|We|Th|Fr|Sa|Su)\s*-\s*(Mo|Tu|We|Th|Fr|Sa|Su)\b", re.IGNORECASE
)
_HAS_TIME = re.compile(r"\d{1,2}:\d{2}")


def _canonical(token: str) -> str:
    return token[0].upper() + token[1:].lower()


def _days_in(fragment: str) -> set[str]:
    """Weekdays named by a fragment, expanding ranges like Tu-Sa."""
    days: set[str] = set()

    for start, end in _DAY_RANGE.findall(fragment):
        i, j = _INDEX[_canonical(start)], _INDEX[_canonical(end)]
        # ranges wrap: Sa-Su, and Fr-Mo
        days.update(DAYS[i:j + 1] if i <= j else DAYS[i:] + DAYS[:j + 1])

    remaining = _DAY_RANGE.sub(" ", fragment)
    days.update(_canonical(t) for t in _DAY_TOKEN.findall(remaining))
    return days


def closed_on(spec: Optional[str], when: date) -> Optional[bool]:
    """Is a place with this `opening_hours` closed on `when`?

    Returns None whenever the answer is not clear from the parts of the
    specification this parser covers.
    """
    if not spec or not spec.strip():
        return None

    text = spec.strip()
    if text in {"24/7", "24/7; PH open"}:
        return False
    if _UNSUPPORTED.search(text):
        return None

    weekday = DAYS[when.weekday()]
    open_days: set[str] = set()
    closed_days: set[str] = set()
    saw_rule = False

    for rule in text.split(";"):
        rule = rule.strip()
        if not rule or rule.lower().startswith("ph"):
            continue  # public holidays are a separate calendar

        days = _days_in(rule)
        says_off = re.search(r"\boff\b|\bclosed\b", rule, re.IGNORECASE)

        if says_off:
            if not days:
                return None  # a blanket "off" with no scope: unclear
            closed_days |= days
            saw_rule = True
        elif _HAS_TIME.search(rule):
            saw_rule = True
            if days:
                open_days |= days
            else:
                # times with no weekday means every day, e.g. "10:00-18:00"
                open_days |= set(DAYS)

    if not saw_rule:
        return None

    if weekday in closed_days:
        return True
    if open_days:
        return weekday not in open_days
    # only "off" rules were present, and this day was not among them
    return False


def describe(spec: Optional[str]) -> str:
    """Short, honest label for a candidate list."""
    if not spec:
        return ""
    closed = [d for d in DAYS if closed_on(spec, _sample_date(d)) is True]
    if not closed:
        return ""
    return "closed " + ",".join(closed)


def _sample_date(day: str) -> date:
    """Any date falling on `day`, for probing a spec."""
    base = date(2026, 1, 5)  # a Monday
    from datetime import timedelta

    return base + timedelta(days=_INDEX[day])
