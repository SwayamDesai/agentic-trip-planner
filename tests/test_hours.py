"""Reading OSM opening_hours well enough to avoid a closed door.

The governing rule: claiming a place is shut when it is open is worse than
saying nothing, so anything the parser does not confidently understand must
return None.
"""

from datetime import date

import pytest

from hours import closed_on, describe

MON, TUE, WED, SAT, SUN = (
    date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7),
    date(2026, 1, 10), date(2026, 1, 11),
)


# --- real specs, taken verbatim from Seville POIs ---


def test_explicit_day_off():
    spec = "Tu-Sa 11:00-17:30; Su 10:00-14:30; Mo off"   # Centro Cerámica Triana
    assert closed_on(spec, MON) is True
    assert closed_on(spec, TUE) is False
    assert closed_on(spec, SUN) is False


def test_days_not_listed_are_closed():
    spec = "Mo-Fr 09:30-14:00; Sa 10:00-14:00; PH off"   # Museo Histórico Militar
    assert closed_on(spec, MON) is False
    assert closed_on(spec, SAT) is False
    assert closed_on(spec, SUN) is True, "Sunday appears in no rule"


def test_open_every_day():
    spec = "Mo-Sa 09:30-17:00; Su 10:00-14:00"           # Archivo de Indias
    for day in (MON, TUE, SAT, SUN):
        assert closed_on(spec, day) is False


def test_range_excludes_both_ends_correctly():
    spec = "Tu-Sa 10:00-17:00"                            # Museo Arqueológico
    assert closed_on(spec, MON) is True
    assert closed_on(spec, SUN) is True
    assert closed_on(spec, TUE) is False
    assert closed_on(spec, SAT) is False


# --- the honest-unknown path ---


def test_seasonal_spec_is_unknown_not_guessed():
    """Real Alcázar: 'Oct-Mar: ...; Apr-Sep: ...'. A month-scoped rule could
    make every day look closed, so month syntax means unknown."""
    spec = "Oct-Mar: 09:30-17:00; Apr-Sep: 09:30-19:00"
    for day in (MON, TUE, SUN):
        assert closed_on(spec, day) is None


@pytest.mark.parametrize("spec", [
    None, "", "   ",
    "week 1-52 Mo-Su 10:00-18:00",       # week numbers
    "Su[-1] off",                        # nth weekday
    "sunrise-sunset",                    # solar times
])
def test_unparseable_returns_none(spec):
    assert closed_on(spec, MON) is None


def test_blanket_off_without_scope_is_unknown():
    """'off' with no day named could mean anything."""
    assert closed_on("off", MON) is None


# --- simple forms ---


def test_always_open():
    assert closed_on("24/7", MON) is False
    assert closed_on("24/7", SUN) is False


def test_times_without_weekdays_mean_every_day():
    assert closed_on("10:00-18:00", MON) is False
    assert closed_on("10:00-18:00", SUN) is False


def test_wrapping_range():
    """Fr-Mo spans the weekend boundary."""
    spec = "Fr-Mo 10:00-16:00"
    assert closed_on(spec, MON) is False
    assert closed_on(spec, SUN) is False
    assert closed_on(spec, WED) is True


def test_public_holiday_rules_are_ignored():
    """PH is a separate calendar; it must not affect weekday reasoning."""
    assert closed_on("Mo-Su 10:00-18:00; PH off", MON) is False


def test_case_insensitive():
    assert closed_on("tu-sa 10:00-17:00", MON) is True


# --- the label shown to the model ---


def test_describe_lists_closed_days():
    assert describe("Tu-Sa 10:00-17:00") == "closed Mo,Su"


def test_describe_is_silent_when_never_closed():
    assert describe("24/7") == ""
    assert describe("Mo-Su 09:00-17:00") == ""


def test_describe_is_silent_when_unknown():
    """Silence means unknown, and the prompt says so explicitly — a missing
    marker must not read as 'open'."""
    assert describe("Oct-Mar: 09:30-17:00") == ""
    assert describe(None) == ""
