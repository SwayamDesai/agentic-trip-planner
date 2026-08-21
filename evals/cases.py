"""Golden set: scenarios chosen to exercise edge cases, not to be typical.

Deliberately NOT expected outputs. There is no single correct itinerary for
three days in Seville, so comparing against a fixed plan would test for
sameness rather than quality. Each case instead declares the PROPERTIES its
output must hold, which the tier-1 scorers check.
"""

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional


def _days_out(n: int) -> str:
    return (date.today() + timedelta(days=n)).isoformat()


@dataclass
class Case:
    id: str
    why: str                       # the edge this case exists to probe
    origin: str
    destination: str
    start_offset: int              # days from today, so cases never go stale
    nights: Optional[int] = None
    travelers: Optional[int] = None
    budget_usd: Optional[int] = None
    preferences: list[str] = field(default_factory=list)
    expect: dict = field(default_factory=dict)

    @property
    def start_date(self) -> str:
        return _days_out(self.start_offset)


CASES: list[Case] = [
    Case(
        "feasible-budget", "budget comfortably achievable; cap must be respected",
        "Chicago", "Lisbon", 40, nights=3, travelers=2, budget_usd=6000,
        preferences=["food", "history"],
        expect={"feasible": True, "status": "ok"},
    ),
    Case(
        "infeasible-budget", "cheapest travel alone busts the budget; must say so and go free-only",
        "Chicago", "Tokyo", 45, nights=4, travelers=2, budget_usd=800,
        expect={"feasible": False, "activities_usd": 0},
    ),
    Case(
        "no-budget", "no cap given, so mid-tier costing and no verdict",
        "Boston", "Seville", 50, nights=3,
        expect={"tier": "mid"},
    ),
    Case(
        "defaults-only", "only origin/destination/date: 2 travellers and a chosen length",
        "New York", "Porto", 55,
        expect={"travelers": 2, "nights_chosen_by_system": True},
    ),
    Case(
        "solo-traveller", "one traveller must not be bumped to the default of 2",
        "Chicago", "Granada", 42, nights=2, travelers=1, budget_usd=3000,
        expect={"travelers": 1},
    ),
    Case(
        "tiny-airport", "GRX has almost no cached fares; must not invent any",
        "Chicago", "Granada", 44, nights=3, travelers=2,
        expect={},
    ),
    Case(
        "non-latin-names", "OSM returns Japanese names; English should be preferred",
        "Chicago", "Kyoto", 60, nights=5, travelers=2, budget_usd=9000,
        preferences=["history"],
        expect={},
    ),
    Case(
        "forecast-window", "start inside ~16 days, so a real forecast should be used",
        "Chicago", "Lisbon", 8, nights=2, travelers=2,
        expect={"weather_source": "forecast"},
    ),
    Case(
        "beyond-forecast", "start far out, so climate normals and honest labelling",
        "Chicago", "Lisbon", 120, nights=3, travelers=2,
        expect={"weather_source": "climate_normals"},
    ),
    Case(
        "single-night", "shortest real trip; one night of lodging",
        "Chicago", "Madrid", 47, nights=1, travelers=2, budget_usd=4000,
        expect={"nights": 1},
    ),
    Case(
        "long-trip", "10 nights; day coverage must not thin out at the end",
        "Chicago", "Rome", 65, nights=10, travelers=2, budget_usd=12000,
        preferences=["history", "food"],
        expect={"nights": 10},
    ),
    Case(
        "wet-destination", "high rain chance should push activities indoors",
        "Chicago", "Bergen", 70, nights=3, travelers=2, budget_usd=6000,
        expect={},
    ),
]


def by_id(case_id: str) -> Case:
    for case in CASES:
        if case.id == case_id:
            return case
    raise KeyError(case_id)
