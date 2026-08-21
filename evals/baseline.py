"""Comparing an eval run against a stored baseline.

Absolute pass/fail answers "is it good enough". A baseline answers "did my last
change make it worse" — which is the question that actually matters when you are
editing six prompts, and the one an absolute gate cannot see: a score that slides
from 1.0 to 0.7 still "passes" any threshold below 0.7.

The baseline is committed, so a regression shows up in review as a diff.
"""

import json
from pathlib import Path
from typing import Optional

BASELINE_PATH = Path(__file__).parent / "baseline.json"

# Scores wobble slightly between runs on a non-deterministic model, so a
# regression must be larger than the noise to count. Only meaningful when
# replaying recorded responses, where runs are actually comparable.
TOLERANCE = 0.05


def load(path: Optional[Path] = None) -> Optional[dict]:
    path = path or BASELINE_PATH
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def save(summary: dict, path: Optional[Path] = None) -> Path:
    path = path or BASELINE_PATH
    payload = {
        "quality": summary.get("quality", {}),
        "reliability": summary.get("reliability"),
        "note": (
            "Recorded from an eval run. Regenerate with `python -m evals.run "
            "--save-baseline`. Compare with EVAL_MODE=replay for stable numbers."
        ),
    }
    path.write_text(json.dumps(payload, indent=1))
    return path


def compare(summary: dict, baseline: Optional[dict]) -> dict:
    """Per-scorer deltas against the baseline.

    A scorer present in the baseline but missing from this run is reported too:
    a check that silently stopped running is a regression in itself.
    """
    if not baseline:
        return {"status": "no_baseline", "regressions": [], "improvements": []}

    now = summary.get("quality", {})
    before = baseline.get("quality", {})

    regressions, improvements = [], []
    for name, was in sorted(before.items()):
        if name not in now:
            regressions.append(
                {"score": name, "before": was, "after": None,
                 "detail": "scorer no longer runs"}
            )
            continue
        delta = now[name] - was
        if delta < -TOLERANCE:
            regressions.append(
                {"score": name, "before": was, "after": now[name],
                 "delta": round(delta, 3)}
            )
        elif delta > TOLERANCE:
            improvements.append(
                {"score": name, "before": was, "after": now[name],
                 "delta": round(delta, 3)}
            )

    new_scorers = sorted(set(now) - set(before))

    return {
        "status": "regressed" if regressions else "ok",
        "regressions": regressions,
        "improvements": improvements,
        "new_scorers": new_scorers,
    }
