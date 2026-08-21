"""Eval runner: tiers 1-3 in one pass.

    python -m evals.run                     # whole golden set, once
    python -m evals.run --case infeasible-budget
    python -m evals.run --repeat 3          # tier 3: reliability across runs
    python -m evals.run --out report.json

Tier 1  quality  — deterministic scorers over the finished plan
Tier 2  cost     — LLM calls, tokens, tool calls, latency, cache hit rate
Tier 3  reliability — agent outcomes and failure taxonomy across repeats

Runs hit live models and live APIs, so they cost quota. That is deliberate: the
point is to measure the real system, and the caches make repeat runs much
cheaper than the first.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evals import baseline as baseline_mod            # noqa: E402
from evals.cases import CASES, Case, by_id           # noqa: E402
from evals.scorers import score_expectations, score_plan                  # noqa: E402
from orchestrator import plan_trip                    # noqa: E402
from scope import resolve_request                     # noqa: E402


def run_case(case: Case, fresh: bool) -> dict:
    request, _ = resolve_request(
        origin=case.origin,
        destination=case.destination,
        start_date=case.start_date,
        nights=case.nights,
        travelers=case.travelers,
        budget_usd=case.budget_usd,
        preferences=case.preferences,
    )
    state = plan_trip(request, remember=not fresh)

    card = score_plan(state)
    # the case's own declared properties, which the generic scorers cannot know
    card.add(*score_expectations(state, case.expect))
    metrics = state.get("metrics", {})

    return {
        "case": case.id,
        "why": case.why,
        "status": state.get("status"),
        "tier1_quality": card.as_dict(),
        "tier2_cost": metrics.get("totals", {}),
        "tier3_outcomes": {
            name: m["outcome"] for name, m in (metrics.get("agents") or {}).items()
        },
        "cache": state.get("cache", {}).get("hit_rate"),
        "errors": state.get("errors") or [],
        "warnings": state.get("warnings") or [],
    }


def summarise(results: list[dict]) -> dict:
    """Aggregate across cases and repeats — the tier-3 view."""
    outcomes: Counter = Counter()
    for r in results:
        outcomes.update(r["tier3_outcomes"].values())

    score_totals: dict[str, list[float]] = {}
    for r in results:
        for s in r["tier1_quality"]["scores"]:
            if s["value"] is not None:
                score_totals.setdefault(s["name"], []).append(s["value"])

    statuses = Counter(r["status"] for r in results)
    tokens = [r["tier2_cost"].get("total_tokens", 0) for r in results]
    seconds = [r["tier2_cost"].get("wall_seconds", 0) for r in results]

    return {
        "runs": len(results),
        "status_mix": dict(statuses),
        "agent_outcomes": dict(outcomes),
        "reliability": round(
            outcomes["done"] / max(sum(outcomes.values()), 1), 3
        ),
        "quality": {
            name: round(sum(v) / len(v), 3) for name, v in sorted(score_totals.items())
        },
        "critical_failures": [
            {"case": r["case"], "score": s["name"], "detail": s["detail"]}
            for r in results
            for s in r["tier1_quality"]["scores"]
            if s["critical"] and not s["passed"]
        ],
        "cost": {
            "tokens_mean": round(sum(tokens) / max(len(tokens), 1)),
            "tokens_max": max(tokens, default=0),
            "seconds_mean": round(sum(seconds) / max(len(seconds), 1), 1),
        },
    }


def main() -> int:
    p = argparse.ArgumentParser(description="trip planner evals")
    p.add_argument("--case", action="append", help="case id; repeatable")
    p.add_argument("--repeat", type=int, default=1, help="runs per case (tier 3)")
    p.add_argument("--fresh", action="store_true", help="ignore saved plans")
    p.add_argument("--out", help="write the full report as JSON")
    p.add_argument(
        "--save-baseline", action="store_true",
        help="overwrite evals/baseline.json with this run's scores",
    )
    p.add_argument(
        "--no-baseline", action="store_true", help="skip the regression check"
    )
    args = p.parse_args()

    from providers import replay

    if replay.active():
        print(f"EVAL_MODE={replay.mode()}  fixtures={replay.stats()['total']}\n")

    cases = [by_id(c) for c in args.case] if args.case else CASES
    results = []

    for case in cases:
        for attempt in range(args.repeat):
            label = f"{case.id}" + (f" #{attempt + 1}" if args.repeat > 1 else "")
            print(f"→ {label:28} {case.origin} to {case.destination}", flush=True)
            try:
                result = run_case(case, args.fresh)
            except Exception as exc:  # noqa: BLE001 - a crash is a result too
                result = {
                    "case": case.id, "why": case.why, "status": "crashed",
                    "tier1_quality": {"scores": [], "passed": False},
                    "tier2_cost": {}, "tier3_outcomes": {},
                    "errors": [f"{type(exc).__name__}: {exc}"], "warnings": [],
                }
            results.append(result)

            q = result["tier1_quality"]
            crit = [s["name"] for s in q["scores"] if s["critical"] and not s["passed"]]
            print(
                f"   status={result['status']}  quality={q.get('mean_value')}  "
                f"tokens={result['tier2_cost'].get('total_tokens', 0)}  "
                + (f"CRITICAL: {', '.join(crit)}" if crit else "critical: ok"),
                flush=True,
            )

    summary = summarise(results)
    report = {"summary": summary, "results": results}

    print("\n" + "=" * 64)
    print(json.dumps(summary, indent=1))

    # regression check: an absolute gate cannot see a score sliding from 1.0
    # to 0.7, because both pass any threshold below 0.7
    comparison = {"status": "skipped"}
    if not args.no_baseline:
        comparison = baseline_mod.compare(summary, baseline_mod.load())
        report["baseline"] = comparison
        print("\n--- vs baseline ---")
        if comparison["status"] == "no_baseline":
            print("  none stored yet; run with --save-baseline to create one")
        elif comparison["regressions"]:
            for r in comparison["regressions"]:
                print(
                    f"  REGRESSED {r['score']}: {r['before']} -> {r['after']}"
                    + (f"  ({r['delta']:+})" if "delta" in r else f"  ({r['detail']})")
                )
        else:
            print("  no regressions")
        for i in comparison.get("improvements", []):
            print(f"  improved  {i['score']}: {i['before']} -> {i['after']}")
        for n in comparison.get("new_scorers", []):
            print(f"  new       {n}")

    if args.save_baseline:
        print(f"\nwrote {baseline_mod.save(summary)}")

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1))
        print(f"wrote {args.out}")

    return 1 if (summary["critical_failures"] or comparison.get("regressions")) else 0


if __name__ == "__main__":
    raise SystemExit(main())
