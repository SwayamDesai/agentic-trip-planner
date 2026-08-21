"""CLI entrypoint.

Only origin, destination and a start date are required. Anything omitted is
filled in: travellers defaults to 2, and if neither --end nor --nights is
given the system picks a trip length suited to the destination.

    python main.py --destination Lisbon --start 2026-09-10
    python main.py --destination Lisbon --start 2026-09-10 --nights 5
    python main.py --destination Lisbon --start 2026-09-10 --end 2026-09-16 \
        --travelers 3 --budget 4000
"""

import argparse
import sys

from orchestrator import plan_trip
from providers.memory import forget, list_trips, thread_id
from scope import resolve_request


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="multi-agent trip planner")
    p.add_argument("--origin", default="Chicago")
    p.add_argument("--destination", default="Lisbon")
    p.add_argument("--start", default="2026-09-10", help="YYYY-MM-DD")
    p.add_argument(
        "--end", default=None, help="YYYY-MM-DD. Omit to let the system decide."
    )
    p.add_argument(
        "--nights", type=int, default=None,
        help="trip length. Omit (with --end) to let the system decide.",
    )
    p.add_argument(
        "--travelers", type=int, default=None, help="defaults to 2 if omitted"
    )
    p.add_argument(
        "--budget", type=int, default=None,
        help="total USD. Omit for a mid-range plan with no cap.",
    )
    p.add_argument(
        "--prefer",
        action="append",
        default=[],
        help="repeatable, e.g. --prefer food --prefer museums",
    )
    p.add_argument(
        "--fresh",
        action="store_true",
        help="ignore saved results for this trip and re-run every agent",
    )
    p.add_argument(
        "--forget", action="store_true", help="delete saved state for this trip and exit"
    )
    p.add_argument("--list", action="store_true", help="list saved trips and exit")
    a = p.parse_args()

    return a


def main() -> None:
    args = parse_args()

    # resolve before anything else: --forget and the plan both key off the
    # complete request, so an assumed trip length must be settled first
    req, scope_reason = resolve_request(
        origin=args.origin,
        destination=args.destination,
        start_date=args.start,
        end_date=args.end,
        nights=args.nights,
        travelers=args.travelers,
        budget_usd=args.budget,
        preferences=args.prefer,
    ) if not args.list else (None, None)

    if args.list:
        trips = list_trips()
        print(f"{len(trips)} saved trip(s)")
        for t in trips:
            print(f"  {t['thread_id']}  {t['checkpoints']} checkpoint(s)")
        return

    if args.forget:
        print(f"removed {forget(req)} row(s) for trip {thread_id(req)}")
        return

    if scope_reason:
        print(
            f"No trip length given; using {req.start_date} to {req.end_date} "
            f"— {scope_reason}"
        )
    mode = "fresh" if args.fresh else f"resuming trip {thread_id(req)}"
    print(f"Planning {req.origin} -> {req.destination} ({mode}) ...\n")
    state = plan_trip(req, remember=not args.fresh)
    print(state.get("plan") or "no plan produced")

    status = state.get("status")
    if status == "failed":
        print(
            "\nRun FAILED: required data was missing. Re-run to retry only the "
            "failed agents.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    if status == "degraded":
        print("\nRun DEGRADED: see the note at the top of the plan.", file=sys.stderr)


if __name__ == "__main__":
    main()
