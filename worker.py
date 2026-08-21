"""The worker: a separate process that takes jobs and runs plans.

    python worker.py            # one worker
    python worker.py --once     # drain the queue and exit, for tests and CI

This is the half of the split that does the work. The API process no longer
plans anything — it enqueues and reads results — which is what makes deploys
safe and workers scalable.

Two things a worker must do that a request handler never had to:

**Heartbeat while working.** A plan runs far longer than the lease, so the
worker extends it on a timer. Stop heartbeating and the job returns to the
queue — which is exactly the behaviour wanted when a worker is killed.

**Report progress out of process.** Agent progress used to be observed in
memory by the same process serving the request. It cannot be now, so the trace
listener writes events to the job's event log and the API reads them from there.
"""

import argparse
import os
import signal
import socket
import threading
import time

import agents.base as agent_base
import jobs
from models import TripRequest
from orchestrator import plan_trip

POLL_SECONDS = float(os.getenv("WORKER_POLL_SECONDS", 1.0))
HEARTBEAT_SECONDS = max(5.0, jobs.LEASE_SECONDS / 3)

# Set by SIGTERM. The worker finishes its current job and then exits, rather
# than dropping it — this is what makes a rolling deploy lossless.
_stopping = threading.Event()


def worker_id() -> str:
    return f"{socket.gethostname()}:{os.getpid()}"


def _run_job(job: jobs.Job) -> None:
    """Run one plan, streaming progress into the job's event log."""
    payload = job.payload
    request = TripRequest.model_validate(payload["request"])

    def listener(event: dict) -> None:
        # called from the fan-out's agent threads
        jobs.append_event(job.id, "agent", event)

    stop_heartbeat = threading.Event()

    def beat() -> None:
        while not stop_heartbeat.wait(HEARTBEAT_SECONDS):
            jobs.heartbeat(job.id)

    heart = threading.Thread(target=beat, daemon=True)
    heart.start()
    agent_base.TRACE_LISTENERS.append(listener)

    try:
        jobs.append_event(job.id, "started", {"worker": worker_id(),
                                              "attempt": job.attempts})
        state = plan_trip(request, remember=payload.get("remember", True))

        result = {
            "status": state.get("status"),
            "plan": state.get("plan"),
            "warnings": state.get("warnings") or [],
            "errors": state.get("errors") or [],
            "metrics": state.get("metrics"),
            "sections": {
                key: (state[key].model_dump() if state.get(key) is not None else None)
                for key in ("flight", "hotels", "weather", "itinerary", "budget")
            },
            "request": request.model_dump(),
        }
        jobs.complete(job.id, result)
        jobs.append_event(job.id, "finished", {"status": result["status"]})
    except Exception as exc:  # noqa: BLE001 - the queue decides whether to retry
        jobs.fail(job.id, f"{type(exc).__name__}: {exc}")
        jobs.append_event(job.id, "failed", {"error": f"{type(exc).__name__}: {exc}"})
    finally:
        stop_heartbeat.set()
        agent_base.TRACE_LISTENERS.remove(listener)


def run(once: bool = False) -> int:
    """Poll and process until stopped, or until the queue drains if `once`.

    Polling is the simplest correct thing and is fine at this scale. A real
    queue pushes (Redis BLPOP, SQS long-poll) rather than being asked every
    second — that is a Phase 2 change, not a correctness one.
    """
    jobs.init()
    processed = 0
    me = worker_id()
    print(f"worker {me} started (lease {jobs.LEASE_SECONDS}s, "
          f"max attempts {jobs.MAX_ATTEMPTS})", flush=True)

    while not _stopping.is_set():
        job = jobs.claim(me)
        if job is None:
            if once:
                break
            _stopping.wait(POLL_SECONDS)
            continue

        print(f"  → job {job.id} (attempt {job.attempts})", flush=True)
        started = time.perf_counter()
        _run_job(job)
        processed += 1
        print(f"  ← job {job.id} in {time.perf_counter() - started:.1f}s", flush=True)

    print(f"worker {me} stopped after {processed} job(s)", flush=True)
    return processed


def _handle_signal(signum, _frame) -> None:
    # graceful: stop taking new work, let the current job finish
    print(f"\nsignal {signum} received; finishing current job then exiting",
          flush=True)
    _stopping.set()


def main() -> None:
    parser = argparse.ArgumentParser(description="trip planner worker")
    parser.add_argument("--once", action="store_true",
                        help="drain the queue and exit")
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)
    run(once=args.once)


if __name__ == "__main__":
    main()
