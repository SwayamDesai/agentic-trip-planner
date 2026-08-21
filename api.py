"""HTTP layer for the trip planner.

Two endpoints, deliberately:

    POST /api/chat    one conversational turn. Returns either a question (when
                      something required is missing) or a ready-to-plan request.
    GET  /api/plan    Server-Sent Events. Streams agent progress as it happens,
                      then the finished plan.

Progress streaming exists because the interesting part of this system is the
agents working in parallel, and a run takes 60-150s on free-tier models. A
spinner for two minutes tells the user nothing; watching four agents call real
tools tells them exactly what they are waiting for.

The graph is synchronous and runs its fan-out in threads, so it executes in a
worker thread and pushes events into a queue that the async endpoint drains.
"""

import asyncio
import json
import os
import queue
import threading
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import agents.base as agent_base
import jobs
from chat import extract, is_ready, missing_fields
from models import TripRequest
from orchestrator import plan_trip
from scope import InvalidTripError, resolve_request
from status import OPTIONAL, REQUIRED

WEB_DIR = Path(__file__).parent / "web"

app = FastAPI(title="Trip Planner")

# The gateway is opt-in: without it the planner behaves exactly as before, which
# keeps the metering layer honestly separable from the application.
if os.getenv("GATEWAY", "").lower() in {"1", "true", "on"}:
    from gateway import install

    install(app, db_path=os.getenv("GATEWAY_DB", ".gateway.sqlite"))


# --- chat -----------------------------------------------------------------


class ChatTurn(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatTurn] = []


@app.post("/api/chat")
def chat(request: ChatRequest) -> dict:
    """Interpret one message and report whether the trip is plannable."""
    try:
        found = extract([t.model_dump() for t in request.history], request.message)
    except Exception as exc:  # noqa: BLE001 - surfaced to the user, not swallowed
        return {
            "reply": (
                "I could not read that just now — the language model is rate "
                "limited. Try again in a moment."
            ),
            "ready": False,
            "missing": list(REQUIRED),
            "error": type(exc).__name__,
        }

    return {
        "reply": found.reply,
        "ready": is_ready(found),
        "missing": missing_fields(found),
        "trip": found.model_dump(exclude={"reply"}),
    }


# --- planning -------------------------------------------------------------


def _sse(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, default=str)}\n\n"


def _plan_payload(state: dict, request: TripRequest) -> dict:
    """Flatten graph state into something the browser can render directly."""

    def dump(key):
        value = state.get(key)
        return value.model_dump() if value is not None else None

    return {
        "request": request.model_dump(),
        "status": state.get("status"),
        "flight": dump("flight"),
        "hotels": dump("hotels"),
        "weather": dump("weather"),
        "itinerary": dump("itinerary"),
        "budget": dump("budget"),
        "warnings": state.get("warnings") or [],
        "errors": state.get("errors") or [],
        "markdown": state.get("plan"),
        "metrics": state.get("metrics"),
        "cache": state.get("cache"),
        "required": list(REQUIRED),
        "optional": list(OPTIONAL),
    }


@app.get("/api/plan")
async def plan(
    origin: str,
    destination: str,
    start: str,
    end: Optional[str] = None,
    nights: Optional[int] = None,
    travelers: Optional[int] = None,
    budget: Optional[int] = None,
    prefer: str = "",
    fresh: bool = False,
):
    """Run the graph, streaming agent progress then the finished plan."""
    events: queue.Queue = queue.Queue()
    loop = asyncio.get_running_loop()

    def listener(event: dict) -> None:
        # called from agent worker threads
        loop.call_soon_threadsafe(events.put_nowait, ("agent", event))

    async def stream():
        request, scope_reason = resolve_request(
            origin=origin,
            destination=destination,
            start_date=start,
            end_date=end,
            nights=nights,
            travelers=travelers,
            budget_usd=budget,
            preferences=[p for p in prefer.split(",") if p.strip()],
        )

        yield _sse(
            "resolved",
            {
                "request": request.model_dump(),
                "scope_reason": scope_reason,
                "agents": list(REQUIRED) + list(OPTIONAL),
            },
        )

        agent_base.TRACE_LISTENERS.append(listener)
        done: dict = {}

        def run() -> None:
            try:
                done["state"] = plan_trip(request, remember=not fresh)
            except Exception as exc:  # noqa: BLE001 - reported to the browser
                done["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                loop.call_soon_threadsafe(events.put_nowait, ("done", {}))

        worker = threading.Thread(target=run, daemon=True)
        worker.start()

        try:
            while True:
                try:
                    kind, payload = events.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(0.15)
                    continue
                if kind == "done":
                    break
                yield _sse("agent", payload)
        finally:
            agent_base.TRACE_LISTENERS.remove(listener)

        # drain anything queued between the last poll and completion
        while True:
            try:
                kind, payload = events.get_nowait()
            except queue.Empty:
                break
            if kind == "agent":
                yield _sse("agent", payload)

        if "error" in done:
            yield _sse("failed", {"error": done["error"]})
        else:
            yield _sse("plan", _plan_payload(done["state"], request))

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# --- jobs -----------------------------------------------------------------
#
# The async model. `/api/plan` (below, kept for comparison) runs a plan INSIDE
# the request: 60-150s of work on one connection, destroyed by any deploy. These
# three endpoints separate accepting the work from doing it.


class PlanRequest(BaseModel):
    origin: str
    destination: str
    start: str
    end: Optional[str] = None
    nights: Optional[int] = None
    travelers: Optional[int] = None
    budget: Optional[int] = None
    prefer: list[str] = []
    fresh: bool = False


@app.post("/plans", status_code=202)
def create_plan(body: PlanRequest) -> dict:
    """Accept a plan and return immediately.

    202 Accepted, not 200 OK: the work has been accepted but not performed, and
    the status code should say so. Returns in milliseconds.

    Validation happens HERE rather than in the worker — a bad request should be
    rejected while the caller is still listening, not turned into a job that
    fails three times and dies in a dead-letter queue nobody reads.
    """
    try:
        request, scope_reason = resolve_request(
            origin=body.origin,
            destination=body.destination,
            start_date=body.start,
            end_date=body.end,
            nights=body.nights,
            travelers=body.travelers,
            budget_usd=body.budget,
            preferences=body.prefer,
        )
    except InvalidTripError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None

    job_id = jobs.enqueue(
        {"request": request.model_dump(), "remember": not body.fresh}
    )
    return {
        "job_id": job_id,
        "status": jobs.QUEUED,
        "request": request.model_dump(),
        "scope_reason": scope_reason,
        "poll": f"/plans/{job_id}",
        "events": f"/plans/{job_id}/events",
    }


@app.get("/plans/{job_id}")
def get_plan(job_id: str) -> dict:
    """Status, and the result once it exists."""
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="no such job")
    return job.as_dict()


@app.get("/plans/{job_id}/events")
async def plan_events(job_id: str, after: int = 0):
    """Stream progress for a job.

    The events come from the job's event log, not from memory: the worker is a
    different process now, so the API cannot observe agents directly. That is
    the first real cost of moving work out of the request, and this endpoint is
    where it is paid.

    Resumable via `?after=<last event id>`, so a browser that reconnects does
    not replay the whole run — which the in-request version could not offer at
    all, because a dropped connection lost the work with it.
    """
    if jobs.get(job_id) is None:
        raise HTTPException(status_code=404, detail="no such job")

    async def stream():
        cursor = after
        idle = 0.0
        while True:
            for event in jobs.events_since(job_id, cursor):
                cursor = event["id"]
                yield _sse(event["kind"], {"id": cursor, **event["data"]})
                idle = 0.0

            job = jobs.get(job_id)
            if job and job.terminal:
                yield _sse(
                    "result",
                    {"status": job.status, "result": job.result, "error": job.error},
                )
                return

            await asyncio.sleep(0.4)
            idle += 0.4
            if idle > 300:
                yield _sse("timeout", {"detail": "no progress for 5 minutes"})
                return

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/jobs/stats")
def job_stats() -> dict:
    """Queue depth. The number you alert on in production."""
    return jobs.stats()


# --- static ---------------------------------------------------------------


@app.get("/health")
def health() -> dict:
    """Liveness and readiness for the platform's health check.

    Reports whether the model provider is configured, because a container that
    boots without keys is running but cannot plan anything — and a health check
    that only proves the process started would call that healthy.

    Deliberately does no LLM call: a health probe that spends quota is a health
    probe that takes the service down.
    """
    from providers.llm import PROFILES

    configured = sorted(
        {
            key_name
            for _, _, key_name in PROFILES.values()
            if os.getenv(key_name)
        }
    )
    return {
        "status": "ok" if configured else "degraded",
        "llm_keys_configured": len(configured),
        "detail": (
            "ready" if configured else "no model provider key is set; planning will fail"
        ),
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


app.mount("/", StaticFiles(directory=WEB_DIR), name="web")
