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

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import agents.base as agent_base
from chat import extract, is_ready, missing_fields
from models import TripRequest
from orchestrator import plan_trip
from scope import resolve_request
from status import OPTIONAL, REQUIRED

WEB_DIR = Path(__file__).parent / "web"

app = FastAPI(title="Trip Planner")

# The gateway is opt-in: without it the planner behaves exactly as before, which
# keeps the metering layer honestly separable from the application.
if os.getenv("GATEWAY", "").lower() in {"1", "true", "on"}:
    from gateway import install

    install(app)


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


# --- static ---------------------------------------------------------------


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


app.mount("/", StaticFiles(directory=WEB_DIR), name="web")
