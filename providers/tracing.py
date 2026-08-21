"""Langfuse tracing. Entirely optional.

Every LLM call becomes a generation observation, and every agent becomes a span
under one trace per plan run, so a run can be inspected end to end: which agent
called which tool, what the prompt was, how many tokens it cost, and where it
failed.

Two design constraints shaped this:

  1. It must no-op cleanly. Without Langfuse keys the whole module degrades to
     empty callback lists and a null context manager — the planner runs
     identically, and nothing in the agents knows tracing exists.

  2. The fan-out runs agents in worker threads. OpenTelemetry context does not
     cross a thread boundary on its own, so agent observations would each start
     their own trace. Fixed by minting one trace id per run and handing it to
     every CallbackHandler via TraceContext, which stitches them back together.
"""

import os
from contextlib import contextmanager
from typing import Any, Optional

from dotenv import load_dotenv

load_dotenv()

_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
_PUBLIC = os.getenv("LANGFUSE_PUBLIC_KEY")
_SECRET = os.getenv("LANGFUSE_SECRET_KEY")

_client: Any = None
_initialised = False

# One trace per run, keyed by run id. A single module-level trace id worked only
# while one plan ran at a time; with workers, two plans' spans would land in one
# trace. The run id is read from the same contextvar `metrics` binds at node
# entry, so this needs no extra plumbing through the agents.
_trace_ids: dict[str, str] = {}


def _run_id() -> Optional[str]:
    from providers.metrics import _CURRENT_RUN

    return _CURRENT_RUN.get()


def _trace_id() -> Optional[str]:
    run_id = _run_id()
    return _trace_ids.get(run_id) if run_id else None


def enabled() -> bool:
    return bool(_PUBLIC and _SECRET)


def client() -> Any:
    """Lazily built Langfuse client, or None when unconfigured."""
    global _client, _initialised
    if _initialised:
        return _client
    _initialised = True
    if not enabled():
        return None
    try:
        from langfuse import Langfuse

        _client = Langfuse(public_key=_PUBLIC, secret_key=_SECRET, host=_HOST)
    except Exception:  # noqa: BLE001 - observability must never break the app
        _client = None
    return _client


def new_trace_id() -> Optional[str]:
    c = client()
    if c is None:
        return None
    try:
        return c.create_trace_id()
    except Exception:  # noqa: BLE001
        return None


def callbacks(agent: str) -> list:
    """LangChain callbacks for one agent's LLM calls.

    Returns [] when tracing is off, so callers need no conditional.
    """
    c = client()
    if c is None:
        return []
    try:
        from langfuse.langchain import CallbackHandler
        from langfuse.types import TraceContext

        trace_id = _trace_id()
        if trace_id:
            return [CallbackHandler(trace_context=TraceContext(trace_id=trace_id))]
        return [CallbackHandler()]
    except Exception:  # noqa: BLE001
        return []


@contextmanager
def trace_run(name: str, **metadata: Any):
    """Wrap a whole plan run in one trace. No-op when unconfigured."""
    c = client()
    if c is None:
        yield None
        return

    run_id = _run_id()
    trace_id = new_trace_id()
    if run_id and trace_id:
        _trace_ids[run_id] = trace_id
    span = None
    try:
        from langfuse.types import TraceContext

        span = c.start_observation(
            name=name,
            as_type="chain",
            input=metadata,
            trace_context=(TraceContext(trace_id=trace_id) if trace_id else None),
        )
        yield span
    except Exception:  # noqa: BLE001 - never let tracing break a run
        yield None
    finally:
        try:
            if span is not None:
                span.end()
            c.flush()
        except Exception:  # noqa: BLE001
            pass
        if run_id:
            # dropped, or a long-lived worker accumulates one entry per plan
            _trace_ids.pop(run_id, None)


@contextmanager
def trace_agent(agent: str, **metadata: Any):
    """Wrap one agent in a span nested under the current run's trace."""
    c = client()
    if c is None:
        yield None
        return

    span = None
    try:
        from langfuse.types import TraceContext

        span = c.start_observation(
            name=f"agent:{agent}",
            as_type="agent",
            input=metadata,
            trace_context=(
                TraceContext(trace_id=_trace_id()) if _trace_id() else None
            ),
        )
        yield span
    except Exception:  # noqa: BLE001
        yield None
    finally:
        try:
            if span is not None:
                span.end()
        except Exception:  # noqa: BLE001
            pass


def flush() -> None:
    c = client()
    if c is not None:
        try:
            c.flush()
        except Exception:  # noqa: BLE001
            pass
