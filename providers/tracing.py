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

# One trace per plan run. Module-level rather than a contextvar because
# contextvars do not propagate into LangGraph's worker threads; the consequence
# is that concurrent plan runs in a single process would share a trace, which
# is acceptable while runs are serialised for quota reasons.
_current_trace_id: Optional[str] = None


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

        if _current_trace_id:
            return [CallbackHandler(trace_context=TraceContext(trace_id=_current_trace_id))]
        return [CallbackHandler()]
    except Exception:  # noqa: BLE001
        return []


@contextmanager
def trace_run(name: str, **metadata: Any):
    """Wrap a whole plan run in one trace. No-op when unconfigured."""
    global _current_trace_id
    c = client()
    if c is None:
        yield None
        return

    _current_trace_id = new_trace_id()
    span = None
    try:
        from langfuse.types import TraceContext

        span = c.start_observation(
            name=name,
            as_type="chain",
            input=metadata,
            trace_context=(
                TraceContext(trace_id=_current_trace_id) if _current_trace_id else None
            ),
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
        _current_trace_id = None


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
                TraceContext(trace_id=_current_trace_id) if _current_trace_id else None
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


def trace_url() -> Optional[str]:
    c = client()
    if c is None:
        return None
    try:
        return c.get_trace_url()
    except Exception:  # noqa: BLE001
        return None


def flush() -> None:
    c = client()
    if c is not None:
        try:
            c.flush()
        except Exception:  # noqa: BLE001
            pass
