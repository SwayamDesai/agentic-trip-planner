"""Langfuse tracing. Entirely optional.

Every LLM call becomes a generation observation, and every agent becomes a span
under one trace per plan run, so a run can be inspected end to end: which agent
called which tool, what the prompt was, how many tokens it cost, and where it
failed.

Who logs what, now that LiteLLM sits in front:

    this module  the run trace and one span per agent — the STRUCTURE
    LiteLLM      one generation per model call — the COST

Both write into the SAME trace, because the app passes its trace id to the proxy
as request metadata. The split is deliberate: the proxy knows things the app
cannot, namely which deployment actually served the call and what it cost in
dollars, while the app knows things the proxy cannot, namely which agent asked
and why. Letting both log generations would double-count every call.

Two further constraints shaped this:

  1. It must no-op cleanly. Without Langfuse keys the whole module degrades to
     empty callback lists and a null context manager — the planner runs
     identically, and nothing in the agents knows tracing exists.

  2. The fan-out runs agents in worker threads. OpenTelemetry context does not
     cross a thread boundary on its own, so agent observations would each start
     their own trace. Fixed by minting one trace id per run and handing it to
     every CallbackHandler via TraceContext, which stitches them back together.
"""

import atexit
import os
import threading
from contextlib import contextmanager
from typing import Any, Optional

from dotenv import load_dotenv

load_dotenv()

_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
_PUBLIC = os.getenv("LANGFUSE_PUBLIC_KEY")
_SECRET = os.getenv("LANGFUSE_SECRET_KEY")

_client: Any = None
_initialised = False

# The fan-out builds its first client from four worker threads at once. Without
# this lock the flag below was set before the client existed, so whichever
# threads arrived during construction read `None` and silently opened no span —
# one agent went missing from the trace, and which one varied per run.
_client_lock = threading.Lock()

# One trace per run, keyed by run id. A single module-level trace id worked only
# while one plan ran at a time; with workers, two plans' spans would land in one
# trace. The run id is read from the same contextvar `metrics` binds at node
# entry, so this needs no extra plumbing through the agents.
_trace_ids: dict[str, str] = {}

# The name every writer puts on the trace. LiteLLM stamps a trace name on each
# generation it logs, and Langfuse takes the last write, so naming the trace
# from the app alone lost: every run showed up as `litellm-acompletion`. Both
# sides now send the same name, which makes the order stop mattering.
TRACE_NAME = "plan_trip"


def _run_id() -> Optional[str]:
    from providers.metrics import _CURRENT_RUN

    return _CURRENT_RUN.get()


def _trace_id() -> Optional[str]:
    run_id = _run_id()
    return _trace_ids.get(run_id) if run_id else None


def enabled() -> bool:
    return bool(_PUBLIC and _SECRET)


def request_metadata(agent: str) -> dict:
    """Metadata to attach to a proxied LLM call.

    LiteLLM reads these keys and files its generation under our trace, so the
    proxy's cost data lands beside the agent span that caused it rather than in
    a separate, unlinked trace.

    Returns {} when tracing is off, so the caller needs no conditional.
    """
    trace_id = _trace_id()
    if not enabled() or not trace_id:
        return {}
    return {
        "trace_id": trace_id,
        "trace_name": TRACE_NAME,
        "generation_name": f"agent:{agent}",
        "tags": ["atlas", f"agent:{agent}"],
    }


def client() -> Any:
    """Lazily built Langfuse client, or None when unconfigured."""
    global _client, _initialised
    if _initialised:
        return _client
    with _client_lock:
        # re-checked inside the lock: another thread may have built it while
        # this one waited
        if _initialised:
            return _client
        if enabled():
            try:
                from langfuse import Langfuse

                _client = Langfuse(
                    public_key=_PUBLIC, secret_key=_SECRET, host=_HOST
                )
                # Spans are exported in batches, so whatever is still queued when
                # the process ends is lost unless something drains it. A
                # long-lived uvicorn hides this; a short-lived one (a script, a
                # worker, `docker exec`) loses whichever agent happened to be in
                # the last batch — which looked like a random missing span.
                atexit.register(shutdown)
            except Exception:  # noqa: BLE001 - observability never breaks the app
                _client = None
        # set LAST, so no thread can observe the flag without the client
        _initialised = True
    return _client


def new_trace_id() -> Optional[str]:
    c = client()
    if c is None:
        return None
    try:
        return c.create_trace_id()
    except Exception:  # noqa: BLE001
        return None


def callbacks(agent: str, proxied: bool = False) -> list:
    """LangChain callbacks for one agent's LLM calls.

    Returns [] when tracing is off, so callers need no conditional.

    Also returns [] when the call is `proxied`: LiteLLM logs that generation
    itself, with the deployment and dollar cost attached. Logging it here as
    well would put two observations in the trace for one call — and the app's
    copy would be the less informative of the two.
    """
    c = client()
    if c is None or proxied:
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
    span = _start(c, name, "chain", metadata) if c is not None else None
    run_id = _run_id()
    if span is not None and run_id:
        _trace_ids[run_id] = span.trace_id
    if span is not None:
        # Name the TRACE, not just this observation. Both the app and LiteLLM
        # write into it, and without this the last writer wins — every run was
        # titled `litellm-acompletion`, which says nothing about the run.
        try:
            span.update_trace(name=TRACE_NAME, metadata=metadata or None)
        except Exception:  # noqa: BLE001
            pass

    try:
        yield span
    finally:
        _end(span)
        if c is not None:
            try:
                c.flush()
            except Exception:  # noqa: BLE001
                pass
        if run_id:
            _trace_ids.pop(run_id, None)


@contextmanager
def trace_agent(agent: str, **metadata: Any):
    """Wrap one agent in a span under the current run's trace."""
    c = client()
    span = _start(c, f"agent:{agent}", "agent", metadata) if c is not None else None
    try:
        yield span
    finally:
        _end(span)


def _start(c: Any, name: str, kind: str, metadata: dict) -> Any:
    """Open an observation, or return None if that fails.

    Separated from the context managers on purpose. Both of them previously
    wrapped their `yield` in a try/except, which made the generator yield a
    SECOND time whenever the wrapped body raised — Python answers that with
    "generator didn't stop after throw()". The failure only appeared once an
    agent actually raised inside a span.

    Now there is exactly one `yield` per context manager, and every way tracing
    can fail is handled here, before it.
    """
    try:
        from langfuse.types import TraceContext

        trace_id = _trace_id()
        return c.start_observation(
            name=name,
            as_type=kind,
            input=metadata or None,
            trace_context=TraceContext(trace_id=trace_id) if trace_id else None,
        )
    except Exception:  # noqa: BLE001 - observability must never break a run
        return None


def _end(span: Any) -> None:
    if span is None:
        return
    try:
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


def shutdown() -> None:
    """Drain and stop the exporter. Idempotent, and safe to call unconfigured.

    Reads `_client` directly rather than calling `client()`: at interpreter exit
    this must never build a client that did not already exist.
    """
    if _client is None:
        return
    try:
        _client.shutdown()
    except Exception:  # noqa: BLE001 - a failed drain must not raise at exit
        pass
