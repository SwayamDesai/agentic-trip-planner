"""Per-run instrumentation.

Tier 2 of the eval plan: what a run actually cost. Tokens are the binding
constraint on this system (200k/day per Groq key), so "how many tokens did that
plan take" is a first-class question, not a curiosity.

Collected via a LangChain callback rather than by inspecting return values,
because `invoke_structured` returns a parsed Pydantic object and discards the
usage data attached to the raw response.

Scoping: one collector PER RUN, not per process. A single global works only
while exactly one plan runs at a time; two concurrent plans would have the
second `reset()` discard the first plan's counters mid-flight. Measured before
this change: ten LLM calls across two plans, seven recorded.

The run id travels two ways, and it needs both:

  * in `TripState`, because LangGraph's fan-out runs each agent in a worker
    thread and a contextvar set in the parent does NOT cross that boundary
  * in a contextvar, bound at the top of each node from the state, so
    everything the node calls downstream — including the LLM callback several
    frames deep — can find its collector without threading an argument through
    every signature
"""

import contextvars
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

from langchain_core.callbacks import BaseCallbackHandler


@dataclass
class AgentMetrics:
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tool_calls: int = 0
    retries: int = 0
    seconds: float = 0.0
    outcome: str = "pending"  # done | skipped | failed | timeout

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class RunMetrics:
    started_at: float = field(default_factory=time.perf_counter)
    agents: dict[str, AgentMetrics] = field(default_factory=lambda: defaultdict(AgentMetrics))
    wall_seconds: float = 0.0

    def totals(self) -> dict:
        return {
            "llm_calls": sum(a.llm_calls for a in self.agents.values()),
            "prompt_tokens": sum(a.prompt_tokens for a in self.agents.values()),
            "completion_tokens": sum(a.completion_tokens for a in self.agents.values()),
            "total_tokens": sum(a.total_tokens for a in self.agents.values()),
            "tool_calls": sum(a.tool_calls for a in self.agents.values()),
            "retries": sum(a.retries for a in self.agents.values()),
            "wall_seconds": round(self.wall_seconds, 2),
        }

    def as_dict(self) -> dict:
        return {
            "totals": self.totals(),
            "agents": {
                name: {
                    "llm_calls": m.llm_calls,
                    "prompt_tokens": m.prompt_tokens,
                    "completion_tokens": m.completion_tokens,
                    "total_tokens": m.total_tokens,
                    "tool_calls": m.tool_calls,
                    "retries": m.retries,
                    "seconds": round(m.seconds, 2),
                    "outcome": m.outcome,
                }
                for name, m in sorted(self.agents.items())
            },
        }


_LOCK = threading.Lock()

# run_id -> collector. Replaces the single module-level collector.
_RUNS: dict[str, RunMetrics] = {}

# The run this thread is currently working on. Set by `bind_run` at node entry.
_CURRENT_RUN: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "trip_run_id", default=None
)


def new_run() -> str:
    """Start a run and return its id. The id goes into TripState."""
    run_id = uuid.uuid4().hex[:16]
    with _LOCK:
        _RUNS[run_id] = RunMetrics()
    _CURRENT_RUN.set(run_id)
    return run_id


def bind_run(state) -> None:
    """Attach this thread to the run described by `state`.

    Called at the top of every node. Without it a fan-out agent's thread has no
    idea which run it belongs to, because contextvars do not cross the thread
    boundary LangGraph creates.
    """
    run_id = (state or {}).get("run_id") if isinstance(state, dict) else None
    if run_id:
        _CURRENT_RUN.set(run_id)


def _collector() -> Optional[RunMetrics]:
    run_id = _CURRENT_RUN.get()
    if run_id is None:
        return None
    with _LOCK:
        return _RUNS.get(run_id)


def get(run_id: str) -> Optional[RunMetrics]:
    with _LOCK:
        return _RUNS.get(run_id)


def record_llm(agent: str, prompt_tokens: int, completion_tokens: int) -> None:
    run = _collector()
    if run is None:
        return          # untracked context (a test, a script): drop silently
    with _LOCK:
        m = run.agents[agent]
        m.llm_calls += 1
        m.prompt_tokens += prompt_tokens
        m.completion_tokens += completion_tokens


def record_tool(agent: str) -> None:
    run = _collector()
    if run is not None:
        with _LOCK:
            run.agents[agent].tool_calls += 1


def record_retry(agent: str) -> None:
    run = _collector()
    if run is not None:
        with _LOCK:
            run.agents[agent].retries += 1


def record_outcome(agent: str, outcome: str, seconds: float) -> None:
    run = _collector()
    if run is None:
        return
    with _LOCK:
        m = run.agents[agent]
        m.outcome = outcome
        m.seconds = seconds


def finish(run_id: str) -> Optional[RunMetrics]:
    """Close a run and drop it from the registry.

    Removal matters: a long-lived worker process would otherwise accumulate one
    collector per plan it has ever run.
    """
    with _LOCK:
        run = _RUNS.pop(run_id, None)
    if run is not None:
        run.wall_seconds = time.perf_counter() - run.started_at
    return run


def active_runs() -> int:
    with _LOCK:
        return len(_RUNS)


class TokenCounter(BaseCallbackHandler):
    """Captures token usage for one agent's LLM calls.

    Providers report usage in more than one shape depending on the client, so
    both the LLMResult summary and the per-message metadata are checked.
    """

    def __init__(self, agent: str):
        self.agent = agent

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:  # noqa: ANN401
        prompt = completion = 0

        usage = (getattr(response, "llm_output", None) or {}).get("token_usage") or {}
        prompt = usage.get("prompt_tokens") or 0
        completion = usage.get("completion_tokens") or 0

        if not (prompt or completion):
            # fall back to per-message metadata
            for batch in getattr(response, "generations", None) or []:
                for gen in batch:
                    meta = getattr(getattr(gen, "message", None), "usage_metadata", None)
                    if meta:
                        prompt += meta.get("input_tokens") or 0
                        completion += meta.get("output_tokens") or 0

        record_llm(self.agent, int(prompt), int(completion))
