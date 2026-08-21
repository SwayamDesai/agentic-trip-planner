"""Per-run instrumentation.

Tier 2 of the eval plan: what a run actually cost. Tokens are the binding
constraint on this system (200k/day per Groq key), so "how many tokens did that
plan take" is a first-class question, not a curiosity.

Collected via a LangChain callback rather than by inspecting return values,
because `invoke_structured` returns a parsed Pydantic object and discards the
usage data attached to the raw response.

Threading: the fan-out runs agents in worker threads, so the collector is
guarded by a lock and keyed by agent name. One collector per process, reset at
the start of each run — concurrent runs in one process would interleave, which
is acceptable while plans are serialised (they must be anyway, for quota).
"""

import threading
import time
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
_CURRENT: RunMetrics = RunMetrics()


def reset() -> RunMetrics:
    """Begin a new run. Returns the fresh collector."""
    global _CURRENT
    with _LOCK:
        _CURRENT = RunMetrics()
        return _CURRENT


def current() -> RunMetrics:
    return _CURRENT


def record_llm(agent: str, prompt_tokens: int, completion_tokens: int) -> None:
    with _LOCK:
        m = _CURRENT.agents[agent]
        m.llm_calls += 1
        m.prompt_tokens += prompt_tokens
        m.completion_tokens += completion_tokens


def record_tool(agent: str) -> None:
    with _LOCK:
        _CURRENT.agents[agent].tool_calls += 1


def record_retry(agent: str) -> None:
    with _LOCK:
        _CURRENT.agents[agent].retries += 1


def record_outcome(agent: str, outcome: str, seconds: float) -> None:
    with _LOCK:
        m = _CURRENT.agents[agent]
        m.outcome = outcome
        m.seconds = seconds


def finish() -> RunMetrics:
    with _LOCK:
        _CURRENT.wall_seconds = time.perf_counter() - _CURRENT.started_at
        return _CURRENT


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
