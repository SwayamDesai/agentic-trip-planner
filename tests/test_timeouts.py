"""Per-node time budgets.

A node that stops making progress — a stalled provider, or a retry loop
grinding against a spent daily quota — must not hold the whole graph. Each
agent gets a wall-clock budget and is failed like any other error when it
overruns, so the rest of the plan survives.
"""

import time

import pytest
from langchain_core.messages import AIMessage

from agents import base
from models import WeatherResult
from providers import llm


# --- budgets ---


def test_every_agent_has_a_budget():
    for agent in ("flight", "hotels", "weather", "itinerary", "budget", "scope", "chat"):
        assert base.timeout_for(agent) > 0


def test_unknown_agent_gets_the_default():
    assert base.timeout_for("mystery") == base.DEFAULT_TIMEOUT


def test_itinerary_has_the_largest_budget():
    """It is the heaviest agent: most tool calls, biggest payloads."""
    others = [base.timeout_for(a) for a in ("flight", "hotels", "weather", "budget")]
    assert base.timeout_for("itinerary") > max(others)


def test_interactive_paths_are_the_tightest():
    """A person is waiting on chat and scope, so they fail fast."""
    assert base.timeout_for("chat") < base.timeout_for("flight")
    assert base.timeout_for("scope") < base.timeout_for("itinerary")


def test_budgets_are_env_overridable(monkeypatch):
    monkeypatch.setenv("TIMEOUT_WEATHER", "7")
    import importlib

    reloaded = importlib.reload(base)
    try:
        assert reloaded.timeout_for("weather") == 7.0
    finally:
        monkeypatch.delenv("TIMEOUT_WEATHER", raising=False)
        importlib.reload(base)


def test_deadline_is_in_the_future():
    assert base.deadline_for("weather") > time.monotonic()


# --- remaining-time arithmetic ---


def test_remaining_is_none_when_untimed():
    assert llm._remaining(None) is None


def test_remaining_counts_down():
    left = llm._remaining(time.monotonic() + 5)
    assert 4 < left <= 5


def test_expired_deadline_raises():
    with pytest.raises(llm.DeadlineExceeded):
        llm._remaining(time.monotonic() - 1)


# --- the socket-level cap ---


def test_request_timeout_is_passed_to_the_client(monkeypatch):
    """Without a per-request cap, a stalled response blocks past the deadline
    and the cooperative check between rounds never runs."""
    monkeypatch.setenv("GROQ_API_KEY", "k")
    seen = {}

    class FakeChat:
        def __init__(self, **kw):
            seen.update(kw)

    monkeypatch.setattr("langchain_openai.ChatOpenAI", FakeChat)
    llm.get_llm("weather", 0.1, 0, timeout=42.5)
    assert seen["timeout"] == 42.5
    assert seen["max_retries"] == 0, "retries are ours, with key rotation"


# --- retry loops respect the deadline ---


def test_backoff_does_not_sleep_past_the_deadline(monkeypatch):
    """Waiting through the remaining budget achieves nothing but delay."""
    monkeypatch.setattr(llm, "profiles_for", lambda a: ["groq-k1"])
    monkeypatch.setattr(llm, "_retry_after", lambda exc, attempt: 30.0)

    slept = []
    monkeypatch.setattr(llm.time, "sleep", lambda s: slept.append(s))

    class L:
        def bind_tools(self, t):
            return self

        def invoke(self, m):
            raise RuntimeError("429 rate_limit_exceeded")

    monkeypatch.setattr(llm, "get_llm", lambda *a, **k: L())

    with pytest.raises(llm.DeadlineExceeded, match="not worth waiting"):
        llm.invoke_with_retry("weather", [], 0.1, deadline=time.monotonic() + 2)
    assert slept == [], "should refuse to wait rather than wait uselessly"


def test_structured_retry_also_refuses_a_pointless_wait(monkeypatch):
    monkeypatch.setattr(llm, "profiles_for", lambda a: ["groq-k1"])
    monkeypatch.setattr(llm, "_retry_after", lambda exc, attempt: 30.0)
    slept = []
    monkeypatch.setattr(llm.time, "sleep", lambda s: slept.append(s))

    class L:
        def with_structured_output(self, schema, method):
            return self

        def invoke(self, m):
            raise RuntimeError("429 rate_limit_exceeded")

    monkeypatch.setattr(llm, "get_llm", lambda *a, **k: L())
    with pytest.raises(llm.DeadlineExceeded):
        llm.invoke_structured("weather", dict, [], deadline=time.monotonic() + 2)
    assert slept == []


# --- the node contract on timeout ---


def test_timed_out_agent_fails_cleanly(monkeypatch):
    """Same contract as any failure: key stays None, reason lands in errors."""

    def boom(*a, **k):
        raise llm.DeadlineExceeded("no time left in the agent's budget")

    monkeypatch.setattr(base, "invoke_with_retry", lambda *a, **k: AIMessage(content=""))
    monkeypatch.setattr(base, "invoke_structured", boom)

    out = base.run_tool_agent(
        name="weather", state={}, schema=WeatherResult,
        system="s", user="u", tools=[],
    )
    assert out["weather"] is None
    message = out["errors"][0]
    assert "timed out after" in message
    assert "Re-run to retry just this agent" in message, "the fix is actionable"


def test_timeout_message_is_short(monkeypatch):
    def boom(*a, **k):
        raise llm.DeadlineExceeded("no time left")

    monkeypatch.setattr(base, "invoke_structured", boom)
    out = base.run_agent(
        name="weather", state={}, schema=WeatherResult, system="s", user="u"
    )
    assert len(out["errors"][0]) < 160


def test_gather_stops_when_the_budget_is_nearly_spent(monkeypatch):
    """Tool gathering must leave room for phase two, or it produces data that
    never gets formatted."""
    calls = []

    def fake(n, m, t, tools=None, **kw):
        calls.append(1)
        return AIMessage(content="")

    monkeypatch.setattr(base, "invoke_with_retry", fake)
    base._gather_with_tools(
        "weather", [], [], 4, 0.1, deadline=time.monotonic() + 5
    )
    assert calls == [], "under the phase-two reserve, so no round should start"


def test_gather_proceeds_with_ample_budget(monkeypatch):
    monkeypatch.setattr(
        base, "invoke_with_retry", lambda n, m, t, tools=None, **kw: AIMessage(content="")
    )
    _, payloads = base._gather_with_tools(
        "weather", [], [], 4, 0.1, deadline=time.monotonic() + 300
    )
    assert payloads == []


def test_a_required_agent_timing_out_fails_the_run():
    """Timeout is a failure, so the criticality tiers apply unchanged."""
    from status import plan_status

    state = {
        "request": None, "flight": None, "hotels": object(),
        "places": object(), "itinerary": object(),
        "weather": object(), "budget": object(),
    }
    assert plan_status(state)[0] == "failed"


def test_an_optional_agent_timing_out_only_degrades():
    from status import plan_status

    state = {
        "request": None, "flight": object(), "hotels": object(),
        "places": object(), "itinerary": object(),
        "weather": None, "budget": object(),
    }
    assert plan_status(state)[0] == "degraded"
