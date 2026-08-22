"""Agent spans must be closed by the agent, not by the garbage collector.

Both helpers used to enter a context manager without a `finally`. That passed
every test and usually worked in production, because CPython finalises the
generator as soon as the last reference drops and that runs the `finally`
inside it. But anything holding the frame alive — an exception traceback, a
reference cycle — defers closure to interpreter shutdown, and by then the
Langfuse exporter has stopped, so the span is silently discarded.

The symptom was one agent missing from a trace at random. These tests pin the
close down by holding a reference to the context manager, which is exactly what
a retained traceback does.
"""

from contextlib import contextmanager

import pytest
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool

from agents import base
from models import WeatherResult


@tool
def ping(value: str) -> dict:
    """Echo a value back. USE WHEN testing."""
    return {"echo": value}


@pytest.fixture
def spans(monkeypatch):
    """Record span enter/exit, keeping the context managers alive.

    The `held` list is the point: without it, a missing `finally` is hidden by
    refcount finalisation and the test passes against the bug.
    """
    events: list[tuple[str, str]] = []
    held: list = []

    @contextmanager
    def fake_span(name, **_metadata):
        events.append(("enter", name))
        try:
            yield None
        finally:
            events.append(("exit", name))

    def tracked(name, **metadata):
        cm = fake_span(name, **metadata)
        held.append(cm)
        return cm

    monkeypatch.setattr(base.tracing, "trace_agent", tracked)
    monkeypatch.setattr(base.tracing, "flush", lambda: None)
    return events


def _weather():
    return WeatherResult(daily=[], packing_advice="layers")


def test_run_agent_closes_its_span_on_success(spans, monkeypatch):
    monkeypatch.setattr(base, "invoke_structured", lambda *a, **k: _weather())
    base.run_agent(
        name="weather",
        state={},
        schema=WeatherResult,
        system="s",
        user="u",
    )
    assert spans == [("enter", "weather"), ("exit", "weather")]


def test_run_agent_closes_its_span_when_the_model_fails(spans, monkeypatch):
    def boom(*_a, **_k):
        raise RuntimeError("provider down")

    monkeypatch.setattr(base, "invoke_structured", boom)
    out = base.run_agent(
        name="weather", state={}, schema=WeatherResult, system="s", user="u"
    )
    assert out["weather"] is None and out["errors"]
    assert ("exit", "weather") in spans


def test_run_agent_closes_its_span_on_timeout(spans, monkeypatch):
    def slow(*_a, **_k):
        raise base.DeadlineExceeded("out of time")

    monkeypatch.setattr(base, "invoke_structured", slow)
    base.run_agent(
        name="weather", state={}, schema=WeatherResult, system="s", user="u"
    )
    assert ("exit", "weather") in spans


def test_run_tool_agent_closes_its_span(spans, monkeypatch):
    def gather(*_a, **_k):
        return [ToolMessage(content='{"echo": "x"}', tool_call_id="1")], [
            {"tool": "ping", "output": {"echo": "x"}}
        ]

    monkeypatch.setattr(base, "_gather_with_tools", gather)
    monkeypatch.setattr(base, "invoke_structured", lambda *a, **k: _weather())
    base.run_tool_agent(
        name="weather",
        state={},
        schema=WeatherResult,
        system="s",
        user="u",
        tools=[ping],
    )
    assert spans == [("enter", "weather"), ("exit", "weather")]


def test_run_tool_agent_closes_its_span_when_gathering_fails(spans, monkeypatch):
    def boom(*_a, **_k):
        raise RuntimeError("tool provider down")

    monkeypatch.setattr(base, "_gather_with_tools", boom)
    out = base.run_tool_agent(
        name="weather",
        state={},
        schema=WeatherResult,
        system="s",
        user="u",
        tools=[ping],
    )
    assert out["weather"] is None
    assert ("exit", "weather") in spans


def test_skipped_agent_opens_no_span(spans):
    """A cached agent should not appear in the trace as a run of anything."""
    base.run_agent(
        name="weather",
        state={"weather": _weather()},
        schema=WeatherResult,
        system="s",
        user="u",
    )
    assert spans == []
