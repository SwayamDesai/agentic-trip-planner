"""The two-phase agent loop: tool gathering, then forced structured output."""

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool

from agents import base
from models import WeatherResult


@tool
def ping(value: str) -> dict:
    """Echo a value back. USE WHEN testing."""
    return {"echo": value}


@tool
def explode(value: str) -> dict:
    """Always raises. USE WHEN testing failure handling."""
    raise RuntimeError("tool blew up")


def _ai(tool_calls=None):
    return AIMessage(content="", tool_calls=tool_calls or [])


def _call(name, args, cid):
    return {"name": name, "args": args, "id": cid, "type": "tool_call"}


# --- skip guard ---


def test_already_done_skips_completed_agent():
    """Regression basis for resume: a finished agent must cost nothing."""
    state = {"weather": WeatherResult(daily=[], packing_advice="x")}
    assert base.already_done("weather", state) is True


def test_missing_and_failed_agents_are_not_skipped():
    assert base.already_done("weather", {}) is False
    assert base.already_done("weather", {"weather": None}) is False, (
        "a failed agent wrote None; it MUST re-run on resume"
    )


def test_run_agent_returns_empty_update_when_done(monkeypatch):
    called = []
    monkeypatch.setattr(base, "invoke_structured", lambda *a, **k: called.append(1))
    out = base.run_agent(
        name="weather",
        state={"weather": WeatherResult(daily=[])},
        schema=WeatherResult,
        system="s",
        user="u",
    )
    assert out == {} and not called, "no LLM call, no state churn"


# --- tool gathering ---


def test_gather_executes_calls_and_pairs_ids(monkeypatch):
    replies = [_ai([_call("ping", {"value": "hi"}, "c1")]), _ai()]
    monkeypatch.setattr(
        base, "invoke_with_retry", lambda n, m, t, tools=None, **kw: replies.pop(0)
    )
    msgs, payloads = base._gather_with_tools("weather", [ping], [], 4, 0.1)
    assert payloads == [{"echo": "hi"}], "payloads are returned for verification"
    tool_msgs = [m for m in msgs if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 1
    assert tool_msgs[0].tool_call_id == "c1", "provider rejects a mismatched id"
    assert "hi" in tool_msgs[0].content


def test_gather_stops_when_model_asks_for_nothing(monkeypatch):
    calls = []

    def fake(n, m, t, tools=None, **kw):
        calls.append(1)
        return _ai()

    monkeypatch.setattr(base, "invoke_with_retry", fake)
    base._gather_with_tools("weather", [ping], [], 4, 0.1)
    assert len(calls) == 1


def test_gather_respects_max_rounds(monkeypatch):
    calls = []

    def fake(n, m, t, tools=None, **kw):
        calls.append(1)
        return _ai([_call("ping", {"value": "x"}, f"c{len(calls)}")])

    monkeypatch.setattr(base, "invoke_with_retry", fake)
    base._gather_with_tools("weather", [ping], [], 3, 0.1)
    assert len(calls) == 3, "a model that never stops must not loop forever"


def test_tool_exception_becomes_a_message_not_a_crash(monkeypatch):
    replies = [_ai([_call("explode", {"value": "x"}, "c1")]), _ai()]
    monkeypatch.setattr(
        base, "invoke_with_retry", lambda n, m, t, tools=None, **kw: replies.pop(0)
    )
    msgs, payloads = base._gather_with_tools("weather", [explode], [], 4, 0.1)
    import json as _json

    payload = _json.loads([m for m in msgs if isinstance(m, ToolMessage)][0].content)
    assert payload["error_kind"] == "unexpected"
    assert "explode" in payload["error"], "names the failing tool"
    assert payload["retryable"] is False
    assert "Do NOT retry" in payload["guidance"]
    assert payloads == [], "a raising tool contributes no payload"


def test_unknown_tool_reported_back(monkeypatch):
    replies = [_ai([_call("nonexistent", {}, "c1")]), _ai()]
    monkeypatch.setattr(
        base, "invoke_with_retry", lambda n, m, t, tools=None, **kw: replies.pop(0)
    )
    msgs, _ = base._gather_with_tools("weather", [ping], [], 4, 0.1)
    content = [m for m in msgs if isinstance(m, ToolMessage)][0].content
    assert "No tool named" in content
    assert "ping" in content, "lists what IS available"


def test_tool_output_is_truncated(monkeypatch):
    @tool
    def huge(value: str) -> dict:
        """Returns a lot. USE WHEN testing truncation."""
        return {"blob": "x" * 50_000}

    replies = [_ai([_call("huge", {"value": "x"}, "c1")]), _ai()]
    monkeypatch.setattr(
        base, "invoke_with_retry", lambda n, m, t, tools=None, **kw: replies.pop(0)
    )
    msgs, _ = base._gather_with_tools("weather", [huge], [], 4, 0.1)
    content = [m for m in msgs if isinstance(m, ToolMessage)][0].content
    assert len(content) <= 2500, "payloads re-enter context each round; must be capped"


# --- phase two ---


def test_phase_two_prompt_drops_model_turns_but_keeps_tool_data(monkeypatch):
    """Regression: replaying the whole chain pushed one request past the entire
    8000 tok/min budget — a size no backoff can fix."""
    replies = [_ai([_call("ping", {"value": "SENTINEL"}, "c1")]), _ai()]
    monkeypatch.setattr(
        base, "invoke_with_retry", lambda n, m, t, tools=None, **kw: replies.pop(0)
    )

    captured = {}

    def fake_structured(name, schema, messages, temperature, **kw):
        captured["messages"] = messages
        return WeatherResult(daily=[], packing_advice="ok")

    monkeypatch.setattr(base, "invoke_structured", fake_structured)

    base.run_tool_agent(
        name="weather",
        state={},
        schema=WeatherResult,
        system="SYSPROMPT",
        user="TASK",
        tools=[ping],
    )

    msgs = captured["messages"]
    assert len(msgs) == 2, "system + one user turn only"
    assert all(isinstance(m, dict) for m in msgs), "no replayed AIMessage objects"
    assert msgs[0]["content"] == "SYSPROMPT"
    body = msgs[1]["content"]
    assert "TASK" in body and "SENTINEL" in body, "task and tool data survive"


def test_agent_failure_is_isolated(monkeypatch):
    """One dead agent must still yield a partial plan."""
    monkeypatch.setattr(base, "invoke_with_retry", lambda *a, **k: _ai())

    def boom(*a, **k):
        raise RuntimeError("model down")

    monkeypatch.setattr(base, "invoke_structured", boom)
    out = base.run_tool_agent(
        name="weather", state={}, schema=WeatherResult,
        system="s", user="u", tools=[ping],
    )
    assert out["weather"] is None
    assert len(out["errors"]) == 1 and "model down" in out["errors"][0]


# --- loop limits ---


def test_identical_repeated_call_is_refused(monkeypatch):
    """A model re-asking a question it already has the answer to burns quota."""
    calls = []

    def fake(n, m, t, tools=None, **kw):
        calls.append(1)
        return _ai([_call("ping", {"value": "same"}, f"c{len(calls)}")])

    monkeypatch.setattr(base, "invoke_with_retry", fake)
    msgs, payloads = base._gather_with_tools("weather", [ping], [], 6, 0.1)

    contents = [m.content for m in msgs if isinstance(m, ToolMessage)]
    refusals = [c for c in contents if "already called" in c]
    assert refusals, "the repeat should be refused, not executed again"
    assert len(payloads) <= base.MAX_REPEATS, "tool ran at most MAX_REPEATS times"


def test_varying_args_are_not_treated_as_repeats(monkeypatch):
    """Different arguments are a different question."""
    n = {"i": 0}

    def fake(m_name, m, t, tools=None, **kw):
        n["i"] += 1
        return _ai([_call("ping", {"value": f"v{n['i']}"}, f"c{n['i']}")])

    monkeypatch.setattr(base, "invoke_with_retry", fake)
    _, payloads = base._gather_with_tools("weather", [ping], [], 4, 0.1)
    assert len(payloads) == 4, "all four distinct calls ran"


def test_total_call_budget_is_enforced(monkeypatch):
    """Guards against a model that keeps asking DIFFERENT things forever."""
    monkeypatch.setattr(base, "MAX_TOTAL_CALLS", 3)
    n = {"i": 0}

    def fake(m_name, m, t, tools=None, **kw):
        n["i"] += 1
        return _ai([_call("ping", {"value": f"v{n['i']}"}, f"c{n['i']}")])

    monkeypatch.setattr(base, "invoke_with_retry", fake)
    msgs, payloads = base._gather_with_tools("weather", [ping], [], 10, 0.1)
    assert len(payloads) == 3
    assert any(
        "budget exhausted" in m.content for m in msgs if isinstance(m, ToolMessage)
    )


def test_max_rounds_still_caps_the_loop(monkeypatch):
    calls = []

    def fake(n, m, t, tools=None, **kw):
        calls.append(1)
        return _ai([_call("ping", {"value": f"v{len(calls)}"}, f"c{len(calls)}")])

    monkeypatch.setattr(base, "invoke_with_retry", fake)
    base._gather_with_tools("weather", [ping], [], 3, 0.1)
    assert len(calls) == 3


# --- verification hook ---


def test_verifier_warnings_reach_state(monkeypatch):
    replies = [_ai([_call("ping", {"value": "x"}, "c1")]), _ai()]
    monkeypatch.setattr(
        base, "invoke_with_retry", lambda n, m, t, tools=None, **kw: replies.pop(0)
    )
    monkeypatch.setattr(
        base, "invoke_structured",
        lambda *a, **k: WeatherResult(daily=[], packing_advice="ok"),
    )

    out = base.run_tool_agent(
        name="weather", state={}, schema=WeatherResult, system="s", user="u",
        tools=[ping], verify=lambda result, payloads: ["something looks off"],
    )
    assert out["warnings"] == ["weather: something looks off"]
    assert out["weather"] is not None, "warnings do not discard the result"


def test_verifier_sees_the_tool_payloads(monkeypatch):
    replies = [_ai([_call("ping", {"value": "SENTINEL"}, "c1")]), _ai()]
    monkeypatch.setattr(
        base, "invoke_with_retry", lambda n, m, t, tools=None, **kw: replies.pop(0)
    )
    monkeypatch.setattr(
        base, "invoke_structured", lambda *a, **k: WeatherResult(daily=[])
    )
    seen = {}
    base.run_tool_agent(
        name="weather", state={}, schema=WeatherResult, system="s", user="u",
        tools=[ping], verify=lambda r, p: seen.update(payloads=p) or [],
    )
    assert seen["payloads"] == [{"echo": "SENTINEL"}]


def test_no_warnings_key_when_clean(monkeypatch):
    replies = [_ai()]
    monkeypatch.setattr(
        base, "invoke_with_retry", lambda n, m, t, tools=None, **kw: replies.pop(0)
    )
    monkeypatch.setattr(
        base, "invoke_structured", lambda *a, **k: WeatherResult(daily=[])
    )
    out = base.run_tool_agent(
        name="weather", state={}, schema=WeatherResult, system="s", user="u",
        tools=[ping], verify=lambda r, p: [],
    )
    assert "warnings" not in out


def test_agent_failure_message_is_summarised(monkeypatch):
    """Regression: the raw provider payload (a ~700-char 429 body) was being
    pasted into the plan. The reader needs the cause, not the JSON."""
    monkeypatch.setattr(base, "invoke_with_retry", lambda *a, **k: _ai())

    raw = (
        "Error code: 429 - {'error': {'message': 'Rate limit reached for model "
        "`openai/gpt-oss-120b` in organization `org_01kzy` service tier "
        "`on_demand` on tokens per day (TPD): Limit 200000, Used 197984, "
        "Requested 2312. Please try again in 2m7.872s. Need more tokens? "
        "Upgrade to Dev Tier today at https://console.groq.com/settings/billing'"
        ", 'type': 'tokens', 'code': 'rate_limit_exceeded'}}"
    )

    def boom(*a, **k):
        raise RuntimeError(raw)

    monkeypatch.setattr(base, "invoke_structured", boom)
    out = base.run_tool_agent(
        name="weather", state={}, schema=WeatherResult,
        system="s", user="u", tools=[ping],
    )
    message = out["errors"][0]
    assert len(message) < 160, f"still a firehose: {message}"
    assert "daily token quota" in message, "states the actual cause"
    assert "console.groq.com" not in message, "no raw payload"
