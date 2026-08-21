"""Key routing, rotation and retry.

Groq free enforces 8k tokens/min, 1000 req/day and 200k tokens/DAY per key.
The daily ceiling appears in no header, so rotation is the only recovery: a
key that has spent its day is still spent after any sleep.
"""

import pytest

from providers import llm


@pytest.fixture(autouse=True)
def keys(monkeypatch):
    for i, name in enumerate(["GROQ_API_KEY", "GROQ_API_KEY_2", "GROQ_API_KEY_3"]):
        monkeypatch.setenv(name, f"key{i + 1}")


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    """Backoff must not make the suite slow."""
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)


def test_every_agent_has_a_chain():
    for agent in ("flight", "hotels", "weather", "itinerary"):
        assert len(llm.profiles_for(agent)) >= 2, "need somewhere to rotate to"


def test_concurrent_agents_have_distinct_primaries():
    """flight, hotels and weather run at the same time in the fan-out.

    Sharing a primary key would put their tokens in one 60s window.
    """
    fanout = ["flight", "weather"]
    primaries = [llm.profiles_for(a)[0] for a in fanout]
    assert len(set(primaries)) == len(primaries)


def test_itinerary_primary_is_unused_by_fanout():
    """itinerary is the heaviest agent and starts the moment the join happens."""
    fanout = {llm.profiles_for(a)[0] for a in ("flight", "hotels", "weather")}
    assert llm.profiles_for("itinerary")[0] not in fanout


def test_unknown_agent_rejected():
    with pytest.raises(ValueError):
        llm.profiles_for("nope")


def test_attempt_index_walks_then_wraps(monkeypatch):
    seen = []

    class FakeChat:
        def __init__(self, **kw):
            seen.append(kw["api_key"])

    monkeypatch.setattr("langchain_openai.ChatOpenAI", FakeChat)
    chain = llm.profiles_for("weather")
    for i in range(len(chain) + 1):
        llm.get_llm("weather", 0.1, i)
    assert len(set(seen[: len(chain)])) == len(chain), "each attempt a different key"
    assert seen[len(chain)] == seen[0], "then wraps around"


def test_missing_key_raises_clearly(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY_2", raising=False)
    idx = llm.profiles_for("flight").index("groq-k2")
    with pytest.raises(llm.MissingKeyError, match="GROQ_API_KEY_2"):
        llm.get_llm("flight", 0.1, idx)


# --- invoke_with_retry ---


class Boom(Exception):
    pass


def _rate_limit(msg="Error code: 429 rate_limit_exceeded tokens per day (TPD)"):
    return Boom(msg)


def test_retry_rotates_keys_before_sleeping(monkeypatch):
    attempts = []

    def fake_get_llm(agent, temp, attempt=0, timeout=None):
        attempts.append(attempt)

        class L:
            def bind_tools(self, t):
                return self

            def invoke(self, m):
                if len(attempts) < 3:
                    raise _rate_limit()
                return "ok"

        return L()

    slept = []
    monkeypatch.setattr(llm, "get_llm", fake_get_llm)
    monkeypatch.setattr(llm.time, "sleep", lambda s: slept.append(s))

    assert llm.invoke_with_retry("weather", [], 0.1) == "ok"
    assert attempts == [0, 1, 2], "walked the chain"
    assert slept == [], "3 keys, 3 attempts: no sleep needed yet"


def test_retry_sleeps_only_after_chain_exhausted(monkeypatch):
    def fake_get_llm(agent, temp, attempt=0, timeout=None):
        class L:
            def bind_tools(self, t):
                return self

            def invoke(self, m):
                raise _rate_limit()

        return L()

    slept = []
    monkeypatch.setattr(llm, "get_llm", fake_get_llm)
    monkeypatch.setattr(llm.time, "sleep", lambda s: slept.append(s))

    with pytest.raises(Boom):
        llm.invoke_with_retry("weather", [], 0.1)
    assert slept, "should have backed off once every key was tried"


def test_non_rate_limit_error_is_not_retried(monkeypatch):
    calls = []

    def fake_get_llm(agent, temp, attempt=0, timeout=None):
        class L:
            def bind_tools(self, t):
                return self

            def invoke(self, m):
                calls.append(1)
                raise Boom("401 Invalid API Key")

        return L()

    monkeypatch.setattr(llm, "get_llm", fake_get_llm)
    with pytest.raises(Boom, match="401"):
        llm.invoke_with_retry("weather", [], 0.1)
    assert len(calls) == 1, "an auth failure can never succeed on retry"


@pytest.mark.parametrize(
    "message,expected",
    [
        # each provider words the wait differently; all three must parse
        ("429 ... 'retry_after_seconds': 7 ...", 7.5),          # openrouter
        ("Please try again in 13.74s. Need more tokens?", 14.24),  # groq prose
        ("'Retry-After': '5'", 5.5),                            # http header
    ],
)
def test_retry_after_hint_is_honoured(message, expected):
    """Regression: the original pattern matched none of these, so the
    provider's own hint was always discarded for generic backoff."""
    assert llm._retry_after(Boom(message), 0) == pytest.approx(expected)


def test_retry_after_is_capped():
    assert llm._retry_after(Boom("'Retry-After': '9999'"), 0) <= 30.0


def test_backoff_grows_without_a_hint():
    plain = Boom("429 rate limited")
    assert llm._retry_after(plain, 0) < llm._retry_after(plain, 2)


# --- invoke_structured ---


def test_structured_rotates_keys_on_rate_limit(monkeypatch):
    used = []

    def fake_get_llm(agent, temp, attempt=0, timeout=None):
        used.append(attempt)

        class L:
            def with_structured_output(self, schema, method):
                return self

            def invoke(self, m):
                if len(used) < 3:
                    raise _rate_limit()
                return {"ok": True}

        return L()

    monkeypatch.setattr(llm, "get_llm", fake_get_llm)
    assert llm.invoke_structured("weather", dict, []) == {"ok": True}
    assert used == [0, 1, 2]


def test_structured_falls_through_methods_on_tool_failure(monkeypatch):
    methods = []

    def fake_get_llm(agent, temp, attempt=0, timeout=None):
        class L:
            def with_structured_output(self, schema, method):
                methods.append(method)
                return self

            def invoke(self, m):
                if len(methods) < 3:
                    raise Boom("tool_use_failed: did not call a tool")
                return {"ok": True}

        return L()

    monkeypatch.setattr(llm, "get_llm", fake_get_llm)
    assert llm.invoke_structured("weather", dict, []) == {"ok": True}
    assert methods == ["function_calling", "function_calling", "json_schema"], (
        "retry on the same method, then switch to json_schema"
    )
