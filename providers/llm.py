"""LLM factory.

A *profile* is a (model, endpoint, key) triple. Agents are mapped to profiles
rather than to providers, so load can be spread across two models on the same
provider — which is what we actually want here.

Measured on this account, 3 structured-output calls against a real schema:

    groq  openai/gpt-oss-120b                     3/3 ok   ~1-2s each
    groq  qwen/qwen3.6-27b                        1/3 ok   ~4s each
    groq  openai/gpt-oss-20b                      0/3 ok   never emits the call
    or    nvidia/nemotron-3-super-120b-a12b:free  ok        1-128s, high variance
    or    z-ai/glm-5.2:free                       429s constantly (shared pool)

Two earlier splits were tried and rejected. Putting two agents on OpenRouter
made the graph *slower* than sequential — one leg stalled 128s and the fan-out
waits for its slowest branch. Splitting across two Groq models then failed on
reliability: qwen only produced valid structured output a third of the time.

So all four agents share one profile. That gives up per-model rate-limit
spreading, but gpt-oss-120b is the only backend found that is both fast and
reliably structured, and 4 calls per run sits well inside Groq's free limits.

OpenRouter profiles stay defined and tested as a spillover target if Groq's
per-minute limit ever binds — its free ceiling is a separate pool (~50/day).

Gemini is defined but unusable on this key: "prepayment credits are depleted",
and gemini-2.5-flash* is closed to new users.
"""

import os
import re
import time

from dotenv import load_dotenv

load_dotenv()

GROQ_URL = "https://api.groq.com/openai/v1"
OPENROUTER_URL = "https://openrouter.ai/api/v1"

# When set, LLM traffic goes through the LiteLLM proxy, which owns key routing,
# per-key cooldown and spend tracking. Unset (local dev, or the proxy being
# absent) and everything below behaves exactly as it did before.
LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL", "").strip()
LITELLM_MODEL = os.getenv("LITELLM_MODEL", "planner")

# name -> (model, base_url, env var holding the key)
#
# Groq's free tier caps TOKENS PER MINUTE (8000), not request count, and the
# cap is per organisation. Three separate Groq accounts therefore give three
# independent 8000 TPM pools — verified: each key reports its own
# x-ratelimit-remaining-tokens. That is what makes the parallel fan-out viable;
# on a single key three concurrent agents reliably 429 mid-run.
GROQ_MODEL = os.getenv("GROQ_MODEL_LARGE", "openai/gpt-oss-120b")

PROFILES = {
    "groq-k1": (GROQ_MODEL, GROQ_URL, "GROQ_API_KEY"),
    "groq-k2": (GROQ_MODEL, GROQ_URL, "GROQ_API_KEY_2"),
    "groq-k3": (GROQ_MODEL, GROQ_URL, "GROQ_API_KEY_3"),
    "openrouter": (
        os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3-super-120b-a12b:free"),
        OPENROUTER_URL,
        "OPENROUTER_API_KEY",
    ),
    # One alias; the proxy decides which of the three Groq keys serves it.
    "litellm": (LITELLM_MODEL, LITELLM_BASE_URL, "LITELLM_MASTER_KEY"),
}

# Each agent gets an ORDERED preference list of keys, not a single key.
#
# Groq free enforces three separate ceilings per key, and they bind in this
# order of surprise:
#   8000   tokens / minute   - visible in x-ratelimit headers
#   1000   requests / day    - visible in x-ratelimit headers
#   200000 tokens / DAY      - NOT in any header; only appears in the 429 body
#
# The TPD ceiling is what actually broke a pinned assignment: itinerary is the
# heaviest agent, so pinning it to k1 siloed it to 200k/day and it exhausted
# that key alone while k2 and k3 sat nearly untouched. Rotating on exhaustion
# pools all three into ~600k/day.
#
# First entry is the primary and is chosen so concurrent agents do not share a
# key; later entries are overflow, tried only when the primary is throttled.
AGENT_PROFILES = {
    "flight": ["groq-k2", "groq-k3", "groq-k1"],
    "hotels": ["groq-k2", "groq-k1", "groq-k3"],
    "weather": ["groq-k3", "groq-k1", "groq-k2"],
    "itinerary": ["groq-k1", "groq-k3", "groq-k2"],
    # budget runs after everything else, so every key is idle again by then;
    # k3 carried only the light weather agent this run.
    "budget": ["groq-k3", "groq-k2", "groq-k1"],
    # runs in the fan-out alongside flight/hotels/weather, so it needs a key
    # none of them hold as a primary
    "places": ["groq-k1", "groq-k3", "groq-k2"],
    # scope runs before the fan-out starts, so every pool is untouched
    "scope": ["groq-k1", "groq-k2", "groq-k3"],
    # conversation turns are small and interactive, so latency matters most
    "chat": ["groq-k2", "groq-k3", "groq-k1"],
}


def profiles_for(agent_name: str) -> list[str]:
    """Ordered preference for an agent, proxy first when one is configured.

    The direct-to-provider profiles stay in the chain BEHIND the proxy rather
    than being replaced by it. A gateway is one more thing that can be down, and
    an outage there should make the planner slower and unmetered — not broken.
    The existing rotation logic supplies the fallthrough for free.
    """
    chain = AGENT_PROFILES.get(agent_name)
    if chain is None:
        raise ValueError(f"no profile mapped for agent {agent_name!r}")
    direct = [chain] if isinstance(chain, str) else list(chain)
    return ["litellm", *direct] if LITELLM_BASE_URL else direct

# Kept for the trace output in agents.base, which labels each agent's backend.
AGENT_PROVIDERS = AGENT_PROFILES


class MissingKeyError(RuntimeError):
    pass


def get_llm(
    agent_name: str,
    temperature: float = 0.3,
    attempt: int = 0,
    timeout: float | None = None,
):
    """Build an LLM for `agent_name`, using the attempt-th key in its chain.

    `attempt` walks the preference list and then wraps, so a caller retrying
    after a 429 lands on a different key rather than the exhausted one.
    """
    chain = profiles_for(agent_name)
    profile = chain[attempt % len(chain)]
    model, base_url, key_name = PROFILES[profile]

    key = os.getenv(key_name)
    if not key:
        raise MissingKeyError(f"{key_name} not set in .env (needed for {profile})")

    from langchain_openai import ChatOpenAI

    from providers import tracing
    from providers.metrics import TokenCounter

    # Token counting is always on: tokens are the binding constraint, so a run
    # that cannot report its cost is not observable. Langfuse is additive and
    # contributes nothing when unconfigured.
    handlers = [TokenCounter(agent_name), *tracing.callbacks(agent_name)]

    return ChatOpenAI(
        model=model,
        api_key=key,
        base_url=base_url,
        temperature=temperature,
        # a socket-level cap: without it a stalled response blocks the node past
        # its deadline, and a cooperative check between rounds never gets to run
        timeout=timeout,
        max_retries=0,  # retries are handled here, with key rotation
        callbacks=handlers,
    )


# Groq raises this when the model answers in prose instead of emitting the
# forced tool call. It is intermittent on every model tested, so a single
# attempt is not enough.
_TOOL_FAILURE_MARKERS = (
    "tool_use_failed",
    "did not call a tool",
    "Failed to call a function",
)

# Because the fan-out fires several agents at once, they throttle each other.
# Lowercase, because _should_rotate lowercases the message before comparing.
# "RESOURCE_EXHAUSTED" spelled in caps here could never match, which a test
# caught. "tokens per day" earns its own entry: Groq's daily-quota message
# does not always carry a 429 in the text we see.
_RATE_LIMIT_MARKERS = (
    "429",
    "rate_limit",
    "rate-limited",
    "rate limit",
    "resource_exhausted",
    "tokens per day",
)

# Reasons to try the NEXT profile rather than give up. Rate limits were the
# original case; connection failures were added when the LiteLLM proxy went in
# front, because a proxy that is down raises a connection error and would
# otherwise fail the agent instead of falling through to the direct keys.
_CONNECTION_MARKERS = (
    "connection error",
    "connection refused",
    "connect timeout",
    "name or service not known",
    "temporary failure in name resolution",
    "apiconnectionerror",
    "max retries exceeded",
)


def _should_rotate(exc: Exception) -> bool:
    text = str(exc).lower() + " " + type(exc).__name__.lower()
    return any(m in text for m in _RATE_LIMIT_MARKERS) or any(
        m in text for m in _CONNECTION_MARKERS
    )


def _retry_after(exc: Exception, attempt: int) -> float:
    """Seconds to wait before retrying a throttled call.

    Providers state the wait in three different shapes, none of which the
    obvious pattern catches:

        Groq        "Please try again in 13.74s"
        OpenRouter  "'retry_after_seconds': 5"
        HTTP        "'Retry-After': '5'"

    Falls back to exponential backoff when none matches.
    """
    text = str(exc)
    for pattern in (
        r"try again in\s+([\d.]+)\s*s",          # groq prose
        r"retry[_-]after[_-]seconds['\"]?\s*[:=]\s*['\"]?([\d.]+)",
        r"['\"]?retry[-_]after['\"]?\s*[:=]\s*['\"]?([\d.]+)",
    ):
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return min(float(match.group(1)) + 0.5, 30.0)
    return min(2.0 * (2**attempt), 30.0)


class DeadlineExceeded(TimeoutError):
    """The agent ran out of its time budget."""


def _remaining(deadline: float | None) -> float | None:
    """Seconds left, or None when untimed. Raises once the budget is spent."""
    if deadline is None:
        return None
    left = deadline - time.monotonic()
    if left <= 0:
        raise DeadlineExceeded("no time left in the agent's budget")
    return left


def invoke_with_retry(
    agent_name: str,
    messages,
    temperature: float = 0.3,
    tools=None,
    deadline: float | None = None,
):
    """Plain (non-structured) invoke that rotates keys on rate limits.

    The tool-gathering phase needs the raw reply so it can read `tool_calls`,
    so it cannot go through `invoke_structured` — but it hits the same ceilings.
    Rotation matters more than backoff here: a key that has exhausted its
    200k/day budget will still be exhausted after any sleep, so the useful
    move is to switch keys, and only sleep once the chain is used up.
    """
    from providers import replay

    if replay.active():
        key = replay.key_for(agent_name, messages, temperature, None)
        if replay.mode() == "replay":
            return replay.lookup(agent_name, key, None)

    chain_len = len(profiles_for(agent_name))
    last_exc = None

    for attempt in range(chain_len * 2):
        left = _remaining(deadline)
        llm = get_llm(agent_name, temperature, attempt, timeout=left)
        if tools:
            llm = llm.bind_tools(tools)
        try:
            reply = llm.invoke(messages)
            if replay.mode() == "record":
                replay.store(agent_name, key, reply)
            return reply
        except Exception as exc:  # noqa: BLE001 - inspected, then re-raised
            if not _should_rotate(exc):
                raise
            # counted here too, not just in invoke_structured — otherwise a
            # rotation during tool gathering is invisible in the metrics that
            # claim to report retries
            from providers.metrics import record_retry

            record_retry(agent_name)
            last_exc = exc
            # sleep only after every key has been tried once, and never past
            # the deadline — waiting through it would be pure waste
            if attempt >= chain_len - 1:
                wait = _retry_after(exc, attempt - chain_len + 1)
                left = _remaining(deadline)
                if left is not None and wait >= left:
                    raise DeadlineExceeded(
                        f"rate limited with {left:.0f}s left; not worth waiting "
                        f"{wait:.0f}s"
                    ) from exc
                time.sleep(wait)

    raise last_exc


def invoke_structured(
    agent_name: str,
    schema,
    messages,
    temperature: float = 0.3,
    deadline: float | None = None,
):
    """Call the model until it returns a valid `schema` instance.

    Three failure modes show up on free tiers, needing different responses:

    tool_use_failed - the model wrote prose instead of calling the forced
        tool. Retried on the same key with a temperature nudge, then via
        json_schema mode; the two paths fail independently often enough that
        trying both materially raises the success rate.

    429 on a per-minute ceiling - waiting works.

    429 on the per-day ceiling - waiting does NOT work, the key is spent for
        the day. Both are handled by rotating to the next key first and only
        sleeping once the whole chain is exhausted.

    Anything else (missing key, unknown model) can never succeed on retry and
    is raised immediately.
    """
    from providers import replay

    if replay.active():
        key = replay.key_for(agent_name, messages, temperature, schema)
        if replay.mode() == "replay":
            return replay.lookup(agent_name, key, schema)

    methods = [
        ("function_calling", temperature),
        ("function_calling", min(temperature + 0.2, 1.0)),
        ("json_schema", temperature),
    ]
    chain_len = len(profiles_for(agent_name))
    last_exc: Exception | None = None
    key_attempt = 0
    throttled = 0

    i = 0
    while i < len(methods):
        method, temp = methods[i]
        try:
            left = _remaining(deadline)
            llm = get_llm(
                agent_name, temp, key_attempt, timeout=left
            ).with_structured_output(schema, method=method)
            result = llm.invoke(messages)
            if result is not None:
                if replay.mode() == "record":
                    replay.store(agent_name, key, result)
                return result
            last_exc = RuntimeError("model returned no structured output")
            i += 1
        except Exception as exc:  # noqa: BLE001 - inspected and possibly retried
            last_exc = exc
            text = str(exc)

            if _should_rotate(exc):
                from providers.metrics import record_retry

                record_retry(agent_name)
                key_attempt += 1
                # only start sleeping once every key in the chain has been tried
                if key_attempt >= chain_len:
                    if throttled >= 3:
                        raise
                    wait = _retry_after(exc, throttled)
                    left = _remaining(deadline)
                    if left is not None and wait >= left:
                        raise DeadlineExceeded(
                            f"rate limited with {left:.0f}s left; not worth "
                            f"waiting {wait:.0f}s"
                        ) from exc
                    time.sleep(wait)
                    throttled += 1
                continue  # same method, different key

            if not any(m in text for m in _TOOL_FAILURE_MARKERS):
                raise
            i += 1

    raise last_exc  # type: ignore[misc]
