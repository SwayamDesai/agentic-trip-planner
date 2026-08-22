"""Common agent plumbing.

Two agent shapes, both honouring the same node contract (state in, a partial
state update touching one key out):

    run_agent       - single structured call, model knowledge only
    run_tool_agent  - gather real data via tools, then one structured call

`run_tool_agent` is deliberately two-phase. Binding tools and forcing
structured output in the same request fights itself: the model has to choose
between emitting a tool call and emitting the answer schema, and on these
free-tier models that reliably produces neither. Instead phase one lets it
call tools freely until it stops asking, then phase two re-sends the gathered
tool output and demands the schema with no tools attached.
"""

import json
import os
import sys
import time
from contextlib import contextmanager
from collections import Counter

from langchain_core.messages import ToolMessage

from models import TripRequest, TripState
from providers import metrics, prompts, tracing
from providers.safety import Fence
from tools.schemas import summarize_exception, tool_error
from providers.llm import (
    AGENT_PROVIDERS,
    DeadlineExceeded,
    invoke_structured,
    invoke_with_retry,
)

# Wall-clock origin for the whole process, so trace lines from concurrently
# running agents can be compared against a common zero.
_T0 = time.perf_counter()
TRACE = os.getenv("TRIP_TRACE", "1") != "0"

# Observers of agent progress. The CLI prints to stderr; the web API pushes to
# a queue so the browser can watch agents work in real time. Registered here so
# neither the agents nor the graph need to know a UI exists.
TRACE_LISTENERS: list = []


def _trace(name: str, event: str, started: float) -> None:
    """Log agent start/finish to stderr with a shared clock.

    Overlapping start/done windows are the evidence that the fan-out agents
    really run concurrently rather than just being wired to look that way.
    """
    at = time.perf_counter() - _T0
    for listener in TRACE_LISTENERS:
        try:
            listener({"agent": name, "event": event, "at": round(at, 2)})
        except Exception:  # noqa: BLE001 - a broken observer must not break a run
            pass
    if not TRACE:
        return
    took = f" ({time.perf_counter() - started:.1f}s)" if event != "start" else ""
    chain = AGENT_PROVIDERS.get(name) or ["?"]
    provider = chain[0] if isinstance(chain, list) else chain
    print(f"[{at:6.1f}s] {name:<9} {provider:<10} {event}{took}", file=sys.stderr)


def describe_request(req: TripRequest) -> str:
    lines = [
        f"Origin: {req.origin}",
        f"Destination: {req.destination}",
        f"Dates: {req.start_date} to {req.end_date}",
        f"Travelers: {req.travelers}",
    ]
    if req.budget_usd:
        lines.append(f"Total budget: ${req.budget_usd} USD")
    if req.preferences:
        lines.append(f"Preferences: {', '.join(req.preferences)}")
    return "\n".join(lines)


def bind(state: TripState) -> None:
    """Attach the calling thread to this run's metrics and trace.

    First line of every node. A fan-out agent runs in a thread LangGraph
    created, which inherits no context, so without this its recordings land
    nowhere and its spans start a separate trace.
    """
    metrics.bind_run(state)


@contextmanager
def agent_span(name: str, **metadata):
    """One Langfuse span per agent, whatever shape the agent is.

    Previously only `run_tool_agent` opened one, so half the agents were
    invisible in a trace — the tool-free ones (itinerary, budget) and the ones
    with their own node bodies (places) simply had no span, even though their
    generations were logged. Putting it here means the span is a property of
    being an agent rather than of using tools.

    Flushed on exit: spans are created in LangGraph's worker threads, and
    relying on a single flush at the end of the run lost some of them.

    Must be closed explicitly — every caller needs a `finally`. Entering a
    context manager and never exiting it appears to work, because CPython
    finalises the generator when the last reference drops and that runs the
    `finally` inside it. But an exception traceback keeps the frame alive, so
    closure slips to interpreter shutdown, by which point the exporter has
    stopped and the span is discarded. That is what made one agent go missing
    from a trace at random.
    """
    span = tracing.trace_agent(name, **metadata)
    span.__enter__()
    try:
        yield
    finally:
        span.__exit__(None, None, None)
        # flushed per agent: spans are created in LangGraph worker threads, and
        # a single flush at the end of the run dropped some of them
        tracing.flush()


def already_done(name: str, state: TripState) -> bool:
    """True if this agent already has a result from an earlier run.

    Checkpointing restores prior state, but a graph that completed with an
    error-isolated failure is "finished" as far as LangGraph is concerned, so
    resuming replays every node. This guard is what makes the rerun cheap:
    agents holding a result return immediately and spend no tokens, leaving
    only the ones that actually failed to run again.
    """
    if state.get(name) is not None:
        _trace(name, "skip (cached from earlier run)", time.perf_counter())
        metrics.record_outcome(name, "skipped", 0.0)
        return True
    return False


def run_agent(
    *,
    name: str,
    state: TripState,
    schema,
    system: str,
    user: str,
    temperature: float = 0.3,
    verify=None,
) -> TripState:
    """Invoke one agent and return a partial state update.

    On failure the agent's own key stays None and the reason lands in
    `errors`, so the orchestrator can still assemble a partial plan.

    `verify` is an optional result -> list[str] check, for agents that make a
    single call and so have no tool payloads to compare against; it reads what
    it needs from state instead.
    """
    bind(state)
    if already_done(name, state):
        return {}

    t0 = time.perf_counter()
    _trace(name, "start", t0)
    deadline = deadline_for(name)
    span = agent_span(name, task=user[:400], prompt=prompts.resolved(name))
    span.__enter__()
    try:
        result = invoke_structured(
            name,
            schema,
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature,
            deadline=deadline,
        )
        metrics.record_outcome(name, "done", time.perf_counter() - t0)
        warnings = list(verify(result)) if verify and result else []
        _trace(
            name,
            "done" + (f" ({len(warnings)} warning(s))" if warnings else ""),
            t0,
        )
        update = {name: result}
        if warnings:
            update["warnings"] = [f"{name}: {w}" for w in warnings]
        return update
    except DeadlineExceeded as exc:
        _trace(name, "TIMEOUT", t0)
        metrics.record_outcome(name, "timeout", time.perf_counter() - t0)
        return {
            name: None,
            "errors": [
                f"{name}: timed out after {timeout_for(name):.0f}s ({exc}). "
                f"Re-run to retry just this agent."
            ],
        }
    except Exception as exc:  # noqa: BLE001 - one bad agent must not kill the run
        _trace(name, f"FAILED {type(exc).__name__}", t0)
        metrics.record_outcome(name, "failed", time.perf_counter() - t0)
        summary = summarize_exception(exc, name)
        return {name: None, "errors": [summary["error"]]}
    finally:
        span.__exit__(None, None, None)


# A model that loops is the main way an agent burns quota without progressing.
# Three independent stops, because they fail differently:
#   MAX_ROUNDS       the model keeps asking for more tools forever
#   MAX_TOTAL_CALLS  it asks for many DIFFERENT tools, never converging
#   MAX_REPEATS      it asks for the SAME call over and over, ignoring the answer
MAX_TOTAL_CALLS = 10
MAX_REPEATS = 2

# Per-agent wall-clock budget, in seconds. A node that overruns is failed and
# error-isolated like any other failure, so the rest of the plan survives.
#
# These are a safety net, not a target: they sit well above the slowest healthy
# run observed (itinerary peaked near 80s, flights near 105s under throttling).
# The point is to bound a node that has stopped making progress — a stalled
# provider, or a retry loop grinding against a spent daily quota — rather than
# let it hold the whole graph.
#
# Enforcement is twofold: a socket timeout on each request, plus a cooperative
# check between rounds. A Python thread cannot be killed from outside, so a
# node cannot be interrupted mid-call; bounding both the call and the loop is
# what keeps the overrun small.
DEFAULT_TIMEOUT = 120.0
AGENT_TIMEOUTS = {
    "flight": float(os.getenv("TIMEOUT_FLIGHT", 150)),
    "hotels": float(os.getenv("TIMEOUT_HOTELS", 150)),
    "weather": float(os.getenv("TIMEOUT_WEATHER", 120)),
    "itinerary": float(os.getenv("TIMEOUT_ITINERARY", 210)),
    # one small decision plus deterministic tool calls
    "places": float(os.getenv("TIMEOUT_PLACES", 120)),
    "budget": float(os.getenv("TIMEOUT_BUDGET", 90)),
    "scope": float(os.getenv("TIMEOUT_SCOPE", 45)),
    "chat": float(os.getenv("TIMEOUT_CHAT", 45)),
}


def timeout_for(name: str) -> float:
    return AGENT_TIMEOUTS.get(name, DEFAULT_TIMEOUT)


def deadline_for(name: str) -> float:
    """Monotonic instant by which this agent must be finished."""
    return time.monotonic() + timeout_for(name)


def _signature(call: dict) -> str:
    """Stable identity for a tool call, so repeats can be counted."""
    return json.dumps(
        {"name": call.get("name"), "args": call.get("args") or {}}, sort_keys=True
    )


def _gather_with_tools(
    name: str,
    tools: list,
    messages: list,
    max_rounds: int,
    temperature: float,
    deadline: float | None = None,
) -> tuple[list, list[dict]]:
    """Let the model call tools until it stops asking or hits a limit.

    Returns the extended message list and the payloads the tools produced, the
    latter so callers can verify the model's answer against its own sources.

    A tool that raises is reported back as a short, actionable error rather
    than aborting: the model can then adapt or report the gap honestly.
    """
    by_name = {t.name: t for t in tools}
    payloads: list[dict] = []
    repeats: Counter[str] = Counter()
    total_calls = 0

    for _ in range(max_rounds):
        # leave room for phase two: burning the whole budget on tool gathering
        # would produce data no one ever gets to format
        if deadline is not None and deadline - time.monotonic() < 15:
            _trace(name, "LIMIT time budget", time.perf_counter())
            break
        reply = invoke_with_retry(
            name, messages, temperature, tools=tools, deadline=deadline
        )
        messages.append(reply)

        calls = getattr(reply, "tool_calls", None) or []
        if not calls:
            break

        for call in calls:
            signature = _signature(call)
            repeats[signature] += 1

            if total_calls >= MAX_TOTAL_CALLS:
                payload = tool_error(
                    f"Tool call budget exhausted ({MAX_TOTAL_CALLS} calls). "
                    "Answer with the data you already have.",
                    "budget_exhausted",
                )
                _trace(name, "LIMIT total calls", time.perf_counter())
            elif repeats[signature] > MAX_REPEATS:
                # the model is re-asking a question it already has the answer to
                payload = tool_error(
                    f"You have already called {call['name']} with these exact "
                    "arguments. The result is above. Do not call it again; use "
                    "that result or proceed without it.",
                    "repeated_call",
                )
                _trace(name, f"LIMIT repeat {call['name']}", time.perf_counter())
            else:
                tool = by_name.get(call["name"])
                if tool is None:
                    payload = tool_error(
                        f"No tool named {call['name']!r}. Available: "
                        f"{', '.join(sorted(by_name))}.",
                        "unknown_tool",
                    )
                else:
                    try:
                        payload = tool.invoke(call["args"])
                        total_calls += 1
                        metrics.record_tool(name)
                        if isinstance(payload, dict):
                            payloads.append(payload)
                    except Exception as exc:  # noqa: BLE001 - summarised for the model
                        payload = summarize_exception(exc, call["name"])
                    _trace(name, f"tool {call['name']}", time.perf_counter())

            messages.append(
                ToolMessage(
                    content=json.dumps(payload, default=str)[:2500],
                    tool_call_id=call["id"],
                )
            )

    return messages, payloads


def _evidence(agent: str, payloads: list[dict]) -> list[dict]:
    """Project tool output down to what verification needs.

    Recorded per run so groundedness is checked against the payloads THIS run
    saw, not whatever happens to be in the cache afterwards. It also makes the
    weather source exact — previously it was inferred by string-matching
    condition text, which was a guess dressed as a measurement.
    """
    records: list[dict] = []
    for payload in payloads:
        if not isinstance(payload, dict) or payload.get("error"):
            continue

        if isinstance(payload.get("places"), list):
            records.append({
                "agent": agent,
                "kind": "places",
                "names": [
                    n for p in payload["places"]
                    for n in (p.get("name"), p.get("local_name"))
                    if n
                ],
            })
        if isinstance(payload.get("options"), list) and payload["options"]:
            first = payload["options"][0]
            if "price_per_night" in first:
                records.append({
                    "agent": agent,
                    "kind": "hotels",
                    "rows": [
                        {"name": o.get("name"), "price_per_night": o.get("price_per_night")}
                        for o in payload["options"]
                    ],
                })
            else:
                records.append({
                    "agent": agent,
                    "kind": "flights",
                    "rows": [
                        {"airline": o.get("airline"), "price_usd": o.get("price_usd")}
                        for o in payload["options"]
                    ],
                })
        if isinstance(payload.get("days"), list) and payload.get("source"):
            records.append({
                "agent": agent,
                "kind": "weather",
                "source": payload["source"],
                "rows": [
                    {"date": d.get("date"), "high_c": d.get("high_c")}
                    for d in payload["days"]
                ],
            })
    return records


def run_tool_agent(
    *,
    name: str,
    state: TripState,
    schema,
    system: str,
    user: str,
    tools: list,
    max_rounds: int = 4,
    temperature: float = 0.3,
    verify=None,
) -> TripState:
    """Gather real data with tools, then emit one validated `schema` object.

    `verify` is an optional (result, tool_payloads) -> list[str] check. It is
    how a prompt instruction becomes an enforced one: the model is asked to use
    only what the tools returned, and the verifier compares its answer against
    those same payloads. Findings are reported as warnings rather than
    corrections — a wrong auto-fix is worse than a visible caveat.

    Same failure contract as `run_agent`: the agent's key stays None and the
    reason lands in `errors`, so a dead tool or model still yields a partial
    plan rather than killing the graph.
    """
    bind(state)
    if already_done(name, state):
        return {}

    t0 = time.perf_counter()
    _trace(name, "start", t0)
    deadline = deadline_for(name)
    span = agent_span(name, task=user[:400], prompt=prompts.resolved(name))
    span.__enter__()
    try:
        messages, payloads = _gather_with_tools(
            name,
            tools,
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_rounds,
            temperature,
            deadline=deadline,
        )

        # Phase two: rebuild a MINIMAL prompt instead of replaying the chain.
        #
        # Re-sending every AIMessage and ToolMessage was pushing this past the
        # whole 8000 TPM budget in one request — a size no backoff can fix,
        # since the request can never fit in any minute. The model does not
        # need its own intermediate reasoning turns, only the data the tools
        # returned, so phase two keeps the system prompt, the original task,
        # and a plain digest of tool output.
        # Phase two inlines tool output into a USER message, which is where
        # data stops being distinguishable from instructions: OSM place names
        # and wiki summaries are world-editable, and an instruction hidden in
        # one reads exactly like the task. So each result is fenced with a
        # per-call nonce that injected text cannot guess, under a preamble
        # written here in code — deliberately not in the prompt, which can be
        # edited in the registry.
        #
        # Phase one keeps its results in ToolMessages, where the role itself
        # separates them, and their contents were already scrubbed at the tool
        # boundary.
        fence = Fence()
        digest = "\n\n".join(
            fence.wrap(f"tool result {i + 1}", str(m.content))
            for i, m in enumerate(m for m in messages if isinstance(m, ToolMessage))
        )
        final = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": (
                    f"{user}\n\n"
                    f"{fence.preamble()}\n\n"
                    f"{digest}\n\n"
                    "Produce the final structured answer using only the data "
                    "above. Do not invent entries the tools did not return."
                ),
            },
        ]
        result = invoke_structured(
            name, schema, final, temperature, deadline=deadline
        )
        metrics.record_outcome(name, "done", time.perf_counter() - t0)
        warnings = list(verify(result, payloads)) if verify and result else []
        _trace(
            name,
            f"done ({len(payloads)} tool calls"
            + (f", {len(warnings)} warning(s)" if warnings else "")
            + ")",
            t0,
        )
        update = {name: result}
        evidence = _evidence(name, payloads)
        if evidence:
            update["evidence"] = evidence
        if warnings:
            # non-destructive: surfaced in the plan, never silently corrected
            update["warnings"] = [f"{name}: {w}" for w in warnings]
        return update
    except DeadlineExceeded as exc:
        _trace(name, "TIMEOUT", t0)
        return {
            name: None,
            "errors": [
                f"{name}: timed out after {timeout_for(name):.0f}s ({exc}). "
                f"Re-run to retry just this agent."
            ],
        }
    except Exception as exc:  # noqa: BLE001 - one bad agent must not kill the run
        _trace(name, f"FAILED {type(exc).__name__}", t0)
        metrics.record_outcome(name, "failed", time.perf_counter() - t0)
        summary = summarize_exception(exc, name)
        return {name: None, "errors": [summary["error"]]}
    finally:
        span.__exit__(None, None, None)
