"""Recording and replaying LLM responses, so evals are free and reproducible.

Without this an eval run costs ~40k tokens and its inputs drift: prices change,
POI lists change, and two runs of the same case are not comparable. So a
regression in a prompt is indistinguishable from a change in the world.

Three modes, via `EVAL_MODE`:

    live     (default) call the provider, record nothing
    record   call the provider AND persist the response
    replay   serve from the recording; a miss is an error, never a silent
             fallthrough to a live call — a replay that quietly went live would
             be neither free nor reproducible, while looking like both

Keys are derived from everything that changes the answer: the agent, the schema,
the full message list and the temperature. A prompt edit therefore misses on
purpose, which is the signal you want when checking whether the edit helped.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Optional

from langchain_core.load import dumpd, load
from langchain_core.messages import BaseMessage

FIXTURES = Path(os.getenv("EVAL_FIXTURES", Path(__file__).parent.parent / "evals" / "fixtures"))


def mode() -> str:
    return os.getenv("EVAL_MODE", "live").strip().lower()


def active() -> bool:
    return mode() in {"record", "replay"}


class ReplayMiss(RuntimeError):
    """Replay was asked for a response that was never recorded."""


def _serialisable(value: Any) -> Any:
    """Render a message or Pydantic object as plain JSON-safe data."""
    if isinstance(value, BaseMessage):
        return {"__kind__": "message", "data": dumpd(value)}
    if hasattr(value, "model_dump"):
        return {"__kind__": "model", "data": value.model_dump()}
    return {"__kind__": "raw", "data": value}


def _restore(payload: dict, schema: Optional[type]) -> Any:
    kind = payload.get("__kind__")
    data = payload.get("data")
    if kind == "message":
        # 'messages' restricts deserialisation to chat message classes: fixtures
        # are project-generated, but a narrow allowlist costs nothing
        return load(data, allowed_objects="messages")
    if kind == "model":
        if schema is None:
            return data
        return schema.model_validate(data)
    return data


def key_for(agent: str, messages: Any, temperature: float, schema: Optional[type]) -> str:
    """Stable identity for one model call.

    Includes the schema name because the same prompt asked for two different
    shapes is two different calls.
    """
    payload = json.dumps(
        {
            "agent": agent,
            "schema": getattr(schema, "__name__", None),
            "temperature": round(float(temperature), 3),
            "messages": _messages_digest(messages),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


def _messages_digest(messages: Any) -> list:
    """Message content in a comparable form, whatever shape it arrives in."""
    out = []
    for m in messages or []:
        if isinstance(m, BaseMessage):
            out.append({"role": m.type, "content": str(m.content)})
        elif isinstance(m, dict):
            out.append({"role": m.get("role"), "content": str(m.get("content"))})
        else:
            out.append({"role": "?", "content": str(m)})
    return out


def _path(agent: str, key: str) -> Path:
    return FIXTURES / agent / f"{key}.json"


def lookup(agent: str, key: str, schema: Optional[type]) -> Any:
    """Recorded response for this call.

    Raises ReplayMiss in replay mode rather than returning None, so a missing
    fixture cannot silently become a live call.
    """
    path = _path(agent, key)
    if not path.exists():
        raise ReplayMiss(
            f"no recording for {agent}/{key}. Re-record with EVAL_MODE=record."
        )
    return _restore(json.loads(path.read_text()), schema)


def store(agent: str, key: str, value: Any) -> None:
    path = _path(agent, key)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_serialisable(value), indent=1, default=str))
    except (OSError, TypeError):
        pass  # a recording failure must not fail the run


def stats() -> dict:
    if not FIXTURES.exists():
        return {"mode": mode(), "agents": {}, "total": 0}
    by_agent = {
        d.name: len(list(d.glob("*.json"))) for d in FIXTURES.iterdir() if d.is_dir()
    }
    return {"mode": mode(), "agents": by_agent, "total": sum(by_agent.values())}
