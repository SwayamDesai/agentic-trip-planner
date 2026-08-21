"""Cross-run persistence for trip plans.

Two layers, doing different jobs:

`providers.cache` memoises *external API responses* — the same Overpass query
should not be re-sent. This module persists *agent results*, which are far more
expensive: every one cost LLM tokens against a metered free tier.

That matters because of how this system fails. Agents are error-isolated, so a
429 on one agent still produces a plan — just an incomplete one. Without
persistence the natural fix (run it again) re-spends tokens on the three agents
that already succeeded, making a partial failure cost more than a total one.

A LangGraph checkpointer keyed on the trip gives resume: rerunning the same
trip loads the prior state, and the node guards in `agents.base` skip agents
that already have a result. Only the agents that actually failed re-run.
"""

import hashlib
import os
import sqlite3
from pathlib import Path

from models import TripRequest

DB_PATH = Path(os.getenv("TRIP_DB", Path(__file__).parent.parent / ".trips.sqlite"))


def thread_id(req: TripRequest) -> str:
    """Stable id for a trip request.

    Derived from the request fields rather than a timestamp, so asking for the
    same trip twice resumes it instead of starting a parallel copy. Changing
    any field — dates, travellers, budget — is a different trip and gets its
    own thread.
    """
    parts = "|".join(
        [
            req.origin.strip().lower(),
            req.destination.strip().lower(),
            req.start_date,
            req.end_date,
            str(req.travelers),
            str(req.budget_usd or ""),
            ",".join(sorted(p.strip().lower() for p in req.preferences)),
        ]
    )
    return hashlib.sha256(parts.encode()).hexdigest()[:16]


def _allowed_models() -> list[tuple[str, str]]:
    """(module, class) pairs the checkpoint deserialiser may reconstruct.

    LangGraph warns on unregistered types today and will block them in a future
    version, so the state schema's models are declared. Enumerated from the
    module rather than hand-listed, so adding a model cannot silently break
    resume — the failure mode being a checkpoint that loads as plain dicts,
    which then fails on the first attribute access.

    An explicit allowlist rather than `True`: a checkpoint file is untrusted
    input the moment it moves between machines.
    """
    import models as models_module
    from pydantic import BaseModel

    return [
        ("models", name)
        for name, obj in vars(models_module).items()
        if isinstance(obj, type) and issubclass(obj, BaseModel) and obj is not BaseModel
    ]


def get_checkpointer() -> "object":
    """SqliteSaver over a file on disk.

    check_same_thread=False because the fan-out runs nodes in worker threads,
    and every one of them writes checkpoints.
    """
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
    from langgraph.checkpoint.sqlite import SqliteSaver

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)

    serde = JsonPlusSerializer(allowed_msgpack_modules=_allowed_models())
    return SqliteSaver(conn, serde=serde)


def forget(req: TripRequest) -> int:
    """Delete all saved state for one trip. Returns rows removed."""
    tid = thread_id(req)
    if not DB_PATH.exists():
        return 0
    conn = sqlite3.connect(str(DB_PATH))
    try:
        removed = 0
        for table in ("checkpoints", "writes"):
            try:
                cur = conn.execute(
                    f"DELETE FROM {table} WHERE thread_id = ?", (tid,)  # noqa: S608
                )
                removed += cur.rowcount or 0
            except sqlite3.OperationalError:
                pass  # table absent on a fresh db
        conn.commit()
        return removed
    finally:
        conn.close()


def list_trips() -> list[dict]:
    """Summarise saved trips, most recently written first."""
    if not DB_PATH.exists():
        return []
    conn = sqlite3.connect(str(DB_PATH))
    try:
        rows = conn.execute(
            "SELECT thread_id, COUNT(*) FROM checkpoints GROUP BY thread_id"
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()
    return [{"thread_id": t, "checkpoints": n} for t, n in rows]
