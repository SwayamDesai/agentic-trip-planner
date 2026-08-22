"""A durable job queue, backed by SQL.

Why a table and not an in-memory list: the point of a queue is that it survives
the process. An in-memory queue loses everything on deploy, which is the exact
failure this replaces.

SQLite here so nothing new has to be installed while the pattern is learned.
The interface is deliberately narrow — `enqueue`, `claim`, `complete`, `fail`,
`append_event` — so moving to Postgres, and later to SQS or Redis, means
reimplementing five functions rather than rewriting callers.

The three semantics worth understanding, because every real queue has them:

**At-least-once delivery.** A worker can die after doing the work but before
reporting it, so a job may run twice. Queues that promise exactly-once are
either lying or very slow. The consequence for us is mild: `plan_trip` is
checkpointed, so a re-run skips agents that already succeeded.

**Lease / visibility timeout.** A claimed job is invisible to other workers for
`LEASE_SECONDS`. If the worker dies, the lease expires and the job returns to
the queue instead of being stuck in `running` forever. Workers extend their
lease with a heartbeat while working, which is why the lease can be much shorter
than the longest job.

**Attempt cap and dead letter.** A job that fails repeatedly stops being
retried and moves to `dead`. Without this a poison job — a malformed request, a
permanently missing API key — is retried forever and starves everything else.
"""

import json
import os
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

DB_PATH = Path(os.getenv("JOBS_DB", Path(__file__).parent / ".jobs.sqlite"))

# How long a claim is held before another worker may take the job. Shorter than
# a plan takes (60-150s) on purpose: workers heartbeat, so a live worker keeps
# its job, while a dead one releases it in under a minute rather than after the
# worst-case runtime.
LEASE_SECONDS = int(os.getenv("JOB_LEASE_SECONDS", 60))

MAX_ATTEMPTS = int(os.getenv("JOB_MAX_ATTEMPTS", 3))

QUEUED, RUNNING, DONE, FAILED, DEAD = "queued", "running", "done", "failed", "dead"


@dataclass
class Job:
    id: str
    status: str
    payload: dict
    result: Optional[dict]
    error: Optional[str]
    attempts: int
    created_at: float
    updated_at: float

    @property
    def terminal(self) -> bool:
        return self.status in (DONE, DEAD)

    def as_dict(self) -> dict:
        return {
            "job_id": self.id,
            "status": self.status,
            "attempts": self.attempts,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=10.0, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")   # the API reads while workers write
    conn.row_factory = sqlite3.Row
    return conn


def init() -> None:
    with _connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id          TEXT PRIMARY KEY,
                status      TEXT NOT NULL,
                payload     TEXT NOT NULL,
                result      TEXT,
                error       TEXT,
                attempts    INTEGER NOT NULL DEFAULT 0,
                lease_until REAL,
                worker_id   TEXT,
                created_at  REAL NOT NULL,
                updated_at  REAL NOT NULL
            );

            -- claim() filters on status and lease, so this is the index that
            -- keeps polling cheap as the table grows
            CREATE INDEX IF NOT EXISTS jobs_claimable
                ON jobs(status, lease_until);

            -- Progress events. Needed because the worker and the API are now
            -- SEPARATE PROCESSES: the API can no longer observe agent progress
            -- in memory, so the worker writes it here and the API reads it.
            -- This is the first real cost of moving work out of the request.
            CREATE TABLE IF NOT EXISTS job_events (
                id     INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL,
                at     REAL NOT NULL,
                kind   TEXT NOT NULL,
                data   TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS job_events_by_job ON job_events(job_id, id);
            """
        )


def enqueue(payload: dict) -> str:
    """Add a job and return its id. Returns in milliseconds."""
    init()
    job_id = uuid.uuid4().hex[:16]
    now = time.time()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO jobs(id, status, payload, created_at, updated_at) "
            "VALUES(?,?,?,?,?)",
            (job_id, QUEUED, json.dumps(payload), now, now),
        )
    return job_id


def claim(worker_id: str) -> Optional[Job]:
    """Take one job, or None if the queue is empty.

    Claimable means: queued, OR running with an expired lease — the second case
    is how a job recovers from a worker that died mid-run.

    The whole read-and-mark runs in one `BEGIN IMMEDIATE` transaction. Without
    it two workers polling at the same moment both read the same row and both
    run it.
    """
    init()
    now = time.time()
    conn = _connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            """
            SELECT * FROM jobs
             WHERE status = ?
                OR (status = ? AND lease_until IS NOT NULL AND lease_until < ?)
             ORDER BY created_at
             LIMIT 1
            """,
            (QUEUED, RUNNING, now),
        ).fetchone()

        if row is None:
            conn.execute("COMMIT")
            return None

        attempts = row["attempts"] + 1

        # a job that has burned its attempts is poison: stop retrying it
        if attempts > MAX_ATTEMPTS:
            conn.execute(
                "UPDATE jobs SET status=?, error=?, updated_at=? WHERE id=?",
                (DEAD, f"exceeded {MAX_ATTEMPTS} attempts", now, row["id"]),
            )
            conn.execute("COMMIT")
            return None

        conn.execute(
            "UPDATE jobs SET status=?, attempts=?, lease_until=?, worker_id=?, "
            "updated_at=? WHERE id=?",
            (RUNNING, attempts, now + LEASE_SECONDS, worker_id, now, row["id"]),
        )
        conn.execute("COMMIT")
    finally:
        conn.close()

    return Job(
        id=row["id"],
        status=RUNNING,
        payload=json.loads(row["payload"]),
        result=None,
        error=None,
        attempts=attempts,
        created_at=row["created_at"],
        updated_at=now,
    )


def heartbeat(job_id: str) -> None:
    """Extend the lease. Called periodically while working.

    This is what lets the lease be 60s while a job takes 150s: a live worker
    keeps pushing the deadline out, a dead one stops and the job is reclaimed.
    """
    now = time.time()
    with _connect() as conn:
        conn.execute(
            "UPDATE jobs SET lease_until=?, updated_at=? WHERE id=? AND status=?",
            (now + LEASE_SECONDS, now, job_id, RUNNING),
        )


def complete(job_id: str, result: dict) -> None:
    now = time.time()
    with _connect() as conn:
        conn.execute(
            "UPDATE jobs SET status=?, result=?, lease_until=NULL, updated_at=? "
            "WHERE id=?",
            (DONE, json.dumps(result, default=str), now, job_id),
        )


def fail(job_id: str, error: str) -> None:
    """Record a failure. Retried unless the attempt cap is spent.

    Left as `failed` rather than `queued` so the state is visible; `claim`
    treats an expired lease as claimable either way.
    """
    now = time.time()
    with _connect() as conn:
        row = conn.execute(
            "SELECT attempts FROM jobs WHERE id=?", (job_id,)
        ).fetchone()
        attempts = row["attempts"] if row else MAX_ATTEMPTS
        status = DEAD if attempts >= MAX_ATTEMPTS else QUEUED
        conn.execute(
            "UPDATE jobs SET status=?, error=?, lease_until=NULL, updated_at=? "
            "WHERE id=?",
            (status, error[:500], now, job_id),
        )


def get(job_id: str) -> Optional[Job]:
    init()
    with _connect() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    if row is None:
        return None
    return Job(
        id=row["id"],
        status=row["status"],
        payload=json.loads(row["payload"]),
        result=json.loads(row["result"]) if row["result"] else None,
        error=row["error"],
        attempts=row["attempts"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


# --- progress events -------------------------------------------------------


def append_event(job_id: str, kind: str, data: dict) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT INTO job_events(job_id, at, kind, data) VALUES(?,?,?,?)",
            (job_id, time.time(), kind, json.dumps(data, default=str)),
        )


def events_since(job_id: str, after_id: int = 0) -> list[dict]:
    """Events newer than `after_id`.

    Cursor-based rather than "all events": the API polls repeatedly, and
    re-sending the whole history each time would grow quadratically.
    """
    init()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, at, kind, data FROM job_events "
            "WHERE job_id=? AND id>? ORDER BY id",
            (job_id, after_id),
        ).fetchall()
    return [
        {"id": r["id"], "at": r["at"], "kind": r["kind"], "data": json.loads(r["data"])}
        for r in rows
    ]


# --- operations ------------------------------------------------------------


def stats() -> dict:
    """Queue depth by status — the number you alert on in production."""
    init()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT status, COUNT(*) AS n FROM jobs GROUP BY status"
        ).fetchall()
        oldest = conn.execute(
            "SELECT MIN(created_at) AS t FROM jobs WHERE status=?", (QUEUED,)
        ).fetchone()
    by_status = {r["status"]: r["n"] for r in rows}
    return {
        "by_status": by_status,
        "queued": by_status.get(QUEUED, 0),
        "running": by_status.get(RUNNING, 0),
        "dead": by_status.get(DEAD, 0),
        # how long the oldest waiting job has waited: the real backlog signal,
        # because depth alone cannot distinguish a burst from a stall
        "oldest_queued_age_seconds": (
            round(time.time() - oldest["t"], 1) if oldest and oldest["t"] else None
        ),
    }


def purge(older_than_seconds: float = 7 * 86_400) -> int:
    """Drop finished jobs and their events. Unbounded growth is a real outage."""
    cutoff = time.time() - older_than_seconds
    with _connect() as conn:
        ids = [
            r["id"]
            for r in conn.execute(
                "SELECT id FROM jobs WHERE status IN (?,?) AND updated_at < ?",
                (DONE, DEAD, cutoff),
            ).fetchall()
        ]
        for job_id in ids:
            conn.execute("DELETE FROM job_events WHERE job_id=?", (job_id,))
            conn.execute("DELETE FROM jobs WHERE id=?", (job_id,))
    return len(ids)
