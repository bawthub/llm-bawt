"""Bell notifications for opted-in operations (TASK-1183).

An operation with ``notify_bell`` on puts one entry in the user's BawtHub bell
when each of its jobs starts and one when it finishes. Jobs read the flag from
their frozen invocation snapshot, so an edit never changes a job already
admitted; a snapshot without the key is off.

Staging rides the single chokepoint for winning job transitions
(:func:`.events.append_job_transition`): the row is inserted in the job's own
transaction, inside a savepoint, so a lost CAS stages nothing and a staging
failure never fails the job write. Each ``{job}:{phase}`` key is unique, so a
retried transition cannot add a second entry. A start is staged only from
real evidence (a RUNNING transition, or ``started_at`` on a terminal job);
none is invented for a job that never ran.

Delivery (:func:`claim_due`/:func:`settle`) is a leased, restart-safe outbox:
the app's pump claims due rows, POSTs them to BawtHub (which upserts on the
same key), and records success or a backed-off retry. A finish waits for its
own job's start, so the bell never shows them out of order.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import Column, DateTime, Integer, String, and_, exists, or_, select, update
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.orm import aliased
from sqlmodel import Field, SQLModel

from ..service.commit_wake import notify_in_transaction
from .models import (JOB_CANCELLED, JOB_FAILED, JOB_LOST, JOB_RUNNING, JOB_SUCCEEDED,
                     JOB_TERMINAL_STATES, JOB_TIMED_OUT, OpsJob)

logger = logging.getLogger(__name__)

STARTED = "started"
FINISHED = "finished"

#: BawtHub notification severities for a finished job.
SEVERITY = {
    JOB_SUCCEEDED: "info",
    JOB_FAILED: "warning",
    JOB_TIMED_OUT: "warning",
    JOB_CANCELLED: "warning",
    JOB_LOST: "critical",
}

#: A claimed row is retried after this long if its sender never settled it.
LEASE_SECONDS = 120
MAX_BACKOFF_SECONDS = 600
#: Rows still undelivered after this long are given up (receiver gone for a week).
GIVE_UP_AFTER = timedelta(days=7)
_ERROR_LIMIT = 256


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class OpsBellOutbox(SQLModel, table=True):
    __tablename__ = "ops_bell_outbox"

    id: int | None = Field(default=None, sa_column=Column(Integer, primary_key=True, autoincrement=True))
    event_key: str = Field(sa_column=Column(String(160), nullable=False, unique=True))
    job_id: str = Field(sa_column=Column(String(64), nullable=False, index=True))
    phase: str = Field(sa_column=Column(String(16), nullable=False))
    created_at: datetime = Field(default_factory=_utcnow,
                                 sa_column=Column(DateTime(timezone=True), nullable=False))
    attempts: int = Field(default=0, sa_column=Column(Integer, nullable=False, server_default="0"))
    next_attempt_at: datetime = Field(default_factory=_utcnow,
                                      sa_column=Column(DateTime(timezone=True), nullable=False, index=True))
    delivered_at: datetime | None = Field(default=None, sa_column=Column(DateTime(timezone=True), nullable=True))
    gave_up_at: datetime | None = Field(default=None, sa_column=Column(DateTime(timezone=True), nullable=True))
    last_error: str | None = Field(default=None, sa_column=Column(String(_ERROR_LIMIT), nullable=True))


_TABLE = OpsBellOutbox.__table__


def wants_bell(job: OpsJob) -> bool:
    """Whether the job's frozen operation opted in. Unreadable or missing is off."""
    try:
        snapshot = json.loads(job.invocation_snapshot_json or "{}")
        operation = snapshot.get("operation") if isinstance(snapshot, dict) else None
        return isinstance(operation, dict) and operation.get("notify_bell") is True
    except (TypeError, ValueError):
        return False


def phases_for(job: OpsJob) -> tuple[str, ...]:
    """The bell phases this winning transition makes due, in delivery order."""
    if job.state == JOB_RUNNING:
        return (STARTED,)
    if job.state in JOB_TERMINAL_STATES:
        # Backfill a start only from evidence the job actually started.
        return ((STARTED,) if job.started_at is not None else ()) + (FINISHED,)
    return ()


def _insert(conn):
    dialect = postgresql if conn.dialect.name == "postgresql" else sqlite
    return dialect.insert(_TABLE)


def stage(conn, job: OpsJob | None) -> int:
    """Stage this transition's bell rows in ``conn``'s transaction. Never raises.

    Returns how many new rows were staged (0 on a repeat, opt-out or error).
    """
    if job is None:
        return 0
    try:
        phases = phases_for(job)
        if not phases or not wants_bell(job):
            return 0
        added = 0
        with conn.begin_nested():
            for phase in phases:
                stmt = _insert(conn).values(
                    event_key=f"{job.id}:{phase}", job_id=job.id, phase=phase,
                    created_at=_utcnow(), attempts=0, next_attempt_at=_utcnow(),
                ).on_conflict_do_nothing(index_elements=["event_key"])
                added += conn.execute(stmt).rowcount or 0
        if added:
            notify_in_transaction(conn)  # wake the app pump at commit
        return added
    except Exception:  # noqa: BLE001 - a bell never blocks its job transition
        logger.exception("ops bell staging failed for job %s", getattr(job, "id", "?"))
        return 0


# ---------------------------------------------------------------------------
# Delivery side (app pump)
# ---------------------------------------------------------------------------

def _pending():
    return and_(_TABLE.c.delivered_at.is_(None), _TABLE.c.gave_up_at.is_(None))


def claim_due(engine, *, limit: int = 20, now: datetime | None = None) -> list[dict[str, Any]]:
    """Lease up to ``limit`` due rows and return them with their job payloads.

    Rows older than :data:`GIVE_UP_AFTER` are given up first. A finish whose
    start is still pending is left for a later pass.
    """
    now = now or _utcnow()
    started = aliased(_TABLE)
    with engine.begin() as conn:
        conn.execute(update(_TABLE).where(_pending(), _TABLE.c.created_at < now - GIVE_UP_AFTER)
                     .values(gave_up_at=now))
        start_pending = exists().where(
            started.c.job_id == _TABLE.c.job_id, started.c.phase == STARTED,
            started.c.delivered_at.is_(None), started.c.gave_up_at.is_(None))
        query = (select(_TABLE).where(_pending(), _TABLE.c.next_attempt_at <= now,
                                      or_(_TABLE.c.phase == STARTED, ~start_pending))
                 .order_by(_TABLE.c.id).limit(limit))
        if conn.dialect.name == "postgresql":
            query = query.with_for_update(skip_locked=True, of=_TABLE)
        rows = [dict(r._mapping) for r in conn.execute(query)]
        if not rows:
            return []
        conn.execute(update(_TABLE).where(_TABLE.c.id.in_([r["id"] for r in rows]))
                     .values(next_attempt_at=now + timedelta(seconds=LEASE_SECONDS)))
        jobs = {j.id: j for j in conn.execute(select(OpsJob.__table__).where(
            OpsJob.__table__.c.id.in_({r["job_id"] for r in rows})))}
    claimed = []
    for row in rows:
        job = jobs.get(row["job_id"])
        if job is None:
            claimed.append({**row, "payload": None})
            continue
        claimed.append({**row, "payload": payload_for(job, row["phase"])})
    return claimed


def settle(engine, row_id: int, *, ok: bool, attempts: int, error: str | None = None,
           now: datetime | None = None) -> None:
    """Record a delivery result: delivered, or retry with exponential backoff."""
    now = now or _utcnow()
    with engine.begin() as conn:
        if ok:
            values = {"delivered_at": now, "last_error": None, "attempts": attempts + 1}
        else:
            backoff = min(MAX_BACKOFF_SECONDS, 5 * (2 ** min(attempts, 10)))
            values = {"attempts": attempts + 1, "next_attempt_at": now + timedelta(seconds=backoff),
                      "last_error": (error or "error")[:_ERROR_LIMIT]}
        conn.execute(update(_TABLE).where(_TABLE.c.id == row_id, _pending()).values(**values))


def give_up(engine, row_id: int, reason: str, *, now: datetime | None = None) -> None:
    """Stop retrying a row that can never be delivered (its job is gone)."""
    with engine.begin() as conn:
        conn.execute(update(_TABLE).where(_TABLE.c.id == row_id, _pending())
                     .values(gave_up_at=now or _utcnow(), last_error=reason[:_ERROR_LIMIT]))


def _operation(job) -> dict[str, Any]:
    try:
        snapshot = json.loads(job.invocation_snapshot_json or "{}")
        operation = snapshot.get("operation")
        return operation if isinstance(operation, dict) else {}
    except (TypeError, ValueError):
        return {}


def _duration(job) -> str | None:
    start, end = job.started_at, job.finished_at
    if start is None or end is None:
        return None
    seconds = max(0, int((end - start).total_seconds()))
    if seconds < 60:
        return f"{seconds} s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes} min {seconds} s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} h {minutes} min"


def payload_for(job, phase: str) -> dict[str, Any]:
    """The BawtHub entry for one phase. Never names the calling bot."""
    operation = _operation(job)
    slug = job.operation_slug
    title = operation.get("title") if isinstance(operation.get("title"), str) else None
    label = title or slug
    if phase == STARTED:
        # The start entry describes the start, even when delivered after the finish.
        state, severity, heading, parts = JOB_RUNNING, "info", f"Started: {label}", []
    else:
        state = job.state
        severity = SEVERITY.get(state, "warning")
        heading = f"{state.replace('_', ' ').capitalize()}: {label}"
        parts = []
        duration = _duration(job)
        if duration:
            parts.append(f"Took {duration}.")
        if job.exit_code is not None:
            parts.append(f"Exit code {job.exit_code}.")
        if job.error_text and state != JOB_SUCCEEDED:
            parts.append(job.error_text.strip()[:300])
    parts.append(f"Job {job.id}.")
    return {
        "jobId": job.id, "phase": phase, "userId": job.caller_user_id or None,
        "operation": slug, "title": heading[:200], "body": " ".join(parts)[:1000],
        "state": state, "severity": severity,
    }


__all__ = ["FINISHED", "GIVE_UP_AFTER", "OpsBellOutbox", "SEVERITY", "STARTED", "claim_due",
           "give_up", "payload_for", "phases_for", "settle", "stage", "wants_bell"]
