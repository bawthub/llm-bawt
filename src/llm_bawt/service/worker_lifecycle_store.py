"""Durable sub-agent (worker) lifecycle records (TASK-2169).

One row per (turn_id, task_id): the actual worker the bridge reported through
``subagent_started`` / ``subagent_progress`` / ``subagent_done``. Worker counts
come only from these rows, never from nested tool calls.

Merge rules (idempotent under replay and duplicate delivery):

* ``tool_use_id`` (the spawning host call) is fixed by the first write that
  knows it. Events that omit it (``task_updated`` status frames) resolve to the
  already-recorded worker by task id.
* Status only moves forward; a terminal status is never replaced by a
  non-terminal one.
* Usage is a sparse snapshot: monotonic counters are max-merged, the SDK
  ``total_tokens`` estimate follows the newest event, and
  ``cumulative_partial`` is sticky.
* Description, task type, provider and failure keep the first known value;
  summary and last tool follow the newest event that carries one.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import JSON, Column, DateTime, Float, Integer, String, Text, UniqueConstraint
from sqlalchemy.exc import IntegrityError
from sqlmodel import Field, Session, SQLModel, select

from ..utils.schema import SchemaBootstrapGuard

logger = logging.getLogger(__name__)

TERMINAL = frozenset({"completed", "failed", "stopped", "timed_out"})
_STATUS_RANK = {"started": 0, "running": 1}
_MAX_COUNTERS = ("tool_uses", "duration_ms", "cumulative_tokens")
_EVENTS = ("subagent_started", "subagent_progress", "subagent_done")


class WorkerLifecycleRecord(SQLModel, table=True):
    __tablename__ = "worker_lifecycle_records"
    __table_args__ = (UniqueConstraint("turn_id", "task_id", name="uq_worker_lifecycle_turn_task"),)

    id: int | None = Field(default=None, sa_column=Column(Integer, primary_key=True, autoincrement=True))
    turn_id: str = Field(sa_column=Column(String(64), nullable=False, index=True))
    task_id: str = Field(sa_column=Column(String(128), nullable=False))
    tool_use_id: str | None = Field(default=None, sa_column=Column(String(128), nullable=True))
    bot_id: str | None = Field(default=None)
    user_id: str | None = Field(default=None)
    description: str | None = Field(default=None, sa_column=Column(Text, nullable=True))
    task_type: str | None = Field(default=None, sa_column=Column(String(64), nullable=True))
    provider: str | None = Field(default=None, sa_column=Column(String(64), nullable=True))
    status: str = Field(default="started", sa_column=Column(String(16), nullable=False))
    summary: str | None = Field(default=None, sa_column=Column(Text, nullable=True))
    last_tool_name: str | None = Field(default=None, sa_column=Column(String(128), nullable=True))
    usage: dict | None = Field(default=None, sa_column=Column(JSON, nullable=True))
    failure: dict | None = Field(default=None, sa_column=Column(JSON, nullable=True))
    first_event_ts: float | None = Field(default=None, sa_column=Column(Float, nullable=True))
    last_event_ts: float | None = Field(default=None, sa_column=Column(Float, nullable=True))
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )


def _count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _text(value: Any, limit: int) -> str | None:
    return value[:limit] if isinstance(value, str) and value else None


def event_status(event_type: str, status: Any) -> str:
    """Normalize one lifecycle event to a worker status."""
    if isinstance(status, str) and status in TERMINAL:
        return status
    if event_type == "subagent_done":
        return "completed"
    if event_type == "subagent_started":
        return "started"
    return "running"


def _advance(current: str, incoming: str) -> str:
    if current in TERMINAL:
        return current
    if incoming in TERMINAL:
        return incoming
    return incoming if _STATUS_RANK.get(incoming, 0) > _STATUS_RANK.get(current, 0) else current


def _merge_usage(current: dict | None, incoming: Any, *, newest: bool) -> dict | None:
    if not isinstance(incoming, dict):
        return current
    usage = dict(current or {})
    for key in _MAX_COUNTERS:
        value = incoming.get(key)
        if _count(value) and value > usage.get(key, -1):
            usage[key] = value
    total = incoming.get("total_tokens")
    if _count(total) and (newest or "total_tokens" not in usage):
        usage["total_tokens"] = total
    if incoming.get("cumulative_partial") is True:
        usage["cumulative_partial"] = True
    return usage or current


def serialize_worker(row: WorkerLifecycleRecord) -> dict:
    return {
        "task_id": row.task_id,
        "tool_use_id": row.tool_use_id,
        "description": row.description or "",
        "task_type": row.task_type,
        "provider": row.provider,
        "status": row.status,
        "summary": row.summary,
        "last_tool_name": row.last_tool_name,
        "usage": row.usage,
        "failure": row.failure,
        "started_at": row.first_event_ts,
        "updated_at": row.last_event_ts,
    }


class WorkerLifecycleStore:
    _schema_guard = SchemaBootstrapGuard()

    def __init__(self, engine) -> None:
        self.engine = engine

    def ensure_schema(self) -> None:
        if self.engine is None:
            return

        def bootstrap(conn) -> None:
            SQLModel.metadata.create_all(bind=conn, tables=[WorkerLifecycleRecord.__table__])

        self._schema_guard.run(self.engine, "worker-lifecycle-store", bootstrap)

    def record(self, event: dict[str, Any]) -> dict | None:
        """Merge one lifecycle event; return the worker's public shape.

        Returns None when the event cannot be attributed to an authorized turn
        and task, or when a frame without ``tool_use_id`` names a worker this
        turn never started (nothing to attach it to).
        """
        event_type = event.get("_type")
        turn_id = event.get("turn_id")
        task_id = event.get("task_id")
        if (
            self.engine is None or event_type not in _EVENTS or not isinstance(turn_id, str)
            or not turn_id or not isinstance(task_id, str) or not task_id or len(task_id) > 128
        ):
            return None
        tool_use_id = _text(event.get("tool_use_id"), 128)
        ts = event.get("ts") if isinstance(event.get("ts"), (int, float)) else None
        for attempt in range(2):
            try:
                with Session(self.engine) as session:
                    row = session.exec(select(WorkerLifecycleRecord).where(
                        WorkerLifecycleRecord.turn_id == turn_id,
                        WorkerLifecycleRecord.task_id == task_id,
                    )).first()
                    if row is None:
                        if tool_use_id is None:
                            return None
                        row = WorkerLifecycleRecord(
                            turn_id=turn_id, task_id=task_id, tool_use_id=tool_use_id,
                            bot_id=event.get("bot_id"), user_id=event.get("user_id"),
                            first_event_ts=ts,
                        )
                    self._merge(row, event, event_type=event_type, tool_use_id=tool_use_id, ts=ts)
                    session.add(row)
                    session.commit()
                    session.refresh(row)
                    return serialize_worker(row)
            except IntegrityError:
                # A concurrent first write won the unique key; merge into it.
                if attempt:
                    raise
        return None

    @staticmethod
    def _merge(row, event, *, event_type, tool_use_id, ts) -> None:
        newest = ts is None or row.last_event_ts is None or ts >= row.last_event_ts
        if row.tool_use_id is None and tool_use_id:
            row.tool_use_id = tool_use_id
        for column, limit in (("description", 4000), ("task_type", 64), ("provider", 64)):
            value = _text(event.get(column), limit)
            if value and not getattr(row, column):
                setattr(row, column, value)
        row.status = _advance(row.status, event_status(event_type, event.get("status")))
        for column, limit in (("summary", 8000), ("last_tool_name", 128)):
            value = _text(event.get(column), limit)
            if value and (newest or not getattr(row, column)):
                setattr(row, column, value)
        row.usage = _merge_usage(row.usage, event.get("usage"), newest=newest)
        if isinstance(event.get("failure"), dict) and not row.failure:
            row.failure = event["failure"]
        if ts is not None:
            if row.first_event_ts is None or ts < row.first_event_ts:
                row.first_event_ts = ts
            if newest:
                row.last_event_ts = ts
        row.updated_at = datetime.now(timezone.utc)

    def for_turns(self, turn_ids: list[str]) -> dict[str, list[dict]]:
        if self.engine is None or not turn_ids:
            return {}
        with Session(self.engine) as session:
            rows = session.exec(
                select(WorkerLifecycleRecord).where(WorkerLifecycleRecord.turn_id.in_(turn_ids))
            ).all()
        out: dict[str, list[dict]] = {}
        for row in sorted(rows, key=lambda r: (r.first_event_ts or 0, r.id or 0)):
            out.setdefault(row.turn_id, []).append(serialize_worker(row))
        return out
