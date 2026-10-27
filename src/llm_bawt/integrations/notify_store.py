"""Audit log for ``notify_user`` that doubles as its rate and dedupe ledger (TASK-1531).

One row per call, whatever the outcome. Only ``sending``, ``sent`` and
``partial:*`` rows count toward the rate limits and dedupe: a refused,
suppressed or failed call
never reached the phone, so it must not block a retry. A ``sending`` row left
by a crash keeps counting, which errs toward fewer pushes, never more.

Admission (checks plus the ``sending`` reservation) is one transaction under
a Postgres advisory lock, so two bots can't both pass the global cap. HTTP to
Home Assistant happens outside it; :meth:`NotifyStore.finish` records the result.
"""
from __future__ import annotations

import hashlib
import threading
import time
import uuid
from dataclasses import dataclass

from sqlalchemy import Column, Float, Index, MetaData, String, Table, func, or_, select, text, update

from llm_bawt.utils.schema import SchemaBootstrapGuard

_metadata = MetaData()
_log = Table(
    "notify_log", _metadata,
    Column("id", String(32), primary_key=True),
    Column("bot_id", String(64), nullable=False),
    Column("created_at", Float, nullable=False),
    Column("updated_at", Float, nullable=False),
    Column("urgency", String(16), nullable=False),
    Column("message_hash", String(64), nullable=False),
    Column("dedupe_key", String(128)),
    Column("result", String(64), nullable=False),
    Index("ix_notify_log_bot_created", "bot_id", "created_at"),
    Index("ix_notify_log_created", "created_at"),
)
DELIVERING = ("sending", "sent")
_ADMIT_LOCK = "SELECT pg_advisory_xact_lock(hashtext('notify-user-admit'))"


@dataclass(frozen=True)
class Limits:
    per_bot_interval_seconds: int
    global_per_hour: int
    dedupe_window_seconds: int


def message_hash(message: str) -> str:
    return hashlib.sha256(message.encode()).hexdigest()


class NotifyStore:
    """SQL transactions only; never hold a transaction across network work."""

    _guard = SchemaBootstrapGuard()
    _local = threading.Lock()  # SQLite has no advisory lock; one process is enough there

    def __init__(self, engine):
        self.engine = engine
        self._guard.run(engine, "notify-log-v1", lambda conn: _metadata.create_all(conn))

    def record(self, bot_id: str, urgency: str, message: str, dedupe_key: str | None,
               result: str) -> str:
        """Write a terminal audit row for a call that never reached admission."""
        row_id = uuid.uuid4().hex
        now = time.time()
        with self.engine.begin() as conn:
            conn.execute(_log.insert().values(
                id=row_id, bot_id=bot_id, created_at=now, updated_at=now, urgency=urgency,
                message_hash=message_hash(message), dedupe_key=dedupe_key, result=result))
        return row_id

    def admit(self, bot_id: str, urgency: str, message: str, dedupe_key: str | None,
              limits: Limits, now: float | None = None) -> tuple[str, str]:
        """Check dedupe and both rate limits, then reserve. Returns ``(row id, result)``.

        ``result`` is ``sending`` (go ahead, then :meth:`finish`) or
        ``suppressed:<dedupe|rate_limit>``; the row is written either way.
        """
        now = time.time() if now is None else now
        row_id = uuid.uuid4().hex
        with self._local, self.engine.begin() as conn:
            if conn.dialect.name == "postgresql":
                conn.execute(text(_ADMIT_LOCK))
            result = self._suppression(conn, bot_id, dedupe_key, limits, now) or "sending"
            conn.execute(_log.insert().values(
                id=row_id, bot_id=bot_id, created_at=now, updated_at=now, urgency=urgency,
                message_hash=message_hash(message), dedupe_key=dedupe_key, result=result))
        return row_id, result

    @staticmethod
    def _suppression(conn, bot_id, dedupe_key, limits: Limits, now: float) -> str | None:
        delivering = or_(_log.c.result.in_(DELIVERING), _log.c.result.like("partial:%"))

        def count(*where) -> int:
            return conn.execute(select(func.count()).select_from(_log)
                                .where(delivering, *where)).scalar_one()

        if dedupe_key and count(_log.c.bot_id == bot_id, _log.c.dedupe_key == dedupe_key,
                                _log.c.created_at > now - limits.dedupe_window_seconds):
            return "suppressed:dedupe"
        if count(_log.c.bot_id == bot_id,
                 _log.c.created_at > now - limits.per_bot_interval_seconds):
            return "suppressed:rate_limit"
        if count(_log.c.created_at > now - 3600) >= limits.global_per_hour:
            return "suppressed:rate_limit"
        return None

    def finish(self, row_id: str, result: str) -> None:
        with self.engine.begin() as conn:
            conn.execute(update(_log).where(_log.c.id == row_id, _log.c.result == "sending")
                         .values(result=result, updated_at=time.time()))

    def get(self, row_id: str) -> dict | None:
        with self.engine.connect() as conn:
            row = conn.execute(select(_log).where(_log.c.id == row_id)).mappings().first()
        return dict(row) if row else None
