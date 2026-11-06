"""Durable per-LLM-response records for tool grouping (TASK-2110).

One small row per (turn_id, response_id), upserted while the turn runs —
including responses that requested no tools, so call numbering stays stable.
Tool rows carry only ``response_id``; nothing here is a separate group entity.

Merge rules (idempotent under replay and duplicate delivery):

* ``host_parent`` and ``first_seen_seq`` are fixed by the first write.
* Usage counters are max-merged per field, never summed; a write that omits a
  field keeps the known value.
* ``usage_state`` only moves provisional -> final.

Display ordinals are derived on read from persisted first-seen order within
(turn_id, host_parent); they are never stored.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import Column, DateTime, Integer, String, UniqueConstraint
from sqlalchemy.exc import IntegrityError
from sqlmodel import Field, Session, SQLModel, select

from ..utils.schema import SchemaBootstrapGuard

logger = logging.getLogger(__name__)

PROVISIONAL = "provisional"
FINAL = "final"
# Read-only state: a provisional record whose turn has ended without a terminal
# signal (interruption, error, or a subagent response with no stream frames).
PARTIAL = "partial"

# Provider counter -> column. Anthropic semantics: input_tokens excludes cache.
_COUNTERS = {
    "input_tokens": "fresh_input_tokens",
    "cache_read_input_tokens": "cache_read_tokens",
    "cache_creation_input_tokens": "cache_write_tokens",
    "output_tokens": "output_tokens",
}


class LlmResponseRecord(SQLModel, table=True):
    __tablename__ = "llm_response_records"
    __table_args__ = (UniqueConstraint("turn_id", "response_id", name="uq_llm_response_turn_response"),)

    id: int | None = Field(default=None, sa_column=Column(Integer, primary_key=True, autoincrement=True))
    turn_id: str = Field(sa_column=Column(String(64), nullable=False, index=True))
    response_id: str = Field(sa_column=Column(String(128), nullable=False))
    bot_id: str | None = Field(default=None)
    user_id: str | None = Field(default=None)
    host_parent: str | None = Field(default=None, sa_column=Column(String(128), nullable=True))
    first_seen_seq: int = Field(sa_column=Column(Integer, nullable=False))
    model: str | None = Field(default=None, sa_column=Column(String(128), nullable=True))
    fresh_input_tokens: int | None = Field(default=None, sa_column=Column(Integer, nullable=True))
    cache_read_tokens: int | None = Field(default=None, sa_column=Column(Integer, nullable=True))
    cache_write_tokens: int | None = Field(default=None, sa_column=Column(Integer, nullable=True))
    output_tokens: int | None = Field(default=None, sa_column=Column(Integer, nullable=True))
    usage_state: str = Field(default=PROVISIONAL, sa_column=Column(String(16), nullable=False))
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


def _order_key(row: LlmResponseRecord) -> tuple:
    return (row.first_seen_seq, row.id or 0)


def serialize_responses(rows: list[LlmResponseRecord], *, turn_ended: bool) -> list[dict]:
    """Public shape for one turn's records, with ordinals per host parent."""
    ordinals: dict[str | None, int] = {}
    out: list[dict] = []
    for row in sorted(rows, key=_order_key):
        ordinal = ordinals.get(row.host_parent, 0) + 1
        ordinals[row.host_parent] = ordinal
        out.append(_public(row, ordinal=ordinal, turn_ended=turn_ended))
    return out


def _public(row: LlmResponseRecord, *, ordinal: int, turn_ended: bool) -> dict:
    state = row.usage_state
    if state != FINAL and turn_ended:
        state = PARTIAL
    usage: dict[str, int] = {}
    if row.fresh_input_tokens is not None:
        # Normalized input: cache read/write are sub-counts of it.
        usage["input_tokens"] = (
            row.fresh_input_tokens + (row.cache_read_tokens or 0) + (row.cache_write_tokens or 0)
        )
    if row.cache_read_tokens is not None:
        usage["cache_read_tokens"] = row.cache_read_tokens
    if row.cache_write_tokens is not None:
        usage["cache_write_tokens"] = row.cache_write_tokens
    # Snapshot output counts are preliminary; only a terminal frame makes them real.
    if row.output_tokens is not None and row.usage_state == FINAL:
        usage["output_tokens"] = row.output_tokens
    return {
        "response_id": row.response_id,
        "host_parent": row.host_parent,
        "ordinal": ordinal,
        "model": row.model,
        "usage": usage,
        "usage_state": state,
    }


class LlmResponseStore:
    _schema_guard = SchemaBootstrapGuard()

    def __init__(self, engine) -> None:
        self.engine = engine

    def ensure_schema(self) -> None:
        if self.engine is None:
            return

        def bootstrap(conn) -> None:
            SQLModel.metadata.create_all(bind=conn, tables=[LlmResponseRecord.__table__])

        self._schema_guard.run(self.engine, "llm-response-store", bootstrap)

    def upsert(
        self,
        *,
        turn_id: str | None,
        response_id: str | None,
        first_seen_seq: Any,
        host_parent: str | None = None,
        model: str | None = None,
        usage: dict | None = None,
        usage_state: str | None = None,
        bot_id: str | None = None,
        user_id: str | None = None,
    ) -> dict | None:
        """Merge one observation; return the record's public shape (live, not ended)."""
        if (
            self.engine is None or not turn_id or not isinstance(response_id, str)
            or not response_id or len(response_id) > 128 or not _count(first_seen_seq)
        ):
            return None
        for attempt in range(2):
            try:
                with Session(self.engine) as session:
                    row = self._get(session, turn_id, response_id)
                    if row is None:
                        row = LlmResponseRecord(
                            turn_id=turn_id, response_id=response_id, bot_id=bot_id,
                            user_id=user_id, host_parent=host_parent or None,
                            first_seen_seq=first_seen_seq,
                        )
                    self._merge(row, model=model, usage=usage, usage_state=usage_state)
                    session.add(row)
                    session.commit()
                    session.refresh(row)
                    return _public(row, ordinal=self._ordinal(session, row), turn_ended=False)
            except IntegrityError:
                # A concurrent first write won the unique key; merge into it.
                if attempt:
                    raise
        return None

    @staticmethod
    def _get(session: Session, turn_id: str, response_id: str) -> LlmResponseRecord | None:
        return session.exec(select(LlmResponseRecord).where(
            LlmResponseRecord.turn_id == turn_id,
            LlmResponseRecord.response_id == response_id,
        )).first()

    @staticmethod
    def _merge(row: LlmResponseRecord, *, model, usage, usage_state) -> None:
        if isinstance(model, str) and model and not row.model:
            row.model = model[:128]
        if isinstance(usage, dict):
            for key, column in _COUNTERS.items():
                value = usage.get(key)
                current = getattr(row, column)
                if _count(value) and (current is None or value > current):
                    setattr(row, column, value)
        if usage_state == FINAL:
            row.usage_state = FINAL
        row.updated_at = datetime.now(timezone.utc)

    @staticmethod
    def _ordinal(session: Session, row: LlmResponseRecord) -> int:
        peers = session.exec(select(LlmResponseRecord).where(
            LlmResponseRecord.turn_id == row.turn_id,
            LlmResponseRecord.host_parent == row.host_parent
            if row.host_parent is not None else LlmResponseRecord.host_parent.is_(None),
        )).all()
        key = _order_key(row)
        return 1 + sum(1 for peer in peers if _order_key(peer) < key)

    def for_turns(self, turn_ids: list[str]) -> dict[str, list[LlmResponseRecord]]:
        if self.engine is None or not turn_ids:
            return {}
        with Session(self.engine) as session:
            rows = session.exec(
                select(LlmResponseRecord).where(LlmResponseRecord.turn_id.in_(turn_ids))
            ).all()
        out: dict[str, list[LlmResponseRecord]] = {}
        for row in rows:
            out.setdefault(row.turn_id, []).append(row)
        return out
