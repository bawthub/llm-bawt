"""Provider-bound request bodies captured at proxy egress (TASK-502).

``turn_logs.request_json`` keeps what llm-bawt authored. For proxy bots the
in-process proxy then translates, adapts and (for ChatGPT Lite) reshapes the
request; this table keeps what actually left for the provider, one row per
upstream call attempt.

Scope and limits:

* Only proxy-routed bots are captured. Native Anthropic bots never pass
  through the proxy, and any preamble a provider adds on its side is never
  visible to us.
* Bodies only. Credentials travel in headers, which are never captured.
* One agent turn issues many upstream calls whose bodies mostly repeat the
  growing history. Every call keeps its metadata (size, hash, transport,
  attempt), but only the turn's first and latest bodies are retained: the
  first shows the opening system/tool preamble, the latest the full request.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import Column, DateTime, Integer, String, Text, UniqueConstraint
from sqlalchemy import text as sa_text
from sqlalchemy.exc import IntegrityError
from sqlmodel import Field, Session, SQLModel, select

from ..utils.schema import SchemaBootstrapGuard

logger = logging.getLogger(__name__)

# One body larger than this is recorded by size and hash only.
MAX_BODY_BYTES = 8 * 1024 * 1024


class EgressCaptureRecord(SQLModel, table=True):
    __tablename__ = "turn_egress_requests"
    __table_args__ = (
        UniqueConstraint("request_id", "seq", name="uq_turn_egress_request_seq"),
    )

    id: int | None = Field(default=None, sa_column=Column(Integer, primary_key=True, autoincrement=True))
    turn_id: str = Field(sa_column=Column(String(64), nullable=False, index=True))
    request_id: str = Field(sa_column=Column(String(128), nullable=False))
    seq: int = Field(sa_column=Column(Integer, nullable=False))
    attempt: int = Field(default=1, sa_column=Column(Integer, nullable=False))
    provider: str | None = Field(default=None, sa_column=Column(String(64), nullable=True))
    transport: str | None = Field(default=None, sa_column=Column(String(64), nullable=True))
    model: str | None = Field(default=None, sa_column=Column(String(128), nullable=True))
    body_bytes: int = Field(sa_column=Column(Integer, nullable=False))
    body_sha256: str = Field(sa_column=Column(String(64), nullable=False))
    # NULL once superseded (neither first nor latest) or when over the cap.
    body_json: str | None = Field(default=None, sa_column=Column(Text, nullable=True))
    body_state: str = Field(default="retained", sa_column=Column(String(16), nullable=False))
    captured_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False, index=True),
    )


def _model_of(body_json: str) -> str | None:
    import json

    try:
        body = json.loads(body_json)
    except ValueError:
        return None
    model = body.get("model") if isinstance(body, dict) else None
    return str(model)[:128] if model else None


class EgressCaptureStore:
    _schema_guard = SchemaBootstrapGuard()

    def __init__(self, engine) -> None:
        self.engine = engine

    def ensure_schema(self) -> None:
        if self.engine is None:
            return
        self._schema_guard.run(
            self.engine, "turn-egress-requests",
            lambda conn: SQLModel.metadata.create_all(
                bind=conn, tables=[EgressCaptureRecord.__table__],
            ),
        )

    def record(self, *, turn_id: str, capture: dict[str, Any]) -> bool:
        """Insert one capture; idempotent per (request_id, seq)."""
        body_json = capture["body_json"]
        encoded = body_json.encode("utf-8")
        over_cap = len(encoded) > MAX_BODY_BYTES
        row = EgressCaptureRecord(
            turn_id=turn_id,
            request_id=capture["request_id"],
            seq=int(capture["seq"]),
            attempt=int(capture.get("attempt") or 1),
            provider=(capture.get("provider") or None),
            transport=(capture.get("transport") or None),
            model=_model_of(body_json),
            body_bytes=len(encoded),
            body_sha256=hashlib.sha256(encoded).hexdigest(),
            body_json=None if over_cap else body_json,
            body_state="over_cap" if over_cap else "retained",
            captured_at=_parse_time(capture.get("captured_at")),
        )
        with Session(self.engine) as session:
            session.add(row)
            try:
                session.commit()
            except IntegrityError:
                session.rollback()
                return False
            self._prune_superseded(session, turn_id)
        return True

    def _prune_superseded(self, session: Session, turn_id: str) -> None:
        """Keep bodies only for the turn's first and latest call."""
        session.exec(sa_text(
            "UPDATE turn_egress_requests SET body_json = NULL, body_state = 'superseded'"
            " WHERE turn_id = :turn AND body_state = 'retained'"
            " AND id NOT IN ("
            "  (SELECT id FROM turn_egress_requests WHERE turn_id = :turn"
            "   ORDER BY captured_at, seq, id LIMIT 1),"
            "  (SELECT id FROM turn_egress_requests WHERE turn_id = :turn"
            "   ORDER BY captured_at DESC, seq DESC, id DESC LIMIT 1))"
        ).bindparams(turn=turn_id))
        session.commit()

    def for_turn(self, turn_id: str) -> list[dict[str, Any]]:
        import json

        with Session(self.engine) as session:
            rows = session.exec(
                select(EgressCaptureRecord)
                .where(EgressCaptureRecord.turn_id == turn_id)
                .order_by(EgressCaptureRecord.captured_at, EgressCaptureRecord.seq)
            ).all()
        items = []
        for row in rows:
            body = None
            if row.body_json is not None:
                try:
                    body = json.loads(row.body_json)
                except ValueError:
                    body = row.body_json
            items.append({
                "seq": row.seq,
                "attempt": row.attempt,
                "provider": row.provider,
                "transport": row.transport,
                "model": row.model,
                "body_bytes": row.body_bytes,
                "body_sha256": row.body_sha256,
                "body_state": row.body_state,
                "captured_at": row.captured_at,
                "body": body,
            })
        return items

    def delete_before(self, session: Session, cutoff: datetime) -> None:
        session.exec(sa_text(
            "DELETE FROM turn_egress_requests WHERE captured_at < :cutoff"
        ).bindparams(cutoff=cutoff))


def _parse_time(value: Any) -> datetime:
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return datetime.now(timezone.utc)
