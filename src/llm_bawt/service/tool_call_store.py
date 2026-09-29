"""Canonical tool-call and full-result persistence.

The normal tool_call_records row intentionally keeps only a small preview. The
one-to-one payload table is queried only by explicit full-result requests.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from sqlalchemy import BigInteger, Boolean, Column, DateTime, Float, Integer, String, Text, text as sa_text
from sqlmodel import Field, SQLModel, Session, select

from agent_bridge.tool_results import ToolResultPayload
from ..utils.schema import SchemaBootstrapGuard

if TYPE_CHECKING:
    from ..media.object_store import BlobBackend

logger = logging.getLogger(__name__)

# TASK-594: overflow tool-call results are offloaded to the object store
# (Garage S3 in prod) instead of hoarded in Postgres. The backend is resolved
# once and cached; if it can't be built (fs misconfig, S3 creds missing) the
# write path degrades to the legacy tool_call_result_payloads table so a result
# is never lost. Default FS root keeps dev/test working without S3.
DEFAULT_TOOLBLOBS_FS_ROOT = "/var/lib/llm-bawt/tool-results"

_TOOL_BLOB_BACKEND: "BlobBackend | None" = None
_TOOL_BLOB_RESOLVED = False


def _get_tool_blob_backend() -> "BlobBackend | None":
    """Return the (cached) blob backend for tool-result overflow, or None.

    None means "no object store available" — callers fall back to the legacy
    Postgres payload table. Resolution never raises; a misconfigured backend is
    treated as absent so a tool turn never dies over storage.
    """
    global _TOOL_BLOB_BACKEND, _TOOL_BLOB_RESOLVED
    if _TOOL_BLOB_RESOLVED:
        return _TOOL_BLOB_BACKEND
    _TOOL_BLOB_RESOLVED = True
    try:
        from ..media.object_store import get_blob_backend, s3_config_from_env

        fs_root = Path(
            os.environ.get("LLM_BAWT_TOOLBLOBS_FS_ROOT", DEFAULT_TOOLBLOBS_FS_ROOT)
        )
        _TOOL_BLOB_BACKEND = get_blob_backend(
            "tool_results", fs_root=fs_root, s3_cfg=s3_config_from_env()
        )
    except Exception:
        logger.exception(
            "tool-result blob backend unavailable; overflow will use Postgres fallback"
        )
        _TOOL_BLOB_BACKEND = None
    return _TOOL_BLOB_BACKEND


def reset_tool_blob_backend() -> None:
    """Drop the cached backend (tests / config reload)."""
    global _TOOL_BLOB_BACKEND, _TOOL_BLOB_RESOLVED
    _TOOL_BLOB_BACKEND = None
    _TOOL_BLOB_RESOLVED = False


class ToolCallRecord(SQLModel, table=True):
    __tablename__ = "tool_call_records"

    id: int | None = Field(default=None, sa_column=Column(Integer, primary_key=True, autoincrement=True))
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False, index=True),
    )
    turn_id: str | None = Field(default=None, index=True)
    bot_id: str | None = Field(default=None, index=True)
    user_id: str | None = Field(default=None, index=True)
    call_id: str | None = Field(default=None, index=True)
    tool_name: str = Field(default="unknown", max_length=128)
    arguments_json: str | None = Field(default=None, sa_column=Column(Text, nullable=True))
    result_text: str | None = Field(default=None, sa_column=Column(Text, nullable=True))
    iteration: int = Field(default=1)
    started_at: float | None = Field(default=None, sa_column=Column(Float, nullable=True))
    ended_at: float | None = Field(default=None, sa_column=Column(Float, nullable=True))
    duration_ms: float | None = Field(default=None, sa_column=Column(Float, nullable=True))
    text_offset: int | None = Field(default=None, sa_column=Column(Integer, nullable=True))
    is_error: bool | None = Field(default=None, sa_column=Column(Boolean, nullable=True))
    tool_use_id: str | None = Field(default=None, index=True)
    parent_tool_use_id: str | None = Field(default=None, index=True)
    approval_request_id: str | None = Field(default=None, index=True)
    approval_status: str | None = Field(default=None)
    preapproved: bool | None = Field(default=None, sa_column=Column(Boolean, nullable=True))
    result_total_chars: int | None = Field(default=None, sa_column=Column(Integer, nullable=True))
    result_total_bytes: int | None = Field(default=None, sa_column=Column(BigInteger, nullable=True))
    result_total_tokens: int | None = Field(default=None, sa_column=Column(Integer, nullable=True))
    result_sha256: str | None = Field(default=None, sa_column=Column(String(64), nullable=True))
    result_content_type: str | None = Field(default=None, sa_column=Column(String(128), nullable=True))
    result_complete: bool | None = Field(default=None, sa_column=Column(Boolean, nullable=True))
    result_payload_available: bool = Field(default=False, sa_column=Column(Boolean, nullable=False))
    # TASK-594: object-store key for overflow results (content-addressed sha256,
    # relative to the tool_results backend's toolblobs/ prefix). NULL => result
    # fits in result_text preview, or overflow lives in the legacy payload table.
    result_blob_key: str | None = Field(default=None, sa_column=Column(String(128), nullable=True))
    # TASK-977: media produced by this call (generate_image, screenshots) as
    # tiny ``[{"asset_id", "kind"}]`` refs; enriched to envelopes on read so the
    # tool card's inline preview survives activity refresh and reload.
    attachments_json: str | None = Field(default=None, sa_column=Column(Text, nullable=True))


class ToolCallResultPayloadRecord(SQLModel, table=True):
    __tablename__ = "tool_call_result_payloads"

    tool_call_record_id: int = Field(sa_column=Column(Integer, primary_key=True))
    content_text: str = Field(sa_column=Column(Text, nullable=False))
    content_type: str = Field(sa_column=Column(String(128), nullable=False))
    total_chars: int = Field(sa_column=Column(Integer, nullable=False))
    total_bytes: int = Field(sa_column=Column(BigInteger, nullable=False))
    sha256: str = Field(sa_column=Column(String(64), nullable=False))
    complete: bool = Field(sa_column=Column(Boolean, nullable=False))
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )


@dataclass(frozen=True)
class ToolResultPage:
    record_id: int
    content_type: str
    offset: int
    content: str
    total_chars: int
    total_bytes: int
    sha256: str
    complete: bool
    next_offset: int | None


@dataclass(frozen=True)
class ToolResultBlob:
    """A (possibly Range-bounded) full-result download body. TASK-594."""

    record_id: int
    data: bytes
    start: int          # inclusive first byte offset
    end: int            # inclusive last byte offset
    total_bytes: int    # size of the whole result
    content_type: str
    sha256: str
    partial: bool       # True => serve HTTP 206 with Content-Range


class ToolCallStore:
    _schema_guard = SchemaBootstrapGuard()

    def __init__(self, engine) -> None:
        self.engine = engine

    def ensure_schema(self) -> None:
        if self.engine is None:
            return

        def bootstrap(conn) -> None:
            SQLModel.metadata.create_all(
                bind=conn,
                tables=[
                    ToolCallRecord.__table__,
                    ToolCallResultPayloadRecord.__table__,
                ],
            )
            statements = (
                "ALTER TABLE tool_call_records ADD COLUMN IF NOT EXISTS result_total_chars INTEGER",
                "ALTER TABLE tool_call_records ADD COLUMN IF NOT EXISTS result_total_bytes BIGINT",
                "ALTER TABLE tool_call_records ADD COLUMN IF NOT EXISTS result_total_tokens INTEGER",
                "ALTER TABLE tool_call_records ADD COLUMN IF NOT EXISTS result_sha256 VARCHAR(64)",
                "ALTER TABLE tool_call_records ADD COLUMN IF NOT EXISTS result_content_type VARCHAR(128)",
                "ALTER TABLE tool_call_records ADD COLUMN IF NOT EXISTS result_complete BOOLEAN",
                "ALTER TABLE tool_call_records ADD COLUMN IF NOT EXISTS result_payload_available BOOLEAN NOT NULL DEFAULT FALSE",
                # TASK-594: object-store key for overflow tool-result bytes.
                "ALTER TABLE tool_call_records ADD COLUMN IF NOT EXISTS result_blob_key VARCHAR(128)",
                # TASK-977: per-call media refs for tool-card previews.
                "ALTER TABLE tool_call_records ADD COLUMN IF NOT EXISTS attachments_json TEXT",
                "CREATE INDEX IF NOT EXISTS ix_tool_call_records_turn_call ON tool_call_records (turn_id, call_id)",
                # TASK-594: content-addressed lookup for the TTL prune keep-set.
                "CREATE INDEX IF NOT EXISTS ix_tool_call_records_blob_key ON tool_call_records (result_blob_key)",
            )
            for statement in statements:
                conn.execute(sa_text(statement))

        self._schema_guard.run(self.engine, "tool-call-store", bootstrap)

    def save_start(
        self,
        *,
        turn_id: str | None,
        bot_id: str | None,
        user_id: str | None,
        call_id: str | None,
        tool_name: str,
        arguments: dict | None = None,
        iteration: int = 1,
        started_at: float | None = None,
        text_offset: int | None = None,
        tool_use_id: str | None = None,
        parent_tool_use_id: str | None = None,
    ) -> int | None:
        if self.engine is None:
            return None
        with Session(self.engine) as session:
            # Serialize starts with parent termination, including late/replayed
            # starts arriving after the terminal event.
            from .turn_logs import TurnLog
            parent = (session.exec(select(TurnLog).where(TurnLog.id == turn_id).with_for_update()).first()
                      if turn_id else None)
            row = None
            if turn_id and call_id:
                row = session.exec(
                    select(ToolCallRecord)
                    .where(ToolCallRecord.turn_id == turn_id)
                    .where(ToolCallRecord.call_id == call_id)
                ).first()
            if row is None and turn_id and tool_use_id:
                row = session.exec(select(ToolCallRecord).where(
                    ToolCallRecord.turn_id == turn_id,
                    ToolCallRecord.tool_use_id == tool_use_id,
                )).first()
            if row is None:
                row = ToolCallRecord(turn_id=turn_id, call_id=call_id)
            row.bot_id = bot_id
            row.user_id = user_id
            row.tool_name = tool_name
            row.arguments_json = json.dumps(arguments or {}, ensure_ascii=False, default=str)
            row.iteration = iteration
            row.started_at = started_at
            row.text_offset = text_offset
            row.tool_use_id = tool_use_id
            row.parent_tool_use_id = parent_tool_use_id
            if parent is not None and parent.ended_at is not None and row.ended_at is None and row.result_text is None:
                row.ended_at = parent.ended_at.replace(tzinfo=timezone.utc).timestamp()
                row.result_complete = False
            session.add(row)
            session.commit()
            session.refresh(row)
            return row.id

    def resolve_approval_result(
        self,
        *,
        tool_use_id: str,
        approval_request_id: str,
        approval_status: str,
        result: object,
        is_error: bool,
    ) -> bool:
        """Replace an MCP approval placeholder with its server-owned result."""
        if self.engine is None or not tool_use_id:
            return False
        try:
            with Session(self.engine) as session:
                row = session.exec(
                    select(ToolCallRecord)
                    .where(ToolCallRecord.tool_use_id == tool_use_id)
                    .order_by(ToolCallRecord.created_at.desc())
                ).first()
            if row is None:
                return False

            record_id, _ = self.save_result(
                turn_id=row.turn_id,
                call_id=row.call_id,
                tool_use_id=row.tool_use_id,
                tool_name=row.tool_name,
                bot_id=row.bot_id,
                user_id=row.user_id,
                payload=ToolResultPayload.from_value(result),
                ended_at=row.ended_at or datetime.now(timezone.utc).timestamp(),
                is_error=is_error,
                iteration=row.iteration,
                parent_tool_use_id=row.parent_tool_use_id,
            )
            if record_id is None:
                return False
            with Session(self.engine) as session:
                current = session.exec(
                    select(ToolCallRecord)
                    .where(ToolCallRecord.tool_use_id == tool_use_id)
                    .order_by(ToolCallRecord.created_at.desc())
                ).first()
                if current is None:
                    return False
                current.approval_request_id = approval_request_id
                current.approval_status = approval_status
                session.add(current)
                session.commit()
            return True
        except Exception:
            logger.exception(
                "Failed to resolve approval result tool_use_id=%s", tool_use_id,
            )
            return False

    def save_result(
        self,
        *,
        turn_id: str | None,
        call_id: str | None,
        tool_use_id: str | None,
        tool_name: str,
        bot_id: str | None,
        user_id: str | None,
        payload: ToolResultPayload,
        ended_at: float | None,
        is_error: bool | None,
        iteration: int = 1,
        parent_tool_use_id: str | None = None,
        attachments: list[dict] | None = None,
    ) -> tuple[int | None, bool]:
        """Persist payload and metadata atomically. Returns (record id, available).

        ``attachments`` are ``{asset_id, kind}`` refs (TASK-977). Empty/None
        leaves any previously stored refs untouched so a replayed or
        approval-resolved tool_end without media never erases them.
        """
        if self.engine is None:
            return None, False
        try:
            with Session(self.engine) as session:
                row = None
                if turn_id and call_id:
                    row = session.exec(
                        select(ToolCallRecord)
                        .where(ToolCallRecord.turn_id == turn_id)
                        .where(ToolCallRecord.call_id == call_id)
                    ).first()
                if row is None and turn_id and tool_use_id:
                    row = session.exec(
                        select(ToolCallRecord)
                        .where(ToolCallRecord.turn_id == turn_id)
                        .where(ToolCallRecord.tool_use_id == tool_use_id)
                    ).first()
                if row is None:
                    row = ToolCallRecord(
                        turn_id=turn_id,
                        bot_id=bot_id,
                        user_id=user_id,
                        call_id=call_id,
                        tool_name=tool_name,
                        iteration=iteration,
                        tool_use_id=tool_use_id,
                        parent_tool_use_id=parent_tool_use_id,
                    )
                    session.add(row)
                    session.flush()
                row.bot_id = bot_id or row.bot_id
                row.user_id = user_id or row.user_id
                row.call_id = call_id or row.call_id
                if tool_name and tool_name != "unknown":
                    row.tool_name = tool_name
                row.result_text = payload.preview
                row.result_total_chars = payload.total_chars
                row.result_total_bytes = payload.total_bytes
                row.result_total_tokens = payload.total_tokens
                row.result_sha256 = payload.sha256
                row.result_content_type = payload.content_type
                row.result_complete = payload.complete
                row.ended_at = ended_at
                row.is_error = is_error
                if attachments:
                    row.attachments_json = json.dumps(attachments)
                if ended_at and row.started_at:
                    row.duration_ms = (ended_at - row.started_at) * 1000
                session.add(row)
                session.flush()
                assert row.id is not None

                # TASK-594 spill gate: the record's result_text already holds the
                # full preview. Only persist a SECOND copy when the result exceeds
                # what the preview captured. Small results (the ~95% case) stop
                # here — no blob, no payload row, no duplication.
                if payload.total_chars <= len(payload.preview):
                    row.result_blob_key = None
                    row.result_payload_available = False
                    session.add(row)
                    # A prior write of this same (turn, call) row may have left an
                    # overflow row behind; drop it so state matches "preview only".
                    self._clear_legacy_payload(session, row.id)
                    session.commit()
                    return row.id, False

                # Overflow: offload the full bytes to the object store (Garage S3
                # in prod). Content-addressed by sha256 -> identical outputs share
                # one object. Falls back to the legacy Postgres payload table when
                # no object store is configured or the write fails.
                blob_key = self._offload_blob(payload)
                if blob_key is not None:
                    row.result_blob_key = blob_key
                    row.result_payload_available = True
                    session.add(row)
                    # New writes never touch Postgres for overflow; clear any
                    # legacy row from a pre-migration write of this record.
                    self._clear_legacy_payload(session, row.id)
                    session.commit()
                    return row.id, True

                # Fallback path: no object store -> keep overflow in Postgres
                # exactly as before (the read path still serves it).
                row.result_blob_key = None
                row.result_payload_available = True
                session.add(row)
                stored = session.get(ToolCallResultPayloadRecord, row.id)
                if stored is None:
                    stored = ToolCallResultPayloadRecord(tool_call_record_id=row.id, content_text="", content_type="text/plain", total_chars=0, total_bytes=0, sha256="", complete=False)
                stored.content_text = payload.content
                stored.content_type = payload.content_type
                stored.total_chars = payload.total_chars
                stored.total_bytes = payload.total_bytes
                stored.sha256 = payload.sha256
                stored.complete = payload.complete
                session.add(stored)
                session.commit()
                return row.id, True
        except Exception:
            logger.exception("Failed to persist full tool result turn=%s call=%s", turn_id, call_id)
            return None, False

    @staticmethod
    def _clear_legacy_payload(session: Session, record_id: int) -> None:
        """Delete the 1:1 Postgres overflow row for a record if present (idempotent)."""
        stale = session.get(ToolCallResultPayloadRecord, record_id)
        if stale is not None:
            session.delete(stale)

    def _offload_blob(self, payload: ToolResultPayload) -> str | None:
        """Write full result bytes to the object store; return the blob key or None.

        Content-addressed: the key is the sha256, so identical results dedupe to
        one object (put is skipped when it already exists). Never raises — any
        failure returns None and the caller falls back to Postgres.
        """
        backend = _get_tool_blob_backend()
        if backend is None:
            return None
        key = payload.sha256
        try:
            if not backend.exists(key):
                backend.put(key, payload.content.encode("utf-8"), payload.content_type)
            return key
        except Exception:
            logger.exception(
                "tool-result blob offload failed sha=%s; falling back to Postgres",
                payload.sha256,
            )
            return None

    def _resolve_full_text(self, session: Session, row: ToolCallRecord) -> tuple[str, str, str] | None:
        """Return (content_text, content_type, sha256) for a record's full result.

        Reads the object store when the record carries a blob key, else the
        legacy Postgres payload row. Returns None if neither source has it.
        """
        if row.result_blob_key:
            backend = _get_tool_blob_backend()
            if backend is not None:
                try:
                    data = backend.get(row.result_blob_key)
                    return (
                        data.decode("utf-8", errors="replace"),
                        row.result_content_type or "text/plain",
                        row.result_sha256 or "",
                    )
                except Exception:
                    logger.exception(
                        "blob read failed for record=%s key=%s; trying Postgres fallback",
                        row.id,
                        row.result_blob_key,
                    )
        payload = session.get(ToolCallResultPayloadRecord, row.id)
        if payload is None:
            return None
        return payload.content_text, payload.content_type, payload.sha256

    def page(self, record_id: int, *, bot_id: str, user_id: str, offset: int, limit: int) -> ToolResultPage | None:
        if self.engine is None:
            return None
        with Session(self.engine) as session:
            row = session.get(ToolCallRecord, record_id)
            if row is None or row.bot_id != bot_id or row.user_id != user_id:
                return None
            resolved = self._resolve_full_text(session, row)
            if resolved is None:
                return None
            content_text, content_type, sha256 = resolved
            total_chars = len(content_text)
            end = min(total_chars, offset + limit)
            return ToolResultPage(
                record_id=record_id,
                content_type=content_type,
                offset=offset,
                content=content_text[offset:end],
                total_chars=total_chars,
                total_bytes=row.result_total_bytes or len(content_text.encode("utf-8")),
                sha256=sha256,
                complete=bool(row.result_complete),
                next_offset=end if end < total_chars else None,
            )

    def raw_range(
        self,
        record_id: int,
        *,
        bot_id: str,
        user_id: str,
        range_header: str | None = None,
    ) -> ToolResultBlob | None:
        """Resolve full-result bytes for download, honoring a Range header.

        Blob-backed records use the object store's native ``get_range`` (so a
        large download resumes without pulling the whole object); legacy
        Postgres records return the full body (Range ignored — the pre-migration
        tail is small). Raises ``ValueError`` for an unsatisfiable range so the
        route can map it to HTTP 416.
        """
        if self.engine is None:
            return None
        with Session(self.engine) as session:
            row = session.get(ToolCallRecord, record_id)
            if row is None or row.bot_id != bot_id or row.user_id != user_id:
                return None
            if row.result_blob_key:
                backend = _get_tool_blob_backend()
                if backend is not None:
                    try:
                        from ..media.object_store import BlobNotFound

                        try:
                            br = backend.get_range(row.result_blob_key, range_header)
                            return ToolResultBlob(
                                record_id=record_id,
                                data=br.data,
                                start=br.start,
                                end=br.end,
                                total_bytes=br.total_size,
                                content_type=row.result_content_type or br.content_type,
                                sha256=row.result_sha256 or "",
                                partial=br.partial,
                            )
                        except BlobNotFound:
                            logger.warning(
                                "blob missing for record=%s key=%s; trying Postgres fallback",
                                record_id,
                                row.result_blob_key,
                            )
                    except ValueError:
                        raise
                    except Exception:
                        logger.exception(
                            "blob range read failed for record=%s key=%s; trying Postgres fallback",
                            record_id,
                            row.result_blob_key,
                        )
            payload = session.get(ToolCallResultPayloadRecord, record_id)
            if payload is None:
                return None
            full = payload.content_text.encode("utf-8")
            total = len(full)
            return ToolResultBlob(
                record_id=record_id,
                data=full,
                start=0,
                end=max(total - 1, 0),
                total_bytes=total,
                content_type=payload.content_type,
                sha256=payload.sha256,
                partial=False,
            )
