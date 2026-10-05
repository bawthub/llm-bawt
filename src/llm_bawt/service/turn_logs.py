"""Persistent turn logs for debugging and UI inspection. Retained indefinitely by default."""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timedelta, timezone
from urllib.parse import quote_plus

from sqlalchemy import Boolean, Column, DateTime, String, Text, exists as sa_exists, func, text as sa_text
from sqlmodel import Field, SQLModel, Session, delete, select

from ..utils.config import Config, has_database_credentials
from ..utils.schema import SchemaBootstrapGuard
from .tool_call_store import ToolCallRecord, ToolCallStore

logger = logging.getLogger(__name__)


class DeliveryTargetBusy(RuntimeError):
    """A durable inter-bot reservation won the target's idle race."""


# Terminal turn statuses — a turn in any of these is finished and no longer
# "in turn".  Non-terminal (in-progress) statuses are "streaming" (streaming
# path) and "pending" (non-streaming path).  Kept in ONE place so the
# ended_at stamp in save_turn/update_turn and any "is this bot in turn?"
# consumer agree on what "done" means.
TERMINAL_TURN_STATUSES = frozenset(
    {"ok", "completed", "error", "timeout", "cancelled", "aborted"}
)


def _is_terminal(status: str | None, end_reason: str | None) -> bool:
    """A turn is terminal once it has a terminal status OR an end_reason set.

    end_reason is only ever written when a turn ends (stop/error/aborted/
    question/tool_limit), so the streaming finalize path that stamps
    end_reason without a terminal status (chat_streaming) still counts.
    """
    return (status in TERMINAL_TURN_STATUSES) or (end_reason is not None)


def _terminal_ended_at(created_at: datetime, latency_ms: float | None = None) -> datetime:
    """Best completion timestamp for a terminal turn update."""
    if latency_ms is not None:
        try:
            return created_at + timedelta(milliseconds=float(latency_ms))
        except Exception:
            pass
    return datetime.now(timezone.utc)


def _extract_trigger_id(request_payload: dict) -> str | None:
    """Extract the last user message ID from a request payload."""
    messages = request_payload.get("messages")
    if not isinstance(messages, list):
        return None
    for message in reversed(messages):
        if not isinstance(message, dict) or str(message.get("role") or "") != "user":
            continue
        for key in ("id", "db_id", "message_id"):
            value = message.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


class TurnLog(SQLModel, table=True):
    """Persistent log for one chat turn."""

    __tablename__ = "turn_logs"

    id: str = Field(primary_key=True)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False, index=True),
    )
    request_id: str | None = Field(default=None, index=True)
    path: str = Field(default="/v1/chat/completions", max_length=128)
    stream: bool = Field(default=False, index=True)
    model: str | None = Field(default=None, index=True)
    bot_id: str | None = Field(default=None, index=True)
    user_id: str | None = Field(default=None, index=True)
    status: str = Field(default="ok", max_length=32)
    latency_ms: float | None = Field(default=None)
    user_prompt: str | None = Field(default=None, sa_column=Column(Text, nullable=True))
    request_json: str | None = Field(default=None, sa_column=Column(Text, nullable=True))
    response_text: str | None = Field(default=None, sa_column=Column(Text, nullable=True))
    # TASK-360 (P4): partial model reasoning ("thinking") flushed DURING the turn
    # (mirrors response_text incremental persistence from TASK-286) so a mid-turn
    # cold reload can recover already-produced reasoning instead of losing it until
    # the final assistant row persists. Nullable; pre-existing rows stay NULL.
    reasoning: str | None = Field(default=None, sa_column=Column(Text, nullable=True))
    tool_calls_json: str | None = Field(default=None, sa_column=Column(Text, nullable=True))
    error_text: str | None = Field(default=None, sa_column=Column(Text, nullable=True))
    trigger_message_id: str | None = Field(default=None, index=True)
    assistant_message_id: str | None = Field(default=None, index=True)
    agent_session_key: str | None = Field(default=None, max_length=128, index=True)
    agent_request_id: str | None = Field(default=None, max_length=128, index=True)
    animation: str | None = Field(default=None, sa_column=Column(String(255), nullable=True))
    # Per-turn token accounting from the upstream SDK (claude_code, etc.).
    # Stored as JSON text so older rows without this column still load cleanly.
    # Surfaced on TurnLogListItem / TurnLogDetail so the chat UI's per-bubble
    # usage pill survives reloads and history syncs.
    token_usage_json: str | None = Field(default=None, sa_column=Column(Text, nullable=True))
    # Whether this turn's assistant output was scrubbed for TTS (voice-optimized
    # bots). Recorded so the scrub decision is tracked on the turn, not just
    # recomputed — single source of truth is should_scrub_for_tts().
    tts_scrubbed: bool | None = Field(default=None, sa_column=Column(Boolean, nullable=True))
    # --- Turn-lifecycle / continuation chain (TASK-269) ---
    # Why a turn ended.  Default "stop" (normal completion).  When the agent
    # asked an AskUserQuestion and deferred it, the turn ends cleanly with
    # end_reason="question" and question_id set to the originating tool_use_id —
    # the UI renders a first-class QuestionMessage off this instead of holding
    # the turn open.  Other terminal reasons: "error", "tool_limit", "aborted".
    end_reason: str | None = Field(default=None, sa_column=Column(String(32), nullable=True))
    # tool_use_id of the AskUserQuestion this turn ended on (FK-ish into
    # chat_pending_questions.tool_use_id).  Only set when end_reason="question".
    question_id: str | None = Field(default=None, index=True, max_length=128)
    # When this turn is the continuation that answered a prior question, point
    # back at the awaiting turn.  Lets the UI thread the chain and resolve the
    # prior QuestionMessage across tabs on turn_start{parent_turn_id}.
    parent_turn_id: str | None = Field(default=None, index=True, max_length=128)
    # Wall-clock completion timestamp.  NULL while the turn is in progress;
    # stamped exactly once on the first terminal transition (see _is_terminal)
    # in save_turn/update_turn.  This is the single, path-agnostic "is this
    # turn still running?" signal — `ended_at IS NULL` works identically for
    # streaming/non-streaming and local/agent-backend turns, unlike the
    # overloaded `status` string.  Also yields true turn duration and a clean
    # zombie sweep (NULL + old created_at).
    ended_at: datetime | None = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), nullable=True, index=True),
    )



class TurnLogStore:
    """DB access helper for persistent turn logs."""

    _last_cleanup_at: float = 0.0
    _cleanup_interval_seconds: float = 300.0
    _backfill_done: bool = False
    _schema_guard = SchemaBootstrapGuard()

    def __init__(self, config: Config, ttl_hours: int | None = None):
        self.config = config
        # None disables expiry entirely — turn logs are retained forever.
        self.ttl_hours = max(1, int(ttl_hours)) if ttl_hours is not None else None
        self.engine = None
        try:
            host = getattr(config, "POSTGRES_HOST", "localhost")
            port = int(getattr(config, "POSTGRES_PORT", 5432))
            user = getattr(config, "POSTGRES_USER", "llm_bawt")
            password = getattr(config, "POSTGRES_PASSWORD", "")
            database = getattr(config, "POSTGRES_DATABASE", "llm_bawt")
            encoded_password = quote_plus(password)
            from ..utils.db import get_shared_engine
            self.engine = get_shared_engine(config)  # TASK-202: shared pool
            if self.engine is None:
                return
            self._ensure_tables_exist()
            self._backfill_trigger_message_ids()
        except Exception as e:
            self.engine = None
            logger.warning("Turn logs DB unavailable: %s", e)

    def _ensure_tables_exist(self) -> None:
        if self.engine is None:
            return
        # Keep orchestration here while each nested store remains independently
        # guarded for direct callers.
        ToolCallStore(self.engine).ensure_schema()
        from .changed_files_store import ChangedFilesStore
        ChangedFilesStore(self.engine).ensure_schema()

        def bootstrap(conn) -> None:
            SQLModel.metadata.create_all(bind=conn, tables=[TurnLog.__table__])
            # Add columns that may not exist on older tables.
            try:
                conn.execute(sa_text(
                    "ALTER TABLE turn_logs ADD COLUMN IF NOT EXISTS"
                    " trigger_message_id VARCHAR"
                ))
                conn.execute(sa_text(
                    "CREATE INDEX IF NOT EXISTS ix_turn_logs_trigger_message_id"
                    " ON turn_logs (trigger_message_id)"
                ))
                conn.execute(sa_text(
                    "ALTER TABLE turn_logs ADD COLUMN IF NOT EXISTS"
                    " assistant_message_id VARCHAR"
                ))
                conn.execute(sa_text(
                    "CREATE INDEX IF NOT EXISTS ix_turn_logs_assistant_message_id"
                    " ON turn_logs (assistant_message_id)"
                ))
                conn.execute(sa_text(
                    "ALTER TABLE turn_logs ADD COLUMN IF NOT EXISTS"
                    " agent_session_key VARCHAR(128)"
                ))
                conn.execute(sa_text(
                    "CREATE INDEX IF NOT EXISTS ix_turn_logs_agent_session_key"
                    " ON turn_logs (agent_session_key)"
                ))
                conn.execute(sa_text(
                    "ALTER TABLE turn_logs ADD COLUMN IF NOT EXISTS"
                    " agent_request_id VARCHAR(128)"
                ))
                conn.execute(sa_text(
                    "CREATE INDEX IF NOT EXISTS ix_turn_logs_agent_request_id"
                    " ON turn_logs (agent_request_id)"
                ))
                conn.execute(sa_text(
                    "ALTER TABLE turn_logs ADD COLUMN IF NOT EXISTS animation VARCHAR(255)"
                ))
                conn.execute(sa_text(
                    "ALTER TABLE turn_logs ADD COLUMN IF NOT EXISTS token_usage_json TEXT"
                ))
                # Tracks whether the turn's output was scrubbed for TTS.
                conn.execute(sa_text(
                    "ALTER TABLE turn_logs ADD COLUMN IF NOT EXISTS tts_scrubbed BOOLEAN"
                ))
                # TASK-360 (P4): mid-turn partial reasoning for cold-reload resume.
                conn.execute(sa_text(
                    "ALTER TABLE turn_logs ADD COLUMN IF NOT EXISTS reasoning TEXT"
                ))
                # TASK-269 turn-lifecycle / continuation columns.
                conn.execute(sa_text(
                    "ALTER TABLE turn_logs ADD COLUMN IF NOT EXISTS end_reason VARCHAR(32)"
                ))
                conn.execute(sa_text(
                    "ALTER TABLE turn_logs ADD COLUMN IF NOT EXISTS question_id VARCHAR(128)"
                ))
                conn.execute(sa_text(
                    "CREATE INDEX IF NOT EXISTS ix_turn_logs_question_id"
                    " ON turn_logs (question_id)"
                ))
                conn.execute(sa_text(
                    "ALTER TABLE turn_logs ADD COLUMN IF NOT EXISTS parent_turn_id VARCHAR(128)"
                ))
                conn.execute(sa_text(
                    "CREATE INDEX IF NOT EXISTS ix_turn_logs_parent_turn_id"
                    " ON turn_logs (parent_turn_id)"
                ))
                # Interleaved-transcript support: char offset of assistant text
                # emitted before each tool call (see ToolCallRecord.text_offset).
                conn.execute(sa_text(
                    "ALTER TABLE tool_call_records ADD COLUMN IF NOT EXISTS text_offset INTEGER"
                ))
                # Persisted tool-call failure flag (see ToolCallRecord.is_error)
                # so the red error ring survives a reload/reconnect.
                conn.execute(sa_text(
                    "ALTER TABLE tool_call_records ADD COLUMN IF NOT EXISTS is_error BOOLEAN"
                ))
                # TASK-344: SDK tool_use ids so sub-agent nesting survives reload
                # (see ToolCallRecord.tool_use_id / parent_tool_use_id).
                conn.execute(sa_text(
                    "ALTER TABLE tool_call_records ADD COLUMN IF NOT EXISTS tool_use_id VARCHAR(128)"
                ))
                conn.execute(sa_text(
                    "ALTER TABLE tool_call_records ADD COLUMN IF NOT EXISTS parent_tool_use_id VARCHAR(128)"
                ))
                conn.execute(sa_text(
                    "CREATE INDEX IF NOT EXISTS ix_tool_call_records_parent_tool_use_id"
                    " ON tool_call_records (parent_tool_use_id)"
                ))
                # TASK-305: approval gate persistence — link tool calls to
                # their approval request so the approval card survives reload.
                conn.execute(sa_text(
                    "ALTER TABLE tool_call_records ADD COLUMN IF NOT EXISTS approval_request_id VARCHAR(128)"
                ))
                conn.execute(sa_text(
                    "CREATE INDEX IF NOT EXISTS ix_tool_call_records_approval_request_id"
                    " ON tool_call_records (approval_request_id)"
                ))
                conn.execute(sa_text(
                    "ALTER TABLE tool_call_records ADD COLUMN IF NOT EXISTS approval_status VARCHAR(24)"
                ))
                conn.execute(sa_text(
                    "ALTER TABLE tool_call_records ADD COLUMN IF NOT EXISTS preapproved BOOLEAN"
                ))
                # Path-agnostic turn-completion timestamp (NULL = in progress).
                conn.execute(sa_text(
                    "ALTER TABLE turn_logs ADD COLUMN IF NOT EXISTS"
                    " ended_at TIMESTAMPTZ"
                ))
                # Partial index makes "is bot X in turn?" a tiny lookup —
                # only in-flight rows are indexed.
                conn.execute(sa_text(
                    "CREATE INDEX IF NOT EXISTS ix_turn_logs_active"
                    " ON turn_logs (bot_id, created_at) WHERE ended_at IS NULL"
                ))
                # One-time backfill: existing already-finished rows have NULL
                # ended_at after the column add, which would make every recent
                # completed turn look in-flight to active_only.  Stamp them from
                # created_at + latency.  Idempotent (only touches terminal rows
                # that are still NULL); zombie streaming/pending rows correctly
                # stay NULL and age out of the scan window.
                conn.execute(sa_text(
                    "UPDATE turn_logs SET"
                    " ended_at = created_at + (COALESCE(latency_ms, 0) * interval '1 millisecond')"
                    " WHERE ended_at IS NULL"
                    " AND (status IN ('ok','completed','error','timeout','cancelled','aborted')"
                    "      OR end_reason IS NOT NULL)"
                ))
            except Exception:
                logger.exception("Turn-log additive schema migration failed")
                raise

        self._schema_guard.run(self.engine, "turn-log-store", bootstrap)

    def _backfill_trigger_message_ids(self) -> None:
        """One-time backfill: populate trigger_message_id for existing rows."""
        if self.engine is None or TurnLogStore._backfill_done:
            return
        TurnLogStore._backfill_done = True
        try:
            with Session(self.engine) as session:
                rows = session.exec(
                    select(TurnLog)
                    .where(TurnLog.trigger_message_id.is_(None))
                    .where(TurnLog.request_json.is_not(None))
                    .limit(5000)
                ).all()
                if not rows:
                    return
                updated = 0
                for row in rows:
                    try:
                        payload = json.loads(row.request_json) if row.request_json else None
                    except Exception:
                        continue
                    if payload:
                        tid = _extract_trigger_id(payload)
                        if tid:
                            row.trigger_message_id = tid
                            session.add(row)
                            updated += 1
                if updated:
                    session.commit()
                    logger.info("Backfilled trigger_message_id for %d turn logs", updated)
        except Exception as e:
            logger.debug("trigger_message_id backfill skipped: %s", e)

    def _cleanup_expired_if_due(self, force: bool = False) -> None:
        if self.engine is None or self.ttl_hours is None:
            return
        now = time.time()
        if not force and (now - self.__class__._last_cleanup_at) < self.__class__._cleanup_interval_seconds:
            return

        cutoff = datetime.now(timezone.utc) - timedelta(hours=self.ttl_hours)
        try:
            with Session(self.engine) as session:
                session.exec(delete(TurnLog).where(TurnLog.created_at < cutoff))
                session.commit()
            self.__class__._last_cleanup_at = now
        except Exception as e:
            logger.debug("Turn log TTL cleanup skipped: %s", e)

    def save_turn(
        self,
        *,
        turn_id: str,
        request_id: str | None,
        path: str,
        stream: bool,
        model: str | None,
        bot_id: str | None,
        user_id: str | None,
        status: str,
        latency_ms: float | None,
        user_prompt: str | None,
        request_payload: dict | None,
        response_text: str | None,
        tool_calls: list[dict] | None,
        error_text: str | None = None,
        trigger_message_id: str | None = None,
        assistant_message_id: str | None = None,
        agent_session_key: str | None = None,
        agent_request_id: str | None = None,
        animation: str | None = None,
        token_usage: dict | None = None,
        parent_turn_id: str | None = None,
    ) -> None:
        """Persist one turn entry (and run TTL cleanup if a TTL is configured)."""
        if self.engine is None:
            return

        self._cleanup_expired_if_due()

        # Auto-extract trigger message ID from request if not provided.
        if trigger_message_id is None and request_payload:
            trigger_message_id = _extract_trigger_id(request_payload)

        created_at = datetime.now(timezone.utc)

        # Durable inter-bot callbacks reserve their deterministic turn under the
        # same per-target advisory lock used by the outbox claim. Normal turn
        # creation also takes that lock: if any OTHER recent turn is open, the
        # reservation lost its idle race and the dispatcher receives a typed
        # busy signal instead of forcing concurrency. Reusing the reserved id is
        # an atomic upsert, so target acceptance cannot create two turn rows.
        delivery_id = None
        if request_payload:
            raw_delivery = request_payload.get("inter_bot_delivery_id")
            if isinstance(raw_delivery, str) and raw_delivery.strip():
                delivery_id = raw_delivery.strip()
        if bot_id:
            with self.engine.begin() as conn:
                conn.execute(
                    sa_text("SELECT pg_advisory_xact_lock(hashtext(:target))"),
                    {"target": bot_id},
                )
                other_open = conn.execute(
                    sa_text(
                        "SELECT id, status FROM turn_logs WHERE bot_id=:target"
                        " AND ended_at IS NULL AND id != :turn_id"
                        " ORDER BY created_at DESC LIMIT 1"
                    ),
                    {"target": bot_id, "turn_id": turn_id},
                ).mappings().first()
                if other_open and (delivery_id or other_open["status"] == "reserved"):
                    if delivery_id:
                        conn.execute(
                            sa_text("DELETE FROM turn_logs WHERE id=:id AND status='reserved'"),
                            {"id": turn_id},
                        )
                    raise DeliveryTargetBusy(
                        f"target '{bot_id}' became busy with turn {other_open['id']}"
                    )
                existing_status = conn.execute(
                    sa_text("SELECT status FROM turn_logs WHERE id=:id FOR UPDATE"),
                    {"id": turn_id},
                ).scalar_one_or_none()
                if existing_status is not None:
                    if existing_status != "reserved":
                        raise RuntimeError(f"turn id already exists: {turn_id}")
                    conn.execute(
                        sa_text(
                            "UPDATE turn_logs SET request_id=:request_id, path=:path,"
                            " stream=:stream, model=:model, bot_id=:bot_id, user_id=:user_id,"
                            " status=:status, latency_ms=:latency_ms, user_prompt=:user_prompt,"
                            " request_json=:request_json, response_text=:response_text,"
                            " error_text=:error_text, trigger_message_id=:trigger_message_id,"
                            " assistant_message_id=:assistant_message_id,"
                            " agent_session_key=:agent_session_key,"
                            " agent_request_id=:agent_request_id, animation=:animation,"
                            " token_usage_json=:token_usage_json, parent_turn_id=:parent_turn_id"
                            " WHERE id=:turn_id"
                        ),
                        {
                            "turn_id": turn_id,
                            "request_id": request_id,
                            "path": path,
                            "stream": stream,
                            "model": model,
                            "bot_id": bot_id,
                            "user_id": user_id,
                            "status": status,
                            "latency_ms": latency_ms,
                            "user_prompt": user_prompt,
                            "request_json": json.dumps(request_payload, ensure_ascii=False, default=str) if request_payload else None,
                            "response_text": response_text,
                            "error_text": error_text,
                            "trigger_message_id": trigger_message_id,
                            "assistant_message_id": assistant_message_id,
                            "agent_session_key": agent_session_key,
                            "agent_request_id": agent_request_id,
                            "animation": animation,
                            "token_usage_json": json.dumps(token_usage, ensure_ascii=False, default=str) if isinstance(token_usage, dict) else None,
                            "parent_turn_id": parent_turn_id,
                        },
                    )
                    return

                # Ordinary turn insert stays inside the advisory-lock transaction
                # so it cannot cross a delivery claim between check and commit.
                conn.execute(
                    TurnLog.__table__.insert().values(
                        id=turn_id,
                        created_at=created_at,
                        request_id=request_id,
                        path=path,
                        stream=stream,
                        model=model,
                        bot_id=(bot_id or None),
                        user_id=(user_id or None),
                        status=status,
                        latency_ms=latency_ms,
                        user_prompt=user_prompt,
                        request_json=json.dumps(request_payload, ensure_ascii=False, default=str) if request_payload else None,
                        response_text=response_text,
                        error_text=error_text,
                        trigger_message_id=trigger_message_id,
                        assistant_message_id=assistant_message_id,
                        agent_session_key=agent_session_key,
                        agent_request_id=agent_request_id,
                        animation=animation,
                        token_usage_json=(json.dumps(token_usage, ensure_ascii=False, default=str) if isinstance(token_usage, dict) else None),
                        parent_turn_id=parent_turn_id,
                        ended_at=(_terminal_ended_at(created_at, latency_ms) if _is_terminal(status, None) else None),
                    )
                )
                return

        row = TurnLog(
            id=turn_id,
            created_at=created_at,
            request_id=request_id,
            path=path,
            stream=stream,
            model=model,
            bot_id=(bot_id or None),
            user_id=(user_id or None),
            status=status,
            latency_ms=latency_ms,
            user_prompt=user_prompt,
            request_json=json.dumps(request_payload, ensure_ascii=False, default=str) if request_payload else None,
            response_text=response_text,
            # TASK-364: tool_calls_json retired — tool_call_records is canonical.
            # Column left in place (tombstoned) but no longer written or read.
            error_text=error_text,
            trigger_message_id=trigger_message_id,
            agent_session_key=agent_session_key,
            agent_request_id=agent_request_id,
            animation=animation,
            token_usage_json=(
                json.dumps(token_usage, ensure_ascii=False, default=str)
                if isinstance(token_usage, dict) else None
            ),
            parent_turn_id=parent_turn_id,
            # Rare: a turn created already-terminal (e.g. errored before any
            # processing) should not look perpetually in-flight.
            ended_at=(
                _terminal_ended_at(created_at, latency_ms)
                if _is_terminal(status, None) else None
            ),
        )

        with Session(self.engine) as session:
            session.add(row)
            session.commit()

    def update_turn(
        self,
        *,
        turn_id: str,
        status: str | None = None,
        latency_ms: float | None = None,
        response_text: str | None = None,
        request_payload: dict | None = None,
        tool_calls: list[dict] | None = None,
        error_text: str | None = None,
        assistant_message_id: str | None = None,
        agent_session_key: str | None = None,
        agent_request_id: str | None = None,
        animation: str | None = None,
        token_usage: dict | None = None,
        end_reason: str | None = None,
        question_id: str | None = None,
        parent_turn_id: str | None = None,
        tts_scrubbed: bool | None = None,
    ) -> None:
        """Update an existing turn log row with new data."""
        if self.engine is None:
            return
        with Session(self.engine) as session:
            row = session.exec(select(TurnLog).where(TurnLog.id == turn_id).with_for_update()).first()
            if row is None:
                logger.debug("update_turn: no row with id=%s", turn_id)
                return
            prior_status = row.status
            prior_end_reason = row.end_reason
            if prior_status == "cancelling":
                if _is_terminal(status, end_reason):
                    status, end_reason = "aborted", "aborted"
                else:
                    status = None  # a late streaming update cannot revoke intent
            # Terminal outcomes are monotonic. Preserve the existing stale-reap
            # repair exception, but never let EOF/late errors undo cancellation.
            repair_timeout = (prior_status == "timeout" and prior_end_reason == "timeout"
                              and status in ("ok", "completed") and end_reason not in (None, "timeout"))
            if row.ended_at is not None and not repair_timeout:
                # Preserve diagnostic enrichment for the same outcome (notably
                # approval_persist_failed), but reject a conflicting late error.
                if status not in (None, prior_status) or (
                    prior_end_reason is not None and end_reason not in (None, prior_end_reason)
                ):
                    error_text = None
                status = None
                if row.end_reason is not None:
                    end_reason = None
            if status is not None:
                row.status = status
            if latency_ms is not None:
                row.latency_ms = latency_ms
            if response_text is not None:
                row.response_text = response_text
            if request_payload is not None:
                from .turn_termination import request_payload_with_audit
                row.request_json = request_payload_with_audit(row.request_json, request_payload)
                # Backfill trigger_message_id if it wasn't set on initial persist
                # (common when prepared_messages were empty at creation time).
                if not row.trigger_message_id:
                    tid = _extract_trigger_id(request_payload)
                    if tid:
                        row.trigger_message_id = tid
            # TASK-364: tool_calls_json retired (tombstoned) — no longer written.
            if error_text is not None:
                row.error_text = error_text
            if assistant_message_id is not None:
                row.assistant_message_id = assistant_message_id
            if agent_session_key is not None:
                row.agent_session_key = agent_session_key
            if agent_request_id is not None:
                row.agent_request_id = agent_request_id
            if animation is not None:
                row.animation = animation
            if token_usage is not None:
                row.token_usage_json = (
                    json.dumps(token_usage, ensure_ascii=False, default=str)
                    if isinstance(token_usage, dict) else None
                )
            if tts_scrubbed is not None:
                row.tts_scrubbed = tts_scrubbed
            if end_reason is not None:
                row.end_reason = end_reason
            if question_id is not None:
                row.question_id = question_id
            if parent_turn_id is not None:
                row.parent_turn_id = parent_turn_id
            # Stamp completion exactly once, on the first terminal transition.
            # Covers every terminal writer (streaming finalize, _finalize_turn,
            # abort route) since they all funnel through update_turn.
            if row.ended_at is None and _is_terminal(status, end_reason):
                row.ended_at = _terminal_ended_at(row.created_at, latency_ms)
            elif (
                prior_status == "timeout"
                and prior_end_reason == "timeout"
                and status in ("ok", "completed")
                and end_reason not in (None, "timeout")
            ):
                # A queued agent turn can be incorrectly stamped timeout by a
                # stale-turn reap, then later complete normally when the bridge
                # drains the queue. In that repair path, the successful
                # finalization is authoritative and should also repair ended_at.
                row.ended_at = _terminal_ended_at(row.created_at, latency_ms)
            if row.ended_at is not None:
                from .turn_termination import interrupt_unresolved_tools
                interrupt_unresolved_tools(session, row.id, row.ended_at)
            session.add(row)
            session.commit()

    def update_partial_response(
        self, *, turn_id: str, response_text: str, reasoning: str | None = None
    ) -> None:
        """Write in-flight partial assistant text (and, TASK-360/P4, partial
        reasoning) without clobbering a finalized turn (TASK-286).

        Conditional on ``status='streaming'`` so once ``_finalize_turn`` flips
        the row to ``ok`` (with the cleaned, authoritative text) this becomes a
        no-op — a late drain write can never overwrite the final response. This
        is the text analogue of the incremental ``tool_call_records`` writes:
        it gives a COLD reload (resumeScan → /v1/turn-logs) the partial text
        that was streamed before a refresh, and survives client disconnect
        because the drain consumer runs server-side, not in the request thread.

        ``reasoning`` is flushed the same way so a mid-turn cold reload can
        recover already-produced thinking; when ``None`` the reasoning column is
        left untouched.
        """
        if self.engine is None or not turn_id:
            return
        try:
            with self.engine.begin() as conn:
                if reasoning is not None:
                    conn.execute(
                        sa_text(
                            "UPDATE turn_logs SET response_text = :t, reasoning = :r"
                            " WHERE id = :id AND status = 'streaming'"
                        ),
                        {"t": response_text, "r": reasoning, "id": turn_id},
                    )
                else:
                    conn.execute(
                        sa_text(
                            "UPDATE turn_logs SET response_text = :t"
                            " WHERE id = :id AND status = 'streaming'"
                        ),
                        {"t": response_text, "id": turn_id},
                    )
        except Exception as e:
            logger.debug("update_partial_response failed for %s: %s", turn_id, e)

    def reap_other_open_turns(
        self, *, bot_id: str, current_turn_id: str
    ) -> list[dict]:
        """Close older still-open turns for ``bot_id`` as a timeout.

        The single source of truth for "a turn began" is the SDK/bridge
        CONFIRMING its first real output (see chat_streaming's confirmed-start
        hook) — not the optimistic ``save_turn`` insert or ``turn_start``
        publish, both of which fire before the backend produces anything. When
        that confirmed signal lands for a new turn, this reaps prior turn rows
        for the same bot that are still ``ended_at IS NULL`` (zombies that never
        terminated — dropped bridge, aborted-without-finalize, server restart
        mid-turn). They are stamped ``status='timeout'``,
        ``end_reason='timeout'`` and ``ended_at=now`` atomically.

        Registered live/queued workers and cancellation intent are excluded:
        another turn's first output is not evidence that those workers stopped.
        Unresolved tools of genuinely reaped turns settle in the same transaction.

        Returns the reaped rows as ``[{"id": ..., "user_id": ...}]`` so the
        caller can emit a ``turn_complete`` per row to clear UI indicators on
        the correct ``{bot_id}:{user_id}`` stream. Excludes ``current_turn_id``
        so the turn that triggered the reap is never closed by it. Newer open
        rows are also preserved: agent session queues can let an older turn
        reach confirmed-start after a newer user message has already been
        inserted, and that newer queued turn must not be reaped as stale.
        """
        if self.engine is None or not bot_id or not current_turn_id:
            return []
        try:
            from .turn_termination import reap_unowned_turns
            return reap_unowned_turns(self, bot_id, current_turn_id)
        except Exception as e:
            logger.debug("reap_other_open_turns failed for %s: %s", bot_id, e)
            return []

    def get_turn(self, turn_id: str) -> TurnLog | None:
        """Get one turn by id."""
        if self.engine is None:
            return None
        self._cleanup_expired_if_due()
        with Session(self.engine) as session:
            return session.exec(select(TurnLog).where(TurnLog.id == turn_id)).first()

    def active_turn_for_bot(
        self, bot_id: str, *, within_seconds: int = 1800
    ) -> TurnLog | None:
        """Return the bot's most recent in-flight turn, or None if idle.

        In-flight == ``ended_at IS NULL``.  ``within_seconds`` bounds the scan
        so a crashed/zombie turn (never stamped) stops blocking after a sane
        window — callers wanting to override that should pass force, not widen
        the window.  Path-agnostic: matches streaming and non-streaming turns.
        """
        if self.engine is None or not bot_id:
            return None
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=max(1, within_seconds))
        with Session(self.engine) as session:
            return session.exec(
                select(TurnLog)
                .where(TurnLog.bot_id == bot_id)
                .where(TurnLog.ended_at.is_(None))
                .where(TurnLog.created_at > cutoff)
                .order_by(TurnLog.created_at.desc())
            ).first()

    def list_turns(
        self,
        *,
        bot_id: str | None = None,
        user_id: str | None = None,
        model: str | None = None,
        request_id: str | None = None,
        status: str | None = None,
        active_only: bool = False,
        stream: bool | None = None,
        has_tools: bool | None = None,
        trigger_message_ids: set[str] | None = None,
        after: float | None = None,
        before: float | None = None,
        since_hours: int = 24,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[TurnLog], int]:
        """List turns with filters and pagination."""
        if self.engine is None:
            return [], 0
        self._cleanup_expired_if_due()

        conditions: list = []
        if after is not None:
            conditions.append(TurnLog.created_at >= datetime.fromtimestamp(after, tz=timezone.utc))
        elif before is None and not active_only:
            # Active turns remain active regardless of when they started.
            # The recency window is only for historical turn browsing.
            since_cutoff = datetime.now(timezone.utc) - timedelta(hours=max(1, int(since_hours)))
            conditions.append(TurnLog.created_at >= since_cutoff)
        if before is not None:
            conditions.append(TurnLog.created_at <= datetime.fromtimestamp(before, tz=timezone.utc))
        if bot_id:
            conditions.append(TurnLog.bot_id == bot_id.strip().lower())
        if user_id:
            conditions.append(TurnLog.user_id == user_id.strip().lower())
        if model:
            conditions.append(TurnLog.model == model.strip())
        if request_id:
            conditions.append(TurnLog.request_id == request_id.strip())
        if status:
            conditions.append(TurnLog.status == status.strip().lower())
        if active_only:
            # In-progress, path-agnostic: streaming AND non-streaming in-flight
            # turns. Supersedes the old status="streaming" proxy.
            conditions.append(TurnLog.ended_at.is_(None))
        if stream is not None:
            conditions.append(TurnLog.stream.is_(stream))
        if has_tools is not None:
            # TASK-364: presence of tools is derived from the canonical
            # tool_call_records table (correlated EXISTS), not the retired
            # tool_calls_json blob — so the filter stays correct once the blob
            # is no longer written.
            tool_exists = sa_exists().where(ToolCallRecord.turn_id == TurnLog.id)
            conditions.append(tool_exists if has_tools else ~tool_exists)
        if trigger_message_ids:
            # Include rows matching the given IDs OR rows with NULL trigger
            # (agent-backend turns that need post-processing via turn_id fallback).
            conditions.append(
                TurnLog.trigger_message_id.in_(trigger_message_ids)
                | TurnLog.trigger_message_id.is_(None)
            )

        statement = select(TurnLog).where(*conditions).order_by(TurnLog.created_at.desc())
        count_statement = select(func.count()).select_from(TurnLog).where(*conditions)

        with Session(self.engine) as session:
            count_result = session.exec(count_statement).one()
            try:
                total_count = int(count_result or 0)
            except (TypeError, ValueError):
                total_count = int(count_result[0] or 0)
            rows = session.exec(statement.offset(offset).limit(limit)).all()

        return rows, total_count

    def recent_turns_by_bot(
        self,
        *,
        user_id: str | None,
        bot_ids: list[str] | None = None,
        since_hours: int = 168,
    ) -> list[TurnLog]:
        """Return the most recent TurnLog per bot_id for a user — one row each.

        Uses Postgres ``DISTINCT ON (bot_id)`` so the entire fan-out is a single
        index-backed query. Designed for "what's been happening with each bot"
        dashboard views — replaces the N round-trip per-bot history poll.

        Args:
            user_id: Scope to a user (lowercased). ``None`` returns the latest
                turn per bot across all users (admin/global view).
            bot_ids: Optional whitelist to restrict the result. ``None`` returns
                all bots the user has talked to in the window.
            since_hours: Only consider turns from the last N hours (default 7d).
        """
        if self.engine is None:
            return []
        self._cleanup_expired_if_due()

        since_cutoff = datetime.now(timezone.utc) - timedelta(
            hours=max(1, int(since_hours))
        )
        conditions: list = [
            TurnLog.bot_id.is_not(None),
            TurnLog.created_at >= since_cutoff,
        ]
        if user_id:
            conditions.append(TurnLog.user_id == user_id.strip().lower())
        if bot_ids:
            normalized = [b.strip().lower() for b in bot_ids if b and b.strip()]
            if normalized:
                conditions.append(TurnLog.bot_id.in_(normalized))

        # DISTINCT ON (bot_id) requires bot_id to be the FIRST column in
        # ORDER BY; the secondary key picks the row to keep per partition.
        statement = (
            select(TurnLog)
            .where(*conditions)
            .order_by(TurnLog.bot_id, TurnLog.created_at.desc())
            .distinct(TurnLog.bot_id)
        )

        with Session(self.engine) as session:
            rows = session.exec(statement).all()
        return list(rows)

    # ---- Tool call records (event-based persistence) ----

    def save_tool_call(
        self,
        *,
        turn_id: str | None,
        bot_id: str | None,
        user_id: str | None,
        call_id: str | None,
        tool_name: str,
        arguments: dict | None = None,
        result: str | None = None,
        iteration: int = 1,
        started_at: float | None = None,
        ended_at: float | None = None,
        text_offset: int | None = None,
        is_error: bool | None = None,
        tool_use_id: str | None = None,
        parent_tool_use_id: str | None = None,
    ) -> None:
        """Persist a single tool call record."""
        if self.engine is None:
            return
        store = ToolCallStore(self.engine)
        store.save_start(
            turn_id=turn_id,
            bot_id=bot_id,
            user_id=user_id,
            call_id=call_id,
            tool_name=tool_name,
            arguments=arguments,
            iteration=iteration,
            started_at=started_at,
            text_offset=text_offset,
            tool_use_id=tool_use_id,
            parent_tool_use_id=parent_tool_use_id,
        )
        if result is not None:
            from agent_bridge.tool_results import ToolResultPayload
            store.save_result(
                turn_id=turn_id,
                call_id=call_id,
                tool_use_id=tool_use_id,
                tool_name=tool_name,
                bot_id=bot_id,
                user_id=user_id,
                payload=ToolResultPayload.from_value(result),
                ended_at=ended_at,
                is_error=is_error,
                iteration=iteration,
                parent_tool_use_id=parent_tool_use_id,
            )

    def update_tool_call_result(
        self,
        *,
        call_id: str,
        result: str,
        ended_at: float | None = None,
        is_error: bool | None = None,
    ) -> None:
        """Compatibility delegate for legacy preview-only callers."""
        if self.engine is None or not call_id:
            return
        from agent_bridge.tool_results import ToolResultPayload
        ToolCallStore(self.engine).save_result(
            turn_id=None,
            call_id=call_id,
            tool_use_id=None,
            tool_name="unknown",
            bot_id=None,
            user_id=None,
            payload=ToolResultPayload.from_value(result),
            ended_at=ended_at,
            is_error=is_error,
        )

    # ---- Changed files (TASK-661) ----

    def _changed_files_store(self):
        from .changed_files_store import ChangedFilesStore
        return ChangedFilesStore(self.engine)

    def save_changed_files(
        self,
        *,
        turn_id: str,
        bot_id: str | None,
        user_id: str | None,
        trigger_message_id: str | None,
        files: list,
    ) -> int:
        """Persist a turn's changed files (list of ChangedFileInput). Returns count."""
        if self.engine is None:
            return 0
        return self._changed_files_store().save_turn_files(
            turn_id=turn_id,
            bot_id=bot_id,
            user_id=user_id,
            trigger_message_id=trigger_message_id,
            files=files,
        )

    def changed_files_summary(self, turn_id: str) -> dict:
        if self.engine is None:
            from .changed_files_store import build_turn_summary
            return build_turn_summary(turn_id, [])
        return self._changed_files_store().summary_for_turn(turn_id)

    def changed_files_summaries_for_triggers(self, message_ids: list[str]) -> dict:
        if self.engine is None:
            return {}
        return self._changed_files_store().summaries_for_triggers(message_ids)

    def changed_files_summaries_for_turns(self, turn_ids: list[str]) -> dict:
        if self.engine is None:
            return {}
        return self._changed_files_store().summaries_for_turns(turn_ids)

    def uncommitted_changed_files_summaries(
        self,
        *,
        session_id: str | None,
        bot_id: str,
        user_id: str,
        anchor_turn_id: str | None = None,
    ) -> tuple[str | None, dict, dict]:
        if self.engine is None:
            from .changed_files_store import build_uncommitted_summary
            empty = build_uncommitted_summary([])
            return None, empty, empty
        return self._changed_files_store().uncommitted_summaries_for_session(
            session_id=session_id,
            anchor_turn_id=anchor_turn_id,
            bot_id=bot_id,
            user_id=user_id,
        )

    def uncommitted_changed_files_summary(
        self,
        *,
        session_id: str | None,
        bot_id: str,
        user_id: str,
        anchor_turn_id: str | None = None,
    ) -> tuple[str | None, dict]:
        resolved, summary, _ = self.uncommitted_changed_files_summaries(
            session_id=session_id,
            anchor_turn_id=anchor_turn_id,
            bot_id=bot_id,
            user_id=user_id,
        )
        return resolved, summary

    def mark_changed_files_commit_requested(
        self,
        *,
        session_id: str | None,
        bot_id: str,
        user_id: str,
        turn_ids: list[str],
    ) -> tuple[str | None, int]:
        if self.engine is None:
            return None, 0
        return self._changed_files_store().mark_commit_requested(
            session_id=session_id,
            bot_id=bot_id,
            user_id=user_id,
            turn_ids=turn_ids,
        )

    def changed_file_content(self, *, turn_id: str, repo_key: str, path: str):
        if self.engine is None:
            return None
        return self._changed_files_store().get_file_content(
            turn_id=turn_id, repo_key=repo_key, path=path
        )

    def get_tool_calls_for_turn(self, turn_id: str) -> list[ToolCallRecord]:
        """Get all tool call records for a turn."""
        if self.engine is None:
            return []
        try:
            with Session(self.engine) as session:
                return list(session.exec(
                    select(ToolCallRecord)
                    .where(ToolCallRecord.turn_id == turn_id)
                    .order_by(ToolCallRecord.created_at)
                ).all())
        except Exception:
            return []

    def set_approval_status(
        self,
        tool_use_id: str,
        approval_request_id: str,
        approval_status: str,
    ) -> bool:
        """Stamp a tool_call_record with its approval resolution.

        Called when an approval request is resolved (approved/denied/cancelled/
        responded). The tool_use_id of the gated call == the approval request id,
        so we match on that. Returns True if a row was updated.
        """
        if self.engine is None or not tool_use_id:
            return False
        try:
            with Session(self.engine) as session:
                stmt = (
                    sa_text(
                        "UPDATE tool_call_records"
                        " SET approval_request_id = :req_id,"
                        "     approval_status = :status"
                        " WHERE tool_use_id = :tuid"
                    )
                )
                result = session.execute(stmt, {
                    "req_id": approval_request_id,
                    "status": approval_status,
                    "tuid": tool_use_id,
                })
                session.commit()
                return result.rowcount > 0  # type: ignore[union-attr]
        except Exception:
            logger.exception("Failed to set approval status for tool_use_id=%s", tool_use_id)
            return False

    def set_preapproved(self, tool_use_id: str) -> bool:
        """Mark a tool_call_record as pre-approved (ran with a live grant).

        Called from the tool_preapproved event handler so the gold "Approved"
        badge survives a page reload. Returns True if a row was updated.
        """
        if self.engine is None or not tool_use_id:
            return False
        try:
            with Session(self.engine) as session:
                result = session.execute(
                    sa_text(
                        "UPDATE tool_call_records SET preapproved = TRUE"
                        " WHERE tool_use_id = :tuid"
                    ),
                    {"tuid": tool_use_id},
                )
                session.commit()
                return result.rowcount > 0  # type: ignore[union-attr]
        except Exception:
            logger.exception("Failed to set preapproved for tool_use_id=%s", tool_use_id)
            return False
