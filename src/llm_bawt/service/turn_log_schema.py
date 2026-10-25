"""The ``turn_logs`` table: row model, lifecycle helpers and schema bootstrap.

Split out of ``turn_logs.py`` (TASK-1056) so the store module holds only data
access. ``turn_logs`` re-exports every public name here, so existing
``from .turn_logs import TurnLog`` imports keep working.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import Boolean, Column, DateTime, String, Text, text as sa_text
from sqlmodel import Field, SQLModel, Session, select

logger = logging.getLogger("llm_bawt.service.turn_logs")


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
    # TASK-1232: the bridge backend the request was published to, persisted
    # with the ids above. Startup recovery's ``run.status`` asks only this
    # bridge; a row without it is never settled as absent.
    agent_backend: str | None = Field(default=None, sa_column=Column(String(64), nullable=True))
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


def bootstrap_turn_log_schema(conn) -> None:
    """Create turn_logs and apply additive column/index migrations.

    Also patches tool_call_records columns that predate ToolCallStore owning
    its own bootstrap. Run under SchemaBootstrapGuard by TurnLogStore.
    """
    turns_were_new = (
        getattr(getattr(conn, "dialect", None), "name", None) == "postgresql"
        and conn.execute(sa_text("SELECT to_regclass('public.turn_logs') IS NULL")).scalar_one()
    )
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
            "ALTER TABLE turn_logs ADD COLUMN IF NOT EXISTS agent_backend VARCHAR(64)"
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
            "ALTER TABLE tool_call_records ADD COLUMN IF NOT EXISTS tool_use_id VARCHAR"
        ))
        conn.execute(sa_text(
            "ALTER TABLE tool_call_records ADD COLUMN IF NOT EXISTS parent_tool_use_id VARCHAR"
        ))
        conn.execute(sa_text(
            "CREATE INDEX IF NOT EXISTS ix_tool_call_records_parent_tool_use_id"
            " ON tool_call_records (parent_tool_use_id)"
        ))
        # TASK-305: approval gate persistence — link tool calls to
        # their approval request so the approval card survives reload.
        conn.execute(sa_text(
            "ALTER TABLE tool_call_records ADD COLUMN IF NOT EXISTS approval_request_id VARCHAR"
        ))
        conn.execute(sa_text(
            "CREATE INDEX IF NOT EXISTS ix_tool_call_records_approval_request_id"
            " ON tool_call_records (approval_request_id)"
        ))
        conn.execute(sa_text(
            "ALTER TABLE tool_call_records ADD COLUMN IF NOT EXISTS approval_status VARCHAR"
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
        # ToolCallStore creates its table before this parent. Attach the
        # existing live SET NULL contract only on a freshly created parent;
        # an old missing FK needs an explicit reviewed migration, not startup
        # repair of potentially orphaned tool rows.
        if turns_were_new:
            conn.execute(sa_text("""
                ALTER TABLE public.tool_call_records ADD CONSTRAINT fk_toolcalls_turn
                FOREIGN KEY (turn_id) REFERENCES public.turn_logs(id)
                ON DELETE SET NULL NOT DEFERRABLE
            """))
        # TASK-1461: the API balance ledger/counter is written in update_turn's
        # transaction, so it must exist whenever turn_logs does.
        from .usage.balance import bootstrap_balance_schema
        bootstrap_balance_schema(conn)
    except Exception:
        logger.exception("Turn-log additive schema migration failed")
        raise


def backfill_trigger_message_ids(engine) -> None:
    """Populate trigger_message_id for up to 5000 rows that predate the column."""
    try:
        with Session(engine) as session:
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
