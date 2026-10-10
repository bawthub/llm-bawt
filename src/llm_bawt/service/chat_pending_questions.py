"""Canonical question/answer registry for interactive SDK tool calls.

AskUserQuestion persists a durable row for both delivery modes. Claude Code
holds a short ``awaiting_live`` window: the answer endpoint and bridge timeout
atomically claim opposite transitions. A winning live answer resumes the same
SDK turn; a timeout flips to ``awaiting`` and the answer arrives in a later
continuation turn. Wait=0 and other harnesses start directly in ``awaiting``.

An unanswered question survives the originating turn until answered, skipped,
or superseded. Rows are kept for audit/recall; callers filter by status.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import Column, DateTime, Text, update, text as sa_text
from sqlmodel import Field, SQLModel, Session, select

from ..utils.config import Config
from ..utils.schema import SchemaBootstrapGuard

logger = logging.getLogger(__name__)


class PendingQuestion(SQLModel, table=True):
    """Persistent record of an in-flight AskUserQuestion pause."""

    __tablename__ = "chat_pending_questions"

    # SDK-supplied tool_use id — unique across all sessions, doubles as the
    # PK so re-posting an event for the same tool_use_id is idempotent.
    tool_use_id: str = Field(primary_key=True, max_length=128)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False, index=True),
    )
    bot_id: str = Field(index=True, max_length=128)
    user_id: str = Field(index=True, max_length=128)
    turn_id: str = Field(index=True, max_length=128)
    trigger_message_id: str | None = Field(default=None, index=True, max_length=128)
    session_key: str | None = Field(default=None, max_length=128)
    tool_name: str = Field(default="AskUserQuestion", max_length=128)
    # Harness that asked it — "claude", "codex", … .  Drives per-harness answer
    # serialization on the continuation turn (TASK-269).
    origin_harness: str = Field(default="claude", max_length=32)
    # Original tool input (the AskUserQuestion `{questions: [...]}` payload).
    # Stored as JSON text so we don't need a JSONB column for portability.
    arguments_json: str = Field(sa_column=Column(Text, nullable=False))
    # awaiting_live → answered (same turn) | awaiting (timeout) | skipped.
    # awaiting → answered (continuation) | skipped | abandoned | superseded.
    # Both pending states stay visible until explicitly resolved.
    status: str = Field(default="awaiting", index=True, max_length=32)
    live_deadline: datetime | None = Field(default=None, sa_column=Column(DateTime(timezone=True), nullable=True))
    # Human-readable answer text (what gets fed to the continuation turn).
    answer: str | None = Field(default=None, sa_column=Column(Text, nullable=True))
    # Canonical structured answer: [{question_id, selected: [...], other?}].
    answer_json: str | None = Field(default=None, sa_column=Column(Text, nullable=True))
    # The continuation turn that carried this answer back to the agent.
    answered_turn_id: str | None = Field(default=None, max_length=128)
    answered_at: datetime | None = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), nullable=True),
    )


class PendingQuestionStore:
    """DB access helper for ``chat_pending_questions``.

    Mirrors the patterns of :class:`TurnLogStore` — additive ALTERs on init,
    shared SQLAlchemy engine, graceful degradation when Postgres is missing.
    """

    _schema_guard = SchemaBootstrapGuard()

    def __init__(self, config: Config) -> None:
        self.config = config
        self.engine = None
        try:
            from ..utils.db import get_shared_engine
            self.engine = get_shared_engine(config)
            if self.engine is None:
                return
            self._ensure_tables_exist()
        except Exception as e:
            self.engine = None
            logger.warning("Pending-question store unavailable: %s", e)

    def _ensure_tables_exist(self) -> None:
        if self.engine is None:
            return

        def bootstrap(conn) -> None:
            SQLModel.metadata.create_all(
                bind=conn, tables=[PendingQuestion.__table__],
            )
            # Additive index + TASK-269 canonical columns — safe to re-apply.
            conn.execute(sa_text(
                "CREATE INDEX IF NOT EXISTS ix_pending_questions_bot_status_created"
                " ON chat_pending_questions (bot_id, user_id, status, created_at DESC)"
            ))
            conn.execute(sa_text(
                "ALTER TABLE chat_pending_questions ADD COLUMN IF NOT EXISTS"
                " origin_harness VARCHAR(32) DEFAULT 'claude'"
            ))
            conn.execute(sa_text(
                "ALTER TABLE chat_pending_questions ADD COLUMN IF NOT EXISTS answer_json TEXT"
            ))
            conn.execute(sa_text(
                "ALTER TABLE chat_pending_questions ADD COLUMN IF NOT EXISTS"
                " answered_turn_id VARCHAR(128)"
            ))
            conn.execute(sa_text(
                "ALTER TABLE chat_pending_questions ADD COLUMN IF NOT EXISTS"
                " live_deadline TIMESTAMPTZ"
            ))

        self._schema_guard.run(self.engine, "pending-question-store", bootstrap)

    # ----- writes -----

    def upsert_awaiting(
        self,
        *,
        tool_use_id: str,
        bot_id: str,
        user_id: str,
        turn_id: str,
        arguments: dict,
        tool_name: str = "AskUserQuestion",
        trigger_message_id: str | None = None,
        session_key: str | None = None,
        origin_harness: str = "claude",
        live_deadline: datetime | None = None,
    ) -> None:
        """Insert (or no-op-replay) a fresh awaiting question.

        Called from the chat_streaming layer when it sees the bridge emit an
        ``AWAIT_TOOL_RESULT`` AgentEvent.  Idempotent on tool_use_id — a
        duplicate event (Redis replay, multi-window race) does not advance
        the row state, even if it was already answered.
        """
        if self.engine is None:
            return
        try:
            with Session(self.engine) as session:
                existing = session.get(PendingQuestion, tool_use_id)
                if existing is not None:
                    # Already recorded — don't clobber the answer if the row
                    # advanced past awaiting.
                    return
                row = PendingQuestion(
                    tool_use_id=tool_use_id,
                    bot_id=(bot_id or "").strip() or "unknown",
                    user_id=(user_id or "").strip() or "unknown",
                    turn_id=(turn_id or "").strip() or "unknown",
                    trigger_message_id=(trigger_message_id or None) or None,
                    session_key=(session_key or None) or None,
                    tool_name=tool_name or "AskUserQuestion",
                    origin_harness=(origin_harness or "claude").strip() or "claude",
                    status="awaiting_live" if live_deadline else "awaiting",
                    live_deadline=live_deadline,
                    arguments_json=json.dumps(
                        arguments if isinstance(arguments, dict) else {"value": arguments},
                        ensure_ascii=False, default=str,
                    ),
                )
                session.add(row)
                session.commit()
        except Exception:
            logger.exception(
                "Failed to upsert pending question tool_use_id=%s", tool_use_id,
            )

    def _transition(self, where: Any, from_status: str, values: dict[str, Any]) -> int:
        """Atomically change matching rows only while their status is unchanged."""
        if self.engine is None:
            return 0
        with Session(self.engine) as session:
            result = session.execute(
                update(PendingQuestion)
                .where(where, PendingQuestion.status == from_status)
                .values(**values)
            )
            session.commit()
            return result.rowcount

    @staticmethod
    def _answer_values(answer: str, answer_json: list | dict | None) -> dict[str, Any]:
        return {
            "status": "answered",
            "answer": answer,
            "answer_json": json.dumps(answer_json, ensure_ascii=False, default=str)
                if answer_json is not None else None,
            "answered_at": datetime.now(timezone.utc),
        }

    def expire_live(self, tool_use_id: str) -> bool:
        """CAS: timeout owns deferral only if the live answer has not won."""
        return self._transition(
            PendingQuestion.tool_use_id == tool_use_id,
            "awaiting_live", {"status": "awaiting", "live_deadline": None},
        ) == 1

    def expire_live_for_turn(self, turn_id: str) -> int:
        """Abort moves any still-live question into ordinary deferred mode."""
        return self._transition(
            PendingQuestion.turn_id == turn_id,
            "awaiting_live", {"status": "awaiting", "live_deadline": None},
        )

    def claim_live_answer(
        self, tool_use_id: str, answer: str, answer_json: list | dict | None,
    ) -> bool:
        """CAS: answer owns the live Future only while its deadline is valid."""
        return self._transition(
            (PendingQuestion.tool_use_id == tool_use_id)
            & (PendingQuestion.live_deadline > datetime.now(timezone.utc)),
            "awaiting_live", self._answer_values(answer, answer_json),
        ) == 1

    def claim_deferred_answer(
        self, tool_use_id: str, answer: str, answer_json: list | dict | None,
    ) -> bool:
        """CAS the deferred answer too; losing tabs must not send another turn."""
        return self._transition(
            PendingQuestion.tool_use_id == tool_use_id,
            "awaiting", self._answer_values(answer, answer_json),
        ) == 1

    def dismiss_deferred(self, tool_use_id: str) -> bool:
        return self._transition(
            PendingQuestion.tool_use_id == tool_use_id,
            "awaiting", {"status": "skipped", "answered_at": datetime.now(timezone.utc)},
        ) == 1

    def dismiss_live(self, tool_use_id: str) -> bool:
        return self._transition(
            PendingQuestion.tool_use_id == tool_use_id,
            "awaiting_live", {"status": "skipped", "answered_at": datetime.now(timezone.utc)},
        ) == 1

    def mark(
        self,
        *,
        tool_use_id: str,
        status: str,
        answer: str | None = None,
    ) -> PendingQuestion | None:
        """Transition a row to a terminal state.

        ``status`` must be one of: ``answered``, ``skipped``, ``abandoned``.
        Returns the updated row (or None if not found).  No-ops if the row
        is already in a terminal state — the first answer wins.
        """
        if status not in {"answered", "skipped", "abandoned"}:
            raise ValueError(f"invalid pending-question status: {status}")
        try:
            values = {"status": status, "answered_at": datetime.now(timezone.utc)}
            if answer is not None:
                values["answer"] = answer
            self._transition(PendingQuestion.tool_use_id == tool_use_id, "awaiting", values)
            return self.get(tool_use_id)
        except Exception:
            logger.exception(
                "Failed to mark pending question tool_use_id=%s status=%s",
                tool_use_id, status,
            )
            return None

    def record_answer(
        self,
        *,
        tool_use_id: str,
        answer: str,
        answer_json: list | dict | None = None,
        answered_turn_id: str | None = None,
    ) -> PendingQuestion | None:
        """Record the user's answer and flip the question to ``answered``.

        Canonical write path for the deferred/continuation model (TASK-269):
        the answer endpoint persists the answer here, then the frontend
        dispatches a continuation turn carrying ``answer`` back to the agent.
        Idempotent — the first answer wins; a re-POST returns the existing row
        unchanged.  Returns the updated row, or None if the question is missing.
        """
        try:
            values = self._answer_values(answer, answer_json)
            if answered_turn_id:
                values["answered_turn_id"] = answered_turn_id
            self._transition(PendingQuestion.tool_use_id == tool_use_id, "awaiting", values)
            return self.get(tool_use_id)
        except Exception:
            logger.exception("Failed to record answer for tool_use_id=%s", tool_use_id)
            return None

    def supersede_awaiting_for_turn(self, turn_id: str, *, keep: str | None = None) -> int:
        """Mark stale awaiting questions for a turn as ``superseded``.

        Used when a single turn asks more than once (rare) or when a new
        question for the same turn replaces an earlier unanswered one.  Pass
        ``keep`` to exempt the current tool_use_id.  Returns rows changed.
        """
        if self.engine is None or not turn_id:
            return 0
        try:
            with Session(self.engine) as session:
                rows = session.exec(
                    select(PendingQuestion)
                    .where(PendingQuestion.turn_id == turn_id)
                    .where(PendingQuestion.status == "awaiting")
                ).all()
                count = 0
                now = datetime.now(timezone.utc)
                for row in rows:
                    if keep and row.tool_use_id == keep:
                        continue
                    row.status = "superseded"
                    row.answered_at = now
                    session.add(row)
                    count += 1
                if count:
                    session.commit()
                return count
        except Exception:
            logger.exception("Failed to supersede questions for turn_id=%s", turn_id)
            return 0

    def set_answered_turn(self, tool_use_id: str, answered_turn_id: str) -> None:
        """Link the continuation turn that carried this answer back.

        The answer endpoint records the answer before the continuation turn
        exists; the continuation turn (which knows answered_question_id) calls
        this to backfill the link.  Best-effort, idempotent.
        """
        if self.engine is None or not tool_use_id or not answered_turn_id:
            return
        try:
            with Session(self.engine) as session:
                row = session.get(PendingQuestion, tool_use_id)
                if row is None or row.answered_turn_id:
                    return
                row.answered_turn_id = answered_turn_id
                session.add(row)
                session.commit()
        except Exception:
            logger.exception(
                "Failed to link answered turn %s for question %s",
                answered_turn_id, tool_use_id,
            )

    def get_by_turn(self, turn_id: str) -> PendingQuestion | None:
        """Return the (most recent) question asked on a turn, any status.

        The UI renders a first-class QuestionMessage off the turn that ended
        with end_reason="question"; this resolves the turn back to its row.
        """
        if self.engine is None or not turn_id:
            return None
        try:
            with Session(self.engine) as session:
                stmt = (
                    select(PendingQuestion)
                    .where(PendingQuestion.turn_id == turn_id)
                    .order_by(PendingQuestion.created_at.desc())
                    .limit(1)
                )
                return session.exec(stmt).first()
        except Exception:
            logger.exception("Failed to fetch question for turn_id=%s", turn_id)
            return None

    def abandon_for_turn(self, turn_id: str) -> int:
        """Mark every awaiting question for a turn as abandoned.

        Called from the turn-completion path when the originating turn ends
        without an answer (timeout, abort, model bailed out, bridge crashed
        and the app noticed via the turn-log status flip).  Returns the
        number of rows changed.
        """
        if self.engine is None or not turn_id:
            return 0
        try:
            with Session(self.engine) as session:
                rows = session.exec(
                    select(PendingQuestion)
                    .where(PendingQuestion.turn_id == turn_id)
                    .where(PendingQuestion.status == "awaiting")
                ).all()
                count = 0
                now = datetime.now(timezone.utc)
                for row in rows:
                    row.status = "abandoned"
                    row.answered_at = now
                    session.add(row)
                    count += 1
                if count:
                    session.commit()
                return count
        except Exception:
            logger.exception("Failed to abandon questions for turn_id=%s", turn_id)
            return 0

    # ----- reads -----

    def get(self, tool_use_id: str) -> PendingQuestion | None:
        if self.engine is None:
            return None
        try:
            with Session(self.engine) as session:
                return session.get(PendingQuestion, tool_use_id)
        except Exception:
            logger.exception("Failed to fetch pending question tool_use_id=%s", tool_use_id)
            return None

    def list_awaiting(
        self,
        *,
        bot_id: str | None = None,
        user_id: str | None = None,
        limit: int = 50,
    ) -> list[PendingQuestion]:
        """Return all currently-awaiting questions for a bot/user scope.

        Used by ``GET /v1/chat/pending-questions`` for UI hydration on page
        load and by other tabs that missed the live AWAIT_TOOL_RESULT event.
        """
        if self.engine is None:
            return []
        try:
            with Session(self.engine) as session:
                stmt = select(PendingQuestion).where(
                    PendingQuestion.status.in_(("awaiting", "awaiting_live"))
                )
                if bot_id:
                    stmt = stmt.where(PendingQuestion.bot_id == bot_id.strip())
                if user_id:
                    stmt = stmt.where(PendingQuestion.user_id == user_id.strip())
                stmt = stmt.order_by(PendingQuestion.created_at.desc()).limit(limit)
                return list(session.exec(stmt).all())
        except Exception:
            logger.exception(
                "Failed to list awaiting questions bot=%s user=%s",
                bot_id, user_id,
            )
            return []

    def list_recent(
        self,
        *,
        bot_id: str | None = None,
        user_id: str | None = None,
        limit: int = 100,
    ) -> list[PendingQuestion]:
        """Return recent questions for a scope REGARDLESS of status.

        Used by the UI to render resolved (answered/skipped) questions as a
        read-only record in the transcript — distinguishing a genuinely
        resolved question from one whose awaiting row simply hasn't been
        hydrated yet (which must NOT render read-only).
        """
        if self.engine is None:
            return []
        try:
            with Session(self.engine) as session:
                stmt = select(PendingQuestion)
                if bot_id:
                    stmt = stmt.where(PendingQuestion.bot_id == bot_id.strip())
                if user_id:
                    stmt = stmt.where(PendingQuestion.user_id == user_id.strip())
                stmt = stmt.order_by(PendingQuestion.created_at.desc()).limit(limit)
                return list(session.exec(stmt).all())
        except Exception:
            logger.exception(
                "Failed to list recent questions bot=%s user=%s",
                bot_id, user_id,
            )
            return []

    @staticmethod
    def tool_answers(row: PendingQuestion) -> dict[str, str]:
        """Render an answered row as the SDK tool's native ``answers`` input.

        Claude Code's AskUserQuestion takes ``answers`` keyed by question text
        (multi-select comma-joined) and returns a normal, non-error tool result.
        Single source for both live delivery and timeout arbitration.
        """
        try:
            questions = (json.loads(row.arguments_json or "{}") or {}).get("questions") or []
        except Exception:
            questions = []
        try:
            responses = json.loads(row.answer_json) if row.answer_json else []
        except Exception:
            responses = []
        if not isinstance(responses, list):
            responses = []
        by_id = {
            str(r.get("question_id")): r
            for r in responses if isinstance(r, dict) and r.get("question_id")
        }
        answers: dict[str, str] = {}
        for index, question in enumerate(questions):
            text = question.get("question") if isinstance(question, dict) else None
            if not text:
                continue
            response = by_id.get(str(question.get("header") or "")) or by_id.get(text)
            if response is None and index < len(responses) and isinstance(responses[index], dict):
                response = responses[index]
            if not response:
                continue
            parts = [s for s in (response.get("selected") or []) if s]
            if response.get("other"):
                parts.append(response["other"])
            if parts:
                answers[text] = ", ".join(parts)
        # Pre-formatted/free-text answers carry no per-question structure.
        if not answers and row.answer and questions and isinstance(questions[0], dict):
            first = questions[0].get("question")
            if first:
                answers[first] = row.answer
        return answers

    @staticmethod
    def row_to_dict(row: PendingQuestion) -> dict:
        """Serialise a row to the wire shape the UI consumes.

        Mirrors the per-stream ``tool.await_result`` SSE chunk so the same
        client-side store reducer can ingest either source.
        """
        try:
            arguments = json.loads(row.arguments_json) if row.arguments_json else {}
        except Exception:
            arguments = {}
        answer_json = None
        if getattr(row, "answer_json", None):
            try:
                answer_json = json.loads(row.answer_json)
            except Exception:
                answer_json = None
        return {
            "tool_use_id": row.tool_use_id,
            "tool_name": row.tool_name,
            "origin_harness": getattr(row, "origin_harness", "claude"),
            "bot_id": row.bot_id,
            "user_id": row.user_id,
            "turn_id": row.turn_id,
            "trigger_message_id": row.trigger_message_id,
            "session_key": row.session_key,
            "arguments": arguments,
            "status": row.status,
            "live_deadline": row.live_deadline.replace(
                tzinfo=row.live_deadline.tzinfo or timezone.utc,
            ).isoformat() if row.live_deadline and row.status == "awaiting_live" else None,
            "answer": row.answer,
            "answer_json": answer_json,
            "answered_turn_id": getattr(row, "answered_turn_id", None),
            "created_at": row.created_at.isoformat() if row.created_at else None,
            "answered_at": row.answered_at.isoformat() if row.answered_at else None,
        }
