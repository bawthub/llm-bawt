"""Bounded dedicated-run receipts on the next main SDK turn (TASK-1994).

Occurrences are the durable outbox; no extra delivery or model turn is started.
Receipts survive main-session rotation, and only real model output acknowledges
those appended to a turn. A failure before SDK consumption leaves them pending.
As with prompt-pin confirmation, a crash between consumption and acknowledgement
can replay a receipt, identified by its stable run id.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

from sqlalchemy import text, update
from sqlmodel import Session, select

from .prompt_schedule_models import PromptOccurrence, PromptSchedule
from .scheduler import JobRun

logger = logging.getLogger(__name__)
NOTICE_KEY = "agents.scheduled_run_notice"
MAX_NOTICES = 5
REPLY_CHARS = 1000


def completion_notice(occurrence, record, turn, state, now):
    """Snapshot only actual finished dedicated turns, never rejected admissions."""
    if (not occurrence.session_id or turn is None
            or getattr(turn, "ended_at", None) is None):
        return None
    snapshot = occurrence.snapshot_json
    reply = str(getattr(turn, "response_text", None) or "")
    return {
        "run_id": occurrence.run_id, "schedule_id": occurrence.job_id,
        "schedule": str(snapshot.get("name") or "")[:120],
        "tasks": list(dict.fromkeys(re.findall(r"\bTASK-\d+\b", snapshot.get("prompt", ""))))[:8],
        "prompt_excerpt": str(snapshot.get("prompt") or "")[:240],
        "scheduled_for": snapshot.get("scheduled_for"),
        "finished_at": (getattr(turn, "ended_at", None) or now).isoformat(),
        "state": state, "reply_excerpt": reply[:REPLY_CHARS],
        "reply_truncated": len(reply) > REPLY_CHARS,
        "error": str(getattr(turn, "error_text", None) or "")[:240],
        "user_message_id": record.user_message_id,
        "assistant_message_id": (getattr(turn, "assistant_message_id", None)
                                 or record.assistant_message_id),
    }


def upgrade_notice_columns(conn, dialect):
    """Add nullable state, without replaying old completed occurrences."""
    if dialect == "postgresql":
        conn.execute(text("ALTER TABLE prompt_occurrences ADD COLUMN IF NOT EXISTS main_notice_json JSON"))
        conn.execute(text("ALTER TABLE prompt_occurrences ADD COLUMN IF NOT EXISTS main_notice_turn_id VARCHAR"))


def render_notice(bot_id, notices, remaining):
    from ..prompt_registry import PromptResolver, get_prompt_resolver

    variables = {"runs": json.dumps(notices, ensure_ascii=False), "remaining": remaining}
    try:
        body = get_prompt_resolver().render(
            key=NOTICE_KEY, variables=variables, scope_type="bot", scope_id=bot_id,
        )
        if body.strip():
            return body
    except Exception:
        logger.warning("Scheduled run notice render failed; using registered default", exc_info=True)
    definition = PromptResolver.definition_for(NOTICE_KEY)
    if definition is None:
        raise RuntimeError(f"{NOTICE_KEY} is not registered")
    return definition.default_body().format(**variables)


def reconcile_finished(engine, service, bot_id, user_id):
    """Close the prompt-sweep race when the user asks immediately after output.

    Reuse canonical reconciliation, never infer success from the visible reply.
    Only already-terminal turns for this bot/owner are read, at most one notice
    batch; additional runs remain for the normal sweep or later main turns.
    """
    from .prompt_delivery import PromptDeliveryWorker

    with engine.connect() as conn:
        ids = conn.execute(text("""
            SELECT o.run_id FROM prompt_occurrences o
            JOIN job_runs r ON r.id=o.run_id
            JOIN prompt_schedules s ON s.job_id=o.job_id
            JOIN inter_bot_deliveries d ON d.id=o.delivery_id
            JOIN turn_logs t ON t.id=d.turn_id
            WHERE r.bot_id=:bot AND s.owner_user_id=:user
              AND o.state IN ('queued','running') AND t.ended_at IS NOT NULL
            ORDER BY o.updated_at, o.run_id LIMIT :batch
        """), {"bot": bot_id, "user": user_id, "batch": MAX_NOTICES}).scalars().all()
    worker = PromptDeliveryWorker(engine, service)
    for run_id in ids:
        worker._reconcile(run_id, datetime.now(timezone.utc))


@dataclass
class ScheduledRunNotices:
    engine: object
    turn_id: str
    run_ids: list[str] = field(default_factory=list)
    body: str | None = None

    @classmethod
    def prepare(cls, engine, *, bot_id, user_id, session_id, turn_id,
                service, request, user_prompt):
        """Select bounded pending receipts only for the canonical active thread."""
        from ..working_memory.turn import is_maintenance_command, scheduler_turn

        notice = cls(engine, turn_id)
        if (engine is None or not session_id or not bot_id or not user_id
                or is_maintenance_command(user_prompt)):
            return notice
        try:
            with engine.connect() as conn:
                active = conn.execute(text("""
                    SELECT id, session_metadata FROM sessions
                    WHERE bot_id=:bot AND user_id=:user AND status='active'
                      AND ended_at IS NULL
                    ORDER BY started_at DESC LIMIT 1
                """), {"bot": bot_id, "user": user_id}).mappings().first()
            if active is None or active["id"] != session_id:
                return notice
            metadata = active["session_metadata"] or {}
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            if (metadata.get("origin") == "automation"
                    or scheduler_turn(service, request) is not False):
                return notice
            reconcile_finished(engine, service, bot_id, user_id)
            with Session(engine) as session:
                pending = session.exec(
                    select(PromptOccurrence).join(JobRun, JobRun.id == PromptOccurrence.run_id)
                    .join(PromptSchedule, PromptSchedule.job_id == PromptOccurrence.job_id)
                    .where(JobRun.bot_id == bot_id, PromptSchedule.owner_user_id == user_id,
                           PromptOccurrence.main_notice_json.is_not(None),
                           PromptOccurrence.main_notice_turn_id.is_(None))
                    .order_by(PromptOccurrence.updated_at, PromptOccurrence.run_id)
                    .limit(MAX_NOTICES + 1)
                ).all()
                selected = pending[:MAX_NOTICES]
                if selected:
                    # A bounded lookahead reports deferred work without scanning it all.
                    remaining = "at least 1" if len(pending) > MAX_NOTICES else "0"
                    notice.body = render_notice(bot_id, [row.main_notice_json for row in selected], remaining)
                    notice.run_ids = [row.run_id for row in selected]
        except Exception:
            logger.warning("Scheduled run notice preparation failed; receipts remain pending", exc_info=True)
        return notice

    def apply(self, messages):
        from ..working_memory.turn import append_to_last_user

        return append_to_last_user(messages, self.body) if self.body else messages

    def observe(self, item):
        """Acknowledge only once output proves this turn consumed its prompt."""
        from .system_prompt_pin import proves_prompt_consumed

        if not self.run_ids or not proves_prompt_consumed(item):
            return
        try:
            with self.engine.begin() as conn:
                conn.execute(update(PromptOccurrence).where(
                    PromptOccurrence.run_id.in_(self.run_ids),
                    PromptOccurrence.main_notice_turn_id.is_(None),
                ).values(main_notice_turn_id=self.turn_id))
            self.run_ids = []
        except Exception:
            logger.warning("Scheduled run notice acknowledgement failed; retrying on output", exc_info=True)
