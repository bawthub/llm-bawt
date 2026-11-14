"""Durable prompt outbox adapter. Scheduling never executes model inference."""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone

from sqlalchemy import text
from sqlmodel import Session, select

from ..inter_bot_delivery import DeliveryRecord
from ..inter_bot_delivery_ledger import DeliveryTransitionEvents, scheduler_sender_id
from .prompt_capabilities import resolve_prompt_target
from .prompt_delivery_state import gate_reason, receipt, schedule_author, transition
from .prompt_schedule_models import NONTERMINAL, PromptOccurrence, PromptSchedule
from .prompt_schedule_store import PromptScheduleStore
from .scheduler import ScheduledJob, JobRun
from .wakeup_staleness import skipped_notice, wakeup_stale_reason

logger = logging.getLogger(__name__)


class PromptDeliveryWorker:
    def __init__(self, engine, service):
        self.engine = engine
        self.service = service
        self.store = PromptScheduleStore(engine)
        self._last_retention = None

    async def sweep(self):
        dispatcher = getattr(self.service, "_inter_bot_dispatcher", None)
        if dispatcher is None or dispatcher.store.engine is None:
            return
        # All DB/queue work is bounded and off the event loop; inference is
        # exclusively the existing dispatcher's independently-owned turn task.
        records = await asyncio.to_thread(self._sweep, dispatcher)
        for record in records:
            try:
                await dispatcher._emit(record)
            except Exception:
                logger.warning("Prompt delivery event publication failed for %s", record.id, exc_info=True)
            finally:
                dispatcher.wake(record.target_bot_id)

    def _sweep(self, dispatcher):
        now = datetime.now(timezone.utc)
        records = []
        self._dispatcher = dispatcher
        self._notices = []
        if self._last_retention is None or (now - self._last_retention).total_seconds() >= 3600:
            from .prompt_retention import prune_prompt_runs
            prune_prompt_runs(self.engine, now)
            self._last_retention = now
        with Session(self.engine) as session:
            runs = session.exec(select(PromptOccurrence.run_id).where(
                PromptOccurrence.state.in_(NONTERMINAL)
            ).order_by(PromptOccurrence.updated_at).limit(100)).all()
        for run_id in runs:
            try:
                record = self._reconcile(run_id, now)
                if record:
                    records.append(record)
            except Exception:
                logger.exception("Prompt reconciliation failed for %s", run_id)
        self.store.reserve_due(now)
        with Session(self.engine) as session:
            pending = session.exec(select(PromptOccurrence).where(
                PromptOccurrence.state == "pending"
            ).order_by(PromptOccurrence.updated_at, PromptOccurrence.run_id).limit(100)).all()
        for occurrence in pending:
            try:
                # Task state is remote: read it before taking any outbox lock.
                stale = (wakeup_stale_reason(occurrence.snapshot_json or {}, occurrence.kind)
                         if occurrence.enqueue_attempts == 0 else None)
                prepared = self._prepare(occurrence.run_id, now, stale=stale)
                if prepared is None:
                    continue
                record = self._enqueue(occurrence.run_id, prepared, dispatcher, now)
                if record:
                    records.append(record)
            except Exception as exc:
                logger.exception("Prompt outbox enqueue failed for %s", occurrence.run_id)
                self._enqueue_error(occurrence.run_id, str(exc), now)
        return records + self._notices

    def _enqueue(self, run_id, prepared, dispatcher, now):
        snapshot, session_id, model, job_id, key = prepared
        sender = scheduler_sender_id(snapshot["owner_user_id"])
        author = schedule_author(snapshot)
        idem = f"schedule:{job_id}:occurrence:{key}"
        # Serialize outbox attempts with pause/cancel and other workers.
        # Dispatch takes target -> job -> delivery; this transaction now takes
        # target before the job row and shares its connection with enqueue.
        with Session(self.engine) as session, session.begin():
            from .turn_admission import lock_target

            # claim_next takes target -> schedule row. Match that order before
            # _rows locks the schedule; enqueue joins this same transaction.
            lock_target(session.connection(), snapshot["bot_id"])
            from .system_gate_admission import schedule_held

            if schedule_held(session.connection(), snapshot["bot_id"]):
                # A live system gate: a held no-op. The occurrence stays pending
                # (not an enqueue error) and the next sweep after release admits it.
                return None
            rows = self._rows(session, run_id)
            if not rows or rows[2].state != "pending":
                return None
            found = session.connection().execute(text("""
                SELECT id FROM inter_bot_deliveries
                WHERE sender_bot_id=:sender AND target_bot_id=:target
                  AND author_entity_type=:author_type AND author_entity_id=:author_id
                  AND idempotency_key=:key
            """), {"sender": sender, "target": snapshot["bot_id"],
                   "author_type": author.entity_type, "author_id": author.entity_id,
                   "key": idem}).scalar_one_or_none()
            if found:
                record = self._locked_delivery(session, found)
            else:
                if snapshot.get("target_mode") == "auto":
                    from .message_routing import route_opted_in

                    if (not route_opted_in(session.connection(), snapshot["owner_user_id"], snapshot["bot_id"])
                            or rows[2].snapshot_json.get("bot_id") != snapshot["bot_id"]):
                        transition(session, *rows, "cancelled", now,
                                   error="Automatic routing opt-in was revoked or selection changed")
                        return None
                reason = gate_reason(*rows, now)
                if reason:
                    transition(session, *rows, reason[0], now, error=reason[1])
                    return None
                if not model or not session_id:
                    transition(session, *rows, "failed", now,
                               error="Prepared occurrence lost its bound model or session")
                    return None
                record, _ = dispatcher.store.enqueue(
                    sender_bot_id=sender,
                    target_bot_id=snapshot["bot_id"], message=snapshot["prompt"],
                    author=author,
                    idempotency_key=idem,
                    task_id=snapshot.get("task_id"),
                    session_policy="continue",
                    payload={
                        "messages": [{"role": "user", "content": snapshot["prompt"]}],
                        "user": snapshot["owner_user_id"], "bot_id": snapshot["bot_id"],
                        "model": model, "session_id": session_id, "prefer_steer": False,
                        "augment_memory": snapshot["augment_memory"],
                        "extract_memory": snapshot["extract_memory"], "stream": True,
                    },
                    metadata={"prompt_schedule": {
                        "schedule_id": job_id, "occurrence_id": run_id,
                        "scheduled_for": snapshot["scheduled_for"],
                        # The audience user; a bot-created run's author is the bot.
                        "owner_user_id": snapshot["owner_user_id"],
                    }},
                    producer="schedule", connection=session.connection(),
                    # schedule_held above already waited out any live gate; hold
                    # keeps this enqueue from re-asking the same question.
                    on_gate="hold",
                )
            rows[2].delivery_id = record.id
            transition(session, *rows, "queued", now, evidence=self._evidence(record))
            return record

    def _notify_skipped(self, session, snapshot, run_id, reason):
        """Tell the creator a stale wake-up was skipped (TASK-2294).

        Enqueued in the skip's own transaction with a per-occurrence key, so the
        skip and its notice commit together and a retry cannot send it twice.
        """
        from ..bot_identities import SYSTEM_BOT_ID

        creator = snapshot.get("creator_bot_id")
        dispatcher = getattr(self, "_dispatcher", None)
        if not creator or dispatcher is None:
            return
        message = skipped_notice(snapshot, run_id, reason)
        record, _ = dispatcher.store.enqueue(
            sender_bot_id=SYSTEM_BOT_ID, target_bot_id=creator, message=message,
            idempotency_key=f"wakeup-skipped:{run_id}", message_kind="WAKEUP_SKIPPED",
            task_id=snapshot.get("task_id"),
            # A system notice: held behind a live gate, never refused by one.
            on_gate="hold",
            metadata={"prompt_schedule_skipped": {"occurrence_id": run_id}},
            payload={
                "bot_id": creator, "user": snapshot["owner_user_id"], "session_id": None,
                "messages": [{"role": "user", "content": message}],
                "prefer_steer": False, "reply_mode": "async",
                "extract_memory": False, "augment_memory": False, "stream": True,
            },
            connection=session.connection(),
        )
        self._notices.append(record)

    def _rows(self, session, run_id):
        job_id = session.exec(select(PromptOccurrence.job_id).where(
            PromptOccurrence.run_id == run_id)).first()
        if job_id is None:
            return None
        job = session.exec(select(ScheduledJob).where(
            ScheduledJob.id == job_id).with_for_update()).one()
        return job, session.get(PromptSchedule, job_id), session.get(PromptOccurrence, run_id), session.get(JobRun, run_id)

    @staticmethod
    def _session_for_scope(conn, schedule, occurrence, snapshot, scope):
        """Choose under the target lock; never move or revive a user's thread."""
        default_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"llm-bawt:prompt:{scope}"))
        if occurrence.session_id:
            return occurrence.session_id, None
        if snapshot["clear_context"]:
            return default_id, None

        def owned(row):
            if row is None:
                return False
            meta = row["session_metadata"] or {}
            if isinstance(meta, str):
                meta = json.loads(meta)
            return (row["bot_id"] == snapshot["bot_id"]
                    and row["user_id"] == snapshot["owner_user_id"]
                    and meta.get("origin") == "automation"
                    and meta.get("schedule_id") == schedule.job_id
                    and (meta.get("prompt_scope") == scope or row["id"] == default_id))

        def read(session_id):
            return conn.execute(text("""
                SELECT id, bot_id, user_id, status, session_metadata
                FROM sessions WHERE id=:id
            """), {"id": session_id}).mappings().one_or_none()

        row = read(schedule.dedicated_session_id) if schedule.dedicated_session_id else None
        if not owned(row):
            row = read(default_id)
        if row is None:
            return default_id, None
        if owned(row) and row["status"] == "archived":
            return row["id"], None
        reason = "Persistent schedule context was replaced because its previous thread is no longer an available automation thread."
        return str(uuid.uuid4()), reason

    def _prepare(self, run_id, now, stale=None):
        from ..memory.postgresql_short_term import PostgreSQLShortTermManager

        with Session(self.engine) as session, session.begin():
            preflight = session.exec(select(PromptOccurrence.snapshot_json).where(
                PromptOccurrence.run_id == run_id,
            )).first()
            if preflight and preflight.get("target_mode") == "auto":
                from .message_routing import lock_routing_pool

                lock_routing_pool(session.connection(), preflight["owner_user_id"])
            rows = self._rows(session, run_id)
            if not rows:
                return None
            job, schedule, occurrence, run = rows
            if occurrence.state != "pending":
                return None
            snapshot = dict(occurrence.snapshot_json)
            result = receipt(run)
            # Automatic targeting selects once per occurrence, never once per
            # retry. The snapshot is the owner-scoped, durable selected identity;
            # different occurrences may resolve to different bots.
            if snapshot.get("target_mode") == "auto" and occurrence.enqueue_attempts == 0:
                from .message_routing import MessageRouter, RouteRequest

                preview = MessageRouter(self.service).preview(RouteRequest(
                    user_id=snapshot["owner_user_id"],
                    candidate_bot_ids=snapshot.get("candidate_bot_ids"),
                    requested_model=snapshot.get("model"), scope="schedule",
                ))
                if not preview["selected_bot_id"]:
                    transition(session, *rows, "failed", now,
                               error="No opted-in compatible bots for automatic occurrence")
                    return None
                snapshot["bot_id"] = preview["selected_bot_id"]
                snapshot["selection_reason"] = preview["reason"]
                chosen = next(row for row in preview["candidates"] if row["bot_id"] == snapshot["bot_id"])
                snapshot["ranking"] = MessageRouter.ranking_evidence(chosen, preview["candidates"])
                occurrence.snapshot_json = snapshot
                run.bot_id = snapshot["bot_id"]
                session.add(occurrence)
                session.add(run)
            # Once enqueue may have happened, recover by key before consulting
            # mutable capability/lifecycle. An accepted delivery must not be lost.
            if occurrence.enqueue_attempts == 0:
                gated = gate_reason(*rows, now)
                reason = gated or (("skipped", stale) if stale else None)
                if reason:
                    transition(session, *rows, reason[0], now, error=reason[1])
                    if not gated:
                        self._notify_skipped(session, snapshot, run_id, stale)
                    return None
                try:
                    model = resolve_prompt_target(self.service, snapshot["bot_id"],
                                                  snapshot.get("model"), snapshot["owner_user_id"])
                    scope = (f"occurrence:{run_id}" if snapshot["clear_context"] else
                             f"schedule:{job.id}:bot:{snapshot['bot_id']}:owner:{snapshot['owner_user_id']}"
                             + (f":model:{model}" if snapshot.get("target_mode") == "auto" else ""))
                    from .turn_admission import lock_target

                    lock_target(session.connection(), snapshot["bot_id"])
                    session_id, replacement_reason = self._session_for_scope(
                        session.connection(), schedule, occurrence, snapshot, scope,
                    )
                    if replacement_reason:
                        snapshot["context_reset_reason"] = replacement_reason
                        occurrence.snapshot_json = snapshot
                        result["context_reset_reason"] = replacement_reason
                    PostgreSQLShortTermManager.create_inactive_session_row(
                        session.connection(), session_id=session_id,
                        bot_id=snapshot["bot_id"], user_id=snapshot["owner_user_id"],
                        session_metadata={"created_inactive": True, "origin": "automation",
                                          "schedule_id": job.id, "title": snapshot["name"],
                                          "title_source": "automation", "prompt_scope": scope},
                    )
                except ValueError as exc:
                    transition(session, *rows, "failed", now, error=str(exc))
                    return None
                occurrence.session_id = session_id
                if not snapshot["clear_context"]:
                    schedule.dedicated_session_id = session_id
                    session.add(schedule)
                result["actual_model"] = model
                run.result_json = json.dumps(result)
                session.add(run)
            occurrence.enqueue_attempts += 1
            occurrence.updated_at = now
            session.add(occurrence)
            return dict(snapshot), occurrence.session_id, result.get("actual_model"), job.id, occurrence.occurrence_key

    def _enqueue_error(self, run_id, error, now):
        with Session(self.engine) as session, session.begin():
            rows = self._rows(session, run_id)
            if rows and rows[2].state == "pending":
                rows[2].last_error = error[:4000]
                rows[2].updated_at = now
                session.add(rows[2])

    @staticmethod
    def _evidence(record):
        return {"delivery_id": record.id, "turn_id": record.turn_id,
                "user_message_id": record.user_message_id,
                "delivery_status": record.status,
                "response_model": record.response_model}

    def _locked_delivery(self, session, delivery_id):
        query = "SELECT * FROM inter_bot_deliveries WHERE id=:id"
        if self.engine.dialect.name == "postgresql":
            query += " FOR UPDATE"
        row = session.connection().execute(text(query), {"id": delivery_id}).mappings().first()
        return DeliveryRecord.from_mapping(row) if row else None

    def _reconcile(self, run_id, now):
        with Session(self.engine) as session, session.begin():
            rows = self._rows(session, run_id)
            if not rows:
                return None
            job, schedule, occurrence, run = rows
            if occurrence.state not in NONTERMINAL or not occurrence.delivery_id:
                return None
            record = self._locked_delivery(session, occurrence.delivery_id)
            if record is None:
                transition(session, *rows, "unknown", now, error="Persisted delivery receipt is missing; not replayed")
                return None
            state, error = self._delivery_state(record)
            if record.status == "QUEUED" and record.transport_accepted_at is None:
                reason = gate_reason(*rows, now)
                if reason:
                    conn = session.connection()
                    DeliveryTransitionEvents.stage(conn, conn.execute(text("""
                        UPDATE inter_bot_deliveries SET status='CANCELLED',
                            last_error=:error, updated_at=CURRENT_TIMESTAMP,
                            next_retry_at=NULL
                        WHERE id=:id AND status='QUEUED' AND transport_accepted_at IS NULL
                        RETURNING *
                    """), {"id": record.id, "error": reason[1]}).mappings().first(),
                        reason_code="schedule_gate")
                    session.connection().execute(text(
                        "DELETE FROM turn_logs WHERE id=:id AND status='reserved'"
                    ), {"id": record.turn_id})
                    state, error = reason
                    record = self._locked_delivery(session, record.id)
            if state not in NONTERMINAL:
                from .scheduled_run_notice import completion_notice

                try:
                    occurrence.main_notice_json = completion_notice(
                        occurrence, record, self.service._turn_log_store.get_turn(record.turn_id), state, now,
                    )
                except Exception:
                    # An optional context receipt must never prevent the
                    # canonical occurrence/run from reaching its terminal state.
                    logger.warning("Completion notice failed for run %s", run_id, exc_info=True)
            transition(session, *rows, state, now, error=error, evidence=self._evidence(record))
            return record

    def _delivery_state(self, record):
        # Delivery success alone is not inference success. Consult the canonical
        # terminal turn; ambiguous accepted recovery stays with the transport.
        turn = self.service._turn_log_store.get_turn(record.turn_id)
        if record.transport_accepted_at is not None and (
                "ambiguous" in (record.last_error or "").lower()
                or (record.status == "FAILED" and not getattr(turn, "model", None))):
            return "unknown", record.last_error or "Accepted outcome is uncertain; not replayed"
        if turn is not None and turn.ended_at is not None:
            status = str(turn.status).lower()
            if status in {"ok", "completed"} and not getattr(turn, "error_text", None):
                return "succeeded", None
            if record.status in {"DELIVERED", "FAILED", "CANCELLED"}:
                return "failed", getattr(turn, "error_text", None) or f"Turn ended: {status}"
        if record.status in {"STEERING", "DISPATCHING"} or (
                record.status == "QUEUED" and record.transport_accepted_at is not None):
            return "running", record.last_error
        if record.status == "QUEUED":
            return "queued", record.last_error
        if record.status == "CANCELLED" and record.transport_accepted_at is None:
            return "cancelled", record.last_error or "Delivery cancelled before acceptance"
        if record.status == "FAILED" and record.transport_accepted_at is None:
            return "failed", record.last_error or "Delivery failed before acceptance"
        return "unknown", record.last_error or "No authoritative terminal turn evidence; not replayed"
