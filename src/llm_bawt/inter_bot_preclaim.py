"""FIFO head loading and optional producer gates, before any turn reservation.

Lock order is target advisory lock (owned by caller), ScheduledJob, then
inter_bot_deliveries. Never lock a delivery and subsequently its schedule.
Ordinary callbacks have no prompt origin and retain their existing semantics.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from sqlalchemy import text
from sqlmodel import Session, select


def load_claim_head(conn, target: str, config):
    """Return the locked FIFO head, or None if its producer vetoed dispatch.

    A DISPATCHING delivery whose fallback turn the target has accepted and not
    yet ended is already delivered in order (it is that turn's trigger). It no
    longer heads the queue, so the next delivery may steer into the very turn it
    started instead of waiting for it to end (TASK-1067). Before acceptance, or
    once the turn has ended, it still blocks until marked terminal.
    """
    candidate = conn.execute(text("""
        SELECT d.* FROM inter_bot_deliveries d
        WHERE d.target_bot_id=:target
          AND d.status IN ('QUEUED','STEERING','DISPATCHING')
          AND NOT (
              d.status='DISPATCHING'
              AND d.transport_accepted_at IS NOT NULL
              AND EXISTS (
                  SELECT 1 FROM turn_logs t
                  WHERE t.id=d.turn_id AND t.ended_at IS NULL
              )
          )
        ORDER BY d.ordinal ASC LIMIT 1
    """), {"target": target}).mappings().first()
    if not candidate:
        return None
    meta = candidate.get("metadata_json") or {}
    if isinstance(meta, str):
        meta = json.loads(meta)
    origin = meta.get("prompt_schedule")
    if not origin:
        return _lock_delivery(conn, candidate["id"])

    from .service.prompt_schedule_models import PromptOccurrence, PromptSchedule
    from .service.prompt_delivery_state import gate_reason, transition, receipt
    from .service.scheduler import ScheduledJob, JobRun

    # Origin metadata is only a hint; validate against the immutable outbox.
    with Session(bind=conn) as session:
        job = session.exec(select(ScheduledJob).where(
            ScheduledJob.id == origin.get("schedule_id")
        ).with_for_update()).first()
        head = _lock_delivery(conn, candidate["id"])
        if not head or head["status"] != "QUEUED":
            return head
        # Accepted recovery must keep its logical RPC, even past pause/deadline.
        if head.get("transport_accepted_at") is not None:
            terminal = conn.execute(text("""
                SELECT status, ended_at, error_text FROM turn_logs WHERE id=:id
            """), {"id": head["turn_id"]}).mappings().first()
            if terminal and terminal["ended_at"] is not None:
                succeeded = terminal["status"] in {"ok", "completed"} and not terminal["error_text"]
                conn.execute(text("""
                    UPDATE inter_bot_deliveries SET status=:status, last_error=:error,
                        next_retry_at=NULL, updated_at=CURRENT_TIMESTAMP WHERE id=:id
                """), {"id": head["id"], "status": "DELIVERED" if succeeded else "FAILED",
                       "error": None if succeeded else terminal["error_text"] or "Target turn ended unsuccessfully"})
                return None
            return head
        occurrence = session.get(PromptOccurrence, origin.get("occurrence_id"))
        schedule = session.get(PromptSchedule, job.id) if job else None
        run = session.get(JobRun, occurrence.run_id) if occurrence else None
        payload = head["payload_json"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        valid = bool(job and schedule and occurrence and run
                     and occurrence.job_id == job.id
                     and head["author_entity_type"] == "user"
                     and head["author_entity_id"] == occurrence.snapshot_json["owner_user_id"]
                     and head["target_bot_id"] == occurrence.snapshot_json["bot_id"]
                     and head["idempotency_key"] == f"schedule:{job.id}:occurrence:{occurrence.occurrence_key}"
                     and payload.get("session_id") == occurrence.session_id
                     and payload.get("user") == occurrence.snapshot_json["owner_user_id"]
                     and payload.get("bot_id") == occurrence.snapshot_json["bot_id"]
                     and payload.get("messages") == [{"role": "user", "content": occurrence.snapshot_json["prompt"]}]
                     and payload.get("augment_memory") == occurrence.snapshot_json["augment_memory"]
                     and payload.get("extract_memory") == occurrence.snapshot_json["extract_memory"]
                     and payload.get("prefer_steer") is False
                     and head.get("session_policy", "continue") == "continue")
        now = datetime.now(timezone.utc)
        reason = gate_reason(job, schedule, occurrence, run, now) if valid else (
            "failed", "Scheduled delivery has no matching immutable outbox")
        if valid and reason is None:
            from types import SimpleNamespace
            from .service.prompt_capabilities import resolve_prompt_target
            try:
                bound = receipt(run).get("actual_model")
                if not bound or payload.get("model") != bound:
                    raise ValueError("Scheduled delivery has no bound model")
                resolve_prompt_target(SimpleNamespace(config=config),
                                      occurrence.snapshot_json["bot_id"], bound,
                                      occurrence.snapshot_json["owner_user_id"])
                owned = conn.execute(text("""
                    SELECT id FROM sessions WHERE id=:id AND bot_id=:bot
                      AND user_id=:owner AND status='archived'
                """), {"id": occurrence.session_id, "bot": head["target_bot_id"],
                       "owner": occurrence.snapshot_json["owner_user_id"]}).first()
                if not owned:
                    raise ValueError("Automation thread is missing, deleted, or active")
            except ValueError as exc:
                reason = "failed", str(exc)
        if reason:
            conn.execute(text("""
                UPDATE inter_bot_deliveries SET status=:status,
                    last_error=:error, updated_at=CURRENT_TIMESTAMP,
                    next_retry_at=NULL, claim_token=NULL, claim_owner=NULL,
                    lease_expires_at=NULL
                WHERE id=:id AND status='QUEUED' AND transport_accepted_at IS NULL
            """), {"id": head["id"], "status": "FAILED" if reason[0] == "failed" else "CANCELLED",
                   "error": reason[1]})
            conn.execute(text(
                "DELETE FROM turn_logs WHERE id=:id AND status='reserved'"
            ), {"id": head["turn_id"]})
            if valid:
                occurrence.delivery_id = head["id"]
                transition(session, job, schedule, occurrence, run, reason[0], now,
                           error=reason[1])
                session.flush()
            return None
        return head


def _lock_delivery(conn, delivery_id):
    lock = " FOR UPDATE" if conn.dialect.name == "postgresql" else ""
    return conn.execute(text(
        "SELECT * FROM inter_bot_deliveries WHERE id=:id" + lock
    ), {"id": delivery_id}).mappings().first()
