"""Durable, ordered inter-bot message delivery.

The delivery row is the canonical outbox record.  Normal chat turns remain the
execution mechanism: the dispatcher claims one FIFO head per target, invokes the
usual ``BackgroundService.chat_completion`` path, and correlates that turn by a
stable user-message/turn identity.  PostgreSQL owns durability and arbitration;
Redis is used only for best-effort lifecycle fanout.
"""

from __future__ import annotations

import json
import logging
import uuid
from contextlib import nullcontext
from datetime import datetime
from collections.abc import Iterable, Mapping
from typing import Any

from sqlalchemy import text as sa_text
from sqlalchemy.engine import Connection, Engine

from .agent_context import SessionPolicy
from .message_authorship import AuthorReference
from .maintenance_delivery import MaintenanceBarrier
from .inter_bot_control_lane import ensure_control_lane_schema
from .service.dispatcher_leadership import (
    LOCK_NAME_INFO_KEY,
    build_lock_engine,
    close_quietly,
    probe_lock_session,
    release_lock_session,
)
from .service.delivery_cancel_notice import ensure_cancel_notice_schema
from .service.turn_gates import ensure_gate_schema
from .inter_bot_delivery_query import DeliveryListFilter, parse_statuses
from .inter_bot_delivery_record import DeliveryRecord
from .inter_bot_delivery_steer_fence import pending_rejection, rejection_error, rejection_merge
from .inter_bot_delivery_user import resolve_delivery_user
from .utils.config import Config
from .utils.db import get_shared_engine
from .utils.schema import SchemaBootstrapGuard

logger = logging.getLogger(__name__)

QUEUED = "QUEUED"
STEERING = "STEERING"
DISPATCHING = "DISPATCHING"
DELIVERED = "DELIVERED"
FAILED = "FAILED"
CANCELLED = "CANCELLED"
TERMINAL_STATES = frozenset({DELIVERED, FAILED, CANCELLED})
ACTIVE_STATES = frozenset({QUEUED, STEERING, DISPATCHING})
# A requeued delivery whose /new reset already rotated (or completed) keeps
# its reservation row; every other requeue releases it.
RESERVATION_KEPT_RESET_STATUSES = frozenset({"ROTATED", "COMPLETED"})


def submission_result(
    record: "DeliveryRecord",
    *,
    duplicate: bool,
    target_exists: bool,
    requested_delivery: str = "steer_or_idle",
) -> tuple[int, dict[str, Any]]:
    """Return HTTP status + public enqueue result for one durable row."""
    accepted = record.status in {QUEUED, STEERING, DISPATCHING, DELIVERED}
    mode = "when_idle" if requested_delivery == "when_idle" else "steer_or_idle"
    result = record.to_api(duplicate=duplicate)
    result.update({
        "success": accepted,
        "queued": record.status == QUEUED,
        "delivery": mode,
        "note": (
            (
                "Delivery is durable and will start one normal target turn when that bot is idle."
                if mode == "when_idle"
                else "Delivery is durable and will steer an active Claude turn or start one safe idle fallback turn."
            )
            if accepted
            else "Delivery was recorded but is already in terminal FAILED state; inspect last_error."
        ),
    })
    status_code = 202 if accepted else (404 if not target_exists else 422)
    return status_code, result




class InterBotDeliveryStore:
    """PostgreSQL outbox with FIFO-per-target claims and durable recovery."""

    _schema_guard = SchemaBootstrapGuard()

    def __init__(self, config: Config, *, provision_schema: bool = True):
        self.config = config
        self.engine = get_shared_engine(config)
        # TASK-1311: the leadership session gets its own one-connection engine
        # (libpq keepalives, bounded probes) instead of pinning a pool slot.
        self._lock_engine: Engine | None = None
        if self.engine is not None and provision_schema:
            self._ensure_schema()

    def _ensure_schema(self) -> None:
        if self.engine is None:
            return

        def bootstrap(conn) -> None:
            conn.execute(sa_text("""
                CREATE TABLE IF NOT EXISTS inter_bot_deliveries (
                    ordinal BIGSERIAL UNIQUE NOT NULL,
                    id VARCHAR(96) PRIMARY KEY,
                    sender_bot_id VARCHAR(128) NOT NULL,
                    author_entity_type VARCHAR(16) NOT NULL DEFAULT 'bot',
                    author_entity_id VARCHAR(128) NOT NULL,
                    target_bot_id VARCHAR(128) NOT NULL,
                    message TEXT NOT NULL,
                    payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                    user_message_id VARCHAR(96) UNIQUE NOT NULL,
                    turn_id VARCHAR(128) UNIQUE NOT NULL,
                    idempotency_key VARCHAR(256),
                    project_id VARCHAR(128),
                    task_id VARCHAR(128),
                    message_kind VARCHAR(32),
                    status VARCHAR(24) NOT NULL,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    max_attempts INTEGER NOT NULL DEFAULT 5,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    available_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    dispatch_started_at TIMESTAMPTZ,
                    lease_expires_at TIMESTAMPTZ,
                    delivered_at TIMESTAMPTZ,
                    next_retry_at TIMESTAMPTZ,
                    last_error TEXT,
                    response_model VARCHAR(255),
                    response_chars INTEGER,
                    claim_token VARCHAR(96),
                    claim_owner VARCHAR(96),
                    transport_accepted_at TIMESTAMPTZ,
                    target_turn_id VARCHAR(128),
                    delivery_mode VARCHAR(24),
                    session_policy VARCHAR(32) NOT NULL DEFAULT 'continue',
                    reset_status VARCHAR(24),
                    reset_reason TEXT,
                    old_session_id VARCHAR(36),
                    new_session_id VARCHAR(36),
                    reset_at TIMESTAMPTZ,
                    overflow_recovery_count INTEGER NOT NULL DEFAULT 0,
                    CHECK (status IN ('QUEUED','STEERING','DISPATCHING','DELIVERED','FAILED','CANCELLED')),
                    CHECK (author_entity_type IN ('user','bot')),
                    CHECK (session_policy IN ('continue','reset_retain_history','reset_without_history'))
                )
            """))
            conn.execute(sa_text(
                "ALTER TABLE inter_bot_deliveries ADD COLUMN IF NOT EXISTS author_entity_type VARCHAR(16)"
            ))
            conn.execute(sa_text(
                "ALTER TABLE inter_bot_deliveries ADD COLUMN IF NOT EXISTS author_entity_id VARCHAR(128)"
            ))
            conn.execute(sa_text("""
                UPDATE inter_bot_deliveries
                SET author_entity_type='bot', author_entity_id=sender_bot_id
                WHERE author_entity_type IS NULL OR author_entity_id IS NULL
            """))
            conn.execute(sa_text(
                "ALTER TABLE inter_bot_deliveries ALTER COLUMN author_entity_type SET NOT NULL"
            ))
            conn.execute(sa_text(
                "ALTER TABLE inter_bot_deliveries ALTER COLUMN author_entity_type SET DEFAULT 'bot'"
            ))
            conn.execute(sa_text(
                "ALTER TABLE inter_bot_deliveries ALTER COLUMN author_entity_id SET NOT NULL"
            ))
            conn.execute(sa_text(
                "ALTER TABLE inter_bot_deliveries ADD COLUMN IF NOT EXISTS claim_token VARCHAR(96)"
            ))
            conn.execute(sa_text(
                "ALTER TABLE inter_bot_deliveries ADD COLUMN IF NOT EXISTS claim_owner VARCHAR(96)"
            ))
            conn.execute(sa_text(
                "ALTER TABLE inter_bot_deliveries ADD COLUMN IF NOT EXISTS transport_accepted_at TIMESTAMPTZ"
            ))
            conn.execute(sa_text(
                "ALTER TABLE inter_bot_deliveries ADD COLUMN IF NOT EXISTS target_turn_id VARCHAR(128)"
            ))
            conn.execute(sa_text(
                "ALTER TABLE inter_bot_deliveries ADD COLUMN IF NOT EXISTS delivery_mode VARCHAR(24)"
            ))
            conn.execute(sa_text(
                "ALTER TABLE inter_bot_deliveries ADD COLUMN IF NOT EXISTS session_policy VARCHAR(32) NOT NULL DEFAULT 'continue'"
            ))
            conn.execute(sa_text(
                "ALTER TABLE inter_bot_deliveries ADD COLUMN IF NOT EXISTS reset_status VARCHAR(24)"
            ))
            conn.execute(sa_text(
                "ALTER TABLE inter_bot_deliveries ADD COLUMN IF NOT EXISTS reset_reason TEXT"
            ))
            conn.execute(sa_text(
                "ALTER TABLE inter_bot_deliveries ADD COLUMN IF NOT EXISTS old_session_id VARCHAR(36)"
            ))
            conn.execute(sa_text(
                "ALTER TABLE inter_bot_deliveries ADD COLUMN IF NOT EXISTS new_session_id VARCHAR(36)"
            ))
            conn.execute(sa_text(
                "ALTER TABLE inter_bot_deliveries ADD COLUMN IF NOT EXISTS reset_at TIMESTAMPTZ"
            ))
            conn.execute(sa_text(
                "ALTER TABLE inter_bot_deliveries ADD COLUMN IF NOT EXISTS overflow_recovery_count INTEGER NOT NULL DEFAULT 0"
            ))
            # TASK-1300: a per-row revision, bumped by the database on every
            # UPDATE. Row locks serialize the updates, so it follows commit
            # order. Lifecycle events carry it and consumers drop any event
            # older than one they've already shown; publishing is async and
            # can't otherwise promise that order.
            conn.execute(sa_text(
                "ALTER TABLE inter_bot_deliveries ADD COLUMN IF NOT EXISTS revision BIGINT NOT NULL DEFAULT 0"
            ))
            conn.execute(sa_text("""
                CREATE OR REPLACE FUNCTION inter_bot_deliveries_bump_revision() RETURNS trigger AS $$
                BEGIN
                    NEW.revision := OLD.revision + 1;
                    RETURN NEW;
                END;
                $$ LANGUAGE plpgsql
            """))
            conn.execute(sa_text("""
                CREATE OR REPLACE TRIGGER inter_bot_deliveries_revision
                BEFORE UPDATE ON inter_bot_deliveries
                FOR EACH ROW EXECUTE FUNCTION inter_bot_deliveries_bump_revision()
            """))
            conn.execute(sa_text(
                "ALTER TABLE inter_bot_deliveries DROP CONSTRAINT IF EXISTS inter_bot_deliveries_status_check"
            ))
            conn.execute(sa_text(
                "ALTER TABLE inter_bot_deliveries DROP CONSTRAINT IF EXISTS inter_bot_deliveries_author_entity_type_check"
            ))
            conn.execute(sa_text(
                "ALTER TABLE inter_bot_deliveries DROP CONSTRAINT IF EXISTS inter_bot_deliveries_session_policy_check"
            ))
            conn.execute(sa_text("""
                ALTER TABLE inter_bot_deliveries ADD CONSTRAINT inter_bot_deliveries_status_check
                CHECK (status IN ('QUEUED','STEERING','DISPATCHING','DELIVERED','FAILED','CANCELLED'))
            """))
            conn.execute(sa_text("""
                ALTER TABLE inter_bot_deliveries ADD CONSTRAINT inter_bot_deliveries_author_entity_type_check
                CHECK (author_entity_type IN ('user','bot'))
            """))
            conn.execute(sa_text("""
                ALTER TABLE inter_bot_deliveries ADD CONSTRAINT inter_bot_deliveries_session_policy_check
                CHECK (session_policy IN ('continue','reset_retain_history','reset_without_history'))
            """))
            conn.execute(sa_text("""
                DROP INDEX IF EXISTS uq_inter_bot_delivery_idempotency
            """))
            conn.execute(sa_text("""
                CREATE UNIQUE INDEX uq_inter_bot_delivery_idempotency
                ON inter_bot_deliveries (
                    author_entity_type, author_entity_id,
                    sender_bot_id, target_bot_id, idempotency_key
                )
                WHERE idempotency_key IS NOT NULL
            """))
            conn.execute(sa_text("""
                CREATE INDEX IF NOT EXISTS ix_inter_bot_delivery_dispatch
                ON inter_bot_deliveries (target_bot_id, ordinal)
                WHERE status IN ('QUEUED','STEERING','DISPATCHING')
            """))
            conn.execute(sa_text("""
                CREATE INDEX IF NOT EXISTS ix_inter_bot_delivery_retry
                ON inter_bot_deliveries (available_at, ordinal)
                WHERE status = 'QUEUED'
            """))
            ensure_control_lane_schema(conn)  # TASK-1431 control lane

        self._schema_guard.run(self.engine, "inter-bot-deliveries", bootstrap)
        # The delivery claim asks the system gate under the target lock (TASK-1233).
        ensure_gate_schema(self.engine)
        ensure_cancel_notice_schema(self.engine)  # TASK-1327 sender notices

    def acquire_dispatcher_lock(
        self, lock_name: str = "inter-bot-dispatcher"
    ) -> Connection | None:
        """Acquire the process-wide dispatcher singleton fence.

        A session-level advisory lock lives for the lifetime of the returned
        connection and is released automatically if the process/connection dies.
        ``lock_name`` exists so integration tests can verify the primitive beside
        a live dispatcher without contending for production leadership.
        """
        if self.engine is None:
            return None
        if self._lock_engine is None:
            self._lock_engine = build_lock_engine(self.config)
        if self._lock_engine is None:
            return None
        # DB errors here (PostgreSQL down or restarting) propagate: the
        # dispatcher's leadership loop retries them on its passive cadence.
        conn = self._lock_engine.connect()
        try:
            acquired = conn.execute(
                sa_text("SELECT pg_try_advisory_lock(hashtext(:lock_name))"),
                {"lock_name": lock_name},
            ).scalar_one()
            # Session-level advisory locks survive COMMIT. End SQLAlchemy's
            # autobegun transaction so the leadership connection is not idle-in-tx.
            conn.commit()
        except Exception:
            close_quietly(conn)
            raise
        if not acquired:
            conn.close()
            return None
        conn.info[LOCK_NAME_INFO_KEY] = lock_name
        return conn

    @staticmethod
    def probe_dispatcher_lock(conn: Connection) -> bool:
        """Whether the leadership session still answers (TASK-1311).

        False means the session, and the lock with it, is gone; the
        connection is closed. Runs a round trip: call it off the event loop.
        """
        return probe_lock_session(conn)

    @staticmethod
    def release_dispatcher_lock(conn: Connection | None) -> None:
        """Unlock and close the leadership session.

        TASK-1311: a session that died (PostgreSQL restarted under it) has
        already released its lock; that case logs a WARNING and never raises,
        so shutdown continues past it.
        """
        if conn is None:
            return
        release_lock_session(conn)

    @staticmethod
    def stable_ids() -> tuple[str, str, str]:
        # Compatibility for test fixtures; production admission mints in the resolver.
        from .request_identity import stable_ids
        return stable_ids(uuid.uuid4())

    def enqueue(
        self,
        *,
        sender_bot_id: str,
        target_bot_id: str,
        message: str,
        payload: dict[str, Any],
        author: AuthorReference | None = None,
        idempotency_key: str | None = None,
        project_id: str | None = None,
        task_id: str | None = None,
        message_kind: str | None = None,
        metadata: dict[str, Any] | None = None,
        max_attempts: int = 5,
        session_policy: SessionPolicy | str = SessionPolicy.CONTINUE,
        reset_reason: str | None = None,
        user_message_id: str | None = None,
        assistant_message_id: str | None = None,
        connection: Connection | None = None,
        producer: str = "bot",
        on_gate: str = "deny",
    ) -> tuple[DeliveryRecord, bool]:
        """``on_gate``: under a deny gate (freeze) a new ``deny`` row raises
        ``SystemGateRefused``; ``hold`` rows are admitted and wait at the claim
        (TASK-1233). A caller passing ``connection`` must let that refusal
        propagate out of its transaction."""
        if self.engine is None:
            raise RuntimeError("Inter-bot delivery database unavailable")
        if connection is not None and connection.engine is not self.engine:
            raise ValueError("delivery enqueue requires the shared admission engine")
        sender = sender_bot_id.strip().lower() or "unknown"
        resolved_author = author or AuthorReference.bot(sender)
        if resolved_author.entity_type not in {"user", "bot"} or not resolved_author.entity_id:
            raise ValueError("delivery author must identify a user or bot")
        target = target_bot_id.strip().lower()
        body = message.strip()
        if not target:
            raise ValueError("target_bot_id is required")
        if not body:
            raise ValueError("message is required")
        idem = idempotency_key.strip() if idempotency_key and idempotency_key.strip() else None
        policy = SessionPolicy(str(session_policy))
        from .request_identity import RequestIdentityResolver

        values = {
            "sender": sender, "author_type": resolved_author.entity_type,
            "author_id": resolved_author.entity_id, "target": target,
            "message": body, "idem": idem, "project": project_id,
            "task": task_id, "kind": message_kind.upper() if message_kind else None,
            "max_attempts": max(1, min(int(max_attempts), 20)),
            "session_policy": policy.value,
            "reset_reason": (reset_reason or "").strip() or None,
        }
        with (nullcontext(connection) if connection is not None else self.engine.begin()) as conn:
            from .service.turn_admission import lock_target

            # Enqueue and idle context rotation share the target lock. Otherwise
            # a queued delivery could appear between the reset's queue check
            # and its durable thread rotation.
            lock_target(conn, target)
            stamped = MaintenanceBarrier(conn, target).stamp_enqueue(metadata)
            record, duplicate = RequestIdentityResolver(conn).admit(
                values=values, payload=payload, metadata=json.loads(stamped),
                producer=producer, explicit_user=user_message_id,
                explicit_assistant=assistant_message_id,
            )
            if not duplicate:
                # After the identity check, so a retried key already accepted
                # before the gate still gets its receipt. A refusal raises out
                # of this transaction, so the new row never commits.
                from .service.system_gate_admission import admit_enqueue

                admit_enqueue(conn, target, on_gate=on_gate)
            return record, duplicate

    def get(self, delivery_id: str, *, allowed_bot_ids: frozenset[str] | None = None) -> DeliveryRecord | None:
        if self.engine is None:
            return None
        where, params = DeliveryListFilter(allowed_bot_ids=allowed_bot_ids).where(self.engine.dialect.name)
        with self.engine.connect() as conn:
            row = conn.execute(
                sa_text(f"SELECT * FROM inter_bot_deliveries WHERE id=:id AND ({where})"),
                {"id": delivery_id, **params},
            ).mappings().first()
        return DeliveryRecord.from_mapping(row) if row else None

    def payload(self, delivery_id: str) -> dict[str, Any] | None:
        if self.engine is None:
            return None
        with self.engine.connect() as conn:
            value = conn.execute(
                sa_text("SELECT payload_json FROM inter_bot_deliveries WHERE id=:id"),
                {"id": delivery_id},
            ).scalar_one_or_none()
        return dict(value) if isinstance(value, dict) else None

    def list(
        self,
        *,
        sender_bot_id: str | None = None,
        target_bot_id: str | None = None,
        bot_id: str | None = None,
        status: str | Iterable[str] | None = None,
        idempotency_key: str | None = None,
        task_id: str | None = None,
        since: datetime | None = None,
        limit: int = 50,
        exclude_prompt_origin: bool = False,
        exclude_user_chat: bool = False,
        allowed_bot_ids: frozenset[str] | None = None,
    ) -> list[DeliveryRecord]:
        """List newest-first; ``status`` accepts one, comma-separated, or many.

        Filters run in SQL before LIMIT so private receipts neither leak nor
        crowd out selected deliveries. Internal scheduling/transport reads
        leave the exclusion flags off.
        """
        if self.engine is None:
            return []
        where, params = DeliveryListFilter(
            sender_bot_id=sender_bot_id,
            target_bot_id=target_bot_id,
            bot_id=bot_id,
            statuses=parse_statuses(status),
            idempotency_key=idempotency_key,
            task_id=task_id,
            since=since,
            exclude_prompt_origin=exclude_prompt_origin,
            exclude_user_chat=exclude_user_chat,
            allowed_bot_ids=allowed_bot_ids,
        ).where(self.engine.dialect.name)
        params["limit"] = max(1, min(int(limit), 200))
        sql = "SELECT * FROM inter_bot_deliveries WHERE " + where + " ORDER BY ordinal DESC LIMIT :limit"
        with self.engine.connect() as conn:
            rows = conn.execute(sa_text(sql), params).mappings().all()
        return [DeliveryRecord.from_mapping(row) for row in rows]

    def fail_expired_accepted_queued(self) -> list[DeliveryRecord]:
        """Dead-letter accepted queued runs outside Redis's recovery horizon."""
        if self.engine is None:
            return []
        with self.engine.begin() as conn:
            rows = conn.execute(sa_text("""
                UPDATE inter_bot_deliveries SET
                    status='FAILED', next_retry_at=NULL, available_at=now(),
                    last_error='accepted bridge run exceeded the seven-day recovery window; outcome is ambiguous and was not replayed',
                    updated_at=now()
                WHERE status='QUEUED' AND transport_accepted_at IS NOT NULL
                  AND transport_accepted_at < now() - interval '7 days'
                RETURNING *
            """)).mappings().all()
            for row in rows:
                from .service.session_lifecycle.recovery import NewSessionRecovery

                if NewSessionRecovery.is_reset(row):
                    conn.execute(sa_text("""
                        UPDATE inter_bot_deliveries
                        SET metadata_json=jsonb_set(metadata_json, '{new_session_ambiguous}', 'true'::jsonb)
                        WHERE id=:id
                    """), {"id": row["id"]})
                conn.execute(sa_text("""
                    UPDATE turn_logs SET status='error', end_reason='error',
                        error_text=:error, ended_at=COALESCE(ended_at, now())
                    WHERE id=:turn_id AND ended_at IS NULL
                """), {
                    "turn_id": row["turn_id"],
                    "error": row["last_error"],
                })
        return [DeliveryRecord.from_mapping(row) for row in rows]

    def eligible_targets(self, limit: int = 100) -> list[str]:
        if self.engine is None:
            return []
        self.fail_expired_accepted_queued()
        with self.engine.connect() as conn:
            rows = conn.execute(sa_text("""
                SELECT target_bot_id, min(ordinal) AS first_ordinal
                FROM inter_bot_deliveries
                WHERE status='QUEUED' AND available_at <= now()
                GROUP BY target_bot_id
                ORDER BY first_ordinal
                LIMIT :limit
            """), {"limit": limit}).all()
        return [row[0] for row in rows]

    @staticmethod
    def _explicit_stop_requested(conn, turn_id: str) -> bool:
        """Only a human Stop of this delivery's own turn ends its retry chain."""
        raw = conn.execute(sa_text(
            "SELECT request_json FROM turn_logs WHERE id=:turn_id"
        ), {"turn_id": turn_id}).scalar_one_or_none()
        try:
            request = json.loads(raw) if raw else {}
        except (TypeError, ValueError):
            return False
        audit = request.get("abort_request") if isinstance(request, dict) else None
        from .service.user_stop_hold import USER_STOP_SOURCES

        return isinstance(audit, dict) and audit.get("source") in USER_STOP_SOURCES

    def cancel_stopped_claim(self, delivery_id: str, claim_token: str) -> DeliveryRecord | None:
        """Fence an explicitly stopped delivery, but never an unrelated steer."""
        if self.engine is None or not claim_token:
            return None
        with self.engine.begin() as conn:
            row = conn.execute(sa_text("""
                SELECT turn_id, delivery_mode FROM inter_bot_deliveries
                WHERE id=:id AND status='DISPATCHING' AND claim_token=:claim
                FOR UPDATE
            """), {"id": delivery_id, "claim": claim_token}).mappings().first()
            if not row or row["delivery_mode"] != "turn" or not self._explicit_stop_requested(conn, row["turn_id"]):
                return None
            cancelled = conn.execute(sa_text("""
                UPDATE inter_bot_deliveries SET status='CANCELLED',
                    lease_expires_at=NULL, claim_token=NULL, claim_owner=NULL,
                    next_retry_at=NULL, last_error='Stopped by user', updated_at=now()
                WHERE id=:id AND status='DISPATCHING' AND claim_token=:claim
                RETURNING *
            """), {"id": delivery_id, "claim": claim_token}).mappings().first()
        return DeliveryRecord.from_mapping(cancelled) if cancelled else None

    def claim_next(
        self, target_bot_id: str, *, claim_owner: str, steer_capable: bool,
        lease_seconds: int = 300,
    ) -> DeliveryRecord | None:
        from .inter_bot_delivery_claim import DeliveryClaim

        return DeliveryClaim.claim_next(
            self, target_bot_id, claim_owner=claim_owner,
            steer_capable=steer_capable, lease_seconds=lease_seconds,
        )

    def mark_transport_accepted(self, delivery_id: str, claim_token: str) -> bool:
        if self.engine is None or not claim_token:
            return False
        with self.engine.begin() as conn:
            target = conn.execute(sa_text(
                "SELECT target_bot_id FROM inter_bot_deliveries WHERE id=:id"
            ), {"id": delivery_id}).scalar_one_or_none()
            if target is None:
                return False
            from .service.turn_admission import lock_target

            lock_target(conn, target)
            row = conn.execute(sa_text(
                "SELECT * FROM inter_bot_deliveries WHERE id=:id FOR UPDATE"
            ), {"id": delivery_id}).mappings().first()
            if not row or row["claim_token"] != claim_token or row["status"] not in {"STEERING", "DISPATCHING"}:
                return False
            if (row.get("metadata_json") or {}).get("message_route") and row["transport_accepted_at"] is None:
                from .route_dispatch_gate import check_route_claim

                if check_route_claim(conn, row, self.config) is None:
                    return False
            result = conn.execute(sa_text("""
                UPDATE inter_bot_deliveries
                SET transport_accepted_at=COALESCE(transport_accepted_at, now()), updated_at=now()
                WHERE id=:id AND status IN ('STEERING','DISPATCHING') AND claim_token=:claim_token
            """), {"id": delivery_id, "claim_token": claim_token})
        return bool(result.rowcount)

    def renew_lease(
        self, delivery_id: str, claim_token: str, *, lease_seconds: int = 300
    ) -> bool:
        if self.engine is None or not claim_token:
            return False
        with self.engine.begin() as conn:
            result = conn.execute(sa_text("""
                UPDATE inter_bot_deliveries
                SET lease_expires_at=now() + (:lease * interval '1 second'), updated_at=now()
                WHERE id=:id AND status IN ('STEERING','DISPATCHING') AND claim_token=:claim_token
            """), {
                "id": delivery_id,
                "claim_token": claim_token,
                "lease": max(60, lease_seconds),
            })
        return bool(result.rowcount)

    def mark_delivered(
        self,
        delivery_id: str,
        claim_token: str,
        *,
        response_model: str | None,
        response_chars: int,
    ) -> DeliveryRecord | None:
        """Settle a claim DELIVERED, fenced by its claim token.

        TASK-1300: a steer claim whose own command was already reported
        rejected (``steer_rejected_pending``) settles as a re-queue instead.
        The marker check and the DELIVERED write share one transaction and row
        lock, so a rejection either lands first (re-queue here) or waits for
        DELIVERED and reopens it; it can never fall between them.
        """
        if self.engine is None or not claim_token:
            return None
        values = {"id": delivery_id, "claim_token": claim_token,
                  "model": response_model, "chars": response_chars}
        with self.engine.begin() as conn:
            claim = conn.execute(sa_text("""
                SELECT * FROM inter_bot_deliveries
                WHERE id=:id AND status IN ('STEERING','DISPATCHING') AND claim_token=:claim_token
                FOR UPDATE
            """), values).mappings().first()
            if claim is None:
                return None
            if claim["status"] == "STEERING" and pending_rejection(claim) is not None:
                return self.requeue_claim(conn, delivery_id, claim_token, "", delay_seconds=1)
            row = conn.execute(sa_text("""
                UPDATE inter_bot_deliveries SET
                    status='DELIVERED', delivered_at=now(), lease_expires_at=NULL,
                    claim_token=NULL, claim_owner=NULL, next_retry_at=NULL, last_error=NULL,
                    response_model=:model, response_chars=:chars, updated_at=now()
                WHERE id=:id AND status IN ('STEERING','DISPATCHING') AND claim_token=:claim_token
                RETURNING *
            """), values).mappings().first()
        return DeliveryRecord.from_mapping(row) if row else None

    def requeue(
        self,
        delivery_id: str,
        claim_token: str,
        error: str,
        *,
        delay_seconds: float,
        clear_delivery_mode: bool = False,
        reject_steer_target: bool = False,
        refund_attempt: bool = False,
        metadata_merge: Mapping[str, Any] | None = None,
    ) -> DeliveryRecord | None:
        """Return a claimed delivery to QUEUED, fenced by its claim token.

        ``metadata_merge`` is shallow-merged (jsonb ``||``) into metadata_json
        in the same UPDATE as the QUEUED transition, so the caller's state
        advances atomically with the requeue (TASK-1251). The explicit-Stop
        CANCELLED branch never applies it.
        """
        if self.engine is None or not claim_token:
            return None
        with self.engine.begin() as conn:
            return self.requeue_claim(
                conn, delivery_id, claim_token, error, delay_seconds=delay_seconds,
                clear_delivery_mode=clear_delivery_mode, reject_steer_target=reject_steer_target,
                refund_attempt=refund_attempt, metadata_merge=metadata_merge,
            )

    @classmethod
    def requeue_claim(
        cls,
        conn: Connection,
        delivery_id: str,
        claim_token: str,
        error: str,
        *,
        delay_seconds: float,
        clear_delivery_mode: bool = False,
        reject_steer_target: bool = False,
        refund_attempt: bool = False,
        metadata_merge: Mapping[str, Any] | None = None,
    ) -> DeliveryRecord | None:
        """``requeue`` inside the caller's transaction (TASK-1233 hold revoke).

        The one QUEUED-return authority for a claimed delivery: the caller
        supplies the transaction and its locks; the claim token fences it.
        """
        if not claim_token:
            return None
        values = {
            "id": delivery_id,
            "claim_token": claim_token,
            "delay": max(0.0, float(delay_seconds)),
            "error": error[:4000],
            "metadata_merge": json.dumps(dict(metadata_merge or {})),
        }
        if clear_delivery_mode and reject_steer_target:
            raise ValueError("clear_delivery_mode and reject_steer_target are mutually exclusive")
        steering = conn.execute(sa_text("""
            SELECT * FROM inter_bot_deliveries
            WHERE id=:id AND status='STEERING' AND claim_token=:claim_token FOR UPDATE
        """), values).mappings().first()
        proof = pending_rejection(steering) if steering is not None else None
        if proof is not None:
            # TASK-1300: this claim's own command is proven not to have run, so
            # whatever the caller saw, the row leaves its target at k+1 rather
            # than replaying the frozen k.
            metadata = steering.get("metadata_json") or {}
            clear_delivery_mode, reject_steer_target, refund_attempt = False, True, True
            metadata_merge = {**dict(metadata_merge or {}), **rejection_merge(metadata, proof)}
            values["metadata_merge"] = json.dumps(metadata_merge)
            values["error"] = rejection_error(proof, int(metadata.get("steer_generation") or 0))
        claim = conn.execute(sa_text("""
            SELECT turn_id, delivery_mode FROM inter_bot_deliveries
            WHERE id=:id AND status='DISPATCHING' AND claim_token=:claim_token FOR UPDATE
        """), values).mappings().first()
        if claim and claim["delivery_mode"] == "turn" and cls._explicit_stop_requested(conn, claim["turn_id"]):
            row = conn.execute(sa_text("""
                UPDATE inter_bot_deliveries SET status='CANCELLED', lease_expires_at=NULL,
                    claim_token=NULL, claim_owner=NULL, next_retry_at=NULL,
                    last_error='Stopped by user', updated_at=now()
                WHERE id=:id AND status='DISPATCHING' AND claim_token=:claim_token RETURNING *
            """), values).mappings().first()
            return DeliveryRecord.from_mapping(row) if row else None
        mode_reset = (
            ", delivery_mode=NULL, target_turn_id=NULL, transport_accepted_at=NULL"
            if clear_delivery_mode else ""
        )
        steer_rejected = (
            ", delivery_mode='steer_rejected', transport_accepted_at=NULL"
            if reject_steer_target else ""
        )
        attempt_refund = ", attempt_count=GREATEST(attempt_count - 1, 0)" if refund_attempt else ""
        metadata = (
            ", metadata_json=metadata_json || CAST(:metadata_merge AS jsonb)"
            if metadata_merge else ""
        )
        row = conn.execute(sa_text(f"""
            UPDATE inter_bot_deliveries SET
                status='QUEUED', available_at=now() + (:delay * interval '1 second'),
                next_retry_at=now() + (:delay * interval '1 second'),
                lease_expires_at=NULL, claim_token=NULL, claim_owner=NULL,
                last_error=:error, updated_at=now(){mode_reset}{steer_rejected}{attempt_refund}{metadata}
            WHERE id=:id AND status IN ('STEERING','DISPATCHING') AND claim_token=:claim_token
            RETURNING *
        """), values).mappings().first()
        if row and row.get("reset_status") not in RESERVATION_KEPT_RESET_STATUSES:
            conn.execute(sa_text(
                "DELETE FROM turn_logs WHERE id=:turn_id AND status='reserved'"
            ), {"turn_id": row["turn_id"]})
        return DeliveryRecord.from_mapping(row) if row else None

    def recover_context_overflow(
        self,
        delivery_id: str,
        claim_token: str,
        *,
        backend: str | None,
        error: str,
    ) -> DeliveryRecord | None:
        if self.engine is None or not claim_token:
            return None
        from .agent_context import recover_delivery_after_overflow

        with self.engine.begin() as conn:
            # Follow claim_next's lock order: target first, then delivery row.
            # The preliminary read is not authority; status, claim and target
            # are checked again under the row lock before any mutation.
            target = conn.execute(sa_text("""
                SELECT target_bot_id FROM inter_bot_deliveries WHERE id=:id
            """), {"id": delivery_id}).scalar_one_or_none()
            if not target:
                return None
            from .service.turn_admission import lock_target

            lock_target(conn, target)
            head = conn.execute(sa_text("""
                SELECT * FROM inter_bot_deliveries
                WHERE id=:id AND status='DISPATCHING' AND claim_token=:claim
                FOR UPDATE
            """), {"id": delivery_id, "claim": claim_token}).mappings().first()
            if not head or head["target_bot_id"] != target:
                return None
            # Generic overflow recovery rotates the active personal thread and
            # changes the logical RPC. Automation must never take that path.
            if (head.get("metadata_json") or {}).get("prompt_schedule"):
                return None
            payload = head["payload_json"] if isinstance(head["payload_json"], dict) else {}
            user_id = resolve_delivery_user(self.config, payload.get("user"))
            row = recover_delivery_after_overflow(
                conn,
                head=head,
                backend=backend,
                user_id=user_id,
                error=error,
            )
        return DeliveryRecord.from_mapping(row) if row else None

    def fail_claim(
        self, delivery_id: str, claim_token: str, error: str, *,
        connection: Connection | None = None,
    ) -> DeliveryRecord | None:
        """Fail a claimed row; ``connection`` joins the caller's transaction
        (TASK-1231B: fail the claim and insert its restart retry atomically)."""
        if self.engine is None or not claim_token:
            return None
        with (nullcontext(connection) if connection is not None else self.engine.begin()) as conn:
            row = conn.execute(sa_text("""
                UPDATE inter_bot_deliveries SET status='FAILED',
                    lease_expires_at=NULL, claim_token=NULL, claim_owner=NULL, next_retry_at=NULL,
                    last_error=:error, updated_at=now()
                WHERE id=:id AND status IN ('STEERING','DISPATCHING') AND claim_token=:claim_token
                RETURNING *
            """), {
                "id": delivery_id,
                "claim_token": claim_token,
                "error": error[:4000],
            }).mappings().first()
            if row:
                conn.execute(sa_text("""
                    UPDATE turn_logs SET status='error', end_reason='error',
                        error_text=:error, ended_at=now()
                    WHERE id=:turn_id AND status='reserved'
                """), {"turn_id": row["turn_id"], "error": error[:4000]})
        return DeliveryRecord.from_mapping(row) if row else None

    def mark_failed(self, delivery_id: str, error: str) -> DeliveryRecord | None:
        """Terminally fail a non-dispatching row (validation/dead-letter setup)."""
        if self.engine is None:
            return None
        with self.engine.begin() as conn:
            row = conn.execute(sa_text("""
                UPDATE inter_bot_deliveries SET status='FAILED',
                    lease_expires_at=NULL, claim_token=NULL, claim_owner=NULL, next_retry_at=NULL,
                    last_error=:error, updated_at=now()
                WHERE id=:id AND status != 'DELIVERED' RETURNING *
            """), {"id": delivery_id, "error": error[:4000]}).mappings().first()
        return DeliveryRecord.from_mapping(row) if row else None

    @staticmethod
    def cancel_unaccepted_locked(
        conn, candidate: dict, reason: str | None = None, *,
        allow_accepted_queued: bool = False,
    ) -> dict | None:
        """Cancel a target-locked, unaccepted delivery without leaving a FIFO head.

        QUEUED rows may keep delivery_mode='turn' after their reservation was
        removed during requeue; absence of that row is not ambiguous evidence.
        An actual DISPATCHING claim must prove its reservation is unstarted.
        """
        if candidate.get("transport_accepted_at") is not None and not (
            allow_accepted_queued and candidate["status"] == QUEUED
        ):
            return None
        if candidate["status"] == "DISPATCHING":
            from .service.turn_admission import release_reservation

            if not release_reservation(
                None, candidate["turn_id"], connection=conn,
                delivery_id=candidate["id"], bot_id=candidate["target_bot_id"],
                trigger_message_id=candidate["user_message_id"],
            ):
                return None
        elif candidate["status"] != QUEUED:
            return None
        row = conn.execute(sa_text("""
            UPDATE inter_bot_deliveries SET status='CANCELLED',
                last_error=COALESCE(:reason, last_error),
                lease_expires_at=NULL, claim_token=NULL, claim_owner=NULL,
                next_retry_at=NULL, updated_at=now()
            WHERE id=:id AND status IN ('QUEUED','DISPATCHING')
              AND (:allow_accepted OR transport_accepted_at IS NULL) RETURNING *
        """), {"id": candidate["id"], "reason": reason,
               "allow_accepted": allow_accepted_queued}).mappings().first()
        if row and candidate["status"] == QUEUED:
            conn.execute(sa_text(
                "DELETE FROM turn_logs WHERE id=:turn_id AND status='reserved'"
            ), {"turn_id": row["turn_id"]})
        return row

    def cancel(self, delivery_id: str) -> DeliveryRecord | None:
        if self.engine is None:
            return None
        with self.engine.begin() as conn:
            target = conn.execute(sa_text(
                "SELECT target_bot_id FROM inter_bot_deliveries WHERE id=:id"
            ), {"id": delivery_id}).scalar_one_or_none()
            if not target:
                return None
            # Same lock order as claim_next: target advisory lock, then row lock.
            conn.execute(
                sa_text("SELECT pg_advisory_xact_lock(hashtext(:target))"),
                {"target": target},
            )
            current = conn.execute(sa_text("""
                SELECT * FROM inter_bot_deliveries WHERE id=:id FOR UPDATE
            """), {"id": delivery_id}).mappings().first()
            row = self.cancel_queued_locked(conn, current) if current else None
        return DeliveryRecord.from_mapping(row) if row else None

    @classmethod
    def cancel_queued_locked(cls, conn, current: dict, reason: str | None = None) -> dict | None:
        """The one rule for cancelling a QUEUED delivery on request.

        ``current`` is locked under its target lock. Shared by ``cancel`` and
        the user-Stop hold's Cancel all (TASK-1327). Accepted, queued
        *ordinary* callbacks may still be cancelled (legacy); routed and
        scheduled work keep the no-abort-after-acceptance invariant.
        """
        if current["status"] != QUEUED:
            return None
        origin = current.get("metadata_json") or {}
        return cls.cancel_unaccepted_locked(
            conn, current, reason,
            allow_accepted_queued=not (origin.get("prompt_schedule") or origin.get("message_route")),
        )

    def turn_state(self, delivery_id: str) -> tuple[str | None, datetime | None]:
        if self.engine is None:
            return None, None
        with self.engine.connect() as conn:
            row = conn.execute(sa_text("""
                SELECT t.status, t.ended_at
                FROM inter_bot_deliveries d
                LEFT JOIN turn_logs t ON t.id=d.turn_id
                WHERE d.id=:id
            """), {"id": delivery_id}).first()
        return (row[0], row[1]) if row else (None, None)

    def validate_claim(
        self,
        *,
        delivery_id: str,
        claim_token: str,
        target_bot_id: str,
        turn_id: str,
        user_message_id: str,
        bridge_request_id: str,
    ) -> bool:
        if self.engine is None:
            return False
        from .inter_bot_claim_validation import validate_delivery_claim

        with self.engine.connect() as conn:
            return validate_delivery_claim(
                conn, delivery_id=delivery_id, claim_token=claim_token,
                target_bot_id=target_bot_id, turn_id=turn_id,
                user_message_id=user_message_id, bridge_request_id=bridge_request_id,
            )

    def _transition(self, delivery_id: str, assignments: str, params: dict[str, Any], *, only: str | None) -> DeliveryRecord | None:
        if self.engine is None:
            return None
        values = {"id": delivery_id, **params}
        predicate = " AND status=:only" if only else ""
        if only:
            values["only"] = only
        with self.engine.begin() as conn:
            row = conn.execute(sa_text(
                f"UPDATE inter_bot_deliveries SET {assignments} WHERE id=:id{predicate} RETURNING *"
            ), values).mappings().first()
        return DeliveryRecord.from_mapping(row) if row else self.get(delivery_id)

    def recover_expired(self, *, live_owner: str | None = None) -> list[DeliveryRecord]:
        """Recover expired claims; singleton fencing excludes live peers.

        Body lives in ``inter_bot_delivery_recovery`` (TASK-1286 D0).
        """
        from .inter_bot_delivery_recovery import DeliveryLeaseRecovery

        return DeliveryLeaseRecovery.recover_expired(self, live_owner=live_owner)

    def expire_own_claims(self, *, claim_owner: str, turn_ids: list[str]) -> list[str]:
        """Release confirmed-stopped turn claims at shutdown (TASK-1286)."""
        from .inter_bot_delivery_recovery import DeliveryLeaseRecovery

        return DeliveryLeaseRecovery.expire_own_claims(self, claim_owner=claim_owner, turn_ids=turn_ids)
