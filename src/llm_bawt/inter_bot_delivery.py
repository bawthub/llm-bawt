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
from datetime import datetime, timedelta, timezone
from collections.abc import Iterable, Mapping
from typing import Any

from sqlalchemy import text as sa_text
from sqlalchemy.engine import Connection

from .agent_context import SessionPolicy
from .message_authorship import AuthorReference
from .maintenance_delivery import MaintenanceBarrier
from .service.turn_gates import ensure_gate_schema
from .inter_bot_delivery_query import DeliveryListFilter, parse_statuses
from .inter_bot_delivery_record import DeliveryRecord
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

    def __init__(self, config: Config):
        self.config = config
        self.engine = get_shared_engine(config)
        if self.engine is not None:
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

        self._schema_guard.run(self.engine, "inter-bot-deliveries", bootstrap)
        # The delivery claim asks the system gate under the target lock (TASK-1233).
        ensure_gate_schema(self.engine)

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
        conn = self.engine.connect()
        acquired = conn.execute(
            sa_text("SELECT pg_try_advisory_lock(hashtext(:lock_name))"),
            {"lock_name": lock_name},
        ).scalar_one()
        # Session-level advisory locks survive COMMIT. End SQLAlchemy's
        # autobegun transaction so the leadership connection is not idle-in-tx.
        conn.commit()
        if not acquired:
            conn.close()
            return None
        conn.info["inter_bot_dispatcher_lock_name"] = lock_name
        return conn

    @staticmethod
    def release_dispatcher_lock(conn: Connection | None) -> None:
        if conn is None:
            return
        try:
            lock_name = conn.info.get(
                "inter_bot_dispatcher_lock_name", "inter-bot-dispatcher"
            )
            conn.execute(
                sa_text("SELECT pg_advisory_unlock(hashtext(:lock_name))"),
                {"lock_name": lock_name},
            )
            conn.commit()
        finally:
            conn.close()

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

    def get(self, delivery_id: str) -> DeliveryRecord | None:
        if self.engine is None:
            return None
        with self.engine.connect() as conn:
            row = conn.execute(
                sa_text("SELECT * FROM inter_bot_deliveries WHERE id=:id"),
                {"id": delivery_id},
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
        return isinstance(audit, dict) and audit.get("source") in ("chat_stop", "bot_list_stop")

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
        return self._claim_transition(delivery_id, claim_token, """
            status='DELIVERED', delivered_at=now(), lease_expires_at=NULL,
            claim_token=NULL, claim_owner=NULL, next_retry_at=NULL, last_error=NULL,
            response_model=:model, response_chars=:chars, updated_at=now()
        """, {"model": response_model, "chars": response_chars})

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

    def _claim_transition(
        self,
        delivery_id: str,
        claim_token: str,
        assignments: str,
        params: dict[str, Any],
    ) -> DeliveryRecord | None:
        if self.engine is None or not claim_token:
            return None
        values = {"id": delivery_id, "claim_token": claim_token, **params}
        with self.engine.begin() as conn:
            row = conn.execute(sa_text(
                f"UPDATE inter_bot_deliveries SET {assignments} "
                "WHERE id=:id AND status IN ('STEERING','DISPATCHING') AND claim_token=:claim_token RETURNING *"
            ), values).mappings().first()
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
            if not current or current["status"] != QUEUED:
                return None
            if ((current.get("metadata_json") or {}).get("prompt_schedule")
                    and current.get("transport_accepted_at") is not None):
                return None
            # Preserve legacy cancellation for accepted, queued *ordinary*
            # callbacks. Routed and scheduled work retain the no-abort-after-
            # acceptance invariant.
            origin = current.get("metadata_json") or {}
            row = self.cancel_unaccepted_locked(
                conn, current,
                allow_accepted_queued=not (origin.get("prompt_schedule") or origin.get("message_route")),
            )
        return DeliveryRecord.from_mapping(row) if row else None

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

    @staticmethod
    def _settle_stopped_turn(conn: Connection, turn_id: str) -> None:
        """Settle a stopped delivery's open turn in the caller's transaction."""
        from sqlmodel import Session, select

        from .service.turn_logs import TurnLog
        from .service.turn_termination import settle_open_turn

        with Session(bind=conn) as session:
            row = session.exec(select(TurnLog).where(TurnLog.id == turn_id).with_for_update()).first()
            if row is not None:
                settle_open_turn(session, row, audit_key="user_stop", now=datetime.now(timezone.utc), audit={
                    "bridge_state": "lease_expired", "confirmed": False, "by": "recover_expired",
                })

    def recover_expired(self, *, live_owner: str | None = None) -> list[DeliveryRecord]:
        """Recover expired claims; singleton fencing excludes live peers.

        TASK-1286 #2: a user-stopped ordinary turn whose claim no live
        dispatcher owns (claim_owner != live_owner, so no in-process _dispatch
        settles it) is cancelled now. Other foreign claims, accepted resets
        included (their ambiguity check precedes Stop), wait out the lease.
        """
        if self.engine is None:
            return []
        from .service.session_lifecycle.recovery import NewSessionRecovery
        recovered: list[DeliveryRecord] = []
        orphan = (" OR (d.delivery_mode='turn' AND t.ended_at IS NOT NULL"
                  " AND d.claim_owner IS DISTINCT FROM :live_owner)") if live_owner else ""
        with self.engine.begin() as conn:
            rows = conn.execute(sa_text(f"""
                SELECT d.*, t.status AS turn_status, t.ended_at AS turn_ended_at,
                       t.error_text AS turn_error_text,
                       d.lease_expires_at < now() AS lease_expired
                FROM inter_bot_deliveries d
                LEFT JOIN turn_logs t ON t.id=d.turn_id
                WHERE d.status IN ('STEERING','DISPATCHING')
                  AND (d.lease_expires_at < now(){orphan})
                ORDER BY d.ordinal FOR UPDATE OF d
            """), {"live_owner": live_owner} if orphan else {}).mappings().all()
            for row in rows:
                is_turn = row.get("delivery_mode") != "steer"
                accepted = row.get("transport_accepted_at") is not None
                if not row["lease_expired"] and ((accepted and NewSessionRecovery.is_reset(row))
                                                 or not self._explicit_stop_requested(conn, row["turn_id"])):
                    continue
                if is_turn and accepted:
                    if NewSessionRecovery.is_reset(row) and (
                        row.get("turn_ended_at") is None
                        or row.get("turn_status") not in ("ok", "completed")
                        or row.get("turn_error_text")
                    ):
                        ambiguous = NewSessionRecovery.mark_ambiguous(
                            conn, delivery_id=row["id"], claim_token=row["claim_token"],
                            status=row["status"],
                            reason="accepted reset outcome is ambiguous; never replay provider reset",
                        )
                        if ambiguous:
                            recovered.append(ambiguous)
                        continue
                if is_turn and self._explicit_stop_requested(conn, row["turn_id"]):
                    assignments = "status='CANCELLED', claim_token=NULL, claim_owner=NULL, lease_expires_at=NULL, next_retry_at=NULL, last_error='Stopped by user'"
                elif is_turn and row.get("turn_ended_at") is not None:
                    success = (
                        row.get("turn_status") in ("ok", "completed")
                        and not row.get("turn_error_text")
                    )
                    assignments = (
                        "status='DELIVERED', delivered_at=COALESCE(delivered_at, now()), claim_token=NULL, claim_owner=NULL, lease_expires_at=NULL, last_error=NULL"
                        if success else
                        "status='FAILED', claim_token=NULL, claim_owner=NULL, lease_expires_at=NULL, last_error=COALESCE(last_error, 'target turn ended unsuccessfully')"
                    )
                elif (
                    row.get("transport_accepted_at") is not None
                    and row["transport_accepted_at"]
                    < datetime.now(timezone.utc) - timedelta(days=7)
                ):
                    assignments = "status='FAILED', claim_token=NULL, claim_owner=NULL, lease_expires_at=NULL, last_error='accepted bridge run exceeded the seven-day recovery window; outcome is ambiguous and was not replayed'"
                elif int(row["attempt_count"] or 0) < int(row["max_attempts"] or 0):
                    assignments = "status='QUEUED', claim_token=NULL, claim_owner=NULL, lease_expires_at=NULL, available_at=now(), next_retry_at=now(), last_error='delivery lease expired; retrying same logical message'"
                else:
                    assignments = "status='FAILED', claim_token=NULL, claim_owner=NULL, lease_expires_at=NULL, last_error='delivery attempts exhausted before completion'"
                updated = conn.execute(sa_text(
                    f"UPDATE inter_bot_deliveries SET {assignments}, updated_at=now() WHERE id=:id RETURNING *"
                ), {"id": row["id"]}).mappings().one()
                if updated["status"] == "CANCELLED" and is_turn:
                    # TASK-1232 P4: a user Stop ends the turn too. Cancelling
                    # only the delivery left the row open, and admission
                    # blocked every later message (the 2026-10-21 deadlock).
                    self._settle_stopped_turn(conn, row["turn_id"])
                if updated["status"] == QUEUED and is_turn:
                    conn.execute(sa_text("""
                        UPDATE turn_logs SET status='reserved', ended_at=NULL,
                            end_reason=NULL, error_text=NULL
                        WHERE id=:turn_id AND ended_at IS NULL
                    """), {"turn_id": row["turn_id"]})
                recovered.append(DeliveryRecord.from_mapping(updated))
        return recovered
