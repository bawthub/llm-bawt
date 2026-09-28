"""Durable, ordered inter-bot message delivery.

The delivery row is the canonical outbox record.  Normal chat turns remain the
execution mechanism: the dispatcher claims one FIFO head per target, invokes the
usual ``BackgroundService.chat_completion`` path, and correlates that turn by a
stable user-message/turn identity.  PostgreSQL owns durability and arbitration;
Redis is used only for best-effort lifecycle fanout.
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import text as sa_text
from sqlalchemy.engine import Connection

from .agent_context import SessionPolicy
from .message_authorship import AuthorReference
from .inter_bot_delivery_record import DeliveryRecord
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
        token = uuid.uuid4()
        # Message partitions store ids in VARCHAR(36) and every ordinary caller
        # supplies a canonical UUID. Keep the durable callback on that same
        # contract; delivery/turn ids use the hex token in their own wider fields.
        return (
            f"delivery-{token.hex}",
            str(token),
            f"turn-delivery-{token.hex}",
        )

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
    ) -> tuple[DeliveryRecord, bool]:
        if self.engine is None:
            raise RuntimeError("Inter-bot delivery database unavailable")
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
        delivery_id, message_id, turn_id = self.stable_ids()
        payload = dict(payload)
        payload["user_message_id"] = message_id
        payload["assistant_message_id"] = str(
            uuid.uuid5(uuid.NAMESPACE_URL, f"inter-bot-assistant:{message_id}")
        )
        payload["inter_bot_delivery_id"] = delivery_id
        payload["inter_bot_turn_id"] = turn_id
        payload["inter_bot_bridge_request_id"] = (
            f"req_delivery_{delivery_id.removeprefix('delivery-')}"
        )
        values = {
            "id": delivery_id,
            "sender": sender,
            "author_type": resolved_author.entity_type,
            "author_id": resolved_author.entity_id,
            "target": target,
            "message": body,
            "payload": json.dumps(payload, ensure_ascii=False, default=str),
            "metadata": json.dumps(metadata or {}, ensure_ascii=False, default=str),
            "message_id": message_id,
            "turn_id": turn_id,
            "idem": idem,
            "project": project_id,
            "task": task_id,
            "kind": message_kind.upper() if message_kind else None,
            "max_attempts": max(1, min(int(max_attempts), 20)),
            "session_policy": policy.value,
            "reset_reason": (reset_reason or "").strip() or None,
        }
        with self.engine.begin() as conn:
            if idem:
                # Serialize same-key submissions so callers always receive one stable row.
                digest = hashlib.sha256(
                    f"{resolved_author.entity_type}\0{resolved_author.entity_id}\0"
                    f"{sender}\0{target}\0{idem}".encode()
                ).hexdigest()[:16]
                conn.execute(sa_text("SELECT pg_advisory_xact_lock(hashtext(:key))"), {"key": digest})
                existing = conn.execute(sa_text("""
                    SELECT * FROM inter_bot_deliveries
                    WHERE author_entity_type=:author_type
                      AND author_entity_id=:author_id
                      AND sender_bot_id=:sender AND target_bot_id=:target
                      AND idempotency_key=:idem
                """), values).mappings().first()
                if existing:
                    record = DeliveryRecord.from_mapping(existing)
                    if record.session_policy != policy.value:
                        raise ValueError(
                            "idempotency key already exists with "
                            f"session_policy={record.session_policy!r}"
                        )
                    return record, True
            row = conn.execute(sa_text("""
                INSERT INTO inter_bot_deliveries (
                    id, sender_bot_id, author_entity_type, author_entity_id,
                    target_bot_id, message, payload_json,
                    metadata_json, user_message_id, turn_id, idempotency_key,
                    project_id, task_id, message_kind, status, max_attempts,
                    session_policy, reset_status, reset_reason
                ) VALUES (
                    :id, :sender, :author_type, :author_id,
                    :target, :message, CAST(:payload AS jsonb),
                    CAST(:metadata AS jsonb), :message_id, :turn_id, :idem,
                    :project, :task, :kind, 'QUEUED', :max_attempts,
                    :session_policy,
                    CASE WHEN :session_policy='continue' THEN NULL ELSE 'PENDING' END,
                    :reset_reason
                ) RETURNING *
            """), values).mappings().one()
        return DeliveryRecord.from_mapping(row), False

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
        status: str | None = None,
        limit: int = 50,
        exclude_prompt_origin: bool = False,
    ) -> list[DeliveryRecord]:
        if self.engine is None:
            return []
        where = ["1=1"]
        # Filter before LIMIT so private receipts neither leak nor crowd out
        # ordinary deliveries. Internal scheduling/transport reads stay intact.
        if exclude_prompt_origin:
            if self.engine.dialect.name == "postgresql":
                where.append("NOT (COALESCE(metadata_json, '{}'::jsonb) ? 'prompt_schedule')")
            else:
                where.append("json_type(metadata_json, '$.prompt_schedule') IS NULL")
        params: dict[str, Any] = {"limit": max(1, min(int(limit), 200))}
        if sender_bot_id:
            where.append("sender_bot_id=:sender")
            params["sender"] = sender_bot_id.strip().lower()
        if target_bot_id:
            where.append("target_bot_id=:target")
            params["target"] = target_bot_id.strip().lower()
        if status:
            where.append("status=:status")
            params["status"] = status.strip().upper()
        sql = "SELECT * FROM inter_bot_deliveries WHERE " + " AND ".join(where) + " ORDER BY ordinal DESC LIMIT :limit"
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
        self,
        target_bot_id: str,
        *,
        claim_owner: str,
        steer_capable: bool,
        lease_seconds: int = 300,
    ) -> DeliveryRecord | None:
        """Atomically choose STEERING for an active Claude turn or safe dispatch."""
        if self.engine is None:
            return None
        target = target_bot_id.strip().lower()
        claim_token = f"claim-{uuid.uuid4().hex}"
        with self.engine.begin() as conn:
            conn.execute(
                sa_text("SELECT pg_advisory_xact_lock(hashtext(:target))"),
                {"target": target},
            )
            from .inter_bot_preclaim import load_claim_head
            head = load_claim_head(conn, target, self.config)
            if (
                not head
                or head["status"] != QUEUED
                or head["available_at"] > datetime.now(timezone.utc)
            ):
                return None
            if self._explicit_stop_requested(conn, head["turn_id"]):
                conn.execute(sa_text("""
                    UPDATE inter_bot_deliveries SET status='CANCELLED',
                        next_retry_at=NULL, last_error='Stopped by user', updated_at=now()
                    WHERE id=:id AND status='QUEUED'
                """), {"id": head["id"]})
                return None
            payload = head["payload_json"] if isinstance(head["payload_json"], dict) else {}
            policy = SessionPolicy(head.get("session_policy") or SessionPolicy.CONTINUE.value)
            prefer_steer = bool(payload.get("prefer_steer", True))
            delivery_user_id = str(
                payload.get("user") or getattr(self.config, "DEFAULT_USER", "nick")
            )
            active = conn.execute(sa_text("""
                SELECT id, agent_session_key, agent_request_id, user_id
                FROM turn_logs
                WHERE bot_id=:target AND ended_at IS NULL AND id != :turn_id
                ORDER BY created_at DESC LIMIT 1
            """), {"target": target, "turn_id": head["turn_id"]}).mappings().first()
            if active and str(active.get("user_id") or "") != delivery_user_id:
                return None
            if active and policy != SessionPolicy.CONTINUE:
                # Reset is an ordered precondition, never an interrupt. Wait until
                # the authoritative active turn ends; do not steer or rotate under it.
                return None
            existing_steer = (
                head.get("delivery_mode") == "steer"
                and head.get("target_turn_id")
                and prefer_steer
            )
            rejected_same_turn = (
                head.get("delivery_mode") == "steer_rejected"
                and head.get("target_turn_id")
                and active
                and active["id"] == head["target_turn_id"]
            )
            if existing_steer:
                # A steer RPC may have reached the bridge even if the app timed
                # out before seeing its result. Always retry the SAME deterministic
                # RPC/message against the SAME turn until it proves not accepted;
                # only then may the dispatcher clear steer intent for idle fallback.
                status = STEERING
                target_turn_id = head["target_turn_id"]
                delivery_mode = "steer"
            elif active:
                if rejected_same_turn:
                    # This exact active row definitively rejected the deterministic
                    # steer RPC. Wait for it to end rather than hot-looping or
                    # starting an overlapping fallback turn. A newer active row
                    # may still be steered on the next claim.
                    return None
                if not steer_capable or not prefer_steer:
                    return None
                status = STEERING
                target_turn_id = active["id"]
                delivery_mode = "steer"
            else:
                status = DISPATCHING
                target_turn_id = head["turn_id"]
                delivery_mode = "turn"
                user_id = delivery_user_id
                from .agent_context import rotate_delivery_session
                payload, _, _ = rotate_delivery_session(
                    conn,
                    head=head,
                    payload=payload,
                    target_bot_id=target,
                    user_id=user_id,
                )
                conn.execute(sa_text("""
                    INSERT INTO turn_logs (
                        id, created_at, path, stream, bot_id, user_id, status,
                        user_prompt, request_json, response_text,
                        trigger_message_id, ended_at
                    ) VALUES (
                        :turn_id, now(), '/v1/chat/completions', false, :target,
                        :user_id, 'reserved', :message, CAST(:request_json AS jsonb)::text,
                        '', :message_id, NULL
                    )
                    ON CONFLICT (id) DO UPDATE SET
                        status='reserved', ended_at=NULL, error_text=NULL,
                        request_json=EXCLUDED.request_json, user_prompt=EXCLUDED.user_prompt
                """), {
                    "turn_id": head["turn_id"],
                    "target": target,
                    "user_id": user_id,
                    "message": head["message"],
                    "request_json": json.dumps(payload, ensure_ascii=False, default=str),
                    "message_id": head["user_message_id"],
                })
            row = conn.execute(sa_text("""
                UPDATE inter_bot_deliveries SET
                    status=:status, attempt_count=attempt_count+1,
                    dispatch_started_at=now(),
                    lease_expires_at=now() + (:lease * interval '1 second'),
                    claim_token=:claim_token, claim_owner=:claim_owner,
                    target_turn_id=:target_turn_id, delivery_mode=:delivery_mode,
                    next_retry_at=NULL, last_error=NULL, updated_at=now()
                WHERE id=:id AND status='QUEUED'
                RETURNING *
            """), {
                "id": head["id"],
                "status": status,
                "lease": max(60, lease_seconds),
                "claim_token": claim_token,
                "claim_owner": claim_owner,
                "target_turn_id": target_turn_id,
                "delivery_mode": delivery_mode,
            }).mappings().first()
        return DeliveryRecord.from_mapping(row) if row else None

    def mark_transport_accepted(self, delivery_id: str, claim_token: str) -> bool:
        if self.engine is None or not claim_token:
            return False
        with self.engine.begin() as conn:
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
    ) -> DeliveryRecord | None:
        if self.engine is None or not claim_token:
            return None
        values = {
            "id": delivery_id,
            "claim_token": claim_token,
            "delay": max(0.0, float(delay_seconds)),
            "error": error[:4000],
        }
        with self.engine.begin() as conn:
            if clear_delivery_mode and reject_steer_target:
                raise ValueError("clear_delivery_mode and reject_steer_target are mutually exclusive")
            claim = conn.execute(sa_text("""
                SELECT turn_id, delivery_mode FROM inter_bot_deliveries
                WHERE id=:id AND status='DISPATCHING' AND claim_token=:claim_token FOR UPDATE
            """), values).mappings().first()
            if claim and claim["delivery_mode"] == "turn" and self._explicit_stop_requested(conn, claim["turn_id"]):
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
            row = conn.execute(sa_text(f"""
                UPDATE inter_bot_deliveries SET
                    status='QUEUED', available_at=now() + (:delay * interval '1 second'),
                    next_retry_at=now() + (:delay * interval '1 second'),
                    lease_expires_at=NULL, claim_token=NULL, claim_owner=NULL,
                    last_error=:error, updated_at=now(){mode_reset}{steer_rejected}{attempt_refund}
                WHERE id=:id AND status IN ('STEERING','DISPATCHING') AND claim_token=:claim_token
                RETURNING *
            """), values).mappings().first()
            if row and row.get("reset_status") != "COMPLETED":
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
            head = conn.execute(sa_text("""
                SELECT * FROM inter_bot_deliveries
                WHERE id=:id AND status='DISPATCHING' AND claim_token=:claim
                FOR UPDATE
            """), {"id": delivery_id, "claim": claim_token}).mappings().first()
            if not head:
                return None
            # Generic overflow recovery rotates the active personal thread and
            # changes the logical RPC. Automation must never take that path.
            if (head.get("metadata_json") or {}).get("prompt_schedule"):
                return None
            payload = head["payload_json"] if isinstance(head["payload_json"], dict) else {}
            user_id = str(payload.get("user") or getattr(self.config, "DEFAULT_USER", "nick"))
            row = recover_delivery_after_overflow(
                conn,
                head=head,
                backend=backend,
                user_id=user_id,
                error=error,
            )
        return DeliveryRecord.from_mapping(row) if row else None

    def fail_claim(self, delivery_id: str, claim_token: str, error: str) -> DeliveryRecord | None:
        if self.engine is None or not claim_token:
            return None
        with self.engine.begin() as conn:
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
            row = conn.execute(sa_text("""
                UPDATE inter_bot_deliveries SET status='CANCELLED',
                    lease_expires_at=NULL, claim_token=NULL, claim_owner=NULL,
                    next_retry_at=NULL, updated_at=now()
                WHERE id=:id AND status='QUEUED' RETURNING *
            """), {"id": delivery_id}).mappings().first()
            if row:
                conn.execute(sa_text(
                    "DELETE FROM turn_logs WHERE id=:turn_id AND status='reserved'"
                ), {"turn_id": row["turn_id"]})
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
        if self.engine is None or not claim_token:
            return False
        with self.engine.connect() as conn:
            row = conn.execute(sa_text("""
                SELECT payload_json FROM inter_bot_deliveries
                WHERE id=:id AND status IN ('STEERING','DISPATCHING') AND claim_token=:claim_token
                  AND target_bot_id=:target AND turn_id=:turn_id
                  AND user_message_id=:message_id
            """), {
                "id": delivery_id,
                "claim_token": claim_token,
                "target": target_bot_id,
                "turn_id": turn_id,
                "message_id": user_message_id,
            }).scalar_one_or_none()
        return bool(
            isinstance(row, dict)
            and row.get("inter_bot_bridge_request_id") == bridge_request_id
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

    def recover_expired(self) -> list[DeliveryRecord]:
        """Recover only expired claims; singleton fencing excludes live peers."""
        if self.engine is None:
            return []
        recovered: list[DeliveryRecord] = []
        with self.engine.begin() as conn:
            rows = conn.execute(sa_text("""
                SELECT d.*, t.status AS turn_status, t.ended_at AS turn_ended_at,
                       t.error_text AS turn_error_text
                FROM inter_bot_deliveries d
                LEFT JOIN turn_logs t ON t.id=d.turn_id
                WHERE d.status IN ('STEERING','DISPATCHING')
                  AND d.lease_expires_at < now()
                ORDER BY d.ordinal FOR UPDATE OF d
            """)).mappings().all()
            for row in rows:
                is_turn = row.get("delivery_mode") != "steer"
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
                if updated["status"] == QUEUED and is_turn:
                    conn.execute(sa_text("""
                        UPDATE turn_logs SET status='reserved', ended_at=NULL,
                            end_reason=NULL, error_text=NULL
                        WHERE id=:turn_id AND ended_at IS NULL
                    """), {"turn_id": row["turn_id"]})
                recovered.append(DeliveryRecord.from_mapping(updated))
        return recovered
