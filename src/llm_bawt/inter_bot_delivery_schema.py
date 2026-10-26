"""Idempotent DDL for ``inter_bot_deliveries`` (split from inter_bot_delivery.py).

``InterBotDeliveryStore._ensure_schema`` runs this once per process through
its ``SchemaBootstrapGuard``. Every statement is safe to re-run.
"""

from __future__ import annotations

from sqlalchemy import text as sa_text

from .inter_bot_control_lane import ensure_control_lane_schema


def bootstrap_delivery_schema(conn) -> None:
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
            sender_turn_id VARCHAR(128),
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
    # TASK-1249: the sending bot's signed turn, stamped at admission.
    conn.execute(sa_text(
        "ALTER TABLE inter_bot_deliveries ADD COLUMN IF NOT EXISTS sender_turn_id VARCHAR(128)"
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
