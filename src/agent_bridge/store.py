from __future__ import annotations
import logging

from sqlalchemy import text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)


def create_agent_event_tables(engine: Engine) -> None:
    """Create agent-bridge tables if they don't exist.

    Also performs a one-time, idempotent rename of the legacy
    ``openclaw_*`` tables (and their indexes) into the new
    ``agent_*`` names. Safe to run on fresh DBs and on already-migrated
    DBs — both branches are guarded with ``IF EXISTS`` / ``IF NOT EXISTS``.
    """
    with engine.connect() as conn:
        # ---- One-time rename of legacy openclaw_* tables --------------
        # Each statement is independent and idempotent.
        conn.execute(text("ALTER TABLE IF EXISTS openclaw_events RENAME TO agent_events"))
        conn.execute(text("ALTER TABLE IF EXISTS openclaw_session_state RENAME TO agent_session_state"))
        conn.execute(text("ALTER TABLE IF EXISTS openclaw_runs RENAME TO agent_runs"))
        conn.execute(text("ALTER INDEX IF EXISTS idx_openclaw_events_session_created RENAME TO idx_agent_events_session_created"))
        conn.execute(text("ALTER INDEX IF EXISTS idx_openclaw_events_session_run RENAME TO idx_agent_events_session_run"))
        conn.execute(text("ALTER INDEX IF EXISTS idx_openclaw_events_session_seq RENAME TO idx_agent_events_session_seq"))

        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS agent_events (
                id BIGSERIAL PRIMARY KEY,
                event_dedupe_key VARCHAR(255) NOT NULL UNIQUE,
                session_key VARCHAR(255) NOT NULL,
                run_id VARCHAR(255),
                seq BIGINT,
                kind VARCHAR(50) NOT NULL,
                origin VARCHAR(50) NOT NULL DEFAULT 'unknown',
                text TEXT,
                tool_name VARCHAR(255),
                payload_json JSONB NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_agent_events_session_created
            ON agent_events (session_key, created_at)
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_agent_events_session_run
            ON agent_events (session_key, run_id)
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_agent_events_session_seq
            ON agent_events (session_key, seq)
        """))

        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS agent_session_state (
                session_key VARCHAR(255) PRIMARY KEY,
                last_event_id BIGINT,
                last_cursor VARCHAR(255),
                ws_connected BOOLEAN NOT NULL DEFAULT FALSE,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """))

        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS agent_runs (
                run_id VARCHAR(255) PRIMARY KEY,
                session_key VARCHAR(255) NOT NULL,
                status VARCHAR(50) NOT NULL DEFAULT 'running',
                model VARCHAR(255),
                origin VARCHAR(50),
                full_text TEXT,
                tool_calls_json JSONB,
                started_at TIMESTAMPTZ,
                completed_at TIMESTAMPTZ,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """))
        conn.commit()
    logger.info("Agent-bridge tables ensured (renamed legacy openclaw_* if present)")
