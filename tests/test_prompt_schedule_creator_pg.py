"""Private-PG legacy creator upgrade; no operational rows are read or changed."""
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import text

from llm_bawt.service.prompt_schedule_management import (
    schedule_identifier, upgrade_prompt_schedules, wakeup_create_key,
)
from llm_bawt.service.turn_logs import TurnLogStore
from tests.pg_isolation import require_test_db_config

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]


def test_legacy_creator_upgrade_preserves_schedules_and_never_reinfers_new_rows():
    engine = TurnLogStore(require_test_db_config()).engine
    with engine.begin() as conn:
        # All three tables are local temporary fixtures. Shadowing prevents
        # even incidental reads of the private app's persistent schedule data.
        conn.execute(text("""
            CREATE TEMP TABLE prompt_schedules (
                job_id VARCHAR PRIMARY KEY, owner_user_id VARCHAR NOT NULL,
                name VARCHAR NOT NULL, lifecycle VARCHAR NOT NULL, revision INTEGER,
                timing_json JSON, dedicated_session_id VARCHAR
            ) ON COMMIT DROP
        """))
        conn.execute(text("""
            CREATE TEMP TABLE scheduled_jobs (
                id VARCHAR PRIMARY KEY, bot_id VARCHAR, enabled BOOLEAN, next_run_at TIMESTAMPTZ
            ) ON COMMIT DROP
        """))
        conn.execute(text("""
            CREATE TEMP TABLE prompt_occurrences (
                run_id VARCHAR PRIMARY KEY, job_id VARCHAR, occurrence_key VARCHAR,
                state VARCHAR, snapshot_json JSON
            ) ON COMMIT DROP
        """))
        now = datetime(2026, 11, 9, tzinfo=timezone.utc)
        ids = {}
        for bot in ["al", "snark", "byte"]:
            nonce = f"old-{bot}"
            identifier = schedule_identifier("test-owner", wakeup_create_key(bot, nonce))
            ids[identifier] = bot
            conn.execute(text("""
                INSERT INTO prompt_schedules VALUES
                (:id, 'test-owner', :name, 'active', 7,
                 '{"kind":"cron","cron_expression":"0 */3 * * *","timezone":"UTC"}', 'existing-session')
            """), {"id": identifier, "name": f"wakeup:{bot}:{nonce}:old"})
            conn.execute(text("INSERT INTO scheduled_jobs VALUES (:id, :bot, true, :due)"),
                         {"id": identifier, "bot": bot, "due": now})
            for n, state in enumerate(["succeeded", "pending"]):
                conn.execute(text("""
                    INSERT INTO prompt_occurrences VALUES
                    (:run, :job, :key, :state, '{"original":"snapshot"}')
                """), {"run": f"{bot}-{n}", "job": identifier,
                       "key": (now - timedelta(hours=3*n)).isoformat(), "state": state})
        conn.execute(text("""
            INSERT INTO prompt_schedules VALUES
            ('handmade', 'test-owner', 'wakeup:al:fake:handmade', 'paused', 1, '{}', NULL)
        """))
        before_schedules = conn.execute(text("SELECT * FROM prompt_schedules ORDER BY job_id")).all()
        before_jobs = conn.execute(text("SELECT * FROM scheduled_jobs ORDER BY id")).all()
        before_occurrences = conn.execute(text("SELECT * FROM prompt_occurrences ORDER BY run_id")).all()
        upgrade_prompt_schedules(conn, "postgresql")
        creators = dict(conn.execute(text("SELECT job_id, creator_bot_id FROM prompt_schedules")).all())
        assert creators == {**ids, "handmade": None}
        assert conn.execute(text("""
            SELECT job_id, owner_user_id, name, lifecycle, revision, timing_json, dedicated_session_id
            FROM prompt_schedules ORDER BY job_id
        """)).all() == before_schedules
        # A new user row can exactly reproduce the public formula. Subsequent
        # startups must not promote it, or reassign an already persisted creator.
        forged = schedule_identifier("test-owner", wakeup_create_key("al", "new"))
        conn.execute(text("""
            INSERT INTO prompt_schedules(job_id, owner_user_id, name, lifecycle)
            VALUES (:id, 'test-owner', 'wakeup:al:new:forged', 'active')
        """), {"id": forged})
        existing = next(identifier for identifier, bot in ids.items() if bot == "snark")
        conn.execute(text("UPDATE prompt_schedules SET name='Completely renamed' WHERE job_id=:id"),
                     {"id": existing})
        upgrade_prompt_schedules(conn, "postgresql")
        upgrade_prompt_schedules(conn, "postgresql")
        assert dict(conn.execute(text("SELECT job_id, creator_bot_id FROM prompt_schedules")).all()) == {
            **ids, "handmade": None, forged: None}
        assert conn.execute(text("SELECT * FROM scheduled_jobs ORDER BY id")).all() == before_jobs
        assert conn.execute(text("SELECT * FROM prompt_occurrences ORDER BY run_id")).all() == before_occurrences
