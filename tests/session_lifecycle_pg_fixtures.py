"""Shared private-PostgreSQL lifecycle fixtures, independent of test collection."""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from sqlalchemy import text

from llm_bawt.agent_context import AgentContextStore
from llm_bawt.bot_session_state import BotSessionState
from llm_bawt.memory.postgresql import PARENT_TABLES, ensure_bot_partitions, partition_name
from llm_bawt.runtime_settings import BotProfileStore
from llm_bawt.service.scheduler import create_scheduler_tables
from tests.pg_isolation import require_test_db_config


@pytest.fixture
def rig(private_app_pg):
    config = require_test_db_config()
    store = AgentContextStore(config)
    from llm_bawt.inter_bot_delivery import InterBotDeliveryStore
    InterBotDeliveryStore(config)
    profiles = BotProfileStore(config)
    bot = f"testidle{uuid.uuid4().hex}"
    profiles.upsert({"slug": bot, "name": "Disposable idle context target"})
    session_id, message_id = str(uuid.uuid4()), str(uuid.uuid4())
    user, turn_id = f"owner-{uuid.uuid4().hex}", f"turn-{uuid.uuid4().hex}"
    now = datetime.now(timezone.utc)
    with store.engine.begin() as conn:
        ensure_bot_partitions(conn, bot)
        conn.execute(text("""
            INSERT INTO sessions (id, bot_id, user_id, status, started_at)
            VALUES (:id, :bot, :user, 'active', :started)
        """), {"id": session_id, "bot": bot, "user": user, "started": now - timedelta(hours=2)})
        conn.execute(text("""
            INSERT INTO messages (bot_id, id, role, content, timestamp, session_id)
            VALUES (:bot, :id, 'user', 'test', 1, :session)
        """), {"bot": bot, "id": message_id, "session": session_id})
        conn.execute(text("""
            INSERT INTO turn_logs (id, created_at, ended_at, path, stream,
                bot_id, user_id, status, user_prompt, response_text,
                trigger_message_id, token_usage_json)
            VALUES (:id, :created, :ended, '/test', false, :bot, :user,
                'ok', 'test', 'done', :message, :usage)
        """), {"id": turn_id, "created": now - timedelta(minutes=20),
               "ended": now - timedelta(minutes=16), "bot": bot, "user": user,
               "message": message_id, "usage": json.dumps({"resident_tokens": 100001})})
        BotSessionState.accept(conn, bot_id=bot, session_id=session_id)
    create_scheduler_tables(store.engine)
    fixture = SimpleNamespace(
        config=config, engine=store.engine, bot=bot, user=user, session_id=session_id,
        message_id=message_id, turn_id=turn_id, jobs=[], now=now,
    )
    try:
        yield fixture
    finally:
        with store.engine.begin() as conn:
            for job_id in fixture.jobs:
                conn.execute(text("DELETE FROM job_runs WHERE job_id=:id"), {"id": job_id})
                conn.execute(text("DELETE FROM scheduled_jobs WHERE id=:id"), {"id": job_id})
            conn.execute(text("DELETE FROM inter_bot_deliveries WHERE target_bot_id=:bot"), {"bot": bot})
            conn.execute(text("DELETE FROM turn_logs WHERE bot_id=:bot"), {"bot": bot})
            for parent in PARENT_TABLES:
                conn.execute(text(f"DROP TABLE IF EXISTS {partition_name(parent, bot)}"))
            conn.execute(text("DELETE FROM sessions WHERE bot_id=:bot"), {"bot": bot})
            conn.execute(text("DELETE FROM bot_profiles WHERE slug=:bot"), {"bot": bot})



@pytest.fixture
def real_lifecycle(rig, monkeypatch):
    from llm_bawt.inter_bot_delivery import InterBotDeliveryStore
    from llm_bawt.service.session_lifecycle.lifecycle import NewSessionLifecycle

    store = InterBotDeliveryStore(rig.config)
    fake_bot = SimpleNamespace(slug=rig.bot, agent_backend="claude-code", harness="claude-code")
    monkeypatch.setattr("llm_bawt.service.session_lifecycle.lifecycle.BotManager",
                        lambda config: SimpleNamespace(get_bot=lambda slug: fake_bot))
    service = SimpleNamespace(config=rig.config, _inter_bot_dispatcher=SimpleNamespace(
        store=store, wake=lambda bot: None,
    ))
    return NewSessionLifecycle(service)



def scheduled_intent(rig, *, run_id="test-run"):
    from llm_bawt.message_authorship import AuthorReference
    from llm_bawt.service.session_lifecycle.intent import Admission, Continuity, NewSessionIntent, Origin

    return NewSessionIntent(
        bot_id=rig.bot, user_id=rig.user, author=AuthorReference.bot("system"),
        continuity=Continuity.CONFIGURED, admission=Admission.SKIP, origin=Origin.SCHEDULER,
        idempotency_key=f"test-idle:{rig.session_id}:{rig.turn_id}",
        expected_session_id=rig.session_id, expected_usage_turn_id=rig.turn_id,
        job_id="test-job", run_id=run_id,
    )
