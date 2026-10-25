"""Private PostgreSQL regressions for session context projections (TASK-1188)."""
from __future__ import annotations
import json
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
import pytest
from sqlalchemy import event, text
from llm_bawt.agent_context import AgentContextStore
from llm_bawt.inter_bot_delivery import InterBotDeliveryStore
from llm_bawt.memory.postgresql import MESSAGES_PARENT, PARENT_TABLES, ensure_bot_partitions, partition_name
from tests.pg_isolation import require_test_db_config
pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]

@pytest.fixture
def rig():
    config = require_test_db_config()
    InterBotDeliveryStore(config)  # The job checks the durable queue too.
    store = AgentContextStore(config)
    bot = f"testidle{uuid.uuid4().hex}"
    user = f"owner-{uuid.uuid4().hex}"
    session_id = str(uuid.uuid4())
    message_id = str(uuid.uuid4())
    table = partition_name(MESSAGES_PARENT, bot)
    from llm_bawt.runtime_settings import BotProfileStore

    profiles = BotProfileStore(config)
    assert profiles.engine is store.engine
    profiles.upsert({"slug": bot, "name": "Disposable idle context target"})
    with store.engine.begin() as conn:
        ensure_bot_partitions(conn, bot)
        conn.execute(text("""
            INSERT INTO sessions (id, bot_id, user_id, status, started_at)
            VALUES (:id, :bot, :user, 'active', now() - interval '2 hours')
        """), {"id": session_id, "bot": bot, "user": user})
        conn.execute(text(f"""
            INSERT INTO {table} (bot_id, id, role, content, timestamp, session_id)
            VALUES (:bot, :id, 'user', 'test', 1, :session)
        """), {"bot": bot, "id": message_id, "session": session_id})
    service = SimpleNamespace(config=config)
    job = SimpleNamespace(engine=store.engine, config=config, service=service, context=store)
    try:
        yield job, bot, user, session_id, message_id
    finally:
        with store.engine.begin() as conn:
            conn.execute(text("DELETE FROM inter_bot_deliveries WHERE target_bot_id=:bot"), {"bot": bot})
            conn.execute(text("DELETE FROM turn_logs WHERE bot_id=:bot"), {"bot": bot})
            for parent in PARENT_TABLES:
                conn.execute(text(f"DROP TABLE IF EXISTS {partition_name(parent, bot)}"))
            conn.execute(text("DELETE FROM sessions WHERE bot_id=:bot"), {"bot": bot})
            conn.execute(text("DELETE FROM bot_profiles WHERE slug=:bot"), {"bot": bot})



def add_turn(rig, *, tokens=100001, minutes_ago=16, finished=True):
    job, bot, user, _session_id, message_id = rig
    ended = datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)
    turn_id = f"turn-{uuid.uuid4().hex}"
    with job.engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO turn_logs (
                id, created_at, ended_at, path, stream, bot_id, user_id,
                status, user_prompt, response_text, trigger_message_id, token_usage_json
            ) VALUES (
                :id, :created, :ended, '/test', false, :bot, :user, 'ok',
                'test', 'done', :message, :usage
            )
        """), {
            "id": turn_id, "created": ended - timedelta(minutes=1),
            "ended": ended if finished else None, "bot": bot, "user": user,
            "message": message_id, "usage": json.dumps({"resident_tokens": tokens}),
        })
    return turn_id



def test_completion_writes_bot_owner_session_snapshot(rig):
    from llm_bawt.service.turn_logs import TurnLogStore

    job, bot, user, old, _ = rig
    turn_id = add_turn(rig, finished=False)
    store = TurnLogStore(job.config)
    store.update_turn(turn_id=turn_id, status="ok", end_reason="stop",
                      token_usage={"resident_tokens": 100001},
                      agent_session_key="test-provider-key")
    with job.engine.connect() as conn:
        snapshot = conn.execute(text("""
            SELECT session_metadata->'context_stats' FROM sessions WHERE id=:id
        """), {"id": old}).scalar_one()
    assert snapshot["turn_id"] == turn_id
    assert snapshot["usage"]["resident_tokens"] == 100001
    assert snapshot["agent_session_key"] == "test-provider-key"
    assert snapshot["ended_at"] is not None
    statements = []

    def capture(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)

    event.listen(job.engine, "before_cursor_execute", capture)
    try:
        health = job.context.health(bot_id=bot, user_id=user, backend="claude-code",
                                    configured_ceiling=372000)
    finally:
        event.remove(job.engine, "before_cursor_execute", capture)
    assert health["resident_prompt_tokens"] == 100001
    assert len(statements) == 1 and "JOIN messages" not in statements[0]



def test_missing_trigger_defers_projection_until_owner_is_known(rig):
    from llm_bawt.service.turn_logs import TurnLogStore

    job, bot, user, active, _ = rig
    old = add_turn(rig, finished=False)
    store = TurnLogStore(job.config)
    store.update_turn(turn_id=old, status="ok", end_reason="stop",
                      token_usage={"resident_tokens": 120001})
    missing_message = str(uuid.uuid4())
    pending = f"turn-{uuid.uuid4().hex}"
    store.save_turn(
        turn_id=pending, request_id=None, path="/test", stream=True,
        model=None, bot_id=bot, user_id=user, status="streaming", latency_ms=None,
        user_prompt="pending", request_payload={"messages": [{"role": "user", "id": missing_message}]},
        response_text="", tool_calls=None, trigger_message_id=missing_message,
    )
    health = job.context.health(bot_id=bot, user_id=user, backend="claude-code",
                                configured_ceiling=372000)
    # The trigger has no owner yet: never guess which session to invalidate.
    assert health["usage_turn_id"] == old
    assert health["resident_prompt_tokens"] == 120001
    with job.engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO messages (bot_id, id, role, content, timestamp, session_id)
            VALUES (:bot, :id, 'user', 'pending', 1, :session)
        """), {"bot": bot, "id": missing_message, "session": active})
    store.update_turn(turn_id=pending, status="ok", end_reason="stop",
                      token_usage={"resident_tokens": 5000})
    assert job.context.health(bot_id=bot, user_id=user, backend="claude-code",
                              configured_ceiling=372000)["resident_prompt_tokens"] == 5000



def test_archived_trigger_does_not_invalidate_active_snapshot(rig):
    from llm_bawt.service.turn_logs import TurnLogStore

    job, bot, user, active, _ = rig
    old = add_turn(rig, finished=False)
    store = TurnLogStore(job.config)
    store.update_turn(turn_id=old, status="ok", end_reason="stop",
                      token_usage={"resident_tokens": 120001})
    archived, message = str(uuid.uuid4()), str(uuid.uuid4())
    with job.engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO sessions (id, bot_id, user_id, status, started_at, ended_at)
            VALUES (:id, :bot, :user, 'archived', now()-interval '2 hours', now()-interval '1 hour')
        """), {"id": archived, "bot": bot, "user": user})
    archived_turn = f"turn-{uuid.uuid4().hex}"
    store.save_turn(
        turn_id=archived_turn, request_id=None, path="/test",
        stream=False, model=None, bot_id=bot, user_id=user, status="streaming",
        latency_ms=None, user_prompt="archived",
        request_payload={"messages": [{"role": "user", "id": message}]},
        response_text="", tool_calls=None, trigger_message_id=message,
    )
    with job.engine.connect() as conn:
        snapshot = conn.execute(text("""
            SELECT session_metadata->'context_stats' FROM sessions WHERE id=:id
        """), {"id": active}).scalar_one()
    assert snapshot["turn_id"] == old
    with job.engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO messages (bot_id, id, role, content, timestamp, session_id)
            VALUES (:bot, :id, 'user', 'archived', 1, :session)
        """), {"bot": bot, "id": message, "session": archived})
    store.update_turn(turn_id=archived_turn, status="error", end_reason="upstream_error")
    with job.engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT id, session_metadata->'context_stats' AS snapshot
            FROM sessions WHERE id IN (:active, :archived)
        """), {"active": active, "archived": archived}).mappings().all()
    snapshots = {row["id"]: row["snapshot"] for row in rows}
    assert snapshots[active]["turn_id"] == old
    assert snapshots[active]["usage"]["resident_tokens"] == 120001
    assert snapshots[archived]["turn_id"] == archived_turn



def test_admission_and_terminal_insert_invalidate_older_session_snapshot(rig):
    from llm_bawt.service.turn_logs import TurnLogStore

    job, bot, user, old, _ = rig
    first = add_turn(rig, finished=False)
    store = TurnLogStore(job.config)
    store.update_turn(turn_id=first, status="ok", end_reason="stop",
                      token_usage={"resident_tokens": 120001})
    message_id = str(uuid.uuid4())
    with job.engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO messages (bot_id, id, role, content, timestamp, session_id)
            VALUES (:bot, :id, 'user', 'later', 1, :session)
        """), {"bot": bot, "id": message_id, "session": old})
    second = f"turn-{uuid.uuid4().hex}"
    store.save_turn(
        turn_id=second, request_id=None, path="/test", stream=False,
        model=None, bot_id=bot, user_id=user, status="streaming", latency_ms=None,
        user_prompt="later", request_payload={"messages": [{"role": "user", "id": message_id}]},
        response_text="", tool_calls=None, trigger_message_id=message_id,
    )
    health = job.context.health(bot_id=bot, user_id=user, backend="claude-code",
                                configured_ceiling=372000)
    assert health["usage_turn_id"] == second
    assert health["resident_prompt_tokens"] is None
    store.update_turn(turn_id=second, status="error", end_reason="upstream_error")
    third_message_id = str(uuid.uuid4())
    with job.engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO messages (bot_id, id, role, content, timestamp, session_id)
            VALUES (:bot, :id, 'user', 'terminal', 1, :session)
        """), {"bot": bot, "id": third_message_id, "session": old})
    third = f"turn-{uuid.uuid4().hex}"
    store.save_turn(
        turn_id=third, request_id=None, path="/test", stream=False,
        model=None, bot_id=bot, user_id=user, status="error", latency_ms=None,
        user_prompt="terminal", request_payload={"messages": [{"role": "user", "id": third_message_id}]},
        response_text="", tool_calls=None, trigger_message_id=third_message_id,
    )
    health = job.context.health(bot_id=bot, user_id=user, backend="claude-code",
                                configured_ceiling=372000)
    assert health["usage_turn_id"] == third and health["resident_prompt_tokens"] is None
    with job.engine.connect() as conn:
        snapshot = conn.execute(text("""
            SELECT session_metadata->'context_stats' FROM sessions WHERE id=:id
        """), {"id": old}).scalar_one()
    assert snapshot["turn_id"] == third and snapshot["usage"] is None



def test_completion_snapshot_is_owner_scoped(rig):
    from llm_bawt.service.turn_logs import TurnLogStore

    job, bot, user, old, _ = rig
    other_user = f"other-{uuid.uuid4().hex}"
    other_session, message_id = str(uuid.uuid4()), str(uuid.uuid4())
    with job.engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO sessions (id, bot_id, user_id, status, started_at)
            VALUES (:id, :bot, :user, 'active', now()-interval '1 hour')
        """), {"id": other_session, "bot": bot, "user": other_user})
        conn.execute(text("""
            INSERT INTO messages (bot_id, id, role, content, timestamp, session_id)
            VALUES (:bot, :id, 'user', 'other', 1, :session)
        """), {"bot": bot, "id": message_id, "session": other_session})
    TurnLogStore(job.config).save_turn(
        turn_id=f"turn-{uuid.uuid4().hex}", request_id=None,
        path="/test", stream=False, model=None, bot_id=bot, user_id=other_user,
        status="ok", latency_ms=None, user_prompt="other",
        request_payload={"messages": [{"role": "user", "id": message_id}]},
        response_text="ok", tool_calls=None, trigger_message_id=message_id,
        token_usage={"resident_tokens": 120001},
    )
    with job.engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT id, session_metadata->'context_stats' AS snapshot
            FROM sessions WHERE id IN (:old, :other)
        """), {"old": old, "other": other_session}).mappings().all()
    snapshots = {row["id"]: row["snapshot"] for row in rows}
    assert snapshots[old] is None
    assert snapshots[other_session]["usage"]["resident_tokens"] == 120001
    assert job.context.health(bot_id=bot, user_id=user, backend="claude-code",
                              configured_ceiling=372000)["resident_prompt_tokens"] is None



def test_late_older_completion_cannot_overwrite_tied_newer_turn(rig):
    from llm_bawt.service.turn_logs import TurnLogStore

    job, bot, user, old, message_id = rig
    older, newer = "turn-" + "a" * 32, "turn-" + "f" * 32
    with job.engine.begin() as conn:
        for turn_id in (older, newer):
            conn.execute(text("""
                INSERT INTO turn_logs (id, created_at, path, stream, bot_id,
                    user_id, status, user_prompt, response_text, trigger_message_id)
                VALUES (:id, now()-interval '16 minutes', '/test', false,
                    :bot, :user, 'streaming', '', '', :message)
            """), {"id": turn_id, "bot": bot, "user": user,
                   "message": message_id})
    store = TurnLogStore(job.config)
    store.update_turn(turn_id=newer, status="ok", end_reason="stop",
                      token_usage={"resident_tokens": 120001})
    store.update_turn(turn_id=older, status="ok", end_reason="stop",
                      token_usage={"resident_tokens": 5000})
    health = job.context.health(bot_id=bot, user_id=user, backend="claude-code",
                                configured_ceiling=372000)
    assert health["usage_turn_id"] == newer
    assert health["resident_prompt_tokens"] == 120001
    with job.engine.connect() as conn:
        snapshot = conn.execute(text("""
            SELECT session_metadata->'context_stats' FROM sessions WHERE id=:id
        """), {"id": old}).scalar_one()
    assert snapshot["turn_id"] == newer



def test_orphan_reap_invalidates_older_session_snapshot(rig):
    from llm_bawt.service.turn_logs import TurnLogStore

    job, bot, user, old, message_id = rig
    first = add_turn(rig, finished=False)
    store = TurnLogStore(job.config)
    store.update_turn(turn_id=first, status="ok", end_reason="stop",
                      token_usage={"resident_tokens": 120001})
    abandoned = f"turn-{uuid.uuid4().hex}"
    current = f"turn-{uuid.uuid4().hex}"
    with job.engine.begin() as conn:
        for turn_id, minutes in ((abandoned, 2), (current, 1)):
            conn.execute(text("""
                INSERT INTO turn_logs (id, created_at, path, stream, bot_id,
                    user_id, status, user_prompt, response_text, trigger_message_id)
                VALUES (:id, now()-make_interval(mins=>:minutes), '/test',
                    false, :bot, :user, 'streaming', '', '', :message)
            """), {"id": turn_id, "minutes": minutes, "bot": bot,
                   "user": user, "message": message_id})
    assert store.reap_other_open_turns(bot_id=bot, current_turn_id=current) == [
        {"id": abandoned, "user_id": user, "balance_update": None}
    ]
    health = job.context.health(bot_id=bot, user_id=user, backend="claude-code",
                                configured_ceiling=372000)
    assert health["usage_turn_id"] == abandoned
    assert health["resident_prompt_tokens"] is None
    with job.engine.connect() as conn:
        conn.execute(text("DELETE FROM turn_logs WHERE id=:id"), {"id": current})
