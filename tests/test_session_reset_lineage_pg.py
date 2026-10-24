"""Private PostgreSQL proof for a queued /new delivery's atomic successor."""
from __future__ import annotations

import uuid
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from sqlalchemy import text

from llm_bawt.memory.postgresql_short_term import PostgreSQLShortTermManager
from llm_bawt.service.agent_new_rotation import delivery_new_predecessor
from llm_bawt.service.turn_logs import TurnLogStore
from tests.pg_isolation import isolated_pg_store

pytestmark = pytest.mark.usefixtures("private_app_pg")


@pytest.mark.integration
def test_concurrent_claims_reuse_one_delivery_new_thread():
    engine = isolated_pg_store(TurnLogStore).engine
    if engine is None:
        pytest.skip("private PostgreSQL unavailable")
    bot = f"test-new-{uuid.uuid4().hex}"
    user = "nick"
    old_id = str(uuid.uuid4())
    key = f"delivery-{uuid.uuid4().hex}"
    with engine.begin() as conn:
        # private_app_pg bootstraps sessions through the app-owned schema path.
        conn.execute(text("""INSERT INTO sessions
            (id, bot_id, user_id, status, started_at, session_metadata)
            VALUES (:id, :bot, :user, 'active', now(), '{}'::jsonb)
        """), {"id": old_id, "bot": bot, "user": user})

    def rotate():
        manager = object.__new__(PostgreSQLShortTermManager)
        manager._backend = SimpleNamespace(engine=engine)
        manager.bot_id, manager.user_id = bot, user
        return manager.rotate_session(idempotency_key=key, working_memory_mode="pinned")

    try:
        with ThreadPoolExecutor(max_workers=2) as workers:
            first, second = [future.result(timeout=10) for future in [workers.submit(rotate), workers.submit(rotate)]]
        assert first == second
        with engine.connect() as conn:
            rows = conn.execute(text("""SELECT id, status, session_metadata FROM sessions
                WHERE bot_id=:bot AND user_id=:user"""), {"bot": bot, "user": user}).mappings().all()
        assert len(rows) == 2
        successor = next(row for row in rows if row["id"] == first)
        assert successor["status"] == "active"
        assert successor["session_metadata"]["rotation_delivery_id"] == key
        assert successor["session_metadata"]["reset_predecessor_ids"] == [old_id]
        assert next(row for row in rows if row["id"] == old_id)["session_metadata"]["reset_successor_id"] == first
        # A subsequent active thread must not become this delivery's successor
        # or seed source when a claimed delivery is replayed after a delay.
        later_id = str(uuid.uuid4())
        with engine.begin() as conn:
            conn.execute(text("""UPDATE sessions
                SET status='archived', ended_at=now(), archived_at=now()
                WHERE id=:id AND bot_id=:bot AND user_id=:user
            """), {"id": first, "bot": bot, "user": user})
            conn.execute(text("""INSERT INTO sessions
                (id, bot_id, user_id, status, started_at, session_metadata)
                VALUES (:id, :bot, :user, 'active', now(), '{}'::jsonb)
            """), {"id": later_id, "bot": bot, "user": user})
        assert rotate() == first
        with engine.connect() as conn:
            later = conn.execute(text("SELECT status FROM sessions WHERE id=:id"), {"id": later_id}).scalar_one()
            assert later == "active"
            assert conn.execute(text("""SELECT count(*) FROM sessions
                WHERE bot_id=:bot AND user_id=:user"""), {"bot": bot, "user": user}).scalar_one() == 3
        backend = object.__new__(PostgreSQLShortTermManager)
        backend._backend = SimpleNamespace(engine=engine)
        assert delivery_new_predecessor(
            SimpleNamespace(history_manager=SimpleNamespace(_db_backend=backend)),
            bot_id=bot, user_id=user, delivery_id=key,
        ) == old_id
    finally:
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM sessions WHERE bot_id=:bot AND user_id=:user"),
                         {"bot": bot, "user": user})
