"""TASK-259: the PostgreSQL fork loader in the runner's private PostgreSQL."""
import uuid

import pytest
from sqlalchemy import text

from llm_bawt.memory.postgresql import PostgreSQLShortTermManager, PARENT_TABLES, partition_name
from tests.pg_isolation import require_test_db_config

pytestmark = pytest.mark.integration


@pytest.fixture
def threads(private_app_pg):
    config = require_test_db_config()
    bot = "_t259_" + uuid.uuid4().hex[:12]
    manager = PostgreSQLShortTermManager(config, bot_id=bot)
    backend = manager._backend
    older = manager.get_or_create_active_session(bot_id=bot, user_id="user_a")
    parent = manager.rotate_session(bot_id=bot, user_id="user_a", working_memory_mode="pinned")
    fork = manager.rotate_session(bot_id=bot, user_id="user_a", working_memory_mode="pinned")
    other = manager.get_or_create_active_session(bot_id=bot, user_id="user_b")
    ids = {}
    rows = [
        ("old_summary", "summary", 5, older, False),
        ("p1", "user", 10, parent, True),       # summarized before the fork was made
        ("p2", "assistant", 11, parent, True),
        ("p3", "user", 12, parent, False),      # fork point
        ("p4", "assistant", 13, parent, False),  # discarded branch
        ("late_summary", "summary", 13, parent, False),
        ("f1", "user", 20, fork, False),
        ("b1", "user", 12, other, False),
    ]
    table = backend._messages_table_name
    with backend.engine.begin() as conn:
        for label, role, ts, sid, summarized in rows:
            ids[label] = str(uuid.uuid4())
            conn.execute(text(f"""INSERT INTO {table}
                (id, bot_id, role, content, timestamp, session_id, summarized,
                 author_entity_type, author_entity_id)
                VALUES (:id, :bot, :role, :content, :ts, :sid, :summarized, 'bot', :bot)"""),
                {"id": ids[label], "bot": bot, "role": role, "content": label, "ts": ts,
                 "sid": sid, "summarized": summarized})
    yield manager, ids, parent, fork, other
    with backend.engine.begin() as conn:
        for p in PARENT_TABLES:
            conn.execute(text(f"DROP TABLE IF EXISTS {partition_name(p, bot)}"))
        conn.execute(text("DELETE FROM sessions WHERE bot_id=:bot"), {"bot": bot})


def test_fork_pool_through_the_cut_including_summarized_rows(threads):
    manager, ids, parent, fork, _other = threads
    pool = manager.load_fork_scoped(session_id=fork, parent_session_id=parent,
                                    at_message_id=ids["p3"], user_id="user_a")
    assert [m.content for m in pool] == ["old_summary", "p1", "p2", "p3", "f1"]


def test_fork_of_a_fork_reads_the_grandparent_through_its_cut(threads):
    # TASK-2453: the fork itself becomes a parent; its pool must still carry
    # the original parent's history through p3.
    manager, ids, parent, fork, _other = threads
    backend = manager._backend
    child = manager.rotate_session(bot_id=manager.bot_id, user_id="user_a",
                                   working_memory_mode="pinned")
    with backend.engine.begin() as conn:
        conn.execute(text("""UPDATE sessions SET session_metadata =
            COALESCE(session_metadata, '{}'::jsonb) || jsonb_build_object(
              'forked_from', jsonb_build_object('session_id', :p, 'at_message_id', :at))
            WHERE id = :fork"""), {"p": parent, "at": ids["p3"], "fork": fork})
        conn.execute(text(f"""INSERT INTO {backend._messages_table_name}
            (id, bot_id, role, content, timestamp, session_id, summarized,
             author_entity_type, author_entity_id)
            VALUES (:id, :bot, 'user', 'c1', 30, :sid, false, 'bot', :bot)"""),
            {"id": str(uuid.uuid4()), "bot": manager.bot_id, "sid": child})
    pool = manager.load_fork_scoped(session_id=child, parent_session_id=fork,
                                    at_message_id=ids["f1"], user_id="user_a")
    assert [m.content for m in pool] == ["old_summary", "p1", "p2", "p3", "f1", "c1"]


def test_fork_from_another_users_thread_reads_nothing(threads):
    manager, ids, _parent, fork, other = threads
    assert manager.load_fork_scoped(session_id=fork, parent_session_id=other,
                                    at_message_id=ids["b1"], user_id="user_a") is None
