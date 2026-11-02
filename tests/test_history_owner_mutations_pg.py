"""TASK-1778: destructive proofs exclusively in the runner's private PostgreSQL."""
import uuid

import pytest
from sqlalchemy import text

from llm_bawt.memory.postgresql import PostgreSQLShortTermManager, PARENT_TABLES, partition_name
from llm_bawt.mcp_server.storage import MemoryStorage
from tests.pg_isolation import require_test_db_config

pytestmark = pytest.mark.integration


@pytest.fixture
def owners(private_app_pg):
    config = require_test_db_config()  # Verifies actual *_test connection before constructors/DML.
    bot = "_t1778_" + uuid.uuid4().hex[:12]
    manager = PostgreSQLShortTermManager(config, bot_id=bot)
    backend = manager._backend
    alice1 = manager.get_or_create_active_session(bot_id=bot, user_id="user_a")
    alice2 = manager.rotate_session(bot_id=bot, user_id="user_a", working_memory_mode="pinned")
    bob = manager.get_or_create_active_session(bot_id=bot, user_id="user_b")
    table = backend._messages_table_name
    rows = [("a_user", "user", 1, alice1), ("a_answer", "assistant", 2, alice1),
            ("a_summary", "summary", 0.5, alice1), ("a_new", "assistant", 4, alice2),
            ("b_answer", "assistant", 9, bob), ("b_summary", "summary", 8, bob),
            ("legacy", "assistant", 10, None)]
    with backend.engine.begin() as conn:
        for label, role, timestamp, session in rows:
            conn.execute(text(f"""INSERT INTO {table}
                (id, bot_id, role, content, timestamp, session_id, author_entity_type, author_entity_id)
                VALUES (:id, :bot, :role, :content, :ts, :sid, 'bot', :bot)"""),
                {"id": str(uuid.uuid4()), "bot": bot, "role": role, "content": label,
                 "ts": timestamp, "sid": session})
        conn.execute(text(f"INSERT INTO {backend._memories_table_name} (id, bot_id, content, created_at) "
                          "VALUES (:id, :bot, 'semantic_sentinel', CURRENT_TIMESTAMP)"),
                     {"id": str(uuid.uuid4()), "bot": bot})
    storage = MemoryStorage.__new__(MemoryStorage)
    storage.get_short_term_manager = lambda *a, **kw: manager
    storage._get_backend = lambda *a, **kw: backend
    storage.hydrate_message_authors = lambda rows, **kw: rows
    yield bot, manager, storage
    with backend.engine.begin() as conn:
        for parent in PARENT_TABLES:
            conn.execute(text(f"DROP TABLE IF EXISTS {partition_name(parent, bot)}"))
        conn.execute(text("DELETE FROM sessions WHERE bot_id=:bot"), {"bot": bot})


def contents(manager):
    with manager._backend.engine.connect() as conn:
        return set(conn.execute(text(f"SELECT content FROM {manager._backend._messages_table_name}")).scalars())


def assert_semantic_memory_retained(manager):
    with manager._backend.engine.connect() as conn:
        assert conn.execute(text(f"SELECT content FROM {manager._backend._memories_table_name}")).scalars().all() == ["semantic_sentinel"]


@pytest.mark.parametrize("via_storage", [False, True])
@pytest.mark.asyncio
async def test_clear_preserves_other_user_and_semantic_memories(owners, via_storage):
    bot, manager, storage = owners
    if via_storage:
        assert await storage.clear_messages(bot_id=bot, user_id="user_a") == 4
        assert await storage.clear_messages(bot_id=bot, user_id="user_a") == 0
    else:
        manager.user_id = "user_a"
        assert manager.clear() is True
    assert contents(manager) == {"b_answer", "b_summary", "legacy"}
    assert_semantic_memory_retained(manager)


@pytest.mark.parametrize("via_storage", [False, True])
@pytest.mark.asyncio
async def test_partial_selects_latest_within_owner_before_role(owners, via_storage):
    bot, manager, storage = owners
    async def remove():
        if via_storage:
            return await storage.remove_last_message_if_partial(bot_id=bot, role="assistant", user_id="user_a")
        return manager.remove_last_message_if_partial("assistant", user_id="user_a")
    assert await remove() is True
    assert contents(manager) == {"a_user", "a_answer", "a_summary", "b_answer", "b_summary", "legacy"}
    assert await remove() is True
    assert await remove() is False  # Latest owned row is now USER, not older summary/foreign assistant.
    assert contents(manager) == {"a_user", "a_summary", "b_answer", "b_summary", "legacy"}
    assert_semantic_memory_retained(manager)


@pytest.mark.asyncio
@pytest.mark.parametrize("missing", [None, "", " "])
async def test_missing_owner_fails_closed_before_mutation(owners, missing):
    bot, manager, storage = owners
    before = contents(manager)
    for mutate in [lambda: manager.clear(user_id=missing),
                   lambda: manager.remove_last_message_if_partial("assistant", user_id=missing)]:
        with pytest.raises(ValueError, match="requires user_id"):
            mutate()
    for mutate in [storage.clear_messages(bot_id=bot, user_id=missing),
                   storage.remove_last_message_if_partial(bot_id=bot, user_id=missing)]:
        with pytest.raises(ValueError, match="requires user_id"):
            await mutate
    assert contents(manager) == before
    assert_semantic_memory_retained(manager)


@pytest.mark.asyncio
@pytest.mark.parametrize("user,expected", [("user_a", {"a_user", "a_answer", "a_new", "a_summary"}),
                                          ("user_b", {"b_answer", "b_summary"})])
async def test_two_user_storage_reads_include_only_owned_summaries(owners, user, expected):
    bot, _, storage = owners
    rows = await storage.get_messages(bot_id=bot, user_id=user)
    assert {row["content"] for row in rows} == expected
    rows = await storage.get_messages(bot_id=bot, user_id=user, summaries_only=True)
    assert {row["content"] for row in rows} == {x for x in expected if x.endswith("summary")}


@pytest.mark.asyncio
async def test_explicit_internal_operator_clear_all_is_still_bot_wide(owners):
    bot, manager, storage = owners
    assert await storage.clear_all_messages(bot_id=bot) == 7
    assert contents(manager) == set()
    assert_semantic_memory_retained(manager)
