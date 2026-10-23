"""Private-PG authorization/provenance proofs; synthetic bots and isolated schema."""
from __future__ import annotations

import json
import uuid
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, text

from llm_bawt.bot_access import BotAccessError, BotPolicyResolver, TrustedCaller
from llm_bawt.bot_access.provenance import MemoryProvenance, ScopedMessageAuthority
from llm_bawt.memory.postgresql_memories import PostgreSQLSemanticMemoryMixin
from llm_bawt.utils import db
from tests.pg_isolation import require_test_db_config

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]


@pytest.fixture
def rig(private_app_pg):
    url = db.build_postgres_url(require_test_db_config())
    root = create_engine(url)
    schema = "access_" + uuid.uuid4().hex
    with root.begin() as conn:
        conn.execute(text(f"CREATE SCHEMA {schema}"))
    engine = create_engine(url, connect_args={"options": f"-csearch_path={schema},public"})
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE bot_profiles (slug text, include_in_global_search boolean)"))
        conn.execute(text("INSERT INTO bot_profiles VALUES ('public-fixture', true), ('peer', true), ('private_fixture', false)"))
        conn.execute(text("CREATE TABLE sessions (id text, bot_id text, user_id text, status text)"))
        conn.execute(text("""INSERT INTO sessions VALUES
            ('pub', 'public-fixture', 'user_a', 'active'),
            ('priv', 'private_fixture', 'user_a', 'archived'),
            ('other', 'public-fixture', 'user_b', 'active')"""))
        conn.execute(text("""CREATE TABLE messages (
            id text, bot_id text, session_id text, role text,
            author_entity_type text, author_entity_id text)"""))
        conn.execute(text("""CREATE TABLE memories (
            id text PRIMARY KEY, bot_id text, content text, tags jsonb DEFAULT '[]',
            importance float DEFAULT 0.5, source_message_ids jsonb,
            embedding vector(3), meaning_embedding vector(3),
            access_count int DEFAULT 0, last_accessed timestamptz,
            created_at timestamptz DEFAULT now(), intent text, stakes text,
            emotional_charge float)"""))
    instance = SimpleNamespace(engine=engine)
    try:
        yield instance
    finally:
        engine.dispose()
        with root.begin() as conn:
            conn.execute(text(f"DROP SCHEMA {schema} CASCADE"))
        root.dispose()


def add_message(rig, id="good", *, partition="publicfixture", session="pub",
                role="assistant", author_type="bot", author="public-fixture"):
    with rig.engine.begin() as conn:
        conn.execute(text("INSERT INTO messages VALUES (:id, :bot, :session, :role, :type, :author)"),
                     {"id": id, "bot": partition, "session": session,
                      "role": role, "type": author_type, "author": author})


def add_memory(rig, id, sources, *, partition="publicfixture", vector="[1,0,0]"):
    with rig.engine.begin() as conn:
        conn.execute(text("""INSERT INTO memories (id, bot_id, content, source_message_ids, embedding)
            VALUES (:id, :bot, :id, CAST(:sources AS jsonb), CAST(:vector AS vector))"""),
                     {"id": id, "bot": partition, "sources": json.dumps(sources), "vector": vector})


def eligibility(rig, bot="public-fixture"):
    caller = TrustedCaller("private_fixture", "user_a", "service")
    scope = BotPolicyResolver(rig.engine).resolve_sync(caller).aggregate_scope(require_user=True)
    return MemoryProvenance(ScopedMessageAuthority.for_partition(scope, bot))


class Backend(PostgreSQLSemanticMemoryMixin):
    _memories_table_name = "memories"

    def __init__(self, engine, decay):
        self.engine = engine
        self.config = SimpleNamespace(MEMORY_DECAY_ENABLED=decay, MEMORY_DIVERSITY_ENABLED=False)

    def update_memory_access(self, id):
        with self.engine.begin() as conn:
            conn.execute(text("UPDATE memories SET access_count=access_count+1 WHERE id=:id"), {"id": id})


@pytest.mark.parametrize("decay", [True, False])
def test_provenance_precedes_both_limits_and_denied_access_writes(rig, decay):
    add_message(rig)
    add_message(rig, "bad", role="user", author="private_fixture")
    for n in range(12):
        add_memory(rig, f"denied-{n}", ["good", "bad"])
    add_memory(rig, "allowed", ["good"], vector="[0.9,0.1,0]")
    result = Backend(rig.engine, decay).search_memories_by_embedding(
        [1, 0, 0], n_results=1, eligibility=eligibility(rig))
    assert [row["id"] for row in result] == ["allowed"]
    with rig.engine.connect() as conn:
        counts = dict(conn.execute(text("SELECT id, access_count FROM memories")).all())
    assert counts.pop("allowed") == 1
    assert set(counts.values()) == {0}


@pytest.mark.parametrize("sources", [None, {}, [], "good", ["missing"], [3], [None], ["good"] * 65])
def test_invalid_source_sets_fail_closed_without_sql_errors(rig, sources):
    add_message(rig)
    add_memory(rig, "invalid", sources)
    add_memory(rig, "control", ["good"])
    results = Backend(rig.engine, False).search_memories_by_embedding([1, 0, 0], eligibility=eligibility(rig))
    assert [row["id"] for row in results] == ["control"]


@pytest.mark.parametrize("overrides", [
    {"session": "priv"}, {"session": "other"}, {"session": "missing"},
    {"role": "summary"}, {"role": "system"},
    {"author_type": None}, {"author": None}, {"author_type": "unknown"},
    {"author": "peer"},
    {"role": "user", "author_type": None, "author": None},
    {"role": "user", "author": "private_fixture"},
    {"role": "user", "author": "unknown_sender"},
    {"role": "user", "author_type": "user", "author": "user_b"},
])
def test_one_ineligible_source_omits_whole_memory(rig, overrides):
    add_message(rig)
    add_message(rig, "bad", **overrides)
    add_memory(rig, "mixed", ["good", "bad"])
    add_memory(rig, "control", ["good"])
    result = Backend(rig.engine, True).search_memories_by_embedding([1, 0, 0], eligibility=eligibility(rig))
    assert [r["id"] for r in result] == ["control"]


def test_public_and_self_have_distinct_authorities(rig):
    add_message(rig, "pub_blank", author_type=None, author=None)
    add_message(rig, "pub_human", role="user", author_type="user", author="user_a")
    add_message(rig, "pub_delivery", role="user", author="peer")
    add_memory(rig, "public", ["pub_blank", "pub_human", "pub_delivery"])
    add_message(rig, "self_blank", partition="private_fixture", session="priv", author_type=None, author=None)
    add_message(rig, "self_delivery", partition="private_fixture", session="priv", role="user", author="peer")
    add_message(rig, "self_owned", partition="private_fixture", session="priv", author="private_fixture")
    for source in ["self_blank", "self_delivery", "self_owned"]:
        add_memory(rig, source, [source], partition="private_fixture")
    backend = Backend(rig.engine, False)
    assert [r["id"] for r in backend.search_memories_by_embedding([1, 0, 0], eligibility=eligibility(rig))] == ["public"]
    own = eligibility(rig, "private_fixture")
    assert [r["id"] for r in backend.search_memories_by_embedding([1, 0, 0], eligibility=own)] == ["self_owned"]
    with rig.engine.connect() as conn:
        history = conn.execute(text("SELECT m.id FROM messages m JOIN sessions s ON s.id=m.session_id WHERE "
                                    + own.authority.predicate(history=True)), own.params).scalars().all()
    assert set(history) == {"self_owned", "self_delivery"}
    with rig.engine.begin() as conn:
        conn.execute(text("UPDATE bot_profiles SET include_in_global_search=false WHERE slug='peer'"))
    assert backend.search_memories_by_embedding([1, 0, 0], eligibility=eligibility(rig)) == []


@pytest.mark.parametrize("decay", [True, False])
def test_guarded_backend_failure_is_not_successful_empty_result(rig, decay):
    guard = eligibility(rig)
    with rig.engine.begin() as conn:
        conn.execute(text("ALTER TABLE messages DROP COLUMN author_entity_type"))
    with pytest.raises(BotAccessError, match="^source_unavailable$"):
        Backend(rig.engine, decay).search_memories_by_embedding([1, 0, 0], eligibility=guard)


def test_session_get_close_owner_is_bound_in_sql(rig):
    from llm_bawt.memory.postgresql_short_term import PostgreSQLShortTermManager

    with rig.engine.begin() as conn:
        conn.execute(text("""ALTER TABLE sessions ADD COLUMN started_at timestamptz DEFAULT now(),
            ADD COLUMN ended_at timestamptz, ADD COLUMN archived_at timestamptz,
            ADD COLUMN session_metadata jsonb"""))
    manager = object.__new__(PostgreSQLShortTermManager)
    manager._backend = SimpleNamespace(engine=rig.engine)
    assert manager.get_session("priv", bot_id="public-fixture") is None
    assert manager.close_session("priv", bot_id="public-fixture") is False
    assert manager.get_session("priv", bot_id="private_fixture", user_id="user_b") is None
    assert manager.close_session("priv", bot_id="private_fixture", user_id="user_b") is False
    own = manager.get_session("priv", bot_id="private_fixture", user_id="user_a")
    assert own["id"] == "priv" and own["ended_at"] is None
    assert manager.close_session("priv", bot_id="private_fixture", user_id="user_a") is True
    assert manager.get_session("priv", bot_id="private_fixture", user_id="user_a")["ended_at"] is not None


@pytest.mark.parametrize("mode", ["text", "embedding-decay", "embedding-plain"])
def test_tag_values_cannot_override_sql_authority(rig, mode):
    add_message(rig)
    add_memory(rig, "control", ["good"])
    add_memory(rig, "denied", ["missing"])
    backend = Backend(rig.engine, mode == "embedding-decay")
    injected_tag = "x'] OR true --"
    if mode == "text":
        result = backend.search_memories_by_text("unmatched", tags=[injected_tag])
    else:
        result = backend.search_memories_by_embedding([1, 0, 0], tags=[injected_tag], eligibility=eligibility(rig))
    assert result == []
    with rig.engine.connect() as conn:
        assert conn.execute(text("SELECT sum(access_count) FROM memories")).scalar() == 0


@pytest.mark.asyncio
async def test_userless_service_session_storage_filters_without_losing_bot_owner(rig):
    from llm_bawt.bot_access.context import bind_bot_access
    from llm_bawt.mcp_server.storage_sessions import StorageSessionsMixin
    from llm_bawt.memory.postgresql_short_term import PostgreSQLShortTermManager

    with rig.engine.begin() as conn:
        conn.execute(text("""ALTER TABLE sessions ADD COLUMN started_at timestamptz DEFAULT now(),
            ADD COLUMN ended_at timestamptz, ADD COLUMN archived_at timestamptz,
            ADD COLUMN session_metadata jsonb"""))
        conn.execute(text("INSERT INTO sessions (id, bot_id, user_id, status) VALUES "
                          "('priv_b', 'private_fixture', 'user_b', 'active')"))
        conn.execute(text("ALTER TABLE messages ADD COLUMN timestamp float, ADD COLUMN content text"))
    manager = object.__new__(PostgreSQLShortTermManager)
    manager._backend = SimpleNamespace(engine=rig.engine, _messages_table_name="messages")

    class Storage(StorageSessionsMixin):
        def get_short_term_manager(self, bot, **kwargs):
            return manager

    storage = Storage()
    access = BotPolicyResolver(rig.engine).resolve_sync(TrustedCaller("private_fixture", None, "service"))
    with bind_bot_access(access):
        assert (await storage.get_session("priv", "private_fixture"))["user_id"] == "user_a"
        assert await storage.get_session("pub", "private_fixture") is None
        assert await storage.close_session("pub", "private_fixture") is False
        assert {r["id"] for r in await storage.list_sessions("private_fixture")} == {"priv", "priv_b"}
        assert [r["id"] for r in await storage.list_sessions("private_fixture", user_id="user_b")] == ["priv_b"]
        assert await storage.close_session("priv", "private_fixture") is True
        for operation in (storage.get_session, storage.close_session):
            with pytest.raises(BotAccessError, match="access_denied"):
                await operation("pub", "public-fixture")
        with pytest.raises(BotAccessError, match="access_denied"):
            await storage.list_sessions("public-fixture")
        with pytest.raises(BotAccessError):
            await storage.list_sessions("")
    bound = BotPolicyResolver(rig.engine).resolve_sync(TrustedCaller("private_fixture", "user_b", "service"))
    with bind_bot_access(bound):
        assert await storage.get_session("priv", "private_fixture") is None
        assert await storage.close_session("priv", "private_fixture") is False
        assert [r["id"] for r in await storage.list_sessions("private_fixture")] == ["priv_b"]
        with pytest.raises(BotAccessError, match="access_denied"):
            await storage.list_sessions("private_fixture", user_id="user_a")


def test_delivery_participant_predicate_precedes_limit_and_receipt_materialization(rig):
    from llm_bawt.inter_bot_delivery import InterBotDeliveryStore

    # Minimal local table: no production receipts or store provisioning.
    with rig.engine.begin() as conn:
        conn.execute(text("""CREATE TABLE inter_bot_deliveries (
            id text, sender_bot_id text, target_bot_id text, ordinal int,
            message text, user_message_id text, turn_id text, status text,
            attempt_count int DEFAULT 0, max_attempts int DEFAULT 5,
            created_at timestamptz DEFAULT now(), updated_at timestamptz DEFAULT now(),
            available_at timestamptz DEFAULT now(), metadata_json jsonb DEFAULT '{}')"""))
        for ordinal, sender, target in [
            (1, 'public-fixture', 'peer'), (2, 'private_fixture', 'peer'),
            (3, 'peer', 'private_fixture'), (4, 'unknown', 'peer'),
        ]:
            conn.execute(text("""INSERT INTO inter_bot_deliveries
                (id, sender_bot_id, target_bot_id, ordinal, message, user_message_id, turn_id, status)
                VALUES (:id, :sender, :target, :ordinal, 'fixture', 'u', 't', 'QUEUED')"""),
                {"id": str(ordinal), "sender": sender, "target": target, "ordinal": ordinal})
    store = object.__new__(InterBotDeliveryStore)
    store.engine = rig.engine
    public = frozenset({'public-fixture', 'peer'})
    assert [r.id for r in store.list(allowed_bot_ids=public, limit=1)] == ['1']
    for id in ['2', '3', '4']:
        assert store.get(id, allowed_bot_ids=public) is None
    own = public | {'private_fixture'}
    assert [r.id for r in store.list(allowed_bot_ids=own, limit=1)] == ['3']
    assert store.list(allowed_bot_ids=frozenset()) == []
    assert store.get('1', allowed_bot_ids=frozenset()) is None
