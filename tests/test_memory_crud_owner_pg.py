"""Two synthetic users sharing one bot must not share personal memory writes."""
from uuid import uuid4

import pytest
from sqlalchemy import text

from llm_bawt.mcp_server.client import MemoryClient
from tests.test_bot_access_mcp import signer  # noqa: F401
from tests.private_postgres import private_app_pg  # noqa: F401
from tests.test_memory_update_pg import memory_rig  # noqa: F401

pytestmark = pytest.mark.integration


@pytest.fixture
def owned_clients(memory_rig):  # noqa: F811
    storage, backend, bot = memory_rig
    clients, sources = {}, {}
    from llm_bawt.profiles import EntityType, ProfileManager
    profiles = ProfileManager(storage.config)
    for owner in ("owner-a", "owner-b"):
        profiles.get_or_create_profile(EntityType.USER, owner)
    with backend.engine.begin() as conn:
        from llm_bawt.memory.ownership_migration import mark_new_bot
        mark_new_bot(conn, bot)
        for owner in ("owner-a", "owner-b"):
            session, source = str(uuid4()), str(uuid4())
            conn.execute(text("""
                INSERT INTO sessions (id, bot_id, user_id, started_at, status)
                VALUES (:id, :bot, :owner, now(), 'active')
            """), {"id": session, "bot": bot, "owner": owner})
            conn.execute(text(f"""
                INSERT INTO {backend._messages_table_name}
                    (id, role, content, timestamp, session_id, processed,
                     author_entity_type, author_entity_id)
                VALUES (:id, 'user', 'Synthetic personal fact', 1, :session,
                        false, 'user', :owner)
            """), {"id": source, "session": session, "owner": owner})
            # Avoid starting a refresh worker; exercise the real embedded facade
            # against this test's private storage, never the singleton/live DB.
            client = MemoryClient(storage.config, bot_id=bot, user_id=owner,
                                  server_url="http://unused.invalid")
            client.server_url = None
            client._storage = storage
            client._initialized = True
            clients[owner] = client
            sources[owner] = source
    return backend, clients["owner-a"], clients["owner-b"], sources


@pytest.mark.parametrize("vector", [True, False], ids=["semantic", "exact"])
def test_general_add_deduplicates_only_within_owner(owned_clients, monkeypatch, vector):
    _backend, a, b, sources = owned_clients
    if not vector:
        monkeypatch.setattr("llm_bawt.memory.embeddings.generate_embedding", lambda *a, **k: None)
        monkeypatch.setattr("llm_bawt.mcp_server.storage.generate_embedding", lambda *a, **k: None)
    first = a.store_memory("Same personal fact", source_message_ids=[sources["owner-a"]])
    second = b.store_memory("Same personal fact", source_message_ids=[sources["owner-b"]])
    assert second.id != first.id
    assert a.store_memory("Same personal fact", source_message_ids=[sources["owner-a"]]).id == first.id
    assert b.store_memory("Same personal fact", source_message_ids=[sources["owner-b"]]).id == second.id


def test_recent_list_is_owned_before_limit(owned_clients):
    _backend, a, b, sources = owned_clients
    first = a.store_memory("A personal fact", source_message_ids=[sources["owner-a"]])
    b.store_memory("B personal fact", source_message_ids=[sources["owner-b"]])
    assert [row["id"] for row in a.list_recent(n=1)] == [first.id]


def test_foreign_update_is_refused(owned_clients):
    backend, a, b, sources = owned_clients
    target = b.store_memory("B personal fact", source_message_ids=[sources["owner-b"]])
    assert a.update_memory(target.id, content="Wrong owner's edit") is None
    with backend.engine.connect() as conn:
        assert conn.execute(text(f"SELECT content FROM {backend._memories_table_name} WHERE id=:id"),
                            {"id": target.id}).scalar_one() == "B personal fact"


def test_foreign_delete_is_refused(owned_clients):
    _backend, a, b, sources = owned_clients
    target = b.store_memory("B personal fact", source_message_ids=[sources["owner-b"]])
    assert a.delete_memory(target.id) is False
    assert [row["id"] for row in b.list_recent()] == [target.id]


def test_http_foreign_patch_and_delete_refused_with_real_store(owned_clients, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from types import SimpleNamespace
    from llm_bawt.service.routes import memory as routes

    backend, a, b, _sources = owned_clients
    target = b.store_memory("B's HTTP-owned fact", importance=0.57)
    clients = {"owner-a": a, "owner-b": b}
    monkeypatch.setattr(routes, "get_service", lambda: SimpleNamespace(
        get_memory_client=lambda _bot, user_id=None: clients[user_id]))
    app = FastAPI()
    app.include_router(routes.router)
    app.dependency_overrides[routes.require_mutation_service_token] = lambda: None
    with TestClient(app) as http:
        params = {"bot_id": backend.bot_id, "user_id": "owner-a"}
        assert http.patch(f"/v1/memory/{target.id}", params=params,
                          json={"importance": 0.58}).status_code == 404
        assert http.delete(f"/v1/memory/{target.id}", params=params).status_code == 404
        with backend.engine.connect() as conn:
            assert conn.execute(text(f"SELECT importance FROM {backend._memories_table_name} WHERE id=:id"),
                                {"id": target.id}).scalar_one() == pytest.approx(0.57)
        params["user_id"] = "owner-b"
        response = http.patch(f"/v1/memory/{target.id}", params=params,
                              json={"importance": 0.59})
        assert response.status_code == 200, response.text
        with backend.engine.connect() as conn:
            assert conn.execute(text(f"SELECT importance FROM {backend._memories_table_name} WHERE id=:id"),
                                {"id": target.id}).scalar_one() == pytest.approx(0.59)
        assert http.delete(f"/v1/memory/{target.id}", params=params).status_code == 200


@pytest.mark.asyncio
@pytest.mark.usefixtures("signer")
@pytest.mark.parametrize("method,extra", [
    ("memory_update", {"content": "Foreign change"}),
    ("memory_delete", {}),
    ("memory_update_meaning", {"intent": "Foreign intent"}),
    ("memory_supersede", {"new_memory_id": "synthetic-new-id"}),
])
async def test_registered_mcp_foreign_ids_never_mutate_real_store(owned_clients, monkeypatch, method, extra):
    from llm_bawt.mcp_server import server as tools
    from llm_bawt.mcp_server.catalog import CatalogFastMCP
    from tests.test_bot_access_mcp import Resolver, identity, turn

    backend, a, b, _sources = owned_clients
    target = b.store_memory("B's registered MCP fact", importance=0.57)
    monkeypatch.setattr(tools, "_get_storage", lambda: a._storage)
    resolver = Resolver()
    resolver.rows.append({"slug": backend.bot_id, "include_in_global_search": True})
    catalog = CatalogFastMCP("durable-owner-probe", bot_policy_resolver=resolver)
    handlers = {
        "memory_update": tools.update_memory, "memory_delete": tools.delete_memory,
        "memory_update_meaning": tools.update_memory_meaning,
        "memory_supersede": tools.supersede_memory,
    }
    catalog.add_tool(handlers[method], name=method)
    id_key = "old_memory_id" if method == "memory_supersede" else "memory_id"
    with identity(turn(bot=backend.bot_id, user="owner-a")):
        await catalog._tool_manager.call_tool(method, {
            "bot_id": backend.bot_id, id_key: target.id, **extra})
    with backend.engine.connect() as conn:
        row = conn.execute(text(f"""SELECT content, importance, intent, owner_user_id
            FROM {backend._memories_table_name} WHERE id=:id"""), {"id": target.id}).mappings().one()
    assert row["content"] == "B's registered MCP fact"
    assert row["importance"] == pytest.approx(0.57)
    assert row["intent"] is None
    assert row["owner_user_id"] == "owner-b"


def test_executor_foreign_ids_refused_by_its_personal_client(owned_clients):
    from llm_bawt.tools.executor import ToolExecutor
    from llm_bawt.tools.parser import ToolCall

    backend, a, b, _sources = owned_clients
    target = b.store_memory("B's executor fact")
    executor = ToolExecutor.__new__(ToolExecutor)
    executor.memory_client = a
    executor.user_id = "owner-a"
    for action, extra in (("update", {"content": "Foreign change"}), ("delete", {})):
        result = getattr(executor, "_memory_" + action)(ToolCall(
            name="memory", arguments={"memory_id": target.id, **extra}, raw_text=""))
        assert 'status="error"' in result
    with backend.engine.connect() as conn:
        assert conn.execute(text(f"SELECT content FROM {backend._memories_table_name} WHERE id=:id"),
                            {"id": target.id}).scalar_one() == "B's executor fact"


def test_duplicate_response_matches_durable_row(owned_clients):
    backend, a, _b, sources = owned_clients
    first = a.store_memory("Stable fact", tags=["original"], importance=0.3,
                           source_message_ids=[sources["owner-a"]])
    duplicate = a.store_memory("Stable fact", tags=["incoming"], importance=0.9,
                               source_message_ids=[sources["owner-a"]])
    with backend.engine.connect() as conn:
        durable = conn.execute(text(f"""
            SELECT id, content, tags, importance, source_message_ids
            FROM {backend._memories_table_name} WHERE id=:id
        """), {"id": first.id}).mappings().one()
    assert duplicate.id == durable["id"]
    assert duplicate.content == durable["content"]
    assert duplicate.tags == durable["tags"]
    assert duplicate.importance == durable["importance"]
    assert duplicate.source_message_ids == durable["source_message_ids"]


def test_manual_facts_belong_to_the_authenticated_owner(owned_clients):
    _backend, a, b, _sources = owned_clients
    first = a.store_memory("Remember my preference")
    second = b.store_memory("Remember my preference")
    assert first.id != second.id
    assert [row["id"] for row in a.list_recent()] == [first.id]
    assert [row["id"] for row in b.list_recent()] == [second.id]


def test_manual_owner_is_immutable_and_can_delete_own_prefix(owned_clients):
    backend, a, b, _sources = owned_clients
    first = a.store_memory("Personal manual row")
    assert a.update_memory(first.id, content="Updated personal row") is not None
    assert b.delete_memory(first.id[:8]) is False
    with backend.engine.connect() as conn:
        assert conn.execute(text(f"SELECT owner_user_id FROM {backend._memories_table_name} WHERE id=:id"),
                            {"id": first.id}).scalar_one() == "owner-a"
    assert a.delete_memory(first.id[:8]) is True


def test_forged_sources_are_refused_without_mutating_original(owned_clients):
    backend, a, _b, sources = owned_clients
    with pytest.raises(RuntimeError, match="sources do not belong"):
        a.store_memory("Forged fact", source_message_ids=[sources["owner-b"]])
    own = a.store_memory("My valid fact", source_message_ids=[sources["owner-a"]])
    with pytest.raises(ValueError, match="sources do not belong"):
        a.update_memory(own.id, source_message_ids=[sources["owner-b"]])
    with backend.engine.connect() as conn:
        rows = conn.execute(text(f"SELECT source_message_ids FROM {backend._memories_table_name}")).all()
    assert rows == [([sources["owner-a"]],)]


def test_absent_authority_is_not_operator_and_operator_is_explicit(owned_clients):
    from llm_bawt.memory.ownership import OPERATOR_MEMORY_AUTHORITY
    backend, a, _b, _sources = owned_clients
    own = a.store_memory("Personal fact")
    assert backend.list_recent() == []
    assert backend.get_memory_by_id(own.id) is None
    assert backend.update_memory(own.id, content="Anonymous overwrite") is None
    from llm_bawt.memory.ownership import NO_MEMORY_AUTHORITY
    assert backend.delete_memory(own.id, authority=NO_MEMORY_AUTHORITY) is False
    assert not backend.add_memory(str(uuid4()), "Anonymous insertion").persisted
    assert backend.delete_memory(own.id, authority=OPERATOR_MEMORY_AUTHORITY) is True


def test_legacy_null_reads_remain_but_personal_mutation_and_dedup_do_not(owned_clients):
    from llm_bawt.memory.ownership import OPERATOR_MEMORY_AUTHORITY
    backend, a, b, _sources = owned_clients
    legacy = str(uuid4())
    assert backend.add_memory(legacy, "Ambiguous historical fact",
                              authority=OPERATOR_MEMORY_AUTHORITY).persisted
    assert [row["id"] for row in a.list_recent()] == [legacy]
    assert [row["id"] for row in b.list_recent()] == [legacy]
    assert a.update_memory(legacy, content="Claimed history") is None
    assert a.delete_memory(legacy) is False
    assert a.store_memory("Ambiguous historical fact").id != legacy


def test_unmarked_bot_keeps_legacy_behavior(owned_clients):
    from llm_bawt.memory.ownership_migration import ACTIVATIONS
    backend, a, b, _sources = owned_clients
    with backend.engine.begin() as conn:
        conn.execute(text(f"DELETE FROM {ACTIVATIONS} WHERE bot_id=:bot"),
                     {"bot": backend.bot_id_sanitized})
    first = a.store_memory("Legacy shared fact")
    assert b.store_memory("Legacy shared fact").id == first.id
    assert [row["id"] for row in b.list_recent()] == [first.id]
    assert b.update_memory(first.id, content="Legacy update") is not None


def test_legacy_session_provenance_never_proves_personal_ownership(owned_clients):
    from llm_bawt.memory.ownership import OPERATOR_MEMORY_AUTHORITY
    backend, a, _b, sources = owned_clients
    with backend.engine.begin() as conn:
        conn.execute(text("""UPDATE sessions SET session_metadata=CAST(:meta AS jsonb)
            WHERE bot_id=:bot AND user_id='owner-a'"""),
                     {"bot": backend.bot_id, "meta": '{"legacy_backfill":true}'})
    legacy = str(uuid4())
    assert backend.add_memory(legacy, "Guessed historical owner", source_message_ids=[sources["owner-a"]],
                              authority=OPERATOR_MEMORY_AUTHORITY).persisted
    assert a.delete_memory(legacy) is False
    assert a.update_memory(legacy, content="Claimed guessed history") is None
    with pytest.raises(RuntimeError, match="sources do not belong"):
        a.store_memory("New fact using guessed provenance", source_message_ids=[sources["owner-a"]])


@pytest.mark.parametrize("mode", ["text", "vector", "client", "high", "convenience"])
def test_reads_scope_before_limit_and_keep_legacy_visibility(owned_clients, monkeypatch, mode):
    from llm_bawt.memory import embeddings
    from llm_bawt.memory.ownership import OPERATOR_MEMORY_AUTHORITY
    backend, a, b, _sources = owned_clients
    monkeypatch.setattr(backend.config, "MEMORY_DECAY_ENABLED", False)
    monkeypatch.setattr(backend.config, "MEMORY_DIVERSITY_ENABLED", False)
    foreign = b.store_memory("scopedtoken", importance=0.9)
    own = a.store_memory("scopedtoken", importance=0.8)
    legacy = str(uuid4())
    # A weaker vector avoids an arbitrary SQL tie between the two eligible rows.
    legacy_vector = [0.9, 0.1] + [0.0] * (embeddings.EMBED_DIM - 2)
    assert backend.add_memory(legacy, "scopedtoken", importance=0.3, embedding=legacy_vector,
                              authority=OPERATOR_MEMORY_AUTHORITY).persisted

    def read(limit):
        if mode == "text":
            return backend.search_memories_by_text("scopedtoken", limit, authority=a._memory_authority)
        if mode == "vector":
            return backend.search_memories_by_embedding(
                embeddings.generate_embedding("scopedtoken"), limit, authority=a._memory_authority)
        if mode == "client":
            return [vars(row) for row in a.search("scopedtoken", n_results=limit)]
        if mode == "high":
            return a.list_memories(limit=limit)
        return backend.search("scopedtoken", limit, authority=a._memory_authority)

    assert [row["id"] for row in read(1)] == [own.id]
    assert {row["id"] for row in read(10)} == {own.id, legacy}
    with backend.engine.connect() as conn:
        assert conn.execute(text(f"SELECT access_count FROM {backend._memories_table_name} WHERE id=:id"),
                            {"id": foreign.id}).scalar_one() == 0


def test_stats_are_owner_scoped_and_do_not_expose_partition_size(owned_clients):
    from llm_bawt.memory.ownership import OPERATOR_MEMORY_AUTHORITY
    backend, a, b, _sources = owned_clients
    a.store_memory("Own fact")
    b.store_memory("Foreign fact")
    backend.add_memory(str(uuid4()), "Legacy fact", authority=OPERATOR_MEMORY_AUTHORITY)
    stats = a.stats()
    assert stats["memories"]["total_count"] == 2
    assert stats["messages"]["total_count"] == 1
    assert stats["memories"]["storage_size_bytes"] is None
    assert backend.stats()["memories"]["total_count"] == 0
    operator = backend.stats(authority=OPERATOR_MEMORY_AUTHORITY)
    assert operator["memories"]["total_count"] == 3
    assert operator["messages"]["total_count"] == 2
    assert operator["memories"]["storage_size_bytes"] > 0


def test_clear_keeps_foreign_and_ambiguous_legacy_rows(owned_clients):
    from llm_bawt.memory.ownership import OPERATOR_MEMORY_AUTHORITY
    backend, a, b, _sources = owned_clients
    a.store_memory("Own fact")
    foreign = b.store_memory("Foreign fact")
    legacy = str(uuid4())
    backend.add_memory(legacy, "Legacy fact", authority=OPERATOR_MEMORY_AUTHORITY)
    assert not backend.clear()
    assert a.clear_memories()
    assert {row["id"] for row in b.list_recent()} == {foreign.id, legacy}
    assert backend.clear(authority=OPERATOR_MEMORY_AUTHORITY)
    assert b.list_recent() == []


@pytest.mark.parametrize("vector", [False, True], ids=["exact", "semantic"])
def test_operator_dedup_never_crosses_owner_groups(owned_clients, monkeypatch, vector):
    from llm_bawt.memory.errors import MemoryUpdateDuplicateError
    from llm_bawt.memory.ownership import OPERATOR_MEMORY_AUTHORITY as operator
    backend, a, b, _sources = owned_clients
    if not vector:
        monkeypatch.setattr("llm_bawt.memory.embeddings.generate_embedding", lambda *a, **k: None)
        monkeypatch.setattr("llm_bawt.mcp_server.storage.generate_embedding", lambda *a, **k: None)
    foreign = b.store_memory("Same fact")
    own = a.store_memory("Different fact")
    legacy = str(uuid4())
    outcome = backend.add_memory(legacy, "Same fact", authority=operator)
    assert outcome.persisted and outcome.memory_id == legacy
    assert backend.add_memory(str(uuid4()), "Same fact", authority=operator).memory_id == legacy
    assert backend.update_memory(own.id, content="Same fact", authority=operator) is not None
    another = a.store_memory("Another fact")
    with pytest.raises(MemoryUpdateDuplicateError):
        backend.update_memory(another.id, content="Same fact", authority=operator)
    with backend.engine.connect() as conn:
        owners = dict(conn.execute(text(f"SELECT id, owner_user_id FROM {backend._memories_table_name}")).all())
    assert owners == {foreign.id: "owner-b", own.id: "owner-a", legacy: None, another.id: "owner-a"}


def test_existing_id_cannot_bypass_owner_check_by_deduplicating_elsewhere(owned_clients):
    backend, a, b, _sources = owned_clients
    foreign = b.store_memory("Foreign fact")
    a.store_memory("Own fact")
    outcome = backend.add_memory(foreign.id, "Own fact", authority=a._memory_authority)
    assert not outcome.persisted and outcome.status == "failed"


def test_convenience_search_message_fallback_is_owner_scoped(owned_clients):
    backend, a, _b, sources = owned_clients
    rows = backend.search("Synthetic personal fact", authority=a._memory_authority)
    assert [row["id"] for row in rows] == [sources["owner-a"]]
    assert backend.search("Synthetic personal fact") is None


def test_context_composition_does_not_reintroduce_foreign_messages(owned_clients):
    backend, a, _b, sources = owned_clients
    with backend.engine.begin() as conn:
        conn.execute(text(f"UPDATE {backend._messages_table_name} SET timestamp=EXTRACT(EPOCH FROM now())"))
    context = a.get_recent_context(query="personal fact")
    assert [row["id"] for row in context["messages"]] == [sources["owner-a"]]


@pytest.mark.parametrize("forgotten", [False, True])
@pytest.mark.parametrize("legacy", [False, True])
def test_source_delete_validates_caller_but_deletes_mixed_source_rows(owned_clients, forgotten, legacy):
    from llm_bawt.memory.ownership import OPERATOR_MEMORY_AUTHORITY
    backend, a, b, sources = owned_clients
    foreign = b.store_memory("Foreign fact", source_message_ids=[sources["owner-b"]])
    mixed = str(uuid4())
    backend.add_memory(mixed, "Mixed legacy fact", source_message_ids=list(sources.values()),
                       authority=OPERATOR_MEMORY_AUTHORITY)
    if legacy:
        with backend.engine.begin() as conn:
            conn.execute(text("""UPDATE sessions SET session_metadata=jsonb_build_object('legacy_backfill', true)
                WHERE bot_id=:bot AND user_id='owner-a'"""), {"bot": backend.bot_id})
    if forgotten:
        assert backend.ignore_message_by_id(sources["owner-a"], user_id="owner-a")
    assert not a.delete_memory(mixed)  # By-ID rules are not weakened for forget.
    # No partial deletion when even one supplied source is foreign or unknown.
    assert a.delete_memories_by_source_message_ids(list(sources.values())) == 0
    assert a.delete_memories_by_source_message_ids([sources["owner-b"]]) == 0
    assert a.delete_memories_by_source_message_ids([str(uuid4())]) == 0
    assert backend.delete_memories_by_source_message_ids([sources["owner-a"]]) == 0
    assert a.delete_memories_by_source_message_ids([sources["owner-a"]]) == 1
    assert [row["id"] for row in b.list_recent()] == [foreign.id]


@pytest.mark.parametrize("kind", ["service", "missing_profile"])
def test_nonhuman_sessions_never_prove_personal_mutation_or_supplied_sources(owned_clients, kind):
    from llm_bawt.memory.ownership import OPERATOR_MEMORY_AUTHORITY
    backend, a, b, sources = owned_clients
    with backend.engine.begin() as conn:
        if kind == "service":
            conn.execute(text("UPDATE entity_profiles SET is_service=true WHERE entity_id='owner-a'"))
        else:
            conn.execute(text("DELETE FROM entity_profiles WHERE entity_id='owner-a'"))
    target = str(uuid4())
    assert backend.add_memory(target, "Unproven personal history", source_message_ids=[sources["owner-a"]],
                              authority=OPERATOR_MEMORY_AUTHORITY).persisted
    assert {row["id"] for row in a.list_recent()} == {target}
    assert {row["id"] for row in b.list_recent()} == {target}
    assert a.update_memory(target, content="Claimed history") is None
    assert not a.delete_memory(target)
    with pytest.raises(RuntimeError, match="sources do not belong"):
        a.store_memory("Forged provenance", source_message_ids=[sources["owner-a"]])
    # A manual write gets a new explicit owner, never dedups into ambiguous history.
    assert a.store_memory("Unproven personal history").id != target


def test_embedded_fact_extraction_keeps_sourced_only_dedup_and_durable_result(owned_clients, monkeypatch):
    from types import SimpleNamespace
    from llm_bawt.memory.extraction.storage import ExtractionMemoryStore
    backend, a, b, sources = owned_clients
    policy = SimpleNamespace(partitions={backend.bot_id: backend.bot_id_sanitized},
                             readable_bots=frozenset([backend.bot_id]), assert_current=lambda: None)
    monkeypatch.setattr("llm_bawt.memory.extraction.storage.BotPolicyResolver",
                        lambda engine: SimpleNamespace(resolve_sync=lambda: policy))
    manual = a.store_memory("Same extracted fact")
    fact = {"content": "Same extracted fact", "tags": ["original"], "importance": 0.4,
            "source_message_ids": [sources["owner-a"]]}

    async def extract(**kwargs):
        return [fact]

    monkeypatch.setattr("llm_bawt.mcp_server.extraction.extract_facts_from_messages", extract)
    messages = [{"role": "user", "content": "Synthetic", "id": sources["owner-a"]}]
    first = a.extract_facts(messages, use_llm=False)[0]
    assert first.id != manual.id
    fact["tags"] = ["incoming"]
    duplicate = a.extract_facts(messages, use_llm=False)[0]
    assert duplicate.id == first.id and duplicate.tags == ["original"]
    writer = ExtractionMemoryStore(backend, backend.bot_id, "owner-a")
    assert [row["id"] for row in writer.list_memories()] == [first.id]
    assert not writer.delete_memory(manual.id)
    # Even valid source attribution must not override an explicit foreign owner.
    with backend.engine.begin() as conn:
        conn.execute(text(f"UPDATE {backend._memories_table_name} SET owner_user_id='owner-b' WHERE id=:id"),
                     {"id": first.id})
    assert writer.list_memories() == []
    assert not writer.delete_memory(first.id)
    assert [row["id"] for row in b.list_recent()] == [first.id]


def test_owner_change_while_embedding_does_not_overwrite_the_new_owner(owned_clients, monkeypatch):
    backend, a, _b, _sources = owned_clients
    own = a.store_memory("Original own fact")

    def transfer(*args, **kwargs):
        with backend.engine.begin() as conn:
            conn.execute(text(f"UPDATE {backend._memories_table_name} SET owner_user_id='owner-b' WHERE id=:id"),
                         {"id": own.id})
        return None

    monkeypatch.setattr("llm_bawt.memory.embeddings.generate_embedding", transfer)
    assert a.update_memory(own.id, content="Stale edit") is None
    with backend.engine.connect() as conn:
        assert conn.execute(text(f"SELECT content FROM {backend._memories_table_name} WHERE id=:id"),
                            {"id": own.id}).scalar_one() == "Original own fact"


def test_activation_during_embedding_is_seen_by_existing_backend(owned_clients, monkeypatch):
    from llm_bawt.memory.ownership_migration import ACTIVATIONS
    backend, _a, _b, _sources = owned_clients
    with backend.engine.begin() as conn:
        conn.execute(text(f"DELETE FROM {ACTIVATIONS} WHERE bot_id=:bot"), {"bot": backend.bot_id_sanitized})

    def activate(*args, **kwargs):
        with backend.engine.begin() as conn:
            conn.execute(text(f"INSERT INTO {ACTIVATIONS} (bot_id, mode, actor_id) "
                              "VALUES (:bot, 'new', 'synthetic') ON CONFLICT DO NOTHING"),
                         {"bot": backend.bot_id_sanitized})
        return None

    monkeypatch.setattr("llm_bawt.memory.embeddings.generate_embedding", activate)
    outcome = backend.add_memory(str(uuid4()), "Callerless write begun before activation")
    assert outcome.status == "failed"
    with backend.engine.connect() as conn:
        assert conn.execute(text(f"SELECT count(*) FROM {backend._memories_table_name}")).scalar_one() == 0


def test_validated_sources_cannot_change_owner_until_write_commits(owned_clients):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    from sqlalchemy import event
    from tests.pg_lock_wait import wait_blocked_by
    backend, a, _b, sources = owned_clients
    at_insert, release = Event(), Event()
    holder = {}

    def pause(conn, _cursor, statement, _params, _context, _many):
        if " ".join(statement.split()).startswith(f"INSERT INTO {backend._memories_table_name} "):
            holder["pid"] = conn.execute(text("SELECT pg_backend_pid()")).scalar_one()
            at_insert.set()
            assert release.wait(15), "memory writer was not released"

    def reassign():
        with backend.engine.begin() as conn:
            conn.execute(text(f"""UPDATE sessions SET user_id='owner-b', status='archived'
                WHERE id=(SELECT session_id FROM {backend._messages_table_name} WHERE id=:source)"""),
                         {"source": sources["owner-a"]})

    event.listen(backend.engine, "before_cursor_execute", pause)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(a.store_memory, "Validated source fact", source_message_ids=[sources["owner-a"]])
            try:
                assert at_insert.wait(15), "writer did not validate sources and reach insert"
                second = pool.submit(reassign)
                wait_blocked_by(backend.engine, holder["pid"], second)
            finally:
                release.set()
            stored = first.result(timeout=15)
            second.result(timeout=15)
        with backend.engine.connect() as conn:
            assert conn.execute(text(f"SELECT owner_user_id FROM {backend._memories_table_name} WHERE id=:id"),
                                {"id": stored.id}).scalar_one() == "owner-a"
    finally:
        release.set()
        event.remove(backend.engine, "before_cursor_execute", pause)
