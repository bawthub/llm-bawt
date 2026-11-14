"""General memory CRUD uses registered caller identity, not cached backend state."""
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from llm_bawt.bot_access import BotAccessError
from llm_bawt.memory.ownership import MemoryAuthority
from llm_bawt.mcp_server import server as tools
from llm_bawt.mcp_server.catalog import CatalogFastMCP
from tests.test_bot_access_mcp import Resolver, identity, signer, turn  # noqa: F401


@pytest.mark.asyncio
@pytest.mark.usefixtures("signer")
@pytest.mark.parametrize("name,handler,method,extra", [
    ("memory_store", tools.store_memory, "store_memory", {"content": "Synthetic fact"}),
    ("memory_update", tools.update_memory, "update_memory", {"memory_id": "synthetic-id"}),
    ("memory_list_recent", tools.list_recent, "list_recent_memories", {}),
    ("memory_search", tools.search_memories, "search_memories", {"query": "Synthetic"}),
    ("memory_list_high_importance", tools.list_memories, "get_high_importance_memories", {}),
    ("system_stats", tools.stats, "stats", {}),
    ("memory_clear", tools.clear_memories, "clear_memories", {}),
    ("memory_regenerate_embeddings", tools.regenerate_embeddings, "regenerate_embeddings", {}),
    ("memory_consolidate", tools.consolidate_memories, "consolidate_memories", {}),
    ("memory_update_meaning", tools.update_memory_meaning, "update_memory_meaning",
     {"memory_id": "synthetic-id"}),
    ("system_run_maintenance", tools.run_maintenance, "run_maintenance", {}),
    ("memory_delete_by_source_messages", tools.delete_memories_by_source_message_ids,
     "delete_memories_by_source_message_ids", {"message_ids": ["source"]}),
])
@pytest.mark.parametrize("user", ["user_a", "user_b"])
async def test_registered_crud_passes_personal_authority(monkeypatch, name, handler, method, extra, user):
    calls = []

    async def record(**kwargs):
        calls.append(kwargs)
        if method in {"store_memory", "update_memory"}:
            return SimpleNamespace(to_dict=lambda: {"id": "synthetic-id"})
        return []

    monkeypatch.setattr(tools, "_get_storage", lambda: SimpleNamespace(**{method: record}))
    catalog = CatalogFastMCP("memory-owner-probe", bot_policy_resolver=Resolver())
    catalog.add_tool(handler, name=name)
    with identity(turn(user=user)):
        await catalog._tool_manager.call_tool(name, {"bot_id": "public_fixture", **extra})
    assert len(calls) == 1
    assert calls[0]["authority"] == MemoryAuthority(user_id=user)
    assert not calls[0]["authority"].operator


@pytest.mark.asyncio
@pytest.mark.parametrize("handler,extra", [
    (tools.store_memory, {"content": "Synthetic fact"}),
    (tools.update_memory, {"memory_id": "synthetic-id"}),
    (tools.list_recent, {}),
    (tools.search_memories, {"query": "Synthetic"}),
    (tools.list_memories, {}),
    (tools.stats, {}),
    (tools.clear_memories, {}),
    (tools.regenerate_embeddings, {}),
    (tools.consolidate_memories, {}),
    (tools.update_memory_meaning, {"memory_id": "synthetic-id"}),
    (tools.run_maintenance, {}),
    (tools.supersede_memory, {"old_memory_id": "old", "new_memory_id": "new"}),
    (tools.get_recent_context, {}),
    (tools.delete_memories_by_source_message_ids, {"message_ids": ["source"]}),
    (tools.extract_facts, {"messages": [{"id": "source", "role": "user", "content": "Synthetic"}],
                           "user_id": "user_a"}),
])
async def test_unbound_crud_fails_before_storage(monkeypatch, handler, extra):
    def forbidden_storage():
        raise AssertionError("unbound caller reached storage")

    monkeypatch.setattr(tools, "_get_storage", forbidden_storage)
    with pytest.raises(BotAccessError):
        await handler(bot_id="public_fixture", **extra)


@pytest.mark.asyncio
@pytest.mark.usefixtures("signer")
async def test_registered_fact_extraction_uses_sourced_writer_and_bound_owner(monkeypatch):
    calls = []

    async def extract(**kwargs):
        return [{"content": "Synthetic fact", "source_message_ids": ["source"]}]

    async def store(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(to_dict=lambda: {"id": "durable-id", "tags": ["durable"]})

    monkeypatch.setattr("llm_bawt.mcp_server.extraction.extract_facts_from_messages", extract)
    monkeypatch.setattr(tools, "_get_storage", lambda: SimpleNamespace(store_extracted_memory=store))
    catalog = CatalogFastMCP("extracted-owner-probe", bot_policy_resolver=Resolver())
    catalog.add_tool(tools.extract_facts, name="facts_extract")
    arguments = {"bot_id": "public_fixture", "user_id": "user_a",
                 "messages": [{"id": "source", "role": "user", "content": "Synthetic fact"}],
                 "use_llm": False}
    with identity(turn(user="user_a")):
        await catalog._tool_manager.call_tool("facts_extract", arguments)
        with pytest.raises(Exception, match="access_denied"):
            await catalog._tool_manager.call_tool("facts_extract", {**arguments, "user_id": "user_b"})
    assert len(calls) == 1
    assert calls[0]["authority"] == MemoryAuthority(user_id="user_a")
    assert calls[0]["source_message_ids"] == ["source"]


@pytest.mark.parametrize("authority", [MemoryAuthority(user_id="user_a"), MemoryAuthority(operator=True)])
def test_embedded_source_search_keeps_bound_authority_and_bot_policy(monkeypatch, authority):
    from contextlib import nullcontext
    from llm_bawt.mcp_server import client_crossbot
    calls = []

    def target(bot):
        if bot != "permitted-target":
            raise BotAccessError("access_denied")
        return bot

    async def resolve(_caller):
        return SimpleNamespace(require_target=target)

    async def search(**kwargs):
        calls.append(kwargs)
        return []

    storage = SimpleNamespace(_require_engine=lambda: object(), search_memories=search)
    client = SimpleNamespace(_get_storage=lambda: storage, _memory_authority=authority,
                             _service_principal=SimpleNamespace(caller=lambda: object()))
    monkeypatch.setattr(client_crossbot, "BotPolicyResolver", lambda _engine: SimpleNamespace(resolve=resolve))
    monkeypatch.setattr(client_crossbot, "bind_bot_access", lambda _access: nullcontext())
    invoke = client_crossbot.MemoryClientCrossBotMixin._embedded_crossbot
    assert invoke(client, "search_memories", target="permitted-target", query="synthetic") == []
    assert calls == [{"bot_id": "permitted-target", "query": "synthetic", "authority": authority}]
    with pytest.raises(BotAccessError):
        invoke(client, "search_memories", target="forbidden-target", query="synthetic")
    assert len(calls) == 1


@pytest.mark.parametrize("authority", [None, MemoryAuthority(user_id="synthetic-owner"),
                                       MemoryAuthority(operator=True)])
def test_maintenance_defers_backend_access_without_changing_authority(authority):
    from llm_bawt.memory.maintenance import MemoryMaintenance
    from llm_bawt.memory.ownership import NO_MEMORY_AUTHORITY

    maint = MemoryMaintenance(backend=None, authority=authority)
    assert maint.authority == (authority or NO_MEMORY_AUTHORITY)
    assert maint.run(run_consolidation=False, run_recurrence_detection=False).errors == []
    with pytest.raises(AttributeError):
        _ = maint.ownership
    maint.backend = SimpleNamespace(bot_id="synthetic-bot")
    assert maint.ownership == maint.authority.scope("synthetic-bot")


def test_authority_is_immutable_and_cannot_mix_operator_and_personal_modes():
    authority = MemoryAuthority(user_id="synthetic-owner")
    with pytest.raises(FrozenInstanceError):
        authority.user_id = "different-owner"
    with pytest.raises(ValueError):
        MemoryAuthority(user_id="synthetic-owner", operator=True)
    with pytest.raises(ValueError):
        MemoryAuthority(user_id="")


@pytest.mark.parametrize("authority", [MemoryAuthority(user_id="user_a"), MemoryAuthority(operator=True)])
def test_embedded_embedding_repair_forwards_exact_authority(monkeypatch, authority):
    from llm_bawt.mcp_server.client_admin import MemoryClientAdminMixin

    calls = []
    backend = SimpleNamespace(regenerate_embeddings=lambda **kw: calls.append(kw) or {})
    storage = SimpleNamespace(engine=object(), get_backend=lambda *_a, **_k: backend)
    client = SimpleNamespace(
        _ensure_initialized=lambda: None, server_url=None, bot_id="public_fixture",
        _get_storage=lambda: storage, _memory_authority=authority,
    )
    monkeypatch.setattr("llm_bawt.memory.embedding_repair._eligible", lambda *_a: True)
    MemoryClientAdminMixin.regenerate_embeddings(client, batch_size=3)
    assert calls == [{"batch_size": 3, "authority": authority}]


@pytest.mark.asyncio
async def test_storage_embedding_repair_forwards_exact_authority(monkeypatch):
    from llm_bawt.mcp_server.storage_admin import StorageAdminMixin

    authority = MemoryAuthority(user_id="user_b")
    calls = []
    backend = SimpleNamespace(regenerate_embeddings=lambda **kw: calls.append(kw) or {})
    storage = SimpleNamespace(engine=object(), _get_backend=lambda *_a, **_k: backend)
    monkeypatch.setattr("llm_bawt.memory.embedding_repair._eligible", lambda *_a: True)
    await StorageAdminMixin.regenerate_embeddings(
        storage, bot_id="public_fixture", batch_size=3, authority=authority)
    assert calls == [{"batch_size": 3, "authority": authority}]


@pytest.mark.asyncio
@pytest.mark.usefixtures("signer")
@pytest.mark.parametrize("user", ["user_a", "user_b"])
async def test_supersede_is_noop_for_either_registered_user(monkeypatch, user):
    def forbidden_storage():
        raise AssertionError("unimplemented supersede reached storage")

    monkeypatch.setattr(tools, "_get_storage", forbidden_storage)
    catalog = CatalogFastMCP("supersede-owner-probe", bot_policy_resolver=Resolver())
    catalog.add_tool(tools.supersede_memory, name="memory_supersede")
    with identity(turn(user=user)):
        await catalog._tool_manager.call_tool("memory_supersede", {
            "bot_id": "public_fixture", "old_memory_id": "foreign", "new_memory_id": "own"})
