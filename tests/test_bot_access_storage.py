"""Guarded aggregate error propagation and embedded principal boundaries."""
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from llm_bawt.bot_access import BotAccessError, BotPolicyResolver, TrustedCaller
from llm_bawt.bot_access.context import bind_bot_access, current_bot_access
from llm_bawt.mcp_server.client import MemoryClient
from llm_bawt.mcp_server.storage_search import StorageSearchMixin

ROWS = [{"slug": "public_fixture", "include_in_global_search": True},
        {"slug": "private_fixture", "include_in_global_search": False}]


def access():
    return BotPolicyResolver._compile(ROWS, TrustedCaller("private_fixture", "user_a", "service"))


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [BotAccessError("source_unavailable"), RuntimeError("backend details")])
async def test_later_aggregate_partition_failure_never_returns_partial_success(monkeypatch, failure):
    from llm_bawt.memory import embeddings
    monkeypatch.setattr(embeddings, "generate_embedding", lambda *_a, **_k: [1, 0, 0])
    storage = StorageSearchMixin()
    storage.config = SimpleNamespace(MEMORY_EMBEDDING_MODEL="test", VERBOSE=False)
    first, second = Mock(), Mock()
    first.search_memories_by_embedding.return_value = [{"id": "allowed", "content": "ok", "similarity": 1}]
    second.search_memories_by_embedding.side_effect = failure
    storage._get_backend = Mock(side_effect=[first, second])
    with bind_bot_access(access()), pytest.raises(BotAccessError, match="^source_unavailable$"):
        await storage.search_all_memories("query")
    assert storage._get_backend.call_count == 2
    own = first.search_memories_by_embedding.call_args.kwargs["eligibility"]
    public = second.search_memories_by_embedding.call_args.kwargs["eligibility"]
    assert own.authority.partition.authority == "caller"
    assert public.authority.partition.authority == "global"
    assert current_bot_access() is None


@pytest.mark.asyncio
async def test_provenance_user_pin_precedes_embedding_work(monkeypatch):
    from llm_bawt.memory import embeddings
    monkeypatch.setattr(embeddings, "generate_embedding", lambda *_a, **_k: pytest.fail("embedding reached"))
    storage = StorageSearchMixin()
    with bind_bot_access(BotPolicyResolver._compile(ROWS, None)):
        with pytest.raises(BotAccessError, match="caller_unavailable"):
            await storage.search_all_memories("query")


def test_embedded_client_rejects_private_other_before_storage(monkeypatch):
    from llm_bawt.mcp_server import client_crossbot
    policy = SimpleNamespace(resolve=AsyncMock(return_value=BotPolicyResolver._compile(
        ROWS, TrustedCaller("public_fixture", "user_a", "service"))))
    monkeypatch.setattr(client_crossbot, "BotPolicyResolver", lambda _engine: policy)
    client = MemoryClient(SimpleNamespace(), bot_id="public_fixture", user_id="user_a")
    storage = Mock()
    client._storage, client._initialized = storage, True
    with pytest.raises(BotAccessError, match="access_denied"):
        client.search_memory_source("private_fixture", "query")
    storage.search_memories.assert_not_called()
    assert current_bot_access() is None


def test_embedded_client_binds_principal_inside_async_adapter(monkeypatch):
    from llm_bawt.mcp_server import client_crossbot
    policy = SimpleNamespace(resolve=AsyncMock(return_value=access()))
    monkeypatch.setattr(client_crossbot, "BotPolicyResolver", lambda _engine: policy)
    client = MemoryClient(SimpleNamespace(), bot_id="private_fixture", user_id="user_a")
    seen = []

    async def search(**kwargs):
        seen.append(current_bot_access().caller)
        return []

    client._storage = SimpleNamespace(_require_engine=lambda: object(), search_all_memories=search)
    client._initialized = True
    assert client.search_all_memories("query") == []
    assert seen == [client._service_principal.caller()]
    assert current_bot_access() is None
