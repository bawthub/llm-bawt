"""Summary recall always uses the owner-bound canonical reader, not nearby history."""

import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from llm_bawt.mcp_server.client import MemoryClient, _MCPShortTermManager
from llm_bawt.mcp_server.storage_admin import StorageAdminMixin
from llm_bawt.models.message import Message


@pytest.mark.parametrize("user", ["owner", "other", None])
@pytest.mark.parametrize("server_mode", [False, True])
def test_client_summary_reader_carries_its_bound_user_without_mutating_manager(user, server_mode):
    client = MemoryClient(SimpleNamespace(), bot_id="public_fixture", user_id=user,
                          server_url="http://mcp.invalid" if server_mode else None)
    calls = []

    async def read(**kwargs):
        calls.append(kwargs)
        return []

    client._initialized = True
    client._storage = SimpleNamespace(get_messages_for_summary=read)
    client._call_server = Mock(return_value=[])
    assert client.get_messages_for_summary("summary") == []
    if server_mode:
        client._call_server.assert_called_once_with("messages_get_for_summary", {
            "bot_id": "public_fixture", "summary_id": "summary"})
        assert client._service_principal.user_id == user
    else:
        assert calls == [{"bot_id": "public_fixture", "summary_id": "summary", "user_id": user}]


def test_storage_and_server_adapter_preserve_session_and_typed_author():
    message = Message(role="user", content="Bot-authored request", timestamp=1,
                      db_id="original", session_id="session", author_entity_type="bot", author_entity_id="peer")
    manager = SimpleNamespace(get_messages_for_summary=Mock(return_value=[message]))

    class Storage(StorageAdminMixin):
        def get_short_term_manager(self, bot):
            assert bot == "public_fixture"
            return manager

    rows = asyncio.run(Storage().get_messages_for_summary("public_fixture", "summary", user_id="owner"))
    manager.get_messages_for_summary.assert_called_once_with("summary", user_id="owner")
    adapter = _MCPShortTermManager(SimpleNamespace(get_messages_for_summary=lambda ident: rows))
    [rehydrated] = adapter.get_messages_for_summary("summary")
    assert rehydrated.db_id == "original" and rehydrated.session_id == "session"
    assert rehydrated.author_entity_type == "bot" and rehydrated.author_entity_id == "peer"
