"""Summary recall always uses the owner-bound canonical reader, not nearby history."""

import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from llm_bawt.mcp_server.client import MemoryClient, _MCPShortTermManager
from llm_bawt.mcp_server.storage_admin import StorageAdminMixin
from llm_bawt.models.message import Message
from llm_bawt.tools.executor import ToolExecutor
from llm_bawt.tools.parser import ToolCall


@pytest.mark.parametrize("in_memory", [False, True])
def test_executor_uses_canonical_reader_and_marks_only_its_results(in_memory):
    client = Mock()
    client.get_messages_for_summary.return_value = [
        {"id": "original", "role": "user", "content": "Early canonical original", "timestamp": 1000}]
    history = SimpleNamespace(messages=[
        Message(role="summary", content="Summary A", db_id="summary", timestamp=10000),
        Message(role="user", content="UNRELATED-NEARBY", db_id="nearby", timestamp=10001),
    ]) if in_memory else None
    executor = ToolExecutor(user_id="owner", memory_client=client, history_manager=history)
    response = executor.execute(ToolCall(name="history", arguments={
        "action": "recall", "summary_id": "summary"}, raw_text=""))
    assert "Early canonical original" in response and "UNRELATED" not in response
    client.get_messages_for_summary.assert_called_once_with("summary")
    client.mark_messages_recalled.assert_called_once_with(["original"])


def test_empty_canonical_reader_never_falls_back_to_nearby_history():
    client = Mock()
    client.get_messages_for_summary.return_value = []
    history = SimpleNamespace(messages=[
        Message(role="summary", content="Summary A", db_id="summary", timestamp=10000),
        Message(role="user", content="UNRELATED-NEARBY", db_id="nearby", timestamp=10001)])
    executor = ToolExecutor(user_id="owner", memory_client=client, history_manager=history)
    response = executor.execute(ToolCall(name="history", arguments={
        "action": "recall", "summary_id": "summary"}, raw_text=""))
    assert "UNRELATED" not in response and "verified session provenance" in response
    client.mark_messages_recalled.assert_not_called()


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
