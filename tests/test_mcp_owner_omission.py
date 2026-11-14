"""MCP history/memory mutations bind to the caller even when user_id is omitted (TASK-2279)."""
from types import SimpleNamespace
import pytest
from llm_bawt.mcp_server import server as tools
from llm_bawt.mcp_server.catalog import CatalogFastMCP
from tests.test_bot_access_mcp import Resolver, identity, signer, turn  # noqa: F401 (signer is a fixture)

pytestmark = pytest.mark.usefixtures("signer")

@pytest.mark.asyncio
@pytest.mark.parametrize("name,handler,extra", [
    ("memory_delete", tools.delete_memory, {"memory_id": "other-users-memory"}),
    ("messages_ignore_by_id", tools.ignore_message_by_id, {"message_id": "other-users-message"}),
    ("messages_restore_ignored", tools.restore_ignored_messages, {}),
])
@pytest.mark.parametrize("bot", ["public_fixture", "private_fixture"])
async def test_owner_omission(monkeypatch, name, handler, extra, bot):
    calls = []
    async def record(**kwargs):
        calls.append(kwargs)
        return 1
    storage = SimpleNamespace(delete_memory=record, ignore_message_by_id=record, restore_ignored_messages=record)
    monkeypatch.setattr(tools, "_get_storage", lambda: storage)
    catalog = CatalogFastMCP("owner-omission-probe", bot_policy_resolver=Resolver())
    catalog.add_tool(handler, name=name)
    with identity(turn(user="user_a")):
        await catalog._tool_manager.call_tool(name, {"bot_id": bot, **extra})
    assert len(calls) == 1
    assert calls[0]["user_id"] == "user_a", "user-bound call reached unscoped storage"


@pytest.mark.asyncio
@pytest.mark.parametrize("name,handler,extra", [
    ("memory_delete", tools.delete_memory, {"memory_id": "m"}),
    ("messages_ignore_by_id", tools.ignore_message_by_id, {"message_id": "m"}),
    ("messages_restore_ignored", tools.restore_ignored_messages, {}),
])
async def test_other_user_is_refused(monkeypatch, name, handler, extra):
    calls = []
    async def record(**kwargs):
        calls.append(kwargs)
        return 1
    storage = SimpleNamespace(delete_memory=record, ignore_message_by_id=record, restore_ignored_messages=record)
    monkeypatch.setattr(tools, "_get_storage", lambda: storage)
    catalog = CatalogFastMCP("owner-omission-probe", bot_policy_resolver=Resolver())
    catalog.add_tool(handler, name=name)
    with identity(turn(user="user_a")), pytest.raises(Exception):
        await catalog._tool_manager.call_tool(name, {"bot_id": "public_fixture", "user_id": "user_b", **extra})
    assert calls == []
