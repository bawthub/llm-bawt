"""Registered receipt tools must use participant-scoped SQL, never operator HTTP."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from llm_bawt.bot_access import BotAccessError, BotPolicyResolver, TrustedCaller
from llm_bawt.mcp_server import server  # noqa: F401
from llm_bawt.mcp_server import inter_bot_tools
from llm_bawt.mcp_server.bot_access_deliveries import ScopedDeliveryReader
from llm_bawt.mcp_server.catalog import CatalogFastMCP
from tests.test_bot_access_mcp import Resolver


@pytest.mark.asyncio
async def test_registered_receipt_tools_use_scoped_reader_not_http(monkeypatch):
    calls = []

    class Reader:
        def __init__(self, access):
            calls.append(access.searchable_bots)

        async def get(self, delivery_id):
            return {"delivery_id": delivery_id, "error": "not found"}

        async def list(self, **kwargs):
            calls.append(kwargs)
            return {"deliveries": [], "total": 0}

    monkeypatch.setattr(inter_bot_tools, "ScopedDeliveryReader", Reader)
    monkeypatch.setattr(inter_bot_tools.httpx, "AsyncClient", lambda **kw: pytest.fail("operator HTTP bypass"))
    catalog = CatalogFastMCP("receipts", bot_policy_resolver=Resolver())
    catalog.add_tool(inter_bot_tools.get_delivery, name="bots_delivery_get")
    catalog.add_tool(inter_bot_tools.list_deliveries, name="bots_deliveries_list")
    await catalog._tool_manager.call_tool("bots_delivery_get", {"delivery_id": "private-receipt"})
    await catalog._tool_manager.call_tool("bots_deliveries_list", {"limit": 500})
    assert calls[:2] == [frozenset({"public_fixture"})] * 2
    assert calls[2]["limit"] == 50
    for field in ("bot_id", "sender_bot_id", "target_bot_id"):
        with pytest.raises(Exception, match="access_denied"):
            await catalog._tool_manager.call_tool("bots_deliveries_list", {field: "private_fixture"})
    assert len(calls) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("caller", [None, TrustedCaller("private_fixture", "user_a", "service")])
async def test_reader_passes_positive_participants_and_preserves_compact_options(caller):
    access = BotPolicyResolver._compile(Resolver().rows, caller)
    seen = []
    row = SimpleNamespace(to_compact=lambda: {"delivery_id": "allowed"}, to_api=lambda: {"full": True})
    store = SimpleNamespace(engine=object(),
        get=lambda id, **kw: seen.append(kw) or row,
        list=lambda **kw: seen.append(kw) or [row])
    reader = ScopedDeliveryReader(access, store)
    assert await reader.get("allowed") == {"full": True}
    assert await reader.list(bot_id=None, sender_bot_id=None, target_bot_id=None,
        status=None, active_only=True, task_id="task", since_minutes=5,
        include_chat=False, limit=1) == {"deliveries": [{"delivery_id": "allowed"}], "total": 1}
    allowed = frozenset({"public_fixture", "private_fixture"} if caller else {"public_fixture"})
    assert all(call["allowed_bot_ids"] == allowed for call in seen)
    assert seen[1]["exclude_prompt_origin"] is True
    assert seen[1]["exclude_user_chat"] is True
    assert seen[1]["status"] == ("QUEUED", "STEERING", "DISPATCHING")
    assert seen[1]["since"] is not None


@pytest.mark.asyncio
async def test_receipt_storage_errors_are_sanitized_not_empty(monkeypatch):
    access = BotPolicyResolver._compile(Resolver().rows, None)
    reader = ScopedDeliveryReader(access, SimpleNamespace(engine=None))
    with pytest.raises(BotAccessError, match="^source_unavailable$"):
        await reader.get("id")
    with pytest.raises(BotAccessError, match="^source_unavailable$"):
        await reader.list(bot_id=None, sender_bot_id=None, target_bot_id=None,
            status=None, active_only=False, task_id=None, since_minutes=None,
            include_chat=False, limit=10)
