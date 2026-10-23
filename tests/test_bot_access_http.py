"""Service-client wire format through the actual MCP HTTP ASGI transport."""
from __future__ import annotations

import io
from types import SimpleNamespace

import pytest
from starlette.testclient import TestClient

from llm_bawt.bot_access.context import current_bot_access
from llm_bawt.mcp_server.bot_access_identity import BotAccessMiddleware
from llm_bawt.mcp_server.catalog import CatalogFastMCP
from llm_bawt.mcp_server.client import MemoryClient
from tests.test_bot_access_mcp import Resolver, signer  # noqa: F401
from tests.test_mcp_approval_interceptor import FakeStore


def test_service_client_signed_http_roundtrip_and_private_other_denial(monkeypatch, signer):
    seen = []
    server = CatalogFastMCP("http-proof", bot_policy_resolver=Resolver(),
                            stateless_http=True, json_response=True,
                            approval_store_provider=lambda: FakeStore(None, []))

    async def messages_get(bot_id: str, limit: int = 50, session_id: str | None = None,
                           since: float | None = None, include_summaries: bool = False) -> list[dict]:
        seen.append((bot_id, current_bot_access().caller))
        return [{"id": "fixture", "role": "assistant", "content": "private synthetic history"}]

    from llm_bawt.mcp_server import server as tools
    from llm_bawt.tools.executor import ToolExecutor
    from llm_bawt.tools.parser import ToolCall
    from llm_bawt.utils.history import HistoryManager

    async def history_rows(**kwargs):
        return await messages_get(kwargs["bot_id"])

    async def memory_rows(**kwargs):
        assert kwargs["bot_id"] == current_bot_access().caller.bot_id == "private_fixture"
        return [SimpleNamespace(to_dict=lambda: {"id": "m", "content": "synthetic private memory",
                                               "relevance": 1.0, "importance": 0.5, "tags": []})]

    monkeypatch.setattr(tools, "_get_storage", lambda: SimpleNamespace(
        get_messages=history_rows, search_memories=memory_rows))
    server.add_tool(tools.get_messages, name="messages_get")
    server.add_tool(tools.search_memories, name="memory_search")
    server.mark_tools_registered()
    app = server.streamable_http_app()
    with TestClient(BotAccessMiddleware(app), base_url="http://localhost:8000") as http:
        def urlopen(request, **kwargs):
            response = http.post("/mcp", content=request.data, headers=dict(request.header_items()))
            response.raise_for_status()
            return io.BytesIO(response.content)

        monkeypatch.setattr("urllib.request.urlopen", urlopen)
        own = MemoryClient(SimpleNamespace(), bot_id="private_fixture", user_id="user_a",
                           server_url="http://testserver")
        result = own._call_server("messages_get", {"bot_id": "private_fixture"})
        assert result[0]["content"] == "private synthetic history"
        assert (seen[0][1].bot_id, seen[0][1].user_id, seen[0][1].kind) == (
            "private_fixture", "user_a", "service")
        public = MemoryClient(SimpleNamespace(), bot_id="public_fixture", user_id="user_a",
                              server_url="http://testserver")
        with pytest.raises(RuntimeError, match="access_denied"):
            public._call_server("messages_get", {"bot_id": "private_fixture"})
        assert len(seen) == 1
        # Chat-type callers have no bridge capability: context assembly and native
        # tools use the construction-bound service client, including protected self.
        config = SimpleNamespace(HISTORY_FILE=None, SYSTEM_MESSAGE="fixture", MEMORY_N_RESULTS=5,
                                 MEMORY_MIN_RELEVANCE=0.0)
        own.config = config
        history = HistoryManager(SimpleNamespace(), config,
                                 db_backend=own.get_short_term_manager(), bot_id="private_fixture")
        history.load_history()
        payload = history.build_context_payload(delivery="inline", max_tokens=0)
        assert [m.content for m in payload.regular_messages] == ["private synthetic history"]
        seed = history.build_context_payload(delivery="seed", max_tokens=0)
        assert seed.regular_messages == payload.regular_messages
        executor = ToolExecutor(memory_client=own, config=config, user_id="user_a", bot_id="private_fixture")
        output = executor.execute(ToolCall("memory", {"action": "search", "query": "fixture"}, ""))
        assert "synthetic private memory" in output
    assert current_bot_access() is None


def test_userless_operator_history_route_uses_real_signed_mcp(monkeypatch, signer):
    from llm_bawt.mcp_server import server as tools
    from llm_bawt.service.routes.history_pages import _load_sorted_visible_messages

    server = CatalogFastMCP("operator-history", bot_policy_resolver=Resolver(),
                            stateless_http=True, json_response=True,
                            approval_store_provider=lambda: FakeStore(None, []))
    seen = []

    async def history_rows(**kwargs):
        seen.append((kwargs, current_bot_access().caller))
        return [{"id": "synthetic", "session_id": "owned", "role": "assistant",
                 "content": "synthetic history", "timestamp": 1.0}]

    monkeypatch.setattr(tools, "_get_storage", lambda: SimpleNamespace(get_messages=history_rows))
    monkeypatch.setattr(server.bot_access, "_session_owner", lambda sid: ("private_fixture", "user_a"))
    server.add_tool(tools.get_messages, name="messages_get")
    server.mark_tools_registered()
    with TestClient(BotAccessMiddleware(server.streamable_http_app()),
                    base_url="http://localhost:8000") as http:
        def urlopen(request, **kwargs):
            response = http.post("/mcp", content=request.data, headers=dict(request.header_items()))
            response.raise_for_status()
            return io.BytesIO(response.content)

        monkeypatch.setattr("urllib.request.urlopen", urlopen)
        own = MemoryClient(SimpleNamespace(), bot_id="private_fixture",
                           server_url="http://localhost:8000")
        service = SimpleNamespace(get_memory_client=lambda bot: own)
        rows = _load_sorted_visible_messages(service, "private_fixture", session_id="owned")
        assert [row["id"] for row in rows] == ["synthetic"]
        assert seen[0][0]["session_id"] == "owned"
        assert seen[0][1].kind == "service" and seen[0][1].user_id is None
    assert current_bot_access() is None
