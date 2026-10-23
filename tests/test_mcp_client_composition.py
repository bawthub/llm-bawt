"""MemoryClient facade contracts after the responsibility split (TASK-1222).

Extraction/dispatch tests only: the split must not change imports, logger
identity, MRO dispatch, or the server/embedded routing of moved methods.
"""

from __future__ import annotations

import io
import json
import logging
from types import SimpleNamespace
from unittest.mock import Mock

from llm_bawt.mcp_server import client as client_module
from llm_bawt.mcp_server import client_transport
from llm_bawt.mcp_server.client import (
    MemoryClient,
    MemoryResult,
    MessageResult,
    _MCPShortTermManager,
    _run_async,
    get_memory_client,
)
from llm_bawt.mcp_server.client_admin import MemoryClientAdminMixin
from llm_bawt.mcp_server.client_crossbot import MemoryClientCrossBotMixin
from llm_bawt.mcp_server.client_transport import MemoryClientTransportMixin


def test_facade_exposes_each_mixin_method_without_shadowing():
    """Removing an inheritance edge or shadowing a moved method must fail."""
    seen = set()
    for mixin in (MemoryClientAdminMixin, MemoryClientCrossBotMixin, MemoryClientTransportMixin):
        for name, method in vars(mixin).items():
            if not callable(method):
                continue
            assert name not in seen
            seen.add(name)
            assert name not in vars(MemoryClient), f"{name} shadowed on the facade"
            assert getattr(MemoryClient, name) is method
    assert {"stats", "run_maintenance", "search_all_messages", "_call_server"} <= seen


def test_public_imports_stay_on_client_module():
    for name in ("MemoryResult", "MessageResult", "_MCPShortTermManager", "_run_async", "get_memory_client"):
        assert hasattr(client_module, name)
    assert MemoryResult.from_dict({"id": "m", "content": "fact"}).tags == []
    assert MessageResult.from_dict({"id": "x", "role": "user"}).bot_id == ""
    assert isinstance(get_memory_client(SimpleNamespace(), bot_id="b"), MemoryClient)


def test_transport_keeps_client_logger_identity():
    assert client_transport.logger is client_module.logger
    assert client_module.logger.name == "llm_bawt.mcp_server.client"


def test_run_async_still_runs_inside_a_running_loop():
    async def value():
        return 7

    async def outer():
        return _run_async(value())

    assert _run_async(value()) == 7
    assert _run_async(outer()) == 7


def _server_client(monkeypatch, value):
    calls = []

    def urlopen(request, **kwargs):
        payload = json.loads(request.data)
        calls.append((payload["params"], kwargs))
        wire = {"jsonrpc": "2.0", "id": payload["id"],
                "result": {"content": [], "structuredContent": {"result": value}}}
        return io.BytesIO(json.dumps(wire).encode())

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    return MemoryClient(SimpleNamespace(), bot_id="public_fixture", user_id="nick",
                        server_url="http://mcp.invalid"), calls


def test_moved_admin_and_crossbot_methods_keep_server_wire_params(monkeypatch):
    client, calls = _server_client(monkeypatch, [{"id": "m1", "content": "c"}])

    assert client.preview_recent_messages(3) == [{"id": "m1", "content": "c"}]
    [hit] = client.search_memory_source("other", "q", n_results=2, tags=["t"])
    assert isinstance(hit, MemoryResult) and hit.id == "m1"
    client.search_all_messages("q", n_results=4, role_filter="user")

    assert [params for params, _ in calls] == [
        {"name": "messages_preview_recent", "arguments": {"bot_id": "public_fixture", "count": 3}},
        {"name": "memory_search_source", "arguments": {
            "source": "other", "query": "q", "n_results": 2, "min_relevance": 0.0, "tags": ["t"]}},
        {"name": "messages_search_all", "arguments": {"query": "q", "n_results": 4, "role_filter": "user"}},
    ]
    assert all(kwargs == {"timeout": 30} for _, kwargs in calls)


def test_moved_transport_still_raises_tool_errors(monkeypatch):
    def urlopen(request, **_kwargs):
        payload = json.loads(request.data)
        wire = {"jsonrpc": "2.0", "id": payload["id"],
                "result": {"isError": True, "content": [{"type": "text", "text": "boom"}]}}
        return io.BytesIO(json.dumps(wire).encode())

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    client = MemoryClient(SimpleNamespace(), bot_id="public_fixture", server_url="http://mcp.invalid")
    try:
        client.stats()
    except RuntimeError as exc:
        assert str(exc) == "MCP tool system_stats failed: boom"
    else:  # pragma: no cover - the assertion is the failure
        raise AssertionError("tool error was swallowed")


def test_moved_methods_keep_embedded_backend_provisioning():
    client = MemoryClient(SimpleNamespace(), bot_id="public_fixture")
    backend = Mock()
    backend.ignore_recent_messages.return_value = 2
    backend.stats.return_value = {"ok": True}
    storage = Mock()
    storage.get_backend.return_value = backend
    client._storage, client._initialized = storage, True

    assert client.stats() == {"ok": True}
    storage.get_backend.assert_called_with("public_fixture")
    assert client.ignore_recent_messages(2) == 2
    storage.get_backend.assert_called_with("public_fixture", provision_schema=True)


def test_server_mode_short_term_manager_is_the_reexported_adapter(monkeypatch):
    client = MemoryClient(SimpleNamespace(), bot_id="public_fixture", server_url="http://mcp.invalid")
    manager = client.get_short_term_manager()
    assert type(manager) is _MCPShortTermManager
    client.clear_messages = Mock(return_value=0)
    assert manager.clear() is True


def test_debug_logging_uses_the_client_logger(monkeypatch, caplog):
    client, _ = _server_client(monkeypatch, {"k": "v"})
    with caplog.at_level(logging.DEBUG, logger="llm_bawt.mcp_server.client"):
        client.stats()
    assert any(r.name == "llm_bawt.mcp_server.client" and "MCP response" in r.getMessage()
               for r in caplog.records)
