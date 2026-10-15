"""The real FastMCP envelope must preserve an absent durable /new successor."""
from __future__ import annotations

import io
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from llm_bawt.mcp_server.client import MemoryClient
from llm_bawt.service.agent_new_rotation import (
    delivery_new_session_id, prepare_delivery_new_binding,
)


def _client(monkeypatch, value, *, structured=True):
    calls = []

    def urlopen(request, **_kwargs):
        payload = json.loads(request.data)
        calls.append(payload["params"])
        item = {"content": [], "isError": False}
        if structured:
            item["structuredContent"] = {"result": value}
        else:
            item["content"] = [{"type": "json", "json": value}]
        wire = {"jsonrpc": "2.0", "id": payload["id"], "result": item}
        return io.BytesIO(json.dumps(wire).encode())

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    client = MemoryClient(SimpleNamespace(), bot_id="caid", user_id="nick",
                          server_url="http://mcp.invalid")
    return client, calls


@pytest.mark.parametrize("structured", [True, False])
def test_explicit_null_mcp_session_is_not_a_wrapper(monkeypatch, structured):
    client, calls = _client(monkeypatch, None, structured=structured)
    assert client.get_session("missing-successor") is None
    assert calls[0]["name"] == "sessions_get"
    assert calls[0]["arguments"]["session_id"] == "missing-successor"


def test_first_delivery_new_uses_absent_successor_without_relaxing_ownership(monkeypatch):
    client, calls = _client(monkeypatch, None)
    backend = client.get_short_term_manager()
    llm_bawt = SimpleNamespace(history_manager=SimpleNamespace(_db_backend=backend))
    service = SimpleNamespace(_resolve_thread_sdk_key=Mock())
    request = SimpleNamespace(inter_bot_delivery_id="delivery-first-new",
                              inter_bot_session_policy=None)
    binding = {"thread_session_id": "outgoing-session"}

    resolved, delivery_id = prepare_delivery_new_binding(
        service, llm_bawt, request, bot_id="caid", user_id="nick",
        user_prompt="/new", thread_binding=binding, backend_name="claude-code",
    )
    assert resolved is binding
    assert delivery_id == "delivery-first-new"
    assert calls[0]["arguments"]["session_id"] == delivery_new_session_id(delivery_id)
    service._resolve_thread_sdk_key.assert_not_called()


def test_existing_successor_still_rejects_wrong_delivery(monkeypatch):
    client, _ = _client(monkeypatch, {
        "id": "present", "bot_id": "caid", "user_id": "nick", "status": "archived",
        "session_metadata": {"rotation_delivery_id": "another-delivery"},
    })
    llm_bawt = SimpleNamespace(history_manager=SimpleNamespace(
        _db_backend=client.get_short_term_manager()))
    with pytest.raises(ValueError, match="belongs to another delivery"):
        prepare_delivery_new_binding(
            SimpleNamespace(), llm_bawt,
            SimpleNamespace(inter_bot_delivery_id="delivery-first-new",
                            inter_bot_session_policy=None),
            bot_id="caid", user_id="nick", user_prompt="/new",
            thread_binding={"thread_session_id": "outgoing"},
            backend_name="claude-code",
        )
