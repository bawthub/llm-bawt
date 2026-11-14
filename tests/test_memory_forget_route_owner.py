"""TASK-2279: the forget/preview routes select only one user's messages."""

from __future__ import annotations

from unittest.mock import Mock

import pytest
from fastapi import HTTPException

from llm_bawt.service.routes import memory as routes
from llm_bawt.service.schemas import MemoryForgetRequest


@pytest.fixture
def service(monkeypatch):
    svc = Mock()
    owner_client = Mock()
    owner_client.forget_recent_messages.return_value = {"messages_ignored": 2, "memories_deleted": 1}
    owner_client.preview_messages_since_minutes.return_value = []
    svc.get_memory_client.return_value = owner_client
    monkeypatch.setattr(routes, "get_service", lambda: svc)
    return svc, owner_client


def test_count_forget_without_owner_is_refused(service):
    svc, _ = service
    with pytest.raises(HTTPException) as err:
        routes.forget_messages(MemoryForgetRequest(count=3), bot_id="b")
    assert err.value.status_code == 400
    svc.get_memory_client.assert_not_called()


def test_count_forget_uses_the_owner_bound_client_and_thread(service):
    svc, owner_client = service
    shared = Mock()
    resp = routes.forget_messages(
        MemoryForgetRequest(count=3, user_id="alice", session_id="t1"), bot_id="b")
    svc.get_memory_client.assert_called_once_with("b", "alice")
    owner_client.forget_recent_messages.assert_called_once_with(3, session_id="t1")
    shared.forget_recent_messages.assert_not_called()
    assert (resp.messages_ignored, resp.memories_deleted) == (2, 1)


def test_preview_requires_owner(service):
    svc, owner_client = service
    with pytest.raises(HTTPException) as err:
        routes.preview_messages_since_minutes(minutes=5, user_id=None, bot_id="b")
    assert err.value.status_code == 400
    routes.preview_messages_since_minutes(minutes=5, user_id="bob", bot_id="b")
    svc.get_memory_client.assert_called_once_with("b", "bob")
    owner_client.preview_messages_since_minutes.assert_called_once_with(5)


def _embedded_client(user_id):
    from types import SimpleNamespace
    from llm_bawt.mcp_server.client import MemoryClient
    client = MemoryClient(SimpleNamespace(), bot_id="b", user_id=user_id)
    backend = Mock()
    backend.forget_owned_messages.return_value = {"messages_ignored": 1, "memories_deleted": 0}
    storage = Mock()
    storage.get_backend.return_value = backend
    client._storage, client._initialized = storage, True
    return client, backend


def test_in_chat_history_forget_is_owner_scoped():
    """The executor's history tool forgets through its user-bound client."""
    from llm_bawt.tools.executor import ToolExecutor
    from llm_bawt.tools.parser import ToolCall
    client, backend = _embedded_client("alice")
    executor = ToolExecutor(memory_client=client, user_id="alice", bot_id="b")
    out = executor._history_forget(ToolCall("history", {"action": "forget", "count": 2}, ""))
    assert "Forgot 1 messages" in out
    backend.forget_owned_messages.assert_called_once_with(count=2, user_id="alice")
    executor._history_forget(ToolCall("history", {"action": "forget", "minutes": 5}, ""))
    backend.forget_owned_messages.assert_called_with(minutes=5, user_id="alice")


def test_mcp_forget_tools_take_the_owner_from_the_signed_caller(monkeypatch):
    import asyncio
    from llm_bawt.mcp_server import server
    seen = []
    monkeypatch.setattr(server, "history_caller_user", lambda bot, user, **kw: seen.append((bot, user, kw)) or "alice")
    storage = Mock()

    async def forget(**kw):
        return {"called": kw}
    storage.forget_owned_messages = forget
    monkeypatch.setattr(server, "_get_storage", lambda: storage)
    fn = getattr(server.forget_messages, "fn", server.forget_messages)
    result = asyncio.run(fn(bot_id="b", count=3, session_id="t1"))
    assert result["called"] == {"bot_id": "b", "user_id": "alice", "session_id": "t1", "count": 3, "minutes": None}
    # Owner is never taken from an argument and never via the operator path.
    assert seen == [("b", None, {})]
