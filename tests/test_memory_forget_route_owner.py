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
        routes.forget_messages(MemoryForgetRequest(count=3), bot_id="b", client=Mock())
    assert err.value.status_code == 400
    svc.get_memory_client.assert_not_called()


def test_count_forget_uses_the_owner_bound_client_and_thread(service):
    svc, owner_client = service
    shared = Mock()
    resp = routes.forget_messages(
        MemoryForgetRequest(count=3, user_id="alice", session_id="t1"), bot_id="b", client=shared)
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
