"""Wire contract regressions for application user-event writers (TASK-1257)."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from agent_bridge.subscriber import RedisSubscriber
from agent_bridge.user_events import publish_user_event
from llm_bawt.service import approval_events
from llm_bawt.service.session_lifecycle.bridge_reset import BridgeSessionResetMixin


def subscriber(client):
    sub = RedisSubscriber.__new__(RedisSubscriber)
    sub._redis = client
    return sub


def assert_write(client, payload):
    client.xadd.assert_called_once_with(
        "events:test-bot:test-user", {"payload": payload},
        maxlen=5000, approximate=True,
    )


@pytest.mark.parametrize("asynchronous", [False, True])
def test_facade_preserves_encoded_bytes_and_sync_async_execution(asynchronous):
    client = SimpleNamespace(xadd=(AsyncMock if asynchronous else Mock)(return_value="7-0"))
    result = publish_user_event(client, "test-bot", "test-user", '{"text": "café 雪"}')
    if asynchronous:
        client.xadd.assert_not_awaited()
        result = asyncio.run(result)
        client.xadd.assert_awaited_once()
    assert result == "7-0"
    assert_write(client, '{"text": "café 雪"}')


@pytest.mark.parametrize("has_result", [False, True])
def test_approval_resolved_preserves_exact_payload(monkeypatch, has_result):
    monkeypatch.setattr(approval_events.time, "time", lambda: 123.5)
    client = SimpleNamespace(xadd=AsyncMock(return_value="7-0"))
    asyncio.run(approval_events.publish_approval_resolved(
        subscriber(client), "test-bot", "test-user", "approval-1", "turn-1",
        "approved", tool_use_id="tool-1", result={"text": "café 雪"},
        is_error=False, execution_state="succeeded", has_result=has_result,
        resolver=SimpleNamespace(resolved_by="operator", resolved_via="user",
                                 resolved_by_bot=None, resolution_message="ok"),
    ))
    result_fields = '"result": {"text": "café 雪"}, "is_error": false, ' if has_result else ""
    assert_write(client,
        '{"_type": "approval_resolved", "bot_id": "test-bot", "user_id": "test-user", '
        '"request_id": "approval-1", "turn_id": "turn-1", "status": "approved", '
        '"tool_use_id": "tool-1", ' + result_fields +
        '"execution_state": "succeeded", "ts": 123.5, "resolved_by": "operator", '
        '"resolved_via": "user", "resolved_by_bot": null, "resolution_message": "ok"}')
    client.xadd.assert_awaited_once()


def test_thread_switched_stays_synchronous_with_exact_payload(monkeypatch):
    from llm_bawt.service import runtime_switch_rotation
    monkeypatch.setattr(runtime_switch_rotation.time, "time", lambda: 123.5)
    client = SimpleNamespace(xadd=Mock(return_value="7-0"))
    owner = BridgeSessionResetMixin()
    owner._thread_event_redis = client
    owner._publish_thread_switched_blocking("test-bot", "test-user", "session-雪", "new_command")
    assert_write(client,
        '{"_type": "thread_switched", "bot_id": "test-bot", "user_id": "test-user", '
        '"new_session_id": "session-雪", "source": "new_command", "ts": 123.5}')


def test_approval_required_awaits_write_and_closes_client(monkeypatch):
    import redis.asyncio as aioredis
    from llm_bawt.mcp_server.approval_interceptor import ApprovalAwareFastMCP
    client = SimpleNamespace(xadd=AsyncMock(return_value="7-0"), aclose=AsyncMock())
    factory = Mock(return_value=client)
    monkeypatch.setattr(aioredis, "from_url", factory)
    server = ApprovalAwareFastMCP("test")
    server._approval_config = SimpleNamespace(REDIS_URL="redis://unused")
    payload = {"_type": "approval_required", "bot_id": "test-bot",
               "user_id": "test-user", "request_id": "approval-雪", "value": {"a",}}
    asyncio.run(server._publish_approval_required(payload))
    assert_write(client,
        '{"_type": "approval_required", "bot_id": "test-bot", "user_id": "test-user", '
        '"request_id": "approval-雪", "value": "{\'a\'}"}')
    client.xadd.assert_awaited_once()
    client.aclose.assert_awaited_once()
    factory.assert_called_once_with("redis://unused", decode_responses=True)


@pytest.mark.parametrize("asynchronous", [False, True])
def test_transport_failure_propagates_to_callers_best_effort_boundary(asynchronous):
    client = SimpleNamespace(xadd=(AsyncMock if asynchronous else Mock)(
        side_effect=ConnectionError("offline")))
    with pytest.raises(ConnectionError, match="offline"):
        result = publish_user_event(client, "test-bot", "test-user", "{}")
        if asynchronous:
            asyncio.run(result)


def test_all_writers_remain_best_effort_and_required_closes_on_failure(monkeypatch):
    import redis.asyncio as aioredis
    from llm_bawt.mcp_server.approval_interceptor import ApprovalAwareFastMCP
    client = SimpleNamespace(xadd=AsyncMock(side_effect=ConnectionError("offline")),
                             aclose=AsyncMock())
    asyncio.run(approval_events.publish_approval_resolved(
        subscriber(client), "test-bot", "test-user", "r", None, "approved"))
    monkeypatch.setattr(aioredis, "from_url", lambda *a, **kw: client)
    server = ApprovalAwareFastMCP("test")
    server._approval_config = SimpleNamespace(REDIS_URL="redis://unused")
    asyncio.run(server._publish_approval_required(
        {"bot_id": "test-bot", "user_id": "test-user"}))
    client.aclose.assert_awaited_once()
    owner = BridgeSessionResetMixin()
    owner._thread_event_redis = SimpleNamespace(xadd=Mock(side_effect=ConnectionError("offline")))
    owner._publish_thread_switched_blocking("test-bot", "test-user", "s", "new_command")


def test_thread_switched_still_starts_a_daemon_without_running_transport(monkeypatch):
    import threading
    constructor = Mock()
    monkeypatch.setattr(threading, "Thread", constructor)
    owner = BridgeSessionResetMixin()
    owner.config = SimpleNamespace(DEFAULT_USER="fallback")
    owner._publish_thread_switched("test-bot", " test-user ", "s", "new_command")
    constructor.assert_called_once_with(
        target=owner._publish_thread_switched_blocking,
        args=("test-bot", "test-user", "s", "new_command"),
        daemon=True, name="thread-switched-pub-test-bot",
    )
    constructor.return_value.start.assert_called_once_with()
