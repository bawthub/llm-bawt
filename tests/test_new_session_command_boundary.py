"""Legacy /new token boundaries must match preflight and preprocessing."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from claude_code_bridge.send_preflight import prepare_send
from claude_code_bridge.session_ops import ClaudeSessionMixin


CASES = [
    ("/newspaper", False, "/newspaper"),
    (" \t/newspaper keep these bytes  ", False, " \t/newspaper keep these bytes  "),
    ("/new/task", False, "/new/task"),
    ("/new​task", False, "/new​task"),
    ("/NEW task", False, "/NEW task"),
    ("mention /new here", False, "mention /new here"),
    ("/new", True, None),
    (" \t/new\r\n ", True, None),
    ("/new task", True, "task"),
    (" \t/new\ttask\nsecond line  ", True, "task\nsecond line"),
    ("/new\r\ntask", True, "task"),
    ("/new task", True, "task"),
]


def make_bridge():
    bridge = ClaudeSessionMixin()
    bridge._trigger_message_ids = {}
    bridge._publisher = SimpleNamespace(publish_run_done=Mock())
    bridge._publish_session_reset_unified = Mock()
    bridge._publish_event = Mock()
    bridge._seed_new_session = AsyncMock(return_value={"clean_start": True})
    return bridge


@pytest.mark.parametrize("message,is_reset,remaining", CASES)
@pytest.mark.parametrize("explicit", [False, True])
def test_claude_legacy_preflight_and_parser_agree(message, is_reset, remaining, explicit):
    bridge = make_bridge()
    redis = SimpleNamespace(xack=AsyncMock())
    result = asyncio.run(prepare_send(bridge, {
        "request_id": "request", "session_key": "review-bot:nick", "bot_id": "review-bot",
        "model": "fixture", "message": message, "thread_session_id": "thread",
        "explicit_thread": "1" if explicit else "0", "inject_messages": "[]",
    }, "redis-message", redis))
    reset = is_reset and not explicit
    if reset and remaining is None:
        assert result is None
        bridge._seed_new_session.assert_awaited_once()
        redis.xack.assert_awaited_once()
    else:
        assert result is not None
        assert result.reset_requested is reset
        assert result.message == (remaining if reset else message)
        bridge._seed_new_session.assert_not_awaited()
        redis.xack.assert_not_awaited()
    assert bridge._publish_session_reset_unified.call_count == int(reset)


@pytest.mark.parametrize("message", [" /new task  ", "/newspaper", "\t/new\nliteral task\n"])
def test_claude_structured_follow_up_remains_byte_preserving(message):
    bridge = make_bridge()
    result = asyncio.run(prepare_send(bridge, {
        "request_id": "request", "session_key": "review-bot:nick", "bot_id": "review-bot",
        "model": "fixture", "message": message, "thread_session_id": "thread",
        "bridge_reset_action": "reset_follow_up", "inject_messages": "[]",
    }, "redis-message", SimpleNamespace(xack=AsyncMock())))
    assert result.reset_requested is True and result.message == message
    bridge._seed_new_session.assert_not_awaited()
