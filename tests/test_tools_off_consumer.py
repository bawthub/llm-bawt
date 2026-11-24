"""TASK-2590: a tools-off house's command consumer and inbox wiring."""
from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock, Mock

import pytest

from agent_bridge.houses import HouseSlot
from claude_code_bridge.bridge import ClaudeCodeBridge
from claude_code_bridge.command_ops import TOOLS_OFF_REFUSED_ACTIONS
from tests.test_claude_command_lifecycle import FakeRedis, bridge_harness, cleanup


@pytest.mark.parametrize("tools_off, expected", [("on", False), ("off", True)])
def test_tools_off_house_never_builds_the_inbox_consumer(monkeypatch, tools_off, expected):
    monkeypatch.setenv("BAWT_HOUSE", "reader-a")
    monkeypatch.setenv("BAWT_HOUSE_TOOLS_OFF", tools_off)
    bridge = ClaudeCodeBridge(Mock(), app_api_url="http://app.invalid")
    assert (bridge._inbox_consumer is not None) is expected


class ScriptedRedis(FakeRedis):
    """Delivers each scripted command once, then blocks like an idle stream."""

    def __init__(self, messages):
        super().__init__()
        self.pending = list(messages)
        self.acked = []
        self.done = asyncio.Event()

    async def xreadgroup(self, group, consumer, streams, **kwargs):
        self.step("read")
        if self.pending:
            msg_id, fields = self.pending.pop(0)
            return [(next(iter(streams)), [(msg_id, fields)])]
        self.done.set()
        await asyncio.Event().wait()

    async def xack(self, stream, group, msg_id):
        self.acked.append(msg_id)


COMMANDS = [
    ("1-0", {"action": "chat.steer", "request_id": "steer-1", "session_key": "s", "message": "x"}),
    ("2-0", {"action": "chat.tool_result", "request_id": "tr-1"}),
    ("3-0", {"action": "approval.grant", "request_id": "ag-1", "backend": "claude-code"}),
    ("4-0", {"action": "rpc.call", "request_id": "ping-1", "method": "ping"}),
]


def _run(monkeypatch, tools_off, caplog):
    async def run():
        redis = ScriptedRedis(COMMANDS)
        bridge = bridge_harness()
        bridge._publisher.redis_url = "redis://redis.invalid:6379/0"
        bridge._inbox_consumer = None
        for name in ("_handle_steer", "_handle_tool_result",
                     "_handle_approval_grant", "_handle_rpc"):
            setattr(bridge, name, AsyncMock())
        monkeypatch.setenv("BAWT_HOUSE", "reader-a")
        monkeypatch.setenv("BAWT_HOUSE_TOOLS_OFF", tools_off)
        monkeypatch.setattr("redis.asyncio.from_url", lambda *a, **k: redis)
        monkeypatch.setattr(HouseSlot, "_heartbeat", AsyncMock(side_effect=lambda: None))
        bridge._command_task = asyncio.create_task(bridge._command_listener())
        await asyncio.wait_for(redis.done.wait(), 3)
        await asyncio.sleep(0)
        await cleanup(bridge)
        return bridge, redis
    with caplog.at_level(logging.INFO, logger="claude_code_bridge.bridge"):
        return asyncio.run(asyncio.wait_for(run(), 5))


def test_tools_off_consumer_refuses_steer_tool_result_and_grant(monkeypatch, caplog):
    bridge, redis = _run(monkeypatch, "on", caplog)
    # Refusals never reach a handler; ping/status/abort RPCs still do.
    bridge._handle_steer.assert_not_called()
    bridge._handle_tool_result.assert_not_called()
    bridge._handle_approval_grant.assert_not_called()
    bridge._handle_rpc.assert_called_once()
    assert {"1-0", "2-0", "3-0"} <= set(redis.acked)
    # Steer gets the normal negative RPC result; the others are dropped.
    bridge._publisher.publish_rpc_result.assert_called_once_with(
        "steer-1", {"ok": False, "error": "tools-off house refuses steer"})
    text = caplog.text
    for action in TOOLS_OFF_REFUSED_ACTIONS:
        assert f"tools-off house refused {action}" in text
    assert "tools-off house active: house=reader-a inbox_consumer=False" in text


def test_ordinary_house_still_dispatches_every_action(monkeypatch, caplog):
    bridge, _redis = _run(monkeypatch, "off", caplog)
    bridge._handle_steer.assert_called_once()
    bridge._handle_tool_result.assert_called_once()
    bridge._handle_approval_grant.assert_called_once()
    bridge._publisher.publish_rpc_result.assert_not_called()
    assert "tools-off house active" not in caplog.text
