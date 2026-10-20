"""Independent failure-boundary checks for structured provider resets."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, Mock

from agent_bridge.events import AgentEventKind
from tests.test_claude_code_send_boundaries import MODEL, SESSION, _SendHarness


def test_reset_follow_up_does_not_run_task_after_continuity_seed_failure(monkeypatch):
    async def run():
        harness = _SendHarness()
        harness._seed_new_session = AsyncMock(return_value={
            "seeded": False, "reason": "seed write failed: disk unavailable",
        })
        provider = Mock(side_effect=AssertionError("task ran without promised continuity"))
        monkeypatch.setattr("claude_code_bridge.send_handler.ClaudeSDKClient", provider)
        await harness._handle_send({
            "request_id": "turn-request", "session_key": SESSION,
            "model": MODEL, "message": "Do the supplied task",
            "thread_session_id": "fresh-thread", "thread_resume_id": "stale-sdk",
            "trigger_message_id": "message-turn-request",
            "bridge_reset_action": "reset_follow_up",
            "inject_messages": '[{"role":"user","content":"required retained context"}]',
        }, "send-turn-request", harness.redis)
        harness._seed_new_session.assert_awaited_once()
        provider.assert_not_called()
        errors = [event for event in harness.events if event.kind == AgentEventKind.ERROR]
        assert len(errors) == 1 and "seed write failed" in errors[0].text
        assert not any(event.raw.get("new_session_ack") for event in harness.events)
    asyncio.run(asyncio.wait_for(run(), timeout=5))
