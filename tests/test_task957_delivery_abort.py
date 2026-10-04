"""TASK-957: stopped durable turns cannot be revived as a new steer."""

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_bridge.events import AgentEventKind
from claude_code_bridge.command_ops import ClaudeCommandMixin
from claude_code_bridge.event_ops import ClaudeEventMixin
from llm_bawt.service.inter_bot_dispatcher import InterBotDeliveryDispatcher


@pytest.mark.parametrize("status,error", [
    ("aborted", "Aborted via chat.abort"),
    ("cancelled", "Cancelled by user"),
])
def test_stopped_delivery_dispatch_does_not_requeue_for_steering(status, error):
    turn = SimpleNamespace(
        id="turn-delivery-old", status=status, ended_at=datetime.now(timezone.utc),
        error_text=error,
    )
    record = SimpleNamespace(
        id="delivery-old", turn_id=turn.id, claim_token="claim-one",
        target_bot_id="loopy", max_attempts=5, attempt_count=1,
    )
    payload = {"messages": [{"role": "user", "content": "old request"}],
               "bot_id": "loopy", "inter_bot_bridge_request_id": "req_delivery_old"}
    store = SimpleNamespace(
        payload=lambda _id: payload,
        mark_transport_accepted=lambda *_: True,
        get=lambda _id: record,
        turn_state=lambda _id: (status, turn.ended_at),
        requeue=MagicMock(), fail_claim=MagicMock(), mark_delivered=MagicMock(),
        cancel_stopped_claim=MagicMock(return_value=SimpleNamespace(status="CANCELLED")),
    )

    async def stream(_request):
        if False:
            yield None

    service = SimpleNamespace(
        chat_completion_stream=stream,
        _turn_log_store=SimpleNamespace(get_turn=lambda _id: turn),
    )
    dispatcher = object.__new__(InterBotDeliveryDispatcher)
    dispatcher.service = service
    dispatcher.store = store
    dispatcher._heartbeat = AsyncMock()
    dispatcher._emit = AsyncMock()
    asyncio.run(dispatcher._dispatch(record))
    store.requeue.assert_not_called()
    store.mark_delivered.assert_not_called()
    store.fail_claim.assert_not_called()
    store.cancel_stopped_claim.assert_called_once_with("delivery-old", "claim-one")


class _Bridge(ClaudeCommandMixin, ClaudeEventMixin):
    def __init__(self):
        self._backend_name = "claude-code"
        self._trigger_message_ids = {"req-current": "new-trigger"}
        run = SimpleNamespace(request_id="req-current", steer=AsyncMock())
        self._session_queue = SimpleNamespace(get_active_client=lambda _key: run)
        self._publisher = SimpleNamespace(publish_rpc_result=lambda *_: None)
        self.published = []

    def _publish_run_event_with_changed_file(self, _request_id, event):
        self.published.append(event)


@pytest.mark.parametrize("priority", ["now", "next"])
def test_steering_does_not_reassign_active_turn_tool_ownership(priority):
    bridge = _Bridge()
    redis = SimpleNamespace(xack=AsyncMock())
    asyncio.run(bridge._handle_steer({
        "backend": "claude-code", "session_key": "loopy:nick",
        "message": "old delivery", "message_id": "old-trigger",
        "target_request_id": "req-current", "request_id": "steer-old",
        "priority": priority,
    }, "event-1", redis))
    bridge._publish_event("req-current", "loopy:nick", 1, kind=AgentEventKind.TOOL_START,
                          tool_name="Read", tool_use_id="tool-one")
    bridge._publish_event("req-current", "loopy:nick", 2, kind=AgentEventKind.ASSISTANT_DELTA,
                          text="response")
    bridge._publish_event("req-current", "loopy:nick", 3, kind=AgentEventKind.RUN_COMPLETED,
                          token_usage={"input_tokens": 1})
    assert [event.trigger_message_id for event in bridge.published] == ["new-trigger"] * 3
    bridge._session_queue.get_active_client("loopy:nick").steer.assert_awaited_once_with(
        "old delivery", priority=priority, message_id="old-trigger", origin="system"
    )
