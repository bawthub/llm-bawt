"""Immediate recovery cannot silently queue a different maintenance turn."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from llm_bawt.service.routes.chat import ChatSteerRequest, steer_active_turn


@pytest.mark.parametrize("prompt,message", [("/compact", "continue"), ("work", "/compact")])
def test_now_rejects_maintenance_without_queue_or_interrupt(monkeypatch, prompt, message):
    queue = AsyncMock(side_effect=AssertionError("must not queue a new turn"))
    monkeypatch.setattr("llm_bawt.service.maintenance_queue.MaintenanceQueue.enqueue_user", queue)
    turn = SimpleNamespace(id="turn", bot_id="bot", user_id="user", user_prompt=prompt)
    service = SimpleNamespace(_turn_log_store=SimpleNamespace(get_turn=lambda _id: turn))
    request = ChatSteerRequest(turn_id="turn", bot_id="bot", user_id="user", message_id="uuid", message=message, priority="now", allow_queue=False)
    with pytest.raises(HTTPException) as raised:
        asyncio.run(steer_active_turn(service, request))
    assert raised.value.status_code == 409
    assert raised.value.detail == "maintenance_requires_separate_turn"
    queue.assert_not_awaited()


def test_next_preserves_explicit_maintenance_queue(monkeypatch):
    queue = AsyncMock(return_value={"ok": True, "detail": "queued", "turn_id": "turn", "message_id": "uuid", "queued": True})
    monkeypatch.setattr("llm_bawt.service.maintenance_queue.MaintenanceQueue.enqueue_user", queue)
    turn = SimpleNamespace(id="turn", bot_id="bot", user_id="user", user_prompt="work")
    service = SimpleNamespace(_turn_log_store=SimpleNamespace(get_turn=lambda _id: turn))
    request = ChatSteerRequest(turn_id="turn", bot_id="bot", user_id="user", message_id="uuid", message="/compact", priority="next")
    assert asyncio.run(steer_active_turn(service, request)).queued
    queue.assert_awaited_once()
