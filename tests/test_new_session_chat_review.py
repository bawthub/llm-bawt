"""Independent stateless-chat reset admission checks (TASK-1188)."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from llm_bawt.service.schemas import ChatCompletionRequest, ChatMessage
from llm_bawt.service.turn_admission import AdmissionUnavailable
from tests.test_nonstream_turn_execution import _claimed, _isolate, _service

__all__ = ["_isolate"]


@pytest.mark.parametrize("stream", [False, True], ids=["nonstream", "stream"])
@pytest.mark.parametrize("command", ["/new", "  /new\nKeep this task  "])
def test_rejected_chat_reset_has_no_lifecycle_side_effects(monkeypatch, stream, command):
    service = _service(Mock(side_effect=AssertionError("unadmitted inference")), model_type="openai")
    service._persist_turn_log.side_effect = AdmissionUnavailable("disposable admission rejection")
    service._rotate_chat_session = Mock(return_value="must-not-rotate")
    monkeypatch.setattr(
        "llm_bawt.service.chat_streaming.prepare_stream_request_attachments",
        AsyncMock(return_value=(command, [], [], None)),
    )
    request = _claimed(ChatCompletionRequest(
        model="fixture-model", bot_id="fixture-chat", user="fixture-owner",
        messages=[ChatMessage(role="user", content=command)], stream=stream,
    ))

    async def run():
        if stream:
            async for _ in service.chat_completion_stream(request):
                pass
        else:
            await service.chat_completion(request)

    with pytest.raises(AdmissionUnavailable):
        asyncio.run(run())
    service._maybe_summarize_on_new.assert_not_called()
    service._rotate_chat_session.assert_not_called()
    service._maybe_rotate_agent_session.assert_not_called()
    service._resolve_active_thread_binding.assert_not_called()
    service._start_generation.assert_not_awaited()
    service._get_llm_bawt.return_value.prepare_messages_for_query.assert_not_called()
    service._get_llm_bawt.return_value.execute_llm_query.assert_not_called()
