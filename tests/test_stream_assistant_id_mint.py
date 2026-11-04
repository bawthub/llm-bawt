"""TASK-1393: streaming turns without a caller-supplied assistant id mint one."""
from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from llm_bawt.models.message import Message
from llm_bawt.service.schemas import ChatMessage
from tests.test_nonstream_turn_execution import _claimed, _isolate, _request, _service

__all__ = ["_isolate"]


@pytest.mark.parametrize("supplied", [None, "11111111-2222-3333-4444-555555555555"])
def test_stream_turn_events_and_row_share_one_assistant_id(monkeypatch, supplied):
    from llm_bawt.service import chat_streaming
    from llm_bawt.service.turn_stream_finalize import TurnStreamFinalizer
    from llm_bawt.service.turn_stream_worker import TurnStreamWorker

    service = _service(Mock(), model_type="openai")
    llm = service._get_llm_bawt.return_value
    llm.history_manager = SimpleNamespace(_db_backend=SimpleNamespace(
        get_session=lambda _sid: None,
        get_active_session=lambda **_kw: {"id": "thread"},
    ))
    service._turn_log_store.get_turn = Mock(return_value=None)
    service._turn_log_store.reap_other_open_turns = Mock(return_value=[])
    llm.bot.uses_tools = False
    llm._get_generation_kwargs = Mock(return_value={})
    llm.adapter = None
    llm.client.stream_raw = Mock(return_value=iter(("reply",)))
    llm.prepare_messages_for_query.return_value = [Message(role="user", content="hi")]
    monkeypatch.setattr(
        chat_streaming, "prepare_stream_request_attachments",
        AsyncMock(return_value=("hi", [], [], None)),
    )
    monkeypatch.setattr(chat_streaming, "ToolEventCoordinator", Mock())
    published = AsyncMock()
    monkeypatch.setattr(TurnStreamWorker, "_publish_unified", published)
    monkeypatch.setattr(
        "llm_bawt.entity_presentation.EntityPresentationResolver",
        lambda *_a: SimpleNamespace(resolve_many_safe=lambda refs: {
            (r.entity_type, r.entity_id): r.to_dict() for r in refs
        }),
    )
    finalized_ids = []
    monkeypatch.setattr(
        TurnStreamFinalizer, "_finalize",
        lambda finalizer, *, prepared_messages: finalized_ids.append(finalizer.ctx.assistant_message_id),
    )
    request = _claimed(_request().model_copy(update={
        "messages": [ChatMessage(role="user", content="hi")], "stream": True,
        "assistant_message_id": supplied,
    }))

    async def run():
        return [chunk async for chunk in service.chat_completion_stream(request)]

    asyncio.run(run())

    persisted = service._persist_turn_log.call_args.kwargs["assistant_message_id"]
    if supplied:
        assert persisted == supplied
    else:
        assert str(uuid.UUID(persisted)) == persisted
    assert finalized_ids == [persisted]
    starts = [c.args[-1] for c in published.call_args_list if c.args and c.args[-1].get("_type") == "turn_start"]
    assert starts and all(event["assistant_message_id"] == persisted for event in starts)
