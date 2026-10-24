"""Actual claimed chat query path: persist command, answer once, never model a bare reset."""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from llm_bawt.memory.session_reset_lineage import SessionResetLineage
from llm_bawt.models.message import Message
from llm_bawt.service.schemas import ChatMessage
from llm_bawt.service.session_lifecycle.receipt import RESET_CONFIRMATION
from tests.test_nonstream_turn_execution import _claimed, _isolate, _request, _service

__all__ = ["_isolate"]


@pytest.mark.parametrize("command,answer", [
    ("/new", RESET_CONFIRMATION),
    ("  /new\nKeep this task  ", "task answer"),
])
def test_claimed_chat_nonstream_persists_command_and_exact_answer(command, answer):
    query = Mock(return_value=("task answer", "", []))
    service = _service(query, model_type="openai")
    llm = service._get_llm_bawt.return_value
    llm.history_manager = SimpleNamespace(_db_backend=SimpleNamespace(
        get_session=lambda _sid: None,
        get_active_session=lambda **_kw: {"id": "old-thread"},
    ))
    successor = SessionResetLineage.delivery_successor_id("delivery-test")
    service._rotate_chat_session = Mock(return_value=successor)
    command_message = Message(role="user", content=command)
    llm.prepare_messages_for_query.return_value = [command_message]
    request = _claimed(_request().model_copy(update={
        "messages": [ChatMessage(role="user", content=command)],
        "inter_bot_typed_follow_up": answer != RESET_CONFIRMATION,
    }))

    response = asyncio.run(service.chat_completion(request))
    assert response.choices[0].message.content == answer
    assert service._persist_turn_log.call_args.kwargs["user_prompt"] == command
    assert llm.prepare_messages_for_query.call_args.args[0] == command
    assert llm._session_id_override == successor
    assert service._finalize_turn.call_args.kwargs["response_text"] == answer
    assert service._finalize_turn.call_args.kwargs["user_prompt"] == command
    assert service._maybe_summarize_on_new.call_args.kwargs["thread_binding"] == {
        "thread_session_id": "old-thread",
    }
    service._rotate_chat_session.assert_called_once_with(
        llm, "caid", idempotency_key="delivery-test", working_memory_mode="pinned",
    )
    if answer == RESET_CONFIRMATION:
        query.assert_not_called()
    else:
        query.assert_called_once()
        assert query.call_args.args[0][-1].content == "Keep this task"
        assert command_message.content == command


def test_explicit_chat_thread_treats_new_as_literal_prompt():
    query = Mock(return_value=("literal answer", "", []))
    service = _service(query, model_type="openai")
    llm = service._get_llm_bawt.return_value
    service._rotate_chat_session = Mock(side_effect=AssertionError("rotated unrelated active thread"))
    llm.prepare_messages_for_query.return_value = [Message(role="user", content="/new")]
    request = _request().model_copy(update={
        "session_id": "selected-archive",
        "messages": [ChatMessage(role="user", content="/new")],
    })

    response = asyncio.run(service.chat_completion(request))
    assert response.choices[0].message.content == "literal answer"
    assert query.call_args.args[0][-1].content == "/new"
    assert llm._session_id_override == "selected-archive"
    service._rotate_chat_session.assert_not_called()
    service._maybe_summarize_on_new.assert_not_called()
    assert service._persist_turn_log.call_args.kwargs["user_prompt"] == "/new"


@pytest.mark.parametrize("command,answer", [
    ("/new", RESET_CONFIRMATION),
    ("/new Keep this task", "task answer"),
])
def test_claimed_chat_stream_uses_shared_finalizer_and_app_confirmation(monkeypatch, command, answer):
    from llm_bawt.service import chat_streaming
    from llm_bawt.service.turn_stream_finalize import TurnStreamFinalizer
    from llm_bawt.service.turn_stream_worker import TurnStreamWorker

    service = _service(Mock(), model_type="openai")
    llm = service._get_llm_bawt.return_value
    llm.history_manager = SimpleNamespace(_db_backend=SimpleNamespace(
        get_session=lambda _sid: None,
        get_active_session=lambda **_kw: {"id": "old-thread"},
    ))
    successor = SessionResetLineage.delivery_successor_id("delivery-test")
    service._rotate_chat_session = Mock(return_value=successor)
    service._turn_log_store.get_turn = Mock(return_value=None)
    service._turn_log_store.reap_other_open_turns = Mock(return_value=[])
    llm.bot.uses_tools = False
    llm._get_generation_kwargs = Mock(return_value={})
    llm.adapter = None
    llm.client.stream_raw = Mock(return_value=iter(("task answer",)))
    original = Message(role="user", content=command)
    llm.prepare_messages_for_query.return_value = [original]
    monkeypatch.setattr(
        chat_streaming, "prepare_stream_request_attachments",
        AsyncMock(return_value=(command, [], [], None)),
    )
    monkeypatch.setattr(chat_streaming, "ToolEventCoordinator", Mock())
    monkeypatch.setattr(TurnStreamWorker, "_publish_unified", AsyncMock())
    monkeypatch.setattr(
        "llm_bawt.entity_presentation.EntityPresentationResolver",
        lambda *_a: SimpleNamespace(resolve_many_safe=lambda refs: {
            (r.entity_type, r.entity_id): r.to_dict() for r in refs
        }),
    )
    finalized = []

    def finalize(finalizer, *, prepared_messages):
        finalized.append((finalizer.ctx.full_response_holder[0], prepared_messages))

    monkeypatch.setattr(TurnStreamFinalizer, "_finalize", finalize)
    request = _claimed(_request().model_copy(update={
        "messages": [ChatMessage(role="user", content=command)], "stream": True,
        "inter_bot_typed_follow_up": answer != RESET_CONFIRMATION,
    }))

    async def run():
        return [chunk async for chunk in service.chat_completion_stream(request)]

    chunks = asyncio.run(run())
    texts = []
    for chunk in chunks:
        if not chunk.startswith("data: {"):
            continue
        data = json.loads(chunk[6:])
        texts.extend(
            choice.get("delta", {}).get("content", "")
            for choice in data.get("choices", [])
        )
    assert "".join(texts) == answer
    assert finalized[0][0] == answer
    assert llm.prepare_messages_for_query.call_args.args[0] == command
    assert original.content == command
    assert llm._session_id_override == successor
    if answer == RESET_CONFIRMATION:
        llm.client.stream_raw.assert_not_called()
    else:
        assert llm.client.stream_raw.call_args.args[0][-1].content == "Keep this task"
