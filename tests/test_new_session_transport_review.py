"""Independent checks at the real app execution and typed receipt boundaries."""
from __future__ import annotations

import asyncio
from unittest.mock import Mock

import pytest
from sqlalchemy import text

from llm_bawt.models.message import Message
from llm_bawt.service.schemas import ChatMessage
from llm_bawt.service.session_lifecycle.receipt import NewSessionReceipt
from tests.test_nonstream_turn_execution import _claimed, _isolate, _request, _service

# Reuse the execution harness's claim/capability isolation, not live services.
__all__ = ["_isolate"]


@pytest.mark.parametrize("provider", ["claude-code", "codex"])
@pytest.mark.parametrize("command", [
    "  /new\nKeep\tthese  interior bytes\n/new is task text  ",
    "/new /newspaper is literal task text",
])
def test_nonstream_follow_up_transport_preserves_task_and_visible_command(monkeypatch, provider, command):
    task = command.lstrip()[len("/new"):].strip()
    execute = Mock(return_value=("unchanged task answer", "", []))
    service = _service(execute)
    bawt = service._get_llm_bawt.return_value
    bawt.bot.agent_backend = provider
    visible = Message(role="user", content=command, db_id="trigger")
    bawt.prepare_messages_for_query.return_value = [visible]
    service._resolve_active_thread_binding.return_value = {"thread_session_id": "successor"}
    service._maybe_rotate_agent_session.return_value = "successor"
    monkeypatch.setattr("llm_bawt.service.routes.history.maybe_build_session_seed", lambda *a, **k: [])
    request = _claimed(_request()).model_copy(update={
        "messages": [ChatMessage(role="user", content=command)],
        "inter_bot_typed_follow_up": True,
    })

    response = asyncio.run(service.chat_completion(request))

    execute.assert_called_once()
    args, kwargs = execute.call_args
    assert args[0][-1].content == task
    assert args[0][-1] is not visible
    assert visible.content == command
    assert kwargs["bridge_reset_action"] == "reset_follow_up"
    assert kwargs["thread_binding"]["thread_session_id"] == "successor"
    assert response.choices[0].message.content == "unchanged task answer"
    assert service._persist_turn_log.call_args.kwargs["user_prompt"] == command
    assert bawt.prepare_messages_for_query.call_args.args[0] == command
    service._maybe_rotate_agent_session.assert_called_once()


@pytest.mark.integration
@pytest.mark.usefixtures("private_app_pg")
@pytest.mark.parametrize("provider", ["claude-code", "codex"])
@pytest.mark.parametrize("shape", ["text", "split-image-trailing-assistant"])
def test_text_part_follow_up_receipt_uses_same_command_as_admission(rig, monkeypatch, provider, shape):
    """OpenAI text-part messages are accepted by both chat entrypoints."""
    from tests import test_new_session_typed_follow_up_pg as fixtures

    original_request = fixtures.ChatCompletionRequest

    def text_part_request(**kwargs):
        command = kwargs["messages"][0].content
        parts = [{"type": "text", "text": command}]
        if shape != "text":
            parts = [
                {"type": "text", "text": command[:5]},
                {"type": "image_url", "image_url": {"url": "fixture-only"}},
                {"type": "text", "text": command[5:]},
            ]
        kwargs["messages"] = [ChatMessage(role="user", content=parts)]
        if shape != "text":
            kwargs["messages"].append(ChatMessage(role="assistant", content="not the command"))
        return original_request(**kwargs)

    monkeypatch.setattr(fixtures, "ChatCompletionRequest", text_part_request)
    deliveries, claim, payload, successor, event = fixtures._finished_follow_up(
        rig, provider, "  /new\nKeep this task  ",
    )
    assert isinstance(payload["messages"][0]["content"], list)
    assert NewSessionReceipt.record_provider_ack(
        rig.engine, request_id=event.run_id, bot_id=rig.bot, event=event,
    )
    delivered = deliveries.mark_delivered(
        claim.id, claim.claim_token, response_model="fixture", response_chars=11,
    )
    asyncio.run(fixtures._dispatcher(deliveries)._finish_delivered(delivered))
    settled = deliveries.get(claim.id)
    assert settled.status == "DELIVERED", settled.metadata
    assert "new_session_receipt" in settled.metadata
    with rig.engine.connect() as conn:
        assert conn.execute(text("""
            SELECT count(*) FROM new_session_event_outbox WHERE delivery_id=:id
        """), {"id": claim.id}).scalar_one() == 1
