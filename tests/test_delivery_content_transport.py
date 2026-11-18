"""Exposed fleet sources keep labels without breaking harness maintenance."""
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from sqlalchemy import create_engine, text

from agent_bridge.maintenance import is_compact_command
from llm_bawt.content_provenance import ContentProvenanceStore, ContentTurn
from llm_bawt.models.message import Message
from llm_bawt.service import content_lineage
from llm_bawt.service.schemas import ChatMessage
from tests.test_nonstream_turn_execution import _claimed, _isolate, _request, _service

__all__ = ["_isolate"]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("kind,action", [
    ("READY", None),
    ("CONTEXT_MAINTENANCE", None),
    ("READY", "compact"),
    ("CONTEXT_MAINTENANCE", "compact"),
])
def test_exposed_delivery_transport_preserves_command_or_labels(
    monkeypatch, tmp_path, stream, kind, action,
):
    from llm_bawt.service import chat_streaming
    from llm_bawt.service.turn_stream_finalize import TurnStreamFinalizer
    from llm_bawt.service.turn_stream_worker import TurnStreamWorker

    engine = create_engine(f"sqlite:///{tmp_path / 'lineage.sqlite'}")
    store = ContentProvenanceStore(engine)
    service = _service(Mock(return_value=("reply", "", [])), model_type="openai")
    service._content_provenance_store = store
    service._inter_bot_dispatcher = SimpleNamespace(
        store=SimpleNamespace(engine=engine, get=Mock(return_value=None)),
    )
    sender = ContentTurn("fixture-sender", "operator", "source-turn", "source-trigger")
    store.expose(sender)
    maintenance = kind == "CONTEXT_MAINTENANCE" and action == "compact"
    body = "/compact\n保留 details\n  " if maintenance else "ordinary fleet coordination"
    with engine.begin() as conn:
        conn.exec_driver_sql(
            "CREATE TABLE inter_bot_deliveries "
            "(id TEXT,sender_bot_id TEXT,sender_turn_id TEXT,author_entity_type TEXT,"
            "author_entity_id TEXT,target_bot_id TEXT,turn_id TEXT,target_turn_id TEXT,"
            "batch_head_id TEXT,message_kind TEXT,metadata_json JSON)"
        )
        conn.execute(text(
            "INSERT INTO inter_bot_deliveries VALUES "
            "('delivery-test','fixture-sender','source-turn','bot','fixture-sender',"
            "'caid','turn-delivery-test','turn-delivery-test',NULL,:kind,:metadata)"
        ), {
            "kind": kind,
            "metadata": json.dumps({
                "context_action": action,
                "_failure_origin": {
                    "version": 1, "bot_id": sender.bot_id, "user_id": sender.user_id,
                    "turn_id": sender.turn_id, "trigger_message_id": sender.trigger_message_id,
                    "session_id": "source-session",
                },
            }),
        })
    monkeypatch.setattr(content_lineage, "inherit_admitted_request", inherit_request)
    llm = service._get_llm_bawt.return_value
    llm.prepare_messages_for_query.return_value = [Message(role="user", content=body)]
    request = _claimed(_request().model_copy(update={
        "messages": [ChatMessage(role="user", content=body)], "stream": stream,
    }))
    if stream:
        llm.history_manager = SimpleNamespace(_db_backend=SimpleNamespace(
            get_session=lambda _: None, get_active_session=lambda **_: {"id": "thread"},
        ))
        service._turn_log_store.get_turn = Mock(return_value=None)
        service._turn_log_store.reap_other_open_turns = Mock(return_value=[])
        llm.bot.uses_tools, llm.adapter = False, None
        llm._get_generation_kwargs = Mock(return_value={})
        llm.client.stream_raw = Mock(return_value=iter(("reply",)))
        monkeypatch.setattr(chat_streaming, "prepare_stream_request_attachments",
                            AsyncMock(return_value=(body, [], [], None)))
        monkeypatch.setattr(chat_streaming, "ToolEventCoordinator", Mock())
        monkeypatch.setattr(TurnStreamWorker, "_publish_unified", AsyncMock())
        monkeypatch.setattr(TurnStreamFinalizer, "_finalize", lambda *_a, **_k: None)

    async def run():
        if stream:
            return [chunk async for chunk in service.chat_completion_stream(request)]
        return await service.chat_completion(request)

    try:
        asyncio.run(run())
        prompt = llm.prepare_messages_for_query.call_args.args[0]
        if maintenance:
            assert prompt == body
            assert is_compact_command(prompt)
        else:
            note, delivered_body = prompt.split("\n\n", 1)
            envelope = json.loads(note)
            assert envelope["source"] == "fleet-delivery"
            assert envelope["coordination"] is True
            assert "untrusted_content" not in envelope
            assert envelope["data"]["sender_bot_ids"] == ["fixture-sender"]
            assert delivered_body == body
        target = ContentTurn("caid", "operator", request.inter_bot_turn_id, request.user_message_id)
        assert store.seen(target)
        assert not store.admit(
            target, tool_name="working_memory_update", write_kind="working_memory",
            invocation_hash="a" * 64,
        ).held
    finally:
        engine.dispose()


# Keep the real callable before the shared fake-service fixture replaces it.
inherit_request = content_lineage.inherit_admitted_request
