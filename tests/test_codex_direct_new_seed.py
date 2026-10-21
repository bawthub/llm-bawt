"""A direct bare /new must carry an explicit empty seed to Codex."""
from __future__ import annotations

from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

from llm_bawt.service.core import ServiceLLMBawt
from llm_bawt.service.turn_stream_worker import TurnStreamWorker
from llm_bawt.utils.history import Message


def test_nonstream_direct_reset_forwards_empty_seed_to_agent_client():
    client = SimpleNamespace(model_definition={"type": "agent_backend"}, query=Mock(return_value="ok"))
    service = SimpleNamespace(client=client, tool_format="none", bot=SimpleNamespace(uses_tools=False))
    reply, _, _ = ServiceLLMBawt.execute_llm_query(
        service, [Message(role="user", content="/new")],
        inject_messages=[], bridge_reset_action="reset_only",
        thread_binding={"thread_session_id": "new-thread"},
    )
    assert reply == "ok"
    assert client.query.call_args.kwargs["inject_messages"] == []
    assert client.query.call_args.kwargs["bridge_reset_action"] == "reset_only"


def test_stream_direct_reset_forwards_empty_seed_to_agent_client(monkeypatch):
    captured = {}

    def stream_raw(_messages, **kwargs):
        captured.update(kwargs)
        raise RuntimeError("stop after outbound config")

    client = SimpleNamespace(model_definition={"type": "agent_backend"}, stream_raw=stream_raw)
    llm_bawt = SimpleNamespace(
        client=client,
        bot=SimpleNamespace(tts_mode=False, uses_tools=False),
        prepare_messages_for_query=lambda *_args, **_kwargs: [Message(role="user", content="/new")],
        _get_generation_kwargs=lambda: {},
        adapter=None,
    )
    service = SimpleNamespace(
        _turn_log_store=SimpleNamespace(engine=None, get_turn=lambda *_args: None),
        _update_turn_log=Mock(), _resolve_turn_token_usage=Mock(return_value=None),
        config=SimpleNamespace(),
    )
    monkeypatch.setattr("llm_bawt.service.tool_changed_files.ToolChangedFilesCoordinator", Mock())
    monkeypatch.setattr("llm_bawt.service.turn_stream_worker.TurnUsageCoordinator", Mock())
    monkeypatch.setattr("llm_bawt.service.turn_stream_worker.put_queue_item_threadsafe", Mock())
    monkeypatch.setattr("llm_bawt.service.turn_stream_worker.TurnStreamFinalizer.finalize", Mock())
    ctx = SimpleNamespace(
        svc=service, _redis_sub=None, _upstream_model=[None],
        agent_attachments_holder=[], animation_holder=[None], approval_id_holder=[None],
        approval_persist_failed_holder=[False], assistant_message_id=None,
        attachments_to_persist=[], bot_id="codex", cancel_event=Event(),
        cancelled_holder=[False], chunk_queue=None, done_event=Event(),
        full_response_holder=[""], inject_seed_messages=[], is_agent_backend=True,
        llm_bawt=llm_bawt, loop=None, media_store=None, model_alias="fixture",
        question_id_holder=[None], reasoning_holder=[""],
        request=SimpleNamespace(client_system_context=None, ha_mode=False,
                                include_summaries=True, tts_mode=False,
                                inject_user_prefix=False, session_id=None,
                                animations=None, avatar_visible=False,
                                inter_bot_bridge_request_id=None),
        timing_holder=[0, 0], token_usage_holder=[None], tool_call_details_holder=[],
        tool_context_holder=[""], trigger_message_id=None, tts_scrub=None,
        tts_scrubber=None, turn_log_id="turn", user_attachments=[],
        user_author=None, user_id="nick", user_prompt="/new",
        thread_binding={"thread_session_id": "new-thread"},
        bridge_reset_action="reset_only", bridge_prompt=None, chat_reset=None,
        task_turn_capability=None, execution=None,
    )
    TurnStreamWorker(ctx)._stream_to_queue()
    assert captured["inject_messages"] == []
    assert captured["bridge_reset_action"] == "reset_only"
