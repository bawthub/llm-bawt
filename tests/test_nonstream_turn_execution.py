"""Non-stream agent turns own a live execution handle (TASK-1015).

Before TASK-1015 ``chat_completion`` (the waited ``bots_send_message`` path)
never registered a ``TurnExecution`` and never minted the task-turn
capability, so the bridge never persisted ``agent_session_key`` /
``agent_request_id``: steer answered 409 "Active bridge run is not ready",
Stop answered 503 ``execution_not_confirmed``, the reaper could not tell the
turn was owned, and ``tasks_associate_current`` saw no trusted context.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from llm_bawt.service.background_service import BackgroundService
from llm_bawt.service.schemas import ChatCompletionRequest, ChatMessage
from llm_bawt.service.turn_execution import turn_executions


def _service(execute_llm_query, *, model_type: str = "claude-code") -> BackgroundService:
    service = BackgroundService.__new__(BackgroundService)
    service.config = SimpleNamespace(DEFAULT_BOT="nova", DEFAULT_USER="nick")
    service._redis_subscriber = SimpleNamespace(publish_tool_event=AsyncMock())
    service._resolve_request_model = Mock(return_value=("test-model", []))
    service._persist_turn_log = Mock(return_value=None)
    service._update_turn_log = Mock()
    service._end_generation = Mock()
    service._start_generation = AsyncMock(
        return_value=(threading.Event(), threading.Event())
    )
    service._bind_agent_thread = Mock(return_value=None)
    service._resolve_active_thread_binding = Mock(
        return_value={"thread_session_id": "sess-1"}
    )
    service._maybe_summarize_on_new = Mock()
    service._maybe_rotate_agent_session = Mock()
    service._finalize_turn = Mock()
    service._turn_log_store = SimpleNamespace(engine=object(), accept_session_binding=Mock())
    service._get_llm_bawt = Mock(return_value=SimpleNamespace(
        client=SimpleNamespace(
            model_definition={"type": model_type},
            get_token_usage=Mock(return_value=None),
        ),
        bot=SimpleNamespace(tts_mode=False, agent_backend="claude-code"),
        prepare_messages_for_query=Mock(return_value=[]),
        execute_llm_query=execute_llm_query,
    ))
    return service


def _request() -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="test-model",
        messages=[ChatMessage(role="user", content="do the task")],
        bot_id="caid",
        user="nick",
        stream=False,
        user_message_id="00000000-0000-4000-8000-000000000001",
    )


def _claimed(request: ChatCompletionRequest) -> ChatCompletionRequest:
    """Run execution tests as dispatcher-owned turns, not HTTP direct starts."""
    return request.model_copy(update={
        "inter_bot_delivery_id": "delivery-test",
        "inter_bot_turn_id": "turn-delivery-test",
        "inter_bot_claim_token": "claim-test",
        "inter_bot_bridge_request_id": "req_delivery_test",
    })


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    from llm_bawt.service import chat_nonstream, chat_streaming
    from llm_bawt.service.turn_admission import AdmissionClaim

    for module in (chat_nonstream, chat_streaming):
        validate, claim = module.validate_inter_bot_claim, module.turn_admission_claim
        monkeypatch.setattr(module, "validate_inter_bot_claim", lambda svc, req, original=validate: (
            None if getattr(req, "inter_bot_delivery_id", None) == "delivery-test"
            else original(svc, req)
        ))
        monkeypatch.setattr(module, "turn_admission_claim", lambda svc, req, original=claim: (
            AdmissionClaim(owner="claim-test", fence=lambda _conn: True)
            if getattr(req, "inter_bot_delivery_id", None) == "delivery-test"
            else original(svc, req)
        ))
    monkeypatch.setattr("llm_bawt.service.chat_nonstream.get_bot", lambda _b: None)
    monkeypatch.setattr(
        "llm_bawt.service.routes.history.maybe_build_session_seed",
        lambda *_a, **_k: None,
    )
    minted: list[dict] = []

    def mint(**kwargs):
        minted.append(kwargs)
        return "cap-token"

    monkeypatch.setattr("llm_bawt.task_turn_context.mint_for_agent_turn", mint)
    before = set(turn_executions.active_ids())
    yield minted
    for turn_id in set(turn_executions.active_ids()) - before:
        turn_executions.remove(turn_id)


@pytest.mark.parametrize("stream,publish_fails", [(True, False), (False, False), (False, True)])
@pytest.mark.parametrize("explicit", [True, False], ids=["explicit-archive", "implicit-return"])
def test_resumption_event_is_after_accepted_binding_and_user_anchor(monkeypatch, _isolate, stream, publish_fails,
                                                                   explicit):
    import json
    from llm_bawt.service.turn_stream_worker import TurnStreamWorker

    order = []
    receipt = {}
    service = _service(Mock(return_value=("answer", "", [])))
    binding = {"thread_session_id": "old-thread", "explicit_thread": explicit}
    service._bind_agent_thread.return_value = binding
    service._resolve_active_thread_binding.return_value = binding

    def accept(**kwargs):
        assert kwargs["record_resumption"] is True
        order.append("accepted")
        receipt.update(turn_id=kwargs["turn_id"], session_id="old-thread",
                       previous_session_id="current", accepted_at=100.0,
                       session_started_at=10.0, title="Earlier work")
        return dict(receipt)

    service._turn_log_store.accept_session_binding.side_effect = accept
    from llm_bawt.message_authorship import AuthorReference
    for module in ("chat_streaming", "chat_nonstream"):
        monkeypatch.setattr(f"llm_bawt.service.{module}.validate_inter_bot_claim",
                            lambda *_a: AuthorReference.bot("fixture-sender"))
    events = []

    async def publish(event):
        events.append(event)
        order.append(event["_type"])
        if publish_fails and event["_type"] == "session_resumed":
            raise RuntimeError("optional resumption publication failed")

    async def nonstream_publish(**kw):
        await publish(kw["event"])
    service._publish_nonstream_turn_event = nonstream_publish
    monkeypatch.setattr(TurnStreamWorker, "_publish_unified", lambda _worker, event: publish(event))
    monkeypatch.setattr("llm_bawt.service.chat_streaming.ToolEventCoordinator", Mock())
    monkeypatch.setattr("llm_bawt.service.chat_streaming.prepare_stream_request_attachments",
                        AsyncMock(return_value=("do the task", [], [], None)))
    request = _claimed(_request().model_copy(update={"stream": stream,
                                                     "session_id": "old-thread" if explicit else None}))

    async def run():
        if not stream:
            await service.chat_completion(request)
            return None
        generator = service.chat_completion_stream(request)
        try:
            # Ignore the transport's initial connected comment. First data
            # is committed metadata, before starting inference.
            async for chunk in generator:
                if chunk.startswith("data:"):
                    return chunk
        finally:
            await generator.aclose()

    chunk = asyncio.run(run())
    assert any(event["_type"] == "session_resumed" for event in events), (chunk, order)
    resumed = next(event for event in events if event["_type"] == "session_resumed")
    assert resumed["session_resumption"] == receipt
    assert resumed["session_id"] == "old-thread" and resumed["trigger_message_id"] == request.user_message_id
    assert order.index("session_resumed") > order.index("accepted")
    assert order.index("session_resumed") > order.index("turn_start")
    if stream:
        http = json.loads(chunk.removeprefix("data: ").strip())
        assert http["object"] == "chat.completion.session_resumed"
        assert http["session_resumption"] == receipt
        service._get_llm_bawt.return_value.execute_llm_query.assert_not_called()
    else:
        # An unavailable optional metadata channel must not stop inference.
        service._get_llm_bawt.return_value.execute_llm_query.assert_called_once()


def test_agent_nonstream_turn_registers_execution_and_capability(_isolate) -> None:
    seen: dict = {}

    def execute_llm_query(*_args, **kwargs):
        seen["kwargs"] = kwargs
        # Live while the bridge runs: Stop/steer/reaper can find the owner.
        execution = kwargs["turn_execution"]
        seen["registered"] = turn_executions.get(execution.turn_id)
        return "done", "", []

    service = _service(execute_llm_query)
    asyncio.run(service.chat_completion(_claimed(_request())))

    execution = seen["kwargs"]["turn_execution"]
    assert execution is not None and seen["registered"] is execution
    assert execution.turn_id.startswith("turn-")
    assert seen["kwargs"]["task_turn_capability"] == "cap-token"
    assert _isolate[0]["turn_id"] == execution.turn_id
    assert _isolate[0]["session_id"] == "sess-1"
    # Removed once the turn ends.
    assert turn_executions.get(execution.turn_id) is None


@pytest.mark.parametrize("model_type", ["claude-code", "openai"])
def test_claimed_nonstream_delivery_gets_outbound_only_reply_note(_isolate, model_type) -> None:
    """Loopy r1: claimed stream=false turns must carry the note too (agents only)."""
    from llm_bawt.inter_bot_reply_channel import delivery_reply_channel_note
    from llm_bawt.message_authorship import AuthorReference

    record = SimpleNamespace(reply_mode="async", author=AuthorReference.bot("vex"),
                             message_kind=None, target_bot_id="caid")
    lookups: list[str] = []
    service = _service(Mock(return_value=("done", "", [])), model_type=model_type)
    service._inter_bot_dispatcher = SimpleNamespace(store=SimpleNamespace(
        get=lambda delivery_id: lookups.append(delivery_id) or record,
    ))
    asyncio.run(service.chat_completion(_claimed(_request())))

    prepare = service._get_llm_bawt.return_value.prepare_messages_for_query
    assert prepare.call_args.args[0] == "do the task"  # stored prompt stays clean
    if model_type == "claude-code":
        assert prepare.call_args.kwargs["context_suffix"] == delivery_reply_channel_note(record)
        assert lookups == ["delivery-test"]
    else:
        assert prepare.call_args.kwargs["context_suffix"] is None


def test_nonstream_multipart_text_survives_response_accounting(_isolate) -> None:
    service = _service(Mock(return_value=("task answer", "", [])))
    request = _claimed(_request().model_copy(update={"messages": [ChatMessage(
        role="user", content=[
            {"type": "text", "text": "say "},
            {"type": "image_url", "image_url": {"url": "asset"}},
            {"type": "text", "text": "hello"},
        ],
    )]}))

    response = asyncio.run(service.chat_completion(request))
    assert response.choices[0].message.content == "task answer"
    assert response.usage.prompt_tokens == len("say hello") // 4
    assert service._get_llm_bawt.return_value.prepare_messages_for_query.call_args.args[0] == "say hello"


def test_execution_is_released_when_the_query_raises(_isolate) -> None:
    seen: dict = {}

    def execute_llm_query(*_args, **kwargs):
        seen["turn_id"] = kwargs["turn_execution"].turn_id
        seen["live"] = turn_executions.get(seen["turn_id"]) is not None
        raise RuntimeError("bridge exploded")

    service = _service(execute_llm_query)
    try:
        asyncio.run(service.chat_completion(_claimed(_request())))
    except Exception:
        pass
    assert seen["live"] is True
    assert turn_executions.get(seen["turn_id"]) is None


def test_chat_backend_nonstream_turn_does_not_register(_isolate) -> None:
    seen: dict = {}

    def execute_llm_query(*_args, **kwargs):
        seen["kwargs"] = kwargs
        return "done", "", []

    service = _service(execute_llm_query, model_type="openai")
    asyncio.run(service.chat_completion(_request()))
    assert seen["kwargs"]["turn_execution"] is None
    assert seen["kwargs"]["task_turn_capability"] is None
    assert _isolate == []


@pytest.mark.parametrize("stream", [True, False], ids=["stream", "nonstream"])
def test_failed_admission_does_not_cancel_running_chat_generation(stream, monkeypatch):
    from llm_bawt.service.turn_admission import AdmissionUnavailable

    execute = Mock(side_effect=AssertionError("unadmitted request reached inference"))
    service = _service(execute, model_type="openai")
    service._persist_turn_log.side_effect = AdmissionUnavailable("DB unavailable")
    if stream:
        monkeypatch.setattr(
            "llm_bawt.service.chat_streaming.prepare_stream_request_attachments",
            AsyncMock(return_value=("do the task", [], [], None)),
        )
        request = _request().model_copy(update={"stream": True})

        async def run():
            async for _ in service.chat_completion_stream(request):
                pass
    else:
        async def run():
            await service.chat_completion(_request())

    with pytest.raises(AdmissionUnavailable):
        asyncio.run(run())
    service._start_generation.assert_not_awaited()
    execute.assert_not_called()


@pytest.mark.parametrize("stream", [True, False], ids=["stream", "nonstream"])
@pytest.mark.parametrize("explicit", [False, True], ids=["active", "explicit"])
def test_rejected_agent_new_has_no_thread_side_effects(explicit, monkeypatch, _isolate, stream):
    from llm_bawt.service.turn_admission import AdmissionUnavailable

    service = _service(Mock(side_effect=AssertionError("unadmitted bridge execution")))
    service._persist_turn_log.side_effect = AdmissionUnavailable("DB unavailable")
    if explicit:
        service._bind_agent_thread.return_value = {
            "thread_session_id": "selected-thread", "explicit_thread": True,
        }
    seed = Mock(return_value=[])
    monkeypatch.setattr("llm_bawt.service.dependencies.get_service", lambda: service)
    monkeypatch.setattr("llm_bawt.service.routes.history.maybe_build_session_seed", seed)
    monkeypatch.setattr(
        "llm_bawt.service.chat_streaming.prepare_stream_request_attachments",
        AsyncMock(return_value=("/new", [], [], None)),
    )
    request = ChatCompletionRequest(
        model="test-model", messages=[ChatMessage(role="user", content="/new")],
        bot_id="caid", user="nick", stream=stream,
    )
    if explicit:
        request.session_id = "selected-thread"

    async def run():
        if stream:
            async for _ in service.chat_completion_stream(_claimed(request)):
                pass
        else:
            await service.chat_completion(_claimed(request))

    with pytest.raises(AdmissionUnavailable):
        asyncio.run(run())
    # A prior /new rotated the durable thread, summarized, and seeded before
    # save_turn rejected the request. Even a cold-start resolver can INSERT a
    # fresh active session, so no unscoped binding is allowed before admission.
    service._resolve_active_thread_binding.assert_not_called()
    service._maybe_summarize_on_new.assert_not_called()
    seed.assert_not_called()
    service._maybe_rotate_agent_session.assert_not_called()
    assert _isolate == []


@pytest.mark.parametrize("stream", [True, False], ids=["stream", "nonstream"])
@pytest.mark.parametrize("mode", ["typed", "retain", "clean"])
def test_accepted_agent_new_seeds_outgoing_then_routes_to_fresh_thread(monkeypatch, _isolate, mode, stream):
    class StopAfterSetup(Exception):
        pass

    order = []
    execute = Mock(side_effect=StopAfterSetup)
    service = _service(execute)
    service._persist_turn_log.side_effect = lambda **_kw: order.append("admit")
    binding = (
        {"thread_session_id": "old-thread", "thread_resume_id": "old-sdk"}
        if mode == "typed" else {"thread_session_id": "fresh-thread"}
    )
    service._resolve_active_thread_binding.side_effect = lambda *_a: (
        order.append("bind") or dict(binding)
    )
    service._maybe_summarize_on_new.side_effect = lambda *_a, **_kw: order.append("summarize")

    def rotate(*_args, **_kwargs):
        order.append("rotate")
        binding.clear()
        binding["thread_session_id"] = "fresh-thread"

    service._maybe_rotate_agent_session.side_effect = rotate

    def seed(*_args, **kwargs):
        assert kwargs["thread_binding"]["thread_session_id"] == "old-thread"
        order.append("seed")
        return [{"role": "user", "content": "outgoing context"}]

    def retained_seed(*_args, **kwargs):
        assert kwargs["session_id"] == "old-thread"
        order.append("seed")
        return {"messages": [{"role": "user", "content": "outgoing context"}]}

    monkeypatch.setattr("llm_bawt.service.dependencies.get_service", lambda: service)
    monkeypatch.setattr("llm_bawt.service.routes.history.maybe_build_session_seed", seed)
    monkeypatch.setattr("llm_bawt.service.routes.history.build_context_seed", retained_seed)
    monkeypatch.setattr("llm_bawt.service.chat_streaming.ToolEventCoordinator",
                        Mock(side_effect=StopAfterSetup))
    prompt = "/new" if mode == "typed" else "do the task"
    monkeypatch.setattr(
        "llm_bawt.service.chat_streaming.prepare_stream_request_attachments",
        AsyncMock(return_value=(prompt, [], [], None)),
    )
    request = ChatCompletionRequest(
        model="test-model", messages=[ChatMessage(role="user", content=prompt)],
        bot_id="caid", user="nick", stream=stream,
    )
    if mode != "typed":
        request.inter_bot_session_policy = (
            "reset_retain_history" if mode == "retain" else "reset_without_history"
        )
        request.inter_bot_seed_session_id = "old-thread"

    async def run():
        if stream:
            async for _ in service.chat_completion_stream(_claimed(request)):
                pass
        else:
            await service.chat_completion(_claimed(request))

    with pytest.raises(StopAfterSetup):
        asyncio.run(run())
    expected = {
        "typed": ["admit", "bind", "summarize", "seed", "rotate", "bind"],
        "retain": ["admit", "bind", "summarize", "seed"],
        "clean": ["admit", "bind"],
    }
    expected_order = expected[mode]
    if not stream:
        # Nonstream resolves the same current binding for its early turn_start.
        expected_order = ["admit", "bind", *expected_order[1:]]
    assert order == expected_order
    assert _isolate[0]["session_id"] == "fresh-thread"
    if mode != "typed":
        service._maybe_rotate_agent_session.assert_not_called()
    if stream:
        execute.assert_not_called()
        service._update_turn_log.assert_not_called()
    else:
        execute.assert_called_once()
        assert execute.call_args.kwargs["thread_binding"]["thread_session_id"] == "fresh-thread"
        assert execute.call_args.kwargs["inject_messages"] == (
            [] if mode == "clean" else [{"role": "user", "content": "outgoing context"}]
        )
        service._update_turn_log.assert_called_once()
        assert service._update_turn_log.call_args.kwargs["status"] == "error"


@pytest.mark.parametrize("row_kind", ["owned", "other_user", "missing"])
def test_original_runtime_stream_passes_only_row_verified_service_context(monkeypatch, _isolate, row_kind):
    """Actual chat coordinator + original worker, not an agent-tool assumption."""
    from llm_bawt.bot_access.service import _runtime_turn
    from llm_bawt.service.session_lifecycle.bridge_reset import BridgeSessionResetMixin
    from llm_bawt.service.turn_stream_finalize import TurnStreamFinalizer
    from llm_bawt.service.turn_stream_worker import TurnStreamWorker

    service = _service(Mock())
    llm = service._get_llm_bawt.return_value
    llm.bot.slug = "caid"
    llm.user_id = "nick"
    row = {"id": "successor", "bot_id": "caid", "user_id": "nick", "session_metadata": {}}
    if row_kind == "other_user":
        row["user_id"] = "other_fixture_user"
    llm.history_manager = SimpleNamespace(_db_backend=SimpleNamespace(
        get_session=lambda _sid: row if row_kind != "missing" else None))
    service._resolve_thread_sdk_key = BridgeSessionResetMixin._resolve_thread_sdk_key.__get__(service)
    service._bind_agent_thread = BridgeSessionResetMixin._bind_agent_thread.__get__(service)
    service._turn_log_store.get_turn = Mock(return_value=None)
    seen = []

    def prepare(*_a, **_kw):
        seen.append((_runtime_turn.get(), llm._session_id_override))
        raise RuntimeError("fixture stop after runtime history boundary")

    llm.prepare_messages_for_query = prepare
    monkeypatch.setattr(TurnStreamFinalizer, "_finalize", lambda *_a, **_kw: None)
    monkeypatch.setattr(TurnStreamWorker, "_publish_unified", AsyncMock())
    monkeypatch.setattr("llm_bawt.service.chat_streaming.ToolEventCoordinator", Mock())
    monkeypatch.setattr("llm_bawt.service.chat_streaming.prepare_stream_request_attachments",
                        AsyncMock(return_value=("/new", [], [], None)))
    request = _claimed(_request().model_copy(update={"stream": True, "session_id": "successor",
        "messages": [ChatMessage(role="user", content="/new")],
        "inter_bot_session_policy": "reset_without_history"}))

    async def run():
        return [chunk async for chunk in service.chat_completion_stream(request)]

    asyncio.run(run())
    assert seen == [("cap-token" if row_kind == "owned" else None, "successor")]
    assert _isolate[0]["session_id"] == "successor"
    service._maybe_rotate_agent_session.assert_not_called()


@pytest.mark.parametrize("stream", [True, False], ids=["stream", "nonstream"])
def test_post_admission_agent_setup_failure_closes_turn(monkeypatch, stream):
    service = _service(Mock(side_effect=AssertionError("bridge execution started")))
    service._maybe_rotate_agent_session.side_effect = RuntimeError("rotation failed")
    monkeypatch.setattr("llm_bawt.service.dependencies.get_service", lambda: service)
    monkeypatch.setattr(
        "llm_bawt.service.chat_streaming.prepare_stream_request_attachments",
        AsyncMock(return_value=("/new", [], [], None)),
    )
    request = ChatCompletionRequest(
        model="test-model", messages=[ChatMessage(role="user", content="/new")],
        bot_id="caid", user="nick", stream=stream,
    )

    async def run():
        if stream:
            async for _ in service.chat_completion_stream(_claimed(request)):
                pass
        else:
            await service.chat_completion(_claimed(request))

    with pytest.raises(RuntimeError, match="rotation failed"):
        asyncio.run(run())
    service._persist_turn_log.assert_called_once()
    service._update_turn_log.assert_called_once()
    assert service._update_turn_log.call_args.kwargs["status"] == "error"
    assert "rotation failed" in service._update_turn_log.call_args.kwargs["error_text"]


def test_nonstream_route_answers_a_queued_ack_instead_of_202(monkeypatch):
    """TASK-1144: no header brings back the 202 receipt shortcut."""
    from starlette.requests import Request
    from llm_bawt.service import chat_response_adapter as adapter
    from llm_bawt.service.routes import chat as chat_routes

    receipt = {
        "object": "chat.completion.queued", "delivery_id": "delivery-test",
        "user_message_id": "00000000-0000-4000-8000-000000000001",
        "turn_id": "turn-delivery-test", "assistant_message_id": "assistant-test",
        "status": "QUEUED",
    }
    service = SimpleNamespace(chat_completion=AsyncMock(return_value=receipt))
    monkeypatch.setattr(chat_routes, "get_service", lambda: service)
    observer = AsyncMock(return_value={"status": "DELIVERED", "response_text": "the answer"})
    monkeypatch.setattr(adapter, "observe_receipt", observer)
    http_request = Request({"type": "http", "headers": [], "method": "POST", "path": "/v1/chat/completions"})
    response = asyncio.run(chat_routes.chat_completions(_request(), http_request))
    assert response.object == "chat.completion"
    assert response.choices[0].message.content == "the answer"
    assert observer.await_args.args[2] == receipt  # the exact admitted turn, not a resubmit
    service.chat_completion.assert_awaited_once()


def test_agent_client_threads_request_local_identity_into_backend_config() -> None:
    from llm_bawt.clients.agent_backend_client import AgentBackendClient

    captured: dict = {}

    class Backend:
        async def chat_full(self, prompt, config, **_kwargs):
            captured.update(config)
            return SimpleNamespace(text="ok", tool_calls=[], model="m", usage={})

    client = AgentBackendClient.__new__(AgentBackendClient)
    client._bot_config = {"bot_id": "caid"}
    client._backend = Backend()
    marker = object()
    asyncio.run(client._chat_full(
        "hi", request_local={"turn_execution": marker, "task_turn_capability": "cap"},
    ))
    assert captured["turn_execution"] is marker
    assert captured["task_turn_capability"] == "cap"
    assert "turn_execution" not in client._bot_config  # shared config untouched


def _system_prompt_capture_client(captured: dict):
    from llm_bawt.clients.agent_backend_client import AgentBackendClient

    class Backend:
        def system_prompt_delivery(self, is_boundary):
            return "boundary" if is_boundary else "cached"

        async def chat_full(self, prompt, config, **_kwargs):
            captured.update(config, prompt=prompt)
            return SimpleNamespace(text="ok", tool_calls=[], model="m", usage={})

    client = AgentBackendClient.__new__(AgentBackendClient)
    client._bot_config = {"bot_id": "caid"}
    client._backend = Backend()
    return client


def test_agent_nonstream_query_forwards_the_turn_system_prompt() -> None:
    """TASK-1321: the non-stream path sends the system prompt like stream_raw."""
    from llm_bawt.models.message import Message

    captured: dict = {}
    client = _system_prompt_capture_client(captured)
    text = client.query([Message(role="system", content="You are Caid."),
                         Message(role="user", content="hi")])
    assert text == "ok"
    assert captured["prompt"] == "hi"
    assert captured["system_prompt"] == "You are Caid."
    assert "system_prompt_pinned" not in captured
    assert "system_prompt" not in client._bot_config  # shared config untouched


def test_agent_nonstream_query_honors_the_session_pin() -> None:
    """TASK-1321: a pinned prompt is marked so the backend sends the bytes verbatim."""
    from llm_bawt.models.message import Message

    captured: dict = {}
    client = _system_prompt_capture_client(captured)
    client.query([Message(role="system", content="pinned bytes"), Message(role="user", content="hi")],
                 system_prompt_pin=SimpleNamespace(is_boundary=True))
    assert captured["system_prompt"] == "pinned bytes"
    assert captured["system_prompt_pinned"] is True
    assert captured["system_prompt_delivery"] == "boundary"


@pytest.mark.parametrize("model_type,recorded", [("openai", True), ("claude-code", False)])
def test_chat_bot_nonstream_tool_calls_are_recorded(_isolate, monkeypatch, model_type, recorded) -> None:
    """TASK-1775: stream=false chat-bot tool calls reach tool_call_records
    through ToolEventCoordinator; agent turns keep the bridge event path."""
    from llm_bawt.service.tool_event_coordinator import ToolEventCoordinator

    starts: list[dict] = []
    ends: list[dict] = []
    monkeypatch.setattr(ToolEventCoordinator, "start", lambda self, e: starts.append(e) or e)
    monkeypatch.setattr(ToolEventCoordinator, "end", lambda self, e: ends.append(e) or e)
    monkeypatch.setattr(
        "llm_bawt.service.tool_event_coordinator.ToolCallStore", lambda _engine: object()
    )
    details = [{"tool": "web_search", "parameters": {"query": "iceland"},
                "result": "Found 5 results", "iteration": 1}]
    service = _service(Mock(return_value=("done", "ctx", details)), model_type=model_type)
    asyncio.run(service.chat_completion(_claimed(_request())))

    if not recorded:
        assert starts == [] and ends == []
        return
    turn_id = service._finalize_turn.call_args.kwargs["turn_id"]
    assert [(e["event"], e["tool_name"], e["turn_id"]) for e in starts + ends] == [
        ("tool_start", "web_search", turn_id), ("tool_end", "web_search", turn_id),
    ]
    assert starts[0]["arguments"] == {"query": "iceland"}
    assert starts[0]["call_id"] == ends[0]["call_id"]
    assert ends[0]["result"] == "Found 5 results"


def _protocol_admission(service, monkeypatch):
    """Inject the internal opt-in result, without enabling runtime admissions."""
    from llm_bawt.service.turn_terminal_follower import TerminalFollower
    from llm_bawt.service.turn_terminal_protocol import FollowerLease

    protocol = Mock()
    protocol.renew.side_effect = lambda lease: lease
    protocol.commit.return_value = SimpleNamespace(snapshot={
        "response_text": "answer", "completion": {"status": "completed", "end_reason": "stop"},
    }, won=True)
    service._turn_log_store.get_turn = Mock(return_value=SimpleNamespace(
        id="turn-delivery-test", terminal_protocol_state="open", terminal_snapshot_json=None,
        status="streaming", response_text="",
    ))
    monkeypatch.setattr(
        "llm_bawt.service.turn_terminal_finalize.project_execution_terminal_from_worker", Mock(),
    )
    service._persist_turn_log.return_value = FollowerLease(
        "turn-delivery-test", "insert-token", datetime.now(timezone.utc),
    )
    monkeypatch.setattr(
        "llm_bawt.service.turn_terminal_follower.TurnTerminalProtocol", lambda store: protocol,
    )
    original_start = TerminalFollower.start

    def start(follower):
        follower._stop.wait = Mock(return_value=True)
        return original_start(follower)

    monkeypatch.setattr(TerminalFollower, "start", start)
    llm = service._get_llm_bawt.return_value
    llm.history_manager = SimpleNamespace(_db_backend=SimpleNamespace(
        bot_id="caid", user_id="nick",
        get_or_create_active_session=Mock(return_value="canonical-native"),
    ))
    return protocol, llm


def test_protocol_nonstream_pins_post_reset_session_before_user_history(monkeypatch):
    from llm_bawt.service.session_lifecycle.execution import NewSessionExecution, PreparedNewSession

    order = []
    service = _service(Mock(return_value=("answer", "", [])))
    protocol, llm = _protocol_admission(service, monkeypatch)

    def prepare(*args, **kwargs):
        execution = turn_executions.get("turn-delivery-test")
        assert execution is not None
        protocol.renew.assert_called_once()
        order.append("reset")
        return PreparedNewSession(None, {"thread_session_id": "final-successor"}, None)

    def bind(lease, *, session_id):
        assert lease.token == "insert-token"
        assert session_id == "final-successor"
        order.append("bind")

    def messages(*args, **kwargs):
        assert llm._session_id_override is None
        assert kwargs["history_write_session_id"] == "final-successor"
        order.append("user-history")
        return []

    monkeypatch.setattr(NewSessionExecution, "prepare_agent", prepare)
    protocol.bind_history_session.side_effect = bind
    llm.prepare_messages_for_query.side_effect = messages
    response = asyncio.run(service.chat_completion(_claimed(_request())))
    assert response.choices[0].message.content == "answer"
    assert order == ["reset", "bind", "user-history"]
    llm.history_manager._db_backend.get_or_create_active_session.assert_not_called()
    assert llm.execute_llm_query.call_args.kwargs["turn_execution"].turn_id == "turn-delivery-test"
    assert turn_executions.get("turn-delivery-test") is None
    service._end_generation.assert_called_once()


def test_protocol_nonstream_returns_only_the_terminal_winner(monkeypatch):
    service = _service(Mock(return_value=("losing local answer", "", [])))
    protocol, llm = _protocol_admission(service, monkeypatch)
    protocol.commit.return_value = SimpleNamespace(
        won=False, snapshot={
            "response_text": "Stop-owned partial",
            "completion": {"status": "cancelled", "end_reason": "aborted"},
        },
    )
    response = asyncio.run(service.chat_completion(_claimed(_request())))
    assert response.choices[0].message.content == "Stop-owned partial"
    assert response.choices[0].finish_reason == "cancelled"
    service._finalize_turn.assert_not_called()
    events = [call.args[2] for call in service._redis_subscriber.publish_tool_event.await_args_list]
    assert not any(event["_type"] in {"text_delta", "turn_complete"} for event in events)
    assert turn_executions.get("turn-delivery-test") is None


def test_protocol_nonstream_inference_error_commits_before_raising(monkeypatch):
    service = _service(Mock(side_effect=RuntimeError("provider failed")))
    protocol, _ = _protocol_admission(service, monkeypatch)
    with pytest.raises(RuntimeError, match="provider failed"):
        asyncio.run(service.chat_completion(_claimed(_request())))
    candidate = protocol.commit.call_args.kwargs
    assert candidate["status"] == "error" and candidate["end_reason"] == "upstream_error"
    assert candidate["error_text"] == "provider failed"
    service._finalize_turn.assert_not_called()
    service._update_turn_log.assert_not_called()
    events = [call.args[2] for call in service._redis_subscriber.publish_tool_event.await_args_list]
    assert not any(event["_type"] == "turn_complete" for event in events)


def test_protocol_nonstream_voice_strips_emotes_only_from_returned_winner(monkeypatch):
    service = _service(Mock(return_value=("*smiles* Hello", "", [])))
    protocol, _ = _protocol_admission(service, monkeypatch)
    frozen = "*smiles* Hello"
    protocol.commit.return_value = SimpleNamespace(
        won=True, snapshot={
            "response_text": frozen, "completion": {"status": "completed", "end_reason": "stop"},
        },
    )
    monkeypatch.setattr(
        "llm_bawt.service.chat_nonstream.get_bot",
        lambda _bot: SimpleNamespace(voice_optimized=True),
    )
    response = asyncio.run(service.chat_completion(_claimed(_request())))
    assert response.choices[0].message.content == "Hello"
    assert protocol.commit.return_value.snapshot["response_text"] == frozen


def test_nonstream_projection_waits_on_approval_publish_lock(monkeypatch):
    from llm_bawt.service import turn_terminal_finalize

    service = _service(Mock(return_value=("answer", "", [])))
    _protocol_admission(service, monkeypatch)
    asyncio.run(service.chat_completion(_claimed(_request())))
    call = turn_terminal_finalize.project_execution_terminal_from_worker.call_args
    assert isinstance(call.kwargs["publish_lock"], asyncio.Lock)


def test_failed_nonstream_commit_never_retries_as_an_empty_error(monkeypatch):
    service = _service(Mock(return_value=("provider reply", "", [])))
    protocol, _ = _protocol_admission(service, monkeypatch)
    protocol.commit.side_effect = ConnectionError("commit unavailable")
    with pytest.raises(ConnectionError, match="commit unavailable"):
        asyncio.run(service.chat_completion(_claimed(_request())))
    protocol.commit.assert_called_once()
    candidate = protocol.commit.call_args.kwargs
    assert candidate["response_text"] == "provider reply"
    assert candidate["status"] == "ok"


def test_protocol_nonstream_freezes_actual_bridge_metadata(monkeypatch):
    def query(*args, **kwargs):
        callback = kwargs["bridge_event_callback"]
        callback({"event": "attachments", "attachments": [{"asset_id": "image"}]})
        callback({"event": "reasoning", "text": "thinking"})
        callback({"event": "token_usage", "token_usage": {"input_tokens": 11}})
        callback({
            "event": "approval_required", "tool_use_id": "approval", "tool_name": "Bash",
            "arguments": {}, "provider": "claude-code",
        })
        return "answer", "", []

    service = _service(query)
    service._tool_approval_policy_store = Mock()
    protocol, _ = _protocol_admission(service, monkeypatch)
    asyncio.run(service.chat_completion(_claimed(_request())))
    candidate = protocol.commit.call_args.kwargs
    assert candidate["end_reason"] == "approval"
    assert candidate["attachments"] == [{"asset_id": "image"}]
    assert candidate["reasoning"] == "thinking"
    assert candidate["token_usage"]["input_tokens"] == 11
    assert candidate["completion_fields"]["approval_id"] == "approval"
    service._tool_approval_policy_store.record_request.assert_called_once()


@pytest.mark.parametrize("stream", [True, False])
def test_protocol_setup_failure_closes_early_owner_before_any_user_write(monkeypatch, stream):
    from llm_bawt.service.session_lifecycle.execution import NewSessionExecution

    service = _service(Mock(side_effect=AssertionError("no inference after setup failure")))
    protocol, llm = _protocol_admission(service, monkeypatch)

    def fail(*args, **kwargs):
        assert turn_executions.get("turn-delivery-test") is not None
        protocol.renew.assert_called_once()
        raise ValueError("reset failed")

    monkeypatch.setattr(NewSessionExecution, "prepare_agent", fail)

    async def run():
        request = _claimed(_request())
        if stream:
            async for _ in service.chat_completion_stream(request):
                pass
        else:
            await service.chat_completion(request)

    with pytest.raises(ValueError, match="reset failed"):
        asyncio.run(run())
    assert turn_executions.get("turn-delivery-test") is None
    llm.prepare_messages_for_query.assert_not_called()
    llm.execute_llm_query.assert_not_called()
    protocol.bind_history_session.assert_not_called()
    protocol.commit.assert_called_once_with(
        service._persist_turn_log.return_value, status="error",
        end_reason="setup_error", error_text="reset failed",
    )
    service._end_generation.assert_called_once()


def test_nonstream_setup_failure_projection_uses_same_publish_lock(monkeypatch):
    from llm_bawt.service import turn_terminal_finalize
    from llm_bawt.service.session_lifecycle.execution import NewSessionExecution

    service = _service(Mock(side_effect=AssertionError("no inference")))
    _protocol_admission(service, monkeypatch)
    monkeypatch.setattr(
        NewSessionExecution, "prepare_agent", Mock(side_effect=RuntimeError("setup failed")),
    )
    with pytest.raises(RuntimeError, match="setup failed"):
        asyncio.run(service.chat_completion(_claimed(_request())))
    call = turn_terminal_finalize.project_execution_terminal_from_worker.call_args
    assert isinstance(call.kwargs["publish_lock"], asyncio.Lock)


def test_protocol_stream_worker_preserves_canonical_native_pin(monkeypatch):
    from llm_bawt.service.turn_execution import TurnExecutionDetached
    from llm_bawt.working_memory.turn import WorkingMemoryTurn

    service = _service(Mock(side_effect=AssertionError("no provider needed")), model_type="openai")
    protocol, llm = _protocol_admission(service, monkeypatch)
    service._start_generation = AsyncMock(side_effect=lambda *args, **kwargs: (
        kwargs["cancel_event"], kwargs["done_event"],
    ))
    monkeypatch.setattr(
        WorkingMemoryTurn, "begin",
        lambda ctx: SimpleNamespace(extend_suffix=lambda suffix: suffix),
    )
    pins = []

    def messages(*args, **kwargs):
        protocol.bind_history_session.assert_called_once()
        assert llm._session_id_override is None
        pins.append(kwargs["history_write_session_id"])
        # Stop before inference, using the real worker's observation-only
        # detach path and real finalizer cleanup.
        execution = turn_executions.get("turn-delivery-test")
        execution.detach()
        raise TurnExecutionDetached()

    llm.prepare_messages_for_query.side_effect = messages

    async def run():
        return [event async for event in service.chat_completion_stream(_claimed(_request()))]

    asyncio.run(run())
    assert pins == ["canonical-native"]
    protocol.bind_history_session.assert_called_once_with(
        service._persist_turn_log.return_value, session_id="canonical-native",
    )
    llm.execute_llm_query.assert_not_called()
    assert turn_executions.get("turn-delivery-test") is None
