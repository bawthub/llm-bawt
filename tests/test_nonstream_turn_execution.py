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
    service._persist_turn_log = Mock()
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


def test_nonstream_route_returns_202_for_queued_ack(monkeypatch):
    from starlette.requests import Request
    from llm_bawt.service.routes import chat as chat_routes

    receipt = {
        "object": "chat.completion.queued", "delivery_id": "delivery-test",
        "user_message_id": "00000000-0000-4000-8000-000000000001",
        "turn_id": "turn-delivery-test", "assistant_message_id": "assistant-test",
        "status": "QUEUED",
    }
    monkeypatch.setattr(chat_routes, "get_service", lambda: SimpleNamespace(
        chat_completion=AsyncMock(return_value=receipt),
    ))
    http_request = Request({"type": "http", "headers": [], "method": "POST", "path": "/v1/chat/completions"})
    response = asyncio.run(chat_routes.chat_completions(_request(), http_request))
    assert response.status_code == 202
    assert response.media_type == "application/json"
    import json
    assert json.loads(response.body) == receipt


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
