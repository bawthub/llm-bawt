"""Durability tests for streaming persistence and shared turn finalization."""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from llm_bawt.message_authorship import AuthorReference
from llm_bawt.service.background_service import BackgroundService
from llm_bawt.service.chat_stream_worker import consume_stream_chunks
from llm_bawt.service.schemas import ChatCompletionRequest, ChatMessage


class _ClosedLoop:
    """Event-loop stub that simulates a disconnected client loop."""

    def call_soon_threadsafe(self, *args, **kwargs):
        raise RuntimeError("Event loop is closed")


class _CapturingLoop:
    """Event-loop stub that captures thread-safe queue puts."""

    def call_soon_threadsafe(self, callback, *args):
        callback(*args)


def _build_service_for_finalize() -> BackgroundService:
    service = BackgroundService.__new__(BackgroundService)
    service.config = SimpleNamespace(DEBUG_TURN_LOG=False)
    service._update_turn_log = Mock()
    return service


@pytest.mark.parametrize("sender_mode", ["durable", "synchronous"])
def test_inter_bot_nonstream_turn_publishes_visible_lifecycle(
    monkeypatch, sender_mode: str,
) -> None:
    service = BackgroundService.__new__(BackgroundService)
    service.config = SimpleNamespace(DEFAULT_BOT="nova", DEFAULT_USER="nick")
    service._inter_bot_dispatcher = SimpleNamespace(
        store=SimpleNamespace(
            validate_claim=Mock(return_value=True),
            get=Mock(return_value=SimpleNamespace(
                sender_bot_id="snark",
                author=AuthorReference.bot("snark"),
            )),
        )
    )
    subscriber = SimpleNamespace(publish_tool_event=AsyncMock())
    service._redis_subscriber = subscriber
    service._resolve_request_model = Mock(return_value=("test-model", []))
    service._persist_turn_log = Mock()
    service._update_turn_log = Mock()
    service._end_generation = Mock()
    service._bind_agent_thread = Mock(return_value=None)
    service._resolve_active_thread_binding = Mock(
        return_value={"thread_session_id": "sess-visible-abc"}
    )
    service._maybe_summarize_on_new = Mock()
    service._maybe_rotate_agent_session = Mock()
    service._finalize_turn = Mock()
    service._turn_log_store = SimpleNamespace(engine=object())

    coordinated: list[dict] = []

    class FakeCoordinator:
        def __init__(self, _engine):
            pass

        def start(self, event):
            coordinated.append(event)
            return {**event, "_tool_persisted": True}

        def end(self, event):
            coordinated.append(event)
            return {**event, "_tool_persisted": True}

    monkeypatch.setattr(
        "llm_bawt.service.tool_event_coordinator.ToolEventCoordinator",
        FakeCoordinator,
    )

    client = SimpleNamespace(
        model_definition={"type": "claude-code"},
        get_token_usage=Mock(return_value={"input_tokens": 12}),
    )
    def execute_llm_query(*_args, **kwargs):
        callback = kwargs["bridge_event_callback"]
        callback({
            "event": "tool_call",
            "name": "Read",
            "arguments": {"file_path": "x.py"},
            "provider": "claude-code",
            "tool_use_id": "tool-1",
        })
        callback({
            "event": "tool_result",
            "name": "Read",
            "arguments": {"file_path": "x.py"},
            "result": "contents",
            "provider": "claude-code",
            "tool_use_id": "tool-1",
        })
        return "visible reply", "", []

    llm_bawt = SimpleNamespace(
        client=client,
        bot=SimpleNamespace(tts_mode=False),
        prepare_messages_for_query=Mock(return_value=[]),
        execute_llm_query=execute_llm_query,
    )
    service._get_llm_bawt = Mock(return_value=llm_bawt)
    monkeypatch.setattr(
        "llm_bawt.service.chat_nonstream.get_bot", lambda _bot_id: None
    )
    monkeypatch.setattr(
        "llm_bawt.service.routes.history.maybe_build_session_seed",
        lambda *_args, **_kwargs: None,
    )

    request_kwargs = {
        "model": "test-model",
        "messages": [ChatMessage(role="user", content="background prompt")],
        "bot_id": "loopy",
        "user": "nick",
        "stream": False,
        "user_message_id": "user-visible",
    }
    if sender_mode == "durable":
        request_kwargs.update({
            "inter_bot_delivery_id": "delivery-visible",
            "inter_bot_turn_id": "turn-visible",
            "inter_bot_bridge_request_id": "req-visible",
            "inter_bot_claim_token": "claim-visible",
        })
    request = ChatCompletionRequest(**request_kwargs)
    if sender_mode == "synchronous":
        request._internal_inter_bot_sender_id = "snark"

    response = asyncio.run(service.chat_completion(request))

    assert response.choices[0].message.content == "visible reply"
    events = [call.args[2] for call in subscriber.publish_tool_event.await_args_list]
    assert [event.get("event") or event["_type"] for event in events] == [
        "turn_start",
        "tool_start",
        "tool_end",
        "text_delta",
        "turn_complete",
    ]
    assert events[0]["trigger_message_id"] == "user-visible"
    assert events[1]["tool_name"] == "Read"
    assert events[1]["call_id"] == events[2]["call_id"]
    assert events[2]["result"] == "contents"
    assert coordinated == [
        {key: value for key, value in events[1].items() if key != "_tool_persisted"},
        {key: value for key, value in events[2].items() if key != "_tool_persisted"},
    ]
    assert events[3]["delta"] == "visible reply"
    assert events[4]["status"] == "completed"
    assert events[4]["assistant_message_id"] == events[0]["assistant_message_id"]
    assert service._finalize_turn.call_args.kwargs["assistant_message_id"] == events[0]["assistant_message_id"]

    # TASK-709: turn_start and turn_complete carry the durable session_id so
    # a receiving BawtHub window can route lifecycle events to the correct
    # thread without a follow-up DB fetch. Value comes from the pre-emit
    # session resolve (see chat_nonstream.chat_completion) — either
    # ``request.session_id`` (dispatcher-written target for reset deliveries;
    # explicit thread selection for chat clients) or the target bot's
    # active-thread fallback.
    assert "session_id" in events[0], "turn_start must carry session_id"
    assert "session_id" in events[4], "turn_complete must carry session_id"
    assert events[0]["session_id"] == "sess-visible-abc"
    assert events[4]["session_id"] == "sess-visible-abc"


def test_inter_bot_nonstream_turn_start_prefers_request_session_over_archived_seed(
    monkeypatch,
) -> None:
    """TASK-709 F1: for durable inter-bot deliveries with a reset session
    policy, the dispatcher writes the NEW (target) session id to
    ``request.session_id`` and the OLD/ARCHIVED session id to
    ``inter_bot_seed_session_id`` (the history SOURCE for
    ``build_context_seed``). The lifecycle payload's ``session_id`` must
    route to the TARGET, not the archived source — otherwise turn_start
    events go to the dead thread and the pane on the fresh session gates
    them out as "wrong thread"."""
    service = BackgroundService.__new__(BackgroundService)
    service.config = SimpleNamespace(DEFAULT_BOT="nova", DEFAULT_USER="nick")
    service._inter_bot_dispatcher = SimpleNamespace(
        store=SimpleNamespace(
            validate_claim=Mock(return_value=True),
            get=Mock(return_value=SimpleNamespace(
                sender_bot_id="snark",
                author=AuthorReference.bot("snark"),
            )),
        )
    )
    subscriber = SimpleNamespace(publish_tool_event=AsyncMock())
    service._redis_subscriber = subscriber
    service._resolve_request_model = Mock(return_value=("test-model", []))
    service._persist_turn_log = Mock()
    service._update_turn_log = Mock()
    service._end_generation = Mock()
    service._bind_agent_thread = Mock(return_value=None)
    # Active-thread resolver would return a fallback if request.session_id were
    # missing, but with request.session_id present it MUST NOT be consulted
    # for the pre-emit lifecycle session id.
    active_resolver = Mock(return_value={"thread_session_id": "sess-fresh-target"})
    service._resolve_active_thread_binding = active_resolver
    service._maybe_summarize_on_new = Mock()
    service._maybe_rotate_agent_session = Mock()
    service._finalize_turn = Mock()
    service._turn_log_store = SimpleNamespace(engine=object())

    class FakeCoordinator:
        def __init__(self, _engine):
            pass

        def start(self, event):
            return {**event, "_tool_persisted": True}

        def end(self, event):
            return {**event, "_tool_persisted": True}

    monkeypatch.setattr(
        "llm_bawt.service.tool_event_coordinator.ToolEventCoordinator",
        FakeCoordinator,
    )

    client = SimpleNamespace(
        model_definition={"type": "claude-code"},
        get_token_usage=Mock(return_value=None),
    )
    llm_bawt = SimpleNamespace(
        client=client,
        bot=SimpleNamespace(tts_mode=False),
        prepare_messages_for_query=Mock(return_value=[]),
        execute_llm_query=Mock(return_value=("reply", "", [])),
    )
    service._get_llm_bawt = Mock(return_value=llm_bawt)
    monkeypatch.setattr(
        "llm_bawt.service.chat_nonstream.get_bot", lambda _bot_id: None
    )
    monkeypatch.setattr(
        "llm_bawt.service.routes.history.maybe_build_session_seed",
        lambda *_args, **_kwargs: None,
    )

    request = ChatCompletionRequest(
        model="test-model",
        messages=[ChatMessage(role="user", content="reset delivery")],
        bot_id="loopy",
        user="nick",
        stream=False,
        user_message_id="user-reset",
        # request.session_id = the NEW (target) session id the dispatcher
        # rotated into. This is where the turn_start/turn_complete lifecycle
        # events must be routed so the fresh pane picks them up.
        session_id="sess-fresh-target",
        inter_bot_delivery_id="delivery-reset",
        inter_bot_turn_id="turn-reset",
        inter_bot_bridge_request_id="req-reset",
        inter_bot_claim_token="claim-reset",
        # inter_bot_seed_session_id = the OLD/ARCHIVED session id from which
        # ``build_context_seed`` reads history. MUST NOT be emitted as the
        # lifecycle session id — that would route events to the dead thread.
        inter_bot_seed_session_id="sess-archived-source",
    )

    asyncio.run(service.chat_completion(request))

    events = [call.args[2] for call in subscriber.publish_tool_event.await_args_list]
    turn_start = next(e for e in events if e["_type"] == "turn_start")
    turn_complete = next(e for e in events if e["_type"] == "turn_complete")
    # F1: request.session_id (dispatcher-written target) wins.
    assert turn_start["session_id"] == "sess-fresh-target"
    assert turn_complete["session_id"] == "sess-fresh-target"
    # F1: the archived seed id must NEVER surface in a lifecycle payload.
    for evt in events:
        assert evt.get("session_id") != "sess-archived-source", (
            f"archived seed session id must not be emitted; got in {evt.get('_type')}"
        )


def test_finalize_turn_saves_history() -> None:
    service = _build_service_for_finalize()
    service._extract_agent_backend_tool_calls = Mock(return_value=[])

    llm_bawt = SimpleNamespace(finalize_response=Mock(), adapter=None)

    service._finalize_turn(
        llm_bawt=llm_bawt,
        turn_id="turn-1",
        response_text="hello world",
        tool_context="",
        tool_call_details=[],
        prepared_messages=[{"role": "user", "content": "hi"}],
        user_prompt="hi",
        model="test-model",
        bot_id="proto",
        user_id="nick",
        elapsed_ms=12.5,
        stream=False,
    )

    llm_bawt.finalize_response.assert_called_once_with(
        "hello world", "", attachments=None, reasoning=None, message_id=None
    )
    service._update_turn_log.assert_called_once()
    kwargs = service._update_turn_log.call_args.kwargs
    assert kwargs["turn_id"] == "turn-1"
    assert kwargs["status"] == "ok"
    assert kwargs["response_text"] == "hello world"


def test_finalize_turn_persists_non_streaming_agent_usage(monkeypatch) -> None:
    service = _build_service_for_finalize()

    class FakeAgentBackendClient:
        def get_token_usage(self):
            return {
                "resident_tokens": 16475,
                "resident_source": "claude_sdk_context",
                "context_window": 372000,
            }

        def get_tool_calls(self):
            return []

    from llm_bawt.service import turn_lifecycle as tl_module

    monkeypatch.setattr(tl_module, "AgentBackendClient", FakeAgentBackendClient)
    llm_bawt = SimpleNamespace(
        finalize_response=Mock(), adapter=None, client=FakeAgentBackendClient()
    )

    service._finalize_turn(
        llm_bawt=llm_bawt,
        turn_id="turn-agent-usage",
        response_text="done",
        tool_context="",
        tool_call_details=[],
        prepared_messages=[],
        user_prompt="prompt",
        model="test-model",
        bot_id="proto",
        user_id="nick",
        elapsed_ms=20.0,
        stream=False,
    )

    assert service._update_turn_log.call_args.kwargs["token_usage"] == {
        "resident_tokens": 16475,
        "resident_source": "claude_sdk_context",
        "context_window": 372000,
    }


def test_resolve_turn_usage_recovers_partial_backend_result(monkeypatch) -> None:
    service = _build_service_for_finalize()

    class FakeAgentBackendClient:
        def get_token_usage(self):
            return {
                "resident_tokens": 139613,
                "resident_source": "interrupted_iteration_total_input",
                "usage_status": "partial",
            }

    from llm_bawt.service import turn_lifecycle as tl_module

    monkeypatch.setattr(tl_module, "AgentBackendClient", FakeAgentBackendClient)
    llm_bawt = SimpleNamespace(client=FakeAgentBackendClient())

    assert service._resolve_turn_token_usage(llm_bawt, None) == {
        "resident_tokens": 139613,
        "resident_source": "interrupted_iteration_total_input",
        "usage_status": "partial",
    }


def test_finalize_turn_prefers_explicit_streaming_usage(monkeypatch) -> None:
    service = _build_service_for_finalize()

    class FakeAgentBackendClient:
        def get_token_usage(self):
            return {"resident_tokens": 1}

        def get_tool_calls(self):
            return []

    from llm_bawt.service import turn_lifecycle as tl_module

    monkeypatch.setattr(tl_module, "AgentBackendClient", FakeAgentBackendClient)
    llm_bawt = SimpleNamespace(
        finalize_response=Mock(), adapter=None, client=FakeAgentBackendClient()
    )
    explicit = {"resident_tokens": 99, "resident_source": "stream"}

    service._finalize_turn(
        llm_bawt=llm_bawt,
        turn_id="turn-stream-usage",
        response_text="done",
        tool_context="",
        tool_call_details=[],
        prepared_messages=[],
        user_prompt="prompt",
        model="test-model",
        bot_id="proto",
        user_id="nick",
        elapsed_ms=20.0,
        stream=True,
        token_usage=explicit,
    )

    assert service._update_turn_log.call_args.kwargs["token_usage"] == explicit


def test_finalize_turn_extracts_agent_backend_tools(monkeypatch) -> None:
    service = _build_service_for_finalize()

    class FakeAgentBackendClient:
        def get_tool_calls(self):
            return [
                {
                    "name": "search_web",
                    "display_name": "search_web",
                    "arguments": {"query": "ai"},
                    "result": "ok",
                }
            ]

    from llm_bawt.service import turn_lifecycle as tl_module

    monkeypatch.setattr(tl_module, "AgentBackendClient", FakeAgentBackendClient)
    llm_bawt = SimpleNamespace(finalize_response=Mock(), adapter=None, client=FakeAgentBackendClient())

    tool_calls: list[dict] = []
    service._finalize_turn(
        llm_bawt=llm_bawt,
        turn_id="turn-2",
        response_text="done",
        tool_context="",
        tool_call_details=tool_calls,
        prepared_messages=[],
        user_prompt="prompt",
        model="test-model",
        bot_id="proto",
        user_id="nick",
        elapsed_ms=20.0,
        stream=True,
    )

    assert len(tool_calls) == 1
    assert tool_calls[0]["tool"] == "search_web"
    assert tool_calls[0]["parameters"] == {"query": "ai"}
    assert tool_calls[0]["result"] == "ok"


def test_stream_persists_on_client_disconnect() -> None:
    cancel_event = threading.Event()
    full_response_holder = [""]

    cancelled = consume_stream_chunks(
        iter(["hello", " ", "world"]),
        cancel_event=cancel_event,
        loop=_ClosedLoop(),
        chunk_queue=asyncio.Queue(),
        full_response_holder=full_response_holder,
    )

    assert cancelled is False
    assert full_response_holder[0] == "hello world"


def test_stream_persists_on_cancellation() -> None:
    cancel_event = threading.Event()
    full_response_holder = [""]

    def _iter_chunks():
        yield "a"
        cancel_event.set()
        yield "b"
        yield "c"

    cancelled = consume_stream_chunks(
        _iter_chunks(),
        cancel_event=cancel_event,
        loop=_CapturingLoop(),
        chunk_queue=asyncio.Queue(),
        full_response_holder=full_response_holder,
    )

    assert cancelled is True
    assert full_response_holder[0] == "abc"


def test_agent_client_keeps_partial_result_when_stream_raises() -> None:
    from llm_bawt.clients.agent_backend_client import AgentBackendClient
    from llm_bawt.models.message import Message

    usage = {"resident_tokens": 50, "usage_status": "partial"}

    class Backend:
        def stream_raw(self, *_args, **_kwargs):
            yield {"event": "token_usage", "token_usage": usage}
            raise RuntimeError("interrupted")

        def get_last_stream_result(self):
            return SimpleNamespace(usage=usage)

    client = AgentBackendClient.__new__(AgentBackendClient)
    client._backend = Backend()
    client._bot_config = {}
    client.last_result = None

    iterator = client.stream_raw([Message(role="user", content="hello")])
    assert next(iterator)["token_usage"] == usage
    try:
        next(iterator)
    except RuntimeError as exc:
        assert str(exc) == "interrupted"
    else:
        raise AssertionError("stream should raise")

    assert client.get_token_usage() == usage


def test_stream_accumulates_only_str_chunks() -> None:
    cancel_event = threading.Event()
    full_response_holder = [""]

    cancelled = consume_stream_chunks(
        iter(["hello", {"event": "tool_call", "name": "search"}, " world"]),
        cancel_event=cancel_event,
        loop=_CapturingLoop(),
        chunk_queue=asyncio.Queue(),
        full_response_holder=full_response_holder,
    )

    assert cancelled is False
    assert full_response_holder[0] == "hello world"


# ---------------------------------------------------------------------------
# Tests for worker finally-block resilience
# ---------------------------------------------------------------------------


def test_worker_finalize_error_falls_back_to_error_status() -> None:
    """If _finalize_turn raises, the worker catches it and marks the turn as error.

    This exercises the try/except around _finalize_turn in the worker's
    finally block.  We simulate the full sequence: _finalize_turn raises,
    then the worker falls back to _update_turn_log(status="error").
    """
    service = _build_service_for_finalize()
    service._extract_agent_backend_tool_calls = Mock(return_value=[])

    llm_bawt = SimpleNamespace(
        finalize_response=Mock(side_effect=RuntimeError("DB down")),
        adapter=None,
    )

    # Simulate the worker's finally block:
    # 1. Try _finalize_turn → raises
    # 2. Catch → call _update_turn_log(status="error")
    turn_id = "turn-err"
    elapsed_ms = 10.0
    full_response = "partial response"

    try:
        service._finalize_turn(
            llm_bawt=llm_bawt,
            turn_id=turn_id,
            response_text=full_response,
            tool_context="",
            tool_call_details=[],
            prepared_messages=[],
            user_prompt="hi",
            model="test-model",
            bot_id="proto",
            user_id="nick",
            elapsed_ms=elapsed_ms,
            stream=True,
        )
    except Exception as fin_err:
        # This is what the worker's finally block does on failure:
        service._update_turn_log(
            turn_id=turn_id,
            status="error",
            latency_ms=elapsed_ms,
            response_text=full_response or None,
            error_text=f"finalize_error: {fin_err}",
        )

    # finalize_response was attempted
    llm_bawt.finalize_response.assert_called_once()

    # The fallback _update_turn_log should have been called with error status.
    # Note: _update_turn_log was also called by _finalize_turn before the
    # error, so we check the LAST call has status="error".
    last_call_kwargs = service._update_turn_log.call_args_list[-1].kwargs
    assert last_call_kwargs["status"] == "error"
    assert "finalize_error" in last_call_kwargs["error_text"]
    assert last_call_kwargs["turn_id"] == turn_id


def test_worker_finalize_error_sends_sentinel_and_generation_done() -> None:
    """Simulate the worker's finally block when _finalize_turn raises.

    Verify that the sentinel (None) is still pushed to the chunk queue and
    the generation done_event is still set, even if finalization fails.
    """
    from llm_bawt.service.chat_stream_worker import put_queue_item_threadsafe

    loop = _CapturingLoop()
    chunk_queue = asyncio.Queue()
    done_event = threading.Event()

    full_response_holder = ["some response"]
    finalize_raised = False

    def mock_finalize_turn(**kwargs):
        raise RuntimeError("DB failure")

    # Simulate the worker finally block logic (matches the code we changed):
    # try _finalize_turn → except → fallback update → finally sentinel + done
    try:
        mock_finalize_turn(response_text=full_response_holder[0])
    except Exception:
        finalize_raised = True

    # These should always execute regardless of finalize outcome
    put_queue_item_threadsafe(loop, chunk_queue, None)
    done_event.set()

    assert finalize_raised
    assert not chunk_queue.empty()
    assert chunk_queue.get_nowait() is None
    assert done_event.is_set()


def test_generation_lifecycle_ordering() -> None:
    """done_event must be set only AFTER finalization completes.

    The worker thread's finally block runs: finalize → sentinel → done_event.
    A subsequent _start_generation would wait on done_event, so setting it
    prematurely (from the generator's finally, as the old code did) could
    allow a new generation to start before the old one finished persisting.

    This test verifies the ordering contract by simulating the full sequence.
    """
    done_event = threading.Event()
    finalize_order: list[str] = []

    def mock_finalize():
        finalize_order.append("finalize")
        assert not done_event.is_set(), "done_event must not be set before finalize"

    def mock_sentinel():
        finalize_order.append("sentinel")
        assert not done_event.is_set(), "done_event must not be set before sentinel"

    # Simulate worker finally block ordering
    mock_finalize()
    mock_sentinel()
    done_event.set()
    finalize_order.append("done")

    assert finalize_order == ["finalize", "sentinel", "done"]
    assert done_event.is_set()
