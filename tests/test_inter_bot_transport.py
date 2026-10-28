"""Shared transport and dispatcher idempotency tests (TASK-710)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest

from agent_bridge.publisher import COMMANDS_STREAM
from agent_bridge.subscriber import RedisSubscriber
from llm_bawt.agent_context import (
    SessionPolicy,
    capabilities_for_backend,
    normalize_session_policy,
    preventive_session_policy,
    validate_session_policy,
)
from llm_bawt.service.inter_bot_dispatcher import InterBotDeliveryDispatcher
from llm_bawt.mcp_server.server import _get_profile_manager
from llm_bawt.model_catalog import AccessPath, ModelCatalog, ModelEndpoint, ModelIdentity
from llm_bawt.service.routes.inter_bot_deliveries import _target_context_window
from llm_bawt.utils.config import Config


def test_profile_manager_compatibility_alias_remains_importable():
    assert callable(_get_profile_manager)


def test_delivery_context_health_uses_pinned_endpoint_for_ambiguous_model():
    model = ModelIdentity(
        id=1,
        key="claude-opus-5",
        vendor="anthropic",
        display_name="Claude Opus 5",
        default_context_window=200_000,
    )
    oauth = AccessPath(
        id=1,
        key="anthropic-oauth",
        vendor="anthropic",
        protocol="anthropic-messages",
        base_url=None,
        auth_mechanism="oauth",
    )
    api = AccessPath(
        id=2,
        key="anthropic-api",
        vendor="anthropic",
        protocol="anthropic-messages",
        base_url=None,
        auth_mechanism="api-key",
    )
    config = Config()
    config.install_model_catalog(ModelCatalog([
        ModelEndpoint(1, model, oauth, "claude-opus-5"),
        ModelEndpoint(2, model, api, "claude-opus-5", context_window_override=1_000_000),
    ]))
    bot = SimpleNamespace(
        default_model="claude-opus-5",
        endpoint_id=2,
        harness="claude-code",
    )

    assert _target_context_window(config, bot) == 1_000_000



def _run(coro):
    return asyncio.run(coro)


def test_stable_bridge_request_id_publishes_one_command():
    subscriber = RedisSubscriber("redis://localhost:6379/0")
    fake = MagicMock()
    published: dict[str, str] = {}
    commands: list[dict] = []

    async def eval_script(
        _script, _numkeys, dedupe_key, stream, _maxlen, request_id, *flat
    ):
        if dedupe_key in published:
            return [0, published[dedupe_key]]
        fields = dict(zip(flat[::2], flat[1::2]))
        assert request_id == fields["request_id"]
        stream_id = f"1-{len(commands)}"
        published[dedupe_key] = stream_id
        commands.append({"stream": stream, **fields})
        return [1, stream_id]

    fake.eval = eval_script
    subscriber._pub_redis = fake

    async def send_twice():
        await subscriber.send_command(
            session_key="snark:nick",
            message="TASK-700 READY",
            request_id="req_delivery_stable",
            bot_id="snark",
        )
        await subscriber.send_command(
            session_key="snark:nick",
            message="TASK-700 READY",
            request_id="req_delivery_stable",
            bot_id="snark",
        )

    _run(send_twice())
    assert len(commands) == 1
    assert commands[0]["stream"] == COMMANDS_STREAM
    assert commands[0]["request_id"] == "req_delivery_stable"


def test_session_policy_normalization_and_capabilities():
    assert normalize_session_policy() == SessionPolicy.CONTINUE
    assert normalize_session_policy(
        reset_session_before_delivery=True, retain_history=True
    ) == SessionPolicy.RESET_RETAIN_HISTORY
    assert normalize_session_policy(
        reset_session_before_delivery=True, retain_history=False
    ) == SessionPolicy.RESET_WITHOUT_HISTORY
    with pytest.raises(ValueError, match="contradicts"):
        normalize_session_policy(
            "continue", reset_session_before_delivery=True, retain_history=False
        )
    with pytest.raises(ValueError, match="requires"):
        normalize_session_policy(retain_history=True)

    claude = capabilities_for_backend("claude-code")
    assert claude.inspect and claude.compact
    assert claude.reset_retain_history and claude.reset_without_history
    codex = capabilities_for_backend("codex")
    assert not codex.inspect and not codex.compact
    # TASK-1579: retained history is seeded into a fresh Codex thread.
    assert codex.reset_retain_history and codex.reset_without_history
    validate_session_policy("codex", SessionPolicy.RESET_RETAIN_HISTORY)
    with pytest.raises(ValueError, match="does not support"):
        validate_session_policy("local", SessionPolicy.RESET_WITHOUT_HISTORY)

    selected, reason = preventive_session_policy(
        requested=SessionPolicy.CONTINUE,
        health_state="critical",
        configured_critical_policy="reset_retain_history",
        backend="claude-code",
    )
    assert selected == SessionPolicy.RESET_RETAIN_HISTORY
    assert "critical" in reason
    selected, reason = preventive_session_policy(
        requested=SessionPolicy.CONTINUE,
        health_state="critical",
        configured_critical_policy="reset_retain_history",
        backend="codex",
    )
    # Codex seeds retained history into a fresh thread (TASK-1579).
    assert selected == SessionPolicy.RESET_RETAIN_HISTORY
    assert "critical" in reason
    selected, reason = preventive_session_policy(
        requested=SessionPolicy.CONTINUE,
        health_state="critical",
        configured_critical_policy="reset_retain_history",
        backend="local",
    )
    assert selected == SessionPolicy.CONTINUE
    assert "no safe reset" in reason


def test_context_overflow_classifier_is_narrow():
    classify = InterBotDeliveryDispatcher._is_context_overflow
    assert classify("Your input exceeds the context window of this model")
    assert classify("maximum context length exceeded")
    assert not classify("HTTP 500 upstream overloaded")
    assert not classify("request timed out")


def test_agent_context_compact_is_capability_gated(monkeypatch):
    from llm_bawt.mcp_server import inter_bot_tools

    async def unsupported(*_args, **_kwargs):
        return {"backend": "codex", "capabilities": {"compact": False}}

    called = []

    async def enqueue(**kwargs):
        called.append(kwargs)
        return {"success": True, "delivery_id": "delivery-compact"}

    monkeypatch.setattr(inter_bot_tools, "agent_context_health", unsupported)
    monkeypatch.setattr(inter_bot_tools, "_enqueue_durable", enqueue)
    result = _run(inter_bot_tools.agent_context_compact("codex", "compact-1"))
    assert result["success"] is False
    assert called == []

    async def supported(*_args, **_kwargs):
        return {"backend": "claude-code", "capabilities": {"compact": True}}

    monkeypatch.setattr(inter_bot_tools, "agent_context_health", supported)
    result = _run(inter_bot_tools.agent_context_compact(
        "snark", "compact-2", sender_bot_id="snark"
    ))
    assert result["delivery_id"] == "delivery-compact"
    assert called[0]["prefer_steer"] is False
    assert called[0]["message"] == "/compact"
    assert called[0]["session_policy"] == "continue"
    assert called[0]["idempotency_key"] == "compact-2"


def test_empty_seed_is_serialized_as_explicit_decision():
    subscriber = RedisSubscriber("redis://localhost:6379/0")
    fake = MagicMock()
    commands = []

    async def eval_script(_script, _numkeys, _dedupe, _stream, _maxlen, _request, *flat):
        commands.append(dict(zip(flat[::2], flat[1::2])))
        return [1, "1-0"]

    fake.eval = eval_script
    subscriber._pub_redis = fake
    _run(subscriber.send_command(
        session_key="snark:nick",
        message="clean",
        request_id="req-empty-seed",
        inject_messages=[],
    ))
    assert commands[0]["inject_messages"] == "[]"

    from claude_code_bridge.send_request import SendRequest
    parsed = SendRequest.from_fields(commands[0])
    assert parsed.inject_messages == []


def test_responses_transport_is_serialized_into_bridge_command():
    subscriber = RedisSubscriber("redis://localhost:6379/0")
    fake = MagicMock()
    commands = []

    async def eval_script(_script, _numkeys, _dedupe, _stream, _maxlen, _request, *flat):
        commands.append(dict(zip(flat[::2], flat[1::2])))
        return [1, "1-0"]

    fake.eval = eval_script
    subscriber._pub_redis = fake
    _run(subscriber.send_command(
        session_key="snark:nick",
        message="parallel",
        request_id="req-transport",
        responses_transport="sse",
    ))

    assert commands[0]["responses_transport"] == "sse"


def test_agent_client_preserves_empty_seed_per_call():
    from llm_bawt.clients.agent_backend_client import AgentBackendClient

    captured = {}

    class Backend:
        async def chat_full(self, prompt, config):
            captured.update(config)
            return SimpleNamespace(text="ok")

    client = AgentBackendClient.__new__(AgentBackendClient)
    client._bot_config = {"bot_id": "snark"}
    client._backend = Backend()
    result = _run(client._chat_full("hello", inject_messages=[]))
    assert result.text == "ok"
    assert captured["inject_messages"] == []
    assert client._bot_config == {"bot_id": "snark"}


def test_durable_message_id_fits_canonical_history_schema():
    import uuid as uuid_module

    from llm_bawt.inter_bot_delivery import InterBotDeliveryStore

    _delivery_id, message_id, _turn_id = InterBotDeliveryStore.stable_ids()
    assert str(uuid_module.UUID(message_id)) == message_id
    assert len(message_id) == 36


def test_spoofed_delivery_correlation_is_rejected_before_turn_start():
    from llm_bawt.service.background_service import BackgroundService
    from llm_bawt.service.schemas import ChatCompletionRequest, ChatMessage

    service = BackgroundService.__new__(BackgroundService)
    service._default_bot = "snark"
    service._inter_bot_dispatcher = SimpleNamespace(
        store=SimpleNamespace(validate_claim=lambda **_kwargs: False)
    )
    request = ChatCompletionRequest(
        bot_id="snark",
        messages=[ChatMessage(role="user", content="spoof")],
        user_message_id="interbot-msg",
        inter_bot_delivery_id="delivery-spoof",
        inter_bot_turn_id="turn-delivery-spoof",
        inter_bot_bridge_request_id="req_delivery_spoof",
        inter_bot_claim_token="claim-spoof",
    )

    try:
        _run(service.chat_completion(request))
    except ValueError as exc:
        assert "invalid or stale" in str(exc)
    else:
        raise AssertionError("spoofed delivery claim was accepted")


def test_spoofed_delivery_correlation_is_rejected_on_stream_path():
    from llm_bawt.service.background_service import BackgroundService
    from llm_bawt.service.schemas import ChatCompletionRequest, ChatMessage

    service = BackgroundService.__new__(BackgroundService)
    service._default_bot = "snark"
    service._inter_bot_dispatcher = SimpleNamespace(
        store=SimpleNamespace(validate_claim=lambda **_kwargs: False)
    )
    request = ChatCompletionRequest(
        bot_id="snark",
        messages=[ChatMessage(role="user", content="spoof")],
        user_message_id="interbot-msg",
        inter_bot_delivery_id="delivery-spoof",
        inter_bot_turn_id="turn-delivery-spoof",
        inter_bot_bridge_request_id="req_delivery_spoof",
        inter_bot_claim_token="claim-spoof",
    )

    async def consume():
        chunks = []
        async for chunk in service.chat_completion_stream(request):
            chunks.append(chunk)
        return chunks

    try:
        _run(consume())
    except ValueError as exc:
        assert "invalid or stale" in str(exc)
    else:
        raise AssertionError("spoofed streaming delivery claim was accepted")


def test_durable_dispatch_drains_stream_and_uses_persisted_outcome():
    from datetime import datetime, timezone

    record = SimpleNamespace(
        id="delivery-1",
        turn_id="turn-delivery-1",
        claim_token="claim-1",
        target_bot_id="loopy",
        metadata={}, session_policy="continue", transport_accepted_at=None,
    )
    marked = []
    emitted = []
    payload = {
        "messages": [{"role": "user", "content": "work"}],
        "bot_id": "loopy",
        "user_message_id": "00000000-0000-4000-8000-000000000001",
        "assistant_message_id": "00000000-0000-4000-8000-000000000002",
        "inter_bot_delivery_id": "delivery-1",
        "inter_bot_turn_id": "turn-delivery-1",
        "inter_bot_bridge_request_id": "req_delivery_1",
    }
    turn = SimpleNamespace(
        ended_at=datetime.now(timezone.utc),
        status="ok",
        error_text=None,
        response_text="streamed reply",
        model="test-model",
    )
    store = SimpleNamespace(
        payload=lambda _id: payload,
        mark_transport_accepted=lambda *_args: True,
        mark_delivered=lambda *args, **kwargs: (
            marked.append((args, kwargs)) or SimpleNamespace(
                status="DELIVERED", metadata={}, session_policy="continue",
            )
        ),
        renew_lease=lambda *_args: False,
    )

    async def stream(request):
        assert request.stream is True
        assert request.inter_bot_turn_id == record.turn_id
        yield ": connected\n\n"
        yield "data: [DONE]\n\n"

    dispatcher = InterBotDeliveryDispatcher.__new__(InterBotDeliveryDispatcher)
    dispatcher.store = store
    dispatcher.service = SimpleNamespace(
        chat_completion_stream=stream,
        _turn_log_store=SimpleNamespace(get_turn=lambda _id: turn),
    )
    dispatcher._heartbeat = lambda *_args: asyncio.sleep(3600)

    async def emit(value):
        emitted.append(value)

    dispatcher._emit = emit
    _run(dispatcher._dispatch(record))

    assert marked[0][1] == {
        "response_model": "test-model",
        "response_chars": len("streamed reply"),
    }
    assert emitted[0].status == "DELIVERED"


def test_durable_dispatch_rejects_ok_turn_with_persisted_error():
    from datetime import datetime, timezone

    record = SimpleNamespace(
        id="delivery-err",
        turn_id="turn-delivery-err",
        claim_token="claim-err",
        target_bot_id="loopy",
        attempt_count=1,
        max_attempts=1,
        overflow_recovery_count=0,
        metadata={}, session_policy="continue", transport_accepted_at=None,
    )
    payload = {
        "messages": [{"role": "user", "content": "work"}],
        "bot_id": "loopy",
        "user_message_id": "00000000-0000-4000-8000-000000000003",
        "assistant_message_id": "00000000-0000-4000-8000-000000000004",
        "inter_bot_delivery_id": "delivery-err",
        "inter_bot_turn_id": "turn-delivery-err",
        "inter_bot_bridge_request_id": "req_delivery_err",
    }
    turn = SimpleNamespace(
        ended_at=datetime.now(timezone.utc),
        status="ok",
        error_text="upstream exploded",
        response_text="partial",
        model="test-model",
    )
    failed = []
    store = SimpleNamespace(
        payload=lambda _id: payload,
        mark_transport_accepted=lambda *_args: True,
        get=lambda _id: record,
        turn_state=lambda _id: ("ok", turn.ended_at),
        fail_claim=lambda *args: (
            failed.append(args) or SimpleNamespace(status="FAILED")
        ),
        renew_lease=lambda *_args: False,
    )

    async def stream(_request):
        yield "data: [DONE]\n\n"

    dispatcher = InterBotDeliveryDispatcher.__new__(InterBotDeliveryDispatcher)
    dispatcher.store = store
    dispatcher.service = SimpleNamespace(
        config=SimpleNamespace(),
        chat_completion_stream=stream,
        _turn_log_store=SimpleNamespace(get_turn=lambda _id: turn),
    )
    dispatcher._heartbeat = lambda *_args: asyncio.sleep(3600)
    dispatcher._emit = lambda _value: asyncio.sleep(0)
    _run(dispatcher._dispatch(record))

    assert failed
    assert "upstream exploded" in failed[0][2]


@pytest.mark.parametrize("ended", [True, False])
def test_deterministic_run_requeues_only_while_turn_may_still_be_live(ended):
    """TASK-1095: Snark's errored turn was requeued 4x; each retry re-read the
    finished `agent:run:req_delivery_*` stream from 0, re-broadcast ~300 tool
    events to the browser, and failed on the same error frame."""
    from datetime import datetime, timezone

    record = SimpleNamespace(
        id="delivery-dead", turn_id="turn-delivery-dead", claim_token="claim",
        target_bot_id="snark", attempt_count=1, max_attempts=5,
        overflow_recovery_count=0,
        metadata={}, session_policy="continue", transport_accepted_at=None,
    )
    payload = {
        "messages": [{"role": "user", "content": "work"}],
        "bot_id": "snark",
        "user_message_id": "00000000-0000-4000-8000-000000000005",
        "assistant_message_id": "00000000-0000-4000-8000-000000000006",
        "inter_bot_delivery_id": "delivery-dead",
        "inter_bot_turn_id": "turn-delivery-dead",
        "inter_bot_bridge_request_id": "req_delivery_dead",
    }
    ended_at = datetime.now(timezone.utc) if ended else None
    turn = SimpleNamespace(
        ended_at=ended_at, status="error" if ended else "streaming",
        error_text="Proxy stream failed: ChatGPTEventTimeout" if ended else None,
        response_text="", model="m",
    )
    failed, requeued = [], []
    store = SimpleNamespace(
        payload=lambda _id: payload,
        mark_transport_accepted=lambda *_args: True,
        get=lambda _id: record,
        turn_state=lambda _id: (turn.status, ended_at),
        fail_claim=lambda *args: failed.append(args) or SimpleNamespace(status="FAILED"),
        requeue=lambda *args, **kw: requeued.append(args) or SimpleNamespace(status="QUEUED"),
        renew_lease=lambda *_args: False,
    )

    async def stream(_request):
        yield "data: [DONE]\n\n"

    dispatcher = InterBotDeliveryDispatcher.__new__(InterBotDeliveryDispatcher)
    dispatcher.store = store
    dispatcher.service = SimpleNamespace(
        config=SimpleNamespace(),
        chat_completion_stream=stream,
        _turn_log_store=SimpleNamespace(get_turn=lambda _id: turn),
    )
    dispatcher._heartbeat = lambda *_args: asyncio.sleep(3600)
    dispatcher._emit = lambda _value: asyncio.sleep(0)
    _run(dispatcher._dispatch(record))

    if ended:
        assert len(failed) == 1 and not requeued
        assert "ChatGPTEventTimeout" in failed[0][2]
    else:
        assert len(requeued) == 1 and not failed


def test_passive_dispatcher_retries_and_acquires_leadership(monkeypatch):
    attempts = iter([None, object()])
    stop_event = asyncio.Event()
    dispatcher = InterBotDeliveryDispatcher.__new__(InterBotDeliveryDispatcher)
    dispatcher.store = SimpleNamespace(
        acquire_dispatcher_lock=lambda: next(attempts)
    )
    dispatcher._dispatcher_lock = None
    dispatcher._stop_event = stop_event
    dispatcher._task = None

    async def fake_run():
        stop_event.set()

    dispatcher.run = fake_run

    async def immediate_wait(awaitable, timeout):
        awaitable.close()
        raise asyncio.TimeoutError

    monkeypatch.setattr(asyncio, "wait_for", immediate_wait)
    _run(dispatcher._leadership_loop())

    assert dispatcher._dispatcher_lock is not None
    assert dispatcher._task is not None


def test_heartbeat_renews_live_claim(monkeypatch):
    renewals: list[tuple[str, str]] = []
    store = SimpleNamespace(
        renew_lease=lambda delivery_id, claim_token: (
            renewals.append((delivery_id, claim_token)) or False
        )
    )
    dispatcher = InterBotDeliveryDispatcher.__new__(InterBotDeliveryDispatcher)
    dispatcher.store = store

    original_sleep = asyncio.sleep

    async def immediate_sleep(_seconds):
        return None

    monkeypatch.setattr(asyncio, "sleep", immediate_sleep)
    _run(dispatcher._heartbeat("delivery-1", "claim-1"))
    monkeypatch.setattr(asyncio, "sleep", original_sleep)

    assert renewals == [("delivery-1", "claim-1")]


def test_agent_client_threads_stable_request_and_timeout_into_config(monkeypatch):
    from llm_bawt.clients.agent_backend_client import AgentBackendClient

    captured = {}

    class Backend:
        async def chat_full(self, prompt, config):
            captured.update(config)
            return SimpleNamespace(text="ok")

    client = AgentBackendClient.__new__(AgentBackendClient)
    client._bot_config = {"timeout_seconds": 600, "bot_id": "snark"}
    client._backend = Backend()

    result = _run(client._chat_full(
        "hello",
        bridge_request_id="req_delivery_abc",
        bridge_timeout_seconds=7200,
    ))

    assert result.text == "ok"
    assert captured["request_id"] == "req_delivery_abc"
    assert captured["timeout_seconds"] == 7200
    assert client._bot_config["timeout_seconds"] == 600


def test_agent_client_query_threads_bridge_event_callback_without_running_loop():
    from llm_bawt.clients.agent_backend_client import AgentBackendClient
    from llm_bawt.models.message import Message

    captured = {}
    callback_events = []
    callback = callback_events.append

    class Backend:
        async def chat_full(self, prompt, config):
            captured.update(config)
            config["event_callback"]({"event": "tool_call", "name": "Read"})
            return SimpleNamespace(text="ok")

    client = AgentBackendClient.__new__(AgentBackendClient)
    client._bot_config = {"bot_id": "loopy"}
    client._backend = Backend()
    client.last_result = None

    response = client.query(
        [Message(role="user", content="inspect")],
        bridge_request_id="req_delivery_callback",
        bridge_event_callback=callback,
    )

    assert response == "ok"
    assert captured["request_id"] == "req_delivery_callback"
    assert captured["event_callback"] is callback
    assert callback_events == [{"event": "tool_call", "name": "Read"}]


def test_failed_delivery_response_is_non_2xx_with_stable_id():
    from llm_bawt.inter_bot_delivery import submission_result

    failed = SimpleNamespace(
        status="FAILED",
        to_api=lambda duplicate=False: {
            "delivery_id": "delivery-failed",
            "status": "FAILED",
            "last_error": "Unknown target bot 'missing'",
            "duplicate": duplicate,
        },
    )
    status_code, result = submission_result(
        failed,
        duplicate=False,
        target_exists=False,
        requested_delivery="steer_or_idle",
    )

    assert status_code == 404
    assert result["delivery_id"] == "delivery-failed"
    assert result["delivery"] == "steer_or_idle"
    assert result["success"] is False


def test_ended_never_ready_turn_is_definitively_not_active():
    from datetime import datetime, timezone

    from fastapi import HTTPException
    from llm_bawt.service.routes.chat import ChatSteerRequest, steer_active_turn

    turn = SimpleNamespace(
        id="turn-ended-before-ready",
        bot_id="snark",
        user_id="nick",
        status="error",
        ended_at=datetime.now(timezone.utc),
        agent_session_key=None,
        agent_request_id=None,
    )
    service = SimpleNamespace(
        _turn_log_store=SimpleNamespace(get_turn=lambda _turn_id: turn)
    )

    with pytest.raises(HTTPException) as caught:
        _run(steer_active_turn(
            service,
            ChatSteerRequest(
                turn_id=turn.id,
                message="continue safely",
                message_id="interbot-ended-before-ready",
                bot_id="snark",
                user_id="nick",
            ),
            steer_request_id="steer_delivery_ended_before_ready",
        ))
    assert caught.value.status_code == 409
    assert caught.value.detail == "no_active_run"


def test_interrupt_anchor_uses_canonical_turn_trigger():
    from llm_bawt.service.routes.chat import _persist_interrupt_anchor

    captured: dict[str, object] = {}

    class FakeBackend:
        def set_interrupt_anchor(self, message_id, source_message_id, content_offset):
            captured.update(
                message_id=message_id,
                source_message_id=source_message_id,
                content_offset=content_offset,
            )
            return True

    manager = SimpleNamespace(_backend=FakeBackend())
    memory_client = SimpleNamespace(get_short_term_manager=lambda: manager)
    service = SimpleNamespace(get_memory_client=lambda _bot_id: memory_client)

    assert _persist_interrupt_anchor(
        service,
        bot_id="snark",
        message_id="interrupt-1",
        source_message_id="canonical-trigger-1",
        content_offset=42,
    ) is True
    assert captured == {
        "message_id": "interrupt-1",
        "source_message_id": "canonical-trigger-1",
        "content_offset": 42,
    }


def test_steer_failure_policy_never_falls_back_after_acceptance():
    classify = InterBotDeliveryDispatcher._steer_failure_action

    assert classify("Active bridge run is not ready", accepted=False) == "wait_ready"
    assert classify("no_active_run", accepted=False) == "fallback"
    assert classify("active_run_mismatch", accepted=False) == "fallback"
    assert classify("chat.steer timed out after 10s", accepted=False) == "retry_same_steer"
    assert classify("no_active_run", accepted=True) == "retry_same_steer"
    assert classify("Turn not found", accepted=True) == "retry_same_steer"


def test_dispatcher_classifies_terminal_and_transient_errors():
    assert InterBotDeliveryDispatcher._is_terminal(ValueError("bad payload")) is True

    request = httpx.Request("POST", "http://app/v1/chat/completions")
    terminal = httpx.HTTPStatusError(
        "not found",
        request=request,
        response=httpx.Response(404, request=request),
    )
    transient = httpx.HTTPStatusError(
        "unavailable",
        request=request,
        response=httpx.Response(503, request=request),
    )
    busy = httpx.HTTPStatusError(
        "busy",
        request=request,
        response=httpx.Response(409, request=request),
    )

    assert InterBotDeliveryDispatcher._is_terminal(terminal) is True
    assert InterBotDeliveryDispatcher._is_terminal(transient) is False
    assert InterBotDeliveryDispatcher._is_terminal(busy) is False
