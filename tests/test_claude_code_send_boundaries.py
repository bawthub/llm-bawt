import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from claude_agent_sdk import AssistantMessage, ResultMessage, StreamEvent, TextBlock

from agent_bridge.events import AgentEventKind
from agent_bridge.session_queue import SessionQueue
from claude_code_bridge.command_ops import ClaudeCommandMixin
from claude_code_bridge.event_ops import ClaudeEventMixin
from claude_code_bridge.send_handler import ClaudeSendMixin
from claude_code_bridge.send_boundaries import (
    publish_run_done_once,
    separator_before_new_block,
    should_read_native_context_usage,
)


def test_new_reasoning_block_gets_markdown_paragraph_boundary() -> None:
    assert separator_before_new_block("**first thought**") == "\n\n"


def test_first_or_already_separated_reasoning_block_gets_no_extra_boundary() -> None:
    assert separator_before_new_block("") == ""
    assert separator_before_new_block("**first thought**\n") == ""
    assert separator_before_new_block("**first thought**\n\n") == ""


def test_proxy_turn_skips_native_context_control_request() -> None:
    assert should_read_native_context_usage(use_proxy=True) is False


def test_direct_turn_keeps_native_context_control_request() -> None:
    assert should_read_native_context_usage(use_proxy=False) is True


def test_run_done_is_published_exactly_once() -> None:
    class Publisher:
        def __init__(self) -> None:
            self.request_ids: list[str] = []

        def publish_run_done(self, request_id: str) -> None:
            self.request_ids.append(request_id)

    publisher = Publisher()
    published = publish_run_done_once(
        publisher,
        "req-terminal",
        already_published=False,
    )
    published = publish_run_done_once(
        publisher,
        "req-terminal",
        already_published=published,
    )

    assert published is True
    assert publisher.request_ids == ["req-terminal"]


SESSION = "claude-code:test:thread"
MODEL = "claude-sonnet-4-6"
CANCELLED = {"end_reason": "aborted", "status": "cancelled"}


class _SendHarness(ClaudeSendMixin, ClaudeCommandMixin, ClaudeEventMixin):
    def __init__(self):
        self._session_queue = SessionQueue()
        self._publisher = Mock()
        self._proxy_base_url = None
        self._proxy_request_sessions = {}
        self._trigger_message_ids = {}
        self._backend_name = "claude-code"
        self._request_timeout = 5
        self.events = []
        self.redis = SimpleNamespace(xack=AsyncMock())
        self._preprocess_new_command = AsyncMock(side_effect=lambda message, **kw: message)
        self._build_sdk_env = Mock(return_value={})
        self._build_agent_options = Mock(return_value=None)
        self._make_can_use_tool = Mock()
        self._make_pre_tool_use_hook = Mock()
        self._make_post_tool_use_hook = Mock()
        self._discard_changed_file_request = Mock()
        self._read_native_context_usage = AsyncMock(return_value=None)
        self.queued = asyncio.Event()

    @staticmethod
    def _model_provider_prefix(model):
        return None

    def _publish_run_event_with_changed_file(self, request_id, event):
        self.events.append(event)
        if event.raw.get("turn_health", {}).get("phase") == "queued":
            self.queued.set()

    def send(self, request_id="turn-request"):
        return asyncio.create_task(self._handle_send({
            "request_id": request_id,
            "session_key": SESSION,
            "model": MODEL,
            "message": "hello",
            "thread_resume_id": "sdk-session",
            "trigger_message_id": f"message-{request_id}",
        }, f"send-{request_id}", self.redis))

    async def abort(self, request_id="turn-request", *, method="chat.abort"):
        await self._handle_rpc({
            "request_id": f"rpc-{request_id}",
            "method": method,
            "params": {"sessionKey": SESSION, "requestId": request_id},
        }, f"abort-{request_id}", self.redis)

    def terminal(self, request_id="turn-request"):
        return [e for e in self.events
                if e.run_id == request_id and e.kind == AgentEventKind.ASSISTANT_DONE]


class _SDKStream:
    def __init__(self, *, result=False, delay_disconnect=False):
        self.result = result
        self.waiting = asyncio.Event()
        self.release_stream = asyncio.Event()
        self.disconnect_entered = asyncio.Event()
        self.release_disconnect = asyncio.Event()
        if not delay_disconnect:
            self.release_disconnect.set()
        self.disconnect_calls = 0

    async def connect(self, prompt):
        pass

    async def receive_messages(self):
        yield StreamEvent(uuid="delta", session_id="sdk-session", event={
            "type": "content_block_delta",
            "delta": {"type": "text_delta", "text": "partial answer"},
        })
        yield AssistantMessage(
            content=[TextBlock(text="partial answer")], model=MODEL,
            usage={"input_tokens": 23, "output_tokens": 7},
        )
        self.waiting.set()
        await self.release_stream.wait()
        if self.result:
            yield ResultMessage(
                subtype="success", duration_ms=1, duration_api_ms=1,
                is_error=False, num_turns=1, session_id="sdk-session",
                usage={"input_tokens": 23, "output_tokens": 11},
            )

    async def disconnect(self):
        self.disconnect_calls += 1
        self.disconnect_entered.set()
        self.release_stream.set()
        # Only the RPC disconnect is delayed; send cleanup may finish first.
        if self.disconnect_calls == 1:
            await self.release_disconnect.wait()


def _assert_cancelled(harness, request_id="turn-request", *, partial=True):
    terminal, = harness.terminal(request_id)
    assert terminal.raw == CANCELLED
    assert terminal.session_key == SESSION
    assert terminal.run_id == request_id
    assert terminal.trigger_message_id == f"message-{request_id}"
    assert terminal.text == ("partial answer" if partial else "")
    if partial:
        assert terminal.token_usage["input_tokens"] == 23
        assert terminal.token_usage["output_tokens"] >= 7
    assert [call.args[0] for call in harness._publisher.publish_run_done.call_args_list].count(request_id) == 1
    assert not [e for e in harness.events if e.kind == AgentEventKind.ERROR]


@pytest.mark.parametrize("result", [False, True], ids=["EOF", "ResultMessage"])
def test_disconnect_terminal_is_cancelled_not_success(monkeypatch, result):
    async def run():
        harness = _SendHarness()
        sdk = _SDKStream(result=result, delay_disconnect=True)
        monkeypatch.setattr("claude_code_bridge.send_handler.ClaudeSDKClient", lambda **kw: sdk)
        send = harness.send()
        await sdk.waiting.wait()
        # Exercise the cooperative path even if the SDK consumes task cancellation:
        # disconnect produces EOF/a ResultMessage before its await returns.
        harness._session_queue.signal_cancel(SESSION)
        disconnect = asyncio.create_task(sdk.disconnect())
        await sdk.disconnect_entered.wait()
        await send
        assert not disconnect.done()
        _assert_cancelled(harness)
        if result:
            assert harness.terminal()[0].token_usage["output_tokens"] == 11
        sdk.release_disconnect.set()
        await disconnect
    asyncio.run(asyncio.wait_for(run(), timeout=5))


def test_delayed_abort_never_cancels_successor_or_repeats_done(monkeypatch):
    async def run():
        harness = _SendHarness()
        first = _SDKStream(delay_disconnect=True)
        second = _SDKStream(result=True)
        clients = iter([first, second])
        monkeypatch.setattr("claude_code_bridge.send_handler.ClaudeSDKClient", lambda **kw: next(clients))
        first_send = harness.send()
        await first.waiting.wait()
        successor = harness.send("successor")
        await harness.queued.wait()
        abort = asyncio.create_task(harness.abort())
        await first.disconnect_entered.wait()
        # Repeat while cleanup is pending: no second task.cancel injection.
        await harness.abort()
        await asyncio.gather(first_send, return_exceptions=True)
        await second.waiting.wait()
        _assert_cancelled(harness)
        first.release_disconnect.set()
        await abort
        await harness.abort()  # stale target after the successor owns the session
        assert not successor.done()
        assert not harness._session_queue.cancel_event(SESSION).is_set()
        assert second.disconnect_calls == 0
        second.release_stream.set()
        await successor
        terminal, = harness.terminal("successor")
        assert terminal.raw == {}
        assert terminal.text == "partial answer"
        _assert_cancelled(harness)
    asyncio.run(asyncio.wait_for(run(), timeout=5))


@pytest.mark.parametrize("method", ["chat.abort", "chat.cancel"])
def test_queued_cancellation_leaves_active_run_untouched(monkeypatch, method):
    async def run():
        harness = _SendHarness()
        sdk = _SDKStream(result=True)
        monkeypatch.setattr("claude_code_bridge.send_handler.ClaudeSDKClient", lambda **kw: sdk)
        active = harness.send()
        await sdk.waiting.wait()
        queued = harness.send("queued")
        await harness.queued.wait()
        await harness.abort("queued", method=method)
        await asyncio.gather(queued, return_exceptions=True)
        _assert_cancelled(harness, "queued", partial=False)
        assert not active.done()
        assert not harness._session_queue.cancel_event(SESSION).is_set()
        assert sdk.disconnect_calls == 0
        sdk.release_stream.set()
        await active
        assert harness.terminal()[0].raw == {}
    asyncio.run(asyncio.wait_for(run(), timeout=5))


def test_steer_defaults_to_next_and_requires_origin_for_user_lifecycle():
    async def run():
        harness = _SendHarness()
        active = SimpleNamespace(request_id="target-run", steer=AsyncMock(), cancel_queued=AsyncMock())
        harness._session_queue.set_active_client(SESSION, active)
        await harness._handle_steer({
            "session_key": SESSION, "request_id": "steer-1",
            "target_request_id": "target-run", "message_id": "id-1",
            "message": "note", "origin": "user",
        }, "command-1", harness.redis)
        active.steer.assert_awaited_once_with(
            "note", priority="next", message_id="id-1", origin="user",
        )
        harness._publisher.publish_rpc_result.assert_called_once_with(
            "steer-1", {"ok": True, "detail": "steered", "active_request_id": "target-run", "can_escalate": False, "can_cancel": True},
        )
    asyncio.run(run())


def test_escalation_calls_active_run_without_resending_message():
    async def run():
        harness = _SendHarness()
        active = SimpleNamespace(request_id="target-run", steer=AsyncMock(),
                                 escalate=AsyncMock(return_value="escalated"))
        harness._session_queue.set_active_client(SESSION, active)
        fields = {
            "session_key": SESSION, "request_id": "escalate-1",
            "target_request_id": "target-run", "message_id": "id-1", "escalate": "1",
        }
        await harness._handle_steer(fields, "command-1", harness.redis)
        active.escalate.assert_awaited_once_with("id-1")
        active.steer.assert_not_awaited()
        harness._publisher.publish_rpc_result.assert_called_once_with(
            "escalate-1", {"ok": True, "detail": "escalated", "active_request_id": "target-run", "can_escalate": False, "can_cancel": False},
        )
    asyncio.run(run())


def test_cancel_queued_targets_one_active_message_without_interrupting():
    async def run():
        harness = _SendHarness()
        active = SimpleNamespace(request_id="target-run", steer=AsyncMock(),
                                 cancel_queued=AsyncMock(return_value="cancelled"))
        harness._session_queue.set_active_client(SESSION, active)
        await harness._handle_steer({
            "session_key": SESSION, "request_id": "cancel-1", "steer_action": "cancel",
            "target_request_id": "target-run", "message_id": "id-1",
        }, "command-1", harness.redis)
        active.cancel_queued.assert_awaited_once_with("id-1")
        active.steer.assert_not_awaited()
        harness._publisher.publish_rpc_result.assert_called_once_with(
            "cancel-1", {"ok": True, "detail": "cancelled", "active_request_id": "target-run", "can_escalate": False, "can_cancel": False},
        )
    asyncio.run(run())


def test_escalation_validates_target_and_refuses_message_payload():
    async def run():
        harness = _SendHarness()
        active = SimpleNamespace(request_id="target-run", steer=AsyncMock())
        harness._session_queue.set_active_client(SESSION, active)
        base = {"session_key": SESSION, "request_id": "escalate-1", "escalate": "1"}
        cases = [
            ({"target_request_id": "other-run", "message_id": "id-1"}, "active_run_mismatch"),
            ({"target_request_id": "target-run", "message_id": "id-1", "message": "duplicate"}, "must not include message"),
            ({"target_request_id": "target-run"}, "requires message_id and target_request_id"),
            ({"message_id": "id-1"}, "requires message_id and target_request_id"),
        ]
        for i, (extra, error) in enumerate(cases):
            await harness._handle_steer(base | extra, f"command-{i}", harness.redis)
            assert error in harness._publisher.publish_rpc_result.call_args.args[1]["error"]
        active.steer.assert_not_awaited()
    asyncio.run(run())


@pytest.mark.parametrize("result", [False, True], ids=["EOF", "ResultMessage"])
def test_normal_completion_still_emits_one_success(monkeypatch, result):
    async def run():
        harness = _SendHarness()
        sdk = _SDKStream(result=result)
        monkeypatch.setattr("claude_code_bridge.send_handler.ClaudeSDKClient", lambda **kw: sdk)
        send = harness.send()
        await sdk.waiting.wait()
        sdk.release_stream.set()
        await send
        terminal, = harness.terminal()
        assert terminal.raw == {}
        assert terminal.text == "partial answer"
        assert terminal.trigger_message_id == "message-turn-request"
        harness._publisher.publish_run_done.assert_called_once_with("turn-request")
    asyncio.run(asyncio.wait_for(run(), timeout=5))


def test_abort_during_result_context_read_preserves_result_usage(monkeypatch):
    async def run():
        harness = _SendHarness()
        sdk = _SDKStream(result=True)
        reading_context = asyncio.Event()

        async def read_context(client):
            reading_context.set()
            await asyncio.Event().wait()

        harness._read_native_context_usage = read_context
        monkeypatch.setattr("claude_code_bridge.send_handler.ClaudeSDKClient", lambda **kw: sdk)
        send = harness.send()
        await sdk.waiting.wait()
        sdk.release_stream.set()
        await reading_context.wait()
        await harness.abort()
        await asyncio.gather(send, return_exceptions=True)
        _assert_cancelled(harness)
        assert harness.terminal()[0].token_usage["output_tokens"] == 11
    asyncio.run(asyncio.wait_for(run(), timeout=5))


def test_abort_after_terminal_during_cleanup_does_not_duplicate_done(monkeypatch):
    async def run():
        harness = _SendHarness()
        sdk = _SDKStream(result=True, delay_disconnect=True)
        monkeypatch.setattr("claude_code_bridge.send_handler.ClaudeSDKClient", lambda **kw: sdk)
        send = harness.send()
        await sdk.waiting.wait()
        sdk.release_stream.set()
        await sdk.disconnect_entered.wait()
        terminal, = harness.terminal()
        assert terminal.raw == {}
        await harness.abort()
        await send
        assert harness.terminal() == [terminal]
        harness._publisher.publish_run_done.assert_called_once_with("turn-request")
    asyncio.run(asyncio.wait_for(run(), timeout=5))


def test_abort_during_sdk_setup_emits_cancelled_terminal(monkeypatch):
    async def run():
        harness = _SendHarness()
        sdk = _SDKStream()
        connecting = asyncio.Event()

        async def connect(prompt):
            connecting.set()
            await asyncio.Event().wait()

        sdk.connect = connect
        monkeypatch.setattr("claude_code_bridge.send_handler.ClaudeSDKClient", lambda **kw: sdk)
        send = harness.send()
        await connecting.wait()
        await harness.abort()
        await asyncio.gather(send, return_exceptions=True)
        _assert_cancelled(harness, partial=False)
    asyncio.run(asyncio.wait_for(run(), timeout=5))


def test_session_only_abort_is_rejected(monkeypatch):
    async def run():
        harness = _SendHarness()
        sdk = _SDKStream(result=True)
        monkeypatch.setattr("claude_code_bridge.send_handler.ClaudeSDKClient", lambda **kw: sdk)
        send = harness.send()
        await sdk.waiting.wait()
        await harness._handle_rpc({
            "request_id": "rpc", "method": "chat.abort", "params": {"sessionKey": SESSION},
        }, "abort", harness.redis)
        assert harness._publisher.publish_rpc_result.call_args.args[1]["ok"] is False
        assert not send.done()
        assert sdk.disconnect_calls == 0
        sdk.release_stream.set()
        await send
    asyncio.run(asyncio.wait_for(run(), timeout=5))
