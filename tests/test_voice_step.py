"""DB-free voice next-step attribution and public streaming regressions (TASK-1753)."""

import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from llm_bawt.service import chat_response_adapter as adapter
from llm_bawt.service import voice_step
from llm_bawt.service.delivery_reply import DeliveryReplyMismatch
from llm_bawt.service.schemas import ChatCompletionRequest, ChatMessage


BINDING = {"turn_id": "turn-1", "run_request_id": "run-1", "steer_rpc_id": "rpc-1"}
RECEIPT = {
    "object": "chat.completion.queued", "status": "QUEUED", "busy": True,
    "steer_capable": True, "delivery_id": "delivery-1", "user_message_id": "message-1",
    "bot_id": "test-bot", "user_id": "test-user",
}
DONE = "data: [DONE]\n\n"


def _step():
    return voice_step.VoiceStep(dict(BINDING), "message-1", "test-bot", "test-user")


def _event(kind, **fields):
    return {"_type": kind, "turn_id": "turn-1", "bot_id": "test-bot",
            "user_id": "test-user", **fields}


def _started(**fields):
    # No proof field: the exact ordered lifecycle barrier is the evidence.
    return _event("steer_lifecycle", message_id="message-1", run_request_id="run-1",
                  steer_rpc_id="rpc-1", state="started", text_offset=12, **fields)


def _delta(text, offset=12):
    return _event("text_delta", text_offset=offset, delta=text)


def _complete(offset=12, **fields):
    return _event("turn_complete", text_offset=offset, status="ok", **fields)


def _tool(offset=12, **fields):
    return _event("tool_event", event="tool_start", text_offset=offset, **fields)


def _replace(payload, **fields):
    return {**payload, **fields}


@pytest.mark.parametrize("field", [
    "turn_id", "bot_id", "user_id", "message_id", "run_request_id", "steer_rpc_id",
])
@pytest.mark.parametrize("value", [None, "other"])
def test_started_requires_exact_turn_message_run_rpc_and_conversation(field, value):
    step = _step()
    assert step.accept(_replace(_started(), **{field: value})) == (False, None)
    assert step.offset is None
    assert step.accept(_delta("not attributable")) == (False, None)
    assert step.accept(_complete(28)) == (True, None)


def test_prestart_text_and_tools_are_ignored_and_started_needs_no_proof_field():
    step = _step()
    assert step.accept(_delta("old turn answer", 0)) == (False, None)
    assert step.accept(_delta("lost pre-barrier text", 1000)) == (False, None)
    assert step.accept(_tool(1000)) == (False, None)
    assert step.accept(_started()) == (False, None)
    assert step.offset == 12
    assert step.accept(_delta("fresh")) == (False, None)
    assert step.accept(_complete(17)) == (True, "fresh")


@pytest.mark.parametrize("offset", [None, -1, True, 1.5, "12"])
def test_started_invalid_utf16_offset_fails_closed(offset):
    assert _step().accept(_replace(_started(), text_offset=offset)) == (True, None)


def test_duplicate_started_does_not_reset_text_or_utf16_frontier():
    step = _step()
    step.accept(_started())
    step.accept(_delta("answer"))
    assert step.accept(_started()) == (False, None)
    assert step.offset == 18
    assert step.accept(_complete(18)) == (True, "answer")


def test_emoji_utf16_overlap_and_duplicate_delta_are_spoken_once():
    step = _step()
    step.accept(_started())
    assert step.accept(_delta(" Hi\U0001f600")) == (False, None)
    assert step.offset == 17  # Space + two letters + a two-unit emoji.
    assert step.accept(_delta(" Hi\U0001f600")) == (False, None)
    assert step.accept(_delta("\U0001f600 there ", 15)) == (False, None)
    assert step.offset == 24
    assert step.units == 12
    assert step.accept(_delta("old", 0)) == (False, None)
    assert step.accept(_tool(24)) == (True, "Hi\U0001f600 there")


@pytest.mark.parametrize("event", [
    _delta("gap", 13),
    _delta("negative", -1),
    _replace(_delta("bad"), text_offset=True),
    _replace(_delta("bad"), text_offset="12"),
    _replace(_delta("bad"), text_offset=None),
    _replace(_delta("bad"), delta=None),
    _replace(_delta("bad"), delta=123),
    _delta("\ud800"),
    # Frontier 12 would split the emoji's surrogate pair beginning at 11.
    _delta("\U0001f600tail", 11),
])
def test_gap_or_malformed_delta_rejects_the_entire_step(event):
    step = _step()
    step.accept(_started())
    assert step.accept(event) == (True, None)


@pytest.mark.parametrize("boundary", [_tool, _complete])
@pytest.mark.parametrize("offset", [None, 16, 18])
def test_boundary_rejects_gap_or_missing_tail_even_after_valid_text(boundary, offset):
    step = _step()
    step.accept(_started())
    step.accept(_delta("valid"))  # Frontier 17; a terminal 18 proves a lost tail.
    assert step.accept(boundary(offset)) == (True, None)


def test_child_tool_does_not_end_step_but_first_root_tool_does():
    step = _step()
    step.accept(_started())
    step.accept(_delta("one"))
    assert step.accept(_tool(999, parent_tool_use_id="parent")) == (False, None)
    assert step.accept(_replace(_tool(999), event="tool_result")) == (False, None)
    assert step.accept(_delta(" two", 15)) == (False, None)
    assert step.accept(_tool(19)) == (True, "one two")


@pytest.mark.parametrize("boundary", [_tool, _complete])
@pytest.mark.parametrize("text", ["", " \n\t"])
def test_zero_or_whitespace_only_step_is_unavailable(boundary, text):
    step = _step()
    step.accept(_started())
    step.accept(_delta(text))
    assert step.accept(boundary(12 + len(text))) == (True, None)


@pytest.mark.parametrize("status", ["cancelled", "error", "failed", None])
def test_unsuccessful_turn_does_not_speak_partial_text(status):
    step = _step()
    step.accept(_started())
    step.accept(_delta("partial"))
    assert step.accept(_replace(_complete(19), status=status)) == (True, None)


@pytest.mark.parametrize("state", ["cancelled", "rejected", "uncertain"])
@pytest.mark.parametrize("started", [False, True])
def test_exact_rejected_or_uncertain_injection_discards_partial_text(state, started):
    step = _step()
    if started:
        step.accept(_started())
        step.accept(_delta("partial"))
    assert step.accept(_replace(_started(), state=state)) == (True, None)


@pytest.mark.parametrize("field", ["message_id", "run_request_id", "steer_rpc_id"])
def test_other_injection_after_barrier_makes_attribution_ambiguous(field):
    step = _step()
    step.accept(_started())
    step.accept(_delta("partial"))
    assert step.accept(_replace(_started(), **{field: "other"})) == (True, None)


@pytest.mark.parametrize("field", ["turn_id", "bot_id", "user_id"])
@pytest.mark.parametrize("event", [_delta("foreign"), _tool(), _complete(), _started()])
def test_other_turn_or_conversation_events_cannot_affect_active_step(field, event):
    step = _step()
    step.accept(_started())
    assert step.accept(_replace(event, **{field: "other"})) == (False, None)
    assert step.offset == 12
    step.accept(_delta("own"))
    assert step.accept(_complete(15)) == (True, "own")


def test_other_injection_rejection_is_not_our_rejection():
    step = _step()
    step.accept(_started())
    other = _replace(_started(), message_id="other", state="rejected")
    assert step.accept(other) == (False, None)
    step.accept(_delta("own"))
    assert step.accept(_complete(15)) == (True, "own")


def test_step_size_limit_counts_utf16_units(monkeypatch):
    monkeypatch.setattr(voice_step, "MAX_STEP_UNITS", 3)
    step = _step()
    step.accept(_started())
    assert step.accept(_delta("x\U0001f600")) == (False, None)
    assert step.units == 3
    assert step.accept(_delta("y", 15)) == (True, None)


class _Reader:
    """No Redis: only the actual reader's claim/events interface plus probes."""

    def __init__(self, events=(), *, block=False, claim_error=None):
        self.items = list(events)
        self.block = block
        self.claim_error = claim_error
        self.entered = asyncio.Event()
        self.closed = asyncio.Event()
        self.release = asyncio.Event()
        self.claimed = False
        self.superseded = False
        self.timeouts = []
        self.claim = AsyncMock(side_effect=self._claim)

    async def _claim(self):
        if self.claim_error is not None:
            raise self.claim_error
        self.claimed = True

    async def events(self, *, timeout_s):
        self.timeouts.append(timeout_s)
        self.entered.set()
        try:
            for event in self.items:
                yield event
            if self.block:
                await self.release.wait()
            self.superseded = True
        finally:
            self.closed.set()


def _request(*, voice=True, explicit_identity=True):
    return ChatCompletionRequest(
        model="test-model", bot_id="test-bot" if explicit_identity else None,
        user="test-user" if explicit_identity else None, stream=True, voice_input=voice,
        messages=[ChatMessage(role="user", content="spoken input")],
    )


def _service():
    subscriber = SimpleNamespace(close=AsyncMock(), _redis=SimpleNamespace(aclose=AsyncMock()))
    return SimpleNamespace(
        _default_bot="default-bot", config=SimpleNamespace(DEFAULT_USER="default-user"),
        _inter_bot_dispatcher=SimpleNamespace(store=object()), _turn_log_store=object(),
        _redis_subscriber=subscriber, abort=AsyncMock(), chat_completion=AsyncMock(),
    )


@pytest.fixture
def resolver(monkeypatch):
    resolved = Mock(return_value={"status": "IN_FLIGHT", "voice_binding": dict(BINDING)})
    monkeypatch.setattr(voice_step, "resolve_delivery_reply", resolved)

    async def inline(function, *args, **kwargs):
        # Resolve only against a fake store, without worker threads or a DB.
        return function(*args, **kwargs)

    monkeypatch.setattr(voice_step.asyncio, "to_thread", inline)
    return resolved


async def _observe(service, reader, *, receipt=None, deadline=None):
    return await voice_step.observe_voice_step(
        service, _request(), dict(RECEIPT) if receipt is None else receipt, reader,
        deadline=asyncio.get_running_loop().time() + 60 if deadline is None else deadline,
    )


def _assert_shared_resources_untouched(service):
    service._redis_subscriber.close.assert_not_awaited()
    service._redis_subscriber._redis.aclose.assert_not_awaited()
    service.abort.assert_not_awaited()
    service.chat_completion.assert_not_awaited()


def test_observer_uses_exact_durable_identity_and_closes_at_first_root_boundary(resolver):
    async def scenario():
        service = _service()
        reader = _Reader([
            None, _delta("old", 0), _started(), _delta("answer"), _tool(18),
            _delta("must not read whole turn", 18), _complete(42),
        ])
        assert await _observe(service, reader) == "answer"
        resolver.assert_called_once_with(
            service._inter_bot_dispatcher.store, service._turn_log_store, "delivery-1",
            bot_id="test-bot", user_id="test-user", user_message_id="message-1",
            include_voice_binding=True,
        )
        assert reader.closed.is_set()
        assert not reader.superseded  # The consumer closed at the boundary, not EOF.
        assert len(reader.timeouts) == 1
        assert 0 < reader.timeouts[0] <= 60
        _assert_shared_resources_untouched(service)

    asyncio.run(scenario())


def test_observer_retains_pre_settlement_lifecycle_until_binding_is_verified(monkeypatch, resolver):
    async def scenario():
        settling = asyncio.Event()
        release = asyncio.Event()
        resolver.side_effect = [
            {"status": "STEERING"},
            {"status": "DELIVERED", "voice_binding": dict(BINDING),
             "response_text": "whole old turn must not be spoken"},
        ]

        async def next_poll(_delay):
            settling.set()
            await release.wait()

        monkeypatch.setattr(voice_step.asyncio, "sleep", next_poll)
        service = _service()
        reader = _Reader([_started(), _delta("fresh"), _complete(17)])
        task = asyncio.create_task(_observe(service, reader))
        try:
            await settling.wait()
            assert not task.done()
            assert not reader.entered.is_set()
            release.set()
            assert await task == "fresh"
            assert resolver.call_count == 2
            assert reader.closed.is_set()
            _assert_shared_resources_untouched(service)
        finally:
            if not task.done():
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task

    asyncio.run(scenario())


@pytest.mark.parametrize("resolved", [
    None, {"status": "FAILED"}, {"status": "CANCELLED"},
    {"status": "DELIVERED", "response_text": "not a proven step"},
    {"status": "UNRESOLVED", "voice_binding": {"turn_id": "turn-1"}},
])
def test_observer_missing_binding_or_terminal_receipt_never_reads_whole_turn(resolver, resolved):
    resolver.return_value = resolved

    async def scenario():
        service, reader = _service(), _Reader([_started()])
        assert await _observe(service, reader) is None
        assert not reader.entered.is_set()
        _assert_shared_resources_untouched(service)

    asyncio.run(scenario())


def test_observer_delivery_identity_mismatch_fails_closed(resolver):
    resolver.side_effect = DeliveryReplyMismatch("wrong injected message")

    async def scenario():
        service, reader = _service(), _Reader([_started()])
        assert await _observe(service, reader) is None
        assert not reader.entered.is_set()
        _assert_shared_resources_untouched(service)

    asyncio.run(scenario())


@pytest.mark.parametrize("missing", ["reader", "store", "delivery_id"])
def test_observer_missing_prerequisite_does_not_resolve_or_submit(resolver, missing):
    async def scenario():
        service, reader, receipt = _service(), _Reader(), dict(RECEIPT)
        if missing == "reader":
            reader = None
        elif missing == "store":
            service._inter_bot_dispatcher = None
        else:
            receipt.pop("delivery_id")
        assert await _observe(service, reader, receipt=receipt) is None
        resolver.assert_not_called()
        _assert_shared_resources_untouched(service)

    asyncio.run(scenario())


@pytest.mark.parametrize("events", [
    [], [_delta("pre-start only", 0)], [_started(), _delta("partial")],
    [_started(), _delta("partial"), _replace(_started(), state="rejected")],
    [_started(), _delta("partial"), _replace(_started(), message_id="other")],
    [_started(), _delta("partial"), _complete(20)],
    [_started(), _complete()],
])
def test_observer_supersession_rejection_or_incomplete_stream_discards_partial(resolver, events):
    async def scenario():
        service, reader = _service(), _Reader(events)
        assert await _observe(service, reader) is None
        assert reader.closed.is_set()
        _assert_shared_resources_untouched(service)

    asyncio.run(scenario())


def test_observer_timeout_closes_iterator_without_aborting_shared_run(resolver):
    async def scenario():
        service = _service()
        reader = _Reader([_started(), _delta("partial")], block=True)
        # An already-expired deadline cancels at the reader's first suspension.
        deadline = asyncio.get_running_loop().time() - 1
        assert await _observe(service, reader, deadline=deadline) is None
        assert reader.entered.is_set()
        assert reader.closed.is_set()
        _assert_shared_resources_untouched(service)

    asyncio.run(scenario())


def test_observer_disconnect_propagates_cancellation_and_closes_iterator(resolver):
    async def scenario():
        service = _service()
        reader = _Reader([_started(), _delta("partial")], block=True)
        task = asyncio.create_task(_observe(service, reader))
        try:
            await reader.entered.wait()
            assert not task.done()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert reader.closed.is_set()
            _assert_shared_resources_untouched(service)
        finally:
            if not task.done():
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task

    asyncio.run(scenario())


def _sse(payload):
    return f"data: {json.dumps(payload)}\n\n"


def _chunks(frames):
    return [json.loads(frame[6:]) for frame in frames
            if frame.startswith("data: ") and frame != DONE]


def _stream_service(receipt, reader=None):
    service = _service()
    service._redis_subscriber.ui_stream = Mock(return_value=reader)
    service.admissions = []

    async def stream(request):
        service.admissions.append(request)
        yield _sse(receipt)
        raise AssertionError("internal receipt trailer must not be read")

    service.chat_completion_stream = stream
    return service


async def _collect(service, request=None, **kwargs):
    return [frame async for frame in adapter.completion_stream(
        service, _request() if request is None else request, **kwargs,
    )]


def test_adapter_waits_for_cursor_claim_to_finish_before_admitting(monkeypatch):
    async def scenario():
        claim_entered, release_claim = asyncio.Event(), asyncio.Event()
        reader = _Reader()

        async def claim():
            claim_entered.set()
            await release_claim.wait()
            reader.claimed = True

        reader.claim.side_effect = claim
        service = _stream_service(dict(RECEIPT), reader)
        observer = AsyncMock(return_value="fresh step")
        whole_turn = AsyncMock(side_effect=AssertionError("busy voice used whole turn"))
        monkeypatch.setattr(voice_step, "observe_voice_step", observer)
        monkeypatch.setattr(adapter, "observe_receipt", whole_turn)
        task = asyncio.create_task(_collect(service))
        try:
            await claim_entered.wait()
            assert not task.done()
            assert service.admissions == []
            observer.assert_not_awaited()
            release_claim.set()
            frames = await task
            assert reader.claimed
            assert len(service.admissions) == 1
            observer.assert_awaited_once()
            args, kwargs = observer.await_args
            assert args == (service, service.admissions[0], RECEIPT, reader)
            assert isinstance(kwargs["deadline"], float)
            assert _chunks(frames)[0]["choices"][0]["delta"]["content"] == "fresh step"
            assert _chunks(frames)[1]["choices"][0]["finish_reason"] == "stop"
            assert frames[-1] == DONE
            whole_turn.assert_not_awaited()
            _assert_shared_resources_untouched(service)
        finally:
            if not task.done():
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task

    asyncio.run(scenario())


@pytest.mark.parametrize("mode", ["none", "comment", "data"])
@pytest.mark.parametrize("steer_capable", [False, True])
def test_adapter_unavailable_busy_voice_warns_then_ends_without_whole_turn(
    monkeypatch, mode, steer_capable,
):
    async def scenario():
        receipt = {**RECEIPT, "steer_capable": steer_capable}
        service = _stream_service(receipt, _Reader())
        observer = AsyncMock(return_value=None)
        whole_turn = AsyncMock(side_effect=AssertionError("busy voice used whole turn"))
        monkeypatch.setattr(voice_step, "observe_voice_step", observer)
        monkeypatch.setattr(adapter, "observe_receipt", whole_turn)
        frames = await _collect(service, extensions=mode)
        warning = {"object": "service.warning", "warnings": ["voice_step_unavailable"]}
        expected = [] if mode == "none" else [
            _sse(warning) if mode == "data" else f"{adapter.EXTENSION_COMMENT}{json.dumps(warning)}\n\n",
        ]
        assert frames == [*expected, DONE]
        assert len(service.admissions) == 1
        if steer_capable:
            observer.assert_awaited_once()
        else:
            observer.assert_not_awaited()
        whole_turn.assert_not_awaited()
        _assert_shared_resources_untouched(service)

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["no_subscriber", "claim_error"])
def test_adapter_reader_failure_still_admits_once_and_fails_closed(monkeypatch, resolver, failure):
    async def scenario():
        reader = _Reader(claim_error=RuntimeError("cursor unavailable"))
        service = _stream_service(dict(RECEIPT), reader)
        if failure == "no_subscriber":
            service._redis_subscriber = None
        whole_turn = AsyncMock(side_effect=AssertionError("busy voice used whole turn"))
        monkeypatch.setattr(adapter, "observe_receipt", whole_turn)
        frames = await _collect(service, extensions="data")
        assert _chunks(frames) == [
            {"object": "service.warning", "warnings": ["voice_step_unavailable"]},
        ]
        assert frames[-1] == DONE
        assert len(service.admissions) == 1
        assert not reader.entered.is_set()
        resolver.assert_not_called()
        whole_turn.assert_not_awaited()
        service.abort.assert_not_awaited()

    asyncio.run(scenario())


@pytest.mark.parametrize("voice,busy", [(True, False), (False, False), (False, True)])
def test_adapter_idle_voice_uses_durable_voice_observer_and_typed_keeps_whole_turn(monkeypatch, voice, busy):
    async def scenario():
        receipt = {**RECEIPT, "busy": busy}
        reader = _Reader()
        service = _stream_service(receipt, reader)
        step_observer = AsyncMock(return_value="ordinary reply")
        whole_turn = AsyncMock(return_value={"status": "DELIVERED", "response_text": "ordinary reply"})
        monkeypatch.setattr(voice_step, "observe_voice_step", step_observer)
        monkeypatch.setattr(adapter, "observe_receipt", whole_turn)
        request = _request(voice=voice)
        frames = await _collect(service, request)
        assert _chunks(frames)[0]["choices"][0]["delta"]["content"] == "ordinary reply"
        assert _chunks(frames)[1]["choices"][0]["finish_reason"] == "stop"
        assert frames[-1] == DONE
        if voice:
            step_observer.assert_awaited_once()
            assert step_observer.await_args.args == (service, request, receipt, reader)
            whole_turn.assert_not_awaited()
        else:
            whole_turn.assert_awaited_once_with(service, request, receipt)
            step_observer.assert_not_awaited()
        assert len(service.admissions) == 1
        assert reader.claim.await_count == int(voice)
        assert service._redis_subscriber.ui_stream.call_count == int(voice)
        _assert_shared_resources_untouched(service)

    asyncio.run(scenario())


@pytest.mark.parametrize("voice", [False, True])
def test_adapter_direct_idle_stream_remains_byte_for_byte_unchanged(monkeypatch, voice):
    async def scenario():
        service, reader = _service(), _Reader()
        service._redis_subscriber.ui_stream = Mock(return_value=reader)
        direct = [": connected\n\n", _sse({
            "id": "direct", "object": "chat.completion.chunk", "model": "test-model",
            "choices": [{"index": 0, "delta": {"content": "live reply"}, "finish_reason": None}],
        }), DONE]

        async def stream(_request):
            for frame in direct:
                yield frame

        service.chat_completion_stream = stream
        whole_turn, step_observer = AsyncMock(), AsyncMock()
        monkeypatch.setattr(adapter, "observe_receipt", whole_turn)
        monkeypatch.setattr(voice_step, "observe_voice_step", step_observer)
        assert await _collect(service, _request(voice=voice)) == direct
        assert reader.claim.await_count == int(voice)
        whole_turn.assert_not_awaited()
        step_observer.assert_not_awaited()
        _assert_shared_resources_untouched(service)

    asyncio.run(scenario())


@pytest.mark.parametrize("explicit_identity", [False, True])
def test_adapter_uses_unique_voice_cursor_without_stealing_ui_or_other_call(monkeypatch, explicit_identity):
    async def scenario():
        service = _stream_service(dict(RECEIPT))
        readers = [_Reader(), _Reader()]
        service._redis_subscriber.ui_stream.side_effect = readers
        monkeypatch.setattr(voice_step, "observe_voice_step", AsyncMock(return_value="fresh"))
        # Each call also allocates a response ID, so cursor IDs use draws 0 and 2.
        monkeypatch.setattr(adapter.uuid, "uuid4", Mock(side_effect=[
            SimpleNamespace(hex=f"{value:032x}") for value in range(4)
        ]))
        request = _request(explicit_identity=explicit_identity)
        await _collect(service, request)
        await _collect(service, request)
        calls = service._redis_subscriber.ui_stream.call_args_list
        expected_identity = ("test-bot", "test-user") if explicit_identity else ("default-bot", "default-user")
        assert [call.args[:2] for call in calls] == [expected_identity, expected_identity]
        cursors = [call.args[2] for call in calls]
        assert cursors == [f"voice-step-{value:032x}" for value in (0, 2)]
        for reader in readers:
            reader.claim.assert_awaited_once()
        assert len(service.admissions) == 2
        _assert_shared_resources_untouched(service)

    asyncio.run(scenario())


@pytest.mark.parametrize("busy", [False, True])
def test_adapter_idle_snapshot_can_durably_become_steer_without_speaking_old_turn(monkeypatch, resolver, busy):
    async def scenario():
        old = "old target's whole response must never be spoken"
        resolver.return_value = {
            "status": "DELIVERED", "voice_binding": dict(BINDING), "response_text": old,
        }
        reader = _Reader([
            _delta(old, 0), _started(), _delta("fresh step"), _tool(22),
            _delta("later autonomous work", 22), _complete(43),
        ])
        service = _stream_service({**RECEIPT, "busy": busy}, reader)
        whole_turn = AsyncMock(side_effect=AssertionError("voice used snapshot whole turn"))
        monkeypatch.setattr(adapter, "observe_receipt", whole_turn)
        frames = await _collect(service, extensions="comment")
        assert _chunks(frames)[0]["choices"][0]["delta"]["content"] == "fresh step"
        assert old not in "".join(frames)
        assert frames[-1] == DONE
        assert reader.claimed and reader.closed.is_set()
        assert len(service.admissions) == 1
        assert resolver.call_args.kwargs["include_voice_binding"] is True
        whole_turn.assert_not_awaited()
        _assert_shared_resources_untouched(service)

    asyncio.run(scenario())


@pytest.mark.parametrize("busy", [False, True])
def test_adapter_busy_snapshot_can_durably_become_direct_and_return_own_full_reply(monkeypatch, resolver, busy):
    async def scenario():
        own_reply = "own first step\n\nplus own final answer"
        resolver.return_value = {
            "status": "DELIVERED", "voice_direct_reply": True, "response_text": own_reply,
        }
        reader = _Reader([_delta("foreign running turn", 0)])
        service = _stream_service({**RECEIPT, "busy": busy}, reader)
        whole_turn = AsyncMock(side_effect=AssertionError("voice used whole-turn observer"))
        monkeypatch.setattr(adapter, "observe_receipt", whole_turn)
        frames = await _collect(service)
        assert _chunks(frames)[0]["choices"][0]["delta"]["content"] == own_reply
        assert _chunks(frames)[1]["choices"][0]["finish_reason"] == "stop"
        assert frames[-1] == DONE
        assert reader.claimed and not reader.entered.is_set()
        assert len(service.admissions) == 1
        whole_turn.assert_not_awaited()
        _assert_shared_resources_untouched(service)

    asyncio.run(scenario())


@pytest.mark.parametrize("marker", [None, False, 1, "true"])
@pytest.mark.parametrize("busy", [False, True])
def test_adapter_unmarked_delivered_voice_never_leaks_full_response(monkeypatch, resolver, marker, busy):
    async def scenario():
        resolved = {"status": "DELIVERED", "response_text": "unproven old full answer"}
        if marker is not None:
            resolved["voice_direct_reply"] = marker
        resolver.return_value = resolved
        service = _stream_service({**RECEIPT, "busy": busy}, _Reader())
        whole_turn = AsyncMock(side_effect=AssertionError("voice used whole-turn fallback"))
        monkeypatch.setattr(adapter, "observe_receipt", whole_turn)
        frames = await _collect(service, extensions="data")
        assert frames == [_sse({
            "object": "service.warning", "warnings": ["voice_step_unavailable"],
        }), DONE]
        assert len(service.admissions) == 1
        whole_turn.assert_not_awaited()
        _assert_shared_resources_untouched(service)

    asyncio.run(scenario())


def test_idle_voice_verified_direct_reply_does_not_require_reader(resolver):
    async def scenario():
        service = _service()
        resolver.return_value = {
            "status": "DELIVERED", "voice_direct_reply": True, "response_text": "own full reply",
        }
        assert await _observe(service, None, receipt={**RECEIPT, "busy": False}) == "own full reply"
        assert resolver.call_args.kwargs["include_voice_binding"] is True
        _assert_shared_resources_untouched(service)

    asyncio.run(scenario())


@pytest.mark.parametrize("busy,expected_timeout", [(True, 60.0), (False, 300.0)])
def test_adapter_deadline_is_60_seconds_for_busy_and_300_for_own_idle_reply(monkeypatch, busy, expected_timeout):
    async def scenario():
        base_time = 1000.0
        deadlines = []

        @asynccontextmanager
        async def no_real_timer(deadline):
            deadlines.append(deadline)
            yield

        monkeypatch.setattr(adapter.asyncio, "get_running_loop", lambda: SimpleNamespace(time=lambda: base_time))
        monkeypatch.setattr(adapter.asyncio, "timeout_at", no_real_timer)
        observer = AsyncMock(return_value="own reply")
        monkeypatch.setattr(voice_step, "observe_voice_step", observer)
        service = _stream_service({**RECEIPT, "busy": busy}, _Reader())
        assert (await _collect(service))[-1] == DONE
        # Claim is always bounded by the short pre-admission window; idle own
        # reply observation keeps the ordinary five-minute completion budget.
        assert deadlines == [base_time + 60.0]
        assert observer.await_args.kwargs["deadline"] == base_time + expected_timeout
        assert len(service.admissions) == 1
        _assert_shared_resources_untouched(service)

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["resolver", "reader"])
def test_adapter_database_or_reader_failure_is_unavailable_not_failed_work(monkeypatch, resolver, failure):
    async def scenario():
        reader = _Reader()
        if failure == "resolver":
            resolver.side_effect = RuntimeError("database observation failed")
        else:
            async def broken_events(*, timeout_s):
                reader.entered.set()
                try:
                    yield _started()
                    yield _delta("partial")
                    raise OSError("reader observation failed")
                finally:
                    reader.closed.set()

            reader.events = broken_events
        service = _stream_service(dict(RECEIPT), reader)
        whole_turn = AsyncMock(side_effect=AssertionError("voice used whole-turn fallback"))
        monkeypatch.setattr(adapter, "observe_receipt", whole_turn)
        frames = await _collect(service, extensions="data")
        assert frames == [_sse({
            "object": "service.warning", "warnings": ["voice_step_unavailable"],
        }), DONE]
        assert len(service.admissions) == 1
        if failure == "reader":
            assert reader.entered.is_set() and reader.closed.is_set()
        else:
            assert not reader.entered.is_set()
        whole_turn.assert_not_awaited()
        _assert_shared_resources_untouched(service)

    asyncio.run(scenario())


def test_adapter_keepalive_is_15_seconds_and_disconnect_cancels_and_joins_observer(monkeypatch):
    async def scenario():
        entered, cleanup_started, release_cleanup, finished = (asyncio.Event() for _ in range(4))
        never = asyncio.Event()
        waits = []
        real_wait = asyncio.wait

        async def observer(*_args, **_kwargs):
            entered.set()
            try:
                await never.wait()
            finally:
                cleanup_started.set()
                await release_cleanup.wait()
                finished.set()

        async def force_keepalive(tasks, *, timeout):
            waits.append(timeout)
            await entered.wait()
            if len(waits) == 1:
                return set(), set(tasks)  # Deterministic idle interval, no clock wait.
            return await real_wait(tasks)

        monkeypatch.setattr(voice_step, "observe_voice_step", observer)
        monkeypatch.setattr(adapter.asyncio, "wait", force_keepalive)
        service = _stream_service(dict(RECEIPT), _Reader())
        whole_turn = AsyncMock(side_effect=AssertionError("voice used whole-turn fallback"))
        monkeypatch.setattr(adapter, "observe_receipt", whole_turn)
        stream = adapter.completion_stream(service, _request())
        close_task = None
        try:
            assert await anext(stream) == ": waiting for voice reply\n\n"
            assert waits == [15.0]
            assert not finished.is_set()
            close_task = asyncio.create_task(stream.aclose())
            await cleanup_started.wait()
            assert not close_task.done()  # Closing must join, not just cancel.
            release_cleanup.set()
            await close_task
            assert finished.is_set()
            assert len(service.admissions) == 1
            whole_turn.assert_not_awaited()
            _assert_shared_resources_untouched(service)
        finally:
            release_cleanup.set()
            if close_task is not None:
                await close_task
            else:
                await stream.aclose()

    asyncio.run(scenario())


def test_observer_superseded_while_waiting_does_not_return_partial_text(resolver):
    async def scenario():
        service = _service()
        reader = _Reader([_started(), _delta("partial")], block=True)
        task = asyncio.create_task(_observe(service, reader))
        try:
            await reader.entered.wait()
            assert not task.done()
            # UiStreamReader ends iteration when another owner supersedes it.
            reader.release.set()
            assert await task is None
            assert reader.superseded and reader.closed.is_set()
            _assert_shared_resources_untouched(service)
        finally:
            if not task.done():
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task

    asyncio.run(scenario())


def test_observer_timeout_while_waiting_for_durable_binding_never_opens_reader(monkeypatch, resolver):
    async def scenario():
        service, reader = _service(), _Reader([_started()])
        resolver.return_value = {"status": "STEERING"}
        poll_entered = asyncio.Event()
        never = asyncio.Event()

        async def wait_for_next_poll(_delay):
            poll_entered.set()
            await never.wait()

        monkeypatch.setattr(voice_step.asyncio, "sleep", wait_for_next_poll)
        deadline = asyncio.get_running_loop().time() - 1
        assert await _observe(service, reader, deadline=deadline) is None
        assert poll_entered.is_set()
        assert not reader.entered.is_set()
        _assert_shared_resources_untouched(service)

    asyncio.run(scenario())


def test_adapter_claim_timeout_still_admits_once_without_whole_turn_fallback(monkeypatch, resolver):
    async def scenario():
        reader = _Reader()
        claim_entered, never = asyncio.Event(), asyncio.Event()

        async def claim():
            claim_entered.set()
            await never.wait()

        reader.claim.side_effect = claim
        service = _stream_service(dict(RECEIPT), reader)
        monkeypatch.setattr(voice_step, "VOICE_STEP_TIMEOUT_SECONDS", -1)
        whole_turn = AsyncMock(side_effect=AssertionError("busy voice used whole turn"))
        monkeypatch.setattr(adapter, "observe_receipt", whole_turn)
        frames = await _collect(service, extensions="comment")
        assert claim_entered.is_set()
        assert len(service.admissions) == 1
        assert frames == [
            f'{adapter.EXTENSION_COMMENT}{{"object": "service.warning", '
            '"warnings": ["voice_step_unavailable"]}\n\n', DONE,
        ]
        assert not reader.entered.is_set()
        resolver.assert_not_called()
        whole_turn.assert_not_awaited()
        _assert_shared_resources_untouched(service)

    asyncio.run(scenario())


def test_adapter_disconnect_during_step_closes_reader_and_leaves_run_alive(monkeypatch, resolver):
    async def scenario():
        reader = _Reader([_started(), _delta("partial")], block=True)
        service = _stream_service(dict(RECEIPT), reader)
        whole_turn = AsyncMock(side_effect=AssertionError("busy voice used whole turn"))
        monkeypatch.setattr(adapter, "observe_receipt", whole_turn)
        stream = adapter.completion_stream(service, _request())
        task = asyncio.create_task(anext(stream))
        try:
            await reader.entered.wait()
            assert not task.done()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert reader.closed.is_set()
            assert len(service.admissions) == 1
            whole_turn.assert_not_awaited()
            _assert_shared_resources_untouched(service)
        finally:
            if not task.done():
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            await stream.aclose()

    asyncio.run(scenario())


def test_adapter_busy_voice_end_to_end_uses_only_post_barrier_step(monkeypatch, resolver):
    async def scenario():
        reader = _Reader([
            _delta("whole old turn answer", 0), _started(), _delta("fresh step"), _tool(22),
            _delta("later autonomous work", 22), _complete(43),
        ])
        service = _stream_service(dict(RECEIPT), reader)
        whole_turn = AsyncMock(side_effect=AssertionError("busy voice used whole turn"))
        monkeypatch.setattr(adapter, "observe_receipt", whole_turn)
        frames = await _collect(service, admission_comment=True, extensions="comment")
        assert frames[0] == f": llm-bawt-admission {json.dumps(RECEIPT)}\n\n"
        assert _chunks(frames)[0]["choices"][0]["delta"]["content"] == "fresh step"
        assert frames[-1] == DONE
        assert reader.claimed and reader.closed.is_set()
        assert len(service.admissions) == 1
        whole_turn.assert_not_awaited()
        _assert_shared_resources_untouched(service)

    asyncio.run(scenario())
