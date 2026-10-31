"""TASK-1344: a steer that lands after a run's final result is refused.

Once ``messages()`` hands the send handler the ResultMessage that ends the
run, the handler finalizes and tears down. A ``next`` steer written in that
window was consumed by the CLI as a stray follow-up the user never saw, then
published ``uncertain`` at teardown. The run now refuses such a steer with the
definite :data:`STEER_RUN_FINISHING` code, so the app sends it as a fresh turn.

A steer written before that point is reconciled from the lifecycle frames the
SDK reader buffered but the consumer never took, before teardown settles it.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import anyio
import pytest
from claude_agent_sdk import AssistantMessage, ResultMessage, UserMessage
from claude_agent_sdk._internal.message_parser import parse_message
from claude_agent_sdk._internal.query import Query

from agent_bridge.steer_contract import STEER_RUN_FINISHING
from claude_code_bridge.active_run import ClaudeActiveRun
from tests.test_claude_active_run_next_steer import (
    BOT,
    CMD,
    _FakeClient,
    _finish,
    _lifecycle,
    _next,
    _result,
    _results,
    _run,
    _start,
)


async def _consume_until_result(run: ClaudeActiveRun) -> list:
    """Consume like the send handler: drain interrupt boundaries, stop at the terminal result."""
    yielded: list = []
    async for msg in run.messages():
        yielded.append(msg)
        if isinstance(msg, ResultMessage):
            if run.consume_replaced_result(msg):
                continue
            break
    return yielded


@pytest.mark.anyio
async def test_next_steer_after_the_terminal_result_is_refused_as_run_finishing():
    client = _FakeClient()
    run = _run(client)
    consumer = asyncio.create_task(_consume_until_result(run))
    await asyncio.sleep(0)
    await client._query.frames.put(_result("final answer"))
    yielded = await asyncio.wait_for(consumer, 1)
    assert _results(yielded) == ["final answer"]

    # Teardown has not started: the run is still registered and not completed.
    with pytest.raises(RuntimeError, match=f"^{STEER_RUN_FINISHING}$"):
        await _next(run, "omg add indexes", CMD)

    assert client.written == []
    assert not run.has_unstarted_command()


@pytest.mark.anyio
async def test_now_steer_after_the_terminal_result_is_refused_without_interrupting():
    client = _FakeClient()
    run = _run(client)
    consumer = asyncio.create_task(_consume_until_result(run))
    await asyncio.sleep(0)
    await client._query.frames.put(_result("final answer"))
    await asyncio.wait_for(consumer, 1)

    with pytest.raises(RuntimeError, match=f"^{STEER_RUN_FINISHING}$"):
        await run.steer("stop", priority="now", origin="user", framed_message="NOW[stop]")

    client.interrupt.assert_not_awaited()
    assert client.replacements == []


@pytest.mark.anyio
async def test_completed_run_still_reports_no_active_run():
    """After teardown begins the run is gone, which callers already handle."""
    client = _FakeClient()
    run = _run(client)
    consumer = asyncio.create_task(_consume_until_result(run))
    await asyncio.sleep(0)
    await client._query.frames.put(_result("final answer"))
    await asyncio.wait_for(consumer, 1)
    run.mark_completed()

    with pytest.raises(RuntimeError, match="^no_active_run$"):
        await _next(run, "late", CMD)


@pytest.mark.anyio
async def test_interrupt_boundary_result_does_not_close_steering():
    client = _FakeClient()
    run = _run(client)
    consumer = asyncio.create_task(_consume_until_result(run))
    await asyncio.sleep(0)

    await run.steer("redirect", priority="now", origin="user", framed_message="NOW[redirect]")
    # The interrupted request's result is a boundary, not the run's end.
    await client._query.frames.put(_result("interrupted"))
    await asyncio.sleep(0.01)
    assert not consumer.done()

    await _next(run, "still live", CMD, origin="bot")
    assert [frame["uuid"] for frame in client.written] == [CMD]
    for frame in (_lifecycle(CMD, "started"), _lifecycle(CMD, "completed"), _result("replacement")):
        await client._query.frames.put(frame)
    yielded = await asyncio.wait_for(consumer, 1)
    assert _results(yielded) == ["interrupted", "replacement"]


@pytest.mark.anyio
async def test_steer_written_before_the_result_is_held_then_steering_closes():
    """A steer that beats the result keeps today's behavior: the run waits for it."""
    client = _FakeClient()
    run = _run(client)
    consumer = asyncio.create_task(_consume_until_result(run))
    await asyncio.sleep(0)

    await _next(run, "late note", BOT, origin="bot")
    await client._query.frames.put(_result("first turn"))
    await asyncio.sleep(0.01)
    assert not consumer.done()
    # Held for the queued command: the run is not finishing yet.
    await _next(run, "another", CMD, origin="user")

    for frame in (_lifecycle(BOT, "started"), _lifecycle(CMD, "started"),
                  _lifecycle(BOT, "completed"), _result("follow-up"),
                  _lifecycle(CMD, "completed")):
        await client._query.frames.put(frame)
    yielded = await asyncio.wait_for(consumer, 1)
    assert _results(yielded) == ["follow-up"]

    with pytest.raises(RuntimeError, match=f"^{STEER_RUN_FINISHING}$"):
        await _next(run, "too late", "736a3206-0000-4000-8000-000000000003")


@pytest.mark.anyio
async def test_replay_of_an_accepted_command_keeps_its_answer_after_the_result():
    """A retried RPC for a command the CLI already ran must not become a fresh turn."""
    client = _FakeClient()
    run = _run(client)
    consumer = asyncio.create_task(_consume_until_result(run))
    await asyncio.sleep(0)
    await _next(run, "fold me", CMD, origin="bot")
    for frame in (_lifecycle(CMD, "started"), _lifecycle(CMD, "completed"), _result("final")):
        await client._query.frames.put(frame)
    await asyncio.wait_for(consumer, 1)

    await _next(run, "fold me", CMD, origin="bot")  # accepted, not refused

    assert [frame["uuid"] for frame in client.written] == [CMD]


@pytest.mark.anyio
async def test_lifecycle_less_sdk_closes_steering_at_its_result_too():
    """Without raw frames ``next`` is already refused; ``now`` must not interrupt a finished run."""

    class _Plain:
        def __init__(self) -> None:
            self.interrupted = False

        async def receive_messages(self):
            yield ResultMessage(
                subtype="success", duration_ms=1, duration_api_ms=1, is_error=False,
                num_turns=1, session_id="s", result="done",
            )

        async def interrupt(self):
            self.interrupted = True

    client = _Plain()
    run = ClaudeActiveRun(client=client, request_id="req-1")
    await _consume_until_result(run)

    with pytest.raises(RuntimeError, match=f"^{STEER_RUN_FINISHING}$"):
        await run.steer("stop", priority="now", origin="user", framed_message="NOW[stop]")
    assert client.interrupted is False


@pytest.mark.anyio
@pytest.mark.parametrize("raw_reader", [False, True], ids=["sdk-reader", "raw-reader"])
@pytest.mark.parametrize("stale_kind", ["task-notification", "hook-response"])
@pytest.mark.parametrize("own_origin", [None, {"kind": "human"}], ids=["unmarked-prompt", "human-prompt"])
async def test_nonhuman_result_does_not_finish_the_current_prompt(
    raw_reader, stale_kind, own_origin,
):
    """A background completion may arrive before the SDK's own prompt starts."""
    stale = {**_result(""), "origin": {"kind": stale_kind}}
    own_user = {
        "type": "user", "session_id": "s", "origin": own_origin,
        "message": {"role": "user", "content": "the current prompt"},
    }
    own_assistant = {
        "type": "assistant", "session_id": "s",
        "message": {"role": "assistant", "model": "test-model", "content": [
            {"type": "text", "text": "working on the current prompt"},
            {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "pwd"}},
        ]},
    }
    tool_response = {
        "type": "user", "session_id": "s",
        "message": {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "done"},
        ]},
    }
    frames = [
        {**own_user, "origin": {"kind": stale_kind}}, stale,
        own_user, own_assistant, tool_response,
        {**_result("the actual answer"), "origin": own_origin},
    ]

    if raw_reader:
        client = _FakeClient()
        for frame in frames:
            client._query.frames.put_nowait(frame)
    else:
        class _Plain:
            async def receive_messages(self):
                for frame in frames:
                    yield parse_message(frame)

        client = _Plain()
    run = ClaudeActiveRun(client=client, request_id="current-prompt")
    yielded = await _consume_until_result(run)

    assert _results(yielded) == ["the actual answer"]
    assert any(isinstance(msg, AssistantMessage) for msg in yielded)
    assert any(isinstance(msg, UserMessage) and msg.content == "the current prompt" for msg in yielded)
    assert any(isinstance(msg, UserMessage) and isinstance(msg.content, list) for msg in yielded)
    assert run._finishing is True


@pytest.mark.anyio
async def test_stale_result_does_not_consume_interrupt_boundary_or_queued_steer():
    client = _FakeClient()
    run = _run(client)
    consumer = asyncio.create_task(_consume_until_result(run))
    await asyncio.sleep(0)  # enable lifecycle support before writing the queued steer
    run._expected_interrupt_results = 1
    await _next(run, "follow-up", CMD, origin="bot")
    for frame in (
        {**_result(""), "origin": {"kind": "task-notification"}},
        _result("interrupted"),
        _lifecycle(CMD, "started"),
        _lifecycle(CMD, "completed"),
        _result("replacement answer"),
    ):
        client._query.frames.put_nowait(frame)
    yielded = await asyncio.wait_for(consumer, 1)

    assert _results(yielded) == ["interrupted", "replacement answer"]
    assert not run.has_unstarted_command()
    assert run._expected_interrupt_results == 0
    assert run._deferred_result is None


@pytest.mark.anyio
async def test_stale_result_cannot_be_released_by_a_rejected_queued_steer():
    client = _FakeClient()
    run = _run(client)
    consumer = asyncio.create_task(_consume_until_result(run))
    await asyncio.sleep(0)
    await _next(run, "queued note", CMD, origin="bot")
    for frame in (
        {**_result(""), "origin": {"kind": "task-notification"}},
        _lifecycle(CMD, "discarded"),
        _result("own answer"),
    ):
        client._query.frames.put_nowait(frame)

    yielded = await asyncio.wait_for(consumer, 1)
    assert _results(yielded) == ["own answer"]
    assert run._deferred_result is None
    assert run._finishing is True


# --- reconciling buffered lifecycle frames before teardown settles -----------


class _StreamQuery(Query):
    """The SDK's real buffer: an anyio memory stream fed by the stdout reader."""

    def __init__(self) -> None:  # noqa: D401 - deliberately skip SDK init
        self._message_send, self._message_receive = anyio.create_memory_object_stream(100)

    async def receive_messages(self):
        async for frame in self._message_receive:
            if frame.get("type") == "end":
                return
            yield frame

    def cli_writes(self, *frames: dict) -> None:
        """The reader buffers frames the consumer may never take."""
        for frame in frames:
            self._message_send.send_nowait(frame)

    def disconnect(self) -> None:
        """``ClaudeSDKClient.disconnect``: close the send side, then the receive side."""
        self._message_send.close()
        self._message_receive.close()


class _StreamClient:
    def __init__(self) -> None:
        self._query = _StreamQuery()
        self.written: list[dict] = []

    async def query(self, prompt):
        async for frame in prompt:
            self.written.append(frame)
            self._query.cli_writes(_lifecycle(frame["uuid"], "queued"))

    async def interrupt(self):  # pragma: no cover - not used
        raise AssertionError("unexpected interrupt")


async def _stopped_consumer(run: ClaudeActiveRun) -> asyncio.Task:
    """A consumer that stops reading (abort/error) while the CLI keeps writing."""
    task = asyncio.create_task(_consume_until_result(run))
    await asyncio.sleep(0)
    return task


@pytest.mark.anyio
async def test_started_frame_buffered_after_the_consumer_stopped_is_not_uncertain():
    client = _StreamClient()
    events: list[dict] = []
    run = _run(client, events)
    consumer = await _stopped_consumer(run)
    await _next(run, "note", CMD, origin="user")
    await asyncio.sleep(0.01)
    consumer.cancel()  # chat.abort: the send handler stops consuming
    await asyncio.gather(consumer, return_exceptions=True)

    # The CLI ran it; the reader buffered the frames, nobody took them.
    client._query.cli_writes(_lifecycle(CMD, "started"), _result("unseen"))
    run.mark_completed()
    client._query.disconnect()  # the run's teardown

    assert run.release_unstarted(SimpleNamespace(exited=True)) == []
    assert [(e["message_id"], e["state"]) for e in events] == [(CMD, "queued"), (CMD, "started")]
    assert not any(e["state"] == "uncertain" for e in events)


@pytest.mark.anyio
async def test_cli_terminal_frame_buffered_before_teardown_is_a_proven_rejection():
    client = _StreamClient()
    events: list[dict] = []
    run = _run(client, events)
    consumer = await _stopped_consumer(run)
    await _next(run, "note", BOT, origin="bot", rpc="steer_delivery_d1_g0")
    await asyncio.sleep(0.01)
    consumer.cancel()
    await asyncio.gather(consumer, return_exceptions=True)

    client._query.cli_writes(_lifecycle(BOT, "discarded"))
    run.mark_completed()
    client._query.disconnect()

    assert run.release_unstarted(SimpleNamespace(exited=True)) == []
    assert events[-1]["state"] == "discarded" and events[-1]["proof"] == "cli_terminal"


@pytest.mark.anyio
async def test_command_with_no_buffered_evidence_is_still_uncertain():
    client = _StreamClient()
    events: list[dict] = []
    run = _run(client, events)
    consumer = await _stopped_consumer(run)
    await _next(run, "note", CMD, origin="user")
    await asyncio.sleep(0.01)
    consumer.cancel()
    await asyncio.gather(consumer, return_exceptions=True)

    run.mark_completed()
    client._query.disconnect()

    assert run.release_unstarted(SimpleNamespace(exited=False)) == [CMD]
    assert (events[-1]["state"], events[-1]["proof"]) == ("uncertain", "teardown")


@pytest.mark.anyio
async def test_release_without_a_capture_still_settles_from_known_state():
    """The fake SDK used elsewhere has no anyio buffer: reconciliation is a no-op."""
    client = _FakeClient()
    events: list[dict] = []
    run = _run(client, events)
    _, task = await _start(run)
    await _next(run, "note", CMD, origin="bot")
    await _finish(client, task)
    run.mark_completed()

    assert run.release_unstarted(SimpleNamespace(exited=True)) == [CMD]
