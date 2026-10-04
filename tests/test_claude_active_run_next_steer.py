"""TASK-934: bot-originated steers deliver at the next tool boundary.

Frame sequences mirror live Claude Code CLI 2.1.280 probes:

* fold — a ``next`` message injected while Bash runs is ``queued``, then
  ``started`` right after the tool_result and ``completed`` before the single
  ResultMessage.
* late — a ``next`` message injected while the final answer streams stays
  ``queued`` past the first ResultMessage, then ``started`` runs a follow-up
  turn that ends in a second ResultMessage.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest
from claude_agent_sdk import ResultMessage, UserMessage
from claude_agent_sdk._internal.query import Query

from claude_code_bridge.active_run import (
    STEER_INTERRUPTED_TOOL_RESULT,
    ClaudeActiveRun,
)

CMD = "736a3206-0000-4000-8000-000000000001"


class _FakeQuery(Query):
    """Raw-frame source that satisfies the SDK ``Query`` isinstance check."""

    def __init__(self) -> None:  # noqa: D401 - deliberately skip SDK init
        self.frames: asyncio.Queue = asyncio.Queue()
        self.receipt = {"still_queued": [], "cancelled": [CMD]}
        self.control_requests: list[dict] = []

    async def _send_control_request(self, request, timeout=60):
        self.control_requests.append(request)
        return self.receipt

    async def receive_messages(self):
        while True:
            frame = await self.frames.get()
            if frame is None:
                return
            yield frame


class _FakeClient:
    def __init__(self) -> None:
        self._query = _FakeQuery()
        self.written: list[dict] = []
        self.replacements: list[str] = []
        self.interrupt = AsyncMock()
        self.disconnect = AsyncMock()

    async def query(self, prompt):
        if isinstance(prompt, str):
            self.replacements.append(prompt)
            return
        async for frame in prompt:
            self.written.append(frame)
            # The CLI acknowledges uuid-tagged stdin commands immediately.
            await self._query.frames.put(_lifecycle(frame["uuid"], "queued"))


def _lifecycle(command_uuid: str, state: str) -> dict:
    return {"type": "command_lifecycle", "command_uuid": command_uuid, "state": state}


def _result(text: str) -> dict:
    return {
        "type": "result", "subtype": "success", "duration_ms": 1,
        "duration_api_ms": 1, "is_error": False, "num_turns": 1,
        "session_id": "s", "result": text,
    }


def _tool_result(content: str) -> dict:
    return {"type": "user", "message": {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "t1", "content": content, "is_error": True}
    ]}}


async def _start(run: ClaudeActiveRun):
    yielded: list = []

    async def consume():
        async for msg in run.messages():
            yielded.append(msg)

    task = asyncio.create_task(consume())
    await asyncio.sleep(0)  # let messages() detect lifecycle support
    return yielded, task


async def _finish(client: _FakeClient, task: asyncio.Task) -> None:
    await client._query.frames.put(None)
    await asyncio.wait_for(task, 1)


def _results(yielded: list) -> list[str]:
    return [m.result for m in yielded if isinstance(m, ResultMessage)]


@pytest.mark.anyio
async def test_next_steer_folds_at_tool_boundary_without_interrupt():
    client = _FakeClient()
    run = ClaudeActiveRun(client=client, request_id="req-1")
    yielded, task = await _start(run)

    await run.steer("Message from bot 'al': reply", priority="next", message_id=CMD)

    client.interrupt.assert_not_awaited()
    [frame] = client.written
    assert frame["priority"] == "next" and frame["uuid"] == CMD
    assert "reply" in frame["message"]["content"]
    for f in (_tool_result("slept"), _lifecycle(CMD, "started"),
              _lifecycle(CMD, "completed"), _result("final")):
        await client._query.frames.put(f)
    await _finish(client, task)

    assert _results(yielded) == ["final"]
    # Tool result passes through untouched (not an interrupt).
    [user] = [m for m in yielded if isinstance(m, UserMessage)]
    assert user.content[0].content == "slept"


@pytest.mark.anyio
async def test_late_next_steer_holds_result_until_follow_up_turn_ends():
    client = _FakeClient()
    run = ClaudeActiveRun(client=client, request_id="req-1")
    yielded, task = await _start(run)

    await run.steer("late note", priority="next", message_id=CMD)
    await client._query.frames.put(_result("first turn"))
    await asyncio.sleep(0.01)
    assert _results(yielded) == []  # held: the queued command has not run

    for f in (_lifecycle(CMD, "started"), _result("follow-up turn"),
              _lifecycle(CMD, "completed")):
        await client._query.frames.put(f)
    await _finish(client, task)

    assert _results(yielded) == ["follow-up turn"]


@pytest.mark.anyio
async def test_rejected_queued_command_releases_held_result():
    client = _FakeClient()
    run = ClaudeActiveRun(client=client, request_id="req-1")
    yielded, task = await _start(run)

    await run.steer("late note", priority="next", message_id=CMD)
    await client._query.frames.put(_result("only turn"))
    await client._query.frames.put(_lifecycle(CMD, "discarded"))
    await _finish(client, task)

    assert _results(yielded) == ["only turn"]


@pytest.mark.anyio
async def test_next_steer_refused_without_lifecycle_support():
    client = AsyncMock()  # not a real SDK client: no raw frame access
    run = ClaudeActiveRun(client=client, request_id="req-1")

    with pytest.raises(RuntimeError, match="next_unsupported"):
        await run.steer("note", priority="next", message_id=CMD)

    client.query.assert_not_awaited()
    client.interrupt.assert_not_awaited()


@pytest.mark.anyio
async def test_next_steer_after_completion_is_no_active_run():
    client = _FakeClient()
    run = ClaudeActiveRun(client=client, request_id="req-1")
    yielded, task = await _start(run)
    run.mark_completed()

    with pytest.raises(RuntimeError, match="no_active_run"):
        await run.steer("note", priority="next", message_id=CMD)

    assert client.written == []
    await _finish(client, task)


@pytest.mark.anyio
async def test_now_steer_still_interrupts_and_relabels_killed_tools():
    client = _FakeClient()
    run = ClaudeActiveRun(client=client, request_id="req-1")
    yielded, task = await _start(run)
    client.query = AsyncMock()

    await run.steer("stop and do X")  # default priority: now

    client.interrupt.assert_awaited_once_with()
    await client._query.frames.put(_tool_result(
        "The user doesn't want to proceed with this tool use. The tool use was rejected"
    ))
    await client._query.frames.put(_result("interrupted"))
    await _finish(client, task)

    [user] = [m for m in yielded if isinstance(m, UserMessage)]
    assert user.content[0].content == STEER_INTERRUPTED_TOOL_RESULT
    # The interrupt boundary result passes through for the consumer to drain.
    assert _results(yielded) == ["interrupted"]
    assert run.consume_replaced_result(yielded[-1]) is True


@pytest.mark.anyio
async def test_user_next_lifecycle_reports_only_owned_uuid_and_no_duplicate_frames():
    client = _FakeClient()
    events: list[tuple[str, str]] = []
    run = ClaudeActiveRun(client=client, request_id="req-1", on_user_lifecycle=lambda *e: events.append(e))
    yielded, task = await _start(run)

    await run.steer("user note", priority="next", message_id=CMD, origin="user")
    await run.steer("user note", priority="next", message_id=CMD, origin="user")
    assert len(client.written) == 1
    for frame in (
        _lifecycle("12345678-0000-4000-8000-000000000000", "queued"),
        _lifecycle(CMD, "queued"),
        _lifecycle(CMD, "started"),
        _lifecycle(CMD, "started"),
        _lifecycle(CMD, "completed"),
        _result("final"),
    ):
        await client._query.frames.put(frame)
    await _finish(client, task)

    assert events == [(CMD, "queued"), (CMD, "started"), (CMD, "completed")]
    assert _results(yielded) == ["final"]
    client.interrupt.assert_not_awaited()


@pytest.mark.anyio
async def test_user_late_completion_is_published_before_result_reaches_send_handler():
    client = _FakeClient()
    events: list[tuple[str, str]] = []
    run = ClaudeActiveRun(client=client, request_id="req-1", on_user_lifecycle=lambda *e: events.append(e))
    stream = run.messages()
    pending = asyncio.create_task(stream.__anext__())
    await asyncio.sleep(0)
    await run.steer("late user note", priority="next", message_id=CMD, origin="user")
    await client._query.frames.put(_result("first turn"))
    await client._query.frames.put(_lifecycle(CMD, "started"))
    await client._query.frames.put(_result("follow-up"))
    await asyncio.sleep(0.01)
    assert not pending.done()
    await client._query.frames.put(_lifecycle(CMD, "completed"))
    result = await asyncio.wait_for(pending, 1)
    assert result.result == "follow-up"
    assert events == [(CMD, "queued"), (CMD, "started"), (CMD, "completed")]
    await stream.aclose()


@pytest.mark.anyio
async def test_bot_origin_next_has_no_user_lifecycle_and_rejected_user_id_is_not_queued():
    client = _FakeClient()
    events: list[tuple[str, str]] = []
    run = ClaudeActiveRun(client=client, request_id="req-1", on_user_lifecycle=lambda *e: events.append(e))
    yielded, task = await _start(run)
    with pytest.raises(ValueError, match="UUID message_id"):
        await run.steer("user note", priority="next", message_id="not-a-uuid", origin="user")
    await run.steer("bot note", priority="next", message_id=CMD, origin="bot")
    await client._query.frames.put(_lifecycle(CMD, "started"))
    await client._query.frames.put(_lifecycle(CMD, "completed"))
    await _finish(client, task)
    assert len(client.written) == 1
    assert events == []
    assert yielded == []


@pytest.mark.anyio
async def test_escalation_atomically_replaces_queued_message_once():
    client = _FakeClient()
    await client._query.frames.put({
        "type": "system", "subtype": "init", "capabilities": ["interrupt_cancel_queued_v1"],
    })
    events: list[tuple[str, str]] = []
    run = ClaudeActiveRun(client=client, request_id="req-1", on_user_lifecycle=lambda *e: events.append(e))
    yielded, task = await _start(run)
    await run.steer("urgent direction", priority="next", message_id=CMD, origin="user")
    await client._query.frames.put(_result("old turn"))
    await asyncio.sleep(0.01)
    assert _results(yielded) == []
    assert await run.escalate(CMD) == "escalated"
    assert await run.escalate(CMD) == "escalated"
    assert client._query.control_requests == [{"subtype": "interrupt", "cancel_queued": True}]
    assert len(client.replacements) == 1
    assert "urgent direction" in client.replacements[0]
    await client._query.frames.put(_lifecycle(CMD, "cancelled"))
    await client._query.frames.put(_result("replacement"))
    await _finish(client, task)
    assert _results(yielded) == ["replacement"]
    assert run.consume_replaced_result(yielded[-1]) is False
    assert events == [(CMD, "queued"), (CMD, "started"), (CMD, "completed")]


@pytest.mark.anyio
@pytest.mark.parametrize("state", ["started", "completed"])
async def test_escalation_does_not_interrupt_if_command_already_started(state):
    client = _FakeClient()
    await client._query.frames.put({
        "type": "system", "subtype": "init", "capabilities": ["interrupt_cancel_queued_v1"],
    })
    run = ClaudeActiveRun(client=client, request_id="req-1")
    _, task = await _start(run)
    await run.steer("user note", priority="next", message_id=CMD, origin="user")
    run.note_lifecycle(CMD, state)
    with pytest.raises(RuntimeError, match="steer_already_started"):
        await run.escalate(CMD)
    assert client._query.control_requests == []
    assert client.replacements == []
    await _finish(client, task)


@pytest.mark.anyio
async def test_escalation_refuses_missing_capability_or_other_queued_messages():
    client = _FakeClient()
    run = ClaudeActiveRun(client=client, request_id="req-1")
    _, task = await _start(run)
    await run.steer("user note", priority="next", message_id=CMD, origin="user")
    with pytest.raises(RuntimeError, match="escalation_unsupported_cli"):
        await run.escalate(CMD)
    run._cli_capabilities.add("interrupt_cancel_queued_v1")
    other = "736a3206-0000-4000-8000-000000000002"
    await run.steer("bot note", priority="next", message_id=other, origin="bot")
    with pytest.raises(RuntimeError, match="escalation_other_queued_commands"):
        await run.escalate(CMD)
    assert client._query.control_requests == []
    await _finish(client, task)


@pytest.mark.anyio
async def test_escalation_missing_cancellation_receipt_never_replays_on_retry():
    client = _FakeClient()
    client._query.receipt = {"still_queued": [], "cancelled": []}
    await client._query.frames.put({
        "type": "system", "subtype": "init", "capabilities": ["interrupt_cancel_queued_v1"],
    })
    run = ClaudeActiveRun(client=client, request_id="req-1")
    _, task = await _start(run)
    await run.steer("user note", priority="next", message_id=CMD, origin="user")
    with pytest.raises(RuntimeError, match="escalation_not_cancelled"):
        await run.escalate(CMD)
    with pytest.raises(RuntimeError, match="escalation_uncertain"):
        await run.escalate(CMD)
    assert len(client._query.control_requests) == 1
    assert client.replacements == []
    client.disconnect.assert_awaited_once()
    await _finish(client, task)


@pytest.mark.anyio
async def test_cancel_queued_user_message_withdraws_only_that_uuid_and_releases_result():
    client = _FakeClient()
    client._query.receipt = {"cancelled": True}
    events: list[tuple[str, str]] = []
    run = ClaudeActiveRun(client=client, request_id="req-1", on_user_lifecycle=lambda *e: events.append(e))
    yielded, task = await _start(run)
    await run.steer("don't send this", priority="next", message_id=CMD, origin="user")
    await client._query.frames.put(_result("original result"))
    await asyncio.sleep(0.01)
    assert _results(yielded) == []

    assert await run.cancel_queued(CMD) == "cancelled"
    assert await run.cancel_queued(CMD) == "cancelled"
    await client._query.frames.put(_lifecycle(CMD, "cancelled"))
    await _finish(client, task)

    assert client._query.control_requests == [{"subtype": "cancel_async_message", "message_uuid": CMD}]
    assert _results(yielded) == ["original result"]
    assert events == [(CMD, "queued"), (CMD, "cancelled")]
    client.interrupt.assert_not_awaited()
    assert client.replacements == []
    with pytest.raises(RuntimeError, match="steer_already_cancelled"):
        await run.steer("don't send this", priority="next", message_id=CMD, origin="user")


@pytest.mark.anyio
async def test_cancel_rejects_started_or_non_user_message_without_control():
    client = _FakeClient()
    run = ClaudeActiveRun(client=client, request_id="req-1")
    _, task = await _start(run)
    await run.steer("bot note", priority="next", message_id=CMD, origin="bot")
    with pytest.raises(RuntimeError, match="unknown_user_steer"):
        await run.cancel_queued(CMD)
    other = "736a3206-0000-4000-8000-000000000002"
    await run.steer("user note", priority="next", message_id=other, origin="user")
    run.note_lifecycle(other, "started")
    with pytest.raises(RuntimeError, match="steer_already_started"):
        await run.cancel_queued(other)
    assert client._query.control_requests == []
    await _finish(client, task)


@pytest.mark.anyio
async def test_cancel_false_receipt_never_claims_message_was_cancelled():
    client = _FakeClient()
    client._query.receipt = {"cancelled": False}
    run = ClaudeActiveRun(client=client, request_id="req-1")
    _, task = await _start(run)
    await run.steer("note", priority="next", message_id=CMD, origin="user")
    with pytest.raises(RuntimeError, match="steer_already_started"):
        await run.cancel_queued(CMD)
    assert run._commands[CMD] == "queued"
    await _finish(client, task)


@pytest.mark.anyio
async def test_cancel_timeout_does_not_issue_a_second_control_request():
    client = _FakeClient()
    client._query._send_control_request = AsyncMock(side_effect=asyncio.TimeoutError)
    run = ClaudeActiveRun(client=client, request_id="req-1")
    _, task = await _start(run)
    await run.steer("user note", priority="next", message_id=CMD, origin="user")
    with pytest.raises(asyncio.TimeoutError):
        await run.cancel_queued(CMD)
    with pytest.raises(RuntimeError, match="cancellation_uncertain"):
        await run.cancel_queued(CMD)
    client._query._send_control_request.assert_awaited_once()
    await _finish(client, task)


@pytest.mark.anyio
async def test_unknown_priority_rejected():
    run = ClaudeActiveRun(client=_FakeClient(), request_id="req-1")
    with pytest.raises(ValueError, match="priority"):
        await run.steer("note", priority="later")

