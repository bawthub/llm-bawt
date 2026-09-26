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
        self.interrupt = AsyncMock()
        self.disconnect = AsyncMock()

    async def query(self, prompt):
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
async def test_unknown_priority_rejected():
    run = ClaudeActiveRun(client=_FakeClient(), request_id="req-1")
    with pytest.raises(ValueError, match="priority"):
        await run.steer("note", priority="later")

