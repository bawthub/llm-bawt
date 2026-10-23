"""TASK-934: bot-originated steers deliver at the next tool boundary.

Frame sequences mirror live Claude Code CLI 2.1.280 probes:

* fold — a ``next`` message injected while Bash runs is ``queued``, then
  ``started`` right after the tool_result and ``completed`` before the single
  ResultMessage.
* late — a ``next`` message injected while the final answer streams stays
  ``queued`` past the first ResultMessage, then ``started`` runs a follow-up
  turn that ends in a second ResultMessage.

TASK-1300: escalation sweeps the whole CLI queue and replays every swept
command it holds, in send order. A command is released as provably cancelled
only under the four-condition rule; everything else it cannot prove is
published ``uncertain``.
"""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from claude_agent_sdk import ResultMessage, UserMessage
from claude_agent_sdk._internal.query import Query

from agent_bridge.steer_contract import STEER_ITEMS_SLOT
from claude_code_bridge.active_run import (
    STEER_INTERRUPTED_TOOL_RESULT,
    ClaudeActiveRun,
)
from llm_bawt.ops.quiesce_prompts import DEFAULT_QUIESCE_REQUEST

CMD = "736a3206-0000-4000-8000-000000000001"
CMD2 = "736a3206-0000-4000-8000-000000000002"
BOT = "736a3206-0000-4000-8000-0000000000b0"
FRAME = f"INTERRUPT[{STEER_ITEMS_SLOT}]END"


class _Stats:
    def __init__(self, query: "_FakeQuery") -> None:
        self._query = query

    def statistics(self):
        extra = self._query.phantom_buffered
        return SimpleNamespace(current_buffer_used=self._query.frames.qsize() + extra)


class _FakeQuery(Query):
    """Raw-frame source that satisfies the SDK ``Query`` isinstance check.

    Mirrors the SDK reader's ordering: frames the CLI wrote before a control
    receipt are already in ``frames`` when the receipt resolves.
    """

    def __init__(self) -> None:  # noqa: D401 - deliberately skip SDK init
        self.frames: asyncio.Queue = asyncio.Queue()
        self.receipt = {"still_queued": [], "cancelled": [CMD]}
        self.control_requests: list[dict] = []
        # Frames written before the receipt, queued as the receipt resolves.
        self.before_receipt: list[dict] = []
        # Frames the stats claim are buffered but never arrive (barrier timeout).
        self.phantom_buffered = 0
        self._message_receive = _Stats(self)

    async def _send_control_request(self, request, timeout=60):
        self.control_requests.append(request)
        for frame in self.before_receipt:
            self.frames.put_nowait(frame)
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
        self.hang_replacement = False

    async def query(self, prompt):
        if isinstance(prompt, str):
            # The transport write happens; the await may still not return.
            self.replacements.append(prompt)
            if self.hang_replacement:
                await asyncio.Event().wait()
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


_INIT = {"type": "system", "subtype": "init", "capabilities": ["interrupt_cancel_queued_v1"]}


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


def _run(client: _FakeClient, events: list | None = None, **kw) -> ClaudeActiveRun:
    return ClaudeActiveRun(
        client=client, request_id="req-1",
        on_command_lifecycle=(events.append if events is not None else None), **kw,
    )


def _states(events: list[dict], origin: str | None = None) -> list[tuple[str, str]]:
    return [(e["message_id"], e["state"]) for e in events
            if origin is None or e["origin"] == origin]


async def _next(run: ClaudeActiveRun, text: str, uuid: str, origin: str = "user", rpc: str | None = None):
    await run.steer(
        text, priority="next", message_id=uuid, origin=origin,
        framed_message=f"NEXT[{text}]", interrupt_item=f"ITEM[{text}]", steer_rpc_id=rpc,
    )


async def _escalation_rig(*, events: list):
    """Two user steers and one bot steer queued behind a held result."""
    client = _FakeClient()
    await client._query.frames.put(_INIT)
    run = _run(client, events)
    yielded, task = await _start(run)
    await _next(run, "first user", CMD)
    await _next(run, "bot note", BOT, origin="bot", rpc="steer_delivery_d1_g0")
    await _next(run, "second user", CMD2)
    await asyncio.sleep(0.01)
    return client, run, yielded, task


@pytest.mark.anyio
async def test_next_steer_folds_at_tool_boundary_without_interrupt():
    client = _FakeClient()
    run = _run(client)
    yielded, task = await _start(run)

    await _next(run, "Message from bot 'al': reply", CMD, origin="bot")

    client.interrupt.assert_not_awaited()
    [frame] = client.written
    assert frame["priority"] == "next" and frame["uuid"] == CMD
    assert frame["message"]["content"] == "NEXT[Message from bot 'al': reply]"
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
    run = _run(client)
    yielded, task = await _start(run)

    await _next(run, "late note", CMD, origin="bot")
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
    run = _run(client)
    yielded, task = await _start(run)

    await _next(run, "late note", CMD, origin="bot")
    await client._query.frames.put(_result("only turn"))
    await client._query.frames.put(_lifecycle(CMD, "discarded"))
    await _finish(client, task)

    assert _results(yielded) == ["only turn"]


@pytest.mark.anyio
async def test_next_steer_refused_without_lifecycle_support():
    client = AsyncMock()  # not a real SDK client: no raw frame access
    run = ClaudeActiveRun(client=client, request_id="req-1")

    with pytest.raises(RuntimeError, match="next_unsupported"):
        await _next(run, "note", CMD, origin="bot")

    client.query.assert_not_awaited()
    client.interrupt.assert_not_awaited()


@pytest.mark.anyio
async def test_next_steer_after_completion_is_no_active_run():
    client = _FakeClient()
    run = _run(client)
    yielded, task = await _start(run)
    run.mark_completed()

    with pytest.raises(RuntimeError, match="no_active_run"):
        await _next(run, "note", CMD, origin="bot")

    assert client.written == []
    await _finish(client, task)


@pytest.mark.anyio
async def test_now_steer_still_interrupts_and_relabels_killed_tools():
    client = _FakeClient()
    run = _run(client)
    yielded, task = await _start(run)
    client.query = AsyncMock()

    await run.steer("stop and do X", origin="user", framed_message="NOW[stop and do X]")

    client.interrupt.assert_awaited_once_with()
    client.query.assert_awaited_once_with("NOW[stop and do X]")
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
    events: list[dict] = []
    run = _run(client, events)
    yielded, task = await _start(run)

    await _next(run, "user note", CMD)
    await _next(run, "user note", CMD)
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

    assert _states(events) == [(CMD, "queued"), (CMD, "started"), (CMD, "completed")]
    assert _results(yielded) == ["final"]
    client.interrupt.assert_not_awaited()


@pytest.mark.anyio
async def test_user_late_completion_is_published_before_result_reaches_send_handler():
    client = _FakeClient()
    events: list[dict] = []
    run = _run(client, events)
    stream = run.messages()
    pending = asyncio.create_task(stream.__anext__())
    await asyncio.sleep(0)
    await _next(run, "late user note", CMD)
    await client._query.frames.put(_result("first turn"))
    await client._query.frames.put(_lifecycle(CMD, "started"))
    await client._query.frames.put(_result("follow-up"))
    await asyncio.sleep(0.01)
    assert not pending.done()
    await client._query.frames.put(_lifecycle(CMD, "completed"))
    result = await asyncio.wait_for(pending, 1)
    assert result.result == "follow-up"
    assert _states(events) == [(CMD, "queued"), (CMD, "started"), (CMD, "completed")]
    await stream.aclose()


@pytest.mark.anyio
async def test_bot_lifecycle_is_published_with_origin_and_generation_id():
    """TASK-1300 F4: a bot steer's fate reaches the app, naming its RPC."""
    client = _FakeClient()
    events: list[dict] = []
    run = _run(client, events)
    yielded, task = await _start(run)
    with pytest.raises(ValueError, match="UUID message_id"):
        await _next(run, "user note", "not-a-uuid")
    await _next(run, "bot note", CMD, origin="bot", rpc="steer_delivery_d1_g1")
    await client._query.frames.put(_lifecycle(CMD, "started"))
    await client._query.frames.put(_lifecycle(CMD, "completed"))
    await _finish(client, task)
    assert len(client.written) == 1
    assert _states(events, "bot") == [(CMD, "queued"), (CMD, "started"), (CMD, "completed")]
    assert {e["steer_rpc_id"] for e in events} == {"steer_delivery_d1_g1"}
    assert {e["run_request_id"] for e in events} == {"req-1"}
    assert all("proof" not in e for e in events)
    assert yielded == []


@pytest.mark.anyio
async def test_cli_terminal_frame_is_proof_only_without_started_evidence():
    client = _FakeClient()
    events: list[dict] = []
    run = _run(client, events)
    _, task = await _start(run)
    await _next(run, "discarded note", CMD, origin="bot")
    await _next(run, "started note", BOT, origin="bot")
    for frame in (_lifecycle(CMD, "discarded"), _lifecycle(BOT, "started"),
                  _lifecycle(BOT, "cancelled")):
        await client._query.frames.put(frame)
    await _finish(client, task)
    rejected = [e for e in events if e["state"] in {"discarded", "cancelled"}]
    assert [(e["message_id"], e.get("proof")) for e in rejected] == [
        (CMD, "cli_terminal"), (BOT, None),
    ]


@pytest.mark.anyio
@pytest.mark.parametrize("origin", ["user", "bot"])
async def test_next_frame_is_the_app_rendered_text_byte_for_byte(origin):
    client = _FakeClient()
    run = _run(client)
    yielded, task = await _start(run)
    framed = "[registry chat.steer.next_aside]\n  a note  \n"
    await run.steer("a note", priority="next", message_id=CMD, origin=origin,
                    framed_message=framed, interrupt_item="ITEM")
    await _finish(client, task)
    [frame] = client.written
    assert frame["message"]["content"] == framed


@pytest.mark.anyio
@pytest.mark.parametrize("origin", ["user", "bot"])
@pytest.mark.parametrize("missing", ["framed_message", "interrupt_item"])
async def test_unframed_next_steer_is_refused_before_reaching_the_cli(origin, missing):
    client = _FakeClient()
    run = _run(client)
    _, task = await _start(run)
    kwargs = {"framed_message": "F", "interrupt_item": "I", missing: None}
    with pytest.raises(RuntimeError, match="steer_unframed"):
        await run.steer("a note", priority="next", message_id=CMD, origin=origin, **kwargs)
    assert client.written == []
    await _finish(client, task)


@pytest.mark.anyio
async def test_unframed_now_steer_is_refused_without_interrupting():
    client = _FakeClient()
    run = _run(client)
    with pytest.raises(RuntimeError, match="steer_unframed"):
        await run.steer("stop", origin="user")
    client.interrupt.assert_not_awaited()


@pytest.mark.anyio
async def test_system_next_steer_is_delivered_verbatim():
    # TASK-1233: the quiesce control pump's stop request travels as a system
    # steer. Its text is the registry-owned ops.quiesce_request body; the
    # bridge adds nothing (a trailing "continue the unfinished task" would
    # undo it, and any wrapper would be prompt text the registry can't edit).
    client = _FakeClient()
    run = _run(client)
    yielded, task = await _start(run)
    await run.steer(DEFAULT_QUIESCE_REQUEST, priority="next", message_id=CMD, origin="system")
    await _finish(client, task)
    [frame] = client.written
    assert frame["message"]["content"] == DEFAULT_QUIESCE_REQUEST


@pytest.mark.anyio
async def test_escalation_atomically_replaces_queued_message_once():
    client = _FakeClient()
    await client._query.frames.put(_INIT)
    events: list[dict] = []
    run = _run(client, events)
    yielded, task = await _start(run)
    await _next(run, "urgent direction", CMD)
    await client._query.frames.put(_result("old turn"))
    await asyncio.sleep(0.01)
    assert _results(yielded) == []
    assert await run.escalate(CMD, interrupt_frame=FRAME) == "escalated"
    assert await run.escalate(CMD, interrupt_frame=FRAME) == "escalated"
    assert client._query.control_requests == [{"subtype": "interrupt", "cancel_queued": True}]
    assert client.replacements == ["INTERRUPT[ITEM[urgent direction]]END"]
    await client._query.frames.put(_lifecycle(CMD, "cancelled"))
    await client._query.frames.put(_result("replacement"))
    await _finish(client, task)
    assert _results(yielded) == ["replacement"]
    assert run.consume_replaced_result(yielded[-1]) is False
    assert _states(events) == [(CMD, "queued"), (CMD, "started"), (CMD, "completed")]


@pytest.mark.anyio
@pytest.mark.parametrize("frames_before_receipt", [False, True], ids=["receipt-first", "frames-first"])
async def test_escalation_replays_every_swept_command_in_send_order(frames_before_receipt):
    """TASK-1300 acceptance: the Snark incident, with nothing lost."""
    events: list[dict] = []
    client, run, yielded, task = await _escalation_rig(events=events)
    client._query.receipt = {"still_queued": [], "cancelled": [CMD2, BOT, CMD]}
    swept_frames = [_lifecycle(u, "cancelled") for u in (CMD, BOT, CMD2)]
    if frames_before_receipt:
        client._query.before_receipt = swept_frames
    events.clear()

    assert await run.escalate(CMD2, interrupt_frame=FRAME) == "escalated"

    assert len(client._query.control_requests) == 1
    assert client.replacements == [
        "INTERRUPT[ITEM[first user]\n\nITEM[bot note]\n\nITEM[second user]]END"
    ]
    if not frames_before_receipt:
        for frame in swept_frames:
            await client._query.frames.put(frame)
    # The CLI ends the interrupted turn, then the replacement turn.
    await client._query.frames.put(_result("interrupted"))
    await client._query.frames.put(_result("replacement"))
    await _finish(client, task)
    assert [run.consume_replaced_result(m) for m in yielded if isinstance(m, ResultMessage)] == [
        True, False,
    ]
    # Started (replayed) then completed for each; never the CLI's cancelled.
    assert _states(events) == [
        (CMD, "started"), (BOT, "started"), (CMD2, "started"),
        (CMD, "completed"), (BOT, "completed"), (CMD2, "completed"),
    ]
    assert {e["proof"] for e in events} == {"replayed"}
    client.disconnect.assert_not_awaited()


@pytest.mark.anyio
async def test_escalation_never_replays_a_command_that_started_before_the_barrier():
    """A started frame written before the receipt but not yet read."""
    events: list[dict] = []
    client, run, yielded, task = await _escalation_rig(events=events)
    client._query.receipt = {"still_queued": [], "cancelled": [CMD, BOT, CMD2]}
    client._query.before_receipt = [_lifecycle(BOT, "started")]
    events.clear()

    with pytest.raises(RuntimeError, match=rf"escalation_started_in_sweep: {BOT}\(bot\)"):
        await run.escalate(CMD2, interrupt_frame=FRAME)

    assert client.replacements == []
    await _finish(client, task)
    # The started bot steer follows its own lifecycle; the rest are fenced.
    assert ("cancelled" not in {e["state"] for e in events})
    assert _states(events) == [(CMD, "uncertain"), (CMD2, "uncertain"), (BOT, "started")]
    assert {e.get("proof") for e in events if e["state"] == "uncertain"} == {"started_in_sweep"}


@pytest.mark.anyio
async def test_escalation_already_started_command_listed_in_receipt_is_not_replayed():
    events: list[dict] = []
    client, run, yielded, task = await _escalation_rig(events=events)
    run.note_lifecycle(BOT, "started")
    client._query.receipt = {"still_queued": [], "cancelled": [CMD, BOT, CMD2]}

    assert await run.escalate(CMD2, interrupt_frame=FRAME) == "escalated"

    assert client.replacements == ["INTERRUPT[ITEM[first user]\n\nITEM[second user]]END"]
    await _finish(client, task)


@pytest.mark.anyio
async def test_escalation_barrier_timeout_fences_swept_commands_uncertain():
    events: list[dict] = []
    client, run, yielded, task = await _escalation_rig(events=events)
    run._steer_timeout_s = 0.05
    client._query.receipt = {"still_queued": [], "cancelled": [CMD, BOT, CMD2]}
    client._query.phantom_buffered = 3
    events.clear()

    with pytest.raises(RuntimeError, match="escalation_barrier_timeout"):
        await run.escalate(CMD2, interrupt_frame=FRAME)

    assert client.replacements == []
    client.disconnect.assert_awaited_once()
    assert _states(events) == [(CMD, "uncertain"), (BOT, "uncertain"), (CMD2, "uncertain")]
    assert {e["proof"] for e in events} == {"barrier_timeout"}
    # Later CLI frames never upgrade a fenced command to a rejection.
    await client._query.frames.put(_lifecycle(BOT, "cancelled"))
    await _finish(client, task)
    assert [e["state"] for e in events].count("cancelled") == 0


@pytest.mark.anyio
async def test_escalation_survivors_release_swept_commands_as_proven_cancelled():
    """The only path that releases: barrier done, in C, not started, no write."""
    events: list[dict] = []
    client, run, yielded, task = await _escalation_rig(events=events)
    client._query.receipt = {"still_queued": [CMD], "cancelled": [BOT, CMD2]}
    events.clear()

    with pytest.raises(RuntimeError, match=rf"escalation_survivors: {CMD}\(user\)"):
        await run.escalate(CMD2, interrupt_frame=FRAME)

    assert client.replacements == []
    client.disconnect.assert_awaited_once()
    assert _states(events) == [(BOT, "cancelled"), (CMD2, "cancelled")]
    assert {e["proof"] for e in events} == {"sweep_receipt"}
    [bot] = [e for e in events if e["origin"] == "bot"]
    assert bot["steer_rpc_id"] == "steer_delivery_d1_g0"
    # The CLI's own cancelled frames are not published a second time.
    for u in (BOT, CMD2):
        await client._query.frames.put(_lifecycle(u, "cancelled"))
    await _finish(client, task)
    assert len(events) == 2


@pytest.mark.anyio
async def test_escalation_consumed_but_timed_out_write_is_uncertain_not_cancelled():
    events: list[dict] = []
    client, run, yielded, task = await _escalation_rig(events=events)
    run._steer_timeout_s = 0.05
    client.hang_replacement = True
    client._query.receipt = {"still_queued": [], "cancelled": [CMD, BOT, CMD2]}
    events.clear()

    with pytest.raises(asyncio.TimeoutError):
        await run.escalate(CMD2, interrupt_frame=FRAME)

    assert len(client.replacements) == 1  # written exactly once
    client.disconnect.assert_awaited_once()
    assert _states(events) == [(CMD, "uncertain"), (BOT, "uncertain"), (CMD2, "uncertain")]
    assert {e["proof"] for e in events} == {"replacement_write_uncertain"}
    for u in (CMD, BOT, CMD2):
        await client._query.frames.put(_lifecycle(u, "cancelled"))
    await _finish(client, task)
    assert len(events) == 3
    with pytest.raises(RuntimeError, match="escalation_uncertain"):
        await run.escalate(CMD2, interrupt_frame=FRAME)


@pytest.mark.anyio
async def test_escalation_missing_receipt_fences_every_eligible_command():
    events: list[dict] = []
    client, run, yielded, task = await _escalation_rig(events=events)
    client._query.receipt = {"cancelled": None}
    events.clear()

    with pytest.raises(RuntimeError, match="escalation_missing_receipt"):
        await run.escalate(CMD2, interrupt_frame=FRAME)

    assert client.replacements == []
    assert _states(events) == [(CMD, "uncertain"), (BOT, "uncertain"), (CMD2, "uncertain")]
    await _finish(client, task)


@pytest.mark.anyio
async def test_escalation_flushes_frames_of_unswept_commands_after_it_settles():
    events: list[dict] = []
    client, run, yielded, task = await _escalation_rig(events=events)
    # BOT started from the CLI queue while the interrupt was in flight.
    client._query.receipt = {"still_queued": [], "cancelled": [CMD, CMD2]}
    client._query.before_receipt = [_lifecycle(BOT, "started")]
    events.clear()

    assert await run.escalate(CMD2, interrupt_frame=FRAME) == "escalated"

    assert client.replacements == ["INTERRUPT[ITEM[first user]\n\nITEM[second user]]END"]
    assert _states(events) == [(CMD, "started"), (CMD2, "started"), (BOT, "started")]
    assert [e.get("proof") for e in events] == ["replayed", "replayed", None]
    await _finish(client, task)


@pytest.mark.anyio
@pytest.mark.parametrize("state", ["started", "completed"])
async def test_escalation_does_not_interrupt_if_command_already_started(state):
    client = _FakeClient()
    await client._query.frames.put(_INIT)
    run = _run(client)
    _, task = await _start(run)
    await _next(run, "user note", CMD)
    run.note_lifecycle(CMD, state)
    with pytest.raises(RuntimeError, match="steer_already_started"):
        await run.escalate(CMD, interrupt_frame=FRAME)
    assert client._query.control_requests == []
    assert client.replacements == []
    await _finish(client, task)


@pytest.mark.anyio
async def test_escalation_refuses_only_before_interrupting_for_capability_or_framing():
    client = _FakeClient()
    run = _run(client)
    _, task = await _start(run)
    await _next(run, "user note", CMD)
    await _next(run, "bot note", BOT, origin="bot")
    with pytest.raises(RuntimeError, match="escalation_unsupported_cli"):
        await run.escalate(CMD, interrupt_frame=FRAME)
    run._cli_capabilities.add("interrupt_cancel_queued_v1")
    for frame in (None, "no slot here"):
        with pytest.raises(RuntimeError, match="steer_unframed"):
            await run.escalate(CMD, interrupt_frame=frame)
    with pytest.raises(RuntimeError, match="unknown_user_steer"):
        await run.escalate(BOT, interrupt_frame=FRAME)
    assert client._query.control_requests == []
    # Another queued command no longer blocks the interrupt (TASK-1300).
    client._query.receipt = {"still_queued": [], "cancelled": [CMD, BOT]}
    assert await run.escalate(CMD, interrupt_frame=FRAME) == "escalated"
    await _finish(client, task)


@pytest.mark.anyio
async def test_escalation_missing_cancellation_receipt_never_replays_on_retry():
    client = _FakeClient()
    client._query.receipt = {"still_queued": [], "cancelled": []}
    await client._query.frames.put(_INIT)
    run = _run(client)
    _, task = await _start(run)
    await _next(run, "user note", CMD)
    with pytest.raises(RuntimeError, match="escalation_not_cancelled"):
        await run.escalate(CMD, interrupt_frame=FRAME)
    with pytest.raises(RuntimeError, match="escalation_uncertain"):
        await run.escalate(CMD, interrupt_frame=FRAME)
    assert len(client._query.control_requests) == 1
    assert client.replacements == []
    client.disconnect.assert_awaited_once()
    await _finish(client, task)


@pytest.mark.anyio
async def test_release_unstarted_publishes_uncertain_never_a_rejection():
    client = _FakeClient()
    events: list[dict] = []
    run = _run(client, events)
    _, task = await _start(run)
    await _next(run, "never read", CMD, origin="bot", rpc="steer_delivery_d1_g0")
    await _next(run, "ran", BOT, origin="bot")
    await _next(run, "rejected", CMD2, origin="bot")
    await client._query.frames.put(_lifecycle(BOT, "started"))
    await client._query.frames.put(_lifecycle(CMD2, "refused"))
    await _finish(client, task)
    events.clear()

    assert run.release_unstarted(SimpleNamespace(exited=True)) == [CMD]
    assert run.release_unstarted(SimpleNamespace(exited=True)) == []
    [event] = events
    assert (event["message_id"], event["state"], event["proof"], event["exited"]) == (
        CMD, "uncertain", "teardown", True,
    )
    assert event["steer_rpc_id"] == "steer_delivery_d1_g0"


@pytest.mark.anyio
async def test_lifecycle_frames_are_logged_at_info_with_origin(caplog):
    client = _FakeClient()
    run = _run(client)
    _, task = await _start(run)
    with caplog.at_level(logging.INFO, logger="claude_code_bridge.active_run"):
        await _next(run, "bot note", BOT, origin="bot")
        await _finish(client, task)
    assert f"uuid={BOT} origin=bot sent->queued" in caplog.text


@pytest.mark.anyio
async def test_cancel_queued_user_message_withdraws_only_that_uuid_and_releases_result():
    client = _FakeClient()
    client._query.receipt = {"cancelled": True}
    events: list[dict] = []
    run = _run(client, events)
    yielded, task = await _start(run)
    await _next(run, "don't send this", CMD)
    await client._query.frames.put(_result("original result"))
    await asyncio.sleep(0.01)
    assert _results(yielded) == []

    assert await run.cancel_queued(CMD) == "cancelled"
    assert await run.cancel_queued(CMD) == "cancelled"
    await client._query.frames.put(_lifecycle(CMD, "cancelled"))
    await _finish(client, task)

    assert client._query.control_requests == [{"subtype": "cancel_async_message", "message_uuid": CMD}]
    assert _results(yielded) == ["original result"]
    assert _states(events) == [(CMD, "queued"), (CMD, "cancelled")]
    client.interrupt.assert_not_awaited()
    assert client.replacements == []
    with pytest.raises(RuntimeError, match="steer_already_cancelled"):
        await _next(run, "don't send this", CMD)


@pytest.mark.anyio
async def test_cancel_rejects_started_or_non_user_message_without_control():
    client = _FakeClient()
    run = _run(client)
    _, task = await _start(run)
    await _next(run, "bot note", CMD, origin="bot")
    with pytest.raises(RuntimeError, match="unknown_user_steer"):
        await run.cancel_queued(CMD)
    await _next(run, "user note", CMD2)
    run.note_lifecycle(CMD2, "started")
    with pytest.raises(RuntimeError, match="steer_already_started"):
        await run.cancel_queued(CMD2)
    assert client._query.control_requests == []
    await _finish(client, task)


@pytest.mark.anyio
async def test_cancel_false_receipt_never_claims_message_was_cancelled():
    client = _FakeClient()
    client._query.receipt = {"cancelled": False}
    run = _run(client)
    _, task = await _start(run)
    await _next(run, "note", CMD)
    with pytest.raises(RuntimeError, match="steer_already_started"):
        await run.cancel_queued(CMD)
    assert run._commands[CMD] == "queued"
    await _finish(client, task)


@pytest.mark.anyio
async def test_cancel_timeout_does_not_issue_a_second_control_request():
    client = _FakeClient()
    client._query._send_control_request = AsyncMock(side_effect=asyncio.TimeoutError)
    run = _run(client)
    _, task = await _start(run)
    await _next(run, "user note", CMD)
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
