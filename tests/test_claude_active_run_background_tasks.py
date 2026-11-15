"""TASK-595: background subagents finish on the task-notification turn.

With background tasks enabled, the CLI ends the host turn while a subagent is
still running, then re-invokes the model when it reports. The first result
must not end the run; the notification turn's result does.
"""

from __future__ import annotations

import asyncio

import pytest
from claude_agent_sdk._internal.message_parser import parse_message

from tests.test_claude_active_run_finishing import _consume_until_result
from tests.test_claude_active_run_next_steer import _FakeClient, _result, _results, _run

NOTE = {"origin": {"kind": "task-notification"}}


def _started(task_id: str) -> dict:
    return {"type": "system", "subtype": "task_started", "task_id": task_id,
            "description": "d", "uuid": f"u-{task_id}", "session_id": "s"}


def _notified(task_id: str) -> dict:
    return {"type": "system", "subtype": "task_notification", "task_id": task_id,
            "status": "completed", "output_file": "/tmp/o", "summary": "done",
            "uuid": f"n-{task_id}", "session_id": "s"}


def _killed(task_id: str) -> dict:
    return {"type": "system", "subtype": "task_updated", "task_id": task_id,
            "patch": {"status": "killed"}, "session_id": "s"}


async def _drive(frames: list, *, raw: bool = True, enabled: bool = True) -> list[str]:
    if raw:
        client = _FakeClient()
        for frame in frames:
            client._query.frames.put_nowait(frame)
        client._query.frames.put_nowait(None)
    else:
        class _Plain:
            async def receive_messages(self):
                for frame in frames:
                    msg = parse_message(frame)
                    if msg is not None:
                        yield msg

            async def interrupt(self):  # pragma: no cover - unused
                pass

        client = _Plain()
    run = _run(client, background_tasks=enabled)
    return _results(await asyncio.wait_for(_consume_until_result(run), 2))


@pytest.mark.anyio
@pytest.mark.parametrize("raw", [True, False])
async def test_first_result_is_held_until_the_notification_turn(raw):
    frames = [
        _started("a"), _result("launched, standing by"),
        _notified("a"), {**_result("final with agent answer"), **NOTE},
    ]
    assert await _drive(frames, raw=raw) == ["final with agent answer"]


@pytest.mark.anyio
async def test_parallel_agents_wait_for_the_last_one():
    frames = [
        _started("a"), _started("b"), _result("launched"),
        _notified("a"), {**_result("a done, b pending"), **NOTE},
        _killed("b"), {**_result("all done"), **NOTE},
    ]
    assert await _drive(frames) == ["all done"]


@pytest.mark.anyio
async def test_no_subagent_turn_is_unchanged():
    assert await _drive([_result("plain answer")]) == ["plain answer"]


@pytest.mark.anyio
async def test_foreground_agent_settled_before_result_does_not_hold():
    frames = [_started("a"), _notified("a"), _result("sync answer")]
    assert await _drive(frames) == ["sync answer"]


@pytest.mark.anyio
async def test_stale_notification_before_this_run_is_still_ignored():
    frames = [{**_result("old run"), **NOTE}, _result("this run")]
    assert await _drive(frames) == ["this run"]


@pytest.mark.anyio
async def test_stream_end_with_agents_in_flight_finishes_on_last_result():
    frames = [_started("a"), _result("launched")]
    assert await _drive(frames) == ["launched"]


@pytest.mark.anyio
async def test_kill_switch_keeps_first_result_terminal():
    frames = [_started("a"), _result("launched"), _notified("a"),
              {**_result("late"), **NOTE}]
    assert await _drive(frames, enabled=False) == ["launched"]


@pytest.mark.anyio
async def test_now_steer_during_idle_background_wait_owns_next_result():
    """Codex r1 B1: an idle held parent emits no interrupt result."""
    client = _FakeClient()
    run = _run(client, background_tasks=True)
    consumer = asyncio.create_task(_consume_until_result(run))
    client._query.frames.put_nowait(_started("a"))
    client._query.frames.put_nowait(_result("standing by"))
    for _ in range(5):
        await asyncio.sleep(0)
    assert run._background_result is not None

    await run.steer("new instruction", priority="now", framed_message="new instruction")
    assert run._expected_interrupt_results == 0
    assert run._background_result is None
    client._query.frames.put_nowait(_notified("a"))
    client._query.frames.put_nowait(_result("replacement answer"))
    client._query.frames.put_nowait(None)
    try:
        yielded = await asyncio.wait_for(consumer, 5)
        assert _results(yielded) == ["replacement answer"]
    finally:
        run.mark_completed()


@pytest.mark.anyio
async def test_escalation_during_idle_background_wait_owns_next_result():
    """Codex r2 B1: escalation follows the same idle-wait rule as now-steer."""
    from tests.test_claude_active_run_next_steer import CMD, FRAME, _INIT, _next

    client = _FakeClient()
    run = _run(client, background_tasks=True)
    consumer = asyncio.create_task(_consume_until_result(run))
    for frame in (_INIT, _started("a"), _result("standing by")):
        client._query.frames.put_nowait(frame)
    for _ in range(5):
        await asyncio.sleep(0)
    assert run._background_result is not None
    try:
        await _next(run, "replacement answer", CMD)
        await run.escalate(CMD, interrupt_frame=FRAME)
        assert run._expected_interrupt_results == 0
        assert run._background_result is None
        client._query.frames.put_nowait(_notified("a"))
        client._query.frames.put_nowait(_result("replacement answer"))
        client._query.frames.put_nowait(None)
        yielded = await asyncio.wait_for(consumer, 5)
        assert _results(yielded) == ["replacement answer"]
    finally:
        if not consumer.done():
            consumer.cancel()
            await asyncio.gather(consumer, return_exceptions=True)
        run.mark_completed()


def _progress(task_id: str) -> dict:
    return {"type": "system", "subtype": "task_progress", "task_id": task_id,
            "description": "d", "usage": {}, "uuid": f"p-{task_id}", "session_id": "s"}


@pytest.mark.anyio
async def test_endless_progress_hits_the_hard_background_bound():
    """Progress resets the watchdog's idle timer, so the hold needs its own cap."""
    from claude_code_bridge.active_run import BackgroundWaitTimeout

    client = _FakeClient()
    for frame in [_started("a"), _result("launched"), _progress("a"), _progress("a")]:
        client._query.frames.put_nowait(frame)
    run = _run(client, background_tasks=True)
    run.background_wait_max_s = 0.0
    with pytest.raises(BackgroundWaitTimeout, match="still running after 0s: a"):
        await asyncio.wait_for(_consume_until_result(run), 2)


@pytest.mark.anyio
async def test_result_after_a_steer_over_background_ends_the_run():
    """A user correction or ops quiesce steered during the wait must not stall
    behind subagents that are still running."""
    client = _FakeClient()
    run = _run(client, background_tasks=True)
    client._query.frames.put_nowait(_started("a"))
    client._query.frames.put_nowait(_result("launched"))

    async def steer_then_answer():
        while run._background_result is None:
            await asyncio.sleep(0)
        run._mark_steer_over_background()
        client._query.frames.put_nowait(_result("stopping now"))

    task = asyncio.create_task(steer_then_answer())
    got = _results(await asyncio.wait_for(_consume_until_result(run), 2))
    await task
    assert got == ["stopping now"]


@pytest.mark.anyio
async def test_steer_without_background_work_changes_nothing():
    run = _run(_FakeClient(), background_tasks=True)
    run._mark_steer_over_background()
    assert run._steered_over_background is False
