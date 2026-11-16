"""TASK-1252: durable inbox admission needs positive CLI evidence, not RPC return.

All SDK/ledger inputs are fakes. Event barriers exercise ownership changes and
interrupt ordering without real bot traffic or timing-sensitive sleeps.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agent_bridge.session_queue import SessionQueue
from claude_code_bridge.active_run import ClaudeActiveRun
from claude_code_bridge.inbox_input import accept_inbox_input
from tests.test_claude_active_run_next_steer import BOT


class InputClient:
    def __init__(self, states=("queued",), error=None):
        self.states = states
        self.error = error
        self.frames = []
        self.interrupt = AsyncMock()
        self.on_input = None
        self.run = ClaudeActiveRun(self, request_id="run-1", lifecycle_supported=True)

    async def query(self, prompt):
        if isinstance(prompt, str):
            self.frames.append(prompt)
            return
        async for frame in prompt:
            self.frames.append(frame)
            if self.on_input is not None:
                self.on_input()
            for state in self.states:
                self.run.note_lifecycle(frame["uuid"], state)
            if self.error is not None:
                raise self.error


def rig(**kwargs):
    client = InputClient(**kwargs)
    queue = SessionQueue()
    queue.set_active_client("session", client.run)
    prepare = AsyncMock(return_value=True)
    args = {
        "command_uuid": BOT,
        "steer_rpc_id": "steer_delivery_d1",
        "framed_message": "NEXT[trusted bot message]",
        "interrupt_item": "ITEM[trusted bot message]",
        "owns_run": lambda: (
            queue.get_active_client("session") is client.run
            and not queue.is_tombstoned("session", client.run.request_id)
            and not queue.is_stopping("session", client.run.request_id)
        ),
        "prepare_write": prepare,
    }
    return client, queue, prepare, args


@pytest.mark.anyio
@pytest.mark.parametrize("state", ["queued", "started", "completed"])
async def test_positive_receipt_requires_cli_evidence_and_prewrite_fence(state):
    client, _, prepare, args = rig(states=(state,))
    client.on_input = lambda: prepare.assert_awaited_once()
    receipt = await accept_inbox_input(client.run, **args)
    assert receipt.outcome == "accepted"
    assert receipt.state == state and receipt.proof is None
    assert receipt.acceptance == ("queued" if state == "queued" else "started")
    assert (receipt.request_id, receipt.command_uuid) == ("run-1", BOT)
    [frame] = client.frames
    assert frame["uuid"] == BOT and frame["priority"] == "next"
    assert frame["message"]["content"] == args["framed_message"]
    client.interrupt.assert_not_awaited()


@pytest.mark.anyio
async def test_started_proof_survives_a_newer_teardown_state_in_receipt():
    client, _, _, args = rig(states=("started", "refused"))
    receipt = await accept_inbox_input(client.run, **args)
    assert receipt.outcome == "accepted" and receipt.state == "refused"
    assert receipt.acceptance == "started" and receipt.proof is None


@pytest.mark.anyio
@pytest.mark.parametrize("ending", ["missing", "completed", "finishing", "unsupported", "maintenance", "stop"])
async def test_no_eligible_owner_never_prepares_or_writes(ending):
    client, queue, prepare, args = rig()
    if ending == "missing":
        queue.pop_active_client("session")
    elif ending == "completed":
        client.run.mark_completed()
    elif ending == "finishing":
        client.run._finishing = True
    elif ending == "unsupported":
        client.run.lifecycle_supported = False
    elif ending == "maintenance":
        client.run.maintenance = True
    else:
        queue.tombstone("session", client.run.request_id)
    receipt = await accept_inbox_input(client.run, **args)
    assert receipt.outcome == "not_written"
    assert client.frames == [] and not client.run.steering_attempted
    prepare.assert_not_awaited()


@pytest.mark.anyio
@pytest.mark.parametrize("change", ["handoff", "finishing", "stop"])
async def test_ownership_and_stop_rechecked_after_durable_preparation(change):
    client, queue, prepare, args = rig()

    async def prepare_write():
        assert not client.run._steer_lock.locked()
        if change == "handoff":
            queue.set_active_client("session", object())
        elif change == "finishing":
            client.run._finishing = True
        else:
            queue.tombstone("session", client.run.request_id)
        return True

    prepare.side_effect = prepare_write
    receipt = await accept_inbox_input(client.run, **args)
    assert receipt.outcome == "not_written"
    assert client.frames == [] and not client.run.has_unstarted_command()
    prepare.assert_awaited_once()


@pytest.mark.anyio
async def test_stale_claim_or_lost_preparation_never_authorizes_input():
    client, _, prepare, args = rig()
    prepare.return_value = False
    receipt = await accept_inbox_input(client.run, **args)
    assert receipt.outcome == "not_written"
    assert receipt.detail == "claim_write_not_authorized"
    assert client.frames == []
    prepare.side_effect = TimeoutError("lost WRITING response")
    with pytest.raises(TimeoutError, match="lost WRITING"):
        await accept_inbox_input(client.run, **args)
    assert client.frames == []


@pytest.mark.anyio
@pytest.mark.parametrize("state", ["cancelled", "discarded", "refused"])
async def test_proven_rejection_is_not_accepted_or_automatically_rewritten(state):
    client, _, prepare, args = rig(states=(state,))
    first = await accept_inbox_input(client.run, **args)
    second = await accept_inbox_input(client.run, **args)
    assert first == second
    assert first.outcome == "rejected" and first.proof == "cli_terminal"
    assert len(client.frames) == 1
    prepare.assert_awaited_once()


@pytest.mark.anyio
async def test_rejection_after_started_never_proves_nonexecution():
    client, _, _, args = rig(states=("started", "cancelled"))
    receipt = await accept_inbox_input(client.run, **args)
    assert receipt.outcome == "accepted" and receipt.proof is None
    assert len(client.frames) == 1


@pytest.mark.anyio
@pytest.mark.parametrize("now_fails_first", [True, False])
async def test_refused_inbox_and_failed_now_restore_background_hold(now_fails_first):
    """Independent review regression, including reverse rollback ordering."""
    client, _, _, args = rig(states=())
    run = client.run
    run.background_tasks = True
    run._active_tasks.add("still-running-child")
    written, interrupting, release_interrupt = asyncio.Event(), asyncio.Event(), asyncio.Event()
    client.on_input = written.set

    async def failed_interrupt():
        interrupting.set()
        await release_interrupt.wait()
        raise RuntimeError("interrupt refused before write")

    client.interrupt.side_effect = failed_interrupt
    inbox = asyncio.create_task(accept_inbox_input(run, **args))
    now = None
    try:
        await written.wait()
        now = asyncio.create_task(run.steer("redirect", priority="now", origin="system"))
        await interrupting.wait()
        if now_fails_first:
            release_interrupt.set()
            with pytest.raises(RuntimeError, match="interrupt refused before write"):
                await now
        run.note_lifecycle(BOT, "refused")
        assert (await inbox).outcome == "rejected"
        if not now_fails_first:
            # A still-pending now-steer retains its own override.
            assert run._steered_over_background
            release_interrupt.set()
            with pytest.raises(RuntimeError, match="interrupt refused before write"):
                await now
        assert len(client.frames) == 1 and not run._started_ever
        assert not run._steered_over_background
        assert run._background_hold(SimpleNamespace(), None) == "hold"
    finally:
        inbox.cancel()
        if now is not None:
            now.cancel()
        await asyncio.gather(inbox, *([now] if now is not None else []), return_exceptions=True)


@pytest.mark.anyio
async def test_queued_acceptance_is_not_irrevocable_and_cannot_be_rewritten():
    client, _, prepare, args = rig()
    assert (await accept_inbox_input(client.run, **args)).outcome == "accepted"
    client.run.note_lifecycle(BOT, "discarded")
    receipt = await accept_inbox_input(client.run, **args)
    assert receipt.outcome == "rejected" and receipt.proof == "cli_terminal"
    assert len(client.frames) == 1
    prepare.assert_awaited_once()


@pytest.mark.anyio
async def test_rejected_input_restores_background_wait_override():
    client, _, _, args = rig(states=("refused",))
    client.run.background_tasks = True
    client.run._active_tasks.add("background-task")
    assert (await accept_inbox_input(client.run, **args)).outcome == "rejected"
    assert not client.run._steered_over_background


@pytest.mark.anyio
async def test_successful_write_without_ack_remains_uncertain_and_late_proof_repairs(monkeypatch):
    client, _, prepare, args = rig(states=())
    real_wait_for = asyncio.wait_for

    async def expire_ack(awaitable, *, timeout):
        if awaitable.cr_code.co_name == "wait":
            awaitable.close()
            raise asyncio.TimeoutError
        return await real_wait_for(awaitable, timeout=timeout)

    monkeypatch.setattr("claude_code_bridge.inbox_input.asyncio.wait_for", expire_ack)
    first = await accept_inbox_input(client.run, **args)
    second = await accept_inbox_input(client.run, **args)
    assert first.outcome == second.outcome == "uncertain"
    assert first.detail == "cli_ack_timeout" and first.proof is None
    assert len(client.frames) == 1
    client.run.note_lifecycle(BOT, "started")
    repaired = await accept_inbox_input(client.run, **args)
    assert repaired.outcome == "accepted"
    assert len(client.frames) == 1
    prepare.assert_awaited_once()


@pytest.mark.anyio
async def test_write_exception_cannot_authorize_replay_or_hide_late_acceptance():
    client, _, prepare, args = rig(states=(), error=OSError("write outcome unknown"))
    receipt = await accept_inbox_input(client.run, **args)
    assert receipt.outcome == "uncertain" and receipt.state == "sent"
    assert receipt.detail == "sdk_write_failed:OSError"
    client.run.note_lifecycle(BOT, "queued")
    assert (await accept_inbox_input(client.run, **args)).outcome == "accepted"
    assert len(client.frames) == 1
    prepare.assert_awaited_once()


@pytest.mark.anyio
async def test_teardown_uncertainty_overrides_old_queued_receipt():
    client, _, _, args = rig()
    assert (await accept_inbox_input(client.run, **args)).outcome == "accepted"
    client.run.release_unstarted()
    assert (await accept_inbox_input(client.run, **args)).outcome == "uncertain"
    client.run.note_lifecycle(BOT, "started")
    assert (await accept_inbox_input(client.run, **args)).outcome == "accepted"
    assert len(client.frames) == 1


@pytest.mark.anyio
async def test_existing_operation_cannot_be_rebound_to_a_new_identity():
    client, _, _, args = rig()
    await accept_inbox_input(client.run, **args)
    args["steer_rpc_id"] = "steer_delivery_d1_g1"
    receipt = await accept_inbox_input(client.run, **args)
    assert receipt.outcome == "uncertain" and receipt.detail == "command_identity_conflict"
    assert len(client.frames) == 1


@pytest.mark.anyio
async def test_slow_preparation_does_not_hold_input_lock_and_handoff_wins():
    client, queue, _, args = rig()
    preparing, release = asyncio.Event(), asyncio.Event()

    async def prepare_write():
        preparing.set()
        await release.wait()
        return True

    args["prepare_write"] = prepare_write
    task = asyncio.create_task(accept_inbox_input(client.run, **args))
    await preparing.wait()
    try:
        assert not client.run._steer_lock.locked()
        await client.run.steer("redirect", priority="now", origin="system")
        queue.pop_active_client("session")
        release.set()
        assert (await task).outcome == "not_written"
        assert client.frames == ["redirect"]
        client.interrupt.assert_awaited_once()
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.anyio
async def test_waiting_for_cli_receipt_does_not_block_interrupt():
    client, _, _, args = rig(states=())
    client.run.background_tasks = True
    client.run._active_tasks.add("background-task")
    written = asyncio.Event()
    client.on_input = written.set
    task = asyncio.create_task(accept_inbox_input(client.run, **args))
    await written.wait()
    try:
        # Acquiring the lock is deterministic evidence the ack waiter released it.
        async with client.run._steer_lock:
            assert not task.done()
        await client.run.steer("redirect", priority="now", origin="system")
        client.run.note_lifecycle(BOT, "refused")
        assert (await task).outcome == "rejected"
        # The older rejected input must not undo the newer now-steer's override.
        assert client.run._steered_over_background
        assert len(client.frames) == 2
        client.interrupt.assert_awaited_once()
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.anyio
async def test_cancelled_writer_never_returns_a_success_receipt():
    client, _, _, args = rig(states=())
    written = asyncio.Event()
    client.on_input = written.set
    task = asyncio.create_task(accept_inbox_input(client.run, **args))
    await written.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert client.run._commands[BOT] == "sent"
    assert len(client.frames) == 1


@pytest.mark.anyio
async def test_restart_requires_durable_permission_not_empty_local_dedup():
    client, _, _, args = rig(states=(), error=OSError("crash after write"))
    assert (await accept_inbox_input(client.run, **args)).outcome == "uncertain"
    replacement, _, prepare, resumed = rig()
    # Durable WRITING survived the process. Recovery may not issue a new write.
    prepare.return_value = False
    receipt = await accept_inbox_input(replacement.run, **resumed)
    assert receipt.outcome == "not_written"
    assert replacement.frames == []


@pytest.mark.anyio
async def test_concurrent_duplicate_preparations_only_write_once():
    client, _, prepare, args = rig()
    both_preparing = asyncio.Event()
    release = asyncio.Event()
    count = 0

    async def prepare_write():
        nonlocal count
        count += 1
        if count == 2:
            both_preparing.set()
        await release.wait()
        return True

    prepare.side_effect = prepare_write
    tasks = [asyncio.create_task(accept_inbox_input(client.run, **args)) for _ in range(2)]
    try:
        await both_preparing.wait()
        release.set()
        receipts = await asyncio.gather(*tasks)
        assert all(receipt.outcome == "accepted" for receipt in receipts)
        assert len(client.frames) == 1
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.anyio
async def test_completed_run_can_return_old_acceptance_without_new_write():
    client, queue, prepare, args = rig()
    assert (await accept_inbox_input(client.run, **args)).outcome == "accepted"
    client.run.note_lifecycle(BOT, "completed")
    client.run.mark_completed()
    queue.pop_active_client("session")
    assert (await accept_inbox_input(client.run, **args)).outcome == "accepted"
    assert len(client.frames) == 1
    prepare.assert_awaited_once()


@pytest.mark.anyio
async def test_uses_existing_raw_lifecycle_consumer_not_a_second_sdk_reader():
    from tests.test_claude_active_run_next_steer import _FakeClient, _finish, _run, _start

    client = _FakeClient()
    events = []
    run = _run(client, events=events)
    _, consumer = await _start(run)
    _, _, _, args = rig()
    args["owns_run"] = lambda: True
    try:
        assert (await accept_inbox_input(run, **args)).outcome == "accepted"
        [event] = events
        assert event["message_id"] == BOT and event["origin"] == "bot"
        assert event["steer_rpc_id"] == args["steer_rpc_id"]
        assert event["run_request_id"] == run.request_id
    finally:
        await _finish(client, consumer)


@pytest.mark.anyio
@pytest.mark.parametrize("bad", [{"command_uuid": "not-a-uuid"}, {"steer_rpc_id": ""},
                                 {"framed_message": ""}, {"interrupt_item": ""}])
async def test_invalid_identity_or_unframed_input_fails_before_prepare(bad):
    client, _, prepare, args = rig()
    with pytest.raises(ValueError):
        await accept_inbox_input(client.run, **{**args, **bad})
    assert client.frames == []
    prepare.assert_not_awaited()
