"""TASK-1231B: a bridge shutdown cancel is stamped ``abort_cause=host_shutdown``.

Before this, SIGTERM left fire-and-forget sends to be cancelled by
``asyncio.run`` teardown after the publisher closed, and the terminal they did
publish was indistinguishable from a user Stop.
"""

from __future__ import annotations

import asyncio
from unittest.mock import Mock

from agent_bridge.abort_contract import HOST_SHUTDOWN
from agent_bridge.session_queue import SessionQueue
from claude_code_bridge.bridge import ClaudeCodeBridge
from tests.test_claude_code_send_boundaries import (
    CANCELLED,
    SESSION,
    _SDKStream,
    _SendHarness,
)

SHUTDOWN = {**CANCELLED, "abort_cause": HOST_SHUTDOWN}


def _terminal_raw(harness, request_id):
    terminal, = harness.terminal(request_id)
    assert harness._publisher.publish_run_done.call_args_list.count(((request_id,),)) == 1
    return terminal


def test_shutdown_stamps_running_and_queued_sends(monkeypatch):
    async def run():
        harness = _SendHarness()
        sdk = _SDKStream()
        monkeypatch.setattr("claude_code_bridge.send_handler.ClaudeSDKClient", lambda **kw: sdk)
        running = harness.send()
        await sdk.waiting.wait()
        queued = harness.send("queued")
        await harness.queued.wait()
        assert len(harness._session_queue.cancel_all(cause=HOST_SHUTDOWN)) == 2
        await asyncio.gather(running, queued, return_exceptions=True)
        active = _terminal_raw(harness, "turn-request")
        assert active.raw == SHUTDOWN
        assert active.text == "partial answer"  # partial kept, still aborted
        assert _terminal_raw(harness, "queued").raw == SHUTDOWN
    asyncio.run(asyncio.wait_for(run(), timeout=5))


def test_user_abort_keeps_historical_shape_through_a_later_shutdown(monkeypatch):
    async def run():
        harness = _SendHarness()
        sdk = _SDKStream(delay_disconnect=True)
        monkeypatch.setattr("claude_code_bridge.send_handler.ClaudeSDKClient", lambda **kw: sdk)
        send = harness.send()
        await sdk.waiting.wait()
        abort = asyncio.create_task(harness.abort())
        try:
            await sdk.disconnect_entered.wait()
            # Shutdown arrives while the user Stop is still cleaning up.
            harness._session_queue.cancel_all(cause=HOST_SHUTDOWN)
            for _ in range(5):
                await asyncio.sleep(0)
            # The terminal was published before the teardown; the send itself
            # keeps the session until its teardown ends (TASK-1232 P4, PE1).
            assert _terminal_raw(harness, "turn-request").raw == CANCELLED
            assert not send.done()
        finally:
            sdk.release_disconnect.set()
            await asyncio.gather(abort, send, return_exceptions=True)
    asyncio.run(asyncio.wait_for(run(), timeout=5))


def test_shutdown_during_sdk_setup_is_stamped(monkeypatch):
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
        harness._session_queue.cancel_all(cause=HOST_SHUTDOWN)
        await asyncio.gather(send, return_exceptions=True)
        assert _terminal_raw(harness, "turn-request").raw == SHUTDOWN
    asyncio.run(asyncio.wait_for(run(), timeout=5))


# -- ClaudeCodeBridge.stop ordering -------------------------------------------


def _bridge(order):
    bridge = ClaudeCodeBridge.__new__(ClaudeCodeBridge)
    bridge._session_queue = SessionQueue()
    bridge._command_task = bridge._cleanup_task = bridge._approval_reload_task = None
    bridge._publisher = Mock()
    bridge._publisher.close.side_effect = lambda: order.append("publisher.close")
    return bridge


def _register_send(bridge, order, *, cleanup=None):
    """Register one send; its cancel handler records the cause it saw.

    Callers must ``_finish`` the task so a regression fails the test instead
    of leaving a never-ending cleanup for ``asyncio.run`` teardown.
    """
    entered = asyncio.Event()

    async def on_cancel():
        if cleanup is not None:
            await cleanup()
        cause = bridge._session_queue.cancel_cause(SESSION, "r")
        order.append(f"terminal:{cause}")

    async def send():
        async with bridge._session_queue.active(SESSION, request_id="r", on_cancel=on_cancel):
            entered.set()
            await asyncio.Event().wait()

    return asyncio.create_task(send()), entered


async def _finish(task, *events):
    for event in events:
        event.set()
    task.cancel()
    await asyncio.wait({task}, timeout=1)


def test_stop_publishes_shutdown_terminal_before_closing_publisher():
    async def run():
        order: list[str] = []
        bridge = _bridge(order)
        task, entered = _register_send(bridge, order)
        try:
            await entered.wait()
            await bridge.stop()
            assert task.cancelled()
            assert order == [f"terminal:{HOST_SHUTDOWN}", "publisher.close"]
        finally:
            await _finish(task)
    asyncio.run(asyncio.wait_for(run(), timeout=5))


def test_stop_closes_the_listener_before_cancelling_sends():
    # A send the listener admits after cancel_all would never be stamped and
    # would outlive the publisher (Loopy, TASK-1231 r1).
    async def run():
        order: list[str] = []
        bridge = _bridge(order)
        task, entered = _register_send(bridge, order)
        late: list[asyncio.Task] = []

        async def listener():
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                order.append("listener.stopped")
                raise

        def admit_late_send(_task):
            # What a live listener does on the next chat.send.
            late.append(_register_send(bridge, order)[0])

        bridge._command_task = asyncio.create_task(listener())
        try:
            await entered.wait()
            task.add_done_callback(lambda t: bridge._command_task.done() or admit_late_send(t))
            await bridge.stop()
            assert late == []
            assert order == ["listener.stopped", f"terminal:{HOST_SHUTDOWN}", "publisher.close"]
        finally:
            await _finish(task)
            for extra in late:
                await _finish(extra)
    asyncio.run(asyncio.wait_for(run(), timeout=5))


def test_concurrent_stop_is_single_flight():
    async def run():
        order: list[str] = []
        bridge = _bridge(order)
        release = asyncio.Event()
        task, entered = _register_send(bridge, order, cleanup=release.wait)
        first = second = None
        try:
            await entered.wait()
            first = asyncio.create_task(bridge.stop())
            second = asyncio.create_task(bridge.stop())
            await asyncio.sleep(0.05)
            # Neither caller may close the publisher while publication is pending.
            assert order == []
            release.set()
            await asyncio.wait_for(asyncio.gather(first, second), timeout=2)
            assert order == [f"terminal:{HOST_SHUTDOWN}", "publisher.close"]
            bridge._publisher.close.assert_called_once()
            assert task.cancelled()
        finally:
            await _finish(task, release)
            for stopper in (first, second):
                if stopper is not None:
                    await _finish(stopper)
    asyncio.run(asyncio.wait_for(run(), timeout=5))


def test_stop_bounds_wait_on_a_stuck_send(monkeypatch):
    async def run():
        order: list[str] = []
        bridge = _bridge(order)
        monkeypatch.setattr(ClaudeCodeBridge, "SHUTDOWN_SEND_GRACE_SECONDS", 0.05)
        stuck = asyncio.Event()
        task, entered = _register_send(bridge, order, cleanup=stuck.wait)
        try:
            await entered.wait()
            await asyncio.wait_for(bridge.stop(), timeout=2)
            assert task.cancelling()  # shutdown did cancel it
            assert order == ["publisher.close"]
        finally:
            await _finish(task, stuck)
    asyncio.run(asyncio.wait_for(run(), timeout=5))
