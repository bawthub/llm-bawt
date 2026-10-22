"""TASK-1232 P2: a send is registered at receipt; a Stop before or during
preflight yields exactly one cancelled terminal and no reset/seed side effect."""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from agent_bridge.abort_contract import HOST_SHUTDOWN, aborted_terminal_raw
from agent_bridge.command_publisher import (
    COMMAND_DEDUPE_TTL_S,
    DELIVERY_COMMAND_DEDUPE_TTL_S,
    CommandPublisherMixin,
    command_dedupe_ttl_s,
)
from agent_bridge.events import AgentEventKind
from agent_bridge.request_tombstones import RequestTombstones
from agent_bridge.run_contract import UNATTEMPTED, TeardownReport
from agent_bridge.session_queue import Admission, RequestPhase, SessionQueue
from claude_code_bridge.session_ops import ClaudeSessionMixin
from tests.test_claude_code_send_boundaries import CANCELLED, MODEL, SESSION, _SendHarness
from tests.test_other_bridge_abort_targeting import abort, make_bridge, send, until

# -- Claude bridge -----------------------------------------------------------

CLAUDE_SENDS = {
    "bare_new": {"message": "/new"},
    "reset_only": {"message": "/new", "bridge_reset_action": "reset_only",
                   "thread_session_id": "thread-1"},
    "new_with_message": {"message": "/new do the thing"},
    "normal": {"message": "hello"},
}


def _claude_fields(request_id, **extra):
    return {"request_id": request_id, "session_key": SESSION, "model": MODEL,
            "message": "hello", "trigger_message_id": f"message-{request_id}", **extra}


def _claude_harness():
    harness = _SendHarness()
    harness._publish_session_reset_unified = Mock()
    harness._run_send = AsyncMock()
    return harness


def _send_xacks(harness, msg_id):
    return [c.args[-1] for c in harness.redis.xack.call_args_list].count(msg_id)


def _assert_one_cancelled_terminal(harness, request_id):
    terminal, = harness.terminal(request_id)
    assert terminal.raw == CANCELLED
    assert terminal.trigger_message_id == f"message-{request_id}"
    # TASK-1233: refused or stopped before the SDK ran, so no teardown was
    # attempted (a /new seed's CLI may have been in flight).
    harness._publisher.publish_run_done.assert_called_once_with(request_id, teardown=UNATTEMPTED)
    assert _send_xacks(harness, f"send-{request_id}") == 1
    assert harness._session_queue.request_phase(SESSION, request_id) is None
    assert (SESSION, request_id) not in harness._session_queue._request_tasks
    assert request_id not in harness._trigger_message_ids


@pytest.mark.parametrize("kind", list(CLAUDE_SENDS))
def test_claude_send_tombstoned_before_receipt_is_refused_without_side_effects(kind):
    async def run():
        harness = _claude_harness()
        await harness.abort("r1")
        await harness._handle_send(_claude_fields("r1", **CLAUDE_SENDS[kind]), "send-r1", harness.redis)
        _assert_one_cancelled_terminal(harness, "r1")
        harness._preprocess_new_command.assert_not_awaited()
        harness._publish_session_reset_unified.assert_not_called()
        harness._run_send.assert_not_awaited()
    asyncio.run(asyncio.wait_for(run(), timeout=5))


@pytest.mark.parametrize("method", ["chat.abort", "chat.cancel"])
def test_claude_cancel_rpc_also_tombstones(method):
    async def run():
        harness = _claude_harness()
        await harness.abort("r1", method=method)
        assert harness._session_queue.is_tombstoned(SESSION, "r1")
        await harness._handle_send(_claude_fields("r1"), "send-r1", harness.redis)
        _assert_one_cancelled_terminal(harness, "r1")
        harness._run_send.assert_not_awaited()
    asyncio.run(asyncio.wait_for(run(), timeout=5))


@pytest.mark.parametrize("kind", list(CLAUDE_SENDS))
def test_claude_stop_mid_preflight_publishes_one_cancelled_terminal(kind):
    async def run():
        harness = _claude_harness()
        entered = asyncio.Event()

        async def preprocess(message, **kw):
            # The reset is published before the seed's await, like the real path.
            harness._publish_session_reset_unified(kw["bot_slug"], kw["session_key"], had_session=True)
            entered.set()
            await asyncio.Event().wait()

        harness._preprocess_new_command = AsyncMock(side_effect=preprocess)
        task = asyncio.create_task(
            harness._handle_send(_claude_fields("r1", **CLAUDE_SENDS[kind]), "send-r1", harness.redis))
        await entered.wait()
        assert harness._session_queue.request_phase(SESSION, "r1") is RequestPhase.PREFLIGHT
        await harness.abort("r1")
        await asyncio.gather(task, return_exceptions=True)
        assert task.cancelled()
        _assert_one_cancelled_terminal(harness, "r1")
        # A reset already published stays published; nothing runs after it.
        harness._publish_session_reset_unified.assert_called_once()
        harness._run_send.assert_not_awaited()
    asyncio.run(asyncio.wait_for(run(), timeout=5))


def test_claude_stop_at_bare_new_ack_xack_does_not_repeat_its_terminal():
    """Bare /new publishes its own terminal; a Stop on its xack adds none."""
    async def run():
        harness = _claude_harness()
        session_ops = ClaudeSessionMixin
        harness._preprocess_new_command = session_ops._preprocess_new_command.__get__(harness)
        harness._format_seed_ack = session_ops._format_seed_ack
        harness._seed_usage_estimate = session_ops._seed_usage_estimate
        harness._seed_new_session = AsyncMock(return_value={"session_id": "sdk-new", "clean_start": True})
        acking = asyncio.Event()

        async def xack(stream, group, msg_id):
            if msg_id == "send-r1":
                acking.set()
                await asyncio.Event().wait()

        harness.redis = SimpleNamespace(xack=AsyncMock(side_effect=xack))
        task = asyncio.create_task(
            harness._handle_send(_claude_fields("r1", message="/new"), "send-r1", harness.redis))
        await acking.wait()
        await harness.abort("r1")
        await asyncio.gather(task, return_exceptions=True)
        done = harness.terminal("r1")
        assert len(done) == 1 and "new_session_ack" in done[0].raw
        run_done = [c.args[0] for c in harness._publisher.publish_run_done.call_args_list]
        assert run_done.count("r1") == 1
        assert (SESSION, "r1") not in harness._session_queue._request_tasks
    asyncio.run(asyncio.wait_for(run(), timeout=5))


def test_claude_missing_model_exit_releases_registration():
    async def run():
        harness = _claude_harness()
        await harness._handle_send(_claude_fields("r1", model=""), "send-r1", harness.redis)
        errors = [e for e in harness.events if e.kind == AgentEventKind.ERROR]
        assert len(errors) == 1
        harness._publisher.publish_run_done.assert_called_once_with(
            "r1", teardown=TeardownReport(exited=True))
        assert (SESSION, "r1") not in harness._session_queue._request_tasks
        harness._run_send.assert_not_awaited()
    asyncio.run(asyncio.wait_for(run(), timeout=5))


def test_claude_bad_structured_reset_releases_registration():
    async def run():
        harness = _claude_harness()
        with pytest.raises(ValueError, match="Invalid structured reset"):
            await harness._handle_send(
                _claude_fields("r1", bridge_reset_action="bogus", thread_session_id="t"),
                "send-r1", harness.redis)
        assert (SESSION, "r1") not in harness._session_queue._request_tasks
    asyncio.run(asyncio.wait_for(run(), timeout=5))


def test_claude_duplicate_live_send_is_dropped_without_publishing():
    async def run():
        harness = _claude_harness()
        release = asyncio.Event()

        async def run_send(preflight, msg_id, redis, terminal):
            await release.wait()

        harness._run_send = AsyncMock(side_effect=run_send)
        first = asyncio.create_task(harness._handle_send(_claude_fields("r1"), "send-r1", harness.redis))
        await until(lambda: harness._run_send.await_count == 1)
        await harness._handle_send(_claude_fields("r1"), "send-r1-copy", harness.redis)
        assert _send_xacks(harness, "send-r1-copy") == 1
        assert harness._run_send.await_count == 1
        assert harness.terminal("r1") == []
        assert harness._session_queue._request_tasks[(SESSION, "r1")] is first
        release.set()
        await first
        assert (SESSION, "r1") not in harness._session_queue._request_tasks
    asyncio.run(asyncio.wait_for(run(), timeout=5))


def test_claude_host_shutdown_cancels_a_preflight_send():
    async def run():
        harness = _claude_harness()
        entered = asyncio.Event()

        async def preprocess(message, **kw):
            entered.set()
            await asyncio.Event().wait()

        harness._preprocess_new_command = AsyncMock(side_effect=preprocess)
        task = asyncio.create_task(harness._handle_send(_claude_fields("r1"), "send-r1", harness.redis))
        await entered.wait()
        assert harness._session_queue.cancel_all(cause=HOST_SHUTDOWN) == [task]
        await asyncio.gather(task, return_exceptions=True)
        terminal, = harness.terminal("r1")
        assert terminal.raw == aborted_terminal_raw(HOST_SHUTDOWN)
        harness._publisher.publish_run_done.assert_called_once_with("r1", teardown=UNATTEMPTED)
    asyncio.run(asyncio.wait_for(run(), timeout=5))


# -- Other bridges ----------------------------------------------------------

OTHER = ["codex", "local", "openclaw"]


def _xacks(redis, msg_id):
    return [c.args[-1] for c in redis.xack.call_args_list].count(msg_id)


@pytest.mark.parametrize("message", ["/new", "/new do it", "hello"])
@pytest.mark.parametrize("kind", OTHER)
def test_other_bridge_tombstoned_send_is_refused_without_side_effects(kind, message):
    async def run():
        bridge = make_bridge(kind)
        bridge._publish_session_reset_unified = MagicMock()
        bridge._reset_openclaw_session = AsyncMock(return_value=True)
        redis = SimpleNamespace(xack=AsyncMock())
        assert (await abort(bridge, "r1", redis))["state"] == "tombstoned"
        await send(bridge, "r1", redis, message=message)
        bridge._publisher.publish_run_done.assert_called_once_with("r1", teardown=UNATTEMPTED)
        bridge._publisher.publish_run_accepted.assert_not_called()
        assert _xacks(redis, "r1") == 1
        bridge._publish_session_reset_unified.assert_not_called()
        bridge._reset_openclaw_session.assert_not_awaited()
        assert not bridge._publisher.publish_run_event.called
        if kind != "openclaw":
            bridge._publish_event.assert_not_called()
        assert ("bot:user", "r1") not in bridge._session_queue._request_tasks
    asyncio.run(asyncio.wait_for(run(), timeout=5))


@pytest.mark.parametrize("message", ["/new", "/new do it"])
def test_openclaw_stop_during_new_reset_publishes_one_terminal(message):
    async def run():
        bridge = make_bridge("openclaw")
        bridge._publish_session_reset_unified = MagicMock()
        bridge._discard_changed_file_request = MagicMock()
        entered = asyncio.Event()

        async def reset(session_key):
            entered.set()
            await asyncio.Event().wait()

        bridge._reset_openclaw_session = AsyncMock(side_effect=reset)
        redis = SimpleNamespace(xack=AsyncMock())
        task = asyncio.create_task(send(bridge, "r1", redis, message=message))
        await entered.wait()
        assert bridge._session_queue.request_phase("bot:user", "r1") is RequestPhase.PREFLIGHT
        assert (await abort(bridge, "r1", redis))["state"] == "cancelled"
        await asyncio.gather(task, return_exceptions=True)
        assert task.cancelled()
        # Stopped in preflight: the gateway never started a run for it.
        bridge._publisher.publish_run_done.assert_called_once_with(
            "r1", teardown=TeardownReport(exited=True))
        assert _xacks(redis, "r1") == 1
        assert not bridge._publisher.publish_run_event.called
        assert ("bot:user", "r1") not in bridge._session_queue._request_tasks
    asyncio.run(asyncio.wait_for(run(), timeout=5))


@pytest.mark.parametrize("unavailable_run_id", [False, True])
def test_openclaw_failed_abort_of_a_running_request_still_tombstones(unavailable_run_id):
    """The Stop is recorded even when the gateway abort fails (bridge
    ``_abort_request`` tombstones before reading run state)."""
    from tests.test_other_bridge_abort_targeting import Gateway

    async def run():
        ws = Gateway()
        ws.no_events = unavailable_run_id
        if unavailable_run_id:
            ws.send_ack.clear()
        ws.rejected = True
        ws.abort_release.set()
        bridge = make_bridge("openclaw", ws)
        bridge._abort_run_id_timeout_s = 0.01
        bridge._wait_for_history_reply = AsyncMock(return_value=None)
        redis = SimpleNamespace(xack=AsyncMock())
        task = asyncio.create_task(send(bridge, "active", redis))
        await ws.started.wait()
        assert ("bot:user", "active") in bridge._request_runs
        assert (await abort(bridge, "active", redis))["state"] == "error"
        assert bridge._session_queue.is_tombstoned("bot:user", "active")
        ws.send_ack.set()
        ws.release.set()
        await task
        ws.started.clear()
        run_dones = bridge._publisher.publish_run_done.call_count
        await send(bridge, "active", redis, message="hello again")
        assert not ws.started.is_set()
        assert bridge._publisher.publish_run_done.call_count == run_dones + 1
        assert ("bot:user", "active") not in bridge._session_queue._request_tasks
    asyncio.run(asyncio.wait_for(run(), timeout=5))


@pytest.mark.parametrize("kind", OTHER)
def test_other_bridge_validation_exit_releases_registration(kind):
    async def run():
        bridge = make_bridge(kind)
        redis = SimpleNamespace(xack=AsyncMock())
        await send(bridge, "r1", redis, message="")
        assert _xacks(redis, "r1") == 1
        # TASK-1233: an invalid send still ends with exactly one done.
        bridge._publisher.publish_run_done.assert_called_once_with(
            "r1", teardown=TeardownReport(exited=True))
        assert ("bot:user", "r1") not in bridge._session_queue._request_tasks
    asyncio.run(asyncio.wait_for(run(), timeout=5))


# -- SessionQueue -------------------------------------------------------------

def test_accept_registers_at_receipt_and_active_reuses_it():
    async def run():
        queue = SessionQueue()
        phases = []

        async def owner():
            assert queue.accept("s", "r") is Admission.ACCEPTED
            phases.append(queue.request_phase("s", "r"))
            try:
                async with queue.active("s", request_id="r"):
                    phases.append(queue.request_phase("s", "r"))
                # Still registered after the lock: release() owns it.
                phases.append(queue.request_phase("s", "r"))
            finally:
                queue.release("s", "r")
            phases.append(queue.request_phase("s", "r"))

        await owner()
        assert phases == [RequestPhase.PREFLIGHT, RequestPhase.RUNNING, RequestPhase.RUNNING, None]
        assert queue._request_tasks == {} and queue._request_phases == {}
    asyncio.run(run())


def test_queued_phase_and_unaccepted_active_still_cleans_up():
    async def run():
        queue = SessionQueue()
        hold = asyncio.Event()

        async def holder():
            async with queue.active("s", request_id="a"):
                await hold.wait()

        first = asyncio.create_task(holder())
        await until(lambda: queue.is_busy("s"))
        second = asyncio.create_task(holder_for(queue, "b"))
        await until(lambda: queue.request_phase("s", "b") is RequestPhase.QUEUED)
        assert queue.request_phase("s", "a") is RequestPhase.RUNNING
        hold.set()
        await asyncio.gather(first, second)
        assert queue._request_tasks == {} and queue._request_phases == {}
    asyncio.run(run())


async def holder_for(queue, request_id):
    async with queue.active("s", request_id=request_id):
        pass


def test_accept_refuses_tombstone_and_live_duplicate():
    async def run():
        queue = SessionQueue()
        assert queue.stop_request("s", "gone") is False
        assert queue.accept("s", "gone") is Admission.TOMBSTONED
        assert ("s", "gone") not in queue._request_tasks
        assert queue.accept("s", "live") is Admission.ACCEPTED
        dup = await asyncio.create_task(_accept(queue, "s", "live"))
        assert dup is Admission.DUPLICATE
        assert queue._request_tasks[("s", "live")] is asyncio.current_task()
        # Another session's id is independent.
        assert queue.accept("other", "gone") is Admission.ACCEPTED
    asyncio.run(run())


async def _accept(queue, session_key, request_id):
    return queue.accept(session_key, request_id)


def test_release_by_a_non_owner_keeps_the_registration():
    async def run():
        queue = SessionQueue()
        assert queue.accept("s", "r") is Admission.ACCEPTED
        await asyncio.create_task(_release(queue))
        assert ("s", "r") in queue._request_tasks
    asyncio.run(run())


async def _release(queue):
    queue.release("s", "r")


def test_accept_requires_a_task():
    async def run():
        outcome = asyncio.get_running_loop().create_future()

        def from_callback():
            try:
                SessionQueue().accept("s", "r")
            except RuntimeError as exc:
                outcome.set_result(str(exc))

        asyncio.get_running_loop().call_soon(from_callback)
        assert "send's own task" in await outcome
    asyncio.run(run())


def test_stop_request_tombstones_then_cancels():
    async def run():
        queue = SessionQueue()
        entered = asyncio.Event()

        async def owner():
            queue.accept("s", "r")
            try:
                entered.set()
                await asyncio.Event().wait()
            finally:
                queue.release("s", "r")

        task = asyncio.create_task(owner())
        await entered.wait()
        assert queue.stop_request("s", "r") is True
        await asyncio.gather(task, return_exceptions=True)
        assert task.cancelled() and queue.is_tombstoned("s", "r")
        assert queue.accept("s", "r") is Admission.TOMBSTONED
    asyncio.run(run())


# -- Tombstone store ----------------------------------------------------------

class _Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def test_tombstone_lives_for_the_dedupe_ttl():
    clock = _Clock()
    store = RequestTombstones(clock=clock)
    store.add("s", "req_1")
    store.add("s", "req_delivery_1")
    clock.now += COMMAND_DEDUPE_TTL_S - 1
    assert store.contains("s", "req_1") and store.contains("s", "req_delivery_1")
    clock.now += 1
    assert not store.contains("s", "req_1")
    assert store.contains("s", "req_delivery_1")
    clock.now += DELIVERY_COMMAND_DEDUPE_TTL_S - COMMAND_DEDUPE_TTL_S
    assert not store.contains("s", "req_delivery_1")
    assert len(store) == 0


def test_repeat_stop_renews_the_tombstone():
    clock = _Clock()
    store = RequestTombstones(clock=clock)
    store.add("s", "r")
    clock.now += COMMAND_DEDUPE_TTL_S - 1
    store.add("s", "r")
    clock.now += 2
    assert store.contains("s", "r")


def test_tombstone_cap_evicts_oldest_and_logs(caplog):
    store = RequestTombstones(cap=2, clock=_Clock())
    with caplog.at_level(logging.WARNING, logger="agent_bridge.request_tombstones"):
        for rid in ("a", "b", "c"):
            store.add("s", rid)
    assert not store.contains("s", "a")
    assert store.contains("s", "b") and store.contains("s", "c")
    assert len(store) == 2
    assert "evicted before its TTL" in caplog.text and "request=a" in caplog.text


def test_tombstone_cap_must_be_positive():
    with pytest.raises(ValueError):
        RequestTombstones(cap=0)


def test_dedupe_script_ttls_match_the_tombstone_ttls():
    captured = {}

    class _Redis:
        async def eval(self, script, *args):
            captured["script"] = script
            return 1, "1-0"

    publisher = CommandPublisherMixin()
    publisher._pub_redis = _Redis()
    asyncio.run(publisher.send_command("s", "hello", "req_delivery_x"))
    script = captured["script"]
    assert f"local ttl = {COMMAND_DEDUPE_TTL_S}" in script
    assert (f"string.sub(ARGV[2], 1, 13) == 'req_delivery_' then ttl = "
            f"{DELIVERY_COMMAND_DEDUPE_TTL_S}") in script
    assert command_dedupe_ttl_s("req_delivery_x") == DELIVERY_COMMAND_DEDUPE_TTL_S == 604800
    assert command_dedupe_ttl_s("req_x") == COMMAND_DEDUPE_TTL_S == 86400
