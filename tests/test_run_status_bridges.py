"""TASK-1232: every bridge answers ``run.status`` from its own request registry."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agent_bridge.run_status import RUN_STATUS_METHOD, RunState, RunStatusReply
from agent_bridge.session_queue import RequestPhase
from tests.test_claude_code_send_boundaries import SESSION, _SDKStream, _SendHarness
from tests.test_other_bridge_abort_targeting import Gateway, abort, make_bridge, send, until

BACKENDS = ["claude-code", "codex", "local", "openclaw"]
UNSET = object()
OTHER = "bot:user"  # the session make_bridge's send() uses


def _bridge(kind):
    if kind == "claude-code":
        return _SendHarness()
    bridge = make_bridge(kind, Gateway() if kind == "openclaw" else None)
    if kind == "codex":
        # A backend-less RPC makes codex look up bot ownership; claim it so the
        # backend check under test is the only thing that can drop the call.
        bridge._bot_uses_codex = AsyncMock(return_value=True)
    return bridge


async def _call(bridge, target, backend=UNSET, session=SESSION):
    """Send one run.status; return the published reply, or None if unanswered."""
    handler = getattr(bridge, "_handle_rpc", None) or bridge._handle_rpc_command
    fields = {"request_id": "status-rpc", "method": RUN_STATUS_METHOD,
              "params": {"sessionKey": session, "requestId": target}}
    if backend is not UNSET:
        fields["backend"] = backend
    bridge._publisher.publish_rpc_result.reset_mock()
    redis = SimpleNamespace(xack=AsyncMock())
    await asyncio.wait_for(handler(fields, "status-msg", redis), timeout=1)
    redis.xack.assert_awaited_once()
    calls = bridge._publisher.publish_rpc_result.call_args_list
    assert len(calls) <= 1
    return calls[0].args[1] if calls else None


async def _state(bridge, target, kind, session=SESSION):
    reply = await _call(bridge, target, kind, session)
    assert reply is not None
    parsed = RunStatusReply.from_wire(reply, backend=kind)
    assert parsed.request_id == target
    return parsed.state


def _run(coro):
    asyncio.run(asyncio.wait_for(coro, timeout=5))


@pytest.mark.parametrize("kind", BACKENDS)
def test_each_phase_of_a_registered_send_is_reported(kind):
    async def run():
        bridge = _bridge(kind)
        queue = bridge._session_queue
        gate = asyncio.Event()
        entered = {name: asyncio.Event() for name in ("pre", "run")}

        async def preflight():
            queue.accept(SESSION, "pre")
            entered["pre"].set()
            await gate.wait()

        async def running():
            async with queue.active(SESSION, request_id="run"):
                entered["run"].set()
                await gate.wait()

        async def queued():
            async with queue.active(SESSION, request_id="queued"):
                pass

        tasks = [asyncio.create_task(preflight()), asyncio.create_task(running())]
        await entered["pre"].wait()
        await entered["run"].wait()
        tasks.append(asyncio.create_task(queued()))
        await until(lambda: queue.request_phase(SESSION, "queued") is not None)
        assert await _state(bridge, "pre", kind) is RunState.PREFLIGHT
        assert await _state(bridge, "run", kind) is RunState.RUNNING
        assert await _state(bridge, "queued", kind) is RunState.QUEUED
        gate.set()
        await asyncio.gather(*tasks)
        assert await _state(bridge, "run", kind) is RunState.NOT_FOUND
    _run(run())


@pytest.mark.parametrize("kind", BACKENDS)
def test_a_cancelled_send_still_tearing_down_is_stopping(kind):
    async def run():
        bridge = _bridge(kind)
        entered, teardown = asyncio.Event(), asyncio.Event()

        async def hold():
            async with bridge._session_queue.active(SESSION, request_id="r"):
                entered.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    await teardown.wait()
                    raise

        task = asyncio.create_task(hold())
        await entered.wait()
        assert bridge._session_queue.cancel_request(SESSION, "r")
        await asyncio.sleep(0)
        assert await _state(bridge, "r", kind) is RunState.STOPPING
        teardown.set()
        await asyncio.gather(task, return_exceptions=True)
        assert await _state(bridge, "r", kind) is RunState.NOT_FOUND
    _run(run())


@pytest.mark.parametrize("kind", BACKENDS)
def test_unknown_request_is_not_found_and_a_stopped_one_is_tombstoned(kind):
    async def run():
        bridge = _bridge(kind)
        assert await _state(bridge, "missing", kind) is RunState.NOT_FOUND
        if kind == "claude-code":
            await bridge.abort("missing")
        else:
            await abort(bridge, "missing", SimpleNamespace(xack=AsyncMock()), SESSION)
        assert await _state(bridge, "missing", kind) is RunState.TOMBSTONED
    _run(run())


@pytest.mark.parametrize("kind", BACKENDS)
@pytest.mark.parametrize("backend", [UNSET, "", "someone-else"])
def test_run_status_not_addressed_to_this_bridge_is_unanswered(kind, backend):
    async def run():
        bridge = _bridge(kind)
        entered = asyncio.Event()

        async def hold():
            async with bridge._session_queue.active(SESSION, request_id="r"):
                entered.set()
                await asyncio.Event().wait()

        task = asyncio.create_task(hold())
        await entered.wait()
        assert await _call(bridge, "r", backend) is None
        if kind == "openclaw":
            assert bridge._ws_client.abort_calls == []  # never forwarded to the gateway
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    _run(run())


def test_claude_real_sends_report_running_then_queued(monkeypatch):
    async def run():
        harness = _SendHarness()
        sdk = _SDKStream()
        monkeypatch.setattr("claude_code_bridge.send_handler.ClaudeSDKClient", lambda **kw: sdk)
        first = harness.send("first")
        await sdk.waiting.wait()
        second = harness.send("second")
        await until(lambda: harness._session_queue.request_phase(SESSION, "second") is RequestPhase.QUEUED)
        assert await _state(harness, "first", "claude-code") is RunState.RUNNING
        assert await _state(harness, "second", "claude-code") is RunState.QUEUED
        for rid in ("second", "first"):
            await harness.abort(rid)
        await asyncio.gather(first, second, return_exceptions=True)
        for rid in ("first", "second"):
            assert await _state(harness, rid, "claude-code") is RunState.TOMBSTONED
    _run(run())


@pytest.mark.parametrize("kind", ["codex", "local", "openclaw"])
def test_a_real_send_behind_another_reports_queued(kind):
    async def run():
        bridge = _bridge(kind)
        redis = SimpleNamespace(xack=AsyncMock())
        gate, entered = asyncio.Event(), asyncio.Event()

        async def hold():
            async with bridge._session_queue.active(OTHER, request_id="active"):
                entered.set()
                await gate.wait()

        active = asyncio.create_task(hold())
        await entered.wait()
        queued = asyncio.create_task(send(bridge, "queued", redis))
        await until(lambda: bridge._session_queue.request_phase(OTHER, "queued") is not None)
        assert await _state(bridge, "queued", kind, OTHER) is RunState.QUEUED
        await abort(bridge, "queued", redis)
        await asyncio.gather(queued, return_exceptions=True)
        assert await _state(bridge, "queued", kind, OTHER) is RunState.TOMBSTONED
        gate.set()
        await active
    _run(run())


def test_openclaw_is_stopping_while_its_gateway_abort_is_pending():
    async def run():
        bridge = _bridge("openclaw")
        ws = bridge._ws_client
        redis = SimpleNamespace(xack=AsyncMock())
        task = asyncio.create_task(send(bridge, "active", redis))
        await until(lambda: bridge._request_runs.get((OTHER, "active"), {}).get("run_id"))
        assert await _state(bridge, "active", "openclaw", OTHER) is RunState.RUNNING
        rpc = asyncio.create_task(abort(bridge, "active", redis))
        await ws.abort_started.wait()
        assert await _state(bridge, "active", "openclaw", OTHER) is RunState.STOPPING
        ws.abort_release.set()
        await rpc
        ws.release.set()
        await asyncio.gather(task, return_exceptions=True)
        assert await _state(bridge, "active", "openclaw", OTHER) is RunState.TOMBSTONED
    _run(run())


@pytest.mark.parametrize("kind", BACKENDS)
def test_run_status_read_after_a_send_never_overtakes_its_registration(kind):
    # The listeners start tasks in stream order: send first, then the rpc.
    # accept() must run before the send's first await, or a run.status read
    # right behind the send answers not_found for a live request.
    async def run():
        bridge = _bridge(kind)
        session = SESSION if kind == "claude-code" else OTHER
        gate, held = asyncio.Event(), asyncio.Event()

        async def hold():
            async with bridge._session_queue.active(session, request_id="active"):
                held.set()
                await gate.wait()

        active = asyncio.create_task(hold())
        await held.wait()
        if kind == "claude-code":
            sending = bridge.send("next")
        else:
            sending = asyncio.create_task(send(bridge, "next", SimpleNamespace(xack=AsyncMock())))
        status = asyncio.create_task(_state(bridge, "next", kind, session))
        assert await status in (RunState.PREFLIGHT, RunState.QUEUED)
        sending.cancel()
        gate.set()
        await asyncio.gather(sending, active, return_exceptions=True)
    _run(run())
