"""TASK-1233: the bridge run contract (accepted, terminal, teardown report)."""

from __future__ import annotations

import asyncio
import json
import logging
import signal
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from agent_bridge.abort_contract import HOST_SHUTDOWN, process_boot_id, process_is_pid1
from agent_bridge.events import AgentEventKind
from agent_bridge.publisher import RUN_STREAM_PREFIX, RedisPublisher
from agent_bridge.run_contract import (
    QUIESCE_SWEEP_INTERVAL_MAX_S,
    QUIESCE_TIMEOUT_MAX_S,
    REFUSED,
    RUN_TERMINAL_RETENTION_S,
    UNATTEMPTED,
    RunAccepted,
    RunContractError,
    RunTerminal,
    TeardownReport,
    run_terminal_key,
)
from agent_bridge.send_admission import RunOutcome, admit_send
from agent_bridge.session_queue import RequestPhase, SessionQueue
from agent_bridge.subscriber import RedisSubscriber, _run_terminal_event
from claude_code_bridge.bridge import ClaudeCodeBridge
from claude_code_bridge import run_teardown
from claude_code_bridge.run_teardown import RunTeardown
from claude_code_bridge.run_terminal import RunTerminalLatch
from codex_bridge.process_ledger import ProcessLedger, bind_ledger, register_process
from local_model_bridge.bridge import LocalModelBridge
from openclaw_bridge.bridge import SessionBridge
from openclaw_bridge.ingest import EventIngestPipeline
from openclaw_bridge.ws_client import is_run_terminal
from tests import virtual_clock
from tests.test_claude_code_send_boundaries import SESSION, _SDKStream, _SendHarness

REAPED = TeardownReport("ok", "none", True)

# -- Contract -----------------------------------------------------------------


def test_terminal_round_trips_through_stream_fields():
    terminal = RunTerminal("r1", "codex", "boot-1", TeardownReport("ok", "sigterm", True, 42))
    fields = terminal.to_fields()
    assert all(isinstance(v, str) for v in fields.values())
    assert fields["done"] == "1"
    assert RunTerminal.from_fields(fields) == terminal
    assert RunTerminal.from_dict(terminal.to_dict()) == terminal
    assert RunTerminal.from_fields(fields).teardown.elapsed_ms == 42
    assert terminal.proves_exit


def test_pre_contract_sentinel_is_no_evidence():
    assert RunTerminal.from_fields({"done": "1"}) is None


@pytest.mark.parametrize("fields", [
    {"done": "1", "request_id": "r1", "backend": "codex", "boot_id": "b", "teardown": "{"},
    {"done": "1", "request_id": "r1", "backend": "codex", "teardown": json.dumps(UNATTEMPTED.to_dict())},
    {"done": "1", "request_id": "r1", "backend": "codex", "boot_id": "b",
     "teardown": json.dumps({"disconnect": None, "signal": None})},
    {"done": "1", "request_id": "r1", "backend": "codex", "boot_id": "b",
     "teardown": json.dumps({"disconnect": None, "signal": None, "exited": "yes", "elapsed_ms": None})},
    {"done": "1", "request_id": "r1", "backend": "codex", "boot_id": "b",
     "teardown": json.dumps({"disconnect": None, "signal": None, "exited": True})},
    {"done": "1", "request_id": "r1", "backend": "codex", "boot_id": "b",
     "teardown": json.dumps({"disconnect": None, "signal": None, "exited": True, "elapsed_ms": -1})},
    {"done": "1", "request_id": "r1", "backend": "codex", "boot_id": "b",
     "teardown": json.dumps({"disconnect": None, "signal": None, "exited": True, "elapsed_ms": True})},
    {"done": "1", "request_id": "r1", "backend": "codex", "boot_id": "b",
     "teardown": json.dumps({"disconnect": None, "signal": None, "exited": True, "elapsed_ms": 1.5})},
    {"request_id": "r1"},
], ids=["bad-json", "no-boot", "short-teardown", "non-bool-exited", "no-elapsed",
        "negative-elapsed", "bool-elapsed", "float-elapsed", "not-done"])
def test_malformed_v1_sentinel_raises(fields):
    with pytest.raises(RunContractError):
        RunTerminal.from_fields(fields)


@pytest.mark.parametrize("exited", [None, False])
def test_only_a_reap_proves_exit(exited):
    assert not RunTerminal("r1", "codex", "b", TeardownReport(exited=exited)).proves_exit


def test_accepted_round_trips_and_names_this_boot():
    accepted = RunAccepted.here("r1", "openclaw")
    assert accepted.boot_id == process_boot_id()
    assert RunAccepted.from_dict(accepted.to_dict()) == accepted
    with pytest.raises(RunContractError):
        RunAccepted.from_dict({"request_id": "r1", "backend": "openclaw"})


def test_accepted_pid1_is_optional_and_typed():
    """TASK-1233 Vex S2: pid1 names the accepting process's PID-1 status."""
    accepted = RunAccepted.here("r1", "openclaw")
    assert accepted.pid1 is process_is_pid1()
    assert accepted.to_dict()["pid1"] is accepted.pid1
    legacy = {"request_id": "r1", "backend": "openclaw", "boot_id": "b"}
    assert RunAccepted.from_dict(legacy).pid1 is None
    assert RunAccepted.from_dict({**legacy, "pid1": True}).pid1 is True
    for bad in ("true", 1, 0):
        with pytest.raises(RunContractError, match="pid1"):
            RunAccepted.from_dict({**legacy, "pid1": bad})


@pytest.mark.parametrize("reports,expected", [
    ([], UNATTEMPTED),
    ([REAPED, REAPED], REAPED),
    ([REAPED, TeardownReport("ok", "none", None)], TeardownReport("ok", "none", None)),
    ([TeardownReport("timeout", "sigkill", False), TeardownReport("ok", "none", None)],
     TeardownReport("timeout", "sigkill", False)),
    ([TeardownReport("ok", "sigterm", True), TeardownReport("error", "none", True)],
     TeardownReport("error", "sigterm", True)),
])
def test_combine_keeps_the_worst_exit_and_strongest_evidence(reports, expected):
    assert TeardownReport.combine(reports) == expected


@pytest.mark.parametrize("times,total", [([120, None, 30], 150), ([None, None], None), ([0], 0)])
def test_combine_adds_up_the_measured_teardown_times(times, total):
    """Attempts run one after another; unmeasured ones don't count."""
    reports = [TeardownReport("ok", "none", True, ms) for ms in times]
    assert TeardownReport.combine(reports).elapsed_ms == total


def test_terminal_record_outlives_any_quiesce_wait():
    """Vex N4: a quiesce must still find a member's terminal at its deadline."""
    assert RUN_TERMINAL_RETENTION_S >= QUIESCE_TIMEOUT_MAX_S + QUIESCE_SWEEP_INTERVAL_MAX_S
    # The stream's own 300s TTL is not enough on its own.
    assert RUN_TERMINAL_RETENTION_S > 300


# -- Publisher ----------------------------------------------------------------


class _Pipe:
    def __init__(self, ops):
        self.ops = ops

    def __getattr__(self, name):
        return lambda *args, **kwargs: self.ops.append((name, args, kwargs))

    def execute(self):
        self.ops.append(("execute", (), {}))


def _publisher(provider="codex"):
    ops = []
    publisher = RedisPublisher.__new__(RedisPublisher)
    publisher._connected = True
    publisher._default_provider = provider
    publisher._redis = SimpleNamespace(
        pipeline=lambda transaction: _Pipe(ops),
        xadd=lambda *args, **kwargs: ops.append(("xadd", args, kwargs)),
        expire=lambda *args, **kwargs: ops.append(("expire", args, kwargs)),
    )
    return publisher, ops


@pytest.mark.parametrize("request_id,stream_ttl", [("r1", 300), ("req_delivery_x", 604800)])
def test_done_writes_the_record_then_the_sentinel(request_id, stream_ttl):
    publisher, ops = _publisher()
    publisher.publish_run_done(request_id, teardown=REAPED)
    fields = RunTerminal(request_id, "codex", process_boot_id(), REAPED).to_fields()
    record = run_terminal_key(request_id)
    assert [op[0] for op in ops] == ["delete", "hset", "expire", "execute", "xadd", "expire"]
    assert ops[1] == ("hset", (record,), {"mapping": fields})
    name, (key, record_ttl), _ = ops[2]
    assert key == record and record_ttl >= RUN_TERMINAL_RETENTION_S and record_ttl >= stream_ttl
    assert ops[4][1][:2] == (f"{RUN_STREAM_PREFIX}{request_id}", fields)
    assert ops[5][1] == (f"{RUN_STREAM_PREFIX}{request_id}", stream_ttl)


def test_done_logs_one_info_line_with_the_teardown_time(caplog):
    publisher, _ = _publisher()
    with caplog.at_level(logging.INFO, logger="agent_bridge.publisher"):
        publisher.publish_run_done("r1", teardown=TeardownReport("ok", "none", True, 87))
    lines = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    assert lines == ["Run done request_id=r1 exited=True disconnect=ok signal=none teardown_ms=87"]


def test_done_without_a_backend_is_the_bare_sentinel():
    publisher, ops = _publisher(provider=None)
    publisher.publish_run_done("r1", teardown=REAPED)
    assert [op[0] for op in ops] == ["xadd", "expire"]
    assert ops[0][1][1] == {"done": "1"}


def test_accepted_event_names_backend_and_boot():
    publisher, _ = _publisher()
    publisher.publish_run_event = Mock()
    publisher.publish_run_accepted("r1", "bot:user")
    request_id, event = publisher.publish_run_event.call_args.args
    assert request_id == "r1" and event.kind == AgentEventKind.RUN_ACCEPTED
    assert event.session_key == "bot:user"
    assert RunAccepted.from_dict(event.raw) == RunAccepted("r1", "codex", process_boot_id(),
                                                           process_is_pid1())


# -- Subscriber ---------------------------------------------------------------


def test_subscribe_run_yields_the_parsed_terminal_last():
    terminal = RunTerminal("r1", "openclaw", "boot-1", REAPED)

    async def run():
        sub = RedisSubscriber.__new__(RedisSubscriber)
        sub._redis = SimpleNamespace(xread=AsyncMock(return_value=[
            ("agent:run:r1", [("1-0", terminal.to_fields()), ("2-0", {"payload": "{}"})]),
        ]))
        return [event async for event in sub.subscribe_run("r1", timeout_s=5)]

    event, = asyncio.run(run())
    assert event.kind == AgentEventKind.RUN_TERMINAL
    assert event.raw == terminal.to_dict() and event.provider == "openclaw"


def test_legacy_or_malformed_terminal_still_ends_the_run_with_no_evidence(caplog):
    assert _run_terminal_event("r1", {"done": "1"}).raw == {}
    with caplog.at_level(logging.ERROR):
        event = _run_terminal_event("r1", {"done": "1", "request_id": "r1", "teardown": "{"})
    assert event.kind == AgentEventKind.RUN_TERMINAL and event.raw == {}
    assert "Malformed run terminal" in caplog.text


# -- Generic admission (codex, openclaw, local) ---------------------------------


def _admit(queue, run, *, request_id="r1", probe=None):
    publisher = Mock()
    redis = SimpleNamespace(xack=AsyncMock())
    kwargs = {"exit_probe": probe} if probe else {}
    task = asyncio.ensure_future(admit_send(
        queue, "bot:user", request_id, "msg", redis, consumer_group="g",
        publisher=publisher, run=run, **kwargs))
    return task, publisher


def _running(queue, body):
    async def run():
        async with queue.active("bot:user", request_id="r1"):
            await body()
    return run


@pytest.mark.parametrize("body,exited", [
    (AsyncMock(), True),
    (AsyncMock(side_effect=RuntimeError("boom")), True),
], ids=["ok", "error"])
def test_admitted_send_is_accepted_then_done_once_after_run(body, exited):
    async def run():
        queue = SessionQueue()
        task, publisher = _admit(queue, _running(queue, body))
        await asyncio.gather(task, return_exceptions=True)
        assert [c[0] for c in publisher.mock_calls] == ["publish_run_accepted", "publish_run_done"]
        publisher.publish_run_accepted.assert_called_once_with("r1", "bot:user")
        publisher.publish_run_done.assert_called_once_with(
            "r1", teardown=TeardownReport(exited=exited))
        assert queue.request_phase("bot:user", "r1") is None
    asyncio.run(run())


def test_cancelled_running_send_proves_nothing_without_a_probe():
    async def run():
        queue = SessionQueue()
        entered = asyncio.Event()

        async def body():
            entered.set()
            await asyncio.Event().wait()

        task, publisher = _admit(queue, _running(queue, body))
        await entered.wait()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        publisher.publish_run_done.assert_called_once_with(
            "r1", teardown=TeardownReport(exited=False))
    asyncio.run(run())


def test_probe_sees_the_outcome_and_its_failure_is_not_a_reap():
    async def run():
        queue = SessionQueue()
        outcomes = []

        def probe(outcome):
            outcomes.append(outcome)
            raise RuntimeError("probe broke")

        task, publisher = _admit(queue, _running(queue, AsyncMock()), probe=probe)
        await task
        assert outcomes == [RunOutcome(reached_running=True, cancelled=False)]
        publisher.publish_run_done.assert_called_once_with(
            "r1", teardown=TeardownReport(exited=False))
    asyncio.run(run())


def test_a_send_tombstoned_before_receipt_gets_a_refused_done_and_no_accepted():
    async def run():
        queue = SessionQueue()
        queue.tombstone("bot:user", "r1")
        body = AsyncMock()
        task, publisher = _admit(queue, body)
        await task
        body.assert_not_awaited()
        publisher.publish_run_accepted.assert_not_called()
        publisher.publish_run_done.assert_called_once_with("r1", teardown=REFUSED)
    asyncio.run(run())


def test_a_redelivered_send_after_a_stop_gets_an_unattempted_done():
    """admit_send: accept → abort → release → the same send again is refused
    with no evidence, since this process accepted it before (Snark)."""
    async def run():
        queue = SessionQueue()
        entered = asyncio.Event()

        async def body():
            entered.set()
            await asyncio.Event().wait()

        task, _ = _admit(queue, body)
        await entered.wait()
        assert queue.stop_request("bot:user", "r1") is True
        await asyncio.gather(task, return_exceptions=True)
        again = AsyncMock()
        task, publisher = _admit(queue, again)
        await task
        again.assert_not_awaited()
        publisher.publish_run_accepted.assert_not_called()
        publisher.publish_run_done.assert_called_once_with("r1", teardown=UNATTEMPTED)
    asyncio.run(run())


@pytest.mark.parametrize("reached_running,cancelled,exited", [
    (False, False, True), (False, True, True), (True, False, True), (True, True, False),
])
def test_default_report(reached_running, cancelled, exited):
    assert RunOutcome(reached_running, cancelled).default_report() == TeardownReport(exited=exited)


# -- Codex process ledger ----------------------------------------------------------


def _proc(returncode):
    return SimpleNamespace(returncode=returncode)


@pytest.mark.parametrize("codes,expected", [
    ([0], TeardownReport(None, "none", True)),
    ([0, -signal.SIGKILL], TeardownReport(None, "sigkill", True)),
    ([0, None], TeardownReport(None, "none", False)),
])
def test_codex_ledger_reports_every_subprocess_reaped(codes, expected):
    ledger = ProcessLedger()
    for code in codes:
        ledger.register(_proc(code))
    assert ledger.report(RunOutcome(True, True)) == expected


def test_codex_ledger_without_processes_falls_back_to_the_default():
    outcome = RunOutcome(reached_running=True, cancelled=True)
    assert ProcessLedger().report(outcome) == outcome.default_report()


def test_codex_ledger_is_scoped_to_the_send_task():
    async def run():
        ledger = ProcessLedger()
        proc = _proc(0)

        async def send():
            bind_ledger(ledger)
            # The SDK spawns from tasks it creates; they inherit the context.
            await asyncio.create_task(asyncio.sleep(0, register_process(proc)))

        await asyncio.create_task(send())
        register_process(_proc(None))  # outside any send: ignored
        assert ledger.procs == (proc,)
    asyncio.run(run())


# -- OpenClaw ---------------------------------------------------------------------


@pytest.mark.parametrize("raw,terminal", [
    ({"type": "event", "event": "agent", "payload": {"stream": "lifecycle", "data": {"phase": "end"}}}, True),
    ({"type": "event", "event": "agent", "payload": {"stream": "lifecycle", "data": {"phase": "error"}}}, True),
    ({"type": "event", "event": "agent", "payload": {"stream": "error", "data": {}}}, True),
    ({"type": "event", "event": "chat", "payload": {"state": "Final"}}, True),
    ({"type": "event", "event": "agent", "payload": {"stream": "lifecycle", "data": {"phase": "start"}}}, False),
    ({"type": "event", "event": "agent", "payload": {"stream": "assistant", "data": {"delta": "x"}}}, False),
    ({"type": "res", "event": "chat", "payload": {"state": "final"}}, False),
])
def test_openclaw_run_terminal_events(raw, terminal):
    assert is_run_terminal(raw) is terminal


def test_openclaw_exit_needs_a_settled_gateway_run_once_running():
    bridge = SessionBridge(MagicMock(), EventIngestPipeline(), MagicMock(), MagicMock())
    key = ("bot:user", "r1")
    assert bridge._exit_report(key, RunOutcome(True, False)) == TeardownReport(exited=False)
    bridge._gateway_settled.add(key)
    assert bridge._exit_report(key, RunOutcome(True, True)) == TeardownReport(exited=True)
    assert key not in bridge._gateway_settled
    assert bridge._exit_report(key, RunOutcome(False, True)) == TeardownReport(exited=True)


# -- Local model -----------------------------------------------------------------


def test_local_exit_is_the_inference_worker_finishing():
    async def run():
        bridge = LocalModelBridge.__new__(LocalModelBridge)
        bridge._run_workers = {}
        loop = asyncio.get_running_loop()
        finished, pending = loop.create_future(), loop.create_future()
        finished.set_result(None)
        bridge._run_workers.update(r1=finished, r2=pending)
        assert bridge._exit_report("r1", RunOutcome(True, True)) == TeardownReport(exited=True)
        assert bridge._exit_report("r2", RunOutcome(True, True)) == TeardownReport(exited=False)
        assert not bridge._run_workers
        assert bridge._exit_report("r3", RunOutcome(True, True)) == TeardownReport(exited=False)
        assert bridge._exit_report("r3", RunOutcome(False, False)) == TeardownReport(exited=True)
        pending.cancel()
    asyncio.run(run())


# -- Claude ---------------------------------------------------------------------


def _teardown(outcome):
    return SimpleNamespace(outcome=outcome)


def _outcome(exited, disconnect="ok", sig="none", elapsed_ms=None):
    return SimpleNamespace(disconnect=disconnect, signal=sig, exited=exited, elapsed_ms=elapsed_ms)


def test_claude_latch_reports_attempted_teardowns_conservatively():
    latch = RunTerminalLatch(Mock(), "r1")
    latch.record(_teardown(_outcome(True, sig="sigterm", elapsed_ms=40)))
    first = latch.report(phase=RequestPhase.RUNNING, cancelled=True)
    assert first == TeardownReport("ok", "sigterm", True) and first.elapsed_ms == 40
    latch.record(_teardown(_outcome(None, elapsed_ms=5)))  # CLI not identified
    both = latch.report(phase=RequestPhase.RUNNING, cancelled=False)
    assert both == TeardownReport("ok", "sigterm", False) and both.elapsed_ms == 45
    cut_short = RunTerminalLatch(Mock(), "r1")
    cut_short.record(_teardown(None))
    assert cut_short.report(phase=RequestPhase.RUNNING, cancelled=True) == TeardownReport(exited=False)


@pytest.mark.parametrize("phase,cancelled,expected", [
    (RequestPhase.PREFLIGHT, True, UNATTEMPTED),
    (RequestPhase.PREFLIGHT, False, TeardownReport(exited=True)),
    (RequestPhase.RUNNING, True, TeardownReport(exited=True)),
    (None, False, TeardownReport(exited=True)),
])
def test_claude_latch_without_an_sdk_client(phase, cancelled, expected):
    assert RunTerminalLatch(Mock(), "r1").report(phase=phase, cancelled=cancelled) == expected


def test_claude_teardown_measures_its_own_wall_time():
    class _Client:
        async def disconnect(self):
            await asyncio.sleep(0.05)

    async def run():
        return await RunTeardown(_Client()).run_owned()

    outcome = virtual_clock.run(run(), follow=[run_teardown])
    assert outcome.exited is None and outcome.elapsed_ms == 50


def test_claude_latch_publishes_once():
    publisher = Mock()
    latch = RunTerminalLatch(publisher, "r1")
    latch.publish(UNATTEMPTED)
    latch.publish(REAPED)
    publisher.publish_run_done.assert_called_once_with("r1", teardown=UNATTEMPTED)
    assert latch.published


def test_claude_done_follows_the_teardown(monkeypatch):
    async def run():
        harness = _SendHarness()
        sdk = _SDKStream(result=True, delay_disconnect=True)
        monkeypatch.setattr("claude_code_bridge.send_handler.ClaudeSDKClient", lambda **kw: sdk)
        send = harness.send()
        await sdk.waiting.wait()
        harness._publisher.publish_run_accepted.assert_called_once_with("turn-request", SESSION)
        sdk.release_stream.set()
        await sdk.disconnect_entered.wait()
        for _ in range(5):
            await asyncio.sleep(0)
        assert harness.terminal()  # content terminal is out
        harness._publisher.publish_run_done.assert_not_called()
        await asyncio.sleep(0.05)  # the teardown is measured while it is held
        sdk.release_disconnect.set()
        await send
        harness._publisher.publish_run_done.assert_called_once()
        teardown = harness._publisher.publish_run_done.call_args.kwargs["teardown"]
        assert (teardown.disconnect, teardown.signal, teardown.exited) == ("ok", "none", False)
        # timing-ok: a unit check (milliseconds, not s or us); 100x headroom over the 50ms hold
        assert 50 <= teardown.elapsed_ms < 5000
        assert harness._run_terminals == {}
    asyncio.run(asyncio.wait_for(run(), timeout=5))


@pytest.mark.parametrize("teardown_ends_in_grace", [False, True])
def test_host_shutdown_covers_a_stopped_send_still_tearing_down(monkeypatch, teardown_ends_in_grace):
    """Vex r2 R1: cancel_all skips a send already being cancelled (a user
    Stop), so it alone must still get the grace and then a done."""
    async def run():
        harness = _SendHarness()
        harness.SHUTDOWN_SEND_GRACE_SECONDS = 1.0 if teardown_ends_in_grace else 0.05
        sdk = _SDKStream(delay_disconnect=True)
        monkeypatch.setattr("claude_code_bridge.send_handler.ClaudeSDKClient", lambda **kw: sdk)
        send = harness.send()
        await sdk.waiting.wait()
        abort = asyncio.create_task(harness.abort())
        try:
            await sdk.disconnect_entered.wait()  # Stopped, teardown held
            assert harness._session_queue.cancel_all(cause=HOST_SHUTDOWN) == []
            if teardown_ends_in_grace:
                asyncio.get_running_loop().call_later(0.05, sdk.release_disconnect.set)
            await ClaudeCodeBridge._cancel_sends_for_shutdown(harness)
            harness._publisher.publish_run_done.assert_called_once()
            teardown = harness._publisher.publish_run_done.call_args.kwargs["teardown"]
            # Within the grace the send reports its own teardown; past it the
            # shutdown publishes "attempted, not proven".
            expected = (TeardownReport("ok", "none", False) if teardown_ends_in_grace
                        else TeardownReport(exited=False))
            assert teardown == expected
        finally:
            sdk.release_disconnect.set()
            await asyncio.gather(abort, send, return_exceptions=True)
        harness._publisher.publish_run_done.assert_called_once()
        assert harness._run_terminals == {}
    asyncio.run(asyncio.wait_for(run(), timeout=5))


def test_host_shutdown_publishes_a_done_its_teardown_still_holds(monkeypatch):
    """Vex S1: a finished send tearing down when shutdown starts still gets
    its done before the publisher closes, not after a 600s app timeout."""
    async def run():
        harness = _SendHarness()
        harness.SHUTDOWN_SEND_GRACE_SECONDS = 0.05
        sdk = _SDKStream(result=True, delay_disconnect=True)
        monkeypatch.setattr("claude_code_bridge.send_handler.ClaudeSDKClient", lambda **kw: sdk)
        send = harness.send()
        await sdk.waiting.wait()
        sdk.release_stream.set()
        await sdk.disconnect_entered.wait()  # stream over, teardown held
        harness._publisher.publish_run_done.assert_not_called()
        await ClaudeCodeBridge._cancel_sends_for_shutdown(harness)
        assert not send.done()
        harness._publisher.publish_run_done.assert_called_once_with(
            "turn-request", teardown=TeardownReport(exited=False))
        sdk.release_disconnect.set()
        await asyncio.gather(send, return_exceptions=True)
        harness._publisher.publish_run_done.assert_called_once()
        assert harness._run_terminals == {}
    asyncio.run(asyncio.wait_for(run(), timeout=5))


def test_claude_host_shutdown_publishes_before_the_teardown(monkeypatch):
    async def run():
        harness = _SendHarness()
        sdk = _SDKStream(delay_disconnect=True)
        monkeypatch.setattr("claude_code_bridge.send_handler.ClaudeSDKClient", lambda **kw: sdk)
        send = harness.send()
        await sdk.waiting.wait()
        harness._session_queue.cancel_all(cause=HOST_SHUTDOWN)
        await sdk.disconnect_entered.wait()
        # Teardown still pending; the done is already out, claiming nothing.
        assert not send.done()
        harness._publisher.publish_run_done.assert_called_once_with("turn-request", teardown=UNATTEMPTED)
        sdk.release_disconnect.set()
        await asyncio.gather(send, return_exceptions=True)
        harness._publisher.publish_run_done.assert_called_once()
    asyncio.run(asyncio.wait_for(run(), timeout=5))
