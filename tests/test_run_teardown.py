"""TASK-1232 P4: bounded, request-owned teardown of one Claude CLI run."""

from __future__ import annotations

import asyncio
import signal

import pytest

from agent_bridge.abort_contract import (
    ABORT_TEARDOWN_BOUND_S,
    TEARDOWN_DISCONNECT_BACKSTOP_S,
    TEARDOWN_KILL_WAIT_S,
)
from claude_code_bridge import run_teardown
from claude_code_bridge.run_teardown import (
    VERIFIED_SDK_VERSIONS,
    ProcTable,
    RunTeardown,
    TeardownOutcome,
    cli_process,
    loaded_sdk_version,
)
from tests import virtual_clock

CLI, CHILD, GRANDCHILD = 100, 200, 300


class FakeOS:
    """A process table that ``kill`` mutates: pid -> [start_time, state]."""

    def __init__(self, tree: dict[int, list[int]], *, ignore_kill: tuple[int, ...] = ()):
        self.procs = {pid: [1000 + pid, "S"] for pid in {CLI, *tree, *(c for cs in tree.values() for c in cs)}}
        self.tree = tree
        self.ignore_kill = set(ignore_kill)
        self.kills: list[int] = []
        self.on_cli_kill = None

    def kill(self, pid: int, sig: int) -> None:
        assert sig == signal.SIGKILL
        self.kills.append(pid)
        if pid not in self.procs:
            raise ProcessLookupError(pid)
        if pid in self.ignore_kill:
            return
        self.exit(pid)
        if pid == CLI and self.on_cli_kill is not None:
            self.on_cli_kill()

    def exit(self, pid: int) -> None:
        # Exited children are reparented: the tree no longer reaches them.
        self.procs.pop(pid, None)
        self.tree.pop(pid, None)
        for children in self.tree.values():
            if pid in children:
                children.remove(pid)


class FakeProcTable(ProcTable):
    def __init__(self, os_: FakeOS):
        self.os = os_

    def start_time(self, pid):
        entry = self.os.procs.get(pid)
        return entry[0] if entry else None

    def is_alive(self, pid, start_time):
        entry = self.os.procs.get(pid)
        return bool(entry) and entry[0] == start_time and entry[1] not in ("Z", "X")

    def children(self, pid):
        return list(self.os.tree.get(pid, []))


class FakeProcess:
    def __init__(self, pid=CLI):
        self.pid = pid
        self.returncode: int | None = None
        self._exited = asyncio.Event()

    def finish(self, code: int) -> None:
        self.returncode = code
        self._exited.set()

    async def wait(self) -> int:
        await self._exited.wait()
        return self.returncode


class FakeClient:
    """``disconnect()`` mimics the SDK: it may end the CLI, hang, or raise."""

    def __init__(self, process, *, on_disconnect=None, hang=False, error=None):
        self._transport = type("T", (), {})()
        self._transport._process = process
        self.on_disconnect = on_disconnect
        self.hang = hang
        self.error = error
        self.disconnect_calls = 0

    async def disconnect(self):
        self.disconnect_calls += 1
        # The real transport drops its process reference on close.
        self._transport._process = None
        if self.hang:
            await asyncio.Event().wait()
        if self.on_disconnect is not None:
            self.on_disconnect()
        if self.error is not None:
            raise self.error


def _teardown(client, os_, **kw):
    kw.setdefault("disconnect_backstop_s", 0.05)
    kw.setdefault("kill_wait_s", 0.2)
    return RunTeardown(client, procs=FakeProcTable(os_), kill=os_.kill, **kw)


def run(coro):
    return asyncio.run(asyncio.wait_for(coro, timeout=5))


async def bounded(teardown, limit=2.0):
    """Run the teardown and fail (not hang) if it overruns: the owner path
    deliberately can't be cancelled, so a missing bound would hang the test."""
    task = teardown.start()
    done, _ = await asyncio.wait({task}, timeout=limit)
    if not done:
        task.cancel()
        pytest.fail(f"teardown overran its {limit}s bound")
    return task.result()


def test_bound_constants_fit_the_abort_reply_bound():
    assert TEARDOWN_DISCONNECT_BACKSTOP_S + TEARDOWN_KILL_WAIT_S < ABORT_TEARDOWN_BOUND_S


def test_sdk_layout_the_teardown_relies_on_exists_in_the_installed_sdk():
    import inspect

    import anyio.abc
    from claude_agent_sdk import ClaudeSDKClient
    from claude_agent_sdk._internal.transport.subprocess_cli import SubprocessCLITransport

    assert loaded_sdk_version() in VERIFIED_SDK_VERSIONS
    client = ClaudeSDKClient()
    assert cli_process(client) is None  # not connected: nothing to tear down
    assert "self._transport" in inspect.getsource(ClaudeSDKClient.__init__)
    assert "self._process" in inspect.getsource(SubprocessCLITransport.__init__)
    assert "anyio.open_process" in inspect.getsource(SubprocessCLITransport)
    for name in ("pid", "returncode", "wait"):
        assert hasattr(anyio.abc.Process, name)


def test_graceful_exit_sends_no_signal_and_leaves_descendants_alone():
    async def go():
        os_ = FakeOS({CLI: [CHILD]})
        proc = FakeProcess()
        client = FakeClient(proc, on_disconnect=lambda: (os_.exit(CLI), proc.finish(0)))
        outcome = await _teardown(client, os_).run_owned()
        assert outcome == TeardownOutcome("ok", "none", False, CLI, (CHILD,))
        assert os_.kills == []
        # A clean exit had its chance to stop its children: not ours to kill.
        assert CHILD in os_.procs
    run(go())


def test_graceful_exit_without_children_is_confirmed_gone():
    async def go():
        os_ = FakeOS({})
        proc = FakeProcess()
        client = FakeClient(proc, on_disconnect=lambda: (os_.exit(CLI), proc.finish(0)))
        outcome = await _teardown(client, os_).run_owned()
        assert outcome == TeardownOutcome("ok", "none", True, CLI, ())
        assert outcome.detail() == "teardown:disconnect=ok;signal=none;exited=true"
    run(go())


def test_hung_disconnect_hits_the_backstop_then_kills_the_cli_and_its_whole_tree():
    async def go():
        os_ = FakeOS({CLI: [CHILD], CHILD: [GRANDCHILD]})
        proc = FakeProcess()
        os_.on_cli_kill = lambda: proc.finish(-9)
        client = FakeClient(proc, hang=True)
        outcome = await bounded(_teardown(client, os_))
        assert outcome == TeardownOutcome("timeout", "sigkill", True, CLI, ())
        # The CLI first, then every descendant snapshotted before any signal.
        assert os_.kills[0] == CLI and set(os_.kills[1:]) == {CHILD, GRANDCHILD}
        assert outcome.detail() == "teardown:disconnect=timeout;signal=sigkill;exited=true"
    run(go())


def test_signal_killed_cli_has_its_descendants_killed():
    async def go():
        # The SDK's own escalation SIGKILLed the CLI: its children are orphans.
        os_ = FakeOS({CLI: [CHILD], CHILD: [GRANDCHILD]})
        proc = FakeProcess()
        client = FakeClient(proc, on_disconnect=lambda: (os_.exit(CLI), proc.finish(-9)))
        outcome = await _teardown(client, os_).run_owned()
        assert outcome == TeardownOutcome("ok", "sigkill", True, CLI, ())
        assert sorted(os_.kills) == [CHILD, GRANDCHILD]
    run(go())


def test_a_descendant_that_survives_sigkill_is_reported_not_hidden():
    async def go():
        os_ = FakeOS({CLI: [CHILD]}, ignore_kill=(CHILD,))
        proc = FakeProcess()
        client = FakeClient(proc, on_disconnect=lambda: (os_.exit(CLI), proc.finish(-15)))
        outcome = await _teardown(client, os_).run_owned()
        assert (outcome.exited, outcome.survivors) == (False, (CHILD,))
        assert outcome.detail().endswith(f";exited=false;survivors={CHILD}")
    run(go())


def test_reused_pids_are_never_signalled():
    async def go():
        os_ = FakeOS({CLI: [CHILD]})
        proc = FakeProcess()

        def reuse():
            # The SDK killed the CLI; both pids were reaped and handed to
            # unrelated processes (new start times).
            os_.exit(CLI)
            os_.exit(CHILD)
            os_.procs[CLI] = [9_000, "S"]
            os_.procs[CHILD] = [9_001, "S"]

        client = FakeClient(proc, on_disconnect=reuse, hang=False)
        # returncode stays None: reaped by the watcher, not yet recorded.
        outcome = await _teardown(client, os_).run_owned()
        assert os_.kills == []
        assert outcome.signal == "none"
        assert CLI in os_.procs and CHILD in os_.procs
    run(go())


def test_disconnect_error_still_tears_down():
    async def go():
        os_ = FakeOS({CLI: [CHILD]})
        proc = FakeProcess()
        os_.on_cli_kill = lambda: proc.finish(-9)
        client = FakeClient(proc, error=RuntimeError("transport broke"))
        outcome = await _teardown(client, os_).run_owned()
        assert outcome == TeardownOutcome("error", "sigkill", True, CLI, ())
    run(go())


def test_unidentifiable_cli_reports_unknown_exit():
    async def go():
        os_ = FakeOS({})
        client = FakeClient(None)
        outcome = await _teardown(client, os_).run_owned()
        assert outcome == TeardownOutcome("ok", "none", None)
        assert outcome.detail() == "teardown:disconnect=ok;signal=none;exited=unknown"
        assert client.disconnect_calls == 1
    run(go())


def test_owner_cancellation_waits_for_the_teardown_then_propagates():
    async def go():
        os_ = FakeOS({})
        proc = FakeProcess()
        release = asyncio.Event()

        async def slow_disconnect():
            await release.wait()
            os_.exit(CLI)
            proc.finish(0)

        client = FakeClient(proc)
        client.disconnect = slow_disconnect
        teardown = _teardown(client, os_, disconnect_backstop_s=5)
        owner = asyncio.create_task(teardown.run_owned())
        await asyncio.sleep(0.01)
        owner.cancel()  # repeated Stop / host shutdown
        await asyncio.sleep(0.01)
        owner.cancel()
        await asyncio.sleep(0.01)
        assert not owner.done()  # still holding the session lock
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await owner
        assert (await teardown.wait(1)) == TeardownOutcome("ok", "none", True, CLI, ())
    run(go())


def test_wait_returns_none_while_pending_and_the_outcome_once_done():
    async def go():
        os_ = FakeOS({})
        proc = FakeProcess()
        release = asyncio.Event()
        client = FakeClient(proc)

        async def slow_disconnect():
            await release.wait()
            os_.exit(CLI)
            proc.finish(0)

        client.disconnect = slow_disconnect
        teardown = _teardown(client, os_, disconnect_backstop_s=5)
        assert await teardown.wait(0.01) is None  # not started yet
        owner = asyncio.create_task(teardown.run_owned())
        assert await teardown.wait(0.01) is None
        release.set()
        outcome = await teardown.wait(1)
        assert outcome is not None and outcome.exited is True
        assert await owner == outcome
        # Idempotent: a second owner call doesn't disconnect again.
        assert await teardown.run_owned() == outcome
    run(go())


def test_teardown_stays_within_its_bound_when_everything_hangs():
    async def go():
        os_ = FakeOS({CLI: [CHILD]}, ignore_kill=(CLI, CHILD))
        proc = FakeProcess()  # never exits, ignores SIGKILL
        client = FakeClient(proc, hang=True)
        loop = asyncio.get_running_loop()
        start = loop.time()
        outcome = await bounded(_teardown(client, os_, disconnect_backstop_s=0.05, kill_wait_s=0.1), 0.5)
        assert loop.time() - start < 0.5  # timing-ok: virtual clock (tests/virtual_clock.py), exact and runner-independent
        assert outcome == TeardownOutcome("timeout", "sigkill", False, CLI, (CLI, CHILD))
    virtual_clock.run(go(), follow=[run_teardown])


def test_real_proc_table_reads_this_process(tmp_path):
    import os

    procs = ProcTable()
    me = os.getpid()
    start = procs.start_time(me)
    assert start is not None and procs.is_alive(me, start)
    assert not procs.is_alive(me, start + 1)
    assert procs.start_time(2**22 + 12345) is None
    # A comm with spaces and parens still parses (fields resume after the last ')').
    fake = tmp_path / "77"
    (fake / "task" / "77").mkdir(parents=True)
    (fake / "stat").write_text("77 (a (b) c) S " + " ".join(["0"] * 18) + " 4242 0 0\n")
    (fake / "task" / "77" / "children").write_text("")
    table = ProcTable(tmp_path)
    assert table.start_time(77) == 4242 and table.is_alive(77, 4242)
    (fake / "stat").write_text("77 (zombie) Z " + " ".join(["0"] * 18) + " 4242 0 0\n")
    assert not table.is_alive(77, 4242)


def test_real_proc_table_finds_a_live_grandchild():
    import os
    import subprocess
    import sys

    child = subprocess.Popen(
        [sys.executable, "-c", "import subprocess,sys,time; p=subprocess.Popen(['sleep','30']); print(p.pid, flush=True); time.sleep(30)"],
        stdout=subprocess.PIPE, text=True,
    )
    try:
        grandchild = int(child.stdout.readline())
        tree = ProcTable().descendants(os.getpid())
        assert child.pid in tree and grandchild in tree
    finally:
        os.kill(grandchild, signal.SIGKILL)
        child.kill()
        child.wait()


# -- steer escalation shares the run's teardown (Vex S1) ----------------------


def test_a_failed_steer_escalation_tears_down_through_the_runs_one_teardown():
    # A direct disconnect() here nulled the transport before anything looked
    # at it, so the send's teardown reported exited=unknown, never killed the
    # CLI's descendants, and the disconnect itself was unbounded.
    from claude_code_bridge.active_run import ClaudeActiveRun
    from tests.test_claude_active_run_next_steer import CMD, FRAME, _FakeClient, _finish, _start

    async def go():
        os_ = FakeOS({CLI: [CHILD], CHILD: [GRANDCHILD]})
        process = FakeProcess()
        os_.on_cli_kill = lambda: process.finish(-signal.SIGKILL)
        client = _FakeClient()
        client._query.receipt = {"still_queued": [], "cancelled": []}
        sdk = FakeClient(process, hang=True)  # the SDK's close() hangs
        client._transport, client.disconnect = sdk._transport, sdk.disconnect
        teardown = _teardown(client, os_)
        await client._query.frames.put({
            "type": "system", "subtype": "init", "capabilities": ["interrupt_cancel_queued_v1"],
        })
        active = ClaudeActiveRun(client=client, request_id="req-1", teardown=teardown)
        _, consumer = await _start(active)
        # TASK-1300: steers and escalations carry the app's registry framing.
        await active.steer("user note", priority="next", message_id=CMD, origin="user",
                           framed_message="NEXT[user note]", interrupt_item="ITEM[user note]")
        with pytest.raises(RuntimeError, match="escalation_not_cancelled"):
            await asyncio.wait_for(active.escalate(CMD, interrupt_frame=FRAME), 2)
        assert teardown.started
        # The send's finally reuses the same teardown and its outcome.
        outcome = await bounded(teardown)
        assert outcome == await teardown.run_owned()
        assert (outcome.pid, outcome.signal, outcome.exited) == (CLI, "sigkill", True)
        assert sorted(os_.kills) == [CLI, CHILD, GRANDCHILD]
        assert sdk.disconnect_calls == 1
        await _finish(client, consumer)

    run(go())


def test_the_send_hands_its_teardown_to_the_steer_controller(monkeypatch):
    from tests.test_claude_code_send_boundaries import SESSION, _SDKStream, _SendHarness

    async def go():
        harness = _SendHarness()
        sdk = _SDKStream()
        monkeypatch.setattr("claude_code_bridge.send_handler.ClaudeSDKClient", lambda **kw: sdk)
        send = harness.send()
        await sdk.waiting.wait()
        active = harness._session_queue.get_active_client(SESSION)
        assert active.teardown is harness._run_teardowns["turn-request"]
        await harness.abort()
        await asyncio.gather(send, return_exceptions=True)

    run(go())
