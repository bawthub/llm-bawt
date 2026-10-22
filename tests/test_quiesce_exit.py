"""TASK-1233 B1 (b)/(c): the pure exit-evidence rule and the sweep's I/O seams, without a database."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agent_bridge.run_contract import (QUIESCE_SWEEP_INTERVAL_MAX_S, RunContractError, RunTerminal, TeardownReport,
                                       run_terminal_key)
from agent_bridge.run_status import RunSighting, RunState, RunStatusReply
from agent_bridge.subscriber import RedisSubscriber
from llm_bawt.ops.quiesce_exit import SweepMember, judge
from llm_bawt.service import quiesce_exit_sweep
from llm_bawt.service.quiesce_exit_sweep import QuiesceExitSweep
from llm_bawt.service.recovery.orphan_turns import BridgeRunProbe

SIGHTED = RunSighting("req-1", "claude-code", "boot-a", pid1=True)


def _member(*, sighting=SIGHTED, backend="claude-code", request_id="req-1", accepted=None, session_key="snark:nick"):
    return SweepMember(7, 3, "turn-1", True, backend, session_key, request_id, sighting, accepted=accepted)


def _reply(state=RunState.NOT_FOUND, *, boot_id="boot-b", backend="claude-code", request_id="req-1", pid1=True):
    return RunStatusReply(backend, boot_id, request_id, state, "", pid1)


def _terminal(exited, *, boot_id="boot-a", backend="claude-code", request_id="req-1"):
    return RunTerminal(request_id, backend, boot_id, TeardownReport(exited=exited))


def _verdict(member, reply, terminal=None):
    v = judge(member, reply, terminal)
    record = None if v.record is None else (v.record["source"], v.record["exited"], v.record["boot_id"])
    return v.reason, record


# -- (b) restart ---------------------------------------------------------------

@pytest.mark.parametrize("backend", ["claude-code", "codex"])
@pytest.mark.parametrize("state", [RunState.NOT_FOUND, RunState.TOMBSTONED])
def test_a_pid1_restart_of_a_sighted_run_is_b_on_the_answering_boot(backend, state):
    member = _member(sighting=RunSighting("req-1", backend, "boot-a", True), backend=backend)
    v = judge(member, _reply(state, backend=backend), None)
    assert (v.reason, v.sighting) == ("restart", None)
    assert v.record == {**v.record, "source": "b", "exited": True, "boot_id": "boot-b", "backend": backend,
                        "request_id": "req-1"}


@pytest.mark.parametrize(("member", "reply"), [
    (_member(sighting=RunSighting("req-1", "openclaw", "boot-a", True), backend="openclaw"),
     _reply(backend="openclaw")),                                               # openclaw never proves
    (_member(sighting=None, accepted=RunSighting("req-1", "claude-code", "boot-a", None)),
     _reply()),                                                                 # accepted pid1 unknown
    (_member(sighting=RunSighting("req-1", "claude-code", "boot-a", False)), _reply()),  # sighted not PID 1
    (_member(sighting=RunSighting("req-1", "claude-code", "boot-a", None)), _reply()),   # sighted pid1 unknown
    (_member(), _reply(pid1=False)),                                            # answering not PID 1
    (_member(), _reply(pid1=None)),                                             # answering pid1 unknown
    (_member(), _reply(boot_id="boot-a")),                                      # same boot, not_found
])
def test_absence_without_restart_proof_or_terminal_is_never_evidence(member, reply):
    assert _verdict(member, reply) == ("absent_unproven", None)


# -- (c) same boot -------------------------------------------------------------

@pytest.mark.parametrize(("exited", "reason"), [(True, "terminal_exited"), (False, "terminal_survivor")])
def test_a_terminal_record_stamps_its_teardown_exited(exited, reason):
    for reply in (_reply(boot_id="boot-a"), _reply(RunState.TOMBSTONED, boot_id="boot-a"),
                  _reply(boot_id="boot-b", pid1=False)):
        assert _verdict(_member(sighting=None), reply, _terminal(exited)) == (reason, ("c", exited, "boot-a"))


def test_an_unattempted_terminal_is_no_evidence_for_not_found():
    for boot in ("boot-a", "boot-b"):
        assert _verdict(_member(sighting=None), _reply(boot_id=boot), _terminal(None)) == (
            "terminal_unattempted", None)


@pytest.mark.parametrize("terminal", [None, "unattempted"])
@pytest.mark.parametrize("backend", ["claude-code", "openclaw"])
def test_a_tombstone_alone_is_never_evidence(terminal, backend):
    # Vex r1 B1: a bridge unregisters in ``finally`` whatever its teardown did
    # (claude teardown crash -> exited null; openclaw tombstones as the abort
    # starts), so ``tombstoned`` from the very boot that held the run proves nothing.
    held = RunSighting("req-1", backend, "boot-a", False)
    tomb = _reply(RunState.TOMBSTONED, boot_id="boot-a", backend=backend, pid1=False)
    end = None if terminal is None else _terminal(None, backend=backend)
    reason = "absent_unproven" if end is None else "terminal_unattempted"
    assert _verdict(_member(sighting=held, backend=backend), tomb, end) == (reason, None)
    assert _verdict(_member(sighting=None, backend=backend), tomb, end) == (reason, None)


def test_a_known_survivor_is_not_rewritten_every_pass():
    member = _member(sighting=None)
    assert _verdict(member, _reply(boot_id="boot-a"), _terminal(False)) == (
        "terminal_survivor", ("c", False, "boot-a"))
    blocked = SweepMember(7, 3, "turn-1", True, "claude-code", "snark:nick", "req-1", None, "exit_unconfirmed")
    assert _verdict(blocked, _reply(boot_id="boot-a"), _terminal(False)) == ("terminal_survivor", None)
    assert _verdict(blocked, _reply(boot_id="boot-a"), _terminal(True)) == ("terminal_exited", ("c", True, "boot-a"))


def test_restart_proof_ignores_backend_case():
    member = _member(sighting=RunSighting("req-1", "Claude-Code", "boot-a", True))
    assert _verdict(member, _reply(backend="claude-code")) == ("restart", ("b", True, "boot-b"))


def test_restart_proof_wins_over_a_survivor_terminal():
    assert _verdict(_member(), _reply(), _terminal(False)) == ("restart", ("b", True, "boot-b"))


def test_a_terminal_of_another_request_or_backend_is_ignored():
    for terminal in (_terminal(True, request_id="req-2"), _terminal(True, backend="codex")):
        assert _verdict(_member(sighting=None), _reply(boot_id="boot-a"), terminal) == ("absent_unaccepted", None)


# -- (b) holders: the accept first, then the sighting --------------------------

ACCEPTED = RunSighting("req-1", "claude-code", "boot-a", pid1=True)


def test_an_accept_proves_a_restart_with_no_sighting():
    """Snark: the boot changes before any sighting, so only the accept record names the holder."""
    assert _verdict(_member(sighting=None, accepted=ACCEPTED), _reply()) == ("restart", ("b", True, "boot-b"))


@pytest.mark.parametrize(("accepted", "sighting", "reply"), [
    (ACCEPTED, RunSighting("req-1", "claude-code", "boot-c", True), _reply(boot_id="boot-c")),  # sighted boot answers
    (ACCEPTED, RunSighting("req-1", "claude-code", "boot-c", True), _reply(boot_id="boot-a")),  # accepting boot answers
    (ACCEPTED, RunSighting("req-1", "claude-code", "boot-c", None), _reply()),                  # sighting pid1 unknown
    (RunSighting("req-1", "claude-code", "boot-a", False), SIGHTED, _reply(boot_id="boot-c")),  # accept not PID 1
])
def test_every_holder_must_prove_the_restart(accepted, sighting, reply):
    assert _verdict(_member(sighting=sighting, accepted=accepted), reply) == ("absent_unproven", None)


def test_holders_are_the_accept_then_the_sighting():
    assert _member(sighting=SIGHTED, accepted=ACCEPTED).holders == (ACCEPTED, SIGHTED)
    assert _member(sighting=None).holders == ()


# -- abort for a bound member with no bridge record ----------------------------

def test_not_found_with_no_holder_and_no_terminal_asks_for_an_abort():
    v = judge(_member(sighting=None), _reply(boot_id="boot-a"), None)
    assert (v.reason, v.record, v.sighting, v.abort) == ("absent_unaccepted", None, None, True)


@pytest.mark.parametrize(("member", "reply", "terminal"), [
    (_member(sighting=None), _reply(RunState.TOMBSTONED, boot_id="boot-a"), None),  # already stopped there
    (_member(sighting=SIGHTED), _reply(boot_id="boot-a"), None),                    # a sighting holds it
    (_member(sighting=None, accepted=ACCEPTED), _reply(boot_id="boot-a"), None),    # an accept holds it
    (_member(sighting=None), _reply(boot_id="boot-a"), _terminal(None)),            # it already ended
    (_member(sighting=None, session_key=None), _reply(boot_id="boot-a"), None),     # nothing to address
    (_member(sighting=None), _reply(RunState.RUNNING, boot_id="boot-a"), None),     # live
])
def test_no_abort_where_the_bridge_has_or_had_a_record(member, reply, terminal):
    assert judge(member, reply, terminal).abort is False


# -- sightings and ambiguity ---------------------------------------------------

@pytest.mark.parametrize("state", [RunState.PREFLIGHT, RunState.QUEUED, RunState.RUNNING, RunState.STOPPING])
def test_a_live_answer_keeps_a_sighting_and_never_a_record(state):
    v = judge(_member(sighting=None), _reply(state, boot_id="boot-c", pid1=True), _terminal(True))
    assert (v.reason, v.record, v.sighting) == (f"live:{state.value}", None,
                                                RunSighting("req-1", "claude-code", "boot-c", True))


@pytest.mark.parametrize(("member", "reply", "reason"), [
    (_member(), None, "ambiguous"),
    (_member(), _reply(RunState.ERROR), "no_evidence:error"),
    (_member(), _reply(RunState.UNSUPPORTED), "no_evidence:unsupported"),
    (_member(), _reply(request_id="req-2"), "reply_mismatch"),
    (_member(), _reply(backend="codex"), "reply_mismatch"),
    (_member(request_id=None), _reply(), "unaddressed"),
    (_member(backend=None), _reply(), "unaddressed"),
])
def test_every_ambiguous_reply_writes_nothing(member, reply, reason):
    v = judge(member, reply, _terminal(True))
    assert (v.reason, v.record, v.sighting) == (reason, None, None)


# -- loop seams ----------------------------------------------------------------

def test_two_bounded_passes_fit_the_contract_interval():
    assert quiesce_exit_sweep.INTERVAL_SECONDS <= quiesce_exit_sweep.PASS_BUDGET_SECONDS
    assert 2 * quiesce_exit_sweep.PASS_BUDGET_SECONDS <= QUIESCE_SWEEP_INTERVAL_MAX_S
    assert quiesce_exit_sweep.PASS_CAPACITY == 32


def test_a_pass_where_every_probe_times_out_covers_exactly_its_capacity(monkeypatch):
    """Vex r2 R1: the real run_once at 1/100 scale, every probe taking the full timeout."""
    timeout = quiesce_exit_sweep.PROBE_TIMEOUT_SECONDS / 100
    budget = quiesce_exit_sweep.PASS_BUDGET_SECONDS / 100

    class TimesOut:
        async def status(self, **kwargs):
            await asyncio.sleep(timeout)
            return None, None

    monkeypatch.setattr(quiesce_exit_sweep, "undrained_members", lambda engine: _members(60))
    monkeypatch.setattr(quiesce_exit_sweep, "PROBE_TIMEOUT_SECONDS", timeout)
    sweep = QuiesceExitSweep(None, subscriber=lambda: None, probe=TimesOut())
    sweep.budget_seconds = budget
    capacity = quiesce_exit_sweep.pass_capacity(budget, timeout, quiesce_exit_sweep.CONCURRENCY)
    assert capacity == quiesce_exit_sweep.PASS_CAPACITY
    assert asyncio.run(sweep.run_once()) == {"ambiguous": capacity, "deferred": 60 - capacity}


def _members(n):
    return [SweepMember(i, 3, f"turn-{i}", True, "claude-code", "s", f"req-{i}", None) for i in range(1, n + 1)]


def test_a_pass_stops_at_its_budget_and_the_left_out_member_leads_the_next(monkeypatch):
    members = _members(3)
    asked = []

    class Slow:
        async def status(self, *, backend, session_key, request_id, label=None):
            asked.append(request_id)
            await asyncio.sleep(0.2)
            return None, None

    monkeypatch.setattr(quiesce_exit_sweep, "undrained_members", lambda engine: list(members))
    monkeypatch.setattr(quiesce_exit_sweep, "PROBE_TIMEOUT_SECONDS", 0.2)
    sweep = QuiesceExitSweep(None, subscriber=lambda: None, probe=Slow(), concurrency=1)
    sweep.budget_seconds = 0.5  # probes may start until 0.3s: two of three fit

    async def two_passes():
        return await sweep.run_once(), await sweep.run_once()

    first, second = asyncio.run(two_passes())
    assert first == {"ambiguous": 2, "deferred": 1}
    assert second == {"ambiguous": 2, "deferred": 1}
    # Pass 2 leads with the member pass 1 left out, then the least recently probed.
    assert asked == ["req-1", "req-2", "req-3", "req-1"]


def test_a_probe_still_running_at_the_budget_is_cut(monkeypatch):
    class Hung:
        async def status(self, **kwargs):
            await asyncio.sleep(10)

    monkeypatch.setattr(quiesce_exit_sweep, "undrained_members", lambda engine: _members(1))
    monkeypatch.setattr(quiesce_exit_sweep, "PROBE_TIMEOUT_SECONDS", 0.05)
    sweep = QuiesceExitSweep(None, subscriber=lambda: None, probe=Hung())
    sweep.budget_seconds = 0.1  # starts by 0.05s, still running at 0.1s
    assert asyncio.run(sweep.run_once()) == {"cut": 1}


def test_a_member_whose_turn_was_rebound_is_not_probed():
    asked = []

    class Probe:
        async def status(self, **kwargs):
            asked.append(kwargs)
            return None, None

    rebound = SweepMember(7, 3, "turn-1", True, "claude-code", "s", "req-1", None, "request_mismatch", "req-2")
    assert rebound.unstampable
    assert asyncio.run(QuiesceExitSweep(None, subscriber=lambda: None, probe=Probe())._one(rebound)) == "pin_mismatch"
    assert asked == []
    gone = SweepMember(7, 3, "turn-1", False, "claude-code", "s", "req-1", None, "turn_missing:no_execution_exit")
    assert not gone.unstampable


def _subscriber(record):
    sub = RedisSubscriber.__new__(RedisSubscriber)
    sub._pub_redis = SimpleNamespace(hgetall=AsyncMock(return_value=record))
    return sub


def test_read_run_terminal_parses_the_compact_record():
    terminal = _terminal(True)
    sub = _subscriber(terminal.to_fields())
    assert asyncio.run(sub.read_run_terminal("req-1")) == terminal
    sub._pub_redis.hgetall.assert_awaited_once_with(run_terminal_key("req-1"))
    assert asyncio.run(_subscriber({}).read_run_terminal("req-1")) is None
    assert asyncio.run(_subscriber({"done": "1"}).read_run_terminal("req-1")) is None
    with pytest.raises(RunContractError):
        asyncio.run(_subscriber({"done": "1", "request_id": "req-1", "teardown": "{"}).read_run_terminal("req-1"))


class _Rpc:
    def __init__(self, wire, record=None):
        self.wire, self.record, self.reads = wire, record, []

    async def send_rpc(self, method, params, request_id, *, timeout_s, backend):
        return self.wire

    async def read_run_terminal(self, request_id):
        self.reads.append(request_id)
        if isinstance(self.record, Exception):
            raise self.record
        return self.record


def _wire(state, boot="boot-b"):
    return {"backend": "claude-code", "boot_id": boot, "request_id": "req-1", "state": state, "detail": "",
            "pid1": True}


def test_probe_status_returns_the_parsed_reply_with_pid1():
    reply, result = asyncio.run(BridgeRunProbe(lambda: _Rpc(_wire("not_found"))).status(
        backend="claude-code", session_key="s", request_id="req-1"))
    assert reply == _reply() and result.state == "not_found" and result.backend_matched


def test_sweep_member_stamps_proof_and_keeps_sightings(monkeypatch):
    stamped, sighted = [], []
    monkeypatch.setattr(quiesce_exit_sweep, "stamp", lambda engine, m, r: stamped.append((m.member_id, r)) or True)
    monkeypatch.setattr(quiesce_exit_sweep, "record_sighting", lambda engine, s: sighted.append(s))

    def run(wire, record=None, member=None):
        rpc = _Rpc(wire, record)
        reason = asyncio.run(QuiesceExitSweep(None, subscriber=lambda: rpc)._one(member or _member()))
        return reason, rpc.reads

    assert run(_wire("running", boot="boot-c")) == ("live:running", ["req-1"])
    assert sighted == [RunSighting("req-1", "claude-code", "boot-c", True)] and stamped == []
    assert run(_wire("not_found")) == ("restart", ["req-1"])
    assert stamped[-1][0] == 7 and stamped[-1][1]["source"] == "b"
    # A malformed terminal record is no evidence, never a crash.
    assert run(_wire("not_found", boot="boot-a"), RunContractError("bad"), _member(sighting=None)) == (
        "absent_unaccepted", ["req-1"])
    assert len(stamped) == 1


def _abort_wire(state, request_id="req-1"):
    return {"backend": "claude-code", "boot_id": "boot-a", "request_id": request_id,
            "cancelled": state == "cancelled", "state": state, "detail": ""}


class _AbortRpc:
    """run.status says ``not_found``; chat.abort waits ``delay`` and answers ``state``."""

    def __init__(self, delay=0.0, state="tombstoned"):
        self.delay, self.state, self.calls = delay, state, []

    async def send_rpc(self, method, params, request_id, *, timeout_s, backend):
        self.calls.append((method, params["requestId"], timeout_s, backend))
        if method == "chat.abort":
            await asyncio.sleep(self.delay)
            return _abort_wire(self.state, params["requestId"])
        return {**_wire("not_found", boot="boot-a"), "request_id": params["requestId"]}

    async def read_run_terminal(self, request_id):
        return None


def test_the_sweep_aborts_once_per_member_and_boot_with_the_abort_timeout(monkeypatch):
    from agent_bridge.abort_contract import ABORT_RPC_TIMEOUT_S

    rpc = _AbortRpc()
    monkeypatch.setattr(quiesce_exit_sweep, "undrained_members", lambda engine: _members(1))
    sweep = QuiesceExitSweep(None, subscriber=lambda: rpc)

    async def passes():
        first = await sweep.run_once()
        await sweep.settle_aborts()
        second = await sweep.run_once()
        await sweep.settle_aborts()
        return first, second

    assert asyncio.run(passes()) == ({"absent_unaccepted": 1}, {"absent_unaccepted": 1})
    aborts = [c for c in rpc.calls if c[0] == "chat.abort"]
    assert aborts == [("chat.abort", "req-1", ABORT_RPC_TIMEOUT_S, "claude-code")]


def test_an_abort_holds_no_probe_slot_and_no_budget(monkeypatch):
    """The abort waits on the bridge's teardown; the pass must not."""
    rpc = _AbortRpc(delay=10)
    monkeypatch.setattr(quiesce_exit_sweep, "undrained_members", lambda engine: _members(3))
    monkeypatch.setattr(quiesce_exit_sweep, "PROBE_TIMEOUT_SECONDS", 0.2)
    sweep = QuiesceExitSweep(None, subscriber=lambda: rpc, concurrency=1)
    sweep.budget_seconds = 1.0  # probes start by 0.8s; each abort alone would take 10s

    async def one_pass():
        started = asyncio.get_running_loop().time()
        counts = await sweep.run_once()
        elapsed = asyncio.get_running_loop().time() - started
        await sweep.aclose()
        return counts, elapsed

    counts, elapsed = asyncio.run(one_pass())
    assert counts == {"absent_unaccepted": 3} and elapsed < 0.5
    assert sorted(c[1] for c in rpc.calls if c[0] == "chat.abort") == ["req-1", "req-2", "req-3"]


@pytest.mark.parametrize("wire", [TimeoutError("late"), {"backend": "codex"}, _abort_wire("bogus")])
def test_an_unconfirmed_abort_is_none(wire):
    class Rpc:
        async def send_rpc(self, *args, **kwargs):
            if isinstance(wire, Exception):
                raise wire
            return wire

    assert asyncio.run(BridgeRunProbe(lambda: Rpc()).abort(
        backend="claude-code", session_key="s", request_id="req-1")) is None
    assert asyncio.run(BridgeRunProbe(lambda: None).abort(
        backend="claude-code", session_key="s", request_id="req-1")) is None


def test_a_confirmed_abort_is_its_reply():
    from agent_bridge.abort_contract import AbortState

    reply = asyncio.run(BridgeRunProbe(lambda: _AbortRpc(state="cancelled")).abort(
        backend="claude-code", session_key="s", request_id="req-1"))
    assert reply.state is AbortState.CANCELLED and reply.boot_id == "boot-a"
