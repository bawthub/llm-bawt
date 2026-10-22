"""TASK-1232: the startup check's ``run.status`` probe, without a database.

Only ``not_found`` from the bridge the turn was published to may settle a
turn. Every other outcome (no recorded backend, no transport, a timeout, a
pre-contract bridge, a reply from another backend) is ambiguous.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from llm_bawt.service.recovery.base import SweepScope
from llm_bawt.service.recovery.orphan_turns import BridgeRunProbe, OrphanTurnsStep, ProbeResult

TURN = {"turn_id": "turn-1", "bot_id": "snark", "agent_backend": "claude-code",
        "agent_session_key": "snark:nick", "agent_request_id": "req-1"}


class _Subscriber:
    def __init__(self, reply=None, *, raises: Exception | None = None):
        self.reply, self.raises, self.calls = reply, raises, []

    async def send_rpc(self, method, params, request_id, *, timeout_s, backend):
        self.calls.append({"method": method, "params": params, "timeout_s": timeout_s, "backend": backend})
        if self.raises is not None:
            raise self.raises
        return self.reply


def _v1(state: str, *, backend: str = "claude-code") -> dict:
    return {"backend": backend, "boot_id": "boot-9", "request_id": "req-1", "state": state, "detail": ""}


def _probe(subscriber) -> ProbeResult:
    return asyncio.run(BridgeRunProbe(lambda: subscriber, timeout_seconds=7.0).probe(dict(TURN)))


def test_not_found_from_the_recorded_backend_is_the_only_settling_answer():
    subscriber = _Subscriber(_v1("not_found"))
    result = _probe(subscriber)
    assert (result.state, result.backend_matched, result.backend, result.boot_id) == (
        "not_found", True, "claude-code", "boot-9")
    assert subscriber.calls == [{"method": "run.status", "timeout_s": 7.0, "backend": "claude-code",
                                 "params": {"sessionKey": "snark:nick", "requestId": "req-1"}}]


@pytest.mark.parametrize("state", ["preflight", "queued", "running", "stopping", "tombstoned", "error"])
def test_every_other_bridge_state_is_reported_as_is(state):
    assert _probe(_Subscriber(_v1(state))).state == state


@pytest.mark.parametrize(("subscriber", "state"), [
    (_Subscriber(raises=TimeoutError("RPC run.status timed out")), "timeout"),
    (_Subscriber(raises=ConnectionError("redis down")), "rpc_error"),
    (_Subscriber({"ok": False, "error": "unknown method: run.status"}), "unsupported"),
    (_Subscriber(_v1("not_found", backend="codex")), "malformed_reply"),
    (_Subscriber({"backend": "claude-code", "state": "not_found"}), "malformed_reply"),
    (None, "bridge_unreachable"),
], ids=["timeout", "transport_error", "legacy_bridge", "other_backend", "partial_reply", "no_subscriber"])
def test_no_answer_or_a_bad_answer_never_reads_as_absent(subscriber, state):
    result = _probe(subscriber)
    assert result.state == state and result.state != "not_found"


def test_a_turn_without_a_recorded_backend_is_never_asked():
    subscriber = _Subscriber(_v1("not_found"))
    result = asyncio.run(BridgeRunProbe(lambda: subscriber).probe({**TURN, "agent_backend": None}))
    assert result.state == "unaddressed" and subscriber.calls == []


def test_probes_run_concurrently_so_one_slow_bridge_does_not_serialize_the_check():
    """Every probe may wait out a restarting bridge; serial probes would make
    the check take candidates × timeout."""
    in_flight, peak, release = 0, 0, asyncio.Event()

    class Probe:
        async def probe(self, turn):
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            if peak == OrphanTurnsStep.MAX_CONCURRENT_PROBES:
                release.set()
            await release.wait()
            in_flight -= 1
            return ProbeResult("running", backend_matched=True)

    step = OrphanTurnsStep(Probe(), started_at=datetime.now(timezone.utc))
    scope = SweepScope(engine=None, now=datetime.now(timezone.utc), dry_run=True)
    candidates = [{**TURN, "turn_id": f"turn-{i}"} for i in range(OrphanTurnsStep.MAX_CONCURRENT_PROBES + 3)]
    plans = asyncio.run(asyncio.wait_for(step.plan_all(scope, candidates), timeout=2.0))
    assert peak == OrphanTurnsStep.MAX_CONCURRENT_PROBES
    assert [plan.why for plan in plans] == ["run.status running"] * len(candidates)
