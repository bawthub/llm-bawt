"""TASK-1865: batch-gate driver wiring (claim -> child -> terminal -> retire)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_bawt.ops import gate_driver as gd
from llm_bawt.ops.executor import validate_spec
from llm_bawt.ops.models import JOB_DISPATCHING, JOB_FAILED, JOB_LOST, JOB_QUEUED, JOB_SUCCEEDED

CONFIG = gd.GateDriverConfig(Path("/repo"), Path("/scratch"), Path("/receipts"))


class Store:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        return lambda *a, **k: self.calls.append((name, a, k))


class Child:
    def __init__(self, alive=True):
        self.alive = alive

    def poll(self):
        return None if self.alive else 0


def _job(state=JOB_QUEUED, dispatched_at=None):
    return SimpleNamespace(id="j1", state=state, dispatched_at=dispatched_at)


def test_config_requires_all_mounts(monkeypatch):
    for name in (gd.REPOSITORY_ENV, gd.SCRATCH_ENV, gd.RECEIPT_ROOT_ENV):
        monkeypatch.delenv(name, raising=False)
    assert gd.GateDriverConfig.from_env() is None
    monkeypatch.setenv(gd.REPOSITORY_ENV, "/r")
    monkeypatch.setenv(gd.SCRATCH_ENV, "/s")
    assert gd.GateDriverConfig.from_env() is None
    monkeypatch.setenv(gd.RECEIPT_ROOT_ENV, "/x")
    assert gd.GateDriverConfig.from_env() == gd.GateDriverConfig(Path("/r"), Path("/s"), Path("/x"))


def test_spec_is_exact():
    assert validate_spec('{"action": "batch_gate"}') == {"action": "batch_gate"}
    with pytest.raises(ValueError):
        validate_spec('{"action": "batch_gate", "image": "x"}')


def test_snapshot_detection():
    assert gd.is_gate_snapshot('{"gate_claim_schema": "v1"}')
    assert not gd.is_gate_snapshot('{"action": "x"}')
    assert not gd.is_gate_snapshot("not json")
    assert not gd.is_gate_snapshot(None)


def test_admission_requires_turn_capability():
    with pytest.raises(ValueError):
        gd.admit_gate(Store(), operation_slug="s", args={}, turn_token=None, idempotency_key="k")


def test_dispatch_claims_then_spawns_once(monkeypatch):
    import llm_bawt.ops.gate_trusted_claim as claim
    monkeypatch.setattr(claim, "claim_trusted_gate_job", lambda store, job_id: True)
    spawned = []
    driver = gd.GateDriver(Store(), CONFIG, spawn=lambda j: spawned.append(j) or Child())
    driver.dispatch(_job())
    driver.dispatch(_job())  # live child: no second attempt
    assert spawned == ["j1"]


def test_refused_claim_stays_queued_with_reason(monkeypatch):
    import llm_bawt.ops.gate_trusted_claim as claim
    monkeypatch.setattr(claim, "claim_trusted_gate_job", lambda store, job_id: False)
    store = Store()
    gd.GateDriver(store, CONFIG, spawn=lambda j: pytest.fail("spawned")).dispatch(_job())
    assert [c[0] for c in store.calls] == ["note_queued_error", "fail_stale_queued"]


def test_orphan_after_grace_is_lost_and_retired(monkeypatch):
    store = Store()
    driver = gd.GateDriver(store, CONFIG)
    retired = []
    monkeypatch.setattr(driver, "retire", retired.append)
    fresh = datetime.now(timezone.utc).replace(tzinfo=None)
    driver.reconcile(_job(JOB_DISPATCHING, fresh))
    assert store.calls == [] and retired == []
    driver.reconcile(_job(JOB_DISPATCHING, fresh - timedelta(seconds=120)))
    assert store.calls[0][2]["state"] == JOB_LOST and retired == ["j1"]


def test_live_child_is_not_orphaned():
    store = Store()
    driver = gd.GateDriver(store, CONFIG, spawn=lambda j: Child())
    driver._children["j1"] = Child()
    driver.reconcile(_job("running"))
    assert store.calls == []


def _patch_chain(monkeypatch, *, run=None, launch_error=None):
    import contextlib
    import llm_bawt.ops.gate_custody as custody
    import llm_bawt.ops.gate_retire as retire
    import llm_bawt.ops.gate_runner as runner
    import llm_bawt.ops.gate_source as source
    import llm_bawt.ops.gate_start as start
    import llm_bawt.ops.gate_trust as trust

    events = []
    req = SimpleNamespace(plan={}, environment_json="{}", environment_sha256="0")
    monkeypatch.setattr(trust, "read_trusted_gate_request", lambda job: req)

    @contextlib.contextmanager
    def launch(store, job_id, receipt_root):
        events.append("launch")
        yield SimpleNamespace(owner_nonce="n")

    @contextlib.contextmanager
    def begin(store, job_id, receipt_root):
        events.append("start")
        yield None

    @contextlib.contextmanager
    def staged(plan, repository, staging_parent):
        yield "src"

    def run_gate(api, **kw):
        events.append("run")
        if launch_error:
            raise launch_error
        return run

    monkeypatch.setattr(custody, "reserve_gate_launch", launch)
    monkeypatch.setattr(start, "reserve_gate_start", begin)
    monkeypatch.setattr(source, "staged_gate_source", staged)
    monkeypatch.setattr(runner, "run_gate", run_gate)
    monkeypatch.setattr(retire, "retire_gate_slot", lambda *a, **k: events.append("retire"))
    return events


@pytest.mark.parametrize("code,state", [(0, JOB_SUCCEEDED), (3, JOB_FAILED)])
def test_child_runs_chain_in_order(monkeypatch, code, state):
    events = _patch_chain(monkeypatch, run=SimpleNamespace(exit_code=code, output=b"ok"))
    store = Store()
    assert gd.run_child(store, "j1", CONFIG, api=object()) == state
    names = [c[0] for c in store.calls]
    assert names == ["get_job", "mark_running", "mark_terminal"]
    assert events == ["launch", "start", "run", "retire"]
    assert store.calls[-1][2]["state"] == state


def test_uncertain_launch_is_lost_not_failed(monkeypatch):
    from llm_bawt.ops.gate_runner import GateLaunchError
    exc = GateLaunchError("runner_exit_uncertain")
    events = _patch_chain(monkeypatch, launch_error=exc)
    store = Store()
    assert gd.run_child(store, "j1", CONFIG, api=object()) == JOB_LOST
    assert events[-1] == "retire"
