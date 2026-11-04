"""TASK-1991: finished per-job workers are reaped; a wedged ops process exits for restart."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from llm_bawt.ops.executor import DockerExecutor, ExecutorError
from llm_bawt.ops.health_runtime import WEDGE_EXIT_CODE, OpsHealthRuntime, _wedge_seconds
from llm_bawt.ops.health_state import HealthSample, OpsHealthState
from llm_bawt.ops.worker import OPS_JOB_LABEL

DONE = "a" * 32
ACTIVE = "b" * 32
OTHER = "c" * 32


class Worker:
    def __init__(self, job_id, *, name=None, fail=None):
        self.labels = {OPS_JOB_LABEL: job_id}
        self.name = name or f"llm-bawt-ops-{job_id}"
        self.removed = False
        self.fail = fail

    def remove(self, v=False, force=False):
        assert v is False and force is False
        if self.fail:
            raise self.fail
        self.removed = True


def executor_with(workers):
    calls = []

    def list_(all=False, filters=None):
        calls.append((all, filters))
        return list(workers)

    client = SimpleNamespace(containers=SimpleNamespace(list=list_))
    return DockerExecutor(client_factory=lambda: client, worker_image="w", receipt_volume="r",
                          receipt_root="/nonexistent"), calls


def test_reaps_only_stopped_workers_of_finished_jobs():
    done, active = Worker(DONE), Worker(ACTIVE)
    spoofed = Worker(OTHER, name="something-else")
    bad_label = Worker("not-a-job-id")
    executor, calls = executor_with([done, active, spoofed, bad_label])
    removed = executor.reap_finished_workers(lambda job: job in {DONE, OTHER})
    assert removed == 1
    assert done.removed and not active.removed and not spoofed.removed and not bad_label.removed
    # Only stopped containers are ever listed; running/created workers stay.
    assert calls == [(True, {"label": OPS_JOB_LABEL, "status": ["exited", "dead"]})]


def test_reap_is_bounded_and_survives_a_failed_remove():
    gone = Worker(DONE, fail=RuntimeError("busy"))
    later = [Worker(f"{i:032x}") for i in range(1, 5)]
    executor, _ = executor_with([gone, *later])
    assert executor.reap_finished_workers(lambda _job: True, limit=3) == 2
    assert not gone.removed and later[0].removed and later[1].removed and not later[2].removed


def test_unreapable_workers_do_not_starve_older_ones():
    stuck = [Worker(f"{i:032x}") for i in range(1, 61)]
    old = Worker(DONE)
    executor, _ = executor_with([*stuck, old])
    assert executor.reap_finished_workers(lambda job: job == DONE, limit=50) == 1
    assert old.removed and not any(w.removed for w in stuck)


def test_ops_process_bounds_database_connects(monkeypatch):
    from llm_bawt.ops import reconciler
    monkeypatch.delenv("PGCONNECT_TIMEOUT", raising=False)
    monkeypatch.setattr("llm_bawt.ops.health_runtime.OpsHealthRuntime", lambda **_kw: (_ for _ in ()).throw(SystemExit))
    with pytest.raises(SystemExit):
        reconciler.main()
    import os
    assert os.environ["PGCONNECT_TIMEOUT"] == "10"


def test_dispatch_refuses_a_job_whose_receipt_exists(tmp_path):
    executor = DockerExecutor(client_factory=lambda: None, worker_image="w", receipt_volume="r",
                              receipt_root=str(tmp_path))
    executor.preflight = lambda _snapshot: None
    (tmp_path / DONE).mkdir()
    (tmp_path / DONE / "receipt.json").write_text(json.dumps({"state": "succeeded"}))
    with pytest.raises(ExecutorError, match="never replay"):
        executor.dispatch(job_id=DONE, snapshot={"execution": {"receipt_root": str(tmp_path)}})


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def test_silent_lists_only_producers_that_stopped_publishing():
    clock = Clock()
    state = OpsHealthState(clock=clock)
    clock.now += 200
    for name in ("database", "docker", "ops.executor"):
        state.publish(name, HealthSample(state.sample_time(), "down", "reported outage"))
    assert state.silent(180) == ["reconciler.loop"]
    state.publish("reconciler.loop", HealthSample(state.sample_time(), "ok"))
    assert state.silent(180) == []
    state.stop()
    clock.now += 1000
    assert state.silent(180) == []  # shutting down is not a wedge


def test_watchdog_exits_once_when_a_producer_is_silent():
    clock = Clock()
    state = OpsHealthState(clock=clock)
    exits = []
    runtime = OpsHealthRuntime(state=state, collector=SimpleNamespace(), interval=0.01,
                               wedge_seconds=60, exit=exits.append)
    clock.now += 61
    runtime.stop.wait = lambda _timeout: bool(exits)
    runtime._watchdog()
    assert exits == [WEDGE_EXIT_CODE]


def test_wedge_setting(monkeypatch):
    monkeypatch.delenv("LLM_BAWT_OPS_WEDGE_SECONDS", raising=False)
    assert _wedge_seconds() == 180
    monkeypatch.setenv("LLM_BAWT_OPS_WEDGE_SECONDS", "0")
    assert _wedge_seconds() == 0
    for bad in ("30", "-1", "nan"):
        monkeypatch.setenv("LLM_BAWT_OPS_WEDGE_SECONDS", bad)
        with pytest.raises(ValueError):
            _wedge_seconds()
