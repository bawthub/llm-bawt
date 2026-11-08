"""TASK-1263 durable sequence receipts. Fake Docker only; never restart services."""
from __future__ import annotations

import copy
import asyncio
import hashlib
import json
from contextlib import contextmanager
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from llm_bawt.ops import sequence, worker
from llm_bawt.ops.executor import worker_request
from llm_bawt.ops.restart_health import RestartUnhealthy
from tests.test_ops_executor import JOB, setup as executor_setup
from tests.test_ops_sequence_admission import prepare, rig, step
from tests.test_ops_sequence_store import create_job, operation, sequence_snapshot
from tests.test_ops_store import _store


class Crash(BaseException):
    """A process death cannot be handled as an ordinary step failure."""


class Docker(worker.DockerAPI):
    def __init__(self, snapshot, path, *, fail=None, crash=False):
        super().__init__(timeout=30)
        self.path, self.fail, self.crash = path, fail, crash
        self.events = []
        self.containers = {
            item["target"]["name"]: {"Id": item["target"]["id"], "Name": "/" + item["target"]["name"]}
            for item in snapshot["sequence"]["steps"]
        }

    def request(self, method, path, **kwargs):
        parts = path.split("/")
        name = parts[2]
        target = self.containers.get(name) or next(c for c in self.containers.values() if c["Id"] == name)
        if method == "GET":
            return copy.deepcopy(target)
        assert method == "POST" and parts[3].split("?")[0] in ("restart", "start")
        receipt = json.loads(self.path.with_name("receipt.json").read_text())
        active = next(s for s in receipt["sequence"]["steps"] if s["state"] == "running")
        assert active["target"] == target["Name"].lstrip("/")
        assert active["started_at"] and active["exit"] is None  # durable intent precedes POST
        assert self.timeout == 60  # the child's budget, not the 5400s parent budget
        assert 0 < worker.signal.getitimer(worker.signal.ITIMER_REAL)[0] <= 60
        self.events.append(("submit", active["target"]))
        if self.crash:
            raise Crash()
        if self.fail == active["target"]:
            raise TimeoutError("response lost")


def request_file(tmp_path, snapshot, fence=None):
    path = tmp_path / JOB / "request.json"
    path.parent.mkdir(exist_ok=True)
    raw = json.dumps(worker_request(JOB, snapshot, fence), sort_keys=True).encode()
    path.write_bytes(raw)
    return path, hashlib.sha256(raw).hexdigest()


def fixture(tmp_path, monkeypatch, *, fail=None, crash=False):
    r = rig()
    snapshot = prepare(r, [step(), step("test.second")])
    path, digest = request_file(tmp_path, snapshot)
    api = Docker(snapshot, path, fail=fail, crash=crash)

    def health(_api, container_id, spec, **kwargs):
        target = next(c for c in api.containers.values() if c["Id"] == container_id)
        api.events.append(("healthy", target["Name"].lstrip("/")))
        return "healthy"

    monkeypatch.setattr(worker, "wait_healthy", health)
    return snapshot, path, digest, api


def receipt(path):
    return json.loads(path.with_name("receipt.json").read_text())


def invoke(path, digest, api, **kwargs):
    return worker.run(path, digest, api_factory=lambda timeout: api, sleep=lambda seconds: None, **kwargs)


def test_order_health_and_durable_results_survive_text_truncation_and_replay(tmp_path, monkeypatch):
    snapshot, path, digest, api = fixture(tmp_path, monkeypatch)
    # The structured record is not part of the parent's text-output budget.
    snapshot["execution"]["max_output_bytes"] = 1
    path, digest = request_file(tmp_path, snapshot)
    assert invoke(path, digest, api) == 0
    final = receipt(path)
    assert final["state"] == "succeeded"
    assert api.events == [("submit", "first"), ("healthy", "first"),
                          ("submit", "second"), ("healthy", "second")]
    ledger = sequence.read_ledger(final["sequence"])
    assert len(json.dumps(ledger)) > snapshot["execution"]["max_output_bytes"]
    assert [s["state"] for s in ledger["steps"]] == ["succeeded", "succeeded"]
    assert all(s["exit"] == 0 and s["health"] == {"state": "healthy"} for s in ledger["steps"])
    assert invoke(path, digest, api) == 0
    assert len(api.events) == 4


@pytest.mark.parametrize("unknown", [False, True])
def test_failed_health_or_unknown_mutation_stops_remaining_steps(tmp_path, monkeypatch, unknown):
    _, path, digest, api = fixture(tmp_path, monkeypatch, fail="first" if unknown else None)
    if not unknown:
        def unhealthy(*args, **kwargs):
            raise RestartUnhealthy("service did not recover")
        monkeypatch.setattr(worker, "wait_healthy", unhealthy)
    assert invoke(path, digest, api) == 1
    final = receipt(path)
    assert final["state"] == ("lost" if unknown else "failed")
    assert [s["state"] for s in final["sequence"]["steps"]] == [
        "lost" if unknown else "failed", "skipped",
    ]
    assert final["sequence"]["steps"][1]["started_at"] is None
    assert api.events == [("submit", "first")]
    assert final["side_effect_unknown"] is unknown
    assert final["exit_code"] == (None if unknown else 1)


def test_all_step_preflight_refuses_before_any_mutation(tmp_path, monkeypatch):
    _, path, digest, api = fixture(tmp_path, monkeypatch)
    api.containers["second"]["Id"] = "e" * 64
    assert invoke(path, digest, api) == 1
    assert api.events == []
    assert [s["state"] for s in receipt(path)["sequence"]["steps"]] == ["skipped", "skipped"]


def test_each_step_rechecks_after_prior_success_without_erasing_it(tmp_path, monkeypatch):
    _, path, digest, api = fixture(tmp_path, monkeypatch)

    def change_next(*args, **kwargs):
        api.containers["second"]["Id"] = "e" * 64
        return "healthy"

    monkeypatch.setattr(worker, "wait_healthy", change_next)
    assert invoke(path, digest, api) == 1
    final = receipt(path)
    assert final["state"] == "failed"
    assert [s["state"] for s in final["sequence"]["steps"]] == ["succeeded", "skipped"]
    assert api.events == [("submit", "first")]


def test_intent_survives_process_death_and_restart_does_not_replay(tmp_path, monkeypatch):
    _, path, digest, api = fixture(tmp_path, monkeypatch, crash=True)
    with pytest.raises(Crash):
        invoke(path, digest, api)
    assert receipt(path)["sequence"]["steps"][0]["state"] == "running"
    assert invoke(path, digest, api) == 1
    final = receipt(path)
    assert final["state"] == "lost"
    assert final["sequence"]["state"] == "lost_requires_inspection"
    assert [s["state"] for s in final["sequence"]["steps"]] == ["lost", "skipped"]
    assert api.events == [("submit", "first")]


def test_whole_target_lock_set_is_sorted_and_blocks_standalone_restarts(tmp_path, monkeypatch):
    snapshot, path, digest, api = fixture(tmp_path, monkeypatch)
    acquired = []
    original = worker.target_lock

    @contextmanager
    def locks(root, name):
        with original(root, name):
            acquired.append(name)
            yield

    monkeypatch.setattr(worker, "target_lock", locks)
    # A later target is busy. No earlier target may already have restarted.
    with original(tmp_path / ".target-locks", "second"):
        assert invoke(path, digest, api) == 1
    assert acquired == ["first"]
    assert api.events == []
    assert receipt(path)["sequence"]["state"] == "failed"
    child = snapshot["sequence"]["steps"][0]["snapshot"]
    with original(tmp_path / ".target-locks", "first"):
        with pytest.raises(BlockingIOError):
            worker.execute_action(api, child, JOB, lock_root=tmp_path / ".target-locks", receipt={})
    assert api.events == []


def test_expired_fence_reports_all_steps_skipped_not_lost(tmp_path, monkeypatch):
    snapshot, path, _, api = fixture(tmp_path, monkeypatch)
    path, digest = request_file(tmp_path, snapshot, {"start_not_after": "2020-01-01T00:00:00Z"})
    assert invoke(path, digest, api, clock=lambda: datetime(2026, 1, 1, tzinfo=timezone.utc)) == 1
    final = receipt(path)
    assert final["state"] == "failed"
    assert final["sequence"]["state"] == "failed"
    assert [s["state"] for s in final["sequence"]["steps"]] == ["skipped", "skipped"]
    assert api.events == []


@pytest.mark.parametrize("state", ["missing", "exited", "running", "unconfirmed"])
def test_reconciliation_retains_ledger_and_never_replays(tmp_path, monkeypatch, state):
    snapshot, _, _, _ = fixture(tmp_path, monkeypatch)
    executor, client, single = executor_setup(tmp_path, monkeypatch)
    snapshot["execution"] = single["execution"]
    snapshot["operation"]["execution_mode"] = snapshot["execution"]["execution_mode"] = "isolated"
    # Admission has its own coverage; this test exercises durable worker launch/reconciliation.
    monkeypatch.setattr(executor, "preflight", lambda snapshot: None)
    # dispatch refuses an existing request, so use its canonical immutable request.
    (tmp_path / JOB / "request.json").unlink()
    executor.dispatch(job_id=JOB, snapshot=snapshot)
    path = tmp_path / JOB / "request.json"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    ledger = sequence.initial_ledger(snapshot)
    ledger["state"] = "running"
    ledger["steps"][0].update(state="succeeded", exit=0, health={"state": "healthy"})
    ledger["steps"][1].update(state="running", started_at="2026-01-01T00:00:00Z")
    worker.atomic_json(path.with_name("receipt.json"), {
        "job_id": JOB, "request_hash": digest, "state": "running",
        "started_at": "2026-01-01T00:00:00Z", "sequence": ledger,
    })
    container = client.containers.rows[executor.worker_name(JOB)]
    if state == "missing":
        client.containers.rows.pop(executor.worker_name(JOB))
    elif state == "exited":
        container.status = "exited"
    elif state == "unconfirmed":
        from llm_bawt.ops.worker_launch import Liveness
        launcher = executor._launcher_for(snapshot)
        monkeypatch.setattr(launcher, "probe", lambda **kwargs: Liveness.START_UNCONFIRMED)
    result = executor.reconcile(job_id=JOB, snapshot=snapshot, output_tail_bytes=1)
    assert result.state == ("running" if state == "running" else "lost")
    assert [s["state"] for s in result.sequence["steps"]] == [
        "succeeded", "running" if state == "running" else "lost",
    ]
    assert len(client.containers.creates) == 1 and container.starts == 1


def test_queued_failure_has_a_full_parent_ledger_and_terminal_callback(tmp_path):
    from llm_bawt.service.approval_continuations import _terminal_ops_result
    from llm_bawt.approval_policies import KIND_MCP
    from llm_bawt.ops.service import OpsService
    from tests.test_ops_service import FakeExecutor

    store = _store()
    parent, first, second = (operation(store, n) for n in ("parent", "first", "second"))
    snapshot = sequence_snapshot(store, parent, [first, second])
    job = create_job(store, parent, snapshot)
    assert len(store.get_job(job.id).to_api()["sequence"]["steps"]) == 2
    store.mark_terminal(job.id, state="failed", error_text="preflight refused; no worker submitted")
    row = SimpleNamespace(request_kind=KIND_MCP, tool_name="ops_run",
                          result_json=json.dumps({"job_id": job.id, "state": "queued"}))
    # A fresh service (as after app restart) reads the parent column, not a text tail or child job.
    service = SimpleNamespace(_ops_service=OpsService(store, executor=FakeExecutor()))
    result = _terminal_ops_result(service, row)
    assert result["terminal"] and result["state"] == "failed"
    assert [s["state"] for s in result["sequence"]["steps"]] == ["skipped", "skipped"]
    assert store.count_jobs() == 1


@pytest.mark.parametrize("mutation", ["truncated", "wrong-op", "unfinished-success", "unfinished-failure", "oversized"])
def test_malformed_or_incomplete_ledgers_are_rejected(tmp_path, monkeypatch, mutation):
    snapshot, _, _, _ = fixture(tmp_path, monkeypatch)
    ledger = sequence.initial_ledger(snapshot)
    if mutation == "truncated":
        ledger["steps"].pop()
    elif mutation == "wrong-op":
        ledger["steps"][0]["op"] = "different.operation"
    elif mutation == "unfinished-success":
        ledger["state"] = "succeeded"
    elif mutation == "unfinished-failure":
        ledger["state"] = "failed"
    else:
        ledger["steps"][0]["args"] = {"oversized": "x" * sequence.MAX_LEDGER_BYTES}
    with pytest.raises(ValueError):
        sequence.read_ledger(ledger, snapshot)


@pytest.mark.parametrize("state", ["succeeded", "failed", "lost"])
def test_one_terminal_callback_preserves_all_six_steps_after_service_recreation(monkeypatch, state):
    from tests.test_mcp_approval_continuations import (
        OpsServiceHost, _ready_row, _store as approval_store,
    )
    from llm_bawt.service.approval_continuations import dispatch_due_continuations_once
    from tests.test_ops_sequence_admission import child

    r = rig([child(f"test.step-{i}", name=f"target-{i}") for i in range(6)])
    snapshot = prepare(r, [step(f"test.step-{i}") for i in range(6)])
    ledger = sequence.initial_ledger(snapshot)
    ledger["state"] = state if state != "lost" else "lost_requires_inspection"
    for item in ledger["steps"]:
        item.update(state="succeeded", exit=0, health={"state": "healthy"}, output_tail="x" * 1024)
    if state != "succeeded":
        ledger["steps"][4].update(state=state, exit=1 if state == "failed" else None)
        ledger["steps"][5].update(state="skipped", exit=None, health=None)
    sequence.read_ledger(ledger)
    store = approval_store()
    row = _ready_row(store, result={"job_id": "job-1", "state": "accepted"})
    store.mark_continuation_failed(row.id, error="fixture reset", backoff_seconds=0,
                                   claim_token=row.continuation_claim_token)
    monkeypatch.setattr("llm_bawt.service.approval_continuations.recover_approval_requests",
                        lambda _store: asyncio.sleep(0))
    active_service = OpsServiceHost({"job_id": "job-1", "state": "running", "terminal": False})
    asyncio.run(dispatch_due_continuations_once(active_service, store))
    assert active_service.requests == []
    # The original app/calling bridge is gone. A fresh host imports the one
    # parent result and ordinary continuation ownership deduplicates delivery.
    terminal = {"job_id": "job-1", "state": state, "terminal": True, "sequence": ledger}
    fresh = OpsServiceHost(terminal)
    asyncio.run(dispatch_due_continuations_once(fresh, store))
    asyncio.run(dispatch_due_continuations_once(fresh, store))
    assert len(fresh.requests) == 1
    payload = fresh.requests[0].continuation_payload
    assert payload.result["sequence"] == ledger
    assert payload.is_error is (state != "succeeded")
    assert '"step": 6' in fresh.requests[0].messages[0].content
    assert len(json.dumps(payload.result)) > 4096


@pytest.mark.parametrize("kind", ["rollback", "deadline"])
def test_known_rollback_and_pre_submit_deadline_remain_structured(tmp_path, monkeypatch, kind):
    from llm_bawt.ops.image_deploy import DeployFailed

    snapshot, _, _, api = fixture(tmp_path, monkeypatch)
    saved = []

    def execute(item):
        api.side_effect_started = True
        raise DeployFailed("health failed", {"restored": True}, outcome_known=True)

    def sleep(seconds):
        if kind == "deadline":
            raise TimeoutError("deadline before submission")

    result = sequence.run(snapshot, api=api, preflight=lambda item: None, execute=execute,
                          checkpoint=lambda ledger: saved.append(copy.deepcopy(ledger)), sleep=sleep)
    assert result["state"] == "failed"
    first = result["steps"][0]
    assert first["state"] == ("failed" if kind == "rollback" else "skipped")
    assert first["rollback"] == ({"restored": True} if kind == "rollback" else None)
    assert result["steps"][1]["state"] == "skipped"
    assert saved[-1] == result
