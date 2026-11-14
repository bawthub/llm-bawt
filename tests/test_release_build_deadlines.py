"""TASK-2232: durable phase budgets using fake GitHub and a controlled clock."""
# ruff: noqa: F811
import json
from datetime import timedelta

import pytest
from sqlmodel import Session

from llm_bawt.ops.github_workflow import GitHubWorkflowError
from llm_bawt.ops.release_coordinator import BUILD_WINDOW, ReleaseCoordinator, _aware
from llm_bawt.ops.release_models import RELEASE_BUILD_COMPLETE, RELEASE_BUILDING, RELEASE_LOST, ReleaseRun
from tests.test_release_coordinator import SNAPSHOT, _receipt, _run, _start, rig  # noqa: F401


def unavailable(*args, **kwargs):
    raise GitHubWorkflowError("github_unreachable", "offline fixture", retryable=True)


def restart(coordinator, store, gateway):
    return ReleaseCoordinator(releases=store, ops=None, gateway=gateway,
                              approvals=lambda: None, clock=coordinator._clock)


def one_step(coordinator, store, row):
    current = store.claim(row.id)
    assert current is not None
    try:
        return coordinator._step(current, current.claim_token, SNAPSHOT)
    finally:
        store.release_claim(row.id, current.claim_token, next_check_at=coordinator._clock())


def correlate(rig, *, partial=False):
    coordinator, store, gateway, advance = rig
    row = _start(*rig)
    gateway.runs = [_run()]
    gateway.runs[0]["run_started_at"] = coordinator._clock().isoformat()
    if partial:
        gateway.receipts[1] = _receipt(row, attempt=1, status="partial")
    else:
        gateway.runs[0]["status"] = "in_progress"
    advance()
    coordinator.pump(row.parent_job_id, SNAPSHOT)
    return store.get(row.id)


@pytest.mark.parametrize("legacy", [False, True])
def test_initial_outage_is_bounded_across_restart(rig, legacy):
    coordinator, store, gateway, advance = rig
    row = correlate(rig)
    assert row.phase_deadline_at is not None
    if legacy:
        with Session(store.engine) as session:
            old = session.get(ReleaseRun, row.id)
            old.phase_deadline_at = None
            plan = json.loads(old.release_plan_json)
            plan.pop("confirmed_build_attempt")
            old.release_plan_json = json.dumps(plan)
            session.add(old)
            session.commit()
    gateway.get_run = unavailable
    advance(9 * 3600)
    resumed = restart(coordinator, store, gateway)
    resumed.pump(row.parent_job_id, SNAPSHOT)
    assert store.get(row.id).state == RELEASE_LOST
    assert store.get(row.id).phase_deadline_at is not None
    assert len(gateway.dispatches) == 1 and gateway.reruns == []


def test_registered_rerun_gets_build_budget_not_old_registration_deadline(rig):
    coordinator, store, gateway, advance = rig
    row = correlate(rig, partial=True)
    intent = _aware(row.phase_deadline_at) - timedelta(minutes=10)
    assert "confirmed_build_attempt" not in json.loads(row.release_plan_json)
    advance(11 * 60)
    # GitHub retained the original attempt's old timestamp.
    gateway.runs = [{**_run(attempt=2), "status": "in_progress",
                     "run_started_at": (intent - timedelta(hours=9)).isoformat()}]
    coordinator.pump(row.parent_job_id, SNAPSHOT)
    confirmed = store.get(row.id)
    assert confirmed.state == RELEASE_BUILDING
    assert _aware(confirmed.phase_deadline_at) == intent + BUILD_WINDOW
    assert json.loads(confirmed.release_plan_json)["confirmed_build_attempt"] == 2
    assert json.loads(confirmed.release_plan_json)["inputs"] == json.loads(row.release_plan_json)["inputs"]
    gateway.get_run = unavailable
    advance()
    resumed = restart(coordinator, store, gateway)
    assert resumed.pump(row.parent_job_id, SNAPSHOT).state == "running"
    assert _aware(store.get(row.id).phase_deadline_at) == intent + BUILD_WINDOW
    advance(8 * 3600)
    assert resumed.pump(row.parent_job_id, SNAPSHOT).state == "lost"
    assert len(gateway.dispatches) == 1 and gateway.reruns == ["123456"]


def test_missing_rerun_retains_registration_timeout(rig):
    coordinator, store, gateway, advance = rig
    row = correlate(rig, partial=True)
    advance(11 * 60)
    assert restart(coordinator, store, gateway).pump(row.parent_job_id, SNAPSHOT).state == "lost"
    assert store.get(row.id).error_code == "rerun_not_registered"
    assert gateway.reruns == ["123456"]


def test_receipt_failure_in_registration_step_uses_new_deadline(rig):
    coordinator, store, gateway, advance = rig
    row = correlate(rig, partial=True)
    advance(11 * 60)
    gateway.runs = [{**_run(attempt=2, conclusion="success"), "updated_at": coordinator._clock().isoformat()}]
    gateway.download_release_receipt = unavailable
    assert coordinator.pump(row.parent_job_id, SNAPSHOT).state == "running"
    assert _aware(store.get(row.id).phase_deadline_at) > coordinator._clock()
    advance(9 * 3600)
    assert restart(coordinator, store, gateway).pump(row.parent_job_id, SNAPSHOT).state == "lost"
    assert gateway.reruns == ["123456"]


def test_completed_build_is_recoverable_after_budget_expires(rig):
    coordinator, store, gateway, advance = rig
    row = correlate(rig)
    deadline = _aware(row.phase_deadline_at)
    gateway.runs = [{**_run(conclusion="success"), "updated_at": (deadline - timedelta(minutes=10)).isoformat()}]
    gateway.receipts[1] = _receipt(row, attempt=1, status="complete")
    advance(8 * 3600 + 5 * 60)
    one_step(restart(coordinator, store, gateway), store, row)
    assert store.get(row.id).state == RELEASE_BUILD_COMPLETE
    assert store.get(row.id).deployable
    assert len(gateway.dispatches) == 1 and not gateway.reruns


def test_completion_receipt_grace_is_bounded_by_remote_completion_time(rig):
    coordinator, store, gateway, advance = rig
    row = correlate(rig)
    deadline = _aware(row.phase_deadline_at)
    gateway.runs = [{**_run(conclusion="success"), "updated_at": deadline.isoformat()}]
    gateway.download_release_receipt = unavailable
    advance(8 * 3600)
    assert coordinator.pump(row.parent_job_id, SNAPSHOT).state == "running"
    assert _aware(store.get(row.id).phase_deadline_at) == deadline + timedelta(minutes=2)
    advance(3 * 60)
    assert restart(coordinator, store, gateway).pump(row.parent_job_id, SNAPSHOT).state == "lost"


def test_external_attempt_adoption_persists_identity_and_budget_together(rig):
    coordinator, store, gateway, advance = rig
    row = correlate(rig)
    advance(7 * 3600)
    started = coordinator._clock()
    gateway.runs = [{**_run(attempt=2), "status": "in_progress", "run_started_at": started.isoformat()}]
    coordinator.pump(row.parent_job_id, SNAPSHOT)
    adopted = store.get(row.id)
    assert adopted.github_run_attempt == 2
    assert json.loads(adopted.release_plan_json)["confirmed_build_attempt"] == 2
    assert _aware(adopted.phase_deadline_at) == started + BUILD_WINDOW
    gateway.get_run = unavailable
    advance(2 * 3600)
    assert restart(coordinator, store, gateway).pump(row.parent_job_id, SNAPSHOT).state == "running"
    assert _aware(store.get(row.id).phase_deadline_at) == started + BUILD_WINDOW
    assert len(gateway.dispatches) == 1 and not gateway.reruns


def test_running_build_times_out_on_successful_read_too(rig):
    coordinator, store, gateway, advance = rig
    row = correlate(rig)
    advance(9 * 3600)
    assert coordinator.pump(row.parent_job_id, SNAPSHOT).state == "lost"
    assert store.get(row.id).error_code == "build_timeout"
