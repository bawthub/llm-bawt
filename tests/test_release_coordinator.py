"""TASK-1030: fake-remote release recovery without local clones or live deploys."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from sqlmodel import create_engine

from llm_bawt.ops.executor import ReconcileResult
from llm_bawt.ops.release_coordinator import ReleaseCoordinator
from llm_bawt.ops.release_models import (
    RELEASE_AWAITING_DEPLOY_APPROVAL, RELEASE_BUILDING, RELEASE_DISPATCHING_BUILD,
    RELEASE_LOST,
)
from llm_bawt.ops.release_store import ReleaseStore

SOURCE = "1" * 40
BASE = "2" * 40
DIGEST = "sha256:" + "d" * 64
SPEC = {
    "action": "release_orchestrate",
    "github_repository": "bawthub/bawthub",
    "workflow_path": ".github/workflows/release-frontend.yml",
    "canonical_branch": "main",
    "image_repository": "ghcr.io/bawthub/frontend",
    "deploy_operation": "bawthub.deploy-prod-image",
}
SNAPSHOT = {"spec": SPEC, "resolved_args": {"release_task": "TASK-1030", "bump": "patch"},
            "release_source": {"expected_sha": BASE}}


class FakeGateway:
    def __init__(self):
        self.heads = []
        self.dispatches = []
        self.reruns = []
        self.runs = []
        self.receipts = {}

    def resolve_branch_head(self, repo, branch):
        assert repo == "bawthub/bawthub", f"release must not consult {repo}"
        self.heads.append((repo, branch))
        return BASE

    def dispatch_workflow(self, repo, path, branch, inputs):
        self.dispatches.append((repo, path, branch, inputs.copy()))

    def find_correlated_runs(self, repo, path, branch, request_id, *, created_after):
        return self.runs

    def get_run(self, repo, run_id):
        return self.runs[0]

    def rerun_failed_jobs(self, repo, run_id):
        self.reruns.append(run_id)

    def download_release_receipt(self, repo, run_id, attempt):
        return self.receipts[attempt]


@pytest.fixture
def rig(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'release.sqlite'}")
    store = ReleaseStore(None, engine=engine)
    gateway = FakeGateway()
    current = [datetime.now(timezone.utc)]
    coordinator = ReleaseCoordinator(releases=store, ops=None, gateway=gateway,
                                     approvals=lambda: None, clock=lambda: current[0])

    def advance(seconds=30):
        current[0] += timedelta(seconds=seconds)

    return coordinator, store, gateway, advance


def _run(attempt=1, conclusion="failure"):
    return {"id": 123456, "run_attempt": attempt, "html_url": "https://github.com/bawthub/bawthub/actions/runs/123456",
            "status": "completed", "conclusion": conclusion, "updated_at": datetime.now(timezone.utc).isoformat()}


def _receipt(row, *, attempt, status):
    return {"schema": "bawthub.release-receipt/v1", "status": status,
            "repository": "bawthub/bawthub", "workflow_run_id": "123456",
            "workflow_run_attempt": str(attempt), "release_request_id": row.release_request_id,
            "base_sha": BASE, "source_sha": SOURCE, "version": "0.1.63", "tag": "v0.1.63",
            "digest": DIGEST, "image_repository": SPEC["image_repository"],
            "image_ref": f'{SPEC["image_repository"]}@{DIGEST}'}


def _start(coordinator, store, gateway, advance):
    first = coordinator.pump("a" * 32, SNAPSHOT)
    assert isinstance(first, ReconcileResult)
    row = store.get_by_parent_job("a" * 32)
    assert row.state == RELEASE_DISPATCHING_BUILD
    assert gateway.heads == [("bawthub/bawthub", "main")]
    assert gateway.dispatches[0][1] == "release-frontend.yml"
    assert gateway.dispatches[0][3]["expected_sha"] == BASE
    return row


def test_dispatch_intent_survives_restart_and_does_not_double_bump(rig):
    coordinator, store, gateway, advance = rig
    row = _start(coordinator, store, gateway, advance)
    gateway.runs = [_run()]
    gateway.receipts[1] = _receipt(row, attempt=1, status="partial")
    advance()
    # Reinstantiate the coordinator as after a process restart; the persisted
    # intent correlates the SAME run instead of dispatching a second workflow.
    resumed = ReleaseCoordinator(releases=store, ops=None, gateway=gateway,
                                 approvals=lambda: None, clock=coordinator._clock)
    result = resumed.pump(row.parent_job_id, SNAPSHOT)
    assert result.state == "running"
    assert store.get(row.id).state == RELEASE_BUILDING
    assert len(gateway.dispatches) == 1


def test_remote_branch_moved_since_approval_never_dispatches(rig):
    coordinator, store, gateway, _advance = rig
    snapshot = {**SNAPSHOT, "release_source": {"expected_sha": "4" * 40}}
    result = coordinator.pump("a" * 32, snapshot)
    assert result.state == "failed"
    assert store.get_by_parent_job("a" * 32).error_code == "approved_source_moved"
    assert gateway.dispatches == []


def test_partial_receipt_reruns_failed_jobs_on_same_run(rig):
    coordinator, store, gateway, advance = rig
    row = _start(coordinator, store, gateway, advance)
    gateway.runs = [_run()]
    gateway.receipts[1] = _receipt(row, attempt=1, status="partial")
    advance()
    coordinator.pump(row.parent_job_id, SNAPSHOT)
    current = store.get(row.id)
    assert current.state == RELEASE_BUILDING
    assert current.github_run_id == "123456" and current.github_run_attempt == 2
    assert current.rerun_count == 1 and gateway.reruns == ["123456"]
    assert len(gateway.dispatches) == 1


def test_uncertain_rerun_is_never_reissued_when_attempt_registration_lags(rig):
    from llm_bawt.ops.github_workflow import GitHubWorkflowError

    coordinator, store, gateway, advance = rig
    row = _start(coordinator, store, gateway, advance)
    gateway.runs = [_run()]
    gateway.receipts[1] = _receipt(row, attempt=1, status="partial")
    attempts = []
    def timed_out_rerun(repo, run_id):
        attempts.append(run_id)
        raise GitHubWorkflowError("github_unreachable", "timeout", retryable=True)
    gateway.rerun_failed_jobs = timed_out_rerun
    advance()
    coordinator.pump(row.parent_job_id, SNAPSHOT)
    assert attempts == ["123456"]
    advance(60)
    coordinator.pump(row.parent_job_id, SNAPSHOT)
    assert attempts == ["123456"]
    advance(11 * 60)
    assert coordinator.pump(row.parent_job_id, SNAPSHOT).state == "lost"
    assert attempts == ["123456"]


def test_ambiguous_run_correlation_fails_closed(rig):
    coordinator, store, gateway, advance = rig
    row = _start(coordinator, store, gateway, advance)
    gateway.runs = [_run(), _run()]
    advance()
    result = coordinator.pump(row.parent_job_id, SNAPSHOT)
    assert result.state == "lost"
    assert store.get(row.id).state == RELEASE_LOST
    assert store.get(row.id).error_code == "correlation_ambiguous"
    assert len(gateway.dispatches) == 1


@pytest.mark.parametrize("phase, restored, expected", [
    ("inspect", False, "deploy_declined"),
    ("verify-image", False, "deploy_declined"),
    ("restore", True, "deploy_failed_restored"),
    ("stop-old", False, RELEASE_LOST),
])
def test_failed_deploy_state_is_honest_about_restoration(rig, phase, restored, expected):
    import json
    from llm_bawt.ops.image_deploy import DEPLOY_SCHEMA
    from llm_bawt.ops.release_models import RELEASE_DEPLOYING, RELEASE_PREFLIGHT

    coordinator, store, _gateway, _advance = rig
    row = coordinator.create_for_job("a" * 32, SNAPSHOT)
    token = store.claim(row.id).claim_token
    row = store.transition(row.id, from_states=RELEASE_PREFLIGHT, to_state=RELEASE_DEPLOYING,
                           claim_token=token, event_type="test.deploy")
    child = SimpleNamespace(id="c" * 32, state="failed", exit_code=1,
                            dispatched_at=datetime.now(timezone.utc),
                            error_text="worker failed", output_tail=json.dumps({
                                "schema": DEPLOY_SCHEMA, "action": "deploy", "phase": phase,
                                "restored": restored, "error": "failed in " + phase,
                            }))
    coordinator._finish_deploy(row, token, child, {})
    assert store.get(row.id).state == expected


def test_denied_deploy_never_dispatches_child_and_keeps_verified_build(rig):
    from llm_bawt.approval_models import REQ_DENIED
    from llm_bawt.approval_policies import ToolApprovalPolicyStore
    from llm_bawt.ops.release_models import RELEASE_DEPLOY_DECLINED

    coordinator, store, gateway, advance = rig
    approvals = ToolApprovalPolicyStore(None, engine=store.engine)
    parent = SimpleNamespace(caller_bot_id="test-bot", caller_user_id="test-user",
                             caller_turn_id="turn-1", caller_session_key="test-bot:test-user",
                             caller_backend="claude-code")
    class FakeOps:
        def __init__(self):
            self.store = self
            self.dispatches = []

        def get_job(self, _job_id):
            return parent

        def prepare_invocation(self, slug, args):
            binding = store.verified_binding(args["release_run_id"])
            return {"operation": {"slug": slug}, "input_args": args,
                    "release": {**binding, "expected_current_image_id": "sha256:" + "a" * 64}}

        def dispatch_job(self, **kwargs):
            self.dispatches.append(kwargs)
            raise AssertionError("denied release must not dispatch")

    ops = FakeOps()
    coordinator.ops = ops
    coordinator._approvals = lambda: approvals
    row = _start(coordinator, store, gateway, advance)
    gateway.runs = [_run(conclusion="success")]
    gateway.receipts[1] = _receipt(row, attempt=1, status="complete")
    advance()
    coordinator.pump(row.parent_job_id, SNAPSHOT)
    waiting = store.get(row.id)
    assert waiting.state == RELEASE_AWAITING_DEPLOY_APPROVAL
    approvals.resolve_request(waiting.deploy_approval_request_id, status=REQ_DENIED,
                              resolved_by="test-operator", continuation_owner="none")
    advance()
    assert coordinator.pump(row.parent_job_id, SNAPSHOT).state == "cancelled"
    declined = store.get(row.id)
    assert declined.state == RELEASE_DEPLOY_DECLINED and declined.deployable
    assert declined.deploy_job_id is None and ops.dispatches == []


def test_complete_build_requires_separate_approval_and_dispatches_one_child(rig):
    import json

    from llm_bawt.approval_models import REQ_APPROVED
    from llm_bawt.approval_policies import ToolApprovalPolicyStore
    from llm_bawt.ops.image_deploy import DEPLOY_SCHEMA
    from llm_bawt.ops.models import JOB_SUCCEEDED

    coordinator, store, gateway, advance = rig
    approvals = ToolApprovalPolicyStore(None, engine=store.engine)
    published = []
    parent = SimpleNamespace(caller_bot_id="test-bot", caller_user_id="test-user",
                             caller_turn_id="turn-1", caller_session_key="test-bot:test-user",
                             caller_backend="claude-code")
    old_image, new_image = "sha256:" + "a" * 64, "sha256:" + "b" * 64
    child = SimpleNamespace(id="c" * 32, state=JOB_SUCCEEDED, exit_code=0,
                            dispatched_at=datetime.now(timezone.utc), error_text=None,
                            output_tail=None)

    class FakeOps:
        def __init__(self):
            self.store = self
            self.keys = {}
            self.dispatches = []

        def get_job(self, job_id):
            return parent if job_id == "a" * 32 else child if job_id == child.id else None

        def get_job_by_key(self, key):
            return self.keys.get(key)

        def prepare_invocation(self, slug, args):
            assert args == {"release_run_id": store.get_by_parent_job("a" * 32).id}
            binding = store.verified_binding(args["release_run_id"])
            return {"operation": {"slug": slug}, "input_args": args,
                    "execution": {"executor_kind": "docker"},
                    "spec": {"container_name": "bawthub-frontend-prod-1"},
                    "release": {**binding, "expected_current_image_id": old_image}}

        def dispatch_job(self, **kwargs):
            self.dispatches.append(kwargs)
            self.keys[kwargs["idempotency_key"]] = child
            return {"id": child.id}

        def get_job_status(self, *args, **kwargs):
            return {"state": JOB_SUCCEEDED}

        def _resolve_executor(self, kind):
            return SimpleNamespace(inspect_target_image=lambda spec: new_image,
                                   inspect_target_health=lambda spec: "healthy")

    ops = FakeOps()
    coordinator.ops = ops
    coordinator._approvals = lambda: approvals
    coordinator._publisher = published.append
    row = _start(coordinator, store, gateway, advance)
    gateway.runs = [_run(conclusion="success")]
    gateway.receipts[1] = _receipt(row, attempt=1, status="complete")
    advance()
    result = coordinator.pump(row.parent_job_id, SNAPSHOT)
    assert result.state == "running"
    waiting = store.get(row.id)
    assert waiting.state == RELEASE_AWAITING_DEPLOY_APPROVAL
    assert approvals.get_request(waiting.deploy_approval_request_id).status == "pending"
    approval_cards = [event for event in published if event["_type"] == "tool_approval_required"]
    assert len(approval_cards) == 1 and approval_cards[0]["continuation_capable"] is False
    assert any(event["_type"] == "ops_release_transition" for event in published)
    assert ops.dispatches == []
    approvals.resolve_request(waiting.deploy_approval_request_id, status=REQ_APPROVED,
                              resolved_by="test-operator", continuation_owner="none")
    child.output_tail = json.dumps({
        "schema": DEPLOY_SCHEMA, "phase": "done", "action": "deploy",
        "deployed": {"image_id": new_image, "image_ref": f"{SPEC['image_repository']}@{DIGEST}",
                     "release": {"version": "0.1.63", "source_sha": SOURCE,
                                 "workflow_run_id": "123456"}},
        "health": {"status": "healthy", "release": {"version": "0.1.63",
                    "sourceSha": SOURCE, "workflowRunId": "123456"}},
    })
    advance()
    result = coordinator.pump(row.parent_job_id, SNAPSHOT)
    assert result.state == JOB_SUCCEEDED
    assert len(ops.dispatches) == 1
    assert ops.dispatches[0]["approval_request_id"] == waiting.deploy_approval_request_id
    assert coordinator.pump(row.parent_job_id, SNAPSHOT).state == JOB_SUCCEEDED
    assert len(ops.dispatches) == len(gateway.dispatches) == 1
