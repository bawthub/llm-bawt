"""TASK-1030: one-command release exercise; no GitHub, Docker or production I/O."""
from datetime import datetime, timedelta, timezone
import json

import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, create_engine

from llm_bawt.approval_models import REQ_APPROVED, ToolApprovalRequest
from llm_bawt.approval_policies import ToolApprovalPolicyStore
from llm_bawt.ops.executor import DispatchResult, Executor, ReconcileResult
from llm_bawt.ops.image_deploy import DEPLOY_SCHEMA
from llm_bawt.ops.release import ReleaseVerifier
from llm_bawt.ops.release_executor import ReleaseExecutor
from llm_bawt.ops.release_coordinator import ReleaseCoordinator
from llm_bawt.ops.release_store import ReleaseStore
from llm_bawt.ops.seeds import SEEDS
from llm_bawt.ops.service import OpsService
from llm_bawt.ops.store import OpsStore

BASE, SOURCE = "2" * 40, "1" * 40
DIGEST = "sha256:" + "d" * 64
OLD, NEW = "sha256:" + "a" * 64, "sha256:" + "b" * 64
IMAGE_REPO = "ghcr.io/bawthub/frontend"


class FakeGitHub:
    def __init__(self, releases):
        self.releases = releases
        self.dispatches = []
        self.run_visible = False

    def resolve_branch_head(self, repo, branch):
        assert repo == "bawthub/bawthub", f"release must not consult {repo}"
        return BASE

    def dispatch_workflow(self, repo, workflow, branch, inputs):
        self.dispatches.append(dict(inputs))
        self.run_visible = True

    def find_correlated_runs(self, repo, workflow, branch, request_id, *, created_after):
        assert request_id == self.dispatches[0]["release_request_id"]
        return ([{"id": 123456, "run_attempt": 1, "html_url": "https://github.com/bawthub/bawthub/actions/runs/123456"}]
                if self.run_visible else [])

    def get_run(self, repo, run_id):
        return {"run_attempt": 1, "status": "completed", "conclusion": "success"}

    def download_release_receipt(self, repo, run_id, attempt):
        row = self.releases.get_by_request_id(self.dispatches[0]["release_request_id"])
        return {"schema": "bawthub.release-receipt/v1", "status": "complete",
                "repository": repo, "workflow_run_id": str(run_id),
                "workflow_run_attempt": str(attempt), "release_request_id": row.release_request_id,
                "base_sha": BASE, "source_sha": SOURCE, "version": "0.1.63", "tag": "v0.1.63",
                "digest": DIGEST, "image_repository": IMAGE_REPO,
                "image_ref": f"{IMAGE_REPO}@{DIGEST}"}

    def rerun_failed_jobs(self, repo, run_id):
        raise AssertionError("successful release must never rerun")


class FakeVerifier(ReleaseVerifier):
    def verify(self, spec, args):
        return {**args, "workflow_run_attempt": "1", "trigger_sha": BASE,
                "tag": "v0.1.63", "image_repository": IMAGE_REPO,
                "image_ref": f"{IMAGE_REPO}@{DIGEST}"}


class FakeDocker(Executor):
    def __init__(self):
        self.dispatches = []
        self.current = OLD

    def kind(self):
        return "docker"

    def available(self):
        return True

    def check_target(self, spec, args):
        pass

    def inspect_target_image(self, spec):
        return self.current

    def inspect_target_health(self, spec):
        return "healthy"

    def dispatch(self, **kwargs):
        self.dispatches.append(kwargs)
        return DispatchResult(host_unit_name="fake-worker", status_file_path="fake-receipt")

    def reconcile(self, **kwargs):
        self.current = NEW
        record = {"schema": DEPLOY_SCHEMA, "phase": "done", "action": "deploy",
                  "deployed": {"image_id": NEW, "image_ref": f"{IMAGE_REPO}@{DIGEST}",
                               "release": {"version": "0.1.63", "source_sha": SOURCE,
                                           "workflow_run_id": "123456"}},
                  "health": {"status": "healthy", "release": {"version": "0.1.63",
                              "sourceSha": SOURCE, "workflowRunId": "123456"}}}
        return ReconcileResult("succeeded", 0, json.dumps(record))


@pytest.mark.parametrize("caller_backend", ["claude-code", "http-operator"])
def test_one_approval_reconciles_build_and_exactly_one_deploy_without_external_io(caller_backend):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    store = OpsStore(None, engine=engine)
    releases = ReleaseStore(None, engine=engine)
    approvals = ToolApprovalPolicyStore(None, engine=engine)
    for slug in ("bawthub.release-prod", "bawthub.deploy-prod-image"):
        seed = next(row for row in SEEDS if row["slug"] == slug)
        store.create_operation({**seed, "enabled": True})
    docker, github = FakeDocker(), FakeGitHub(releases)
    ops = OpsService(store, executor=docker, release_verifier=FakeVerifier(), release_store=releases)
    time = [datetime.now(timezone.utc)]
    cards = []
    coordinator = ReleaseCoordinator(releases=releases, ops=ops, gateway=github,
                                     approvals=lambda: approvals, publisher=cards.append,
                                     clock=lambda: time[0])
    ops.register_executor(ReleaseExecutor(coordinator))
    # Mirrors trusted MCP approval preparation. No remote side effect yet.
    snapshot = ops.prepare_invocation("bawthub.release-prod", {"release_task": "TASK-1030"})
    source = snapshot["release_source"]
    assert source["expected_sha"] == BASE
    assert source["expected_current_image_id"] == OLD
    assert source["deploy_authorization"] == "after_verified_build"
    assert github.dispatches == []
    approval_id = "approved-release" if caller_backend == "claude-code" else None
    if approval_id:
        with Session(engine) as session:
            session.add(ToolApprovalRequest(
                id=approval_id, bot_id="test-bot", user_id="test-user", turn_id="turn-1",
                backend=caller_backend, tool_name="ops_run", request_kind="mcp", status=REQ_APPROVED,
                subject="operation=bawthub.release-prod", grant_key="approval",
                tool_arguments_json=json.dumps({"operation": "bawthub.release-prod", "args": {"release_task": "TASK-1030"}}),
                operations_snapshot_json=json.dumps(snapshot),
            ))
            session.commit()
    parent = ops.dispatch_job(operation_slug="bawthub.release-prod", args={"release_task": "TASK-1030"},
                              approved_snapshot=snapshot, idempotency_key="TASK-1030:dry-run",
                              caller_actor="test-operator" if caller_backend == "http-operator" else None,
                              caller_bot_id="test-bot", caller_user_id="test-user",
                              caller_turn_id="turn-1", caller_backend=caller_backend,
                              approval_request_id=approval_id)
    assert parent["release_id"] and parent["state"] == "accepted"
    assert ops.dispatch_job(operation_slug="bawthub.release-prod", args={"release_task": "TASK-1030"},
                            approved_snapshot=snapshot, idempotency_key="TASK-1030:dry-run")["release_id"] == parent["release_id"]
    for _ in range(10):
        time[0] += timedelta(seconds=30)
        result = ops.get_job_status(parent["id"])
        if result["terminal"] or releases.get(parent["release_id"]).deploy_approval_request_id:
            break
    row = releases.get(parent["release_id"])
    assert len(github.dispatches) == 1
    if caller_backend == "http-operator":
        assert row.state == "awaiting_deploy_approval" and docker.dispatches == []
        assert len([card for card in cards if card["_type"] == "tool_approval_required"]) == 1
        request = approvals.get_request(row.deploy_approval_request_id)
        approved = json.loads(request.operations_snapshot_json)
        approvals.resolve_request(request.id, status=REQ_APPROVED, resolved_by="operator",
                                  continuation_owner="none")
        for _ in range(5):
            time[0] += timedelta(seconds=30)
            result = ops.get_job_status(parent["id"])
            if result["terminal"]:
                break
        child_approval_id = request.id
    else:
        assert not [card for card in cards if card["_type"] == "tool_approval_required"]
        assert row.deploy_approval_request_id is None
        approved = json.loads(row.release_plan_json)["deploy_snapshot"]
        child_approval_id = approval_id
    row = releases.get(parent["release_id"])
    assert row.state == "deployed"
    assert any(card["_type"] == "ops_release_transition" for card in cards)
    assert len(docker.dispatches) == 1
    assert approved["input_args"] == {"release_run_id": row.id}
    assert result["state"] == "succeeded"
    assert releases.get(row.id).state == "deployed"
    assert len(github.dispatches) == len(docker.dispatches) == 1
    assert docker.current == NEW
    child = store.get_job(releases.get(row.id).deploy_job_id)
    assert json.loads(child.args_json) == {"release_run_id": row.id}
    assert child.approval_request_id == child_approval_id
    without_health = json.loads(child.output_tail)
    without_health.pop("health")
    assert "worker did not attest healthy production" in coordinator._deployment_problem(
        releases.get(row.id), without_health, approved)
    from llm_bawt.service.routes import ops as routes
    original = routes._service
    try:
        routes._service = lambda: ops
        identity = routes.get_release(row.id)["production_identity"]
        assert identity == {"version": "0.1.63", "source_sha": SOURCE, "digest": DIGEST,
                            "workflow_run_id": "123456", "verified_at": releases.get(row.id).to_api()["finished_at"]}
    finally:
        routes._service = original


@pytest.mark.parametrize("change, expected", [
    ("baseline", "production_baseline_moved"),
    ("operation", "deploy_operation_changed"),
    ("missing_approval", "release_authority_invalid"),
    ("tampered_approval", "release_authority_invalid"),
])
def test_conditional_deploy_refuses_changed_or_unapproved_release(change, expected):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    store = OpsStore(None, engine=engine)
    releases = ReleaseStore(None, engine=engine)
    approvals = ToolApprovalPolicyStore(None, engine=engine)
    for slug in ("bawthub.release-prod", "bawthub.deploy-prod-image"):
        store.create_operation({**next(row for row in SEEDS if row["slug"] == slug), "enabled": True})
    docker, github = FakeDocker(), FakeGitHub(releases)
    ops = OpsService(store, executor=docker, release_verifier=FakeVerifier(), release_store=releases)
    time = [datetime.now(timezone.utc)]
    coordinator = ReleaseCoordinator(releases=releases, ops=ops, gateway=github,
                                     approvals=lambda: approvals, clock=lambda: time[0])
    ops.register_executor(ReleaseExecutor(coordinator))
    snapshot = ops.prepare_invocation("bawthub.release-prod", {"release_task": "TASK-1030"})
    approval_id = None if change == "missing_approval" else f"approved-{change}"
    if approval_id:
        with Session(engine) as session:
            session.add(ToolApprovalRequest(
                id=approval_id, bot_id="test-bot", user_id="test-user", turn_id="turn-1",
                backend="claude-code", tool_name="ops_run", request_kind="mcp", status=REQ_APPROVED,
                subject="operation=bawthub.release-prod", grant_key="approval",
                tool_arguments_json=json.dumps({"operation": "bawthub.release-prod", "args": {"release_task": "TASK-1030"}}),
                operations_snapshot_json=json.dumps(snapshot),
            ))
            session.commit()
    parent = ops.dispatch_job(operation_slug="bawthub.release-prod", args={"release_task": "TASK-1030"},
                              approved_snapshot=snapshot, idempotency_key=f"test-{change}",
                              caller_backend="claude-code", caller_bot_id="test-bot",
                              caller_user_id="test-user", caller_turn_id="turn-1",
                              approval_request_id=approval_id)
    if change == "baseline":
        docker.current = "sha256:" + "c" * 64
    elif change == "operation":
        store.update_operation("bawthub.deploy-prod-image", {"title": "Changed deploy revision"})
    elif change == "tampered_approval":
        with Session(engine) as session:
            request = session.get(ToolApprovalRequest, approval_id)
            request.tool_arguments_json = json.dumps({"operation": "bawthub.deploy-prod-image", "args": {}})
            session.add(request)
            session.commit()
    for _ in range(5):
        time[0] += timedelta(seconds=30)
        result = ops.get_job_status(parent["id"])
        if result["terminal"]:
            break
    row = releases.get(parent["release_id"])
    assert row.state == "deploy_declined" and row.error_code == expected
    assert len(github.dispatches) == 1 and docker.dispatches == []
    assert row.deploy_job_id is None and row.deploy_approval_request_id is None
