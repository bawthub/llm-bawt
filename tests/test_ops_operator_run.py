"""TASK-1947: authenticated manual Run is exact, durable whole-job intent.

All operations use disposable SQLite and fake GitHub/Docker. No live actions.
"""
from datetime import datetime, timedelta, timezone
import json
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, create_engine

from llm_bawt.approval_models import REQ_DENIED, ToolApprovalRequest
from llm_bawt.approval_policies import ToolApprovalPolicyStore
from llm_bawt.bawthub_auth import TokenCheck
from llm_bawt.ops.operator_run import OPERATOR_BACKEND, dispatch_operator_run
from llm_bawt.ops.release_coordinator import ReleaseCoordinator
from llm_bawt.ops.release_executor import ReleaseExecutor
from llm_bawt.ops.release_store import ReleaseStore
from tests.ops_install_rows import SEEDS
from llm_bawt.ops.service import OpsDispatchError, OpsService
from llm_bawt.ops.store import OpsStore
from llm_bawt.service.routes import ops as routes, service_auth
from tests.test_release_dry_run import BASE, OLD, NEW, FakeDocker, FakeGitHub, FakeVerifier

USER = "fixture-person"
ARGS = {"release_task": "TASK-1947"}
SLUG = "bawthub.release-prod"


@pytest.fixture
def rig():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    jobs = OpsStore(None, engine=engine)
    releases = ReleaseStore(None, engine=engine)
    approvals = ToolApprovalPolicyStore(None, engine=engine)
    for slug in (SLUG, "bawthub.deploy-prod-image"):
        jobs.create_operation({**next(row for row in SEEDS if row["slug"] == slug), "enabled": True})
    docker, github = FakeDocker(), FakeGitHub(releases)
    ops = OpsService(jobs, executor=docker, release_verifier=FakeVerifier(), release_store=releases)
    now = [datetime.now(timezone.utc)]
    coordinator = ReleaseCoordinator(releases=releases, ops=ops, gateway=github,
                                     approvals=lambda: approvals, clock=lambda: now[0])
    ops.register_executor(ReleaseExecutor(coordinator))

    def run(*, user=USER, args=None, key="fixture-run"):
        return dispatch_operator_run(ops, approvals, user_id=user, operation=SLUG,
                                     args=ARGS if args is None else args, idempotency_key=key)

    def finish(parent):
        for _ in range(10):
            now[0] += timedelta(seconds=30)
            result = ops.get_job_status(parent["id"])
            if result["terminal"]:
                return result
        raise AssertionError("synthetic release did not terminate")

    return SimpleNamespace(ops=ops, jobs=jobs, releases=releases, approvals=approvals,
                           docker=docker, github=github, now=now, run=run, finish=finish,
                           coordinator=coordinator)


def test_manual_run_has_one_audited_authority_and_one_verified_deploy(rig):
    parent = rig.run()
    job = rig.jobs.get_job(parent["id"])
    approval = rig.approvals.get_request(job.approval_request_id)
    snapshot = json.loads(job.invocation_snapshot_json)
    assert snapshot["release_source"]["expected_sha"] == BASE
    assert snapshot["release_source"]["expected_current_image_id"] == OLD
    assert (job.caller_backend, job.caller_user_id, job.caller_bot_id) == (OPERATOR_BACKEND, USER, None)
    assert (approval.status, approval.resolved_by, approval.resolved_via) == ("approved", USER, "user")
    assert approval.operations_snapshot_json and json.loads(approval.operations_snapshot_json) == snapshot
    assert rig.finish(parent)["state"] == "succeeded"
    release = rig.releases.get(parent["release_id"])
    assert release.state == "deployed" and release.deploy_approval_request_id is None
    child = rig.jobs.get_job(release.deploy_job_id)
    assert child.approval_request_id == approval.id
    assert json.loads(child.args_json) == {"release_run_id": release.id}
    assert rig.docker.current == NEW
    assert len(rig.github.dispatches) == len(rig.docker.dispatches) == 1
    # Same operator key is a read, even after completion and process recreation.
    rig.ops = OpsService(rig.jobs, executor=rig.docker, release_verifier=FakeVerifier(), release_store=rig.releases)
    assert dispatch_operator_run(rig.ops, rig.approvals, user_id=USER, operation=SLUG,
                                 args=ARGS, idempotency_key="fixture-run")["id"] == parent["id"]
    assert len(rig.github.dispatches) == len(rig.docker.dispatches) == 1


def test_resume_after_intent_commit_uses_saved_snapshot_not_new_source(rig, monkeypatch):
    dispatch = rig.ops.dispatch_job
    monkeypatch.setattr(rig.ops, "dispatch_job", lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("crash")))
    with pytest.raises(RuntimeError, match="crash"):
        rig.run()
    monkeypatch.setattr(rig.ops, "dispatch_job", dispatch)
    prepare = rig.ops.prepare_invocation
    monkeypatch.setattr(rig.ops, "prepare_invocation", lambda *_args: pytest.fail("retry must not re-prepare"))
    parent = rig.run()
    assert rig.run()["id"] == parent["id"]
    # Admission persists preflight first; the scheduler has not dispatched yet.
    assert rig.github.dispatches == []
    monkeypatch.setattr(rig.ops, "prepare_invocation", prepare)
    assert rig.finish(parent)["state"] == "succeeded"
    assert len(rig.github.dispatches) == len(rig.docker.dispatches) == 1


def test_same_key_with_changed_scope_is_refused(rig):
    parent = rig.run()
    with pytest.raises(OpsDispatchError, match="different arguments") as error:
        rig.run(args={"release_task": "TASK-9999"})
    assert error.value.code == "idempotency_conflict"
    assert rig.run()["id"] == parent["id"] and rig.github.dispatches == []
    assert rig.finish(parent)["state"] == "succeeded"
    assert len(rig.github.dispatches) == len(rig.docker.dispatches) == 1


def test_user_keys_are_isolated(rig):
    first, second = rig.run(), rig.run(user="other-person")
    assert first["id"] != second["id"]
    assert rig.jobs.get_job(first["id"]).caller_user_id == USER
    assert rig.jobs.get_job(second["id"]).caller_user_id == "other-person"


@pytest.mark.parametrize("change,expected", [
    ("baseline", "production_baseline_moved"),
    ("operation", "deploy_operation_changed"),
    ("authority", "release_authority_invalid"),
    ("source", "approved_source_moved"),
])
def test_manual_run_cannot_broaden_saved_authority(rig, monkeypatch, change, expected):
    if change == "source":
        # Crash after the authorization persisted, before the release is created.
        dispatch = rig.ops.dispatch_job
        monkeypatch.setattr(rig.ops, "dispatch_job", lambda **_kw: (_ for _ in ()).throw(RuntimeError("crash")))
        with pytest.raises(RuntimeError):
            rig.run()
        monkeypatch.setattr(rig.ops, "dispatch_job", dispatch)
        monkeypatch.setattr(rig.github, "resolve_branch_head", lambda *_args: "4" * 40)
    parent = rig.run()
    if change == "baseline":
        rig.docker.current = "sha256:" + "c" * 64
    elif change == "operation":
        rig.jobs.update_operation("bawthub.deploy-prod-image", {"title": "different revision"})
    elif change == "authority":
        with Session(rig.jobs.engine) as session:
            row = session.get(ToolApprovalRequest, rig.jobs.get_job(parent["id"]).approval_request_id)
            row.resolved_by = "other-person"
            session.add(row)
            session.commit()
    if not parent["terminal"]:
        rig.finish(parent)
    release = rig.releases.get(parent["release_id"])
    assert release.error_code == expected
    assert rig.docker.dispatches == []
    assert len(rig.github.dispatches) == (0 if change == "source" else 1)


def test_existing_denied_intent_is_not_reinterpreted(rig, monkeypatch):
    dispatch = rig.ops.dispatch_job
    monkeypatch.setattr(rig.ops, "dispatch_job", lambda **_kw: (_ for _ in ()).throw(RuntimeError("crash")))
    with pytest.raises(RuntimeError):
        rig.run()
    row = rig.approvals.list_requests()[0]
    with Session(rig.jobs.engine) as session:
        current = session.get(ToolApprovalRequest, row.id)
        current.status = REQ_DENIED
        session.add(current)
        session.commit()
    monkeypatch.setattr(rig.ops, "dispatch_job", dispatch)
    with pytest.raises(OpsDispatchError) as error:
        rig.run()
    assert error.value.code == "approval_required" and rig.github.dispatches == []


@pytest.fixture
def client(rig, monkeypatch):
    from llm_bawt.service.routes import approval_policies

    monkeypatch.setattr(routes, "_service", lambda: rig.ops)
    monkeypatch.setattr(approval_policies, "_store", lambda: rig.approvals)
    monkeypatch.setattr(service_auth, "service_token_source", lambda: SimpleNamespace(
        check=lambda value: TokenCheck.OK if value == "Bearer synthetic-service-token" else TokenCheck.INVALID))
    app = FastAPI()
    app.include_router(routes.router)
    return TestClient(app)


def _body(**extra):
    return {"operation": SLUG, "args": ARGS, "idempotency_key": "http-key",
            "operator_user_id": USER, "operator_run": True, **extra}


def test_service_authenticated_endpoint_dispatches_whole_job(client, rig):
    response = client.post("/v1/ops/operator-jobs", headers={"Authorization": "Bearer synthetic-service-token"}, json=_body())
    assert response.status_code == 202, response.text
    assert rig.finish(response.json())["state"] == "succeeded"
    assert len(rig.github.dispatches) == len(rig.docker.dispatches) == 1


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer forged"}])
def test_untrusted_requests_refused_before_parsing(client, rig, headers):
    response = client.post("/v1/ops/operator-jobs", headers=headers, content="not json")
    assert response.status_code == 401 and rig.github.dispatches == []
    assert response.headers["cache-control"] == "no-store"


@pytest.mark.parametrize("extra", [{"actor": USER}, {"caller_backend": OPERATOR_BACKEND},
                                    {"snapshot": {}}, {"operator_run": False}, {"operator_user_id": ""}])
def test_caller_cannot_supply_authority_or_empty_identity(client, rig, extra):
    response = client.post("/v1/ops/operator-jobs", headers={"Authorization": "Bearer synthetic-service-token"}, json=_body(**extra))
    assert response.status_code == 422 and rig.github.dispatches == []


def test_missing_operator_marker_is_refused(client, rig):
    body = _body()
    body.pop("operator_run")
    response = client.post("/v1/ops/operator-jobs", headers={"Authorization": "Bearer synthetic-service-token"}, json=body)
    assert response.status_code == 422 and rig.github.dispatches == []


def test_manual_run_preserves_approval_card_only_preparation_fence(rig, monkeypatch):
    from llm_bawt.ops import operator_run
    monkeypatch.setattr(operator_run, "_spec_action", lambda _op: next(iter(operator_run.APPROVAL_ONLY_ACTIONS)))
    monkeypatch.setattr(rig.ops, "prepare_invocation", lambda *_args: pytest.fail("must refuse before source check"))
    with pytest.raises(OpsDispatchError) as error:
        rig.run()
    assert error.value.code == "approval_required"
    assert rig.approvals.list_requests() == [] and rig.github.dispatches == []


@pytest.mark.parametrize("slug", ["bawthub.stop-moshi-tts", "bawthub.start-moshi-stt", "bawthub.restart-tts"])
def test_manual_run_records_no_doomed_gpu_handoff_run(rig, slug):
    """TASK-1707: only a GPU handoff can run it; refuse before any audited Run."""
    rig.jobs.create_operation({**next(row for row in SEEDS if row["slug"] == slug), "enabled": True})
    with pytest.raises(OpsDispatchError) as error:
        dispatch_operator_run(rig.ops, rig.approvals, user_id=USER, operation=slug, args={},
                              idempotency_key="fixture-moshi")
    assert error.value.code == "gpu_handoff_required"
    assert rig.approvals.list_requests() == [] and rig.jobs.list_jobs() == []


def test_legacy_http_cannot_assert_new_operator_authority(client, rig):
    response = client.post("/v1/ops/jobs", json=_body())
    assert response.status_code == 422 and rig.github.dispatches == []


def test_live_ledger_projects_actual_waiting_release_not_parent_running(client, rig):
    parent = rig.run()
    release = rig.releases.get(parent["release_id"])
    from llm_bawt.ops.release_models import ReleaseRun
    with Session(rig.jobs.engine) as session:
        row = session.get(ReleaseRun, release.id)
        row.state = "awaiting_deploy_approval"
        row.error_code = "fixture_wait"
        row.error_text = "pending exact approval"
        session.add(row)
        session.commit()
    # Pure read projection; do not reconcile or execute.
    from llm_bawt.ops.release_status import attach_release_status
    view = attach_release_status(rig.jobs.engine, [rig.jobs.get_job(parent["id"]).to_api()])[0]
    assert view["release"] == {"id": release.id, "state": "awaiting_deploy_approval", "terminal": False,
                               "error_code": "fixture_wait", "error_text": "pending exact approval"}
    ordinary = {"id": "ordinary", "state": "running", "host_unit_name": "worker"}
    assert attach_release_status(None, [ordinary]) == [ordinary]
