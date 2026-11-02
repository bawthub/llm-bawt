"""TASK-1947: exact-ID recovery decides only an existing release deploy card.

Source-only fixtures: isolated SQLite, fake tag snapshots and signed turns,
real snapshot verification and audited approval CAS; no network or ops runner.
"""
from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from llm_bawt.approval_models import (
    CONT_NOT_NEEDED, REQ_APPROVED, REQ_DENIED, REQ_EXPIRED,
    REQ_PENDING, RESOLVED_VIA_BOT_GRANT, ToolApprovalRequest,
)
from llm_bawt.approval_policies import ToolApprovalPolicyStore
from llm_bawt.bot_tags import BehaviorTag
from llm_bawt.bot_tags.facade import TagSnapshot
from llm_bawt.ops.models import JOB_RUNNING, OpsJob
from llm_bawt.ops.release_models import ReleaseRun
from llm_bawt.service.approver_eligibility import approver_eligible
from llm_bawt.service.routes import ops_approvals as routes
from llm_bawt.task_turn_context import TASK_TURN_CONTEXT_HEADER, TaskTurnContextError
from tests.test_ops_image_deploy import (
    RELEASE_ID, SEED, TRIGGER, _rehash, _service,
)

HOLDER, REQUESTER, USER = "fixture-gavel", "fixture-vex", "fixture-user"
PARENT = "0" * 32


def _forbidden(*_args, **_kwargs):
    raise AssertionError("approval recovery must not prepare, execute, grant or resume")


@pytest.fixture
def recovery(monkeypatch):
    ops, jobs, executor = _service()
    store = ToolApprovalPolicyStore(None, engine=jobs.engine)
    events = []
    store.events = SimpleNamespace(append_at_commit=lambda conn, **event: events.append(event))
    jobs.create_operation({**SEED["bawthub.release-prod"], "enabled": True})
    # Build a real legacy build-only snapshot using fake target/source reads.
    monkeypatch.setattr(ops, "_resolve_executor", lambda _kind: executor)
    monkeypatch.setattr(executor, "approval_source", lambda _spec, _args: {"expected_sha": TRIGGER}, raising=False)
    parent_snapshot = ops.prepare_invocation("bawthub.release-prod", {"release_task": "TASK-1030"})
    release_id = hashlib.sha256(f"bawthub-release:{PARENT}".encode()).hexdigest()[:32]
    request_id = f"release-deploy-{release_id}"
    with Session(jobs.engine) as session:
        old = session.get(ReleaseRun, RELEASE_ID)
        fields = old.model_dump()
        session.delete(old)
        session.flush()
        fields.update(id=release_id, deploy_approval_request_id=request_id,
                      phase_deadline_at=datetime.now(timezone.utc) + timedelta(hours=1),
                      release_plan_json=json.dumps({"deploy_operation": "bawthub.deploy-prod-image",
                                                    "image_repository": fields["image_repository"],
                                                    "expected_sha": TRIGGER}))
        session.add(ReleaseRun(**fields))
        operation = parent_snapshot["operation"]
        parent = OpsJob(id=PARENT, operation_slug=operation["slug"], operation_version=operation["version"],
                        operation_script_hash=operation["script_hash"], args_json=json.dumps(parent_snapshot["resolved_args"]),
                        invocation_snapshot_json=json.dumps(parent_snapshot), state=JOB_RUNNING,
                        idempotency_key="fixture-release-existing", caller_bot_id=None, caller_user_id=USER,
                        caller_backend="http-operator")
        session.add(parent)
        session.commit()
    snapshot = ops.prepare_invocation("bawthub.deploy-prod-image", {"release_run_id": release_id})
    row, created = store.record_orchestration_request(
        request_id=request_id, bot_id="ops", user_id=USER, turn_id=f"release:{release_id}",
        backend="http-operator", tool_name="ops_release_deploy",
        tool_arguments={"operation": "bawthub.deploy-prod-image", "args": {"release_run_id": release_id},
                        "release_run_id": release_id}, subject="existing verified deploy", prompt="Deploy it?",
        grant_key=hashlib.sha256(f"release-deploy:{release_id}".encode()).hexdigest(), severity="high",
        operations_snapshot=snapshot,
        caller_context={"release_run_id": release_id, "parent_job_id": PARENT, "release_task": "TASK-1030"})
    assert created
    events.clear()
    state = SimpleNamespace(holds=True, tags=TagSnapshot(tags={HOLDER: frozenset({BehaviorTag.APPROVE_OPS}),
                                                             REQUESTER: frozenset()}))

    def holder(bot):
        return bool(state.holds and bot == HOLDER), state.tags

    def open_turn(token):
        if not token.startswith("turn:"):
            raise TaskTurnContextError("bad fixture turn")
        bot = token.removeprefix("turn:")
        return SimpleNamespace(bot_id=bot, user_id=USER, turn_id=f"turn-of-{bot}")

    monkeypatch.setattr(routes, "_store", lambda: store)
    monkeypatch.setattr(routes, "_release_ops", lambda: ops)
    monkeypatch.setattr(routes, "_holder_and_tags", holder)
    monkeypatch.setattr(routes, "holder_and_tags", lambda _conn, bot: holder(bot))
    monkeypatch.setattr(routes, "open_task_turn_context", open_turn)
    monkeypatch.setattr(routes, "resolve_mcp_card", _forbidden)
    monkeypatch.setattr(ops, "prepare_invocation", _forbidden)
    monkeypatch.setattr(ops, "dispatch_job", _forbidden)
    monkeypatch.setattr(ops, "get_job_status", _forbidden)
    monkeypatch.setattr(ops, "_resolve_executor", _forbidden)
    monkeypatch.setattr(store, "claim_client_grant", _forbidden)
    app = FastAPI()
    app.include_router(routes.router)
    client = TestClient(app)
    headers = {TASK_TURN_CONTEXT_HEADER: f"turn:{HOLDER}"}

    def get(*, bot=HOLDER, headers=None):
        return client.get(f"/v1/ops-approvals/{request_id}", headers=headers if headers is not None else {
            TASK_TURN_CONTEXT_HEADER: f"turn:{bot}"})

    def decide(decision="approve", reason="explicit Gavel recovery decision", *, bot=HOLDER, headers=None):
        return client.post(f"/v1/ops-approvals/{request_id}/resolve", json={"decision": decision, "reason": reason},
                           headers=headers if headers is not None else {TASK_TURN_CONTEXT_HEADER: f"turn:{bot}"})

    def change(model, key, **values):
        with Session(jobs.engine) as session:
            current = session.get(model, key)
            for name, value in values.items():
                setattr(current, name, value)
            session.add(current)
            session.commit()

    def bot_requester(bot=REQUESTER):
        change(OpsJob, PARENT, caller_bot_id=bot, caller_turn_id="turn-requester",
               caller_backend="claude-code", caller_session_key=f"{bot}:{USER}")
        change(ToolApprovalRequest, request_id, bot_id=bot, turn_id="turn-requester", backend="claude-code",
               session_key=f"{bot}:{USER}")

    return SimpleNamespace(ops=ops, jobs=jobs, store=store, executor=executor, row=row, rid=request_id,
                           release_id=release_id, state=state, events=events, snapshot=snapshot,
                           get=get, decide=decide, change=change, bot_requester=bot_requester,
                           client=client, headers=headers)


def _code(response):
    return response.json()["detail"]["code"]


def test_existing_http_operator_card_recovers_without_new_job_or_tool(recovery):
    ctx = recovery
    before = ctx.store.get_request(ctx.rid)
    immutable = (before.id, before.caller_context_json, before.operations_snapshot_json, before.tool_arguments_json)
    with Session(ctx.jobs.engine) as session:
        jobs_before = [j.id for j in session.exec(select(OpsJob)).all()]
    detail = ctx.get()
    assert detail.status_code == 200, detail.text
    assert detail.json()["requested_by"] == "ops"
    assert detail.json()["status"] == REQ_PENDING
    assert ctx.store.get_request(ctx.rid).status == REQ_PENDING  # no automatic approval
    # Listing and fanout remain MCP-only, including after explicit recovery.
    listed = ctx.client.get("/v1/ops-approvals", headers=ctx.headers)
    assert listed.status_code == 200 and listed.json()["data"] == []
    result = ctx.decide()
    assert result.status_code == 200, result.text
    final = ctx.store.get_request(ctx.rid)
    assert (final.status, final.resolved_via, final.resolved_by, final.resolved_turn_id) == (
        REQ_APPROVED, RESOLVED_VIA_BOT_GRANT, f"bot:{HOLDER}", f"turn-of-{HOLDER}")
    assert final.resolved_by_bot == HOLDER
    assert (final.continuation_owner, final.grant_state, final.continuation_state) == (
        "none", "not_applicable", CONT_NOT_NEEDED)
    assert (final.id, final.caller_context_json, final.operations_snapshot_json, final.tool_arguments_json) == immutable
    assert result.json()["awaiting_admission"] is False
    with Session(ctx.jobs.engine) as session:
        assert [j.id for j in session.exec(select(OpsJob)).all()] == jobs_before
        release = session.get(ReleaseRun, ctx.release_id)
        assert release.parent_job_id == PARENT and release.deploy_job_id is None
        assert release.deploy_approval_request_id == ctx.rid
    assert len(ctx.events) == 1 and ctx.events[0]["event_key"] == f"approval:{ctx.rid}:approved"
    assert ctx.get().json()["reason"] == "explicit Gavel recovery decision"


def test_ordinary_bot_release_card_recovers_but_is_not_fanned_out(recovery):
    recovery.bot_requester()
    row = recovery.store.get_request(recovery.rid)
    assert not approver_eligible(holds=True, tags=recovery.state.tags, row=row, approver=HOLDER, user_id=USER)
    assert recovery.get().status_code == 200
    assert recovery.decide().status_code == 200


@pytest.mark.parametrize("kind", ["self", "private", "unknown", "foreign", "no_grant", "tags_unavailable"])
def test_ordinary_bot_authorization_fails_closed(recovery, kind):
    ctx = recovery
    ctx.bot_requester(HOLDER if kind == "self" else REQUESTER)
    expected = 404
    if kind == "private":
        ctx.state.tags = TagSnapshot(tags={HOLDER: frozenset({BehaviorTag.APPROVE_OPS}),
                                         REQUESTER: frozenset({BehaviorTag.PRIVATE})})
    elif kind == "unknown":
        ctx.state.tags = TagSnapshot(tags={HOLDER: frozenset({BehaviorTag.APPROVE_OPS})})
    elif kind == "foreign":
        ctx.change(OpsJob, PARENT, caller_user_id="foreign-user")
        ctx.change(ToolApprovalRequest, ctx.rid, user_id="foreign-user")
    elif kind == "no_grant":
        ctx.state.holds = False
        expected = 403
    elif kind == "tags_unavailable":
        ctx.state.tags = TagSnapshot.unavailable()
    if kind == "self":
        expected = 403
    assert ctx.get().status_code == (403 if kind == "no_grant" else 404)
    result = ctx.decide()
    assert result.status_code == expected, result.text
    assert ctx.store.get_request(ctx.rid).status == REQ_PENDING and ctx.events == []


@pytest.mark.parametrize("headers,code", [({}, "turn_capability_required"),
                                         ({TASK_TURN_CONTEXT_HEADER: "forged"}, "turn_capability_invalid")])
def test_signed_turn_is_still_required(recovery, headers, code):
    for response in (recovery.get(headers=headers), recovery.decide(headers=headers)):
        assert response.status_code == 403 and _code(response) == code
    assert recovery.store.get_request(recovery.rid).status == REQ_PENDING


@pytest.mark.parametrize("forgery", [
    "request_link", "caller_context", "requester", "parent_id", "parent_missing", "parent_inactive",
    "parent_foreign", "parent_snapshot", "snapshot_hash", "snapshot_binding", "snapshot_args",
    "snapshot_target", "tool_arguments", "grant_key", "grant_state", "continuation_owner",
    "receipt", "release_missing", "not_active", "missing_deadline", "elapsed_deadline",
])
def test_forged_or_stale_release_cards_cannot_be_read_or_decided(recovery, forgery):
    ctx = recovery
    if forgery == "request_link":
        ctx.change(ReleaseRun, ctx.release_id, deploy_approval_request_id="foreign-request")
    elif forgery == "caller_context":
        ctx.change(ToolApprovalRequest, ctx.rid, caller_context_json=json.dumps({
            "release_run_id": ctx.release_id, "parent_job_id": "f" * 32, "release_task": "TASK-1030"}))
    elif forgery == "requester":
        ctx.change(ToolApprovalRequest, ctx.rid, bot_id=REQUESTER)
    elif forgery == "parent_id":
        ctx.change(ReleaseRun, ctx.release_id, parent_job_id="f" * 32)
    elif forgery == "parent_missing":
        with Session(ctx.jobs.engine) as session:
            session.delete(session.get(OpsJob, PARENT))
            session.commit()
    elif forgery == "parent_inactive":
        ctx.change(OpsJob, PARENT, state="succeeded")
    elif forgery == "parent_foreign":
        ctx.change(OpsJob, PARENT, operation_slug="foreign.release")
    elif forgery == "parent_snapshot":
        ctx.change(OpsJob, PARENT, invocation_snapshot_json="[]")
    elif forgery.startswith("snapshot_"):
        snapshot = copy.deepcopy(ctx.snapshot)
        if forgery == "snapshot_hash":
            snapshot["snapshot_hash"] = "f" * 64
        elif forgery == "snapshot_binding":
            snapshot["release"]["digest"] = "sha256:" + "e" * 64
            _rehash(snapshot)
        elif forgery == "snapshot_args":
            snapshot["input_args"]["release_run_id"] = "f" * 32
            _rehash(snapshot)
        else:
            snapshot["spec"]["image_repository"] = "ghcr.io/foreign/image"
            _rehash(snapshot)
        ctx.change(ToolApprovalRequest, ctx.rid, operations_snapshot_json=json.dumps(snapshot))
    elif forgery == "tool_arguments":
        ctx.change(ToolApprovalRequest, ctx.rid, tool_arguments_json="{}")
    elif forgery == "grant_key":
        ctx.change(ToolApprovalRequest, ctx.rid, grant_key="forged")
    elif forgery == "grant_state":
        ctx.change(ToolApprovalRequest, ctx.rid, grant_state="pending")
    elif forgery == "continuation_owner":
        ctx.change(ToolApprovalRequest, ctx.rid, continuation_owner="server")
    elif forgery == "receipt":
        ctx.change(ReleaseRun, ctx.release_id, receipt_json="{}")
    elif forgery == "release_missing":
        with Session(ctx.jobs.engine) as session:
            session.delete(session.get(ReleaseRun, ctx.release_id))
            session.commit()
    elif forgery == "not_active":
        ctx.change(ReleaseRun, ctx.release_id, state="deployed")
    elif forgery == "missing_deadline":
        ctx.change(ReleaseRun, ctx.release_id, phase_deadline_at=None)
    else:
        ctx.change(ReleaseRun, ctx.release_id, phase_deadline_at=datetime.now(timezone.utc) - timedelta(seconds=1))
    for response in (ctx.get(), ctx.decide()):
        assert response.status_code == 404 and _code(response) == "not_found", (forgery, response.text)
    assert ctx.store.get_request(ctx.rid).status == REQ_PENDING and ctx.events == []


@pytest.mark.parametrize("kind", ["no_grant", "foreign", "known_private_ops", "tags_unavailable"])
def test_operator_recovery_has_no_general_authorization_exemption(recovery, kind):
    ctx = recovery
    if kind == "no_grant":
        ctx.state.holds = False
    elif kind == "foreign":
        ctx.change(OpsJob, PARENT, caller_user_id="foreign-user")
        ctx.change(ToolApprovalRequest, ctx.rid, user_id="foreign-user")
    elif kind == "known_private_ops":
        ctx.state.tags = TagSnapshot(tags={HOLDER: frozenset({BehaviorTag.APPROVE_OPS}),
                                         "ops": frozenset({BehaviorTag.PRIVATE})})
    else:
        ctx.state.tags = TagSnapshot.unavailable()
    for response in (ctx.get(), ctx.decide()):
        assert response.status_code == (403 if kind == "no_grant" else 404)
    assert ctx.store.get_request(ctx.rid).status == REQ_PENDING and ctx.events == []


@pytest.mark.parametrize("bad_id", ["release-deploy:{id}", "release-deploy-{id}extra", "release-deploy-ABCDEF", "foreign-{id}"])
def test_request_id_must_have_the_exact_deterministic_format(recovery, bad_id):
    ctx = recovery
    forged_id = bad_id.format(id=ctx.release_id)
    ctx.change(ToolApprovalRequest, ctx.rid, id=forged_id)
    ctx.change(ReleaseRun, ctx.release_id, deploy_approval_request_id=forged_id)
    for response in (
        ctx.client.get(f"/v1/ops-approvals/{forged_id}", headers=ctx.headers),
        ctx.client.post(f"/v1/ops-approvals/{forged_id}/resolve", headers=ctx.headers,
                        json={"decision": "approve", "reason": "explicit decision"}),
    ):
        assert response.status_code == 404 and _code(response) == "not_found"
    assert ctx.store.get_request(forged_id).status == REQ_PENDING and ctx.events == []


@pytest.mark.parametrize("field,value", [("request_kind", "mcp"), ("request_kind", "harness"),
                                         ("tool_name", "ops_run"), ("tool_name", "foreign_tool")])
def test_wrong_request_kind_or_tool_never_uses_recovery(recovery, field, value):
    ctx = recovery
    ctx.change(ToolApprovalRequest, ctx.rid, **{field: value})
    for response in (ctx.get(), ctx.decide()):
        assert response.status_code == 404
    assert ctx.store.get_request(ctx.rid).status == REQ_PENDING and ctx.events == []


@pytest.mark.parametrize("status", [REQ_APPROVED, REQ_DENIED, REQ_EXPIRED])
def test_duplicate_or_terminal_decisions_never_replace_first_provenance(recovery, status):
    ctx = recovery
    ctx.store.resolve_request(ctx.rid, status=status, resolved_by="fixture-person",
                              message="first decision", continuation_owner="none")
    ctx.events.clear()
    result = ctx.decide()
    assert result.status_code == 409 and _code(result) == "already_resolved"
    final = ctx.store.get_request(ctx.rid)
    assert (final.status, final.resolved_by, final.resolved_turn_id, final.resolution_message) == (
        status, "fixture-person", None, "first decision")
    assert final.resolved_by_bot is None and ctx.events == []


def test_duplicate_from_same_signed_turn_is_still_first_wins(recovery):
    assert recovery.decide(reason="first").status_code == 200
    response = recovery.decide(decision="deny", reason="second")
    assert response.status_code == 409 and response.json()["detail"]["reason"] == "first"
    assert recovery.store.get_request(recovery.rid).status == REQ_APPROVED
    assert len(recovery.events) == 1


def test_explicit_denial_is_audited_without_any_execution(recovery):
    result = recovery.decide("deny", "do not deploy")
    assert result.status_code == 200 and result.json()["status"] == REQ_DENIED
    final = recovery.store.get_request(recovery.rid)
    assert final.resolved_via == RESOLVED_VIA_BOT_GRANT and final.continuation_owner == "none"
    assert final.resolution_message == "do not deploy"


@pytest.mark.parametrize("change", ["grant", "private", "link", "snapshot", "deadline"])
def test_authority_and_binding_are_rechecked_at_decision_fence(recovery, monkeypatch, change):
    ctx = recovery
    decide = ctx.store.decide_orchestration_request

    def race(*args, **kwargs):
        if change == "grant":
            ctx.state.holds = False
        elif change == "private":
            ctx.bot_requester()
            ctx.state.tags = TagSnapshot(tags={HOLDER: frozenset({BehaviorTag.APPROVE_OPS}),
                                             REQUESTER: frozenset({BehaviorTag.PRIVATE})})
        elif change == "link":
            ctx.change(ReleaseRun, ctx.release_id, deploy_approval_request_id="foreign")
        elif change == "snapshot":
            ctx.change(ToolApprovalRequest, ctx.rid, operations_snapshot_json="{}")
        else:
            ctx.change(ReleaseRun, ctx.release_id,
                       phase_deadline_at=datetime.now(timezone.utc) - timedelta(seconds=1))
        return decide(*args, **kwargs)

    monkeypatch.setattr(ctx.store, "decide_orchestration_request", race)
    response = ctx.decide()
    assert response.status_code == (403 if change == "grant" else 404), response.text
    assert ctx.store.get_request(ctx.rid).status == REQ_PENDING and ctx.events == []


def test_concurrent_person_decision_wins_before_bot_cas(recovery, monkeypatch):
    ctx = recovery
    decide = ctx.store.decide_orchestration_request

    def person_wins(*args, **kwargs):
        ctx.store.resolve_request(ctx.rid, status=REQ_DENIED, resolved_by="fixture-person",
                                  message="person first", continuation_owner="none")
        return decide(*args, **kwargs)

    monkeypatch.setattr(ctx.store, "decide_orchestration_request", person_wins)
    response = ctx.decide()
    assert response.status_code == 409
    assert response.json()["detail"]["resolved_by"] == "fixture-person"
    assert ctx.store.get_request(ctx.rid).resolved_by_bot is None
