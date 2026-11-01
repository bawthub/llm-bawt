"""Owner-scoped browser approvals; all rows/callers are isolated fixtures."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, create_engine

from agent_bridge.mcp_call_context import canonical_invocation_hash
from llm_bawt.approval_policies import (
    EXEC_SUCCEEDED, REQ_APPROVED, REQ_DENIED, ToolApprovalPolicyStore,
    ToolApprovalRequest,
)
from llm_bawt.service.routes import approval_policies as routes
from tests.test_mcp_approval_resolve import _mcp_req

OWNER = "fixture-owner"
OTHER = "fixture-other-user"


@pytest.fixture
def scope(monkeypatch):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    store = ToolApprovalPolicyStore(None, engine=engine)
    events = []

    async def publish(*args, **kwargs):
        events.append((args, kwargs))

    monkeypatch.setattr(routes, "_store", lambda: store)
    monkeypatch.setattr(routes, "_subscriber", lambda: None)
    monkeypatch.setattr(routes, "publish_approval_resolved", publish)
    monkeypatch.setattr(routes, "build_continuation_prompt", lambda *args, **kwargs: "fixture continuation")
    monkeypatch.setattr(routes, "get_turn_log_store", lambda: SimpleNamespace(
        engine=engine, set_approval_status=lambda **kwargs: None))
    monkeypatch.setattr(routes, "ToolCallStore", lambda engine: SimpleNamespace(
        resolve_approval_result=lambda **kwargs: None, mark_approval_status=lambda **kwargs: None))
    app = FastAPI()
    app.include_router(routes.router)
    with TestClient(app) as client:
        yield SimpleNamespace(store=store, client=client, events=events)
    engine.dispose()


def record(store, *, request_id="fixture-request", user=OWNER, bot="fixture-bot", kind="harness"):
    args = {"secret": f"arguments-for-{user}"}
    fields = dict(request_id=request_id, bot_id=bot, user_id=user, turn_id="fixture-turn",
                  backend="claude-code", tool_name="fixture_tool", tool_arguments=args,
                  subject=f"subject-for-{user}", grant_key="fixture-grant", policy_id=None,
                  severity="high", prompt="Fixture approval")
    if kind == "mcp":
        return store.record_mcp_request(**_mcp_req(
            **fields, session_key=f"{bot}:{user}", operations_snapshot=None,
            invocation_hash=canonical_invocation_hash("fixture_tool", args)))
    if kind == "orchestration":
        fields.pop("policy_id")  # Coordinator-owned requests have no policy argument.
        return store.record_orchestration_request(**fields, operations_snapshot={"fixture": True})[0]
    return store.record_request(**fields, session_id="fixture-session")


def test_list_and_count_scope_before_pagination_and_preserve_filters(scope):
    for user in (OWNER, OTHER):
        for bot in ("fixture-bot", "fixture-second-bot"):
            for index in range(3):
                row = record(scope.store, request_id=f"{user}-{bot}-{index}", user=user, bot=bot)
                if index == 0:
                    scope.store.resolve_request(row.id, status=REQ_DENIED)
    page = scope.client.get("/v1/tool-approval-requests", params={
        "user_id": OWNER, "limit": 2, "offset": 1}).json()
    assert page["total"] == 6 and len(page["requests"]) == 2
    assert all(row["user_id"] == OWNER for row in page["requests"])
    assert f"arguments-for-{OTHER}" not in json.dumps(page)
    filters = {"user_id": OWNER, "status": "pending", "bot_id": "fixture-bot"}
    filtered = scope.client.get("/v1/tool-approval-requests", params={**filters, "limit": 1, "offset": 1}).json()
    assert filtered["total"] == 2 and len(filtered["requests"]) == 1
    assert filtered["requests"][0]["user_id"] == OWNER
    assert scope.store.count_requests(**filters) == 2
    assert scope.client.get("/v1/tool-approval-requests", params={**filters, "offset": 99}).json()["total"] == 2
    # Explicit empty scope is not the trusted unscoped read contract.
    assert scope.client.get("/v1/tool-approval-requests?user_id=").json()["total"] == 0
    assert scope.client.get("/v1/tool-approval-requests").json()["total"] == 12


def test_result_rejects_foreign_owner_without_output_or_existence_leak(scope):
    row = record(scope.store, user=OTHER, kind="mcp")
    with Session(scope.store.engine) as session:
        saved = session.get(ToolApprovalRequest, row.id)
        saved.status = REQ_APPROVED
        saved.execution_state = EXEC_SUCCEEDED
        saved.result_json = json.dumps({"secret": "other-owner-output"})
        session.add(saved)
        session.commit()
    path = f"/v1/tool-approval-requests/{row.id}/result"
    own = scope.client.get(path, params={"user_id": OTHER})
    assert own.status_code == 200 and own.json()["result"]["secret"] == "other-owner-output"
    hidden = scope.client.get(path, params={"user_id": OWNER})
    missing = scope.client.get("/v1/tool-approval-requests/absent/result", params={"user_id": OWNER})
    assert hidden.status_code == missing.status_code == 404
    assert hidden.json() == missing.json()
    assert "other-owner-output" not in hidden.text
    assert scope.client.get(path + "?user_id=").status_code == 404
    # Optional unscoped reads remain available to the existing LAN/internal client.
    assert scope.client.get(path).status_code == 200


@pytest.mark.parametrize("kind", ["harness", "mcp", "orchestration"])
@pytest.mark.parametrize("terminal", [False, True])
@pytest.mark.parametrize("decision", ["approve", "deny", "cancel", "respond"])
def test_foreign_resolve_refused_before_any_branch_or_side_effect(scope, monkeypatch, kind, terminal, decision):
    row = record(scope.store, kind=kind)
    if terminal:
        scope.store.resolve_request(row.id, status=REQ_DENIED, message="original decision")
    before = scope.store.get_request(row.id).model_dump()

    def forbidden(*args, **kwargs):
        pytest.fail("foreign owner reached resolution/execution/grant/continuation")

    for name in ("resolve_request", "prepare_harness_continuation", "claim_client_grant"):
        monkeypatch.setattr(scope.store, name, forbidden)
    monkeypatch.setattr(routes, "resolve_mcp_card", forbidden)
    monkeypatch.setattr(routes, "build_continuation_prompt", forbidden)
    response = scope.client.post(f"/v1/chat/approvals/{row.id}/resolve", json={
        "decision": decision, "user_id": OTHER, "resolved_by": OWNER,
        "internal": True, "message": "foreign guidance"})
    assert response.status_code == 403
    assert scope.store.get_request(row.id).model_dump() == before
    assert scope.events == []
    assert f"arguments-for-{OWNER}" not in response.text


@pytest.mark.parametrize("identity", [{}, {"user_id": ""}, {"user_id": OWNER.upper()}, {"user_id": OWNER + " "}])
def test_missing_empty_or_nonexact_owner_never_implies_internal_authority(scope, identity):
    row = record(scope.store, kind="orchestration")
    response = scope.client.post(f"/v1/chat/approvals/{row.id}/resolve", json={
        "decision": "approve", "internal": True, **identity})
    assert response.status_code == 403
    assert scope.store.get_request(row.id).status == "pending" and scope.events == []


@pytest.mark.parametrize("kind", ["harness", "mcp", "orchestration"])
def test_owner_resolves_with_canonical_human_attribution(scope, kind):
    row = record(scope.store, kind=kind)
    response = scope.client.post(f"/v1/chat/approvals/{row.id}/resolve", json={
        "decision": "deny", "user_id": OWNER})
    assert response.status_code == 200, response.text
    saved = scope.store.get_request(row.id)
    assert saved.status == REQ_DENIED and saved.resolved_by == OWNER
    assert saved.resolved_via == "user" and len(scope.events) == 1


def test_owner_cannot_supply_another_human_resolver(scope):
    row = record(scope.store, kind="orchestration")
    response = scope.client.post(f"/v1/chat/approvals/{row.id}/resolve", json={
        "decision": "approve", "user_id": OWNER, "resolved_by": OTHER})
    assert response.status_code == 403
    assert scope.store.get_request(row.id).status == "pending" and scope.events == []


@pytest.mark.parametrize("owner", ["", "unknown", "none", "null", "undefined", "UNKNOWN", " unknown "])
def test_legacy_sentinel_owner_cannot_be_claimed_by_exact_match(scope, owner):
    row = record(scope.store)
    with Session(scope.store.engine) as session:
        saved = session.get(ToolApprovalRequest, row.id)
        saved.user_id = owner
        session.add(saved)
        session.commit()
    response = scope.client.post(f"/v1/chat/approvals/{row.id}/resolve", json={
        "decision": "approve", "user_id": owner})
    assert response.status_code == 403
    assert scope.store.get_request(row.id).status == "pending" and scope.events == []


def test_owner_check_precedes_reserved_resolver_and_preserves_bot_prefix_rule(scope):
    row = record(scope.store, kind="orchestration")
    path = f"/v1/chat/approvals/{row.id}/resolve"
    body = {"decision": "approve", "resolved_by": "bot:fixture-approver"}
    assert scope.client.post(path, json={**body, "user_id": OTHER}).status_code == 403
    assert scope.client.post(path, json={**body, "user_id": OWNER}).status_code == 400
    assert scope.store.get_request(row.id).status == "pending" and scope.events == []


def test_existing_internal_inprocess_resolution_needs_no_browser_body(scope):
    row = record(scope.store, kind="orchestration")
    saved = scope.store.resolve_request(row.id, status=REQ_DENIED, resolved_by="system:release-timeout",
                                        continuation_owner="none")
    assert saved.status == REQ_DENIED and saved.resolved_by == "system:release-timeout"


def test_openapi_declares_query_scope_for_existing_proxy_identity_binder(scope):
    paths = scope.client.get("/openapi.json").json()["paths"]
    for path in ("/v1/tool-approval-requests", "/v1/tool-approval-requests/{request_id}/result"):
        assert any(p["name"] == "user_id" and p["in"] == "query" for p in paths[path]["get"]["parameters"])
