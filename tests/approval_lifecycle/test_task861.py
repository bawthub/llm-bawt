from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, create_engine

from agent_bridge.mcp_call_context import canonical_invocation_hash
from llm_bawt.approval_policies import (
    ApprovalPersistError, ApprovalStoreUnavailable, CONT_DELIVERED, CONT_DISPATCHING,
    CONT_NOT_NEEDED, CONT_PENDING, EXEC_FAILED, EXEC_SUCCEEDED,
    REQ_APPROVED, REQ_DENIED, ToolApprovalPolicyStore, ToolApprovalRequest,
)
from llm_bawt.mcp_server.approval_interceptor import ApprovalAwareFastMCP
from llm_bawt.service import approval_continuations as continuations
from llm_bawt.service.routes import approval_policies as routes


@pytest.fixture
def store():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    value = ToolApprovalPolicyStore(None, engine=engine)
    yield value
    engine.dispose()


@pytest.fixture
def client(store, monkeypatch):
    monkeypatch.setattr(routes, "_store", lambda: store)
    monkeypatch.setattr(routes, "_subscriber", lambda: None)
    monkeypatch.setattr(routes, "get_turn_log_store", lambda: SimpleNamespace(set_approval_status=lambda **kw: None))
    app = FastAPI()
    app.include_router(routes.router)
    with TestClient(app) as value:
        yield value


def record(store, request_id="req-1", tool_name="generic", **kwargs):
    arguments = {"operation": "demo", "args": {}} if tool_name == "ops_run" else {"x": 1}
    caller_context = {
        "session_id": "original-session", "turn_id": "turn-1",
        "trigger_message_id": "message-1", "bot_id": "test-bot",
        "user_id": "test-user", "issued_at": 1,
        "agent_request_id": "agent-request-1", "session_key": "test-bot:test-user",
        "backend": "claude-code", "tool_use_id": "toolu-1",
    }
    return store.record_mcp_request(
        request_id=request_id, tool_use_id="toolu-1", mcp_server="bawthub", bot_id="test-bot",
        user_id="test-user", turn_id="turn-1", trigger_message_id="message-1",
        session_key="test-bot:test-user", backend="claude-code", tool_name=tool_name,
        tool_arguments=arguments, subject="subject", grant_key="g", policy_id=None, severity="high",
        prompt="Approve?", invocation_hash=canonical_invocation_hash(tool_name, arguments),
        continuation_capable=True, caller_context_json=json.dumps(caller_context), **kwargs,
    )


def harness(store, request_id="h-1"):
    return store.record_request(request_id=request_id, bot_id="test-bot", user_id="test-user",
        turn_id="turn-1", backend="claude-code", tool_name="Bash", tool_arguments={"command": "x"},
        subject="x", grant_key="g", policy_id=None, severity="high", prompt="Approve?", session_id="thread-1")


def expire(store, row, **values):
    with Session(store.engine) as session:
        current = session.get(ToolApprovalRequest, row.id)
        for key, value in values.items():
            setattr(current, key, value)
        session.add(current)
        session.commit()


def approve_claim(store, row):
    store.resolve_request(row.id, status=REQ_APPROVED)
    return store.claim_mcp_execution(row.id)


def ready(store):
    row = approve_claim(store, record(store))
    store.complete_mcp_execution(row.id, result_json='{"answer":42}', is_error=False,
                                 claim_token=row.execution_claim_token)
    return store.claim_continuation(row.id)


def test_unavailable_is_not_an_empty_bundle():
    store = object.__new__(ToolApprovalPolicyStore)
    store.engine = None
    with pytest.raises(ApprovalStoreUnavailable):
        store.compile_bundle()


@pytest.mark.parametrize("payload", [
    {"matcher_type": "regex", "pattern": "["},
    {"matcher_type": "prefix", "pattern": ""},
    {"matcher_type": "bogus"}, {"action": "bogus"}, {"severity": "bogus"},
    {"tool_name": " "}, {"pattern": "x\x00"}, {"order": 1, "order_index": 2},
])
def test_invalid_create_is_rejected(client, payload):
    response = client.post("/v1/tool-approval-policies", json=payload)
    assert response.status_code == 422


def test_update_validates_combined_candidate_and_rolls_back(client, store):
    policy = store.create({"matcher_type": "regex", "pattern": "valid"})
    assert client.patch(f"/v1/tool-approval-policies/{policy.id}", json={"pattern": "["}).status_code == 422
    assert store.get(policy.id).pattern == "valid"
    assert store.get(policy.id).version == 1
    other = store.create({"matcher_type": "always", "pattern": "["})
    assert client.patch(f"/v1/tool-approval-policies/{other.id}", json={"matcher_type": "regex"}).status_code == 422


def test_preview_full_candidate_bundle_and_order_alias(client, store):
    store.create({"tool_name": "Bash", "action": "deny"})
    body = {"backend": "claude-code", "tool_name": "Bash", "tool_input": {"command": "echo OK"}}
    assert client.post("/v1/tool-approval-policies/preview", json=body).json()["action"] == "deny"
    assert client.post("/v1/tool-approval-policies/preview", json={**body, "policies": []}).json()["action"] == "allow"
    policies = [{"id": "denied", "action": "deny", "order": 10},
                {"id": "allowed", "action": "allow", "order_index": 1}]
    response = client.post("/v1/tool-approval-policies/preview", json={**body, "policies": policies})
    assert response.status_code == 200
    assert response.json()["policy_id"] == "allowed"
    assert response.json()["subject"] == "echo OK"
    assert len(store.list_all()) == 1
    assert client.post("/v1/tool-approval-policies/preview", json={**body, "policies": [{"matcher_type": "regex", "pattern": "["}]}).status_code == 422


def test_revisions_reconstruct_create_update_delete(client, store):
    policy = store.create({"pattern": "a"}, actor="first")
    store.update(policy.id, {"pattern": "b", "enabled": False}, actor="second")
    store.delete(policy.id)
    response = client.get(f"/v1/tool-approval-policies/{policy.id}/revisions").json()
    assert response["total"] == 3
    assert [r["change"] for r in response["revisions"]] == ["delete", "update", "create"]
    assert response["revisions"][-1]["policy"]["pattern"] == "a"
    assert response["revisions"][1]["actor"] == "second"


def test_deleted_seed_is_not_resurrected(store):
    store.seed_defaults()
    assert store.delete("seed-ops-run-require-approval")
    assert store.seed_defaults() == 0
    assert store.get("seed-ops-run-require-approval") is None


def test_request_pagination_reports_all_matches(client, store):
    for index in range(7):
        record(store, request_id=f"r-{index}")
    store.resolve_request("r-0", status=REQ_DENIED)
    response = client.get("/v1/tool-approval-requests?status=pending&limit=2&offset=2").json()
    assert response["total"] == 6
    assert len(response["requests"]) == 2
    assert response["offset"] == 2 and response["limit"] == 2
    assert client.get("/v1/tool-approval-requests?limit=999&offset=-9").json()["limit"] == 200


def test_mcp_allow_and_deny_are_audited_separately(client, store):
    server = ApprovalAwareFastMCP("test", approval_store_provider=lambda: store)
    executed = []

    @server.tool()
    async def harmless(value: str):
        executed.append(value)
        return {"value": value}

    asyncio.run(server.call_tool("harmless", {"value": "hello"}))
    policy = store.create({"tool_name": "harmless", "action": "deny"})
    denied = asyncio.run(server.call_tool("harmless", {"value": "blocked"}))
    assert denied["status"] == "denied"
    response = client.get("/v1/tool-approval-decisions?limit=1&offset=0").json()
    assert response["total"] == 2 and len(response["decisions"]) == 1
    assert response["decisions"][0]["policy_id"] == policy.id
    assert response["decisions"][0]["policy"]["action"] == "deny"
    assert store.count_requests() == 0 and executed == ["hello"]


def test_raw_gated_mcp_call_fails_before_snapshot_or_persistence(store):
    store.create({"tool_name": "ops_run"})
    snapshot = {"operation_slug": "demo", "rendered_command": "original", "version": 1}
    prepared = []
    server = ApprovalAwareFastMCP("test", approval_store_provider=lambda: store,
        approval_publisher=lambda payload: None,
        operations_preparer=lambda operation, args: prepared.append((operation, args)) or snapshot)
    result = asyncio.run(server.call_tool("ops_run", {"operation": "demo", "args": {}}))
    assert result["status"] == "approval_context_missing"
    assert prepared == []
    assert store.count_requests() == 0


def test_atomic_completion_enqueues_and_cancel_stays_silent(store):
    row = approve_claim(store, record(store))
    done = store.complete_mcp_execution(row.id, result_json="{}", is_error=False, claim_token=row.execution_claim_token)
    assert done.execution_state == EXEC_SUCCEEDED
    assert done.continuation_state == CONT_PENDING
    second = record(store, request_id="cancel")
    store.resolve_request(second.id, status="cancelled")
    done = store.complete_mcp_execution(second.id, result_json="{}", is_error=False, skipped=True)
    assert done.continuation_state == CONT_NOT_NEEDED


def test_generic_stale_execution_is_uncertain_not_replayed(store):
    row = approve_claim(store, record(store))
    expire(store, row, execution_started_at=datetime.now(timezone.utc) - timedelta(minutes=10))
    assert store.claim_mcp_execution(row.id) is None
    done = store.get_request(row.id)
    assert done.execution_state == EXEC_FAILED
    assert json.loads(done.result_json)["uncertain"] is True
    assert done.continuation_state == CONT_PENDING


def test_ops_stale_execution_is_reclaimed_and_old_claim_is_fenced(store):
    row = approve_claim(store, record(store, tool_name="ops_run", operations_snapshot={"version": 1}))
    expire(store, row, execution_started_at=datetime.now(timezone.utc) - timedelta(minutes=10))
    newer = store.claim_mcp_execution(row.id)
    assert newer.execution_attempts == 2 and newer.execution_claim_token != row.execution_claim_token
    with pytest.raises(ApprovalPersistError, match="claim lost"):
        store.complete_mcp_execution(row.id, result_json='{"stale":true}', is_error=False, claim_token=row.execution_claim_token)
    assert store.get_request(row.id).result_json is None


def test_continuation_claim_checks_due_time_and_fences_stale_ack(store):
    row = ready(store)
    expire(store, row, continuation_next_attempt_at=datetime.now(timezone.utc) - timedelta(minutes=10))
    newer = store.claim_continuation(row.id)
    store.mark_continuation_delivered(row.id, claim_token=row.continuation_claim_token)
    assert store.get_request(row.id).continuation_state == CONT_DISPATCHING
    store.mark_continuation_failed(row.id, error="later", claim_token=newer.continuation_claim_token)
    assert store.claim_continuation(row.id) is None


def test_resolve_obeys_persisted_winner_and_error_payload(client, store, monkeypatch):
    from llm_bawt.mcp_server import registry
    row = record(store)
    store.resolve_request(row.id, status=REQ_DENIED, message="first")
    called = []

    async def execute(*args, **kwargs):
        called.append(kwargs)
        return {"status": "failed", "error": "tool failure"}

    monkeypatch.setattr(registry, "mcp", SimpleNamespace(call_approved_tool=execute))
    response = client.post(f"/v1/chat/approvals/{row.id}/resolve", json={"decision": "approve"}).json()
    assert response["status"] == "denied" and response["ok"] is False
    assert called == []
    other = record(store, request_id="other")
    response = client.post(f"/v1/chat/approvals/{other.id}/resolve", json={"decision": "approve"}).json()
    assert response["execution_state"] == EXEC_FAILED and response["ok"] is False
    client.post(f"/v1/chat/approvals/{other.id}/resolve", json={"decision": "approve"})
    assert len(called) == 1


def test_admin_resolve_defaults_durable_server_owner_and_retries_no_duplicate(client, store):
    row = harness(store)
    first = client.post(f"/v1/chat/approvals/{row.id}/resolve", json={"decision": "respond", "message": "original"}).json()
    second = client.post(f"/v1/chat/approvals/{row.id}/resolve", json={"decision": "approve", "message": "changed", "dispatch_continuation": False}).json()
    assert first["server_dispatched"] is True
    assert first["continuation_prompt"] is None
    assert second["status"] == "responded" and second["continuation_owner"] == "server"
    saved = store.get_request(row.id)
    assert saved.resolution_message == "original" and saved.continuation_state == CONT_PENDING
    assert len(store.find_pending_continuations()) == 1


def test_client_owner_gets_explicit_grant_failure_not_success(client, store):
    row = harness(store)
    response = client.post(f"/v1/chat/approvals/{row.id}/resolve", json={"decision": "approve", "dispatch_continuation": False})
    assert response.status_code == 503
    saved = store.get_request(row.id)
    assert saved.status == REQ_APPROVED and saved.grant_state == "failed"
    assert saved.continuation_owner == "client"


class SessionService:
    def get_memory_client(self, bot_id, user_id):
        return SimpleNamespace(get_session=lambda session_id: {
            "id": session_id, "bot_id": bot_id, "user_id": user_id,
            "status": "active", "session_metadata": {},
        })


def test_dispatch_is_session_bound_and_retry_has_stable_identity(store):
    row = ready(store)
    requests = []

    class Service(SessionService):
        _tool_approval_policy_store = store

        async def chat_completion_stream(self, request):
            continuations.validate_approval_continuation_claim(self, request, request._internal_approval_claim)
            requests.append(request)
            if len(requests) == 1:
                raise RuntimeError("offline")
            yield "data: [DONE]\n\n"

    service = Service()
    with pytest.raises(RuntimeError, match="offline"):
        asyncio.run(continuations.dispatch_mcp_result_continuation(service, store, row))
    expire(store, row, continuation_next_attempt_at=datetime.now(timezone.utc) - timedelta(seconds=1))
    newer = store.claim_continuation(row.id)
    asyncio.run(continuations.dispatch_mcp_result_continuation(service, store, newer))
    assert requests[0].session_id == requests[1].session_id == "original-session"
    assert requests[0].inter_bot_turn_id == requests[1].inter_bot_turn_id
    assert requests[0].inter_bot_bridge_request_id == requests[1].inter_bot_bridge_request_id
    assert requests[0].user_message_id == requests[1].user_message_id
    assert store.get_request(row.id).continuation_state == CONT_DELIVERED


def test_error_stream_is_not_marked_delivered(store):
    row = ready(store)

    class Service(SessionService):
        async def chat_completion_stream(self, request):
            yield 'data: {"error":"bridge failed"}\n\n'

    with pytest.raises(RuntimeError, match="bridge failed"):
        asyncio.run(continuations.dispatch_mcp_result_continuation(Service(), store, row))
    assert store.get_request(row.id).continuation_state == CONT_PENDING


def test_recovery_finishes_stranded_generic_claim_without_execution(store, monkeypatch):
    from llm_bawt.mcp_server import registry
    row = approve_claim(store, record(store))
    expire(store, row, execution_started_at=datetime.now(timezone.utc) - timedelta(minutes=10))

    async def forbidden(*args, **kwargs):
        raise AssertionError("generic side effects must not be replayed")

    monkeypatch.setattr(registry, "mcp", SimpleNamespace(call_approved_tool=forbidden))
    asyncio.run(continuations.recover_approval_requests(store))
    assert store.get_request(row.id).execution_state == EXEC_FAILED


def test_ops_recovery_uses_same_key_and_immutable_snapshot(store, monkeypatch):
    from llm_bawt.mcp_server import registry
    snapshot = {"operation_slug": "demo", "script": "approved", "args": {}}
    row = approve_claim(store, record(store, tool_name="ops_run", operations_snapshot=snapshot))
    expire(store, row, execution_started_at=datetime.now(timezone.utc) - timedelta(minutes=10))
    calls = []

    async def execute(name, arguments, **kwargs):
        calls.append(kwargs)
        assert kwargs["caller_context"].operations_snapshot == snapshot
        assert kwargs["trusted_argument_overrides"] == {"idempotency_key": row.id}
        assert kwargs["expected_invocation_hash"] == row.invocation_hash
        return {"job_id": "same-job", "state": "queued"}

    monkeypatch.setattr(registry, "mcp", SimpleNamespace(call_approved_tool=execute))
    asyncio.run(continuations.recover_approval_requests(store))
    asyncio.run(continuations.recover_approval_requests(store))
    assert len(calls) == 1
    assert store.get_request(row.id).execution_state == EXEC_SUCCEEDED


def test_legacy_ops_without_snapshot_fails_without_using_live_catalog(store, monkeypatch):
    from llm_bawt.mcp_server import registry
    row = record(store, tool_name="ops_run")
    calls = []

    async def execute(*args, **kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(registry, "mcp", SimpleNamespace(call_approved_tool=execute))
    response = asyncio.run(routes._resolve_mcp_request(store, row, outcome="approve", message="", resolved_by=None))
    assert calls == []
    assert response["execution_state"] == EXEC_FAILED
    assert "no immutable snapshot" in response["result"]["error"]


def test_recovery_repairs_pre_atomic_result_outbox_gap(store):
    row = approve_claim(store, record(store))
    expire(store, row, execution_state=EXEC_SUCCEEDED, result_json='{"ok":true}',
           continuation_state=CONT_NOT_NEEDED, continuation_id=None)
    asyncio.run(continuations.recover_approval_requests(store))
    assert store.get_request(row.id).continuation_state == CONT_PENDING
    assert store.get_request(row.id).continuation_id


def test_client_owner_retries_do_not_regrant_or_change_response(client, store, monkeypatch):
    row = harness(store)
    grants = []

    class Subscriber:
        async def send_approval_grant(self, **kwargs):
            grants.append(kwargs)

    monkeypatch.setattr(routes, "_subscriber", lambda: Subscriber())
    body = {"decision": "approve", "dispatch_continuation": False}
    first = client.post(f"/v1/chat/approvals/{row.id}/resolve", json=body).json()
    second = client.post(f"/v1/chat/approvals/{row.id}/resolve", json={"decision": "deny"}).json()
    assert len(grants) == 1
    assert first["continuation_prompt"] == second["continuation_prompt"]
    assert first["server_dispatched"] is False and second["status"] == "approved"
    assert store.get_request(row.id).continuation_owner == "client"


def test_uncertain_grant_is_not_silently_retried(client, store, monkeypatch):
    row = harness(store)
    grants = []

    class Subscriber:
        async def send_approval_grant(self, **kwargs):
            grants.append(kwargs)
            raise RuntimeError("response lost after send")

    monkeypatch.setattr(routes, "_subscriber", lambda: Subscriber())
    body = {"decision": "approve", "dispatch_continuation": False}
    assert client.post(f"/v1/chat/approvals/{row.id}/resolve", json=body).status_code == 503
    assert client.post(f"/v1/chat/approvals/{row.id}/resolve", json=body).status_code == 503
    assert len(grants) == 1 and store.get_request(row.id).grant_state == "uncertain"


def test_continuation_turn_error_does_not_count_as_delivery(store):
    row = ready(store)
    completed = False

    class TurnStore:
        def get_turn(self, turn_id):
            return SimpleNamespace(ended_at=datetime.now(timezone.utc), status="error", error_text="provider failed") if completed else None

    class Service(SessionService):
        _turn_log_store = TurnStore()

        async def chat_completion_stream(self, request):
            nonlocal completed
            completed = True
            yield "data: [DONE]\n\n"

    with pytest.raises(RuntimeError, match="provider failed"):
        asyncio.run(continuations.dispatch_mcp_result_continuation(Service(), store, row))
    assert store.get_request(row.id).continuation_state == CONT_PENDING


def test_bundle_outage_returns_503_not_healthy_empty(client, store, monkeypatch):
    def unavailable():
        raise ApprovalStoreUnavailable("offline")

    monkeypatch.setattr(store, "compile_bundle", unavailable)
    assert client.get("/v1/tool-approval-policies/bundle").status_code == 503
    response = client.post("/v1/tool-approval-policies/preview", json={"backend": "mcp", "tool_name": "x", "tool_input": {}, "policies": []})
    assert response.status_code == 200 and response.json()["action"] == "allow"


def test_execution_heartbeat_cannot_renew_stale_worker(store):
    row = approve_claim(store, record(store, tool_name="ops_run", operations_snapshot={"version": 1}))
    expire(store, row, execution_started_at=datetime.now(timezone.utc) - timedelta(minutes=10))
    newer = store.claim_mcp_execution(row.id)
    assert store.renew_mcp_execution_claim(row.id, row.execution_claim_token) is False
    assert store.renew_mcp_execution_claim(row.id, newer.execution_claim_token) is True
