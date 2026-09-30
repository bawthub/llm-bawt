"""TASK-861: SQLite concurrency, immutable snapshots, strict CRUD and APIs.

Run with LLM_BAWT_ENV_FILE=/dev/null and empty Postgres credentials; all stores
explicitly receive local SQLite engines and executors are fakes.
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import threading
import time

import pytest
from sqlmodel import Session, create_engine

from llm_bawt.ops.executor import DispatchResult, Executor, ReconcileResult
from llm_bawt.ops.models import OpsJob
from llm_bawt.ops.service import OpsDispatchError, OpsService
from llm_bawt.ops.store import OpsStore
from llm_bawt.ops.validation import ArgValidationError, validate_args, validate_catalog


class FakeExecutor(Executor):
    def __init__(self):
        self.calls = []
        self.result = ReconcileResult(None)
    def kind(self):
        return "docker"
    def available(self):
        return True
    def dispatch(self, **kwargs):
        self.calls.append(kwargs)
        return DispatchResult("llm-bawt-ops-" + kwargs["job_id"], terminal_state="succeeded", exit_code=0)
    def reconcile(self, **kwargs):
        return self.result


@pytest.fixture
def system(tmp_path):
    store = OpsStore(None, engine=create_engine(f"sqlite:///{tmp_path / 'ops.sqlite'}", connect_args={"check_same_thread": False, "timeout": 10}))
    executor = FakeExecutor()
    service = OpsService(store, executor=executor)
    op = store.create_operation({"slug": "test", "enabled": True,
        "command_script": '{"action":"restart","container_name_from_arg":"name"}',
        "args_schema_json": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
        "args_defaults_json": {"name": "original"}}, actor="creator")
    return store, service, executor, op


def test_new_invocations_have_fresh_keys_and_explicit_replay_conflicts(system):
    store, service, executor, op = system
    a = service.dispatch_job(operation_slug=op.slug, args={})
    b = service.dispatch_job(operation_slug=op.slug, args={})
    assert a["id"] != b["id"]
    assert a["state"] == "accepted"  # Ignore executor's premature terminal hint.
    assert a["id"] == a["job_id"]
    replay = service.dispatch_job(operation_slug=op.slug, args={}, idempotency_key=a["idempotency_key"])
    assert replay["id"] == a["id"]
    with pytest.raises(OpsDispatchError, match="different invocation"):
        service.dispatch_job(operation_slug=op.slug, args={"name": "different"}, idempotency_key=a["idempotency_key"])
    assert len(executor.calls) == 2


def test_approval_exact_revision_defaults_execution_and_current_disable_interlock(system):
    store, service, executor, op = system
    snapshot = service.prepare_invocation(op.slug, {})
    encoded = json.dumps(snapshot)
    store.update_operation(op.slug, {"args_defaults_json": {"name": "edited"}, "timeout_seconds": 71,
                                   "command_script": '{"action":"stop","container_name_from_arg":"name"}'}, actor="editor")
    result = service.dispatch_job(operation_slug=op.slug, args={}, approved_snapshot=json.loads(encoded))
    submitted = executor.calls[0]["snapshot"]
    assert submitted["resolved_args"] == {"name": "original"}
    assert submitted["execution"]["timeout_seconds"] == 300
    assert submitted["spec"]["action"] == "restart"
    assert result["operation_version"] == 1
    assert json.loads(store.get_job(result["id"]).invocation_snapshot_json) == snapshot
    store.update_operation(op.slug, {"enabled": False})
    with pytest.raises(OpsDispatchError, match="disabled"):
        service.dispatch_job(operation_slug=op.slug, args={}, approved_snapshot=snapshot)
    # Existing replay remains a read, even disabled and after edits.
    assert service.dispatch_job(operation_slug=op.slug, args={}, idempotency_key=result["idempotency_key"])["id"] == result["id"]


def test_snapshot_tampering_and_input_mismatch_rejected(system):
    _, service, executor, op = system
    snapshot = service.prepare_invocation(op.slug, {})
    snapshot["spec"]["action"] = "stop"
    with pytest.raises(OpsDispatchError, match="hash"):
        service.dispatch_job(operation_slug=op.slug, args={}, approved_snapshot=snapshot)
    snapshot = service.prepare_invocation(op.slug, {})
    with pytest.raises(OpsDispatchError, match="match requested"):
        service.dispatch_job(operation_slug=op.slug, args={"name": "other"}, approved_snapshot=snapshot)
    assert not executor.calls


def test_atomic_claim_and_queued_replay_dispatch_once(system, monkeypatch):
    store, service, executor, op = system
    original = store.claim_job
    monkeypatch.setattr(store, "claim_job", lambda *a, **kw: False)
    queued = service.dispatch_job(operation_slug=op.slug, args={}, idempotency_key="one")
    monkeypatch.setattr(store, "claim_job", original)
    assert queued["state"] == "queued"
    # Replay is read-only, including queued.
    assert service.dispatch_job(operation_slug=op.slug, args={}, idempotency_key="one")["state"] == "queued"
    assert not executor.calls
    barrier = threading.Barrier(2)
    def dispatch():
        barrier.wait()
        OpsService(store, executor=executor)._dispatch_queued(store.get_job(queued["id"]))
    with ThreadPoolExecutor(2) as pool:
        list(pool.map(lambda _: dispatch(), range(2)))
    assert len(executor.calls) == 1


def test_concurrent_same_key_insert_has_single_job(system):
    store, _, executor, op = system
    barrier = threading.Barrier(2)
    def dispatch():
        barrier.wait()
        return OpsService(store, executor=executor).dispatch_job(operation_slug=op.slug, args={}, idempotency_key="same")
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(lambda _: dispatch(), range(2)))
    assert results[0]["id"] == results[1]["id"]
    assert store.count_jobs() == 1
    assert len(executor.calls) == 1


def test_max_concurrent_and_queued_snapshot_survive_catalog_edits(system):
    store, service, executor, op = system
    store.update_operation(op.slug, {"max_concurrent": 1})
    a = service.dispatch_job(operation_slug=op.slug, args={})
    b = service.dispatch_job(operation_slug=op.slug, args={})
    assert b["state"] == "queued"
    store.update_operation(op.slug, {"args_defaults_json": {"name": "new"}, "max_concurrent": 8})
    assert service.get_job_status(b["id"])["state"] == "queued"
    store.mark_terminal(a["id"], state="succeeded", exit_code=0)
    service.get_job_status(b["id"])
    assert len(executor.calls) == 2
    assert executor.calls[1]["env_args"] == {"name": "original"}
    assert executor.calls[1]["snapshot"]["execution"]["max_concurrent"] == 1


def test_concurrent_terminal_and_running_cas_never_regresses(system):
    store, service, _, op = system
    job = service.dispatch_job(operation_slug=op.slug, args={})
    barrier = threading.Barrier(3)
    def transition(state):
        barrier.wait()
        if state == "running":
            store.mark_running(job["id"])
        else:
            store.mark_terminal(job["id"], state=state, exit_code=0 if state == "succeeded" else 1)
    with ThreadPoolExecutor(3) as pool:
        list(pool.map(transition, ["succeeded", "failed", "running"]))
    final = store.get_job(job["id"])
    assert final.state in ("succeeded", "failed")
    assert final.exit_code == (0 if final.state == "succeeded" else 1)
    store.mark_accepted(job["id"])
    assert store.get_job(job["id"]).state == final.state


def test_reconcile_receipt_timestamps_and_output_retention(system):
    store, service, executor, op = system
    job = service.dispatch_job(operation_slug=op.slug, args={})
    executor.result = ReconcileResult("succeeded", 0, "abcdef", None,
                                    "2026-01-01T00:00:00Z", "2026-01-01T00:00:05Z")
    status = service.get_job_status(job["id"], output_tail_bytes=2)
    assert status["output_tail"] == "ef"
    assert store.get_job(job["id"]).output_tail == "abcdef"
    assert status["started_at"].startswith("2026-01-01T00:00:00")


def test_legacy_active_job_is_lost_without_replay(system):
    store, service, executor, op = system
    job = store.create_job(operation=op, args_json="{}", display_args_json="{}", idempotency_key="legacy")
    assert service.get_job_status(job.id)["state"] == "lost"
    assert not executor.calls


def test_revision_audit_full_history_and_pagination(system):
    store, service, _, op = system
    store.update_operation(op.slug, {"description": "new description"}, actor="editor")
    store.update_operation(op.slug, {"enabled": False}, actor="disabler")
    store.soft_delete_operation(op.slug, actor="deleter")
    rows, total = store.list_revisions(op.slug, limit=2, offset=1)
    assert total == 4
    assert [r.version for r in rows] == [3, 2]
    assert rows[0].to_api()["actor"] == "disabler"
    assert rows[1].to_api()["operation"]["description"] == "new description"
    assert store.count_operations() == 0
    assert store.count_operations(include_disabled=True, include_soft_deleted=True) == 1
    assert store.list_operations(include_disabled=True, include_soft_deleted=True, limit=1, offset=9) == []


@pytest.mark.parametrize("change", [{"args_schema_json": []}, {"args_defaults_json": []},
    {"args_schema_json": {"properties": {"x": {"type": "bad"}}}},
    {"args_defaults_json": {"unknown": 1}}, {"args_defaults_json": {"name": 1}},
    {"timeout_seconds": 0}, {"max_concurrent": 0}, {"start_delay_seconds": -1},
    {"command_script": "echo bad"}, {"target_host": "ssh-host"}])
def test_crud_invalid_update_is_atomic(system, change):
    store, _, _, op = system
    with pytest.raises(ValueError):
        store.update_operation(op.slug, change)
    assert store.get_operation(op.id).version == 1
    assert store.list_revisions(op.slug)[1] == 1


@pytest.mark.parametrize("schema", ["[]", "null", '"x"', '{"type":"invalid"}',
    '{"required":"x"}', '{"properties":[]}', '{"properties":{"x":null}}',
    '{"properties":{"x":{"type":"string","pattern":"["}}}',
    '{"properties":{"x":{"type":"array"}}}', '{"$ref":"remote"}'])
def test_malformed_schema_rejected(schema):
    with pytest.raises(ArgValidationError):
        validate_catalog(schema)


def test_empty_schema_is_closed_and_nested_array_values_checked():
    assert validate_args({}, "{}") == {}
    with pytest.raises(ArgValidationError, match="unknown"):
        validate_args({"unknown": 1}, "{}")
    schema = json.dumps({"properties": {"items": {"type": "array", "items": {
        "type": "object", "properties": {"count": {"type": "integer"}}, "required": ["count"]}}}})
    with pytest.raises(ArgValidationError, match="integer"):
        validate_args({"items": [{"count": True}]}, schema)


def test_mcp_dispatch_does_not_block_loop_and_uses_fresh_keys(system, monkeypatch):
    from llm_bawt.mcp_server import ops_tools
    _, service, executor, op = system
    monkeypatch.setattr(ops_tools, "_get_ops_service", lambda: service)
    original = executor.dispatch
    def slow(**kwargs):
        time.sleep(0.05)
        return original(**kwargs)
    executor.dispatch = slow
    async def exercise():
        ran = []
        async def ticker():
            await asyncio.sleep(0.01)
            ran.append(True)
        fn = getattr(ops_tools.ops_run, "fn", ops_tools.ops_run)
        task = asyncio.create_task(fn(operation=op.slug, args={}))
        await ticker()
        assert ran and not task.done()
        a = await task
        b = await fn(operation=op.slug, args={})
        assert a["id"] != b["id"]
    asyncio.run(exercise())


def test_migration_adds_nullable_columns_and_baselines_only_existing_revision(tmp_path):
    from sqlalchemy import inspect
    from llm_bawt.ops.models import OpsOperation
    engine = create_engine(f"sqlite:///{tmp_path / 'legacy.sqlite'}")
    with engine.begin() as conn:
        OpsOperation.__table__.create(conn)
        # Simulate exactly the old job schema using a copied table definition.
        from sqlalchemy import MetaData, Table
        metadata = MetaData()
        Table("ops_jobs", metadata, *[column._copy() for column in OpsJob.__table__.columns
              if column.name not in {"invocation_snapshot_json", "request_payload_json", "caller_actor"}]).create(conn)
    with Session(engine) as session:
        session.add(OpsOperation(id="legacy-op", slug="legacy", version=7,
            command_script='{"action":"start","container_name":"test"}', updated_by="old-editor"))
        session.commit()
    store = OpsStore(None, engine=engine)
    columns = {c["name"] for c in inspect(engine).get_columns("ops_jobs")}
    assert {"invocation_snapshot_json", "request_payload_json", "caller_actor"} <= columns
    rows, total = store.list_revisions("legacy")
    assert total == 1
    assert rows[0].version == 7
    assert rows[0].to_api()["operation"]["command_script"] == '{"action":"start","container_name":"test"}'
    OpsStore._schema_guard.reset_for_tests(engine)
    OpsStore(None, engine=engine)
    assert store.list_revisions("legacy")[1] == 1


def test_http_pagination_attribution_and_conflict(system, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from llm_bawt.service.routes import ops as routes
    store, service, _, op = system
    monkeypatch.setattr(routes, "_store", lambda: store)
    monkeypatch.setattr(routes, "_service", lambda: service)
    app = FastAPI()
    app.include_router(routes.router)
    with TestClient(app) as client:
        result = client.post("/v1/ops/jobs", json={"operation": op.slug, "actor": "operator", "caller_user_id": "nick", "idempotency_key": "http"})
        assert result.status_code == 202
        job = result.json()
        assert job["id"] == job["job_id"]
        assert job["caller"]["actor"] == "operator"
        assert job["caller"]["user_id"] == "nick"
        response = client.get("/v1/ops/jobs?limit=1&offset=3").json()
        assert response == {"jobs": [], "total": 1, "limit": 1, "offset": 3}
        response = client.get("/v1/ops/operations?limit=1&offset=3").json()
        assert response == {"operations": [], "total": 1, "limit": 1, "offset": 3}
        conflict = client.post("/v1/ops/jobs", json={"operation": op.slug, "args": {"name": "other"}, "idempotency_key": "http"})
        assert conflict.status_code == 409
        assert client.get(f"/v1/ops/operations/{op.slug}/revisions").json()["total"] == 1


@pytest.fixture
def ops_gate(system, monkeypatch):
    """Real interception + SQLite approval ledger; returns (approvals, call).

    ``call(args, tool_use_id)`` issues one trusted ``ops_run`` transport call.
    """
    from cryptography.fernet import Fernet
    from agent_bridge.mcp_call_context import MCP_CALL_CONTEXT_KEY, mint_mcp_call_context
    from llm_bawt import task_turn_context as codec
    from llm_bawt.approval_policies import ToolApprovalPolicyStore
    from llm_bawt.mcp_server import ops_tools, registry, task_association

    store, service, executor, op = system
    approvals = ToolApprovalPolicyStore(None, engine=store.engine)
    approvals.create({"tool_name": "ops_run", "action": "require_approval"})
    registry.ensure_tools_registered()
    monkeypatch.setattr(ops_tools, "_get_ops_service", lambda: service)
    monkeypatch.setattr(registry.mcp, "_approval_store_provider", lambda: approvals)
    monkeypatch.setattr(registry.mcp, "_approval_publisher", lambda payload: None)
    # Deliberately use the production preparer lookup, not a mock inventing its
    # positional/keyword signature or the snapshot format.
    monkeypatch.setattr(registry.mcp, "_operations_preparer", None)
    fernet = Fernet(Fernet.generate_key())
    monkeypatch.setattr(codec, "_get_fernet", lambda: fernet)
    import uuid

    def call(args, tool_use_id="toolu_ops"):
        capability = codec.mint_task_turn_context(session_id=str(uuid.uuid4()),
            turn_id="turn-" + uuid.uuid4().hex, trigger_message_id=str(uuid.uuid4()),
            bot_id="ops-test", user_id="unit-user")
        stamp = mint_mcp_call_context(capability=capability, tool_name="ops_run", tool_input=args,
            tool_use_id=tool_use_id, agent_request_id="req_ops", session_key="ops-test:unit-user",
            backend="claude-code")
        token = task_association.set_current_task_turn_capability(capability)
        try:
            return asyncio.run(registry.mcp.call_tool("ops_run", {**args, MCP_CALL_CONTEXT_KEY: stamp}))
        finally:
            task_association.reset_current_task_turn_capability(token)

    return approvals, call


@pytest.fixture
def approved_ops(system, ops_gate):
    """Real interception, SQLite approval ledger, replay, and ops implementation."""
    store, service, executor, op = system
    approvals, call = ops_gate
    pending = call({"operation": op.slug, "args": {}})
    assert pending["status"] == "approval_required"
    assert not executor.calls and store.count_jobs() == 0
    row = approvals.get_request(pending["approval_request_id"])
    assert json.loads(row.operations_snapshot_json)["resolved_args"] == {"name": "original"}
    return approvals, row


def test_real_interception_approval_ops_execution_preserves_snapshot(system, approved_ops, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from llm_bawt.service.routes import approval_policies as routes
    from llm_bawt.mcp_server.approval_interceptor import current_approved_caller_context
    from llm_bawt.approval_policies import EXEC_SUCCEEDED, CONT_PENDING
    from types import SimpleNamespace

    store, service, executor, op = system
    approvals, row = approved_ops
    store.update_operation(op.slug, {"args_defaults_json": {"name": "edited"},
        "command_script": '{"action":"stop","container_name_from_arg":"name"}', "timeout_seconds": 71})
    monkeypatch.setattr(routes, "_store", lambda: approvals)
    monkeypatch.setattr(routes, "_subscriber", lambda: None)
    monkeypatch.setattr(routes, "get_turn_log_store", lambda: SimpleNamespace(set_approval_status=lambda **kwargs: None))
    app = FastAPI()
    app.include_router(routes.router)
    with TestClient(app) as client:
        response = client.post(f"/v1/chat/approvals/{row.id}/resolve", json={"decision": "approve"})
        assert response.status_code == 200
        data = response.json()
        assert data["ok"] is True and data["execution_state"] == EXEC_SUCCEEDED
        assert isinstance(data["result"], dict)
        assert data["result"]["state"] == "accepted"
        again = client.post(f"/v1/chat/approvals/{row.id}/resolve", json={"decision": "approve"}).json()
        assert again["already_resolved"] is True and again["result"] == data["result"]
    assert current_approved_caller_context() is None
    assert len(executor.calls) == 1
    snapshot = executor.calls[0]["snapshot"]
    assert snapshot == json.loads(row.operations_snapshot_json)
    assert snapshot["spec"]["action"] == "restart"
    assert executor.calls[0]["env_args"] == {"name": "original"}
    job = store.get_job_by_key(row.id)
    assert job.operation_version == 1 and job.caller_bot_id == "ops-test"
    assert job.approval_request_id == row.id and job.caller_user_id == "unit-user"
    assert approvals.get_request(row.id).continuation_state == CONT_PENDING
    executor.result = ReconcileResult("succeeded", 0)
    assert service.get_job_status(job.id)["terminal"] is True


def _approve_via_route(approvals, request_id, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from llm_bawt.service.routes import approval_policies as routes
    from types import SimpleNamespace

    monkeypatch.setattr(routes, "_store", lambda: approvals)
    monkeypatch.setattr(routes, "_subscriber", lambda: None)
    monkeypatch.setattr(routes, "get_turn_log_store", lambda: SimpleNamespace(set_approval_status=lambda **kwargs: None))
    app = FastAPI()
    app.include_router(routes.router)
    with TestClient(app) as client:
        response = client.post(f"/v1/chat/approvals/{request_id}/resolve", json={"decision": "approve"})
    assert response.status_code == 200
    return response.json()


def test_identical_pending_ops_run_is_refused_not_duplicated(system, ops_gate):
    """TASK-959: a re-sent identical call gets the existing approval, no new one."""
    store, _, executor, op = system
    approvals, call = ops_gate
    args = {"operation": op.slug, "args": {}, "idempotency_key": "TASK-958-restart-app-1"}
    first = call(args, tool_use_id="toolu_first")
    second = call(args, tool_use_id="toolu_second")
    assert first["status"] == "approval_required"
    assert second["status"] == "approval_pending" and second["is_error"] is True
    assert second["approval_request_id"] == first["approval_request_id"]
    assert approvals.count_requests(status="pending") == 1
    different = call({**args, "idempotency_key": "other-key"}, tool_use_id="toolu_third")
    assert different["status"] == "approval_required"
    assert not executor.calls and store.count_jobs() == 0


def test_caller_key_survives_approval_and_collapses_repeat_to_one_job(system, ops_gate, monkeypatch):
    """TASK-959: approved replays keep the caller's key, so ops dedupes them."""
    store, _, executor, op = system
    approvals, call = ops_gate
    args = {"operation": op.slug, "args": {}, "idempotency_key": "TASK-958-restart-app-1"}
    first = call(args, tool_use_id="toolu_first")
    done = _approve_via_route(approvals, first["approval_request_id"], monkeypatch)
    assert done["result"]["idempotency_key"] == "TASK-958-restart-app-1"
    job = store.get_job_by_key("TASK-958-restart-app-1")
    assert job is not None and job.approval_request_id == first["approval_request_id"]
    # After resolution an identical call may ask again; if approved it must
    # land on the same job, not dispatch a second one.
    again = call(args, tool_use_id="toolu_second")
    assert again["status"] == "approval_required"
    assert again["approval_request_id"] != first["approval_request_id"]
    replay = _approve_via_route(approvals, again["approval_request_id"], monkeypatch)
    assert replay["result"]["job_id"] == job.id
    assert len(executor.calls) == 1 and store.count_jobs() == 1


@pytest.mark.parametrize("snapshot_json", [None, "", "{}", "null", "[]"])
def test_recovery_missing_or_empty_snapshot_fails_without_dispatch(system, approved_ops, snapshot_json):
    from llm_bawt.approval_policies import ToolApprovalRequest, REQ_APPROVED, EXEC_FAILED
    from llm_bawt.service.approval_continuations import recover_approval_requests

    store, _, executor, _ = system
    approvals, row = approved_ops
    approvals.resolve_request(row.id, status=REQ_APPROVED)
    with Session(approvals.engine) as session:
        current = session.get(ToolApprovalRequest, row.id)
        current.operations_snapshot_json = snapshot_json
        session.add(current)
        session.commit()
    asyncio.run(recover_approval_requests(approvals))
    final = approvals.get_request(row.id)
    assert final.execution_state == EXEC_FAILED and final.result_is_error
    assert store.count_jobs() == 0 and not executor.calls


def test_recovery_after_job_acceptance_reuses_durable_snapshot_job(system, approved_ops, monkeypatch):
    from datetime import datetime, timedelta, timezone
    from llm_bawt.approval_policies import ToolApprovalRequest, EXEC_SUCCEEDED
    from llm_bawt.service.approval_continuations import recover_approval_requests
    from llm_bawt.service.routes.approval_policies import _resolve_mcp_request

    store, _, executor, op = system
    approvals, row = approved_ops
    original = approvals.complete_mcp_execution
    def unavailable(*args, **kwargs):
        raise RuntimeError("simulated result-commit crash")
    monkeypatch.setattr(approvals, "complete_mcp_execution", unavailable)
    with pytest.raises(RuntimeError, match="result-commit crash"):
        asyncio.run(_resolve_mcp_request(approvals, row, outcome="approve", message="", resolved_by=None))
    assert len(executor.calls) == 1
    with Session(approvals.engine) as session:
        current = session.get(ToolApprovalRequest, row.id)
        current.execution_started_at = datetime.now(timezone.utc) - timedelta(seconds=301)
        session.add(current)
        session.commit()
    store.update_operation(op.slug, {"args_defaults_json": {"name": "edited"}})
    monkeypatch.setattr(approvals, "complete_mcp_execution", original)
    asyncio.run(recover_approval_requests(approvals))
    final = approvals.get_request(row.id)
    assert final.execution_state == EXEC_SUCCEEDED
    assert final.execution_attempts == 2
    assert len(executor.calls) == 1 and store.count_jobs() == 1
    assert json.loads(final.result_json)["job_id"] == store.get_job_by_key(row.id).id


@pytest.mark.parametrize("state,is_error", [("succeeded", False), ("failed", True),
    ("timed_out", True), ("lost", True), ("cancelled", True)])
def test_recovered_terminal_ops_job_has_truthful_approval_result(system, approved_ops, state, is_error):
    from llm_bawt.approval_policies import EXEC_FAILED, EXEC_SUCCEEDED
    from llm_bawt.service.routes.approval_policies import _resolve_mcp_request

    store, service, executor, op = system
    approvals, row = approved_ops
    # Simulate an accepted worker whose terminal receipt was reconciled after
    # the app died, before its approval result could be persisted.
    job = service.dispatch_job(operation_slug=op.slug, args={}, idempotency_key=row.id,
        approved_snapshot=json.loads(row.operations_snapshot_json))
    store.mark_terminal(job["id"], state=state, exit_code=0 if state == "succeeded" else None)
    result = asyncio.run(_resolve_mcp_request(approvals, row,
        outcome="approve", message="", resolved_by=None))
    assert result["result"]["state"] == state
    assert result["result_is_error"] is is_error
    assert result["ok"] is (not is_error)
    assert result["execution_state"] == (EXEC_FAILED if is_error else EXEC_SUCCEEDED)
    assert len(executor.calls) == 1


def test_active_job_pagination_eventually_reconciles_beyond_first_page(system):
    from llm_bawt.ops.reconciler import OpsReconciler

    store, service, executor, op = system
    jobs = [service.dispatch_job(operation_slug=op.slug, args={}) for _ in range(5)]
    reconciler = OpsReconciler(service, batch_size=2)
    executor.result = ReconcileResult("succeeded", 0)
    # Transitions shrink the active set; a subsequent cycle must pick up rows
    # skipped by offset shifts rather than starving them permanently.
    for _ in range(5):
        reconciler.tick()
    assert all(store.get_job(job["id"]).state == "succeeded" for job in jobs)
    assert len(executor.calls) == 5
