"""TASK-1220: ops-owned admission of approved ``ops_run``.

Contract parity with the direct MCP tool (M4), reconciler tick isolation (M7),
health reporting (S2/S3) and the no-lazy-import invariant (M6).
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import textwrap
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_bridge.mcp_call_context import canonical_invocation_hash
from llm_bawt.approval_policies import REQ_APPROVED, ToolApprovalPolicyStore
from llm_bawt.ops.approved_admission import (
    TOOL_ERROR_RESULT,
    AdmissionOutcome,
    ApprovedOpsAdmission,
)
from llm_bawt.ops.executor import DispatchResult, Executor, ReconcileResult
from llm_bawt.ops.models import JOB_FAILED
from llm_bawt.ops.reconciler import OpsReconciler
from llm_bawt.ops.service import OpsService
from llm_bawt.ops.store import OpsStore
from llm_bawt.service.health_checks import (
    DEGRADED, DOWN, OK, check_mcp_tool_registry, check_ops_admission,
)
from tests.sqlite_support import fast_sqlite_engine

REPO = Path(__file__).resolve().parents[1]


class FakeExecutor(Executor):
    def __init__(self):
        self.calls = []

    def kind(self):
        return "docker"

    def available(self):
        return True

    def dispatch(self, **kwargs):
        self.calls.append(kwargs)
        return DispatchResult("llm-bawt-ops-" + kwargs["job_id"])

    def reconcile(self, **kwargs):
        return ReconcileResult(None)


@pytest.fixture
def system(tmp_path):
    engine = fast_sqlite_engine(tmp_path / "ops.sqlite", check_same_thread=False, timeout=10)
    store = OpsStore(None, engine=engine)
    executor = FakeExecutor()
    service = OpsService(store, executor=executor)
    store.create_operation({"slug": "demo", "enabled": True,
        "command_script": '{"action":"restart","container_name_from_arg":"name"}',
        "args_schema_json": {"type": "object", "properties": {"name": {"type": "string"}}},
        "args_defaults_json": {"name": "app"}}, actor="creator")
    approvals = ToolApprovalPolicyStore(None, engine=engine)
    return SimpleNamespace(store=store, service=service, executor=executor, approvals=approvals)


def _approved(system, *, request_id="req-1", arguments=None):
    arguments = arguments or {"operation": "demo", "args": {}}
    snapshot = system.service.prepare_invocation(arguments["operation"], arguments.get("args"))
    row = system.approvals.record_mcp_request(
        request_id=request_id, tool_use_id="toolu-" + request_id, mcp_server="bawthub",
        bot_id="bot", user_id="user", turn_id="turn-1", trigger_message_id="message-1",
        session_key="bot:user", backend="claude-code", tool_name="ops_run",
        tool_arguments=arguments, subject="demo", grant_key="g", policy_id=None,
        severity="high", prompt="Approve?",
        invocation_hash=canonical_invocation_hash("ops_run", arguments),
        continuation_capable=True, operations_snapshot=snapshot,
        caller_context_json=json.dumps({"session_id": "s", "turn_id": "turn-1",
            "trigger_message_id": "message-1", "bot_id": "bot", "user_id": "user",
            "issued_at": 1, "agent_request_id": "a", "session_key": "bot:user",
            "backend": "claude-code", "tool_use_id": "toolu-" + request_id}))
    return system.approvals.resolve_request(row.id, status=REQ_APPROVED, continuation_owner="server")


def _direct_ops_run(service, monkeypatch, **arguments):
    from llm_bawt.mcp_server import ops_tools

    monkeypatch.setattr(ops_tools, "_get_ops_service", lambda: service)
    return asyncio.run(ops_tools.ops_run(**arguments))


# ---- M4: one ops_run contract for both paths -------------------------------

def test_admitted_receipt_matches_direct_tool_receipt(system, monkeypatch):
    direct = _direct_ops_run(system.service, monkeypatch, operation="demo", args={},
                             idempotency_key="direct-key")
    row = _approved(system, arguments={"operation": "demo", "args": {}, "idempotency_key": " admitted-key "})
    assert ApprovedOpsAdmission(system.approvals, system.service).admit_due() == 1
    admitted = json.loads(system.approvals.get_request(row.id).result_json)
    assert set(admitted) == set(direct)
    assert admitted["idempotency_key"] == "admitted-key"
    assert {k: admitted[k] for k in ("operation", "state")} == {k: direct[k] for k in ("operation", "state")}


def test_disabled_operation_error_text_matches_direct_tool(system, monkeypatch):
    row = _approved(system)
    system.store.update_operation("demo", {"enabled": False})
    with pytest.raises(RuntimeError) as direct:
        _direct_ops_run(system.service, monkeypatch, operation="demo", args={})
    assert ApprovedOpsAdmission(system.approvals, system.service).admit_due() == 1
    final = system.approvals.get_request(row.id)
    # One formatter, one code; ops words the approved-snapshot interlock differently.
    assert str(direct.value) == "[operation_disabled] operation 'demo' is disabled"
    expected = "[operation_disabled] operation is currently disabled or deleted"
    assert json.loads(final.result_json) == {"status": "failed", "error": expected}
    assert final.result_is_error and final.execution_error == expected
    assert not system.executor.calls


def test_failed_job_receipt_is_an_error_result():
    outcome = AdmissionOutcome.receipt({"id": "j", "operation": "demo", "state": JOB_FAILED,
                                        "submitted_at": "t"}, "k")
    assert outcome.is_error is True and outcome.error == TOOL_ERROR_RESULT
    assert outcome.payload["job_id"] == "j"


def test_direct_tool_refuses_an_approved_caller_context(system, monkeypatch):
    from llm_bawt.mcp_server import approval_interceptor
    from llm_bawt.mcp_server import ops_tools

    monkeypatch.setattr(ops_tools, "current_approved_caller_context", lambda: object())
    with pytest.raises(RuntimeError, match=r"^\[admission_owned\]"):
        _direct_ops_run(system.service, monkeypatch, operation="demo")
    assert not system.executor.calls
    assert approval_interceptor.current_approved_caller_context() is None


@pytest.mark.parametrize("arguments,code", [
    ({"operation": "demo", "args": {}, "surprise": 1}, "[args_invalid]"),
    ({"operation": "", "args": {}}, "[args_invalid]"),
    ({"operation": "demo", "args": []}, "[args_invalid]"),
])
def test_schema_invalid_stored_arguments_are_refused(system, arguments, code):
    row = _approved(system, arguments={"operation": "demo", "args": {}})
    from sqlmodel import Session
    from llm_bawt.approval_policies import ToolApprovalRequest

    with Session(system.approvals.engine) as session:
        current = session.get(ToolApprovalRequest, row.id)
        current.tool_arguments_json = json.dumps(arguments)
        current.invocation_hash = canonical_invocation_hash("ops_run", arguments)
        session.add(current)
        session.commit()
    assert ApprovedOpsAdmission(system.approvals, system.service).admit_due() == 1
    assert json.loads(system.approvals.get_request(row.id).result_json)["error"].startswith(code)
    assert not system.executor.calls


def test_tampered_arguments_fail_the_invocation_hash(system):
    """M5: admission runs exactly the invocation the human approved."""
    from sqlmodel import Session
    from llm_bawt.approval_policies import ToolApprovalRequest

    row = _approved(system)
    with Session(system.approvals.engine) as session:
        current = session.get(ToolApprovalRequest, row.id)
        current.tool_arguments_json = json.dumps({"operation": "demo", "args": {"name": "other"}})
        session.add(current)
        session.commit()
    assert ApprovedOpsAdmission(system.approvals, system.service).admit_due() == 1
    error = json.loads(system.approvals.get_request(row.id).result_json)["error"]
    assert error.startswith("[invocation_mismatch]")
    assert not system.executor.calls and system.store.count_jobs() == 0


# ---- M7: bounded, isolated admission inside the reconciler tick ------------

def test_admission_is_bounded_per_tick(system):
    rows = [_approved(system, request_id=f"req-{i}") for i in range(3)]
    admission = ApprovedOpsAdmission(system.approvals, system.service)
    assert admission.admit_due(limit=2) == 2
    assert admission.admit_due(limit=2) == 1
    assert system.store.count_jobs() == 3 and len(rows) == 3


def test_one_failing_row_does_not_stop_the_batch(system, monkeypatch):
    first, second = _approved(system, request_id="req-a"), _approved(system, request_id="req-b")
    admission = ApprovedOpsAdmission(system.approvals, system.service)
    original = admission._decide

    def flaky(row):
        if row.id == first.id:
            raise RuntimeError("boom")
        return original(row)

    monkeypatch.setattr(admission, "_decide", flaky)
    assert admission.admit_due() == 1
    assert system.approvals.get_request(second.id).result_json is not None
    assert system.approvals.get_request(first.id).result_json is None


class _Service:
    def __init__(self, error=None):
        self.error = error
        self.calls = 0

    def reconcile_active_jobs(self, *, limit, offset):
        self.calls += 1
        if self.error:
            raise self.error
        return 0


class _Admission:
    def __init__(self, error=None):
        self.error = error
        self.calls = 0

    def admit_due(self):
        self.calls += 1
        if self.error:
            raise self.error
        return 0


def test_tick_runs_admission_even_when_reconcile_fails_and_reports_it():
    service, admission = _Service(RuntimeError("reconcile down")), _Admission()
    with pytest.raises(RuntimeError, match="reconcile down"):
        OpsReconciler(service, admission=admission).tick()
    assert service.calls == 1 and admission.calls == 1


def test_tick_runs_reconcile_even_when_admission_fails_and_reports_it():
    service, admission = _Service(), _Admission(RuntimeError("admission down"))
    with pytest.raises(RuntimeError, match="admission down"):
        OpsReconciler(service, admission=admission).tick()
    assert service.calls == 1 and admission.calls == 1


# ---- S2/S3: the app reports a stalled reconciler and a missing registry ----

def test_admission_health_tracks_oldest_waiting_approval(system):
    approvals = system.approvals
    assert check_ops_admission(approvals)["status"] == OK
    row = _approved(system)
    now = row.resolved_at.replace(tzinfo=timezone.utc) if row.resolved_at.tzinfo is None else row.resolved_at
    assert check_ops_admission(approvals, now + timedelta(seconds=30))["status"] == OK
    stalled = check_ops_admission(approvals, now + timedelta(seconds=90))
    assert stalled["status"] == DEGRADED and "ops reconciler" in stalled["reason"]
    assert check_ops_admission(approvals, now + timedelta(seconds=601))["status"] == DOWN
    ApprovedOpsAdmission(approvals, system.service).admit_due()
    assert check_ops_admission(approvals, now + timedelta(seconds=601))["status"] == OK


def test_admission_health_without_a_store_is_down():
    assert check_ops_admission(None)["status"] == DOWN


def test_tool_registry_health(monkeypatch):
    from llm_bawt.service import health_checks

    monkeypatch.setattr(health_checks, "tools_registration_complete", lambda: False)
    assert check_mcp_tool_registry()["status"] == DEGRADED
    monkeypatch.setattr(health_checks, "tools_registration_complete", lambda: True)
    assert check_mcp_tool_registry()["status"] == OK


# ---- M6: no lazy imports on the admission / ops_run approval paths ---------

# The functions an approved ops_run passes through after process startup. The
# unit-tier guard checks them directly; the cold-interpreter test below checks
# the whole transitive path (it needs >1s of imports, so it runs offline-integration).
_PATH_FUNCTIONS = {
    "src/llm_bawt/ops/approved_admission.py": None,  # the whole module
    "src/llm_bawt/ops/reconciler.py": {"tick", "run"},
    "src/llm_bawt/service/routes/approval_policies.py": {
        "resolve_approval", "_resolve_mcp_request", "_stored_mcp_resolution",
        "_subscriber", "_mcp_result_is_error", "resolve_mcp_card"},
    # TASK-1609: a grant holder's decision takes the same path.
    "src/llm_bawt/service/routes/ops_approvals.py": None,  # the whole module
    "src/llm_bawt/service/approval_events.py": {"publish_approval_resolved"},
    "src/llm_bawt/service/approval_continuations.py": {
        "dispatch_due_continuations_once", "_terminal_ops_result", "_admission_card_result",
        "_persist_terminal_ops_tool_result", "continuation_payload_from_row"},
    "src/llm_bawt/service/tool_call_store.py": {"mark_approval_status", "resolve_approval_result"},
    # The outbox's production ops-service lookup (Loopy r1 H2).
    "src/llm_bawt/service/dependencies.py": {
        "get_ops_service", "get_ops_store", "get_tool_approval_policy_store", "_get_or_build_store"},
    "src/llm_bawt/approval_mcp_store.py": {"execution_fence", "complete_mcp_execution", "claim_mcp_execution"},
    "src/llm_bawt/ops/store.py": {"create_job"},
}


def test_ops_run_approval_path_functions_have_no_local_imports():
    import ast

    offenders = []
    for relative, names in _PATH_FUNCTIONS.items():
        tree = ast.parse((REPO / relative).read_text())
        functions = [node for node in ast.walk(tree)
                     if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                     and (names is None or node.name in names)]
        if names is not None:
            assert {f.name for f in functions} >= names, (relative, names - {f.name for f in functions})
        for function in functions:
            offenders += [f"{relative}:{node.lineno} {function.name}"
                          for node in ast.walk(function) if isinstance(node, (ast.Import, ast.ImportFrom))]
    assert offenders == []

_IMPORT_INVARIANT = textwrap.dedent('''
    import asyncio, json, sys, tempfile
    from types import SimpleNamespace
    from pathlib import Path
    from sqlmodel import create_engine

    # Everything the reconciler and app import at startup (build_reconciler,
    # the resolve route and the continuation outbox), and nothing more.
    from agent_bridge.mcp_call_context import canonical_invocation_hash
    from llm_bawt.approval_policy_store import ToolApprovalPolicyStore
    from llm_bawt.approval_policies import REQ_APPROVED
    from llm_bawt.utils.config import Config
    from llm_bawt.ops.approved_admission import ApprovedOpsAdmission
    from llm_bawt.ops.reconciler import OpsReconciler
    from llm_bawt.ops.store import OpsStore
    from llm_bawt.ops.executor import DispatchResult, Executor, ReconcileResult
    from llm_bawt.ops.service import OpsService
    from llm_bawt.service.routes import approval_policies as routes
    from llm_bawt.service import approval_continuations as continuations
    from llm_bawt.service import dependencies

    class FakeExecutor(Executor):
        def kind(self): return "docker"
        def available(self): return True
        def dispatch(self, **kwargs): return DispatchResult("worker-" + kwargs["job_id"])
        def reconcile(self, **kwargs): return ReconcileResult(None)

    engine = create_engine("sqlite:///" + str(Path(tempfile.mkdtemp()) / "ops.sqlite"),
                           connect_args={"check_same_thread": False})
    store = OpsStore(None, engine=engine)
    service = OpsService(store, executor=FakeExecutor())
    store.create_operation({"slug": "demo", "enabled": True,
        "command_script": '{"action":"restart","container_name":"app"}',
        "args_schema_json": {"type": "object", "properties": {}}}, actor="t")
    approvals = ToolApprovalPolicyStore(None, engine=engine)

    def approved(request_id, arguments):
        row = approvals.record_mcp_request(
            request_id=request_id, tool_use_id="toolu-" + request_id, mcp_server="bawthub",
            bot_id="bot", user_id="user", turn_id="turn-1", trigger_message_id="m",
            session_key="bot:user", backend="claude-code", tool_name="ops_run",
            tool_arguments=arguments, subject="s", grant_key="g", policy_id=None,
            severity="high", prompt="p",
            invocation_hash=canonical_invocation_hash("ops_run", arguments),
            continuation_capable=True,
            operations_snapshot=service.prepare_invocation("demo", {}),
            caller_context_json=json.dumps({"session_id": "s", "turn_id": "turn-1",
                "trigger_message_id": "m", "bot_id": "bot", "user_id": "user",
                "issued_at": 1, "agent_request_id": "a", "session_key": "bot:user",
                "backend": "claude-code", "tool_use_id": "toolu-" + request_id}))
        return row

    ok = approved("ok", {"operation": "demo", "args": {}})
    missing = approved("missing", {"operation": "nope", "args": {}})
    before = {m for m in sys.modules if m.split(".")[0] in ("llm_bawt", "agent_bridge")}

    # App: the resolve route records the decision only (new and repeated
    # approve), marks the card and fans out, all through the real handler.
    routes._store = lambda: approvals
    routes.get_turn_log_store = lambda: SimpleNamespace(engine=engine)
    for row in (ok, missing, ok):
        asyncio.run(routes.resolve_approval(row.id, routes.ResolveRequest(decision="approve")))
    # Reconciler: one tick admits both (one job, one dispatch failure).
    OpsReconciler(service, admission=ApprovedOpsAdmission(approvals, service)).tick()
    assert approvals.get_request(ok.id).result_json and approvals.get_request(missing.id).result_json
    # App: the continuation outbox reads the ops receipt and builds its payload.
    # The production lookup: no injected ops service, so the outbox resolves
    # it through dependencies.get_ops_service (cache seeded, not built).
    config = object()
    dependencies._store_cache[("ops_service", id(config))] = service
    host = SimpleNamespace(config=config)
    for row in (ok, missing):
        current = approvals.get_request(row.id)
        continuations._terminal_ops_result(host, current)
        continuations._admission_card_result(current)
        routes._stored_mcp_resolution(current)

    after = {m for m in sys.modules if m.split(".")[0] in ("llm_bawt", "agent_bridge")}
    print(json.dumps(sorted(after - before)))
''')


@pytest.mark.integration
@pytest.mark.offline_integration
def test_admission_and_ops_run_approval_paths_import_nothing_new():
    """M6: a lazy import on these paths reads the live bind-mounted tree into a
    process holding older modules: the incident this task removes."""
    completed = subprocess.run(
        [sys.executable, "-c", _IMPORT_INVARIANT], cwd=REPO, capture_output=True,
        text=True, timeout=120,
        env={**os.environ, "LLM_BAWT_ENV_FILE": "/dev/null"},
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    assert json.loads(completed.stdout.strip().splitlines()[-1]) == []
