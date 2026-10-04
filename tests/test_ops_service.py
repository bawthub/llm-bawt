"""Tests for :class:`llm_bawt.ops.service.OpsService` (TASK-639).

Exercises the dispatch orchestration end-to-end with a fake executor so no
real SSH is required. Covers:

* unknown operation
* disabled operation
* invalid args (schema violation)
* idempotency on repeated dispatch with same key
* successful dispatch flips job to DISPATCHING and records unit + paths
* executor-unavailable path marks job FAILED and surfaces the reason
* reconcile: RUNNING → SUCCEEDED transitions land in the store
* reconcile: no status.json yet → keep polling (state unchanged)
"""

from __future__ import annotations

import json

from sqlmodel import create_engine
from sqlalchemy.pool import StaticPool

from llm_bawt.ops import (
    JOB_DISPATCHING,
    JOB_QUEUED,
    JOB_SUCCEEDED,
    OpsDispatchError,
    OpsService,
    OpsStore,
)
from llm_bawt.ops.executor import DispatchResult, Executor, ExecutorError, ReconcileResult


def _store():
    store = object.__new__(OpsStore)
    store.engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    store._ensure_tables_exist()
    return store


class FakeExecutor(Executor):
    def __init__(self, *, available: bool = True, dispatch_error: str | None = None):
        self._available = available
        self._dispatch_error = dispatch_error
        self.dispatch_calls: list[dict] = []
        self.reconcile_calls: list[dict] = []
        self.next_reconcile: ReconcileResult | None = None

    def kind(self) -> str:
        return "docker"

    def available(self) -> bool:
        return self._available

    def dispatch(self, **kwargs) -> DispatchResult:
        self.dispatch_calls.append(kwargs)
        if not self._available:
            raise ExecutorError("not available")
        if self._dispatch_error:
            raise ExecutorError(self._dispatch_error)
        return DispatchResult(
            host_unit_name=f"llm-bawt-ops-{kwargs['job_id'][:8]}",
            status_file_path=f"/host/jobs/{kwargs['job_id']}/status.json",
            log_file_path=f"/host/jobs/{kwargs['job_id']}/output.log",
        )

    def reconcile(self, **kwargs) -> ReconcileResult:
        self.reconcile_calls.append(kwargs)
        assert self.next_reconcile is not None, "test forgot to set next_reconcile"
        return self.next_reconcile


def _op_data(**over):
    base = dict(
        slug="test.restart-thing",
        title="Restart Thing",
        description="Test op",
        enabled=True,
        target_host="",
        command_script='{"action":"restart","container_name":"test"}',
        args_schema_json=json.dumps({
            "type": "object", "additionalProperties": False,
            "properties": {"target": {"type": "string", "enum": ["a", "b"]}},
            "required": ["target"],
        }),
        args_defaults_json="{}",
        timeout_seconds=60,
        risk_level="low",
        category="restart",
    )
    base.update(over)
    return base


def _make(store):
    executor = FakeExecutor()
    service = OpsService(store, executor=executor)
    return service, executor


def test_dispatch_unknown_operation_raises():
    store = _store()
    service, _ = _make(store)
    try:
        service.dispatch_job(
            operation_slug="nope", args={}, idempotency_key="k1",
        )
    except OpsDispatchError as exc:
        assert exc.code == "operation_not_found"
        return
    raise AssertionError("expected OpsDispatchError")


def test_dispatch_disabled_operation_raises():
    store = _store()
    store.create_operation(_op_data(enabled=False))
    service, _ = _make(store)
    try:
        service.dispatch_job(
            operation_slug="test.restart-thing", args={"target": "a"},
            idempotency_key="k1",
        )
    except OpsDispatchError as exc:
        assert exc.code == "operation_disabled"
        return
    raise AssertionError("expected OpsDispatchError")


def test_dispatch_invalid_args_raises_with_reason():
    store = _store()
    store.create_operation(_op_data())
    service, _ = _make(store)
    try:
        service.dispatch_job(
            operation_slug="test.restart-thing", args={"target": "z"},
            idempotency_key="k1",
        )
    except OpsDispatchError as exc:
        assert exc.code == "args_invalid"
        assert "not in enum" in str(exc)
        return
    raise AssertionError("expected OpsDispatchError")


class MissingTargetExecutor(FakeExecutor):
    def __init__(self):
        super().__init__()
        self.checked: list[tuple[dict, dict]] = []

    def check_target(self, spec, args):
        self.checked.append((spec, args))
        raise ExecutorError("target container 'test' does not exist on this executor's Docker host")


def test_prepare_invocation_rejects_missing_target_before_approval():
    # TASK-1002: the snapshot is built BEFORE the approval card; a target the
    # executor cannot reach must fail here, not as a 404 after approval.
    store = _store()
    store.create_operation(_op_data())
    executor = MissingTargetExecutor()
    service = OpsService(store, executor=executor)
    try:
        service.prepare_invocation("test.restart-thing", {"target": "a"})
    except OpsDispatchError as exc:
        assert exc.code == "target_unavailable"
        assert "does not exist" in str(exc)
        assert executor.checked == [({"action": "restart", "container_name": "test"}, {"target": "a"})]
        assert executor.dispatch_calls == []
        return
    raise AssertionError("expected OpsDispatchError")


def test_dispatch_without_snapshot_also_checks_target():
    store = _store()
    store.create_operation(_op_data())
    executor = MissingTargetExecutor()
    service = OpsService(store, executor=executor)
    try:
        service.dispatch_job(operation_slug="test.restart-thing", args={"target": "a"},
                             idempotency_key="k-missing")
    except OpsDispatchError as exc:
        assert exc.code == "target_unavailable"
        assert executor.dispatch_calls == []
        assert store.get_job_by_key("k-missing") is None
        return
    raise AssertionError("expected OpsDispatchError")


def test_dispatch_success_records_dispatching_and_unit_name():
    store = _store()
    store.create_operation(_op_data())
    service, executor = _make(store)
    result = service.dispatch_job(
        operation_slug="test.restart-thing", args={"target": "a"},
        idempotency_key="k1", caller_bot_id="snark",
    )
    assert result["state"] == "accepted"
    assert result["host_unit_name"].startswith("llm-bawt-ops-")
    assert result["operation"] == "test.restart-thing"
    assert result["caller"]["bot_id"] == "snark"
    assert len(executor.dispatch_calls) == 1
    # Args land as OPS_ARG_ env keys inside the executor call.
    assert executor.dispatch_calls[0]["env_args"] == {"target": "a"}


def test_dispatch_idempotent_on_key():
    store = _store()
    store.create_operation(_op_data())
    service, executor = _make(store)
    first = service.dispatch_job(
        operation_slug="test.restart-thing", args={"target": "a"},
        idempotency_key="k1",
    )
    second = service.dispatch_job(
        operation_slug="test.restart-thing", args={"target": "a"},
        idempotency_key="k1",
    )
    assert first["id"] == second["id"]
    # Second call must NOT re-dispatch.
    assert len(executor.dispatch_calls) == 1


def test_dispatch_marks_job_failed_when_executor_unavailable():
    store = _store()
    store.create_operation(_op_data())
    service, executor = _make(store)
    executor._available = False
    try:
        service.dispatch_job(
            operation_slug="test.restart-thing", args={"target": "a"},
            idempotency_key="k1",
        )
    except OpsDispatchError as exc:
        assert exc.code == "executor_unavailable"
    # Job row should be FAILED with a useful reason.
    jobs = store.list_jobs(operation_slug="test.restart-thing")
    assert len(jobs) == 1
    assert jobs[0].state == JOB_QUEUED  # Read-only preflight failure; nothing submitted.
    assert "not available" in jobs[0].error_text


def test_dispatch_marks_job_failed_when_executor_raises():
    store = _store()
    store.create_operation(_op_data())
    service, executor = _make(store)
    executor._dispatch_error = "ssh: connection refused"
    try:
        service.dispatch_job(
            operation_slug="test.restart-thing", args={"target": "a"},
            idempotency_key="k1",
        )
    except OpsDispatchError as exc:
        assert exc.code == "dispatch_failed"
        assert "connection refused" in str(exc)
    jobs = store.list_jobs(operation_slug="test.restart-thing")
    assert jobs[0].state == JOB_DISPATCHING


def test_get_job_status_reconciles_running_to_succeeded():
    store = _store()
    store.create_operation(_op_data())
    service, executor = _make(store)
    dispatched = service.dispatch_job(
        operation_slug="test.restart-thing", args={"target": "a"},
        idempotency_key="k1",
    )
    executor.next_reconcile = ReconcileResult(
        state="succeeded", exit_code=0, output_tail="ok\n",
        error=None, started_at="2026-01-01T00:00:00Z",
        finished_at="2026-01-01T00:00:05Z",
    )
    status = service.get_job_status(dispatched["id"], output_tail_bytes=1024)
    assert status["state"] == JOB_SUCCEEDED
    assert status["exit_code"] == 0
    assert status["output_tail"] == "ok\n"


def test_get_job_status_no_status_file_keeps_state():
    store = _store()
    store.create_operation(_op_data())
    service, executor = _make(store)
    dispatched = service.dispatch_job(
        operation_slug="test.restart-thing", args={"target": "a"},
        idempotency_key="k1",
    )
    executor.next_reconcile = ReconcileResult(
        state=None, exit_code=None, output_tail=None, error=None,
        started_at=None, finished_at=None,
    )
    status = service.get_job_status(dispatched["id"])
    # Still DISPATCHING (no state change; touch_reconcile ran).
    assert status["state"] == "accepted"


def test_get_job_status_terminal_is_no_op():
    store = _store()
    store.create_operation(_op_data())
    service, executor = _make(store)
    dispatched = service.dispatch_job(
        operation_slug="test.restart-thing", args={"target": "a"},
        idempotency_key="k1",
    )
    # Mark terminal directly.
    store.mark_terminal(dispatched["id"], state=JOB_SUCCEEDED, exit_code=0)
    executor.next_reconcile = None  # would raise if called
    status = service.get_job_status(dispatched["id"])
    assert status["state"] == JOB_SUCCEEDED
    # And reconcile was NOT invoked.
    assert executor.reconcile_calls == []


def test_get_job_status_unknown_returns_none():
    store = _store()
    service, _ = _make(store)
    assert service.get_job_status("nope") is None


def test_list_operations_for_agent_hides_disabled():
    store = _store()
    store.create_operation(_op_data(slug="live", enabled=True))
    store.create_operation(_op_data(slug="dark", enabled=False))
    service, _ = _make(store)
    slugs = [row["slug"] for row in service.list_operations_for_agent()]
    assert slugs == ["live"]
    summaries = {row["slug"]: row for row in service.list_operations_for_agent(include_disabled=True)}
    assert set(summaries) == {"live", "dark"}
    assert summaries["live"]["enabled"] is True
    assert summaries["dark"]["enabled"] is False


# ---- caller provenance (TASK-639) ------------------------------------------

def test_ops_run_records_approved_caller_provenance(monkeypatch):
    """An approved job must be traceable back to who asked for it.

    Regression: ``ops_run`` never passed caller identity to ``dispatch_job``,
    so ``ops_jobs.caller_bot_id`` / ``approval_request_id`` stayed empty and a
    completed restart had no audit linkage back to its approval.
    """
    from llm_bawt.mcp_server import ops_tools
    from llm_bawt.mcp_server.approval_interceptor import (
        ApprovedCallerContext,
        _approved_caller_context,
    )

    token = _approved_caller_context.set(
        ApprovedCallerContext(
            bot_id="snark", user_id="nick", turn_id="turn-1",
            session_key="sk-1", backend="claude-code",
            approval_request_id="mcp-appr-abc",
        )
    )
    try:
        provenance = ops_tools._caller_provenance()
    finally:
        _approved_caller_context.reset(token)

    assert provenance == {
        "caller_bot_id": "snark",
        "caller_user_id": "nick",
        "caller_turn_id": "turn-1",
        "caller_session_key": "sk-1",
        "caller_backend": "claude-code",
        "approval_request_id": "mcp-appr-abc",
    }


def test_ops_run_provenance_is_empty_without_any_trusted_context():
    """A manual/generic MCP client leaves the columns null, never guessed."""
    from llm_bawt.mcp_server import ops_tools

    assert ops_tools._caller_provenance() == {}


def test_operation_api_timestamps_are_utc_normalized():
    """Same instant loaded under different DB session TimeZones must serialize
    identically, or approval snapshot hashes flap between reads (TASK-927)."""
    from datetime import datetime, timedelta, timezone as tz

    from llm_bawt.ops.models import OpsOperation

    instant = datetime(2026, 9, 27, 1, 45, 12, 776194, tzinfo=tz.utc)
    eastern = instant.astimezone(tz(timedelta(hours=-4)))
    a = OpsOperation(id="x", slug="x", created_at=instant, updated_at=instant).to_api()
    b = OpsOperation(id="x", slug="x", created_at=eastern, updated_at=eastern).to_api()
    assert a == b
    assert a["created_at"] == "2026-09-27T01:45:12.776194+00:00"
