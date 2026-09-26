from __future__ import annotations

import asyncio
import json

import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import create_engine

from agent_bridge.mcp_call_context import canonical_invocation_hash
from llm_bawt.approval_policies import (
    CONT_DELIVERED,
    CONT_PENDING,
    REQ_APPROVED,
    ToolApprovalPolicyStore,
)
from llm_bawt.service.approval_continuations import (
    MCP_RESULT_ENVELOPE_PREFIX,
    _terminal_ops_result,
    dispatch_due_continuations_once,
    dispatch_mcp_result_continuation,
)


def _store():
    store = object.__new__(ToolApprovalPolicyStore)
    store.engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    store._ensure_tables_exist()
    return store


def _ready_row(store, *, tool_name="ops_run", result=None):
    args = {"operation": "llm-bawt.restart-app", "args": {}}
    caller_context = {
        "session_id": "session-1", "turn_id": "turn-1",
        "trigger_message_id": "message-1", "bot_id": "snark",
        "user_id": "nick", "issued_at": 1,
        "agent_request_id": "agent-request-1", "session_key": "snark:nick",
        "backend": "claude-code", "tool_use_id": "toolu-1",
    }
    row = store.record_mcp_request(
        request_id="req-mcp-1",
        tool_use_id="toolu-1",
        mcp_server="bawthub",
        bot_id="snark",
        user_id="nick",
        turn_id="turn-1",
        trigger_message_id="message-1",
        session_key="snark:nick",
        backend="claude-code",
        tool_name=tool_name,
        tool_arguments=args,
        subject="operation=llm-bawt.restart-app args={}",
        grant_key="grant",
        policy_id="policy",
        severity="medium",
        prompt="Approve?",
        invocation_hash=canonical_invocation_hash("ops_run", args),
        continuation_capable=True,
        caller_context_json=json.dumps(caller_context),
    )
    store.resolve_request(row.id, status=REQ_APPROVED)
    store.claim_mcp_execution(row.id)
    store.complete_mcp_execution(
        row.id,
        result_json=json.dumps(result or {"job_id": "job-1", "state": "queued"}),
        is_error=False,
    )
    store.enqueue_continuation(row.id)
    return store.claim_continuation(row.id)


class FakeService:
    def __init__(self, error=None):
        self.requests = []
        self.error = error
        self.config = None

    def get_memory_client(self, bot_id, user_id):
        from types import SimpleNamespace
        return SimpleNamespace(get_session=lambda session_id: {
            "id": session_id, "bot_id": bot_id, "user_id": user_id,
            "status": "active", "session_metadata": {},
        })

    async def chat_completion_stream(self, request):
        self.requests.append(request)
        if self.error:
            raise self.error
        yield "data: [DONE]\n\n"


class FakeOps:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def get_job_status(self, job_id, **kwargs):
        self.calls.append((job_id, kwargs))
        return self.result


class OpsServiceHost(FakeService):
    def __init__(self, result):
        super().__init__()
        self._ops_service = FakeOps(result)


def test_outbox_does_not_claim_ops_continuation_before_job_is_terminal(monkeypatch):
    store = _store()
    row = _ready_row(store, result={"job_id": "job-1", "state": "accepted"})
    # Put the fixture back into a due, unclaimed outbox state.
    store.mark_continuation_failed(
        row.id, error="fixture reset", backoff_seconds=0,
        claim_token=row.continuation_claim_token,
    )
    service = OpsServiceHost({"job_id": "job-1", "state": "running", "terminal": False})
    monkeypatch.setattr(
        "llm_bawt.service.approval_continuations.recover_approval_requests",
        lambda _store: asyncio.sleep(0),
    )

    asyncio.run(dispatch_due_continuations_once(service, store))

    current = store.get_request(row.id)
    assert current.continuation_state == CONT_PENDING
    assert service.requests == []


def test_outbox_dispatches_ops_continuation_once_after_terminal_receipt(monkeypatch):
    store = _store()
    row = _ready_row(store, result={"job_id": "job-1", "state": "accepted"})
    store.mark_continuation_failed(
        row.id, error="fixture reset", backoff_seconds=0,
        claim_token=row.continuation_claim_token,
    )
    terminal = {
        "job_id": "job-1",
        "operation": "llm-bawt.restart-app",
        "state": "succeeded",
        "terminal": True,
        "exit_code": 0,
    }
    service = OpsServiceHost(terminal)
    monkeypatch.setattr(
        "llm_bawt.service.approval_continuations.recover_approval_requests",
        lambda _store: asyncio.sleep(0),
    )

    asyncio.run(dispatch_due_continuations_once(service, store))
    asyncio.run(dispatch_due_continuations_once(service, store))

    current = store.get_request(row.id)
    assert current.continuation_state == CONT_DELIVERED
    assert len(service.requests) == 1
    assert service.requests[0].continuation_payload.result == terminal


def test_ops_continuation_waits_for_terminal_job_receipt():
    store = _store()
    row = _ready_row(store, result={"job_id": "job-1", "state": "accepted"})
    service = OpsServiceHost({"job_id": "job-1", "state": "running", "terminal": False})

    assert _terminal_ops_result(service, row) is None
    assert service._ops_service.calls == [(
        "job-1",
        {"output_tail_bytes": 4096, "reconcile_if_active": True},
    )]


def test_terminal_ops_receipt_replaces_initial_accepted_result():
    store = _store()
    row = _ready_row(store, result={"job_id": "job-1", "state": "accepted"})
    terminal = {
        "job_id": "job-1",
        "operation": "llm-bawt.restart-app",
        "state": "succeeded",
        "terminal": True,
        "exit_code": 0,
    }
    service = OpsServiceHost(terminal)

    final = _terminal_ops_result(service, row)
    assert final == terminal
    asyncio.run(
        dispatch_mcp_result_continuation(
            service, store, row, result_override=final,
        )
    )
    assert service.requests[0].continuation_payload.result == terminal
    assert service.requests[0].continuation_payload.is_error is False
    assert '"state": "succeeded"' in service.requests[0].messages[0].content


def test_failed_terminal_ops_receipt_marks_continuation_result_error():
    store = _store()
    row = _ready_row(store, result={"job_id": "job-1", "state": "accepted"})
    failed = {
        "job_id": "job-1",
        "operation": "llm-bawt.restart-app",
        "state": "failed",
        "terminal": True,
        "error_text": "restart refused",
    }
    service = OpsServiceHost(failed)

    asyncio.run(
        dispatch_mcp_result_continuation(
            service, store, row, result_override=failed,
        )
    )
    assert service.requests[0].continuation_payload.is_error is True


def test_dispatch_delivers_actual_result_envelope_and_marks_done():
    store = _store()
    row = _ready_row(store)
    service = FakeService()

    asyncio.run(dispatch_mcp_result_continuation(service, store, row))

    request = service.requests[0]
    prompt = request.messages[0].content
    assert prompt.startswith(MCP_RESULT_ENVELOPE_PREFIX)
    assert '"job_id": "job-1"' in prompt
    assert "Do not retry or re-issue the tool" in prompt
    assert request.parent_turn_id == "turn-1"
    assert request.continuation_payload.approval_request_id == "req-mcp-1"
    assert request.continuation_payload.result == {"job_id": "job-1", "state": "queued"}
    persisted = store.get_request(row.id)
    assert persisted.continuation_state == CONT_DELIVERED
    assert persisted.continuation_delivered_at is not None


def test_dispatch_requires_persisted_success_before_ack():
    from types import SimpleNamespace
    from llm_bawt.service.approval_continuations import _continuation_identity

    store = _store()
    row = _ready_row(store)
    identity = _continuation_identity(row)

    class PersistedService(FakeService):
        def __init__(self):
            super().__init__()
            self.turn = None
            self._turn_log_store = SimpleNamespace(get_turn=lambda key: self.turn)

        async def chat_completion_stream(self, request):
            assert request.user_message_id == identity["user_message_id"]
            assert request.assistant_message_id == identity["assistant_message_id"]
            assert len(request.user_message_id) == len(request.assistant_message_id) == 36
            self.turn = SimpleNamespace(ended_at=1, status="ok", error_text=None)
            yield "data: [DONE]\n\n"

    service = PersistedService()
    asyncio.run(dispatch_mcp_result_continuation(service, store, row))
    assert store.get_request(row.id).continuation_state == CONT_DELIVERED


def test_existing_failed_turn_is_never_replayed():
    from types import SimpleNamespace

    store = _store()
    row = _ready_row(store)
    service = FakeService()
    service._turn_log_store = SimpleNamespace(get_turn=lambda key: SimpleNamespace(
        ended_at=1, status="error", error_text="prior persistence failure"))
    with pytest.raises(RuntimeError, match="manual reconciliation"):
        asyncio.run(dispatch_mcp_result_continuation(service, store, row))
    assert not service.requests
    assert store.get_request(row.id).continuation_state != CONT_DELIVERED


def test_dispatch_failure_reschedules_for_retry():
    store = _store()
    row = _ready_row(store)
    service = FakeService(RuntimeError("bridge offline"))

    with pytest.raises(RuntimeError, match="bridge offline"):
        asyncio.run(dispatch_mcp_result_continuation(service, store, row))

    persisted = store.get_request(row.id)
    assert persisted.continuation_state == CONT_PENDING
    assert persisted.continuation_last_error == "bridge offline"
    assert persisted.continuation_next_attempt_at is not None


class ResetService(FakeService):
    def __init__(self, store):
        super().__init__()
        self._tool_approval_policy_store = store
        self.sessions = {
            "session-1": {"status": "archived", "session_metadata": {
                "reset_successor_id": "session-2", "agent_session_keys": {"claude_code": "old-huge-sdk"}}},
            "session-2": {"status": "active", "session_metadata": {
                "reset_predecessor_ids": ["session-1"], "agent_session_keys": {"claude_code": "fresh-sdk"}}},
        }

    def get_memory_client(self, bot_id, user_id):
        from types import SimpleNamespace
        assert (bot_id, user_id) == ("snark", "nick")
        return SimpleNamespace(get_session=lambda id: {
            "id": id, "bot_id": bot_id, "user_id": user_id, **self.sessions[id],
        })

    async def chat_completion_stream(self, request):
        from llm_bawt.service.approval_continuations import validate_approval_continuation_claim
        validate_approval_continuation_claim(self, request, request._internal_approval_claim)
        async for chunk in super().chat_completion_stream(request):
            yield chunk


def test_pending_restart_result_follows_new_without_rebinding_original_authority():
    store = _store()
    row = _ready_row(store)
    original_context = row.caller_context_json
    service = ResetService(store)
    asyncio.run(dispatch_mcp_result_continuation(service, store, row))
    request = service.requests[0]
    assert request.session_id == "session-2"
    assert request.parent_turn_id == row.turn_id
    assert request.continuation_payload.original_tool_use_id == row.tool_use_id
    assert request.continuation_payload.result == {"job_id": "job-1", "state": "queued"}
    assert "/new" in request.messages[0].content
    assert "old-huge-sdk" not in request.model_dump_json()
    saved = store.get_request(row.id)
    assert saved.caller_context_json == original_context
    assert saved.execution_attempts == 1
    assert saved.continuation_state == CONT_DELIVERED


def test_reset_again_between_dispatch_and_validation_rejects_stale_target():
    from llm_bawt.service.approval_continuations import validate_approval_continuation_claim
    store = _store()
    row = _ready_row(store)

    class RacingResetService(ResetService):
        async def chat_completion_stream(self, request):
            assert request.session_id == "session-2"
            self.sessions["session-2"]["session_metadata"]["reset_successor_id"] = "session-3"
            self.sessions["session-3"] = {"status": "active", "session_metadata": {"reset_predecessor_ids": ["session-2"]}}
            validate_approval_continuation_claim(self, request, request._internal_approval_claim)
            yield "data: [DONE]\n\n"

    with pytest.raises(ValueError, match="Invalid or stale"):
        asyncio.run(dispatch_mcp_result_continuation(RacingResetService(store), store, row))
    assert store.get_request(row.id).continuation_state == CONT_PENDING


def test_already_started_continuation_is_not_replayed_in_another_session():
    from types import SimpleNamespace
    store = _store()
    row = _ready_row(store)
    service = ResetService(store)
    service._turn_log_store = SimpleNamespace(get_turn=lambda id: SimpleNamespace(ended_at=None))
    with pytest.raises(RuntimeError, match="already-started"):
        asyncio.run(dispatch_mcp_result_continuation(service, store, row))
    assert service.requests == []


def test_new_target_retry_retains_deterministic_identity_and_original_execution():
    from datetime import datetime, timedelta, timezone
    from sqlmodel import Session
    from llm_bawt.approval_policies import ToolApprovalRequest
    store = _store()
    row = _ready_row(store)
    service = ResetService(store)
    service.error = RuntimeError("offline before dispatch")
    with pytest.raises(RuntimeError, match="offline"):
        asyncio.run(dispatch_mcp_result_continuation(service, store, row))
    with Session(store.engine) as session:
        saved = session.get(ToolApprovalRequest, row.id)
        saved.continuation_next_attempt_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        session.add(saved)
        session.commit()
    retried = store.claim_continuation(row.id)
    service.error = None
    asyncio.run(dispatch_mcp_result_continuation(service, store, retried))
    first, second = service.requests
    assert first.session_id == second.session_id == "session-2"
    assert first.user_message_id == second.user_message_id
    assert first.inter_bot_turn_id == second.inter_bot_turn_id
    assert store.get_request(row.id).execution_attempts == 1
