"""Invalidate pending ops cards inside the catalog edit's transaction."""
import json

from sqlalchemy import inspect, update
from sqlmodel import select

from .approval_events import append_approval_transition
from .approval_mcp_store import _continuation_id
from .approval_models import (
    CONT_NOT_NEEDED, CONT_PENDING, EXEC_FAILED, KIND_MCP, REQ_PENDING,
    REQ_SUPERSEDED, ToolApprovalRequest, _utcnow,
)
from .ops.operation_revision import changed_reason
from .service.commit_wake import notify_in_transaction


def invalidate_request(session, row, operation, events=None):
    try:
        approved = json.loads(row.operations_snapshot_json or "null")["operation"]
    except (TypeError, ValueError, KeyError):
        return  # malformed snapshots already fail closed at admission
    reason = changed_reason(approved, operation.to_api() if operation else None)
    if not reason:
        return
    now = _utcnow()
    values = dict(
        status=REQ_SUPERSEDED, resolved_at=now, resolved_by="system:operation_changed",
        resolution_message=reason, execution_state=EXEC_FAILED,
        execution_finished_at=now, execution_error=reason,
        result_json=json.dumps({"status": "failed", "error": reason}), result_is_error=True,
        continuation_owner="server",
        continuation_state=CONT_PENDING if row.continuation_capable else CONT_NOT_NEEDED,
        continuation_next_attempt_at=now if row.continuation_capable else None,
        continuation_id=row.continuation_id or _continuation_id(row.id),
    )
    result = session.exec(update(ToolApprovalRequest).where(
        ToolApprovalRequest.id == row.id, ToolApprovalRequest.status == REQ_PENDING,
    ).values(**values))
    if result.rowcount == 1:
        session.refresh(row)
        append_approval_transition(events, session.connection(), row)
        notify_in_transaction(session.connection(), "continuation")


def invalidate_pending_ops(session, operation, events=None):
    # A standalone ops store (including isolated tests) need not have approvals.
    if not inspect(session.connection()).has_table("tool_approval_requests"):
        return
    pending = session.exec(select(ToolApprovalRequest).where(
        ToolApprovalRequest.request_kind == KIND_MCP,
        ToolApprovalRequest.tool_name == "ops_run",
        ToolApprovalRequest.status == REQ_PENDING,
    )).all()
    for row in pending:
        try:
            approved = json.loads(row.operations_snapshot_json or "null")["operation"]
        except (TypeError, ValueError, KeyError):
            continue  # malformed snapshots already fail closed at admission
        if approved.get("slug") != operation.slug:
            continue
        invalidate_request(session, row, operation, events)
