"""Replace pending singleton ops cards, never approved or running work."""
from __future__ import annotations

import json

from sqlalchemy import inspect, update
from sqlmodel import Session, select

from .approval_events import append_approval_transition
from .approval_models import (
    CONT_NOT_NEEDED, CONT_PENDING, EXEC_SKIPPED, KIND_MCP, OPS_RUN_TOOL,
    REQ_PENDING, REQ_SUPERSEDED, ToolApprovalRequest, _utcnow,
)
from .service.commit_wake import notify_in_transaction


def singleton_target(snapshot):
    """Only known, single-target specs with an explicitly singleton executor.

    Arguments such as release_task, expected_sha and idempotency_key describe
    the invocation, not its target. Different restart targets remain separate.
    Unknown/multi-target actions do not opt into replacement.
    """
    if not isinstance(snapshot, dict):
        return None
    operation, spec, execution, args = (
        snapshot.get(key) or {} for key in ("operation", "spec", "execution", "resolved_args")
    )
    if not all(isinstance(part, dict) for part in (operation, spec, execution, args)):
        return None
    if (type(execution.get("max_concurrent")) is not int or execution["max_concurrent"] != 1
            or not isinstance(operation.get("slug"), str) or not operation["slug"]):
        return None
    if spec.get("action") == "release_orchestrate" and execution.get("executor_kind") == "release":
        target = ("release", spec.get("deploy_operation"))
    elif spec.get("action") == "restart":
        container = args.get(spec["container_name_from_arg"]) if spec.get("container_name_from_arg") else spec.get("container_name")
        service = args.get(spec["compose_service_from_arg"]) if spec.get("compose_service_from_arg") else spec.get("compose_service")
        if container:
            target = ("container", container)
        elif spec.get("compose_project") and service:
            target = ("compose", spec["compose_project"], service)
        else:
            return None
    else:
        return None
    if not all(isinstance(value, str) and value for value in target):
        return None
    return (operation["slug"], execution.get("target_host"), *target)


def lock_singleton(session, tool_name, snapshot):
    """Use the catalog row lock also taken by edits and approval decisions.

    Concurrent singleton replacement requires the ops catalog. Standalone
    approval-only fixtures without that table can exercise sequential state
    transitions, but do not provide the catalog-backed serialization contract.
    """
    target = singleton_target(snapshot) if tool_name == OPS_RUN_TOOL else None
    if target is None:
        return None
    if inspect(session.connection()).has_table("ops_operations"):
        from .ops.models import OpsOperation
        from .ops.operation_revision import changed_reason

        operation = session.exec(select(OpsOperation).where(
            OpsOperation.slug == target[0],
        ).with_for_update()).first()
        if operation is None or changed_reason(snapshot["operation"], operation.to_api()):
            return None  # late admission still refuses the obsolete snapshot
    return target


def supersede_pending(session, row, target, events=None):
    """Same transaction as insert; a concurrent approval's pending CAS wins once."""
    if target is None:
        return
    pending = session.exec(select(ToolApprovalRequest).where(
        ToolApprovalRequest.request_kind == KIND_MCP,
        ToolApprovalRequest.tool_name == OPS_RUN_TOOL,
        ToolApprovalRequest.user_id == row.user_id,
        ToolApprovalRequest.status == REQ_PENDING,
        ToolApprovalRequest.id != row.id,
    )).all()
    for previous in pending:
        try:
            previous_target = singleton_target(json.loads(previous.operations_snapshot_json or "null"))
        except (TypeError, ValueError):
            continue
        if previous_target != target:
            continue
        now = _utcnow()
        message = f"Superseded by {row.id}; no action submitted."
        changed = session.exec(update(ToolApprovalRequest).where(
            ToolApprovalRequest.id == previous.id, ToolApprovalRequest.status == REQ_PENDING,
        ).values(
            status=REQ_SUPERSEDED, resolved_at=now, resolved_by="system:singleton_replaced",
            resolution_message=message, execution_state=EXEC_SKIPPED,
            execution_finished_at=now, result_is_error=False,
            result_json=json.dumps({"status": REQ_SUPERSEDED, "message": message, "superseded_by": row.id}),
            continuation_owner="server",
            continuation_state=CONT_PENDING if previous.continuation_capable else CONT_NOT_NEEDED,
            continuation_next_attempt_at=now if previous.continuation_capable else None,
        ))
        if changed.rowcount == 1:
            session.refresh(previous)
            append_approval_transition(events, session.connection(), previous)
            notify_in_transaction(session.connection(), "continuation")


def superseded_by(store, request_id):
    """The committed replacements to project to live cards (no new decision)."""
    with Session(store.engine) as session:
        return session.exec(select(ToolApprovalRequest).where(
            ToolApprovalRequest.status == REQ_SUPERSEDED,
            ToolApprovalRequest.resolved_by == "system:singleton_replaced",
            ToolApprovalRequest.resolution_message == f"Superseded by {request_id}; no action submitted.",
        )).all()
