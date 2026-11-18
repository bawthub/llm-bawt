"""Ops-owned creation of the existing durable, continuation-capable MCP card."""
from __future__ import annotations

import json

from agent_bridge.approval import derive_subject

from ..approval_ops_supersede import singleton_target
from .approval_prompt import approval_text
from .approved_admission import ops_run_arguments_valid
from .gpu_guard import snapshot_handoff_only
from .invocation_attestation import verify
from .service import OpsDispatchError


def request_invocation(ops, approvals, envelope) -> dict:
    invocation = verify(envelope)
    arguments = invocation.arguments
    if not ops_run_arguments_valid(arguments):
        raise OpsDispatchError("args_invalid", "ops_run accepts only operation, args and idempotency_key")
    existing = approvals.get_request(invocation.request_id)
    if existing is not None:
        if (existing.tool_name != "ops_run" or existing.invocation_hash != invocation.invocation_hash
                or json.loads(existing.caller_context_json) != invocation.caller.model_dump()
                or json.loads(existing.tool_arguments_json) != arguments):
            raise OpsDispatchError("idempotency_conflict", "request belongs to a different invocation")
        # A retry is a read, even if the catalog or target changed since creation.
        snapshot = json.loads(existing.operations_snapshot_json)
        return _projection(existing, False, arguments, snapshot)
    operation = ops.store.get_operation_by_slug(arguments["operation"])
    if operation is None:
        raise OpsDispatchError("operation_not_found", "unknown operation")
    if not operation.enabled or operation.soft_deleted_at is not None:
        raise OpsDispatchError("operation_disabled", "operation is disabled or deleted")
    if operation.approval_required is False:
        # Direct admission must independently pin/check the current false flag.
        return {"status": "catalog_allow"}
    snapshot = ops.prepare_invocation(arguments["operation"], arguments.get("args") or {})
    if service := snapshot_handoff_only(snapshot):
        raise OpsDispatchError("gpu_handoff_required",
                               f"only a GPU handoff can run service {service!r}; no approval was created")
    caller = invocation.caller
    row, created = approvals.record_mcp_request(
        request_id=invocation.request_id, tool_use_id=caller.tool_use_id,
        mcp_server="bawthub", bot_id=caller.bot_id, user_id=caller.user_id,
        turn_id=caller.turn_id, backend=caller.backend, tool_name="ops_run",
        tool_arguments=arguments, subject=derive_subject("ops_run", arguments, None),
        grant_key=invocation.invocation_hash, policy_id=None,
        severity=snapshot["operation"].get("risk_level") or "medium",
        prompt=approval_text(snapshot), invocation_hash=invocation.invocation_hash,
        operations_snapshot=snapshot,
        caller_context_json=json.dumps(caller.model_dump(), ensure_ascii=False, sort_keys=True),
        continuation_capable=True, trigger_message_id=caller.trigger_message_id,
        session_key=caller.session_key, with_created=True,
    )
    return _projection(row, created, arguments, snapshot)


def _projection(row, created, arguments, snapshot) -> dict:
    # The committed row, not a proposed card. The app publishes its projection.
    return {
        "status": "approval_required", "created": created,
        "singleton": singleton_target(snapshot) is not None,
        "approval_request_id": row.id, "tool": row.tool_name, "subject": row.subject,
        "continuation_capable": bool(row.continuation_capable),
        "card": {
            "_type": "tool_approval_required", "request_id": row.id,
            "tool_use_id": row.tool_use_id, "turn_id": row.turn_id,
            "trigger_message_id": row.trigger_message_id, "bot_id": row.bot_id,
            "user_id": row.user_id, "tool_name": row.tool_name,
            "arguments": arguments, "subject": row.subject,
            "label": snapshot["operation"].get("title") or snapshot["operation"]["slug"],
            "prompt": row.prompt, "severity": row.severity, "policy_id": row.policy_id,
            "session_key": row.session_key or "", "provider": row.backend,
            "request_kind": "mcp", "continuation_capable": bool(row.continuation_capable),
        },
    }
