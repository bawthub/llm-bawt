"""Authenticated BawtHub Run authorizes one frozen operation (TASK-1947).

Only the service-authenticated operator route calls this module. BawtHub binds
its existing verified-person manager before sending the user assertion; no
browser actor, bot, backend label or client snapshot establishes authority.
The ordinary LAN dispatch route remains a distinct, legacy build-only path.
"""
from __future__ import annotations

import hashlib
import json

from ..approval_models import KIND_ORCHESTRATION, REQ_APPROVED, RESOLVED_VIA_USER
from .gpu_guard import handoff_only_service
from .service import APPROVAL_ONLY_ACTIONS, APPROVAL_REQUIRED, OpsDispatchError, _spec_action
from .validation import canonical_json

OPERATOR_BACKEND = "bawthub-operator"
OPERATOR_TOOL = "ops_operator_run"


def _identity_problem(row, *, user_id: str, key: str, operation: str, args: dict) -> bool:
    return (row.request_kind != KIND_ORCHESTRATION or row.tool_name != OPERATOR_TOOL
            or row.backend != OPERATOR_BACKEND or row.bot_id != "ops"
            or row.user_id != user_id or row.turn_id != key
            or canonical_json(json.loads(row.tool_arguments_json)) != canonical_json(
                {"operation": operation, "args": args}))


def dispatch_operator_run(ops, approvals, *, user_id: str, operation: str,
                          args: dict, idempotency_key: str) -> dict:
    """Persist authenticated Run intent before dispatch; retries reuse it exactly.

    The approval record is server-owned and audited, never a tool/client grant.
    Its pending->approved CAS records the verified operator's Run decision, not
    another approval round. A crash between intent/decision/job is resumable
    under the same key with the original snapshot, not a fresh source/baseline.
    """
    digest = hashlib.sha256(canonical_json([user_id, idempotency_key]).encode()).hexdigest()
    key = f"operator:{digest}"
    request_id = f"operator-run-{digest}"
    row = approvals.get_request(request_id)
    if row is None:
        # Preserve the separate approval-card-only source-reconcile fence. A
        # manual Run cannot prepare its sealed source check or supply its MCP
        # creation fence; do not produce a doomed approved record for it.
        if _spec_action(ops.store.get_operation_by_slug(operation)) in APPROVAL_ONLY_ACTIONS:
            raise OpsDispatchError("approval_required", APPROVAL_REQUIRED)
        snapshot = ops.prepare_invocation(operation, args)
        if (service := handoff_only_service(snapshot["spec"], snapshot["resolved_args"])) is not None:
            # TASK-1707: only a GPU handoff can run it; record no doomed Run.
            raise OpsDispatchError("gpu_handoff_required", (
                f"this operation starts, restarts or stops the GPU voice service {service!r}; "
                "only a GPU handoff can run it"))
        row, _ =approvals.record_orchestration_request(
            request_id=request_id, bot_id="ops", user_id=user_id, turn_id=key,
            backend=OPERATOR_BACKEND, tool_name=OPERATOR_TOOL,
            tool_arguments={"operation": operation, "args": args},
            subject=f"Authenticated operator Run: {operation}", grant_key=digest,
            severity=snapshot["operation"]["risk_level"],
            prompt="Run authorizes this complete frozen operation, including its conditional verified deploy. "
                   "A changed production baseline or deployment operation refuses; rollback is separate.",
            operations_snapshot=snapshot,
            caller_context={"operator_user_id": user_id, "idempotency_key": idempotency_key},
        )
    try:
        if _identity_problem(row, user_id=user_id, key=key, operation=operation, args=args):
            raise ValueError("operator intent belongs to different arguments or identity")
        snapshot = json.loads(row.operations_snapshot_json)
        # Validate the saved authority on every replay, never re-prepare it.
        ops._verify_snapshot(snapshot, operation, args)
    except (ValueError, TypeError, KeyError) as exc:
        raise OpsDispatchError("idempotency_conflict", str(exc)) from exc
    if row.status != REQ_APPROVED:
        row = approvals.resolve_request(
            row.id, status=REQ_APPROVED, resolved_by=user_id, resolved_turn_id=key,
            message="Authenticated operator Run approved the entire selected operation.",
            continuation_owner="none", resolved_via=RESOLVED_VIA_USER)
    if (row is None or row.status != REQ_APPROVED or row.resolved_by != user_id
            or row.resolved_via != RESOLVED_VIA_USER or row.resolved_turn_id != key):
        raise OpsDispatchError("approval_required", "operator Run intent was already decided differently")
    return ops.dispatch_job(
        operation_slug=operation, args=args, idempotency_key=key,
        approved_snapshot=snapshot, caller_actor=user_id, caller_user_id=user_id,
        caller_bot_id=None, caller_turn_id=key, caller_backend=OPERATOR_BACKEND,
        approval_request_id=row.id)


def operator_authority_matches(request, parent) -> bool:
    """An exact audited operator decision, not a self-asserted backend string."""
    return (parent.caller_backend == OPERATOR_BACKEND and parent.caller_bot_id is None
            and request.request_kind == KIND_ORCHESTRATION
            and request.tool_name == OPERATOR_TOOL and request.backend == OPERATOR_BACKEND
            and request.bot_id == "ops" and request.user_id == parent.caller_user_id
            and request.resolved_by == parent.caller_user_id
            and request.resolved_via == RESOLVED_VIA_USER
            and request.resolved_turn_id == parent.caller_turn_id)
