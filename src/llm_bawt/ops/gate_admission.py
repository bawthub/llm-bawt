"""Dormant create-only admission for an immutable gate request.

Internal callers must obtain ``turn_token`` from authenticated request context
and select ``operation_slug`` through trusted configuration, never model/body
identity, grant or destination arguments. No route/tool/reconciler calls this.
Creation is NOT authorization to execute: the gate-only claim still checks the
current DB-owned grant. No claim, callback, runner or generic executor is used.
"""
from __future__ import annotations

from ..task_turn_context import open_task_turn_context
from .gate_claim import gate_claim_snapshot
from .gate_plan import GatePlan
from .models import OpsJob
from .store import IdempotencyConflict, OpsStore
from .validation import canonical_json

GATE_REQUEST_SCHEMA = "llm-bawt.ops.gate-request/v1"


def create_gate_job(store: OpsStore, *, operation_slug: str, plan: dict,
                    turn_token: str, idempotency_key: str) -> OpsJob:
    """Create one inert QUEUED row, or read an identical keyed request.

    The request binds the workload and signed caller/session/trigger identity,
    not the opaque expiring token or its issue time. Reissued capabilities for
    the same turn can read that request; other callers/workloads conflict. A
    replay never rebuilds the snapshot, claims, retries or attaches a callback,
    even after catalog changes or terminal completion. Concurrent creation uses
    the existing ledger's unique key and snapshot conflict check.

    This is deliberately unused until the trusted runner path is ready. The
    original valid capability is still required for a separate gate claim;
    persisted provenance and a content hash are not a durable execution grant.
    """
    workload = GatePlan.from_dict(plan)
    caller = open_task_turn_context(turn_token)
    if (not isinstance(idempotency_key, str) or not idempotency_key.strip()
            or len(idempotency_key) > 128):
        raise ValueError("idempotency_key must be 1..128 nonblank characters")
    payload = canonical_json({"schema": GATE_REQUEST_SCHEMA, "operation": operation_slug,
        "gate_plan": workload.to_dict(), "caller": {key: getattr(caller, key) for key in
            ("bot_id", "user_id", "turn_id", "session_id", "trigger_message_id")}})
    existing = store.get_job_by_key(idempotency_key)
    if existing is not None:
        store.verify_payload(existing, payload)
        # Defense against accepting another internal writer's mismatched row.
        if ((existing.caller_bot_id, existing.caller_user_id, existing.caller_turn_id)
                != (caller.bot_id, caller.user_id, caller.turn_id)):
            raise IdempotencyConflict("stored gate caller differs from the immutable request")
        return existing

    operation = store.get_operation_by_slug(operation_slug)
    if operation is None:
        raise ValueError("gate operation is not configured")
    if (not operation.enabled or operation.soft_deleted_at is not None
            or operation.quiesce_enabled or operation.execution_tag is None):
        raise ValueError("gate operation is disabled, quiescing or has no configured execution grant")
    snapshot = gate_claim_snapshot(operation, workload, turn_token=turn_token)
    return store.create_job(operation=operation, args_json=canonical_json(workload.to_dict()),
        display_args_json=canonical_json(workload.to_dict()), idempotency_key=idempotency_key,
        invocation_snapshot_json=canonical_json(snapshot), request_payload_json=payload,
        caller_actor=f"bot:{caller.bot_id}", caller_bot_id=caller.bot_id,
        caller_user_id=caller.user_id, caller_turn_id=caller.turn_id)
