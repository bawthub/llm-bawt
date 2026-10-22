"""Ops-owned admission of approved ``ops_run`` requests (TASK-1220).

The app records the human decision only. This module turns an APPROVED
``ops_run`` row into exactly one durable ops job and commits the result that
the app's continuation outbox later delivers. It runs in the ops reconciler
(``reconciler.py``), never in the app, and imports only the ops package, the
approval models and the stdlib-only invocation hash: no MCP tool tree, no
FastAPI router. That removes the incident class where an approval lazily
imported newer on-disk tool code into a process still holding older modules.

Safety rests on existing durable primitives, not on this module being the only
admitter:

* ``claim_mcp_execution`` fences every claimant with a token and lease;
* the job idempotency key (caller key, else approval id) makes a repeated or
  concurrent dispatch a read of the same job;
* ``complete_mcp_execution`` commits result + continuation in one transaction
  and rejects a superseded claim.

Kept independent of where it runs so a future ops service (TASK-1169) can call
:meth:`ApprovedOpsAdmission.admit_due` unchanged.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Callable

from agent_bridge.mcp_call_context import canonical_invocation_hash

from ..approval_models import (
    OPS_RUN_TOOL,
    ApprovalPersistError,
    ExecutionDeadlinePassed,
    _utcnow,
)
from .models import JOB_CANCELLED, JOB_FAILED, JOB_LOST, JOB_TIMED_OUT
from .service import OpsDispatchError, OpsService

logger = logging.getLogger(__name__)

# Nick's call (2026-10-21): an approval not admitted within 10 minutes expires
# rather than firing whenever the reconciler returns. A job that already exists
# for the request's key is always reported, whatever its age.
ADMISSION_WINDOW_SECONDS = 600
# Bounded work per reconciler tick so admission can never starve job reconcile.
ADMISSION_BATCH_SIZE = 10
# Same lease as the app's execution claim; re-entry after expiry is safe for
# ops_run because the idempotency key turns a second dispatch into a read.
ADMISSION_LEASE_SECONDS = 300

# Receipt states that make an approved ops_run an error result. "error" is not
# an ops job state; it is kept because generic MCP payloads use it.
OPS_FAILURE_STATES = frozenset({JOB_FAILED, JOB_TIMED_OUT, JOB_LOST, JOB_CANCELLED, "error"})
TOOL_ERROR_RESULT = "Tool returned an error result"
_OPS_RUN_ARGUMENTS = frozenset({"operation", "args", "idempotency_key"})


def ops_run_receipt(result: dict[str, Any], idempotency_key: str, *,
                    caller_instruction: dict[str, Any] | None = None) -> dict[str, Any]:
    """The ``ops_run`` result contract, shared by the MCP tool and admission.

    ``caller_instruction`` (TASK-1233) is set only by the direct tool, for a
    caller its own quiesce holds. Admission leaves it null: an admitted
    receipt reaches its caller only once the job is terminal.
    """
    return {
        "id": result["id"],
        "job_id": result["id"],
        "release_id": result.get("release_id"),
        "operation": result["operation"],
        "state": result["state"],
        "submitted_at": result["submitted_at"],
        "host_unit_name": result.get("host_unit_name"),
        "idempotency_key": idempotency_key,
        "caller_instruction": caller_instruction,
    }


def dispatch_error_text(exc: OpsDispatchError) -> str:
    """Machine-parseable ``[code] message`` error the agent inspects."""
    return f"[{exc.code}] {exc}"


def ops_receipt_is_error(payload: Any) -> bool:
    return isinstance(payload, dict) and (
        payload.get("state") in OPS_FAILURE_STATES or payload.get("status") == "failed"
    )


def approved_provenance(row) -> dict[str, str | None]:
    """``ops_jobs.caller_*`` from the persisted approval row, never tool input."""
    return {
        "caller_bot_id": row.bot_id or None,
        "caller_user_id": row.user_id or None,
        "caller_turn_id": row.turn_id or None,
        "caller_session_key": row.session_key or None,
        "caller_backend": row.backend or None,
        "approval_request_id": row.id,
    }


@dataclass(frozen=True)
class AdmissionOutcome:
    payload: dict[str, Any]
    is_error: bool
    error: str | None

    @classmethod
    def failure(cls, error: str) -> "AdmissionOutcome":
        return cls({"status": "failed", "error": error}, True, error)

    @classmethod
    def receipt(cls, result: dict[str, Any], idempotency_key: str) -> "AdmissionOutcome":
        payload = ops_run_receipt(result, idempotency_key)
        is_error = ops_receipt_is_error(payload)
        return cls(payload, is_error, TOOL_ERROR_RESULT if is_error else None)


@dataclass(frozen=True)
class _Invocation:
    operation: str
    args: dict[str, Any]
    idempotency_key: str
    snapshot: dict[str, Any]


class _Refused(Exception):
    """The stored invocation cannot be admitted; complete it as a failure."""


class ApprovedOpsAdmission:
    """Admit approved ``ops_run`` rows into durable ops jobs, one row at a time."""

    def __init__(self, approvals, ops: OpsService, *,
                 window_seconds: float = ADMISSION_WINDOW_SECONDS,
                 lease_seconds: int = ADMISSION_LEASE_SECONDS,
                 clock: Callable = _utcnow):
        self.approvals = approvals
        self.ops = ops
        self.window = timedelta(seconds=window_seconds)
        self.lease_seconds = lease_seconds
        self.clock = clock

    def admit_due(self, *, limit: int = ADMISSION_BATCH_SIZE) -> int:
        """Admit up to ``limit`` due rows. One row's failure never stops the rest."""
        admitted = 0
        for row in self.approvals.find_admittable_ops_requests(
                limit=limit, lease_seconds=self.lease_seconds):
            try:
                admitted += self.admit(row)
            except Exception:
                # The claim (if taken) stays RUNNING; lease expiry re-enters it
                # with the same idempotency key, so no job can be duplicated.
                logger.exception("ops admission failed for approval %s; retrying after lease expiry", row.id)
        return admitted

    def admit(self, row) -> bool:
        """Claim, dispatch and complete one row. False when another admitter owns it."""
        claimed = self.approvals.claim_mcp_execution(row.id, lease_seconds=self.lease_seconds)
        if claimed is None:
            return False
        try:
            outcome = self._decide(claimed)
            self.approvals.complete_mcp_execution(
                claimed.id,
                result_json=json.dumps(outcome.payload, ensure_ascii=False, default=str),
                is_error=outcome.is_error,
                error=outcome.error,
                claim_token=claimed.execution_claim_token,
            )
        except ApprovalPersistError:
            # Superseded after its lease expired. The job-creation fence means a
            # stale claimant created no job after losing its claim; a job it did
            # create first is under the same key, which the newer claimant
            # reports. Either way the row's outcome belongs to the newer owner.
            logger.warning("ops admission claim for approval %s was superseded; result left to its owner", claimed.id)
            return False
        return True

    def _decide(self, row) -> AdmissionOutcome:
        try:
            invocation = self._invocation(row)
        except _Refused as refused:
            return AdmissionOutcome.failure(str(refused))
        # The window is enforced in one place: the fence, inside the transaction
        # that would create the job. A job already accepted for this key is a
        # plain read in dispatch_job and is reported at any age; only creating a
        # job needs the claim held and the approval within the window.
        fence = self.approvals.execution_fence(
            row.id, row.execution_claim_token, approved_within=self.window, clock=self.clock)
        try:
            result = self.ops.dispatch_job(
                operation_slug=invocation.operation,
                args=invocation.args,
                idempotency_key=invocation.idempotency_key,
                approved_snapshot=invocation.snapshot,
                create_fence=fence,
                **approved_provenance(row),
            )
        except OpsDispatchError as exc:
            return AdmissionOutcome.failure(dispatch_error_text(exc))
        except ExecutionDeadlinePassed:
            minutes = int(self.window.total_seconds() // 60)
            return AdmissionOutcome.failure(
                f"[admission_expired] approved but not admitted within {minutes} minutes; "
                "no job was created. Request a fresh approval if it is still wanted.")
        return AdmissionOutcome.receipt(result, invocation.idempotency_key)

    def _invocation(self, row) -> _Invocation:
        if row.tool_name != OPS_RUN_TOOL:
            raise _Refused(f"[args_invalid] {row.tool_name!r} is not admitted by ops")
        try:
            arguments = json.loads(row.tool_arguments_json or "{}")
        except (json.JSONDecodeError, TypeError):
            raise _Refused("[args_invalid] stored ops_run arguments are malformed") from None
        if not isinstance(arguments, dict):
            raise _Refused("[args_invalid] stored ops_run arguments are not an object")
        # The integrity check the app's approved-call path performs before
        # executing: run exactly the invocation the human approved.
        if canonical_invocation_hash(row.tool_name, arguments) != (row.invocation_hash or ""):
            raise _Refused("[invocation_mismatch] stored arguments do not match the approved invocation; request a fresh approval")
        unknown = set(arguments) - _OPS_RUN_ARGUMENTS
        operation = arguments.get("operation")
        args = arguments.get("args")
        key = arguments.get("idempotency_key")
        if (unknown or not isinstance(operation, str) or not operation
                or not (args is None or isinstance(args, dict))
                or not (key is None or isinstance(key, str))):
            raise _Refused("[args_invalid] stored ops_run arguments do not match the ops_run schema")
        try:
            snapshot = json.loads(row.operations_snapshot_json or "null")
        except (json.JSONDecodeError, TypeError):
            snapshot = None
        if not isinstance(snapshot, dict) or not snapshot:
            raise _Refused("[snapshot_invalid] approved operation has no persisted invocation snapshot; request fresh approval")
        # The caller's key wins so identical approved calls collapse onto one
        # job (TASK-959); otherwise the durable approval id keys the job.
        return _Invocation(operation, {} if args is None else args,
                           (key or "").strip() or row.id, snapshot)


__all__ = [
    "ADMISSION_WINDOW_SECONDS",
    "OPS_FAILURE_STATES",
    "AdmissionOutcome",
    "ApprovedOpsAdmission",
    "approved_provenance",
    "dispatch_error_text",
    "ops_receipt_is_error",
    "ops_run_receipt",
]
