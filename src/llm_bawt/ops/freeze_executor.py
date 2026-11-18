"""Operator freeze and thaw as ops jobs (TASK-1489).

The ``freeze`` executor runs in the ops service and writes the same
``operator_freeze`` rows as BawtHub's Freeze button, with no HTTP or token.
Opening or releasing the gate notifies ``system_gate_changed`` in its own
transaction, which wakes the dispatcher and refreshes state exactly as for a
UI freeze.

* Freeze: the actor is ``ops:job-<job_id>``, so the gate's owner names the
  job. ``freeze(idempotent_owner=True)`` returns any gate that owner ever
  opened, so a recovered job reads back its gate and never opens a second.
* Thaw: only a freeze opened by this operation (owner ``freeze:ops:job-``),
  never a UI freeze. The receipt is a postcondition: success means the gate
  is inactive now. ``released_by_this_job`` is true only when this executor
  released it; after a restart, or when it was already inactive, it is
  ``"unknown"``.

Dispatch and reconcile run the same idempotent step; reconcile reports it.
Every refusal (bad arguments, no PostgreSQL, not an ops freeze) is a terminal
failure with its reason.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from ..service import operator_freeze
from ..service.turn_gates import GateScope
from .executor_types import DispatchResult, ExecutorError, ReconcileResult
from .executor import Executor
from .freeze_spec import FREEZE_EXECUTOR_KIND, validate_freeze_spec
from .models import JOB_FAILED, JOB_SUCCEEDED

logger = logging.getLogger(__name__)

OPS_ACTOR_PREFIX = "ops:job-"
OPS_FREEZE_OWNER_PREFIX = operator_freeze.OWNER_PREFIX + OPS_ACTOR_PREFIX
UNKNOWN = "unknown"


def job_actor(job_id: str) -> str:
    return f"{OPS_ACTOR_PREFIX}{job_id}"


class _Refused(Exception):
    """A terminal refusal; the message is the job's error."""


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


class FreezeExecutor(Executor):
    """In-process executor: one short DB transaction, no worker."""

    def __init__(self, engine, *, scope: GateScope | None = None):
        self.engine = engine
        # Every agent bot, as the UI's default freeze; tests narrow it.
        self.scope = scope or GateScope.every_agent()
        # Thaws this executor instance performed: job id -> released.
        self._released: dict[str, bool] = {}

    def kind(self) -> str:
        return FREEZE_EXECUTOR_KIND

    def available(self) -> bool:
        return self.engine is not None and self.engine.dialect.name == "postgresql"

    def check_target(self, spec: dict, args: dict) -> None:
        if not self.available():
            raise ExecutorError("operator freeze requires the PostgreSQL gate store")

    def dispatch(self, **kwargs) -> DispatchResult:
        job_id = kwargs["job_id"]
        try:
            self._step(job_id, kwargs["snapshot"])
        except _Refused:
            pass  # Reconcile repeats the step and reports the refusal.
        except Exception as exc:  # noqa: BLE001 - uncertain; reconcile re-reads
            raise ExecutorError(f"operator freeze step uncertain: {exc}") from exc
        return DispatchResult(host_unit_name=f"freeze:{job_id}")

    def reconcile(self, **kwargs) -> ReconcileResult:
        now = datetime.now(timezone.utc).isoformat()
        try:
            receipt = self._step(kwargs["job_id"], kwargs["snapshot"])
        except _Refused as exc:
            return ReconcileResult(state=JOB_FAILED, exit_code=1, error=str(exc), finished_at=now)
        except Exception as exc:  # noqa: BLE001 - transient; the pump retries
            logger.exception("freeze reconcile failed for job %s", kwargs.get("job_id"))
            raise ExecutorError(f"operator freeze unavailable: {exc}") from exc
        return ReconcileResult(state=JOB_SUCCEEDED, exit_code=0, finished_at=now,
                               output_tail=json.dumps(receipt, sort_keys=True))

    def _step(self, job_id: str, snapshot: dict) -> dict:
        op = validate_freeze_spec(snapshot["spec"])["op"]
        args = snapshot.get("resolved_args") or {}
        if op == "freeze":
            return self._freeze(job_id, args)
        return self._thaw(job_id, args)

    def _freeze(self, job_id: str, args: dict) -> dict:
        actor = job_actor(job_id)
        try:
            frozen = operator_freeze.freeze(
                self.engine, actor=actor, reason=str(args.get("reason") or "").strip(),
                scope=self.scope, ttl_seconds=args.get("ttl_seconds"),
                admit_user=bool(args.get("admit_user", False)), idempotent_owner=True)
        except (ValueError, TypeError, operator_freeze.FreezeUnavailable,
                operator_freeze.FreezeConflict) as exc:
            raise _Refused(f"freeze refused: {exc}") from exc
        record = operator_freeze.freeze_record(self.engine, frozen.gate_id)
        return {"op": "freeze", "gate_id": frozen.gate_id, "actor": actor,
                "expires_at": frozen.expires_at.isoformat(), "admit_user": frozen.admit_user,
                "scope": frozen.scope, "live": bool(record and record.live)}

    def _thaw(self, job_id: str, args: dict) -> dict:
        gate_id = args.get("gate_id")
        if type(gate_id) is not int or gate_id < 1:
            raise _Refused("thaw refused: gate_id must be a positive integer")
        try:
            record = operator_freeze.freeze_record(self.engine, gate_id)
        except operator_freeze.FreezeUnavailable as exc:
            raise _Refused(f"thaw refused: {exc}") from exc
        if record is None:
            raise _Refused(f"thaw refused: gate {gate_id} is not a freeze")
        if not record.owner_id.startswith(OPS_FREEZE_OWNER_PREFIX):
            raise _Refused(f"thaw refused: freeze {gate_id} was not opened by the freeze operation")
        if record.live:
            # The owner is an ops freeze, so the override reaches only those.
            result = operator_freeze.thaw(self.engine, gate_id, actor=job_actor(job_id), override=True)
            if result is not None and result.released:
                self._released[job_id] = True
            record = operator_freeze.freeze_record(self.engine, gate_id)
            if record is None or record.live:
                raise ExecutorError(f"freeze {gate_id} is still live after thaw")
        released = True if self._released.get(job_id) else UNKNOWN
        receipt = {"op": "thaw", "gate_id": gate_id, "live": False, "released_by_this_job": released,
                   "freeze_actor": record.freeze.actor}
        if record.released_at is not None:
            receipt["released_at"] = _iso(record.released_at)
        else:
            receipt["expired_at"] = record.freeze.expires_at.isoformat()
        return receipt


__all__ = ["FreezeExecutor", "OPS_FREEZE_OWNER_PREFIX", "job_actor"]
