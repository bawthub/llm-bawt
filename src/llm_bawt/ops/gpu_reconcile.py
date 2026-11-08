"""GPU-service reconcile reservation (TASK-1844).

A same-image recreate of a GPU Compose service (``gpu: true`` on an app
reconcile spec) replaces the worker a GPU handoff drives. So the job holds the
handoff row from its insert until its outcome is KNOWN:

* reserve: :meth:`GpuHandoffFence.gpu_reconcile` runs in the job-insert
  transaction and persists ``phase=gpu_reconciling`` with the job's key;
* dispatch: the row must still hold that key, else the job stays queued;
* release: only on a known outcome (:func:`known_outcome`). A lost, timed-out
  or otherwise uncertain job leaves the reservation held, with the reason in
  ``last_error``, until an operator reconciles the target.

A GPU reconcile runs only as its own approved job, never inside a plan or a
sequence, whose step-level outcomes this module does not read.
"""
from __future__ import annotations

import json

from .models import JOB_CANCELLED, JOB_FAILED, JOB_SUCCEEDED

#: Reconcile kinds that may carry ``gpu: true``.
GPU_RECONCILE_ACTIONS = frozenset({"reconcile_app_config"})


def is_gpu_reconcile(spec: dict | None) -> bool:
    return isinstance(spec, dict) and spec.get("gpu") is True and spec.get("action") in GPU_RECONCILE_ACTIONS


def nested_gpu_reconcile(snapshot: dict) -> bool:
    """A plan or sequence that would run a GPU reconcile as one of its steps."""
    for step in (snapshot.get("sequence") or {}).get("steps") or []:
        if is_gpu_reconcile(((step or {}).get("snapshot") or {}).get("spec")):
            return True
    for target in (snapshot.get("plan") or {}).get("targets") or []:
        if is_gpu_reconcile((target or {}).get("spec") if isinstance(target, dict) else None):
            return True
    return False


def known_outcome(job) -> str | None:
    """Why this terminal job's outcome is known (release), else None (hold).

    Structured evidence only, never the bare state:
    * ``succeeded``: the worker verified health and the GPU configuration;
    * never dispatched: no worker ran (``dispatched_at`` unset);
    * ``failed`` after dispatch with the deployer's receipt proving either no
      container was created (``deployed`` null, ``restored`` false) or the
      previous container was restored and verified healthy (``restored``).
    ``lost``, ``timed_out``, a dispatched cancel and a receipt-less failure hold.
    """
    if job.state == JOB_SUCCEEDED and job.exit_code == 0:
        return "succeeded"
    if job.state in (JOB_FAILED, JOB_CANCELLED) and job.dispatched_at is None:
        return "never_dispatched"
    if job.state != JOB_FAILED:
        return None
    try:
        detail = json.loads(job.output_tail or "")
    except ValueError:
        return None
    if not isinstance(detail, dict):
        return None
    if detail.get("restored") is True:
        return "restored"
    if detail.get("deployed") is None and detail.get("restored") is False and "phase" in detail:
        return "nothing_changed"
    return None


def settle(engine, job) -> str | None:
    """Release or keep holding this job's reservation once it is terminal.
    Returns the release reason, or None when it stays held."""
    from ..media.gpu_handoff_store import GpuHandoffStore

    store = GpuHandoffStore(engine)
    reason = known_outcome(job)
    if reason is None:
        store.hold_gpu_reconcile(job.idempotency_key,
                                 f"GPU service reconcile {job.id} ended {job.state} with an uncertain outcome; "
                                 "verify the target before releasing the GPU")
        return None
    store.release_gpu_reconcile(job.idempotency_key)
    return reason


__all__ = ["GPU_RECONCILE_ACTIONS", "is_gpu_reconcile", "known_outcome", "nested_gpu_reconcile", "settle"]
