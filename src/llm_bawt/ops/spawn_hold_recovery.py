"""Dormant hold quarantine; deadlines classify uncertainty, never grant authority.

No dispatcher, worker or operator endpoint enables this policy. The internal
release seam consumes a trusted, existing-authority verifier under the exclusive
receipt-volume flock; this module cannot infer Docker safety from a dead PID,
empty flock, terminal job state or timeout.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping

from sqlalchemy import func, update
from sqlmodel import Session, select

from .events import append_job_transition
from .execution_mode import EXECUTION_ISOLATED, snapshot_execution_mode
from .models import EXECUTOR_DOCKER, JOB_FAILED, JOB_LOST, JOB_QUEUED, JOB_SUCCEEDED, JOB_TERMINAL_STATES, OpsJob
from .process_spawn_fence import exclusive_process_spawn_guard
from .validation import canonical_json


@dataclass(frozen=True)
class VerifiedHoldOutcome:
    """Result of an AUTHORIZED verifier, not a caller's safety/approval flag.

    The verifier must establish the exact action's existing authority, absence
    of an outstanding original mutation, and a healthy intended reconciler or
    verified restore. Its observations belong under the retained exclusive
    flock. These references bind its result to existing immutable ledger data;
    constructing this record does not itself verify or authorize anything.
    """
    job_id: str
    target_identity: Mapping[str, str]
    authority_job_id: str
    authority_snapshot_hash: str
    outcome: str  # verified_success | verified_restore


def _utc(value):
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def _identity(raw):
    if not isinstance(raw, Mapping) or set(raw) != {"container_id", "compose_project", "compose_service"}:
        raise ValueError("spawn hold requires exact target identity")
    if not isinstance(raw["container_id"], str) or not re.fullmatch(r"[0-9a-f]{64}", raw["container_id"]):
        raise ValueError("spawn hold requires full container ID")
    for key in ("compose_project", "compose_service"):
        if not isinstance(raw[key], str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", raw[key]):
            raise ValueError("spawn hold requires Compose target labels")
    return canonical_json(dict(raw))


def prepare_hold(store, job_id, *, target_identity, deadline_at):
    """Pin once BEFORE exclusive-acquisition retries; this does not take a hold.

    Caller has already inspected the approved exact target. The same persisted
    deadline/identity survive restart; duplicates cannot extend or retarget it.
    """
    store._require()
    target = _identity(target_identity)
    if not isinstance(deadline_at, datetime) or deadline_at.utcoffset() is None:
        raise ValueError("spawn hold requires an aware absolute deadline")
    with Session(store.engine) as session:
        job = session.exec(select(OpsJob).where(OpsJob.id == job_id).with_for_update()).first()
        if job is None or job.state != JOB_QUEUED or job.process_spawn_hold_target_json is not None:
            return False
        snapshot = json.loads(job.invocation_snapshot_json or "{}")
        kind = (snapshot.get("execution") or {}).get("executor_kind")
        if (kind != EXECUTOR_DOCKER or snapshot_execution_mode(snapshot, kind) != EXECUTION_ISOLATED
                or (snapshot.get("quiesce") or {}).get("enabled")):
            return False
        root = (snapshot.get("execution") or {}).get("receipt_root")
        if not isinstance(root, str) or not Path(root).is_absolute():
            return False
        now = session.exec(select(func.current_timestamp())).one()
        if _utc(deadline_at) <= _utc(now):
            return False
        changed = session.execute(update(OpsJob).where(
            OpsJob.id == job_id, OpsJob.state == JOB_QUEUED,
            OpsJob.process_spawn_hold_target_json.is_(None), OpsJob.process_spawn_hold_at.is_(None)
        ).values(process_spawn_hold_target_json=target, process_spawn_hold_deadline_at=deadline_at))
        session.commit()
        return changed.rowcount == 1


def classify_hold(store, job_id):
    """One bounded ledger step: waiting, quarantined, released, or not_held.

    On/after persisted deadline, an open holder becomes LOST (no replay), but
    its independent durable quarantine continues blocking process admission.
    A terminal holder without verified release is quarantined immediately.
    No worker/daemon liveness guess is made. No enabled pump calls this yet.
    """
    store._require()
    with Session(store.engine) as session:
        job = session.exec(select(OpsJob).where(OpsJob.id == job_id).with_for_update()).first()
        if job is None or job.process_spawn_hold_at is None or job.process_spawn_hold_target_json is None:
            return "not_held"
        if job.process_spawn_hold_release_json is not None:
            return "released"
        now = session.exec(select(func.current_timestamp())).one()
        if (job.state not in JOB_TERMINAL_STATES and job.process_spawn_hold_deadline_at is not None
                and _utc(now) < _utc(job.process_spawn_hold_deadline_at)):
            return "waiting"
        was_terminal = job.state in JOB_TERMINAL_STATES
        if not was_terminal:
            job.state, job.finished_at = JOB_LOST, now
            job.error_text = "process_spawn_hold_unknown: deadline elapsed; inspection required, spawns quarantined"
        if job.process_spawn_hold_quarantine_at is None:
            job.process_spawn_hold_quarantine_at = now
        session.add(job)
        session.flush()
        if not was_terminal:
            append_job_transition(getattr(store, "events", None), session.connection(), job)
        session.commit()
        return "quarantined"


def release_hold(store, job_id, *, receipt_root, acquire_deadline,
                 verifier: Callable[[OpsJob], VerifiedHoldOutcome]):
    """Verify OUTSIDE the DB transaction but under EX, then commit once.

    ``verifier`` is a trusted integration boundary, not an authorization policy
    or a new approval role. It must check the applicable exact-action authority
    using existing admission/approval machinery before returning evidence. A
    missing/unreadable receipt, unknown restore, unavailable target or authority
    must raise/refuse. There is deliberately no enabled default verifier.

    A later recovery action uses its own authorized acquisition deadline, not
    an extension of the original holder's expired deadline. Release preserves
    terminal history/forensics; a late original receipt cannot replay the job.
    """
    store._require()
    initial = store.get_job(job_id)
    if initial is None:
        return False
    snapshot = json.loads(initial.invocation_snapshot_json or "{}")
    bound_root = (snapshot.get("execution") or {}).get("receipt_root")
    if (not isinstance(bound_root, str) or not Path(bound_root).is_absolute()
            or Path(receipt_root) != Path(bound_root)):
        return False  # An unrelated/empty lock file is not this job's fence.
    with exclusive_process_spawn_guard(bound_root, acquire_deadline=acquire_deadline):
        original = store.get_job(job_id)
        if (original is None or original.process_spawn_hold_at is None
                or original.process_spawn_hold_target_json is None
                or original.state not in JOB_TERMINAL_STATES
                or original.invocation_snapshot_json != initial.invocation_snapshot_json):
            return False
        if original.process_spawn_hold_release_json is not None:
            return False
        target_binding = original.process_spawn_hold_target_json
        deadline_binding = original.process_spawn_hold_deadline_at
        snapshot_binding = original.invocation_snapshot_json
        evidence = verifier(original)
        if (not isinstance(evidence, VerifiedHoldOutcome) or evidence.job_id != job_id
                or _identity(evidence.target_identity) != target_binding
                or evidence.outcome not in ("verified_success", "verified_restore")):
            return False
        with Session(store.engine) as session:
            job = session.exec(select(OpsJob).where(OpsJob.id == job_id).with_for_update()).first()
            authority = session.get(OpsJob, evidence.authority_job_id)
            allowed = (JOB_SUCCEEDED,) if evidence.outcome == "verified_success" else (JOB_SUCCEEDED, JOB_FAILED)
            if (job is None or job.state not in JOB_TERMINAL_STATES
                    or job.process_spawn_hold_release_json is not None
                    or job.process_spawn_hold_target_json != target_binding
                    or job.process_spawn_hold_deadline_at != deadline_binding
                    or job.invocation_snapshot_json != snapshot_binding
                    or authority is None or authority.state not in allowed
                    or not authority.invocation_snapshot_json
                    or hashlib.sha256(authority.invocation_snapshot_json.encode()).hexdigest()
                    != evidence.authority_snapshot_hash):
                return False
            now = session.exec(select(func.current_timestamp())).one()
            job.process_spawn_hold_release_json = canonical_json({
                "job_id": job_id, "target_identity": json.loads(target_binding),
                "authority_job_id": authority.id, "authority_snapshot_hash": evidence.authority_snapshot_hash,
                "approval_request_id": authority.approval_request_id,
                "outcome": evidence.outcome, "released_at": _utc(now).isoformat(),
            })
            session.add(job)
            session.commit()
            return True
