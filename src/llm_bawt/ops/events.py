"""Safe event projection for winning ops-job transitions."""

from __future__ import annotations

from sqlmodel import Session

from ..system_events import SystemEventEmitter
from .models import JOB_SUCCEEDED, JOB_TERMINAL_STATES, OpsJob


def append_job_transition(emitter: SystemEventEmitter | None, conn, job: OpsJob) -> None:
    if emitter is None:
        return
    requester = job.caller_bot_id or job.caller_user_id or job.caller_actor or "system"
    state = job.state
    # Lifecycle states never revert to an earlier state. A retry of the same
    # winning transition therefore uses the same key, not a new receipt.
    emitter.append(
        conn, event_key=f"ops:{job.id}:{state}", kind="ops_job",
        source="ops", user_id=job.caller_user_id,
        tone=("info" if state not in JOB_TERMINAL_STATES else
              "success" if state == JOB_SUCCEEDED else "error"),
        summary=f"{job.operation_slug} · {state} · by {requester}",
        bot_id=job.caller_bot_id, actor_type=("bot" if job.caller_bot_id else "user"),
        actor_id=requester, subject_type="ops_job", subject_id=job.id,
    )


def append_job_transition_by_id(emitter: SystemEventEmitter | None, conn, job_id: str) -> None:
    if emitter is None:
        return
    with Session(bind=conn) as session:
        job = session.get(OpsJob, job_id)
        if job is not None:
            append_job_transition(emitter, conn, job)
