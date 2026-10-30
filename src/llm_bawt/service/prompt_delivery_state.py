"""Shared prompt run transitions; caller holds ScheduledJob before delivery locks."""
from __future__ import annotations

import json
from datetime import datetime

from ..message_authorship import AuthorReference
from .prompt_schedule_models import NONTERMINAL
from .prompt_schedule_store import db_utc
from .scheduler import JobStatus


def schedule_author(snapshot: dict) -> AuthorReference:
    """Who an occurrence's delivery is from: its proven creator bot, else its owner.

    One rule for enqueue and dispatch validation (TASK-1790). The owner stays
    the run's user (visibility, session, gates) either way.
    """
    creator = snapshot.get("creator_bot_id")
    return AuthorReference.bot(creator) if creator else AuthorReference.user(snapshot["owner_user_id"])


def receipt(run) -> dict:
    return json.loads(run.result_json or "{}")


def gate_reason(job, schedule, occurrence, run, now: datetime) -> tuple[str, str] | None:
    """Only apply before transport reservation/acceptance, never to running work."""
    if receipt(run).get("cancel_requested"):
        return "cancelled", "Occurrence cancellation requested"
    if occurrence.state not in NONTERMINAL:
        return occurrence.state, occurrence.last_error or "Occurrence already terminal"
    if schedule.lifecycle == "cancelled":
        return "cancelled", "Schedule cancelled"
    # A manual test is deliberately allowed while paused.
    if occurrence.kind == "scheduled" and (not job.enabled or schedule.lifecycle != "active"):
        return "cancelled", "Schedule paused or no longer active"
    deadline = datetime.fromisoformat(occurrence.snapshot_json["start_deadline"])
    if now >= db_utc(deadline):
        return "skipped", "Start deadline exceeded"
    return None


def transition(session, job, schedule, occurrence, run, state: str, now: datetime,
               *, error: str | None = None, evidence: dict | None = None) -> None:
    occurrence.state = state
    occurrence.last_error = error
    occurrence.updated_at = now
    result = {**receipt(run), **(evidence or {}), "state": state}
    run.result_json = json.dumps(result, default=str)
    run.error_message = error
    run.status = {
        "pending": JobStatus.PENDING, "queued": JobStatus.PENDING,
        "running": JobStatus.RUNNING, "succeeded": JobStatus.SUCCESS,
        "failed": JobStatus.FAILED, "unknown": JobStatus.FAILED,
        "cancelled": JobStatus.SKIPPED, "skipped": JobStatus.SKIPPED,
    }[state]
    if state not in NONTERMINAL:
        run.finished_at = now
        run.duration_ms = max(0, int((now - db_utc(run.started_at)).total_seconds() * 1000))
        job.last_run_at = now
        if (occurrence.kind == "scheduled" and job.next_run_at is None
                and schedule.lifecycle == "active"):
            schedule.lifecycle = "completed"
            schedule.updated_at = now
            job.enabled = False
    for row in (job, schedule, occurrence, run):
        session.add(row)
