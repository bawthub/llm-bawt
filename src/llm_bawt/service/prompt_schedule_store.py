"""Atomic prompt occurrence producer; no chat execution or delivery side effects.

Every mutation locks the parent ScheduledJob first. Delivery adapters must use
that same lock order. Occurrences are immutable outbox snapshots: a retry must
never pick up an edited prompt/model from the schedule definition.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import exists
from sqlmodel import Session, select

from .prompt_schedule_models import NONTERMINAL, PromptOccurrence, PromptSchedule
from .prompt_timing import PromptTiming, utc
from .scheduler import JobRun, JobStatus, JobType, ScheduledJob


def db_utc(value: datetime) -> datetime:
    """SQLite loses tzinfo; our DB contract is UTC, unlike untrusted API input."""
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else utc(value)


class PromptScheduleStore:
    def __init__(self, engine):
        self.engine = engine

    def create(self, *, owner: str, bot_id: str, name: str, prompt: str,
               timing: PromptTiming, now: datetime, enabled: bool = True,
               requested_model: str | None = None, clear_context: bool = True,
               description: str = "", augment_memory: bool = True,
               extract_memory: bool = True, missed_policy: str = "run_latest",
               misfire_grace_seconds: int = 3600) -> str:
        """Persistence primitive; caller owns catalog/auth checks before calling."""
        now = utc(now)
        if not owner.strip() or not bot_id.strip() or bot_id.strip() == "*":
            raise ValueError("A concrete owner and target bot are required")
        if not 1 <= len(name.strip()) <= 120 or not 1 <= len(prompt.strip()) <= 100000:
            raise ValueError("Name or prompt is blank or too long")
        if len(description) > 2000:
            raise ValueError("Description exceeds 2000 characters")
        if missed_policy not in {"skip", "run_latest"} or not 60 <= misfire_grace_seconds <= 604800:
            raise ValueError("Invalid missed-run policy or deadline")
        due = timing.next_after(now)
        if due is None:
            raise ValueError("Schedule has no future execution within its bounds")
        with Session(self.engine) as session, session.begin():
            job = ScheduledJob(job_type=JobType.SEND_PROMPT, bot_id=bot_id.strip(),
                               enabled=enabled, next_run_at=due if enabled else None)
            session.add(job)
            session.flush()
            schedule = PromptSchedule(
                job_id=job.id, owner_user_id=owner.strip(), name=name.strip(),
                description=description, prompt=prompt, timing_json=timing.model_dump(mode="json"),
                requested_model=requested_model, clear_context=clear_context,
                augment_memory=augment_memory, extract_memory=extract_memory,
                missed_policy=missed_policy, misfire_grace_seconds=misfire_grace_seconds,
                lifecycle="active" if enabled else "paused", created_at=now, updated_at=now,
            )
            session.add(schedule)
            return job.id

    def reserve_due(self, now: datetime, limit: int = 100) -> list[str]:
        """Reserve at most one latest occurrence per job, advancing atomically.

        Busy schedules are excluded before LIMIT so they cannot starve unrelated
        due jobs. Their due pointer stays put for bounded coalescing on completion.
        """
        now = utc(now)
        if not 1 <= limit <= 1000:
            raise ValueError("Reservation limit must be between 1 and 1000")
        reserved = []
        with Session(self.engine) as session, session.begin():
            active = exists().where(PromptOccurrence.job_id == ScheduledJob.id,
                                    PromptOccurrence.state.in_(NONTERMINAL))
            jobs = session.exec(
                select(ScheduledJob).join(PromptSchedule, PromptSchedule.job_id == ScheduledJob.id)
                .where(ScheduledJob.job_type == JobType.SEND_PROMPT,
                       ScheduledJob.enabled.is_(True), PromptSchedule.lifecycle == "active",
                       ScheduledJob.next_run_at.is_not(None), ScheduledJob.next_run_at <= now,
                       ~active)
                .order_by(ScheduledJob.next_run_at, ScheduledJob.id).limit(limit)
                .with_for_update(skip_locked=True, of=ScheduledJob)
            ).all()
            for job in jobs:
                schedule = session.get(PromptSchedule, job.id)
                timing = schedule.timing()
                due = db_utc(job.next_run_at)
                latest = timing.latest_at(now)
                if latest is None or latest < due:
                    raise ValueError(f"Schedule {job.id} due pointer does not match its timing")
                # `skip` drops backlog, not routine scheduler tick latency. The
                # configured start deadline defines how late an occurrence may be.
                slot = due if schedule.missed_policy == "skip" else latest
                expired = (now - slot).total_seconds() > schedule.misfire_grace_seconds
                missed = schedule.missed_policy == "skip" and latest > due
                ended = timing.ends_at is not None and now >= utc(timing.ends_at)
                reason = ("Schedule end reached" if ended else
                          "Missed occurrences skipped" if missed else
                          "Start deadline exceeded" if expired else None)
                run = JobRun(job_id=job.id, bot_id=job.bot_id,
                             status=JobStatus.SKIPPED if reason else JobStatus.PENDING,
                             started_at=now, finished_at=now if reason else None,
                             error_message=reason)
                session.add(run)
                session.flush()
                snapshot = {
                    "owner_user_id": schedule.owner_user_id, "creator_bot_id": schedule.creator_bot_id,
                    "bot_id": job.bot_id,
                    "target_mode": schedule.target_mode, "candidate_bot_ids": schedule.candidate_bot_ids,
                    "name": schedule.name, "prompt": schedule.prompt,
                    "model": schedule.requested_model, "clear_context": schedule.clear_context,
                    "augment_memory": schedule.augment_memory, "extract_memory": schedule.extract_memory,
                    "timing": schedule.timing_json, "revision": schedule.revision,
                    "scheduled_for": slot.isoformat(),
                    "start_deadline": min(slot + timedelta(seconds=schedule.misfire_grace_seconds),
                                          utc(timing.ends_at) if timing.ends_at else datetime.max.replace(tzinfo=timezone.utc)).isoformat(),
                    "coalesced_from": due.isoformat() if latest > due else None,
                }
                occurrence = PromptOccurrence(
                    run_id=run.id, job_id=job.id, occurrence_key=slot.isoformat(),
                    scheduled_for=slot, schedule_revision=schedule.revision,
                    snapshot_json=snapshot, state="skipped" if reason else "pending",
                    last_error=reason, created_at=now, updated_at=now,
                )
                session.add(occurrence)
                job.next_run_at = timing.next_after(now)
                # Exhausted schedules with accepted work remain active until
                # terminal reconciliation. No null-due busy-loop in this producer.
                if reason and job.next_run_at is None:
                    job.enabled = False
                    schedule.lifecycle = "completed"
                schedule.updated_at = now
                session.add(job)
                session.add(schedule)
                if not reason:
                    reserved.append(run.id)
        return reserved

    def pending(self, limit: int = 100) -> list[PromptOccurrence]:
        """Recoverable outbox inventory, safe to re-enqueue by occurrence key."""
        if not 1 <= limit <= 1000:
            raise ValueError("Outbox limit must be between 1 and 1000")
        with Session(self.engine) as session:
            return session.exec(select(PromptOccurrence).where(PromptOccurrence.state == "pending")
                                .order_by(PromptOccurrence.created_at).limit(limit)).all()
