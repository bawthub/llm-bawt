"""Owner-scoped schedule management; always lock ScheduledJob before children."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from uuid import NAMESPACE_URL, uuid5

from fastapi import HTTPException
from sqlalchemy import func, text
from sqlmodel import Session, select

from .prompt_schedule_models import NONTERMINAL, PromptOccurrence, PromptSchedule
from .prompt_schedule_store import db_utc
from .scheduler import JobRun, JobType, ScheduledJob


def now_utc():
    return datetime.now(timezone.utc)


def schedule_identifier(owner, idempotency_key):
    """A schedule's id is derived from its owner and create key, never chosen."""
    return str(uuid5(NAMESPACE_URL, f"prompt-schedule:{owner}:{idempotency_key}"))


def run_view(occurrence, run):
    receipt = json.loads(run.result_json or "{}")
    return dict(id=occurrence.run_id, state=occurrence.state,
                scheduled_for=occurrence.scheduled_for, created_at=occurrence.created_at,
                started_at=run.started_at if occurrence.state not in {"pending", "queued"} else None,
                finished_at=run.finished_at, duration_ms=run.duration_ms,
                error=occurrence.last_error or run.error_message, session_id=occurrence.session_id,
                turn_id=receipt.get("turn_id"), delivery_id=occurrence.delivery_id,
                snapshot=occurrence.snapshot_json, actual_model=receipt.get("actual_model") or receipt.get("model"),
                selected_bot_id=(occurrence.snapshot_json.get("bot_id")
                                 if occurrence.snapshot_json.get("bot_id") not in {None, "*"} else None))


def schedule_view(job, schedule, last_run=None):
    return dict(id=job.id, bot_id=job.bot_id if schedule.target_mode == "specific" else None,
                enabled=job.enabled, next_run_at=(db_utc(job.next_run_at) if schedule.lifecycle == "active" and job.enabled and job.next_run_at else None),
                timing=schedule.timing_json, last_run=last_run,
                **{key: getattr(schedule, key) for key in (
                    "owner_user_id", "name", "description", "prompt", "requested_model",
                    "target_mode", "candidate_bot_ids", "clear_context", "augment_memory", "extract_memory", "missed_policy",
                    "misfire_grace_seconds", "lifecycle", "revision", "created_at", "updated_at")})


class PromptManagement:
    def __init__(self, engine):
        self.engine = engine

    def _owned(self, session, owner, identifier, lock=False):
        query = select(ScheduledJob).join(PromptSchedule, PromptSchedule.job_id == ScheduledJob.id).where(
            ScheduledJob.id == identifier, PromptSchedule.owner_user_id == owner,
            ScheduledJob.job_type == JobType.SEND_PROMPT)
        if lock:
            query = query.with_for_update(of=ScheduledJob)
        job = session.exec(query).first()
        if job is None:
            raise HTTPException(404, "Schedule not found")
        return job, session.get(PromptSchedule, job.id)

    def get(self, owner, identifier):
        with Session(self.engine) as session:
            job, schedule = self._owned(session, owner, identifier)
            last = session.exec(select(PromptOccurrence, JobRun).join(JobRun, JobRun.id == PromptOccurrence.run_id)
                                .where(PromptOccurrence.job_id == identifier)
                                .order_by(PromptOccurrence.created_at.desc()).limit(1)).first()
            return schedule_view(job, schedule, run_view(*last) if last else None)

    def list(self, owner, *, bot_id=None, lifecycle=None, search=None, limit=50, offset=0):
        conditions = [PromptSchedule.owner_user_id == owner]
        if bot_id:
            conditions.append(ScheduledJob.bot_id == bot_id)
        if lifecycle:
            conditions.append(PromptSchedule.lifecycle == lifecycle)
        if search:
            conditions.append(PromptSchedule.name.ilike(f"%{search}%"))
        with Session(self.engine) as session:
            query = select(ScheduledJob, PromptSchedule).join(PromptSchedule, PromptSchedule.job_id == ScheduledJob.id).where(*conditions)
            total = session.exec(select(func.count()).select_from(query.subquery())).one()
            rows = session.exec(query.order_by(PromptSchedule.created_at.desc(), ScheduledJob.id).offset(offset).limit(limit)).all()
            ids = [job.id for job, _ in rows]
            latest = {}
            if ids:
                ranked = select(PromptOccurrence.run_id.label("run_id"), func.row_number().over(
                    partition_by=PromptOccurrence.job_id, order_by=(PromptOccurrence.created_at.desc(), PromptOccurrence.run_id)).label("rank")).where(PromptOccurrence.job_id.in_(ids)).subquery()
                runs = session.exec(select(PromptOccurrence, JobRun).join(JobRun, JobRun.id == PromptOccurrence.run_id)
                                    .join(ranked, ranked.c.run_id == PromptOccurrence.run_id).where(ranked.c.rank == 1)).all()
                latest = {occ.job_id: run_view(occ, run) for occ, run in runs}
            return {"schedules": [schedule_view(job, schedule, latest.get(job.id)) for job, schedule in rows], "total_count": total}

    def create(self, owner, data):
        payload = data.model_dump(mode="json", exclude={"idempotency_key"})
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        identifier = schedule_identifier(owner, data.idempotency_key)
        now = now_utc()
        with Session(self.engine) as session, session.begin():
            if self.engine.dialect.name == "postgresql":
                session.execute(text("SELECT pg_advisory_xact_lock(hashtext(:key))"), {"key": f"prompt-owner:{owner}"})
            existing = session.get(PromptSchedule, identifier)
            if existing:
                if existing.owner_user_id != owner or existing.create_request_hash != digest:
                    raise HTTPException(409, "Idempotency key already used for another request")
                return schedule_view(session.get(ScheduledJob, identifier), existing)
            count = session.exec(select(func.count()).select_from(PromptSchedule).where(
                PromptSchedule.owner_user_id == owner, PromptSchedule.lifecycle.in_(["active", "paused"]))).one()
            if count >= 100:
                raise HTTPException(409, "Limit of 100 active or paused schedules reached")
            due = data.timing.next_after(now)
            if due is None:
                raise HTTPException(422, "Schedule has no future execution")
            job = ScheduledJob(id=identifier, job_type=JobType.SEND_PROMPT,
                               bot_id=data.bot_id if data.target_mode == "specific" else "*",
                               enabled=data.enabled, next_run_at=due if data.enabled else None)
            session.add(job)
            session.flush()
            fields = {key: value for key, value in payload.items() if key not in {"bot_id", "enabled", "timing"}}
            schedule = PromptSchedule(job_id=identifier, owner_user_id=owner, timing_json=payload["timing"],
                                      lifecycle="active" if data.enabled else "paused", create_request_hash=digest, **fields)
            session.add(schedule)
            session.flush()
            return schedule_view(job, schedule)

    def update(self, owner, identifier, changes, validate):
        now = now_utc()
        with Session(self.engine) as session, session.begin():
            job, schedule = self._owned(session, owner, identifier, True)
            if changes.pop("expected_revision") != schedule.revision:
                raise HTTPException(409, "Schedule changed; reload before saving")
            if schedule.lifecycle in {"cancelled", "completed"}:
                raise HTTPException(409, "Terminal schedules cannot be edited; duplicate instead")
            active = session.exec(select(PromptOccurrence).where(PromptOccurrence.job_id == identifier,
                                                                 PromptOccurrence.state.in_(NONTERMINAL))).first()
            if active and set(changes) - {"name", "description"}:
                raise HTTPException(409, "Cannot change execution fields while an occurrence is outstanding")
            current = schedule_view(job, schedule)
            fields = {key: current[key] for key in changes if key in current}
            fields.update(changes)
            merged = {key: current[key] for key in validate.model_fields if key in current}
            merged.update(fields)
            parsed = validate.model_validate(merged)
            timing_changed = "timing" in changes
            if timing_changed:
                due = parsed.timing.next_after(now)
                if due is None:
                    raise HTTPException(422, "Schedule has no future execution")
                job.next_run_at = due
            bound_bot = parsed.bot_id if parsed.target_mode == "specific" else "*"
            if ("bot_id" in changes or "target_mode" in changes) and bound_bot != job.bot_id:
                schedule.dedicated_session_id = None
            job.bot_id = bound_bot
            if "enabled" in changes:
                if parsed.enabled and not job.enabled:
                    job.next_run_at = parsed.timing.next_after(now)
                    if job.next_run_at is None:
                        raise HTTPException(409, "Schedule has no future execution")
                job.enabled = parsed.enabled
                schedule.lifecycle = "active" if parsed.enabled else "paused"
            if not job.enabled:
                job.next_run_at = None
            for key, value in parsed.model_dump(mode="json").items():
                if key == "timing":
                    schedule.timing_json = value
                elif key not in {"bot_id", "enabled"}:
                    setattr(schedule, key, value)
            schedule.revision += 1
            schedule.updated_at = now
            session.add(job)
            session.add(schedule)
            return schedule_view(job, schedule)

    def lifecycle(self, owner, identifier, action):
        now = now_utc()
        with Session(self.engine) as session, session.begin():
            job, schedule = self._owned(session, owner, identifier, True)
            target = {"pause": "paused", "resume": "active", "cancel": "cancelled"}[action]
            if schedule.lifecycle == target:
                return schedule_view(job, schedule)
            if schedule.lifecycle in {"completed", "cancelled"}:
                raise HTTPException(409, "Schedule is already terminal")
            if action == "resume":
                if session.exec(select(PromptOccurrence).where(PromptOccurrence.job_id == identifier,
                      PromptOccurrence.state.in_(NONTERMINAL))).first():
                    raise HTTPException(409, "Wait for outstanding occurrence cancellation or completion")
                job.next_run_at = schedule.timing().next_after(now)
                if job.next_run_at is None:
                    raise HTTPException(409, "Schedule has no future execution; duplicate it instead")
            else:
                for occurrence, run in session.exec(select(PromptOccurrence, JobRun).join(JobRun, JobRun.id == PromptOccurrence.run_id)
                    .where(PromptOccurrence.job_id == identifier, PromptOccurrence.state.in_(["pending", "queued"]))).all():
                    if occurrence.kind == "manual" and action == "pause":
                        continue
                    result = json.loads(run.result_json or "{}")
                    result["cancel_requested"] = True
                    run.result_json = json.dumps(result)
                    session.add(run)
            schedule.lifecycle = target
            schedule.cancelled_at = now if action == "cancel" else None
            schedule.revision += 1
            schedule.updated_at = now
            job.enabled = action == "resume"
            if not job.enabled:
                job.next_run_at = None
            session.add(job)
            session.add(schedule)
            return schedule_view(job, schedule)

    def runs(self, owner, identifier, limit=50, offset=0):
        with Session(self.engine) as session:
            self._owned(session, owner, identifier)
            count = session.exec(select(func.count()).select_from(PromptOccurrence).where(PromptOccurrence.job_id == identifier)).one()
            rows = session.exec(select(PromptOccurrence, JobRun).join(JobRun, JobRun.id == PromptOccurrence.run_id)
                .where(PromptOccurrence.job_id == identifier).order_by(PromptOccurrence.created_at.desc())
                .offset(offset).limit(limit)).all()
            return {"runs": [run_view(*row) for row in rows], "total_count": count}

    def run_now(self, owner, identifier, key):
        now = now_utc()
        with Session(self.engine) as session, session.begin():
            job, schedule = self._owned(session, owner, identifier, True)
            occurrence_key = f"manual:{key}"
            prior = session.exec(select(PromptOccurrence).where(PromptOccurrence.job_id == identifier,
                                                                PromptOccurrence.occurrence_key == occurrence_key)).first()
            if prior:
                return run_view(prior, session.get(JobRun, prior.run_id))
            if schedule.lifecycle not in {"active", "paused"}:
                raise HTTPException(409, "Terminal schedule cannot run")
            if session.exec(select(PromptOccurrence).where(PromptOccurrence.job_id == identifier,
                                                           PromptOccurrence.state.in_(NONTERMINAL))).first():
                raise HTTPException(409, "An occurrence is already outstanding")
            end = schedule.timing().ends_at
            deadline = now + timedelta(seconds=schedule.misfire_grace_seconds)
            if end:
                deadline = min(deadline, db_utc(end))
            if deadline <= now:
                raise HTTPException(409, "Schedule end reached")
            run = JobRun(job_id=identifier, bot_id=job.bot_id, started_at=now)
            session.add(run)
            session.flush()
            snapshot = dict(owner_user_id=owner, bot_id=job.bot_id, name=schedule.name, prompt=schedule.prompt,
                            target_mode=schedule.target_mode, candidate_bot_ids=schedule.candidate_bot_ids,
                            model=schedule.requested_model, clear_context=schedule.clear_context,
                            augment_memory=schedule.augment_memory, extract_memory=schedule.extract_memory,
                            timing=schedule.timing_json, revision=schedule.revision, scheduled_for=now.isoformat(),
                            start_deadline=deadline.isoformat(), coalesced_from=None)
            occurrence = PromptOccurrence(run_id=run.id, job_id=identifier, occurrence_key=occurrence_key,
                                          kind="manual", scheduled_for=now, schedule_revision=schedule.revision,
                                          snapshot_json=snapshot, created_at=now, updated_at=now)
            session.add(occurrence)
            session.flush()
            return run_view(occurrence, run)

    def cancel_run(self, owner, identifier, run_id):
        with Session(self.engine) as session, session.begin():
            self._owned(session, owner, identifier, True)
            occurrence = session.get(PromptOccurrence, run_id)
            if occurrence is None or occurrence.job_id != identifier:
                raise HTTPException(404, "Run not found")
            run = session.get(JobRun, run_id)
            if occurrence.state == "cancelled":
                return run_view(occurrence, run)
            if occurrence.state not in {"pending", "queued"}:
                raise HTTPException(409, "Only pending or queued occurrences can be cancelled")
            if occurrence.delivery_id:
                row = session.execute(text("SELECT status, transport_accepted_at FROM inter_bot_deliveries WHERE id=:id"),
                                      {"id": occurrence.delivery_id}).mappings().first()
                if row and (row["status"] != "QUEUED" or row["transport_accepted_at"] is not None):
                    raise HTTPException(409, "Occurrence has already been reserved or accepted")
            result = json.loads(run.result_json or "{}")
            result["cancel_requested"] = True
            run.result_json = json.dumps(result)
            session.add(run)
            return run_view(occurrence, run)
