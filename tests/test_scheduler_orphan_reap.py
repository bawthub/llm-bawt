"""TASK-832: RUNNING job runs stranded by a dead process are reaped, not re-fired."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine, select

from llm_bawt.service.scheduler import (
    JobRun, JobScheduler, JobStatus, JobType, ScheduledJob,
)


def _setup():
    engine = create_engine("sqlite://", poolclass=StaticPool,
                           connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(engine, tables=[ScheduledJob.__table__, JobRun.__table__])
    now = datetime.now(timezone.utc)
    maint = ScheduledJob(job_type=JobType.MEMORY_EXTRACTION, bot_id="b", next_run_at=now + timedelta(hours=1))
    prompt = ScheduledJob(job_type=JobType.SEND_PROMPT, bot_id="b")
    with Session(engine) as s:
        s.add(maint)
        s.add(prompt)
        s.commit()
        s.refresh(maint)
        s.refresh(prompt)
        runs = {
            "stale": JobRun(job_id=maint.id, bot_id="b", status=JobStatus.RUNNING,
                            started_at=now - timedelta(hours=3)),
            "live": JobRun(job_id=maint.id, bot_id="b", status=JobStatus.RUNNING,
                           started_at=now - timedelta(minutes=5)),
            "prompt": JobRun(job_id=prompt.id, bot_id="b", status=JobStatus.RUNNING,
                             started_at=now - timedelta(hours=3)),
            "done": JobRun(job_id=maint.id, bot_id="b", status=JobStatus.SUCCESS,
                           started_at=now - timedelta(hours=3), finished_at=now - timedelta(hours=3)),
        }
        for r in runs.values():
            s.add(r)
        s.commit()
        ids = {k: r.id for k, r in runs.items()}
    return engine, ids


def test_reaps_only_stale_maintenance_runs_without_refiring():
    engine, ids = _setup()
    scheduler = JobScheduler(engine=engine, task_processor=None)
    fired = []
    scheduler._execute_job = lambda job: fired.append(job)  # would re-fire
    assert scheduler._reap_orphaned_runs() == 1
    with Session(engine) as s:
        got = {k: s.get(JobRun, v) for k, v in ids.items()}
        assert got["stale"].status == JobStatus.FAILED
        assert got["stale"].finished_at is not None
        assert got["stale"].error_message.startswith("interrupted")
        assert got["stale"].duration_ms is None
        assert got["live"].status == JobStatus.RUNNING
        assert got["prompt"].status == JobStatus.RUNNING
        assert got["done"].status == JobStatus.SUCCESS
        assert len(s.exec(select(JobRun)).all()) == 4  # no new run created
    assert fired == []
    assert scheduler._reap_orphaned_runs() == 0  # idempotent
