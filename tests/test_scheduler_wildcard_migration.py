"""TASK-2258: legacy per-bot maintenance jobs with run history must not stop startup.

The per-bot HISTORY_SUMMARIZATION and MEMORY_EXTRACTION jobs became one
wildcard job each. Their runs reference the job row without ON DELETE CASCADE,
so migrating has to move the runs onto the wildcard before the legacy rows go.
The PostgreSQL acceptance reuses :func:`run_migration_scenario`.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine, event
from sqlmodel import Session, select

from llm_bawt.service import scheduler
from llm_bawt.service.scheduler import (
    JobRun,
    JobStatus,
    JobType,
    ScheduledJob,
    create_scheduler_tables,
    init_default_jobs,
)
from llm_bawt.utils.config import Config

MIGRATED = (JobType.HISTORY_SUMMARIZATION, JobType.MEMORY_EXTRACTION)


class _Defaults:
    def __init__(self, config=None, **_):
        pass

    def resolve(self, _key, default):
        return default


def _seed_legacy(engine, bots) -> dict[str, list[str]]:
    """Per-bot jobs of both migrated types, each with two runs. Returns run ids per type."""
    runs: dict[str, list[str]] = {t.value: [] for t in MIGRATED}
    with Session(engine) as session:
        for job_type in MIGRATED:
            for bot in bots:
                job = ScheduledJob(job_type=job_type, bot_id=bot, interval_minutes=30)
                session.add(job)
                session.flush()
                for status in (JobStatus.SUCCESS, JobStatus.FAILED):
                    run = JobRun(job_id=job.id, bot_id=bot, status=status,
                                 finished_at=datetime.now(timezone.utc))
                    session.add(run)
                    runs[job_type.value].append(run.id)
        session.commit()
    return runs


def _jobs(session, job_type, bots=None):
    query = select(ScheduledJob).where(ScheduledJob.job_type == job_type)
    if bots is not None:
        query = query.where(ScheduledJob.bot_id.in_(bots))
    return session.exec(query).all()


def run_migration_scenario(engine, monkeypatch, bots) -> None:
    """Seed legacy jobs with runs, start twice, check history moved and nothing duplicated."""
    monkeypatch.setattr(scheduler, "RuntimeSettingsResolver", _Defaults)
    config = Config()
    config.SCHEDULER_ENABLED = True
    create_scheduler_tables(engine)
    runs = _seed_legacy(engine, bots)

    for _ in range(2):  # startup must succeed and be a no-op the second time
        init_default_jobs(engine, config)
        with Session(engine) as session:
            for job_type in MIGRATED:
                assert _jobs(session, job_type, bots) == []
                wildcards = [j for j in _jobs(session, job_type) if j.bot_id == "*"]
                assert len(wildcards) == 1
                kept = session.exec(
                    select(JobRun).where(JobRun.id.in_(runs[job_type.value]))
                ).all()
                assert len(kept) == len(runs[job_type.value])
                assert {r.job_id for r in kept} == {wildcards[0].id}
                assert {r.bot_id for r in kept} == set(bots)  # per-bot attribution survives


@pytest.fixture
def sqlite_engine(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'scheduler.db'}")

    @event.listens_for(engine, "connect")
    def _fks(dbapi_conn, _record):
        dbapi_conn.execute("PRAGMA foreign_keys=ON")

    try:
        yield engine
    finally:
        engine.dispose()


def test_legacy_jobs_with_runs_migrate_on_sqlite(sqlite_engine, monkeypatch):
    run_migration_scenario(sqlite_engine, monkeypatch, ["alpha", "beta"])


def test_existing_wildcard_leaves_jobs_and_runs_alone(sqlite_engine, monkeypatch):
    monkeypatch.setattr(scheduler, "RuntimeSettingsResolver", _Defaults)
    create_scheduler_tables(sqlite_engine)
    with Session(sqlite_engine) as session:
        wildcard = ScheduledJob(job_type=JobType.MEMORY_EXTRACTION, bot_id="*", interval_minutes=15)
        session.add(wildcard)
        session.flush()
        session.add(JobRun(job_id=wildcard.id, bot_id="alpha", status=JobStatus.SUCCESS))
        session.commit()
        wildcard_id = wildcard.id

    init_default_jobs(sqlite_engine, Config())

    with Session(sqlite_engine) as session:
        jobs = _jobs(session, JobType.MEMORY_EXTRACTION)
        assert [(j.id, j.interval_minutes) for j in jobs] == [(wildcard_id, 15)]
        assert session.exec(select(JobRun)).one().job_id == wildcard_id


def test_failed_migration_keeps_legacy_jobs_and_runs(sqlite_engine, monkeypatch):
    monkeypatch.setattr(scheduler, "RuntimeSettingsResolver", _Defaults)
    create_scheduler_tables(sqlite_engine)
    runs = _seed_legacy(sqlite_engine, ["alpha"])

    def broken_delete(_table):  # fails after the wildcard insert and the run move
        raise RuntimeError("delete failed")

    monkeypatch.setattr(scheduler, "sa_delete", broken_delete)
    with Session(sqlite_engine) as session, pytest.raises(RuntimeError):
        scheduler._ensure_wildcard_job(
            session, JobType.HISTORY_SUMMARIZATION,
            lambda: ScheduledJob(job_type=JobType.HISTORY_SUMMARIZATION, bot_id="*"),
        )

    with Session(sqlite_engine) as session:
        jobs = _jobs(session, JobType.HISTORY_SUMMARIZATION)
        assert [j.bot_id for j in jobs] == ["alpha"]
        kept = session.exec(select(JobRun).where(JobRun.id.in_(runs["history_summarization"]))).all()
        assert {r.job_id for r in kept} == {jobs[0].id}
