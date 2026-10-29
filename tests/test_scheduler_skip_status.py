"""TASK-1727: a task result's skipped/error shape is persisted on the JobRun."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine, select

from llm_bawt.service.scheduler import JobRun, JobScheduler, JobStatus, JobType, ScheduledJob
from llm_bawt.service.tasks import TaskStatus


class _Processor:
    def __init__(self, result):
        self.result = result

    async def process_task(self, task):
        return SimpleNamespace(status=TaskStatus.COMPLETED, result=self.result, error=None)


def _run(result) -> JobRun:
    engine = create_engine("sqlite://", poolclass=StaticPool,
                           connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(engine, tables=[ScheduledJob.__table__, JobRun.__table__])
    job = ScheduledJob(job_type=JobType.MEMORY_EXTRACTION, bot_id="test")
    with Session(engine) as session:
        session.add(job)
        session.commit()
        session.refresh(job)
    scheduler = JobScheduler(engine=engine, task_processor=_Processor(result))
    scheduler._has_new_activity_for_job = lambda _job: (True, None)
    scheduler._create_task_for_job = lambda _job: object()
    asyncio.run(scheduler._execute_single_job(job))
    with Session(engine) as session:
        return session.exec(select(JobRun)).one()


@pytest.mark.parametrize(("result", "expected"), [
    ({"skipped": True, "reason": "No human-authored messages in window"}, JobStatus.SKIPPED),
    ({"skipped": True, "error": "boom"}, JobStatus.FAILED),
    ({"facts_stored": 2}, JobStatus.SUCCESS),
])
def test_task_result_shape_persists_job_run_status(result, expected):
    run = _run(result)
    assert run.status == expected
    if expected == JobStatus.FAILED:
        assert run.error_message == "boom"
