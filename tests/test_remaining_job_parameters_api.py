"""TASK-1198: API readback and edits preserve existing rows and inherited controls."""
import json
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, create_engine

from llm_bawt.service.routes import jobs as routes
from llm_bawt.service.scheduler import JobRun, JobType, ScheduledJob, init_default_jobs


@pytest.fixture
def api(monkeypatch):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    ScheduledJob.__table__.create(engine)
    JobRun.__table__.create(engine)
    shared = SimpleNamespace(
        DEFAULT_USER="owner", SCHEDULER_ENABLED=False, PROFILE_MAINTENANCE_INTERVAL_MINUTES=31,
        MEMORY_EXTRACTION_MIN_IMPORTANCE=.9, MEMORY_PROFILE_ATTRIBUTE_ENABLED=False,
        MEMORY_EXTRACTION_ENABLED=True, resolve_summarization_job=lambda: {
            "session_gap_seconds": 3600, "min_messages_per_session": 2,
            "protected_recent_turns": 3, "trigger_tokens": 12000, "model": None,
        },
    )
    with Session(engine) as session:
        for job in JobType:
            if job != JobType.SEND_PROMPT:
                session.add(ScheduledJob(
                    id=job.value, job_type=job, bot_id="*", enabled=False,
                    interval_minutes=47, config_json=None,
                ))
        session.commit()
    monkeypatch.setattr(routes, "_get_scheduler_engine", lambda: engine)
    monkeypatch.setattr(routes, "get_service", lambda: SimpleNamespace(config=shared))
    monkeypatch.setattr(routes, "_mark_jobs_changed", lambda: None)
    app = FastAPI()
    app.include_router(routes.router)
    with TestClient(app) as client:
        yield client, engine, shared
    engine.dispose()


def test_every_existing_job_reads_back_and_empty_reads_never_write(api):
    client, engine, _ = api
    response = client.get("/v1/jobs")
    assert response.status_code == 200
    for row in response.json()["jobs"]:
        assert row["parameter_schema"] and row["parameters"] is not None
        assert row["configuration_error"] is None and row["interval_minutes"] == 47
    with Session(engine) as session:
        for job in JobType:
            if job != JobType.SEND_PROMPT:
                row = session.get(ScheduledJob, job.value)
                assert row.config_json is None and row.enabled is False


def test_patch_does_not_pin_other_inherited_defaults_or_reschedule(api):
    client, engine, shared = api
    response = client.patch("/v1/jobs/memory_extraction", json={"parameters": {"batch_size": 9}})
    assert response.status_code == 200, response.text
    shared.MEMORY_EXTRACTION_MIN_IMPORTANCE = .2
    rows = {row["id"]: row for row in client.get("/v1/jobs").json()["jobs"]}
    assert rows["memory_extraction"]["parameters"]["min_importance"] == .2
    assert rows["memory_extraction"]["parameters"]["batch_size"] == 9
    with Session(engine) as session:
        row = session.get(ScheduledJob, "memory_extraction")
        assert json.loads(row.config_json) == {"batch_size": 9}
        assert row.next_run_at is None and row.interval_minutes == 47 and not row.enabled


def test_startup_does_not_overwrite_config_cadence_or_enablement(api, monkeypatch):
    from llm_bawt.service import scheduler

    _, engine, shared = api
    monkeypatch.setattr(scheduler, "RuntimeSettingsResolver", lambda **kw: SimpleNamespace(
        resolve=lambda key, fallback: fallback,
    ))
    with Session(engine) as session:
        row = session.get(ScheduledJob, "memory_extraction")
        row.config_json = '{"batch_size":11,"min_importance":0}'
        row.interval_minutes = 91
        session.add(row)
        session.commit()
    init_default_jobs(engine, shared)
    init_default_jobs(engine, shared)
    with Session(engine) as session:
        row = session.get(ScheduledJob, "memory_extraction")
        assert row.config_json == '{"batch_size":11,"min_importance":0}'
        assert row.interval_minutes == 91 and row.next_run_at is None and not row.enabled


def test_invalid_patch_never_changes_stored_controls(api):
    client, engine, _ = api
    response = client.patch("/v1/jobs/media_gc", json={
        "enabled": True, "interval_minutes": 10, "parameters": {"dry_run": "false"},
    })
    assert response.status_code == 422
    with Session(engine) as session:
        row = session.get(ScheduledJob, "media_gc")
        assert row.config_json is None and not row.enabled and row.interval_minutes == 47


def test_execution_uses_claimed_defaults_for_both_probe_and_worker_off_loop(api):
    import asyncio
    import threading

    from llm_bawt.service.scheduler import JobScheduler
    from llm_bawt.service.tasks import TaskStatus

    _, engine, shared = api
    main_thread = threading.get_ident()
    observed = []

    class Processor:
        config = shared

        async def process_task(self, task):
            observed.append(task.payload)
            return SimpleNamespace(status=TaskStatus.COMPLETED, result={}, error=None)

    scheduler = JobScheduler(engine, Processor())
    build = scheduler._build_task_for_job

    def probe(job):
        assert threading.get_ident() != main_thread
        assert json.loads(job.config_json)["min_importance"] == .9
        shared.MEMORY_EXTRACTION_MIN_IMPORTANCE = .1
        return True, None

    def checked_build(job, parameters):
        assert threading.get_ident() != main_thread
        return build(job, parameters)

    scheduler._has_new_activity_for_job = probe
    scheduler._build_task_for_job = checked_build
    with Session(engine) as session:
        job = session.get(ScheduledJob, "memory_extraction")
        session.expunge(job)
    asyncio.run(scheduler._execute_single_job(job))
    assert observed[0]["min_importance"] == .9
    with Session(engine) as session:
        assert session.get(ScheduledJob, "memory_extraction").config_json is None
