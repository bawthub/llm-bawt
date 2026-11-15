"""Route list endpoints read related rows in batches, and workspace errors log safely."""

import logging
from datetime import UTC, datetime, timedelta

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import event
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from llm_bawt.profiles import (
    EntityProfile,
    EntityType,
    ProfileAttribute,
    ProfileManager,
)
from llm_bawt.service.routes import jobs, profiles
from llm_bawt.service.routes import workspace as workspace_routes
from llm_bawt.service.scheduler import (
    JobRun,
    JobStatus,
    JobType,
    ScheduledJob,
    create_scheduler_tables,
)

from .test_workspace_routes import client as workspace_client  # noqa: F401
from .test_workspace_routes import upstream


def _engine():
    return create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)


def _count_selects(engine) -> list[str]:
    seen: list[str] = []

    @event.listens_for(engine, "before_cursor_execute")
    def _record(_conn, _cursor, statement, *_args):
        if statement.lstrip().upper().startswith("SELECT"):
            seen.append(statement)

    return seen


@pytest.fixture
def profile_api(monkeypatch):
    engine = _engine()
    SQLModel.metadata.create_all(engine, tables=[EntityProfile.__table__, ProfileAttribute.__table__])
    with Session(engine) as session:
        for kind, prefix in ((EntityType.USER, "u"), (EntityType.BOT, "b")):
            for n in range(4):
                entity_id = f"{prefix}{n}"
                session.add(EntityProfile(entity_type=kind, entity_id=entity_id))
                session.add(ProfileAttribute(entity_type=kind, entity_id=entity_id,
                                             category="fact", key="k", value=entity_id))
        session.commit()
    manager = object.__new__(ProfileManager)
    manager.engine = engine
    monkeypatch.setattr(profiles, "get_service", lambda: type("S", (), {"config": None})())
    monkeypatch.setattr(profiles, "get_profile_manager", lambda _config: manager)
    app = FastAPI()
    app.include_router(profiles.router)
    yield TestClient(app), engine
    engine.dispose()


def test_profile_list_query_count_is_constant(profile_api):
    client, engine = profile_api
    selects = _count_selects(engine)
    body = client.get("/v1/profiles").json()
    assert body["total_count"] == 8
    assert all(p["attributes"][0]["value"] == p["entity_id"] for p in body["profiles"])
    # One list + one attribute batch per entity type, independent of profile count.
    assert len(selects) == 4

    selects.clear()
    typed = client.get("/v1/profiles?entity_type=bot&entity_id=B2").json()
    assert [p["entity_id"] for p in typed["profiles"]] == ["b2"]
    assert len(selects) == 2


@pytest.fixture
def jobs_api(monkeypatch):
    engine = _engine()
    create_scheduler_tables(engine)
    start = datetime(2026, 11, 1, tzinfo=UTC)
    with Session(engine) as session:
        for bot in ("a", "b", "c"):
            job = ScheduledJob(job_type=JobType.MEMORY_CONSOLIDATION, bot_id=bot, interval_minutes=60)
            session.add(job)
            session.flush()
            for minutes, status in ((0, JobStatus.FAILED), (10, JobStatus.SUCCESS), (5, JobStatus.FAILED)):
                session.add(JobRun(job_id=job.id, job_type=job.job_type, bot_id=bot, status=status,
                                   started_at=start + timedelta(minutes=minutes),
                                   duration_ms=minutes))
        session.add(ScheduledJob(job_type=JobType.MEMORY_CONSOLIDATION, bot_id="never", interval_minutes=60))
        session.commit()
    monkeypatch.setattr(jobs, "_get_scheduler_engine", lambda: engine)
    app = FastAPI()
    app.include_router(jobs.router)
    with TestClient(app) as client:
        yield client, engine
    engine.dispose()


def test_job_list_reads_latest_runs_in_one_query(jobs_api):
    client, engine = jobs_api
    selects = _count_selects(engine)
    listed = {j["bot_id"]: j for j in client.get("/v1/jobs").json()["jobs"]}
    assert {bot: (j["last_status"], j["last_duration_ms"]) for bot, j in listed.items()} == {
        "a": ("success", 10), "b": ("success", 10), "c": ("success", 10), "never": (None, None),
    }
    run_selects = [s for s in selects if "job_runs" in s]
    assert len(run_selects) == 1


def test_unexpected_workspace_error_logs_class_only(workspace_client, monkeypatch, caplog):  # noqa: F811
    secret = "https://user:tok-SECRET@bridge.test/body-SECRET"
    upstream(monkeypatch, error=RuntimeError(secret))
    with caplog.at_level(logging.WARNING, logger=workspace_routes.logger.name):
        response = workspace_client.get("/v1/workspace")
    assert response.status_code == 503
    assert "SECRET" not in response.text
    records = [r for r in caplog.records if r.name == workspace_routes.logger.name]
    assert [r.getMessage() for r in records] == ["Workspace bridge GET list failed: RuntimeError"]
    assert not any("SECRET" in r.getMessage() for r in caplog.records)


def test_profile_batch_helper_handles_empty_and_case():
    engine = _engine()
    SQLModel.metadata.create_all(engine, tables=[ProfileAttribute.__table__])
    manager = object.__new__(ProfileManager)
    manager.engine = engine
    assert manager.get_live_attributes_by_entity(EntityType.USER, []) == {}
    assert manager.get_live_attributes_by_entity(EntityType.USER, [" Operator "]) == {"operator": []}
    engine.dispose()
