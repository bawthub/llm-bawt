"""Hermetic scheduler configuration API and execution-policy regression tests."""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, create_engine, select

from llm_bawt.message_authorship import AuthorReference
from llm_bawt.service.job_parameters import (
    IdleContextResetParameters, JobParameterRegistry,
)
from llm_bawt.service.jobs.idle_context_candidates import IdleContextCandidates
from llm_bawt.service.jobs.idle_context_reset import IdleContextResetJob
from llm_bawt.service.routes import jobs as routes
from llm_bawt.service.scheduler import JobRun, JobScheduler, JobType, ScheduledJob
from llm_bawt.service.session_lifecycle.intent import (
    Admission, Continuity, NewSessionIntent, NewSessionResult, Origin, Outcome, Stage,
)
from llm_bawt.service.session_lifecycle.lifecycle import NewSessionLifecycle


@pytest.fixture
def api(monkeypatch):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    ScheduledJob.__table__.create(engine)
    JobRun.__table__.create(engine)
    with Session(engine) as session:
        session.add(ScheduledJob(id="reset", job_type=JobType.IDLE_CONTEXT_RESET, bot_id="*",
                                 enabled=False, interval_minutes=5))
        session.add(ScheduledJob(id="legacy", job_type=JobType.MEMORY_DECAY, bot_id="fixture",
                                 config_json='{"legacy_key":42}'))
        session.add(ScheduledJob(id="prompt", job_type=JobType.SEND_PROMPT, bot_id="fixture"))
        session.commit()
    monkeypatch.setattr(routes, "_get_scheduler_engine", lambda: engine)
    app = FastAPI()
    app.include_router(routes.router)
    with TestClient(app) as client:
        yield client, engine
    engine.dispose()


def test_listing_effective_defaults_and_unmigrated_compatibility(api):
    client, _ = api
    result = client.get("/v1/jobs").json()
    rows = {row["id"]: row for row in result["jobs"]}
    assert "prompt" not in rows
    reset = rows["reset"]
    assert reset["parameters"] == {"resident_tokens_gt": 100000, "idle_minutes_gt": 15, "protected_bot_ids": []}
    assert reset["parameter_schema"]["properties"]["idle_minutes_gt"]["minimum"] == 1
    assert rows["legacy"]["parameter_schema"] is None
    assert rows["legacy"]["parameters"] is None
    assert rows["legacy"]["configuration_error"] is None


def test_patch_persists_and_resumes_without_immediate_execution(api):
    client, engine = api
    before = datetime.now(timezone.utc).replace(tzinfo=None)
    response = client.patch("/v1/jobs/reset", json={
        "enabled": True, "interval_minutes": 9,
        "parameters": {"idle_minutes_gt": 30, "resident_tokens_gt": 150000},
    })
    assert response.status_code == 200, response.text
    with Session(engine) as session:
        row = session.get(ScheduledJob, "reset")
        assert row.enabled and row.interval_minutes == 9
        assert row.next_run_at >= before + timedelta(minutes=9)
        next_run = row.next_run_at
        assert json.loads(row.config_json)["idle_minutes_gt"] == 30
    assert client.patch("/v1/jobs/reset", json={"parameters": {"resident_tokens_gt": 120000}}).status_code == 200
    with Session(engine) as session:
        row = session.get(ScheduledJob, "reset")
        assert row.next_run_at == next_run
        assert json.loads(row.config_json)["idle_minutes_gt"] == 30
        assert json.loads(row.config_json)["resident_tokens_gt"] == 120000


@pytest.mark.parametrize("payload", [
    {"interval_minutes": 0}, {"interval_minutes": 525601}, {"interval_minutes": True},
    {"interval_minutes": "5"}, {"interval_minutes": None}, {"enabled": "true"},
    {"enabled": None}, {"parameters": None}, {"unexpected": 1},
    {"parameters": {"idle_minutes_gt": 0}}, {"parameters": {"idle_minutes_gt": 10081}},
    {"parameters": {"resident_tokens_gt": 0}}, {"parameters": {"resident_tokens_gt": 2000001}},
    {"parameters": {"resident_tokens_gt": True}}, {"parameters": {"resident_tokens_gt": 1.5}},
    {"parameters": {"resident_tokens_gt": "100000"}}, {"parameters": {"unknown": 1}},
    {"parameters": {"protected_bot_ids": "fixture"}}, {"parameters": {"protected_bot_ids": [""]}},
])
def test_invalid_updates_are_atomic(api, payload):
    client, engine = api
    response = client.patch("/v1/jobs/reset", json=payload)
    assert response.status_code == 422, response.text
    with Session(engine) as session:
        row = session.get(ScheduledJob, "reset")
        assert row.config_json is None and not row.enabled and row.interval_minutes == 5


def test_legacy_cadence_changes_preserve_opaque_config(api):
    client, engine = api
    assert client.patch("/v1/jobs/legacy", json={"enabled": False, "interval_minutes": 90}).status_code == 200
    assert client.patch("/v1/jobs/legacy", json={"parameters": {}}).status_code == 422
    with Session(engine) as session:
        row = session.get(ScheduledJob, "legacy")
        assert row.config_json == '{"legacy_key":42}'
        assert row.interval_minutes == 90 and not row.enabled
    assert client.patch("/v1/jobs/prompt", json={"enabled": True}).status_code == 404
    assert client.patch("/v1/jobs/missing", json={"enabled": True}).status_code == 404


def test_invalid_stored_threshold_visible_and_repairable(api):
    client, engine = api
    with Session(engine) as session:
        row = session.get(ScheduledJob, "reset")
        row.config_json = '{"idle_minutes_gt":-5}'
        session.add(row)
        session.commit()
    row = client.get("/v1/jobs?job_type=idle_context_reset").json()["jobs"][0]
    assert row["configuration_error"] and row["parameters"] is None
    assert client.patch("/v1/jobs/reset", json={"enabled": True}).status_code == 422
    assert client.patch("/v1/jobs/reset", json={"enabled": False}).status_code == 200
    assert client.patch("/v1/jobs/reset", json={"parameters": {"idle_minutes_gt": 20}}).status_code == 200
    assert client.get("/v1/jobs?job_type=idle_context_reset").json()["jobs"][0]["configuration_error"] is None


@pytest.mark.parametrize("stored", [
    {"unknown_key": "preserve", "resident_tokens_gt": 350000,
     "idle_minutes_gt": 15, "protected_bot_ids": ["fixture-protected"]},
    {"resident_tokens_gt": 350000, "idle_minutes_gt": -5,
     "protected_bot_ids": ["fixture-protected"]},
])
def test_pause_and_cadence_do_not_repair_or_replace_invalid_parameters(api, stored):
    client, engine = api
    original = json.dumps(stored)
    with Session(engine) as session:
        row = session.get(ScheduledJob, "reset")
        row.enabled, row.config_json = True, original
        session.add(row)
        session.commit()
    listed = client.get("/v1/jobs?job_type=idle_context_reset").json()["jobs"][0]
    assert listed["parameters"] is None and listed["configuration_error"]
    assert listed["stored_parameters"]["resident_tokens_gt"] == 350000
    assert listed["stored_parameters"]["protected_bot_ids"] == ["fixture-protected"]
    for patch in ({"enabled": False}, {"interval_minutes": 45}):
        assert client.patch("/v1/jobs/reset", json=patch).status_code == 200
        with Session(engine) as session:
            row = session.get(ScheduledJob, "reset")
            assert not row.enabled and row.config_json == original
    if "unknown_key" not in stored:
        assert client.patch("/v1/jobs/reset", json={"parameters": {"idle_minutes_gt": 15}}).status_code == 200
        with Session(engine) as session:
            row = session.get(ScheduledJob, "reset")
            assert json.loads(row.config_json) == {**stored, "idle_minutes_gt": 15}


def candidate():
    now = datetime.now(timezone.utc)
    return {"bot_id": "fixture", "session_id": "old", "user_id": "owner",
            "usage_turn_id": "usage", "latest_turn_id": "usage", "ended_at": now - timedelta(minutes=10),
            "last_completed_at": now - timedelta(minutes=10),
            "token_usage_json": json.dumps({"resident_tokens": 50000})}


def intent(parameters=None):
    return NewSessionIntent(
        bot_id="fixture", user_id="owner", author=AuthorReference.bot("system"),
        continuity=Continuity.CONFIGURED, admission=Admission.SKIP, origin=Origin.SCHEDULER,
        idempotency_key="key", expected_session_id="old", expected_usage_turn_id="usage",
        job_id="job", run_id="run", scheduler_parameters=parameters,
    )


def test_custom_policy_roundtrip_and_claim_recheck_use_same_values(monkeypatch):
    from llm_bawt.service.session_lifecycle import lifecycle

    params = IdleContextResetParameters(resident_tokens_gt=40000, idle_minutes_gt=5)
    original = intent(params)
    stored = json.loads(json.dumps(NewSessionLifecycle._encode(original)))
    decoded = NewSessionLifecycle._decode(stored)
    assert decoded.scheduler_parameters == params
    monkeypatch.setattr(IdleContextCandidates, "read_one", lambda *args: candidate())
    monkeypatch.setattr(lifecycle, "classify_target", lambda *args, **kw: SimpleNamespace(idle=True))
    conn = Mock()
    conn.execute.return_value.first.return_value = None
    assert NewSessionLifecycle.scheduled_skip_reason(conn, decoded) is None
    assert NewSessionLifecycle.scheduled_skip_reason(conn, intent()) == "idle less than threshold"
    with pytest.raises(ValidationError):
        params.idle_minutes_gt = 100
    # Stored snapshots, not subsequent global/DB edits, govern in-flight claim checks.
    assert stored["scheduler_parameters"]["idle_minutes_gt"] == 5


def test_legacy_intent_encoding_is_unchanged():
    assert "scheduler_parameters" not in NewSessionLifecycle._encode(intent())
    assert NewSessionLifecycle._decode(NewSessionLifecycle._encode(intent())).scheduler_parameters is None


def test_configured_protection_applies_before_reading_target(monkeypatch):
    reader = Mock(side_effect=AssertionError("protected target must not be read"))
    monkeypatch.setattr(IdleContextCandidates, "read_one", reader)
    parameters = IdleContextResetParameters(protected_bot_ids=[" FIXTURE "])
    assert NewSessionLifecycle.scheduled_skip_reason(Mock(), intent(parameters)) == "protected bot"
    reader.assert_not_called()


def test_strict_threshold_boundaries():
    row = candidate()
    now = row["last_completed_at"] + timedelta(minutes=10)
    assert IdleContextCandidates.skip_reason(row, now=now, parameters=IdleContextResetParameters(
        resident_tokens_gt=49999, idle_minutes_gt=10)) == "idle less than threshold"
    assert IdleContextCandidates.skip_reason(row, now=now, parameters=IdleContextResetParameters(
        resident_tokens_gt=50000, idle_minutes_gt=9)) == "resident context within threshold"
    assert IdleContextCandidates.skip_reason(row, now=now, parameters=IdleContextResetParameters(
        resident_tokens_gt=49999, idle_minutes_gt=9)) is None


def test_job_passes_snapshot_and_records_effective_parameters():
    lifecycle = SimpleNamespace(submit=AsyncMock(return_value=NewSessionResult(
        outcome=Outcome.SKIPPED, stage=Stage.ADMISSION, detail="busy")))
    params = IdleContextResetParameters(resident_tokens_gt=40000, idle_minutes_gt=5)
    job = IdleContextResetJob(None, None, lifecycle=lifecycle, parameters=params)
    job.candidates.read = Mock(return_value=[candidate()])
    result = asyncio.run(job.run(job_id="job", run_id="run", target_bot="*"))
    assert lifecycle.submit.call_args.args[0].scheduler_parameters == params
    assert result["parameters"] == params.model_dump(mode="json")


@pytest.mark.parametrize("original_parameters", [None, IdleContextResetParameters(idle_minutes_gt=7)])
def test_retry_after_settings_edit_reattaches_without_replacing_original_policy(monkeypatch, original_parameters):
    from llm_bawt.service.session_lifecycle import lifecycle as module

    stored = NewSessionLifecycle._encode(intent(original_parameters))
    record = SimpleNamespace(metadata={"new_session_intent": stored})
    engine = MagicMock()
    engine.begin.return_value.__enter__.return_value.execute.return_value.mappings.return_value.first.return_value = {"id": "existing"}
    store = SimpleNamespace(engine=engine, enqueue=Mock())
    service = SimpleNamespace(config=object(), _inter_bot_dispatcher=SimpleNamespace(store=store))
    monkeypatch.setattr(module, "BotManager", lambda config: SimpleNamespace(get_bot=lambda bot: object()))
    monkeypatch.setattr(module, "durable_agent_backend", lambda bot: "claude-code")
    monkeypatch.setattr(module, "lock_target", lambda *args: None)
    monkeypatch.setattr(module.DeliveryRecord, "from_mapping", lambda row: record)
    operation = NewSessionLifecycle(service)
    result, reason = operation._admit(intent(IdleContextResetParameters(idle_minutes_gt=20)))
    assert result is record and reason is None
    assert record.metadata["new_session_intent"] == NewSessionLifecycle._encode(intent(original_parameters))
    store.enqueue.assert_not_called()


def test_claim_reconstructs_durable_policy_before_rechecking(monkeypatch):
    params = IdleContextResetParameters(resident_tokens_gt=25000, idle_minutes_gt=3)
    checked = []
    def recheck(conn, parsed, **kwargs):
        checked.append(parsed.scheduler_parameters)
        return None
    monkeypatch.setattr(NewSessionLifecycle, "scheduled_skip_reason", recheck)
    assert not NewSessionLifecycle.skip_stale_claim(Mock(), {
        "id": "delivery", "metadata_json": {"new_session_intent": NewSessionLifecycle._encode(intent(params))},
    })
    assert checked == [params]


def test_registry_rejects_nonobjects_and_leaves_legacy_configs_unparsed():
    with pytest.raises(ValueError):
        JobParameterRegistry.resolve("idle_context_reset", "[]")
    assert JobParameterRegistry.describe("memory_decay", "not-json")["configuration_error"] is None


def test_scheduler_uses_saved_parameters_and_records_them(api, monkeypatch):
    client, engine = api
    client.patch("/v1/jobs/reset", json={"enabled": True, "parameters": {
        "resident_tokens_gt": 42000, "idle_minutes_gt": 7}})
    # SQLite lacks timestamptz; null is due on both dialects.
    with Session(engine) as session:
        job = session.get(ScheduledJob, "reset")
        job.next_run_at = None
        session.add(job)
        session.commit()
        session.refresh(job)
        session.expunge(job)
    observed = []
    async def run(self, **kwargs):
        observed.append(self.parameters)
        return {"reset": 0, "failed": 0, "ambiguous": 0,
                "parameters": self.parameters.model_dump(mode="json")}
    monkeypatch.setattr(IdleContextResetJob, "run", run)
    scheduler = JobScheduler(engine, SimpleNamespace())
    monkeypatch.setattr(scheduler, "_has_new_activity_for_job", lambda job: (True, None))
    asyncio.run(scheduler._execute_single_job(job))
    assert observed[0].resident_tokens_gt == 42000 and observed[0].idle_minutes_gt == 7
    with Session(engine) as session:
        run = session.exec(select(JobRun)).one()
        data = json.loads(run.result_json)
        assert data["parameters"]["resident_tokens_gt"] == 42000
        assert data["interval_minutes"] == 5


def test_failed_sweep_keeps_snapshot_visible_during_and_after_execution(api, monkeypatch):
    _, engine = api
    saved = {"resident_tokens_gt": 350000, "idle_minutes_gt": 20,
             "protected_bot_ids": ["fixture-protected"]}
    with Session(engine) as session:
        job = session.get(ScheduledJob, "reset")
        job.enabled, job.config_json = True, json.dumps(saved)
        session.add(job)
        session.commit()
        session.refresh(job)
        session.expunge(job)

    async def fail_read(self, **kwargs):
        with Session(engine) as session:
            running = session.exec(select(JobRun)).one()
            assert running.status.value == "running"
            assert json.loads(running.result_json) == {"parameters": saved, "interval_minutes": 5}
            # A concurrent settings edit must not rewrite the running policy receipt.
            current = session.get(ScheduledJob, "reset")
            current.config_json = '{"resident_tokens_gt":999999}'
            current.interval_minutes = 90
            session.add(current)
            session.commit()
        raise RuntimeError("synthetic candidate-read failure")

    monkeypatch.setattr(IdleContextResetJob, "run", fail_read)
    scheduler = JobScheduler(engine, SimpleNamespace())
    monkeypatch.setattr(scheduler, "_has_new_activity_for_job", lambda job: (True, None))
    asyncio.run(scheduler._execute_single_job(job))
    with Session(engine) as session:
        failed = session.exec(select(JobRun)).one()
        assert failed.status.value == "failed"
        assert failed.error_message == "synthetic candidate-read failure"
        assert json.loads(failed.result_json) == {"parameters": saved, "interval_minutes": 5}


def test_invalid_claimed_configuration_is_recorded_without_executing(api, monkeypatch):
    _, engine = api
    with Session(engine) as session:
        job = session.get(ScheduledJob, "reset")
        job.enabled, job.config_json = True, '{"idle_minutes_gt":-1}'
        session.add(job)
        session.commit()
        session.refresh(job)
        session.expunge(job)
    execute = AsyncMock(side_effect=AssertionError("Invalid config must not execute"))
    monkeypatch.setattr(IdleContextResetJob, "run", execute)
    scheduler = JobScheduler(engine, SimpleNamespace())
    monkeypatch.setattr(scheduler, "_has_new_activity_for_job", lambda job: (True, None))
    asyncio.run(scheduler._execute_single_job(job))
    execute.assert_not_awaited()
    with Session(engine) as session:
        failed = session.exec(select(JobRun)).one()
        assert failed.status.value == "failed"
        snapshot = json.loads(failed.result_json)
        assert snapshot["parameters"] is None and snapshot["interval_minutes"] == 5
        assert snapshot["configuration_error"] == failed.error_message
