"""Private PostgreSQL acceptance for pointer selection and ONE scheduler run.

Reset execution is tested through the shared lifecycle, not through a second
scheduler-owned rotation implementation. Snapshot regressions remain in
``test_turn_context_snapshot_pg.py``.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import event, text
from sqlmodel import Session, select

from llm_bawt.service.jobs.idle_context_candidates import IdleContextCandidates
from llm_bawt.service.jobs.idle_context_reset import IdleContextResetJob
from llm_bawt.service.scheduler import JobRun, JobScheduler, JobStatus, JobType, ScheduledJob
from llm_bawt.service.session_lifecycle.intent import NewSessionResult, Outcome, Stage

from llm_bawt.service.session_lifecycle.lifecycle import NewSessionLifecycle
from tests.bot_tag_fixtures import PRIVATE_MODES, clear_tags, make_private, tag
from tests.session_lifecycle_pg_fixtures import scheduled_intent

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]


def select_one(rig):
    return IdleContextCandidates(rig.engine).read(rig.bot)[0]


def schedule(rig, *, enabled=True, wildcard=False):
    job = ScheduledJob(job_type=JobType.IDLE_CONTEXT_RESET,
                       bot_id="*" if wildcard else rig.bot,
                       enabled=enabled, interval_minutes=5)
    rig.jobs.append(job.id)
    with Session(rig.engine) as session:
        session.add(job)
        session.commit()
        session.refresh(job)
        session.expunge(job)
    return job


def completed(rig):
    return NewSessionResult(outcome=Outcome.COMPLETED, stage=Stage.RECEIPT,
                            old_session_id=rig.session_id, new_session_id="successor",
                            receipt_message_id="persisted-receipt", delivery_id="delivery")


def inject_lifecycle(monkeypatch, lifecycle):
    monkeypatch.setattr("llm_bawt.service.jobs.idle_context_reset.IdleContextResetJob",
                        lambda *args, **kwargs: IdleContextResetJob(*args, lifecycle=lifecycle, **kwargs))


def test_one_pointer_not_one_session_per_user(rig):
    with rig.engine.begin() as conn:
        conn.execute(text("UPDATE sessions SET status='archived' WHERE id=:id"), {"id": rig.session_id})
        for index in range(40):
            conn.execute(text("""
                INSERT INTO sessions (id, bot_id, user_id, status, started_at)
                VALUES (:id, :bot, :user, 'active', now())
            """), {"id": str(uuid.uuid4()), "bot": rig.bot, "user": f"other-{index}"})
    rows = IdleContextCandidates(rig.engine).read(rig.bot)
    assert len(rows) == 1
    assert rows[0]["session_id"] == rig.session_id
    assert rows[0]["user_id"] == rig.user
    assert rows[0]["usage_turn_id"] == rig.turn_id
    assert IdleContextCandidates.skip_reason(rows[0], now=rig.now) is None


@pytest.mark.parametrize("kind,mode", [*(("private", m) for m in PRIVATE_MODES), ("inactive", None)])
def test_private_and_inactive_bots_are_never_candidates(rig, kind, mode):
    assert select_one(rig)["session_id"] == rig.session_id  # the control
    with rig.engine.begin() as conn:
        if kind == "private":
            make_private(conn, rig.bot, mode)
        else:
            tag(conn, rig.bot, kind)
    try:
        candidates = IdleContextCandidates(rig.engine)
        assert candidates.read(rig.bot) == []
        assert rig.bot not in {row["bot_id"] for row in candidates.read("*")}
        with rig.engine.begin() as conn:
            assert NewSessionLifecycle.scheduled_skip_reason(conn, scheduled_intent(rig)) == f"{kind} bot"
    finally:
        with rig.engine.begin() as conn:
            clear_tags(conn, rig.bot)


def test_null_pointer_does_not_fall_back_to_active_session(rig):
    with rig.engine.begin() as conn:
        conn.execute(text("UPDATE bot_profiles SET last_session_id=NULL WHERE slug=:bot"), {"bot": rig.bot})
    row = select_one(rig)
    assert row["session_id"] is None
    assert IdleContextCandidates.skip_reason(row, now=rig.now) == "no valid accepted session"


def test_deleted_session_fails_closed(rig):
    with rig.engine.begin() as conn:
        conn.execute(text("UPDATE sessions SET status='deleted' WHERE id=:id"), {"id": rig.session_id})
    assert select_one(rig)["session_id"] is None


def test_synthetic_protected_bot_excluded_before_session_scan(rig):
    selector = IdleContextCandidates(rig.engine, protected_bot_ids=[rig.bot.upper()])
    assert selector.read("*") == []
    with rig.engine.connect() as conn:
        assert selector.read_one(conn, rig.bot) is None


def test_connection_recheck_uses_callers_uncommitted_pointer(rig):
    selector = IdleContextCandidates(rig.engine)
    with rig.engine.connect() as conn:
        tx = conn.begin()
        conn.execute(text("UPDATE bot_profiles SET last_session_id=NULL WHERE slug=:bot"), {"bot": rig.bot})
        assert selector.read_one(conn, rig.bot)["session_id"] is None
        tx.rollback()
    assert select_one(rig)["session_id"] == rig.session_id


def test_newer_unlinked_work_invalidates_old_measurement(rig):
    with rig.engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO turn_logs (id, created_at, ended_at, path, stream,
                bot_id, user_id, status, user_prompt, response_text)
            VALUES (:id, now(), now(), '/test', false, :bot,
                'another-owner', 'error', '', '')
        """), {"id": f"turn-{uuid.uuid4().hex}", "bot": rig.bot})
    row = select_one(rig)
    assert row["usage_turn_id"] == rig.turn_id
    assert IdleContextCandidates.skip_reason(row, now=rig.now) == "newer bot activity invalidated measurement"


def test_selector_performs_one_read_and_no_mutations(rig):
    statements = []
    def capture(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)
    event.listen(rig.engine, "before_cursor_execute", capture)
    try:
        assert select_one(rig)["session_id"] == rig.session_id
    finally:
        event.remove(rig.engine, "before_cursor_execute", capture)
    assert len(statements) == 1
    assert statements[0].lstrip().startswith("WITH targets AS MATERIALIZED")


def test_scheduler_owns_one_run_with_real_duration_and_result(rig, monkeypatch):
    lifecycle = SimpleNamespace(submit=AsyncMock(return_value=completed(rig)))
    inject_lifecycle(monkeypatch, lifecycle)
    job = schedule(rig, wildcard=True)
    scheduler = JobScheduler(rig.engine, SimpleNamespace(config=rig.config))
    asyncio.run(scheduler._execute_job(job))
    with Session(rig.engine) as session:
        runs = session.exec(select(JobRun).where(JobRun.job_id == job.id)).all()
        assert len(runs) == 1 and runs[0].bot_id == "*"
        run = runs[0]
        assert run.status is JobStatus.SUCCESS
        assert run.finished_at >= run.started_at and run.duration_ms >= 0
        result = json.loads(run.result_json)
        assert result["reset"] == 1
        assert result["bots"][rig.bot]["receipt_message_id"] == "persisted-receipt"
        intent = lifecycle.submit.call_args.args[0]
        assert intent.run_id == run.id and intent.job_id == job.id
        assert intent.user_id == rig.user and intent.author.entity_id == "system"
        current = session.get(ScheduledJob, job.id)
        assert current.next_run_at - current.last_run_at == timedelta(minutes=5)
    # The mocked handler didn't rotate: the scheduler must not do it itself.
    assert select_one(rig)["session_id"] == rig.session_id


def test_concurrent_due_checks_claim_only_one_occurrence(rig, monkeypatch):
    lifecycle = SimpleNamespace(submit=AsyncMock(return_value=completed(rig)))
    inject_lifecycle(monkeypatch, lifecycle)
    job = schedule(rig)
    def execute():
        scheduler = JobScheduler(rig.engine, SimpleNamespace(config=rig.config))
        asyncio.run(scheduler._execute_job(job))
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(execute) for _ in range(2)]
        for future in futures:
            future.result(timeout=20)
    lifecycle.submit.assert_awaited_once()
    with Session(rig.engine) as session:
        assert len(session.exec(select(JobRun).where(JobRun.job_id == job.id)).all()) == 1


def test_disabled_job_cannot_be_run_from_stale_due_list(rig, monkeypatch):
    lifecycle = SimpleNamespace(submit=AsyncMock(return_value=completed(rig)))
    inject_lifecycle(monkeypatch, lifecycle)
    job = schedule(rig, enabled=False)
    asyncio.run(JobScheduler(rig.engine, SimpleNamespace(config=rig.config))._execute_job(job))
    lifecycle.submit.assert_not_awaited()
    with Session(rig.engine) as session:
        assert session.exec(select(JobRun).where(JobRun.job_id == job.id)).all() == []


@pytest.mark.parametrize("outcome,expected", [(Outcome.AMBIGUOUS, JobStatus.FAILED),
                                             (Outcome.FAILED, JobStatus.FAILED),
                                             (Outcome.SKIPPED, JobStatus.SKIPPED)])
def test_scheduler_does_not_label_unconfirmed_resets_successful(rig, monkeypatch, outcome, expected):
    lifecycle = SimpleNamespace(submit=AsyncMock(return_value=NewSessionResult(
        outcome=outcome, stage=Stage.RECEIPT, detail="unconfirmed or skipped",
    )))
    inject_lifecycle(monkeypatch, lifecycle)
    job = schedule(rig)
    asyncio.run(JobScheduler(rig.engine, SimpleNamespace(config=rig.config))._execute_job(job))
    with Session(rig.engine) as session:
        run = session.exec(select(JobRun).where(JobRun.job_id == job.id)).one()
        assert run.status is expected
        assert json.loads(run.result_json)["reset"] == 0


def test_real_submission_is_literal_system_command_and_keyed_retry(rig, real_lifecycle):
    first = asyncio.run(real_lifecycle.submit(scheduled_intent(rig), timeout_seconds=0))
    second = asyncio.run(real_lifecycle.submit(scheduled_intent(rig, run_id="next-run"), timeout_seconds=0))
    assert first.outcome is Outcome.AMBIGUOUS and second.outcome is Outcome.AMBIGUOUS
    assert first.delivery_id == second.delivery_id
    record = real_lifecycle.store.get(first.delivery_id)
    assert record.message == "/new"
    assert record.author_entity_type == "bot" and record.author_entity_id == "system"
    with rig.engine.connect() as conn:
        payload = conn.execute(text("SELECT payload_json FROM inter_bot_deliveries WHERE id=:id"),
                               {"id": record.id}).scalar_one()
        assert payload["messages"] == [{"role": "user", "content": "/new"}]
        assert payload["user"] == rig.user
        assert conn.execute(text("SELECT COUNT(*) FROM inter_bot_deliveries WHERE target_bot_id=:bot"),
                            {"bot": rig.bot}).scalar_one() == 1
        assert conn.execute(text("SELECT COUNT(*) FROM sessions WHERE bot_id=:bot"),
                            {"bot": rig.bot}).scalar_one() == 1


@pytest.mark.parametrize("change", ["pointer", "usage", "new_work", "queued_work"])
def test_real_claim_skips_stale_scheduled_intent_without_reset(rig, real_lifecycle, change):
    intent = scheduled_intent(rig)
    first = asyncio.run(real_lifecycle.submit(intent, timeout_seconds=0))
    assert first.delivery_id
    with rig.engine.begin() as conn:
        if change == "pointer":
            conn.execute(text("UPDATE bot_profiles SET last_session_id=NULL WHERE slug=:bot"), {"bot": rig.bot})
        elif change == "usage":
            conn.execute(text("UPDATE turn_logs SET token_usage_json=:usage WHERE id=:id"),
                         {"id": rig.turn_id, "usage": '{"resident_tokens":1000}'})
        elif change == "new_work":
            conn.execute(text("""
                INSERT INTO turn_logs (id, created_at, path, stream, bot_id, user_id,
                    status, user_prompt, response_text)
                VALUES (:id, now(), '/test', true, :bot, :user, 'reserved', '', '')
            """), {"id": f"new-{uuid.uuid4().hex}", "bot": rig.bot, "user": rig.user})
    if change == "queued_work":
        real_lifecycle.store.enqueue(sender_bot_id="test-sender", target_bot_id=rig.bot,
                                     message="later work", payload={"user": rig.user,
                                     "messages": [{"role": "user", "content": "later work"}]})
    assert real_lifecycle.store.claim_next(rig.bot, claim_owner="fixture", steer_capable=False) is None
    observed = asyncio.run(real_lifecycle.submit(intent, timeout_seconds=0))
    assert observed.outcome is Outcome.SKIPPED
    assert real_lifecycle.store.get(first.delivery_id).status == "FAILED"
    with rig.engine.connect() as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM sessions WHERE bot_id=:bot"),
                            {"bot": rig.bot}).scalar_one() == 1
        assert conn.execute(text("SELECT COUNT(*) FROM turn_logs WHERE id=:id"),
                            {"id": first.turn_id}).scalar_one() == 0


def test_real_scheduler_never_confirms_transport_only_delivery(rig, real_lifecycle):
    def wake(bot):
        claim = real_lifecycle.store.claim_next(bot, claim_owner="fixture", steer_capable=False)
        assert claim is not None
        real_lifecycle.store.mark_delivered(claim.id, claim.claim_token,
                                           response_model="test", response_chars=0)
    real_lifecycle.dispatcher.wake = wake
    job = schedule(rig)
    asyncio.run(JobScheduler(rig.engine, real_lifecycle.service)._execute_job(job))
    with Session(rig.engine) as session:
        run = session.exec(select(JobRun).where(JobRun.job_id == job.id)).one()
        result = json.loads(run.result_json)
        assert run.status is JobStatus.FAILED
        assert result["ambiguous"] == 1 and result["reset"] == 0
        assert result["bots"][rig.bot]["delivery_id"]
