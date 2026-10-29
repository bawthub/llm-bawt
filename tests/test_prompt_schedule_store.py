"""Hermetic persistence/producer tests. SQLite does not verify PG row locking."""
import asyncio
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import event
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, create_engine, select

from llm_bawt.service.prompt_schedule_models import PromptOccurrence, PromptSchedule
from llm_bawt.service.prompt_schedule_store import PromptScheduleStore, db_utc
from llm_bawt.service.prompt_timing import PromptTiming
from llm_bawt.service.scheduler import JobRun, JobScheduler, JobType, ScheduledJob, create_scheduler_tables

NOW = datetime(2026, 9, 15, 12, tzinfo=timezone.utc)


@pytest.fixture
def engine():
    engine = create_engine("sqlite://")
    @event.listens_for(engine, "connect")
    def foreign_keys(conn, _):
        conn.execute("PRAGMA foreign_keys=ON")
    create_scheduler_tables(engine)
    yield engine
    engine.dispose()


def create(store, **kwargs):
    return store.create(owner="test-owner", bot_id="test-scheduler", name="Test", prompt="Saved prompt",
                        timing=PromptTiming(kind="interval", interval_seconds=60, anchor_at=NOW + timedelta(minutes=1)),
                        now=NOW, **kwargs)


def test_schema_creation_idempotent_and_maintenance_untouched(engine):
    with Session(engine) as session:
        session.add(ScheduledJob(id="maintenance", job_type=JobType.MEMORY_DECAY, bot_id="test"))
        session.commit()
    create_scheduler_tables(engine)
    assert PromptScheduleStore(engine).reserve_due(NOW) == []
    with Session(engine) as session:
        assert session.get(ScheduledJob, "maintenance").next_run_at is None


def test_legacy_inactive_due_pointers_are_backfilled_without_touching_maintenance(engine):
    store = PromptScheduleStore(engine)
    active = create(store)
    paused = create(store, enabled=False)
    with Session(engine) as session:
        session.add(ScheduledJob(id="maintenance-due", job_type=JobType.MEMORY_DECAY,
                                 bot_id="test", next_run_at=NOW))
        job = session.get(ScheduledJob, paused)
        job.next_run_at = NOW  # Legacy paused row from before the upgrade.
        session.add(job)
        session.commit()
    create_scheduler_tables(engine)
    create_scheduler_tables(engine)  # Repeat startup is safe.
    with Session(engine) as session:
        assert session.get(ScheduledJob, paused).next_run_at is None
        assert session.get(ScheduledJob, active).next_run_at is not None
        assert db_utc(session.get(ScheduledJob, "maintenance-due").next_run_at) == NOW

def test_due_reservation_snapshot_and_pending_survive_new_store(engine):
    store = PromptScheduleStore(engine)
    job = create(store, requested_model="explicit-model")
    assert store.reserve_due(NOW) == []
    ids = store.reserve_due(NOW + timedelta(minutes=1, seconds=3))
    assert len(ids) == 1
    assert store.reserve_due(NOW + timedelta(minutes=1, seconds=4)) == []
    occurrence = PromptScheduleStore(engine).pending()[0]
    assert occurrence.run_id == ids[0]
    assert occurrence.snapshot_json["prompt"] == "Saved prompt"
    assert occurrence.snapshot_json["model"] == "explicit-model"
    assert occurrence.snapshot_json["clear_context"] is True
    with Session(engine) as session:
        assert db_utc(session.get(ScheduledJob, job).next_run_at) == NOW + timedelta(minutes=2)
        schedule = session.get(PromptSchedule, job)
        schedule.prompt = "Changed later"
        session.add(schedule)
        session.commit()
    assert store.pending()[0].snapshot_json["prompt"] == "Saved prompt"


def test_long_downtime_coalesces_without_backlog(engine):
    store = PromptScheduleStore(engine)
    create(store)
    ids = store.reserve_due(NOW + timedelta(days=400, seconds=10))
    assert len(ids) == 1
    assert db_utc(store.pending()[0].scheduled_for) == NOW + timedelta(days=400)
    assert store.reserve_due(NOW + timedelta(days=401)) == []


def test_completed_prior_run_allows_latest_pending_slot(engine):
    store = PromptScheduleStore(engine)
    create(store)
    first = store.reserve_due(NOW + timedelta(minutes=1))[0]
    assert store.reserve_due(NOW + timedelta(minutes=5)) == []
    with Session(engine) as session:
        occurrence = session.get(PromptOccurrence, first)
        occurrence.state = "succeeded"
        session.add(occurrence)
        session.commit()
    assert len(store.reserve_due(NOW + timedelta(minutes=5, seconds=10))) == 1
    assert db_utc(store.pending()[0].scheduled_for) == NOW + timedelta(minutes=5)


def test_skip_policy_skips_backlog_but_not_tick_latency(engine):
    store = PromptScheduleStore(engine)
    create(store, missed_policy="skip")
    assert store.reserve_due(NOW + timedelta(minutes=5)) == []
    with Session(engine) as session:
        occurrence = session.exec(select(PromptOccurrence)).one()
        assert occurrence.state == "skipped"
        assert occurrence.last_error == "Missed occurrences skipped"
    assert len(store.reserve_due(NOW + timedelta(minutes=6, seconds=4))) == 1


def test_expired_one_shot_terminal_skip(engine):
    store = PromptScheduleStore(engine)
    job = store.create(owner="owner", bot_id="test", name="One", prompt="Once", now=NOW,
                       timing=PromptTiming(kind="once", once_at=NOW + timedelta(minutes=1)),
                       misfire_grace_seconds=60)
    assert store.reserve_due(NOW + timedelta(hours=1)) == []
    with Session(engine) as session:
        assert session.get(PromptSchedule, job).lifecycle == "completed"
        assert session.get(ScheduledJob, job).enabled is False
        assert session.exec(select(PromptOccurrence)).one().state == "skipped"


def test_one_shot_pending_is_not_completed_or_reserved_again(engine):
    store = PromptScheduleStore(engine)
    job = store.create(owner="owner", bot_id="test", name="One", prompt="Once", now=NOW,
                       timing=PromptTiming(kind="once", once_at=NOW + timedelta(minutes=1)))
    assert len(store.reserve_due(NOW + timedelta(minutes=1))) == 1
    assert store.reserve_due(NOW + timedelta(minutes=2)) == []
    with Session(engine) as session:
        assert session.get(PromptSchedule, job).lifecycle == "active"
        assert session.get(ScheduledJob, job).next_run_at is None


def test_paused_schedule_never_claimed(engine):
    store = PromptScheduleStore(engine)
    create(store, enabled=False)
    assert store.reserve_due(NOW + timedelta(minutes=1)) == []


def test_active_jobs_do_not_starve_later_due_jobs(engine):
    store = PromptScheduleStore(engine)
    create(store)
    assert len(store.reserve_due(NOW + timedelta(minutes=1), limit=1)) == 1
    create(store)
    assert len(store.reserve_due(NOW + timedelta(minutes=5), limit=1)) == 1


def test_partial_unique_index_blocks_overlapping_occurrence(engine):
    store = PromptScheduleStore(engine)
    job = create(store)
    store.reserve_due(NOW + timedelta(minutes=1))
    with Session(engine) as session:
        run = JobRun(job_id=job, bot_id="test")
        session.add(run)
        session.flush()
        session.add(PromptOccurrence(run_id=run.id, job_id=job, occurrence_key="manual:new-key",
                                     kind="manual", scheduled_for=NOW, schedule_revision=1, snapshot_json={}))
        with pytest.raises(IntegrityError):
            session.commit()


def test_end_bound_prevents_late_start(engine):
    store = PromptScheduleStore(engine)
    job = store.create(owner="owner", bot_id="test", name="End", prompt="Bounded", now=NOW,
                       timing=PromptTiming(kind="interval", interval_seconds=60,
                                           anchor_at=NOW + timedelta(minutes=1), ends_at=NOW + timedelta(minutes=3)))
    assert store.reserve_due(NOW + timedelta(minutes=3)) == []
    with Session(engine) as session:
        assert session.get(PromptSchedule, job).lifecycle == "completed"
        assert session.exec(select(PromptOccurrence)).one().last_error == "Schedule end reached"


def test_reservation_failure_rolls_back_occurrence_and_due_pointer(engine):
    store = PromptScheduleStore(engine)
    job = create(store)
    def fail_occurrence_insert(_conn, _cursor, statement, _parameters, _context, _many):
        if statement.startswith("INSERT INTO prompt_occurrences"):
            raise RuntimeError("injected before outbox commit")
    event.listen(engine, "before_cursor_execute", fail_occurrence_insert)
    try:
        with pytest.raises(RuntimeError, match="injected"):
            store.reserve_due(NOW + timedelta(minutes=1))
    finally:
        event.remove(engine, "before_cursor_execute", fail_occurrence_insert)
    with Session(engine) as session:
        assert session.exec(select(JobRun)).all() == []
        assert session.exec(select(PromptOccurrence)).all() == []
        assert db_utc(session.get(ScheduledJob, job).next_run_at) == NOW + timedelta(minutes=1)
    assert len(store.reserve_due(NOW + timedelta(minutes=1))) == 1


def test_postgresql_schema_contains_partial_unique_and_foreign_keys():
    from sqlalchemy.dialects import postgresql
    from sqlalchemy.schema import CreateIndex, CreateTable
    ddl = str(CreateTable(PromptOccurrence.__table__).compile(dialect=postgresql.dialect()))
    assert "FOREIGN KEY(run_id) REFERENCES job_runs (id)" in ddl
    assert "UNIQUE (job_id, occurrence_key)" in ddl
    index = next(i for i in PromptOccurrence.__table__.indexes if i.name == "uq_prompt_occurrences_active")
    ddl = str(CreateIndex(index).compile(dialect=postgresql.dialect()))
    assert "UNIQUE INDEX" in ddl and "WHERE state IN ('pending','queued','running')" in ddl


def test_scheduler_prompt_loop_independent_of_blocked_maintenance():
    async def check():
        maintenance_started = asyncio.Event()
        prompt_ran = asyncio.Event()
        async def blocked():
            maintenance_started.set()
            await asyncio.Event().wait()
        async def prompt_sweep():
            prompt_ran.set()
        scheduler = JobScheduler(None, None, prompt_sweep=prompt_sweep)
        scheduler._check_and_run_due_jobs = blocked
        await scheduler.start()
        await asyncio.wait_for(maintenance_started.wait(), 1)
        await asyncio.wait_for(prompt_ran.wait(), 1)
        await scheduler.stop()
        assert scheduler._task.done() and scheduler._prompt_task.done()
    asyncio.run(check())


def test_scheduler_does_not_enable_unwired_producer():
    async def check():
        scheduler = JobScheduler(None, None)
        async def no_op():
            pass
        scheduler._check_and_run_due_jobs = no_op
        await scheduler.start()
        assert scheduler._prompt_task is None
        await scheduler.stop()
    asyncio.run(check())
