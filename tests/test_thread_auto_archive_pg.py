"""TASK-261: daily archive of active threads idle past the configured age."""
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from sqlalchemy import create_engine, text

from llm_bawt.service.jobs.thread_auto_archive import run_thread_auto_archive
from tests.pg_lock_wait import wait_blocked_by
from tests.private_postgres import private_url

pytestmark = pytest.mark.integration

DAY = 86400


@pytest.fixture
def db(private_pg, private_module_db, monkeypatch):
    from llm_bawt.utils import db as db_mod

    engine = create_engine(private_url(private_pg, private_module_db))
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE sessions (
                id varchar(36) PRIMARY KEY, bot_id varchar(64) NOT NULL,
                user_id varchar(64), started_at timestamp NOT NULL,
                ended_at timestamp, archived_at timestamp,
                status varchar(16) NOT NULL DEFAULT 'active', session_metadata jsonb)
        """))
        conn.execute(text("""
            CREATE TABLE messages (
                id varchar(64), bot_id varchar(64), session_id varchar(36),
                role varchar(20) NOT NULL, timestamp double precision NOT NULL)
        """))
        # Production has this FK; its KEY SHARE lock is what fences the sweep.
        conn.execute(text(
            "ALTER TABLE messages ADD FOREIGN KEY (session_id) REFERENCES sessions(id)"
        ))
    monkeypatch.setattr(db_mod, "get_shared_engine", lambda config: engine)
    try:
        yield engine
    finally:
        with engine.begin() as conn:
            conn.execute(text("DROP TABLE messages; DROP TABLE sessions"))
        engine.dispose()


def add_thread(engine, sid, *, started_days, status="active", messages=(), meta=None):
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO sessions (id, bot_id, user_id, started_at, status, session_metadata)
            VALUES (:id, 'bot-a', :id, (NOW() AT TIME ZONE 'UTC') - make_interval(days => :d),
                    :status, CAST(:meta AS JSONB))
        """), {"id": sid, "d": started_days, "status": status, "meta": meta})
        for i, (role, days_ago) in enumerate(messages):
            conn.execute(text(
                "INSERT INTO messages VALUES (:mid, 'bot-a', :sid, :role, :ts)"
            ), {"mid": f"{sid}-{i}", "sid": sid, "role": role, "ts": time.time() - days_ago * DAY})


def status(engine, sid):
    with engine.connect() as conn:
        return conn.execute(text(
            "SELECT status, archived_at IS NOT NULL, session_metadata FROM sessions WHERE id=:id"
        ), {"id": sid}).one()


def test_archives_only_active_threads_idle_past_cutoff(db):
    add_thread(db, "stale", started_days=60, messages=[("user", 45), ("assistant", 45)])
    add_thread(db, "empty-old", started_days=40)
    add_thread(db, "recent-msg", started_days=60, messages=[("user", 2)])
    add_thread(db, "young", started_days=5)
    # Only user/assistant messages count as activity.
    add_thread(db, "tool-only", started_days=60, messages=[("user", 50), ("tool", 1)])
    add_thread(db, "already", started_days=90, status="archived")
    add_thread(db, "deleted", started_days=90, status="deleted")
    add_thread(db, "meta", started_days=60, meta='{"title": "kept"}')

    result = run_thread_auto_archive(object(), days=30)

    assert sorted(row["id"] for row in result["sessions"]) == ["empty-old", "meta", "stale", "tool-only"]
    assert result["archived"] == 4
    for sid in ("stale", "empty-old", "tool-only"):
        assert status(db, sid)[:2] == ("archived", True)
        assert status(db, sid)[2] == {"archived_via": "auto"}
    assert status(db, "meta")[2] == {"title": "kept", "archived_via": "auto"}
    assert status(db, "recent-msg")[0] == "active"
    assert status(db, "young")[0] == "active"
    assert status(db, "already")[2] is None
    assert status(db, "deleted")[0] == "deleted"
    # Idempotent: a second sweep finds nothing new.
    assert run_thread_auto_archive(object(), days=30)["archived"] == 0


def _new_message_sql():
    return text("INSERT INTO messages VALUES ('incoming', 'bot-a', 'race', 'user', "
                "EXTRACT(EPOCH FROM NOW()))")


def test_message_committed_while_sweep_waits_keeps_thread_active(db):
    """Codex's TASK-261 reproduction: a writer holds the row, a message commits
    during the sweep's lock wait; the sweep must see it."""
    add_thread(db, "race", started_days=60, messages=[("user", 45)])
    holder = db.connect()
    tx = holder.begin()
    holder.execute(text("UPDATE sessions SET session_metadata = '{}'::jsonb WHERE id = 'race'"))
    holder_pid = holder.execute(text("SELECT pg_backend_pid()")).scalar_one()
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        sweep = pool.submit(run_thread_auto_archive, object(), days=30)
        wait_blocked_by(db, holder_pid, sweep, timeout=8)
        with db.begin() as writer:
            writer.execute(_new_message_sql())
        tx.commit()
        assert sweep.result(timeout=8)["archived"] == 0
        assert status(db, "race")[0] == "active"
    finally:
        if tx.is_active:
            tx.rollback()
        holder.close()
        pool.shutdown(wait=True)


def test_in_flight_message_insert_blocks_sweep_until_commit(db):
    """An uncommitted message INSERT holds KEY SHARE on its thread; the sweep's
    FOR UPDATE waits for it and then sees the message."""
    add_thread(db, "race", started_days=60, messages=[("user", 45)])
    writer = db.connect()
    tx = writer.begin()
    writer.execute(_new_message_sql())
    writer_pid = writer.execute(text("SELECT pg_backend_pid()")).scalar_one()
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        sweep = pool.submit(run_thread_auto_archive, object(), days=30)
        wait_blocked_by(db, writer_pid, sweep, timeout=8)
        tx.commit()
        assert sweep.result(timeout=8)["archived"] == 0
        assert status(db, "race")[0] == "active"
    finally:
        if tx.is_active:
            tx.rollback()
        writer.close()
        pool.shutdown(wait=True)


def test_zero_days_disables(db):
    add_thread(db, "stale", started_days=60)
    result = run_thread_auto_archive(object(), days=0)
    assert result["disabled"] is True and result["archived"] == 0
    assert status(db, "stale")[0] == "active"


def test_scheduler_builds_archive_task():
    from llm_bawt.service.scheduler import JobScheduler, JobType, ScheduledJob
    from llm_bawt.service.tasks import TaskType

    job = ScheduledJob(job_type=JobType.THREAD_AUTO_ARCHIVE, bot_id="system")
    task = JobScheduler._build_task_for_job(None, job, {"days": 12, "user_id": "system"})
    assert task.task_type == TaskType.THREAD_AUTO_ARCHIVE
    assert task.payload == {"days": 12}
