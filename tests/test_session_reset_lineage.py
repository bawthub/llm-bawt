"""Hermetic /new lineage coverage: no live sessions or agent calls."""
import json
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool

from llm_bawt.memory.postgresql_short_term import PostgreSQLShortTermManager
from llm_bawt.memory.session_reset_lineage import SessionResetLineage
from llm_bawt.service.approval_continuation_session import approval_delivery_session


@pytest.fixture
def sessions():
    engine = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    with engine.begin() as conn:
        conn.execute(text("""CREATE TABLE sessions (
            id TEXT PRIMARY KEY, bot_id TEXT, user_id TEXT, status TEXT,
            started_at TIMESTAMP, ended_at TIMESTAMP, archived_at TIMESTAMP, session_metadata TEXT
        )"""))
    yield engine
    engine.dispose()


def insert(engine, id, *, bot="test-bot", user="test-user", status="active", metadata=None):
    with engine.begin() as conn:
        conn.execute(text("INSERT INTO sessions (id,bot_id,user_id,status,session_metadata) VALUES (:id,:bot,:user,:status,:meta)"),
                     dict(id=id, bot=bot, user=user, status=status, meta=json.dumps(metadata or {})))


def read(engine, id):
    with engine.connect() as conn:
        row = conn.execute(text("SELECT * FROM sessions WHERE id=:id"), {"id": id}).mappings().first()
        return dict(row) if row else None


def rotate(engine):
    manager = object.__new__(PostgreSQLShortTermManager)
    manager._backend = SimpleNamespace(engine=engine)
    manager.bot_id, manager.user_id = "test-bot", "test-user"
    return manager.rotate_session()


def resolve(engine, origin):
    return SessionResetLineage.resolve(lambda id: read(engine, id), origin, bot_id="test-bot", user_id="test-user")


def test_new_records_durable_chain_without_copying_old_sdk_context(sessions):
    insert(sessions, "old", metadata={"agent_session_keys": {"claude_code": "huge-sdk"}, "title": "old"})
    insert(sessions, "other-user", user="someone-else")
    insert(sessions, "other-bot", bot="someone-else")
    first = rotate(sessions)
    latest = rotate(sessions)
    # Reconstruct from database reads, not cached Python/bridge state.
    assert resolve(sessions, "old") == latest
    assert resolve(sessions, first) == latest
    assert resolve(sessions, latest) == latest
    assert read(sessions, "old")["status"] == "archived"
    old_meta = json.loads(read(sessions, "old")["session_metadata"])
    new_meta = json.loads(read(sessions, latest)["session_metadata"])
    assert old_meta["agent_session_keys"]["claude_code"] == "huge-sdk"
    assert old_meta["title"] == "old"
    assert "agent_session_keys" not in new_meta
    assert read(sessions, "other-user")["status"] == "active"
    assert read(sessions, "other-bot")["status"] == "active"


def test_archived_task_thread_is_not_redirected_to_unrelated_active_session(sessions):
    insert(sessions, "task", status="archived", metadata={"source": "dispatch"})
    insert(sessions, "chat")
    rotate(sessions)
    assert resolve(sessions, "task") == "task"


def test_rotation_failure_rolls_back_archive_and_link(sessions):
    insert(sessions, "old")
    insert(sessions, "collision", status="archived")
    with pytest.raises(Exception):
        SessionResetLineage.rotate(sessions, bot_id="test-bot", user_id="test-user", new_id="collision")
    assert read(sessions, "old")["status"] == "active"
    assert "reset_successor_id" not in json.loads(read(sessions, "old")["session_metadata"])


@pytest.mark.parametrize("bad", ["missing", "other-user", "other-bot", "deleted", "cycle", "unlinked"])
def test_invalid_reset_chain_fails_closed(sessions, bad):
    insert(sessions, "old", status="archived", metadata={"reset_successor_id": "next", "reset_predecessor_ids": ["next"]})
    if bad != "missing":
        metadata = {"reset_predecessor_ids": ["old"]}
        if bad == "cycle":
            metadata["reset_successor_id"] = "old"
        if bad == "unlinked":
            metadata = {}
        insert(sessions, "next", user="other" if bad == "other-user" else "test-user",
               bot="other" if bad == "other-bot" else "test-bot",
               status="deleted" if bad == "deleted" else "active", metadata=metadata)
    with pytest.raises(ValueError):
        resolve(sessions, "old")


def _approval_row(origin, created_at, kind="mcp"):
    return SimpleNamespace(caller_context_json=json.dumps({"session_id": origin}), bot_id="test-bot",
                           user_id="test-user", request_kind=kind, created_at=created_at)


def _service(engine):
    return SimpleNamespace(get_memory_client=lambda bot, user: SimpleNamespace(get_session=lambda id: read(engine, id)))


def test_rotation_stamps_absolute_reset_time(sessions):
    insert(sessions, "old")
    before = time.time()
    rotate(sessions)
    reset_at = json.loads(read(sessions, "old")["session_metadata"])["reset_at"]
    assert before <= reset_at <= time.time()


@pytest.mark.parametrize("kind", ["mcp", "harness"])
def test_approval_from_reopened_archived_thread_stays_there(sessions, kind):
    # /new happened first; the user then reopened the archived thread and the
    # approval was raised there. Its result belongs to that thread, not /new.
    insert(sessions, "old")
    rotate(sessions)
    created = datetime.now(timezone.utc) + timedelta(seconds=5)
    assert approval_delivery_session(_service(sessions), _approval_row("old", created, kind)) == "old"


def test_reset_while_pending_still_follows_new(sessions):
    insert(sessions, "old")
    created = datetime.now(timezone.utc) - timedelta(seconds=5)
    latest = rotate(sessions)
    assert approval_delivery_session(_service(sessions), _approval_row("old", created)) == latest


def test_reopened_thread_stays_put_even_with_later_resets(sessions):
    insert(sessions, "old")
    rotate(sessions)
    created = datetime.now(timezone.utc) + timedelta(seconds=5)
    rotate(sessions)
    assert approval_delivery_session(_service(sessions), _approval_row("old", created)) == "old"


def test_native_grant_cannot_be_rebound_to_fresh_sdk_session(sessions):
    insert(sessions, "old")
    rotate(sessions)
    row = SimpleNamespace(caller_context_json='{"session_id":"old"}', bot_id="test-bot", user_id="test-user", request_kind="harness")
    service = SimpleNamespace(get_memory_client=lambda bot, user: SimpleNamespace(get_session=lambda id: read(sessions, id)))
    with pytest.raises(ValueError, match="fresh request"):
        approval_delivery_session(service, row)
