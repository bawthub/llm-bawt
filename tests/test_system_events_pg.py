"""Commit-ordered event cursor under competing PostgreSQL writers (TASK-1745)."""

from __future__ import annotations

import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest
from sqlalchemy import text
from sqlmodel import Session

from llm_bawt.ops.store import OpsStore
from llm_bawt.profiles import EntityProfile, EntityType
from llm_bawt.system_events import SystemEventEmitter
from tests.pg_lock_wait import wait_blocked_by

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]


def test_after_cursor_cannot_skip_a_late_commit(private_app_pg):
    engine = OpsStore(private_app_pg).engine
    EntityProfile.__table__.create(engine, checkfirst=True)
    events = SystemEventEmitter(engine)
    user = f"event-reader-{uuid.uuid4().hex}"
    with Session(engine) as session:
        session.add(EntityProfile(entity_type=EntityType.USER, entity_id=user))
        session.commit()

    def competing_append():
        with engine.begin() as conn:
            return events.append(conn, event_key=f"{user}:second", kind="ops",
                                 source="ops", user_id=user, tone="info", summary="second")

    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            with engine.begin() as holder:
                first = events.append(holder, event_key=f"{user}:first", kind="ops",
                                      source="ops", user_id=user, tone="info", summary="first")
                assert first is not None
                pid = holder.execute(text("SELECT pg_backend_pid()")).scalar_one()
                second = pool.submit(competing_append)
                wait_blocked_by(engine, pid, second)
                assert events.list_user(user, after=0) == []
            second_id = second.result()
            assert second_id is not None and second_id > first
            assert [row["id"] for row in events.list_user(user, after=first)] == [str(second_id)]
    finally:
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM system_events WHERE event_key IN (:first, :second)"),
                         {"first": f"{user}:first", "second": f"{user}:second"})
            conn.execute(text("DELETE FROM entity_profiles WHERE entity_type = 'USER' AND entity_id = :user"),
                         {"user": user})
