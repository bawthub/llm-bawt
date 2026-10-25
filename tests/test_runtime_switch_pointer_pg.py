"""Runtime changes preserve bot-pointer ownership and atomic rotation."""
from __future__ import annotations

import uuid
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest
from sqlalchemy import text

from llm_bawt.memory.session_reset_lineage import SessionResetLineage
from llm_bawt.service.runtime_switch_rotation import rotate_active_threads
from llm_bawt.service.turn_admission import lock_target
from tests.pg_lock_wait import wait_blocked_by

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]


def extra_active(rig):
    other = str(uuid.uuid4())
    with rig.engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO sessions (id, bot_id, user_id, status, started_at)
            VALUES (:id, :bot, 'zzz-other-owner', 'active', now())
        """), {"id": other, "bot": rig.bot})
    return other


def test_runtime_switch_pointer_follows_its_owner_only(rig):
    other = extra_active(rig)
    successors = dict(rotate_active_threads(rig.engine, rig.bot))
    with rig.engine.connect() as conn:
        assert conn.execute(text("SELECT last_session_id FROM bot_profiles WHERE slug=:bot"),
                            {"bot": rig.bot}).scalar_one() == successors[rig.user]
        rows = conn.execute(text("SELECT id, status, session_metadata FROM sessions WHERE bot_id=:bot"),
                            {"bot": rig.bot}).mappings().all()
        by_id = {row["id"]: row for row in rows}
        assert by_id[rig.session_id]["session_metadata"]["reset_successor_id"] == successors[rig.user]
        assert by_id[other]["session_metadata"]["reset_successor_id"] == successors["zzz-other-owner"]
        assert by_id[successors[rig.user]]["session_metadata"].get("agent_session_keys") is None
        assert conn.execute(text("SELECT COUNT(*) FROM messages WHERE bot_id=:bot"),
                            {"bot": rig.bot}).scalar_one() == 1  # No confirmation UX.


def test_runtime_switch_multiple_owners_roll_back_as_one_transaction(rig, monkeypatch):
    other = extra_active(rig)
    rotate = SessionResetLineage.rotate
    def fail_second(engine, **kwargs):
        if kwargs["user_id"] == "zzz-other-owner":
            raise RuntimeError("injected second-owner failure")
        return rotate(engine, **kwargs)
    monkeypatch.setattr(SessionResetLineage, "rotate", fail_second)
    with pytest.raises(RuntimeError, match="second-owner"):
        rotate_active_threads(rig.engine, rig.bot)
    with rig.engine.connect() as conn:
        assert conn.execute(text("SELECT last_session_id FROM bot_profiles WHERE slug=:bot"),
                            {"bot": rig.bot}).scalar_one() == rig.session_id
        assert set(conn.execute(text("SELECT id FROM sessions WHERE bot_id=:bot AND status='active'"),
                                {"bot": rig.bot}).scalars()) == {rig.session_id, other}


def test_runtime_switch_takes_target_lock_before_selecting_sessions(rig):
    started = Event()
    def rotate():
        started.set()
        return rotate_active_threads(rig.engine, rig.bot)
    with ThreadPoolExecutor(max_workers=1) as pool:
        with rig.engine.begin() as conn:
            lock_target(conn, rig.bot)
            pending = pool.submit(rotate)
            assert started.wait(timeout=2)
            holder_pid = conn.execute(text("SELECT pg_backend_pid()")).scalar_one()
            wait_blocked_by(rig.engine, holder_pid, pending)
            assert conn.execute(text("SELECT last_session_id FROM bot_profiles WHERE slug=:bot"),
                                {"bot": rig.bot}).scalar_one() == rig.session_id
        successors = dict(pending.result(timeout=5))
        assert successors[rig.user] != rig.session_id
