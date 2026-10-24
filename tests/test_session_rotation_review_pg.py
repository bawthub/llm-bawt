"""Independent rotation transaction/replay checks on private PostgreSQL."""
import pytest
from sqlalchemy import text

from llm_bawt.bot_session_state import BotSessionState
from llm_bawt.memory.session_reset_lineage import SessionResetLineage
from tests.test_bot_session_state_pg import conn, _bots, _sessions  # noqa: F401

pytestmark = pytest.mark.integration


def setup(conn):
    _bots(conn)
    _sessions(conn)
    BotSessionState.ensure_schema(conn)
    BotSessionState.accept(conn, bot_id="fixture", session_id="second")


def rotate(conn):
    return SessionResetLineage.rotate(
        None, connection=conn, bot_id="fixture", user_id="other-user",
        new_id="successor", idempotency_key="delivery-fixture",
        accept_pointer=True, working_memory_mode="pinned",
        context_lifecycle={"action": "reset_retain_history", "old_session_id": "second"},
    )


def test_rotation_pointer_and_lineage_rollback_with_caller_transaction(conn):
    setup(conn)
    with pytest.raises(RuntimeError), conn.begin_nested():
        assert rotate(conn) is True
        assert BotSessionState.resolve(conn, bot_id="fixture")["id"] == "successor"
        raise RuntimeError("reservation write failed after rotation")
    assert BotSessionState.resolve(conn, bot_id="fixture")["id"] == "second"
    assert conn.execute(text("SELECT status FROM sessions WHERE id='second'")).scalar_one() == "active"
    assert conn.execute(text("SELECT count(*) FROM sessions WHERE id='successor'")).scalar_one() == 0
    assert conn.execute(text("SELECT session_metadata FROM sessions WHERE id='second'")).scalar_one() is None


def test_keyed_replay_preserves_newer_pointer_and_original_lineage(conn):
    setup(conn)
    assert rotate(conn) is True
    old = conn.execute(text("SELECT session_metadata FROM sessions WHERE id='second'")).scalar_one()
    successor = conn.execute(text("SELECT session_metadata FROM sessions WHERE id='successor'")).scalar_one()
    assert old["reset_successor_id"] == "successor"
    assert successor["reset_predecessor_ids"] == ["second"]
    assert successor["context_lifecycle"]["old_session_id"] == "second"
    # An explicit archived-session turn accepted after rotation is authoritative.
    BotSessionState.accept(conn, bot_id="fixture", session_id="first")
    assert rotate(conn) is False
    assert BotSessionState.resolve(conn, bot_id="fixture")["id"] == "first"
    assert conn.execute(text("SELECT session_metadata FROM sessions WHERE id='successor'")).scalar_one() == successor


def test_pointer_failure_rolls_back_rotation_in_callers_transaction(conn):
    setup(conn)
    conn.execute(text("DELETE FROM bot_profiles WHERE slug='fixture'"))
    with pytest.raises(ValueError, match="Cannot accept"), conn.begin_nested():
        rotate(conn)
    assert conn.execute(text("SELECT status FROM sessions WHERE id='second'")).scalar_one() == "active"
    assert conn.execute(text("SELECT count(*) FROM sessions WHERE id='successor'")).scalar_one() == 0
