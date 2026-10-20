"""Bot-wide pointer semantics without operational databases or providers."""
from __future__ import annotations

import pytest
from sqlalchemy import create_engine, text

from llm_bawt.bot_session_state import BotSessionState


@pytest.fixture
def conn():
    engine = create_engine("sqlite://")
    with engine.begin() as connection:
        connection.exec_driver_sql("CREATE TABLE bot_profiles (slug TEXT PRIMARY KEY, last_session_id TEXT)")
        connection.exec_driver_sql("""CREATE TABLE sessions (
            id TEXT PRIMARY KEY, bot_id TEXT, user_id TEXT, status TEXT,
            started_at TEXT, ended_at TEXT, archived_at TEXT, session_metadata TEXT)
        """)
        connection.exec_driver_sql("INSERT INTO bot_profiles (slug) VALUES ('fixture'), ('other')")
        connection.exec_driver_sql("""INSERT INTO sessions (id, bot_id, user_id, status) VALUES
            ('old', 'fixture', 'nick', 'archived'),
            ('latest', 'fixture', 'another-user', 'active'),
            ('deleted', 'fixture', 'nick', 'deleted'),
            ('foreign', 'other', 'nick', 'active')
        """)
        yield connection
    engine.dispose()


def test_null_is_not_guessed_from_active_sessions(conn):
    assert BotSessionState.resolve(conn, bot_id="fixture") is None
    assert conn.execute(text("SELECT last_session_id FROM bot_profiles WHERE slug='fixture'")).scalar_one() is None


def test_accepted_session_is_bot_wide_not_author_scoped(conn):
    BotSessionState.accept(conn, bot_id="fixture", session_id="latest")
    resolved = BotSessionState.resolve(conn, bot_id="fixture")
    assert resolved["id"] == "latest"
    assert resolved["user_id"] == "another-user"
    assert "last_session_id" not in resolved


def test_explicit_archived_binding_becomes_last_used_session(conn):
    BotSessionState.accept(conn, bot_id="fixture", session_id="latest")
    BotSessionState.accept(conn, bot_id="fixture", session_id="old")
    assert BotSessionState.resolve(conn, bot_id="fixture")["id"] == "old"
    # Looking up another bot and repeating reads does not mutate this pointer.
    assert BotSessionState.resolve(conn, bot_id="other") is None
    assert BotSessionState.resolve(conn, bot_id="fixture")["id"] == "old"


@pytest.mark.parametrize("session_id", ["missing", "deleted", "foreign"])
def test_invalid_binding_does_not_overwrite_current_session(conn, session_id):
    BotSessionState.accept(conn, bot_id="fixture", session_id="old")
    with pytest.raises(ValueError, match="Cannot accept"):
        BotSessionState.accept(conn, bot_id="fixture", session_id=session_id)
    assert BotSessionState.resolve(conn, bot_id="fixture")["id"] == "old"


@pytest.mark.parametrize("session_id", ["missing", "deleted", "foreign"])
def test_corrupt_pointer_fails_closed_instead_of_falling_back(conn, session_id):
    conn.execute(text("UPDATE bot_profiles SET last_session_id=:id WHERE slug='fixture'"), {"id": session_id})
    with pytest.raises(ValueError, match="Invalid last session"):
        BotSessionState.resolve(conn, bot_id="fixture")


def test_unknown_bot_is_not_a_cold_start(conn):
    with pytest.raises(ValueError, match="Unknown bot"):
        BotSessionState.resolve(conn, bot_id="missing")
    with pytest.raises(ValueError, match="Cannot accept"):
        BotSessionState.accept(conn, bot_id="missing", session_id="old")


def test_same_binding_is_idempotent(conn):
    for _ in range(2):
        BotSessionState.accept(conn, bot_id="fixture", session_id="old")
    assert BotSessionState.resolve(conn, bot_id="fixture")["id"] == "old"


def test_accept_rolls_back_with_admission_transaction(conn):
    BotSessionState.accept(conn, bot_id="fixture", session_id="old")
    with pytest.raises(RuntimeError):
        with conn.begin_nested():
            BotSessionState.accept(conn, bot_id="fixture", session_id="latest")
            raise RuntimeError("admission failed")
    assert BotSessionState.resolve(conn, bot_id="fixture")["id"] == "old"


def test_profile_model_exposes_nullable_runtime_owned_pointer():
    from llm_bawt.runtime_settings import BotProfile

    column = BotProfile.__table__.c.last_session_id
    assert column.nullable
    assert column.type.length == 36
