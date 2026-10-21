"""Pointer bootstrap/constraints on a test-owned, network-isolated PostgreSQL."""
from __future__ import annotations

import uuid

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError

from llm_bawt.bot_session_state import BotSessionState
from tests.private_postgres import private_url

pytestmark = pytest.mark.integration


@pytest.fixture
def conn(private_pg, private_module_db):
    engine = create_engine(private_url(private_pg, private_module_db))
    with engine.connect() as connection:
        transaction = connection.begin()
        schema = "pointer_" + uuid.uuid4().hex
        connection.exec_driver_sql(f'CREATE SCHEMA "{schema}"')
        connection.exec_driver_sql(f'SET LOCAL search_path TO "{schema}"')
        yield connection
        transaction.rollback()
    engine.dispose()


def _bots(conn):
    conn.exec_driver_sql("CREATE TABLE bot_profiles (slug VARCHAR(128) PRIMARY KEY)")
    conn.exec_driver_sql("INSERT INTO bot_profiles VALUES ('fixture')")


def _sessions(conn):
    conn.exec_driver_sql("""CREATE TABLE sessions (
        id VARCHAR(36) PRIMARY KEY, bot_id VARCHAR(64), user_id VARCHAR(64),
        started_at TIMESTAMP, ended_at TIMESTAMP, archived_at TIMESTAMP,
        status VARCHAR(16), session_metadata JSONB)
    """)
    conn.exec_driver_sql("""INSERT INTO sessions (id, bot_id, user_id, status)
        VALUES ('first', 'fixture', 'nick', 'archived'),
               ('second', 'fixture', 'other-user', 'active')
    """)


@pytest.mark.parametrize("order", ["bots_first", "sessions_first"])
def test_bootstrap_both_orders_is_repeatable_without_backfill(conn, order):
    if order == "bots_first":
        _bots(conn)
        BotSessionState.ensure_schema(conn)
        _sessions(conn)
    else:
        _sessions(conn)
        BotSessionState.ensure_schema(conn)
        _bots(conn)
    for _ in range(2):
        BotSessionState.ensure_schema(conn)
    assert BotSessionState.resolve(conn, bot_id="fixture") is None
    constraints = conn.execute(text("""
        SELECT pg_get_constraintdef(oid) FROM pg_constraint
        WHERE conrelid='bot_profiles'::regclass AND conname='fk_bot_profiles_last_session'
    """)).scalars().all()
    assert len(constraints) == 1
    assert "FOREIGN KEY (last_session_id) REFERENCES sessions(id) ON DELETE SET NULL" in constraints[0]
    with pytest.raises(IntegrityError), conn.begin_nested():
        conn.exec_driver_sql("UPDATE bot_profiles SET last_session_id='missing'")
    BotSessionState.accept(conn, bot_id="fixture", session_id="first")
    assert BotSessionState.resolve(conn, bot_id="fixture")["id"] == "first"
    conn.exec_driver_sql("DELETE FROM sessions WHERE id='first'")
    assert BotSessionState.resolve(conn, bot_id="fixture") is None


def test_pointer_rolls_back_with_failed_transition(conn):
    _bots(conn)
    _sessions(conn)
    BotSessionState.ensure_schema(conn)
    BotSessionState.accept(conn, bot_id="fixture", session_id="first")
    with pytest.raises(RuntimeError), conn.begin_nested():
        BotSessionState.accept(conn, bot_id="fixture", session_id="second")
        raise RuntimeError("rotation rolled back")
    assert BotSessionState.resolve(conn, bot_id="fixture")["id"] == "first"
