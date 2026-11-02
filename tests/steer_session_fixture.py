"""Small real owned-session binding for steer route unit fixtures."""
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool


@pytest.fixture
def steer_store_factory():
    with ExitStack() as stack:
        yield lambda turn, session_id="session-target": stack.enter_context(owned_steer_store(turn, session_id))


@contextmanager
def owned_steer_store(turn, session_id="session-target"):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    try:
        with engine.begin() as conn:
            conn.exec_driver_sql("CREATE TABLE sessions (id TEXT,bot_id TEXT,user_id TEXT,status TEXT)")
            conn.exec_driver_sql("CREATE TABLE messages (id TEXT,bot_id TEXT,session_id TEXT)")
            conn.exec_driver_sql("CREATE TABLE turn_logs (id TEXT,bot_id TEXT,user_id TEXT,trigger_message_id TEXT)")
            values = dict(id=turn.id, bot=turn.bot_id, user=turn.user_id,
                          trigger=turn.trigger_message_id, session=session_id)
            conn.execute(text("INSERT INTO sessions VALUES (:session,:bot,:user,'active')"), values)
            conn.execute(text("INSERT INTO messages VALUES (:trigger,:bot,:session)"), values)
            conn.execute(text("INSERT INTO turn_logs VALUES (:id,:bot,:user,:trigger)"), values)
        yield SimpleNamespace(engine=engine, get_turn=lambda _id: turn)
    finally:
        engine.dispose()
