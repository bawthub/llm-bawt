"""The shared rotation primitive keeps lineage and pointer atomic."""
import json

import pytest
from sqlalchemy import create_engine, text

from llm_bawt.memory.session_reset_lineage import SessionResetLineage


@pytest.fixture
def engine():
    engine = create_engine("sqlite://")
    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE bot_profiles (slug TEXT PRIMARY KEY, last_session_id TEXT)")
        conn.exec_driver_sql("""CREATE TABLE sessions (
            id TEXT PRIMARY KEY, bot_id TEXT, user_id TEXT, status TEXT,
            started_at TIMESTAMP, ended_at TIMESTAMP, archived_at TIMESTAMP,
            session_metadata TEXT)
        """)
        conn.exec_driver_sql("INSERT INTO bot_profiles (slug) VALUES ('fixture')")
        conn.exec_driver_sql("""INSERT INTO sessions (id,bot_id,user_id,status,session_metadata)
            VALUES ('old','fixture','nick','active','{}'),
                   ('later','fixture','other','archived','{}')""")
    yield engine
    engine.dispose()


def test_first_rotation_stamps_pointer_and_lineage_in_caller_transaction(engine):
    lifecycle = {"action": "reset_retain_history", "old_session_id": "old", "delivery_id": "delivery-one"}
    with engine.begin() as conn:
        assert SessionResetLineage.rotate(
            engine, bot_id="fixture", user_id="nick", new_id="successor",
            idempotency_key="delivery-one", connection=conn,
            context_lifecycle=lifecycle, accept_pointer=True,
        )
        assert conn.execute(text("SELECT last_session_id FROM bot_profiles WHERE slug='fixture'")).scalar_one() == "successor"
    with engine.connect() as conn:
        previous = conn.execute(text("SELECT session_metadata FROM sessions WHERE id='old'")).scalar_one()
        current = conn.execute(text("SELECT session_metadata FROM sessions WHERE id='successor'")).scalar_one()
    assert json.loads(previous)["reset_successor_id"] == "successor"
    assert json.loads(current)["reset_predecessor_ids"] == ["old"]
    assert json.loads(current)["context_lifecycle"] == lifecycle


def test_keyed_replay_does_not_rewind_later_accepted_session(engine):
    opts = dict(bot_id="fixture", user_id="nick", new_id="successor",
                idempotency_key="delivery-one", accept_pointer=True)
    assert SessionResetLineage.rotate(engine, **opts)
    with engine.begin() as conn:
        conn.execute(text("UPDATE bot_profiles SET last_session_id='later' WHERE slug='fixture'"))
    assert SessionResetLineage.rotate(engine, **opts) is False
    with engine.connect() as conn:
        assert conn.execute(text("SELECT last_session_id FROM bot_profiles WHERE slug='fixture'")).scalar_one() == "later"


def test_typed_new_manager_stamps_pointer_and_keyed_replay_does_not_rewind(engine):
    from types import SimpleNamespace
    from llm_bawt.memory.postgresql_short_term import PostgreSQLShortTermManager

    manager = object.__new__(PostgreSQLShortTermManager)
    manager._backend = SimpleNamespace(engine=engine)
    manager.bot_id, manager.user_id = "fixture", "nick"
    successor = manager.rotate_session(idempotency_key="typed-new", accept_pointer=True)
    with engine.begin() as conn:
        assert conn.execute(text(
            "SELECT last_session_id FROM bot_profiles WHERE slug='fixture'"
        )).scalar_one() == successor
        conn.execute(text("UPDATE bot_profiles SET last_session_id='later' WHERE slug='fixture'"))
    assert manager.rotate_session(idempotency_key="typed-new", accept_pointer=True) == successor
    with engine.connect() as conn:
        assert conn.execute(text(
            "SELECT last_session_id FROM bot_profiles WHERE slug='fixture'"
        )).scalar_one() == "later"


def test_pointer_failure_rolls_back_session_archive(engine):
    with pytest.raises(ValueError, match="Cannot accept"):
        SessionResetLineage.rotate(engine, bot_id="unknown", user_id="nick",
                                   new_id="successor", accept_pointer=True)
    with engine.connect() as conn:
        assert conn.execute(text("SELECT status FROM sessions WHERE id='old'")).scalar_one() == "active"
        assert conn.execute(text("SELECT 1 FROM sessions WHERE id='successor'")).first() is None
