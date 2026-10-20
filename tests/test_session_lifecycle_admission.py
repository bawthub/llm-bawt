"""A delayed binding cannot rewind a newer admitted bot session."""

import pytest
from sqlalchemy import create_engine, text

from llm_bawt.service.session_lifecycle.admission import accept_session_binding
from llm_bawt.service.turn_admission import AdmissionFenceLost


@pytest.fixture
def engine():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE bot_profiles (slug TEXT PRIMARY KEY, last_session_id TEXT)")
        conn.exec_driver_sql("CREATE TABLE sessions (id TEXT PRIMARY KEY, bot_id TEXT, status TEXT)")
        conn.exec_driver_sql("""CREATE TABLE turn_logs (
            id TEXT PRIMARY KEY, bot_id TEXT, user_id TEXT, created_at INTEGER,
            status TEXT, ended_at INTEGER)
        """)
        conn.exec_driver_sql("INSERT INTO bot_profiles (slug) VALUES ('fixture'), ('other')")
        conn.exec_driver_sql("""INSERT INTO sessions (id,bot_id,status) VALUES
            ('first','fixture','active'),('second','fixture','archived'),
            ('foreign','other','active')
        """)
        conn.exec_driver_sql("""INSERT INTO turn_logs (id,bot_id,user_id,created_at,status) VALUES
            ('old','fixture','nick',1,'streaming'),
            ('new','fixture','nick',2,'streaming'),
            ('queued','fixture','nick',3,'reserved')
        """)
    yield engine
    engine.dispose()


def pointer(engine):
    with engine.connect() as conn:
        return conn.execute(text(
            "SELECT last_session_id FROM bot_profiles WHERE slug='fixture'"
        )).scalar_one()


def test_newest_admission_stamps_even_if_later_reservation_exists(engine):
    accept_session_binding(engine, turn_id="new", bot_id="fixture", user_id="nick", session_id="second")
    assert pointer(engine) == "second"


def test_older_delayed_binding_cannot_rewind_or_start_inference(engine):
    accept_session_binding(engine, turn_id="new", bot_id="fixture", user_id="nick", session_id="second")
    with pytest.raises(AdmissionFenceLost, match="superseded"):
        accept_session_binding(engine, turn_id="old", bot_id="fixture", user_id="nick", session_id="first")
    assert pointer(engine) == "second"


@pytest.mark.parametrize("turn,bot,user,session", [
    ("queued", "fixture", "nick", "first"),
    ("new", "fixture", "wrong", "first"),
    ("new", "other", "nick", "first"),
    ("missing", "fixture", "nick", "first"),
])
def test_unaccepted_or_wrong_owner_never_stamps(engine, turn, bot, user, session):
    with pytest.raises(AdmissionFenceLost):
        accept_session_binding(engine, turn_id=turn, bot_id=bot, user_id=user, session_id=session)
    assert pointer(engine) is None


def test_lost_claim_and_cross_bot_session_fail_before_stamp(engine):
    with pytest.raises(AdmissionFenceLost, match="no longer held"):
        accept_session_binding(engine, turn_id="new", bot_id="fixture", user_id="nick",
                               session_id="second", claim=type("Claim", (), {"fence": lambda *_: False})())
    with pytest.raises(ValueError, match="Cannot accept"):
        accept_session_binding(engine, turn_id="new", bot_id="fixture", user_id="nick", session_id="foreign")
    assert pointer(engine) is None


def test_terminal_turn_cannot_accept_a_delayed_binding(engine):
    with engine.begin() as conn:
        conn.execute(text("UPDATE turn_logs SET ended_at=4, status='completed' WHERE id='new'"))
    with pytest.raises(AdmissionFenceLost):
        accept_session_binding(engine, turn_id="new", bot_id="fixture", user_id="nick", session_id="second")
    assert pointer(engine) is None
