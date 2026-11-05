"""Accepted transitions, not navigation/message-presence guesses (TASK-1918)."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from sqlalchemy import create_engine, event, select
from sqlalchemy.pool import StaticPool

from llm_bawt.service import session_resumption as receipts
from llm_bawt.service.session_lifecycle.admission import accept_session_binding
from llm_bawt.service.turn_admission import AdmissionFenceLost


@pytest.fixture
def db():
    engine = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    receipts.ensure_schema(engine)
    receipts.ensure_schema(engine)
    with engine.begin() as conn:
        for ddl in (
            "CREATE TABLE bot_profiles (slug TEXT PRIMARY KEY, last_session_id TEXT)",
            "CREATE TABLE sessions (id TEXT PRIMARY KEY, bot_id TEXT, user_id TEXT, status TEXT, started_at TEXT, session_metadata TEXT)",
            "CREATE TABLE messages (id TEXT PRIMARY KEY, bot_id TEXT, session_id TEXT, role TEXT)",
            "CREATE TABLE turn_logs (id TEXT PRIMARY KEY, bot_id TEXT, user_id TEXT, created_at INTEGER, ended_at INTEGER, status TEXT, trigger_message_id TEXT)",
        ):
            conn.exec_driver_sql(ddl)
        conn.exec_driver_sql("INSERT INTO bot_profiles VALUES ('fixture','current'),('other','foreign')")
        conn.exec_driver_sql("""INSERT INTO sessions VALUES
            ('current','fixture','owner','active','2026-11-01 10:00:00','{}'),
            ('old','fixture','owner','archived','2026-10-01 10:00:00','{"title":"  Earlier work  "}'),
            ('empty','fixture','owner','archived',NULL,'{}'),
            ('foreign','other','owner','active','2026-01-01 10:00:00','{"title":"private"}')""")
        conn.exec_driver_sql("INSERT INTO messages VALUES ('prior','fixture','old','assistant')")
        conn.exec_driver_sql("INSERT INTO turn_logs VALUES ('turn','fixture','owner',1,NULL,'streaming','prompt')")
    yield engine
    engine.dispose()


def accept(db, **overrides):
    args = dict(turn_id="turn", bot_id="fixture", user_id="owner", session_id="old", record_resumption=True)
    return accept_session_binding(db, **(args | overrides))


def stored(db):
    with db.connect() as conn:
        return conn.execute(select(receipts.RECEIPTS)).mappings().all()


def pointer(db):
    with db.connect() as conn:
        return conn.exec_driver_sql("SELECT last_session_id FROM bot_profiles WHERE slug='fixture'").scalar_one()


def test_atomic_accept_before_user_insert_and_reload_without_turn_logs(db):
    result = accept(db)
    assert pointer(db) == "old"
    assert result == dict(turn_id="turn", session_id="old", previous_session_id="current",
                          accepted_at=result["accepted_at"], session_started_at=1790848800.0,
                          title="Earlier work")
    assert len(stored(db)) == 1
    # Streaming accepts before the canonical user row. Do not move its timing.
    assert receipts.for_messages(db, bot_id="fixture", messages=[{"id": "prompt", "role": "user"}]) == {}
    with db.begin() as conn:
        conn.exec_driver_sql("INSERT INTO messages VALUES ('prompt','fixture','old','user')")
        conn.exec_driver_sql("DROP TABLE turn_logs")
    assert receipts.for_messages(db, bot_id="fixture", messages=[{"id": "prompt", "role": "user"}]) == {"prompt": result}
    wire = receipts.event(result, bot_id="fixture", user_id="owner", trigger_message_id="prompt")
    assert wire["session_resumption"] == result and wire["session_id"] == "old"
    assert wire["_type"] == "session_resumed" and wire["ts"] == result["accepted_at"]


def test_accept_replay_is_idempotent_and_does_not_retimestamp(db):
    first = accept(db)
    assert accept(db) == first
    assert len(stored(db)) == 1


def test_same_prompt_cannot_rebind_receipt_identity(db):
    accept(db)
    with pytest.raises(ValueError, match="identity conflict"):
        accept(db, session_id="current")
    assert pointer(db) == "old" and len(stored(db)) == 1


@pytest.mark.parametrize("kind", ["navigation", "null", "unchanged", "empty", "only_current_prompt", "wrong_owner", "reset"])
def test_no_false_resumption_markers(db, kind):
    with db.begin() as conn:
        if kind == "null":
            conn.exec_driver_sql("UPDATE bot_profiles SET last_session_id=NULL WHERE slug='fixture'")
        elif kind == "unchanged":
            conn.exec_driver_sql("UPDATE bot_profiles SET last_session_id='old' WHERE slug='fixture'")
        elif kind == "wrong_owner":
            conn.exec_driver_sql("UPDATE sessions SET user_id='private' WHERE id='old'")
        elif kind == "only_current_prompt":
            conn.exec_driver_sql("DELETE FROM messages")
            conn.exec_driver_sql("INSERT INTO messages VALUES ('prompt','fixture','old','user')")
    if kind != "navigation":
        assert accept(db, session_id="empty" if kind == "empty" else "old",
                      record_resumption=kind != "reset") is None
    assert stored(db) == []


@pytest.mark.parametrize("kind", ["reserved", "terminal", "superseded", "lost_claim", "foreign"])
def test_rejected_or_preaccept_failure_never_creates_receipt(db, kind):
    with db.begin() as conn:
        if kind == "reserved":
            conn.exec_driver_sql("UPDATE turn_logs SET status='reserved'")
        elif kind == "terminal":
            conn.exec_driver_sql("UPDATE turn_logs SET status='error',ended_at=2")
        elif kind == "superseded":
            conn.exec_driver_sql("INSERT INTO turn_logs VALUES ('new','fixture','owner',2,NULL,'streaming','new-prompt')")
    with pytest.raises((AdmissionFenceLost, ValueError)):
        accept(db, **({"claim": SimpleNamespace(fence=lambda _c: False)} if kind == "lost_claim" else
                      {"session_id": "foreign"} if kind == "foreign" else {}))
    assert pointer(db) == "current" and stored(db) == []


def test_receipt_insert_failure_rolls_back_pointer_and_accept_failure_rolls_back_receipt(db, monkeypatch):
    from llm_bawt.bot_session_state import BotSessionState

    monkeypatch.setattr(BotSessionState, "accept", Mock(side_effect=ValueError("stamp failed")))
    with pytest.raises(ValueError, match="stamp failed"):
        accept(db)
    assert pointer(db) == "current" and stored(db) == []


def test_switch_away_and_back_is_a_distinct_prompt_receipt(db):
    first = accept(db)
    with db.begin() as conn:
        conn.exec_driver_sql("UPDATE bot_profiles SET last_session_id='current' WHERE slug='fixture'")
        conn.exec_driver_sql("UPDATE turn_logs SET ended_at=2,status='completed'")
        conn.exec_driver_sql("INSERT INTO turn_logs VALUES ('turn2','fixture','owner',2,NULL,'streaming','prompt2')")
    second = accept(db, turn_id="turn2")
    assert first["turn_id"] == "turn" and second["turn_id"] == "turn2"
    assert len(stored(db)) == 2


def test_default_B_explicit_A_default_B_live_receipts_survive_reload(db):
    # B is already the status-active/default; accepting A never changes that
    # status. Returning to B is an accepted transition even with no explicit ID.
    with db.begin() as conn:
        conn.exec_driver_sql("INSERT INTO messages VALUES ('B-prior','fixture','current','assistant')")
    away = accept(db)
    assert away["previous_session_id"] == "current" and away["session_id"] == "old"
    with db.begin() as conn:
        conn.exec_driver_sql("INSERT INTO messages VALUES ('prompt','fixture','old','user')")
        conn.exec_driver_sql("UPDATE turn_logs SET ended_at=2,status='completed'")
        conn.exec_driver_sql("INSERT INTO turn_logs VALUES ('return','fixture','owner',2,NULL,'streaming','return-prompt')")
    back = accept(db, turn_id="return", session_id="current")
    assert back["previous_session_id"] == "old" and back["session_id"] == "current"
    wire = receipts.event(back, bot_id="fixture", user_id="owner", trigger_message_id="return-prompt")
    assert wire["session_resumption"] == back and wire["session_id"] == "current"
    assert accept(db, turn_id="return", session_id="current") == back  # stable retry
    with db.begin() as conn:
        conn.exec_driver_sql("INSERT INTO messages VALUES ('return-prompt','fixture','current','user')")
        conn.exec_driver_sql("UPDATE turn_logs SET ended_at=3,status='completed' WHERE id='return'")
        conn.exec_driver_sql("INSERT INTO turn_logs VALUES ('again','fixture','owner',3,NULL,'streaming','again-prompt')")
    assert accept(db, turn_id="again", session_id="current") is None
    assert len(stored(db)) == 2
    with db.begin() as conn:
        conn.exec_driver_sql("DROP TABLE turn_logs")
        assert conn.exec_driver_sql("SELECT status FROM sessions WHERE id='current'").scalar_one() == "active"
        assert conn.exec_driver_sql("SELECT status FROM sessions WHERE id='old'").scalar_one() == "archived"
    assert receipts.for_messages(db, bot_id="fixture", user_id="owner", messages=[
        {"id": "prompt", "role": "user"}, {"id": "return-prompt", "role": "user"},
    ]) == {"prompt": away, "return-prompt": back}


def test_history_is_exact_bot_message_session_and_owner_scoped(db):
    accept(db)
    with db.begin() as conn:
        conn.exec_driver_sql("INSERT INTO messages VALUES ('prompt','fixture','old','user')")
    page = [{"id": "prompt", "role": "user"}, {"id": "prior", "role": "assistant"}]
    assert set(receipts.for_messages(db, bot_id="fixture", messages=page, user_id="owner")) == {"prompt"}
    assert receipts.for_messages(db, bot_id="other", messages=page) == {}
    assert receipts.for_messages(db, bot_id="fixture", messages=page, user_id="private") == {}
    with db.begin() as conn:
        conn.exec_driver_sql("UPDATE messages SET session_id='current' WHERE id='prompt'")
    assert receipts.for_messages(db, bot_id="fixture", messages=page) == {}


def test_inference_failure_does_not_erase_accepted_transition(db):
    receipt = accept(db)
    with db.begin() as conn:
        conn.exec_driver_sql("UPDATE turn_logs SET status='error',ended_at=2")
    assert stored(db)[0]["accepted_at"] == receipt["accepted_at"]
    assert pointer(db) == "old"


def test_history_enrichment_does_not_bootstrap_missing_table(db, monkeypatch):
    from llm_bawt.service.routes.history_resumption import hydrate_resumptions_for_page

    with db.begin() as conn:
        conn.exec_driver_sql("DROP TABLE session_resumption_receipts")
    monkeypatch.setattr("llm_bawt.media.assets._build_engine", lambda _config: db)
    statements = []
    event.listen(db, "before_cursor_execute", lambda _c, _cur, stmt, *_: statements.append(stmt))
    assert hydrate_resumptions_for_page(SimpleNamespace(config=None), "fixture", [{"id": "prompt", "role": "user"}]) == {}
    assert statements and all(s.lstrip().upper().startswith("SELECT") for s in statements)


def test_shared_history_builder_keeps_receipt_on_exact_user_anchor(db, monkeypatch):
    from llm_bawt.service.routes import history_pages as pages

    receipt = accept(db)
    with db.begin() as conn:
        conn.exec_driver_sql("INSERT INTO messages VALUES ('prompt','fixture','old','user')")
    monkeypatch.setattr("llm_bawt.media.assets._build_engine", lambda _config: db)
    for name in ("_hydrate_attachments_for_page", "_hydrate_reasoning_for_page", "_hydrate_turn_meta_for_page",
                 "_hydrate_interrupt_anchors_for_page", "hydrate_scheduler_for_page", "hydrate_approver_notices_for_page"):
        monkeypatch.setattr(pages, name, lambda *_a: {})
    monkeypatch.setattr(pages, "_message_author_payload", lambda _m: {"entity_type": "user", "entity_id": "owner"})
    messages = [dict(id="prompt", role="user", content="continue", timestamp=10),
                dict(id="prior", role="assistant", content="old", timestamp=9)]
    result = pages._build_history_response(SimpleNamespace(config=None), "fixture", messages, messages, 2,
                                          has_older=True, has_newer=True, anchor_id="prompt", origin_user_id="owner")
    assert result.messages[0].session_resumption.model_dump() == receipt
    assert result.messages[1].session_resumption is None
    assert result.has_older and result.has_newer and result.anchor_id == "prompt"


def _set_old_metadata(db, metadata: str):
    with db.begin() as conn:
        conn.exec_driver_sql("UPDATE sessions SET session_metadata=? WHERE id='old'", (metadata,))


def test_receipt_keeps_a_long_given_name_whole(db):
    """TASK-2061: a rename longer than the old 300 cut is stored uncut."""
    name = "Renamed " + "x" * 400
    _set_old_metadata(db, '{"title": "%s", "title_source": "user"}' % name)
    assert accept(db)["title"] == name
    assert stored(db)[0]["title"] == name


def test_receipt_never_copies_an_auto_title(db):
    """TASK-2061: the stored auto copy of a first message is not a name."""
    _set_old_metadata(db, '{"title": "first words of the chat…", "title_source": "auto"}')
    assert accept(db)["title"] is None


def test_schema_bootstrap_widens_a_legacy_varchar_title(monkeypatch):
    """The ALTER runs only on PostgreSQL; elsewhere bootstrap stays create_all."""
    calls = []

    class Conn:
        class dialect:
            name = "postgresql"

        def execute(self, statement, *_args):
            sql = str(statement)
            calls.append(sql)
            result = Mock()
            result.scalar_one_or_none.return_value = "character varying"
            return result

    monkeypatch.setattr(receipts._metadata, "create_all", lambda conn: None)
    receipts._bootstrap(Conn())
    assert any("ALTER COLUMN title TYPE text" in sql for sql in calls)
