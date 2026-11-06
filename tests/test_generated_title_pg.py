"""TASK-2064: generated titles against real SQL in a private PostgreSQL cluster.

Run: pytest -m integration tests/test_generated_title_pg.py
"""
import json
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, text

from llm_bawt.sessions.generated_title import generate_session_title
from tests.private_postgres import private_url

pytestmark = pytest.mark.integration


@pytest.fixture()
def engine(private_pg, private_module_db):
    engine = create_engine(private_url(private_pg, private_module_db))
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS sessions, messages"))
        conn.execute(text("CREATE TABLE sessions (id text, bot_id text, user_id text, status text, session_metadata jsonb)"))
        conn.execute(text("CREATE TABLE messages (bot_id text, id text, role text, content text, timestamp float8, session_id text)"))
    try:
        yield engine
    finally:
        engine.dispose()


def _session(engine, sid, meta, messages):
    with engine.begin() as conn:
        conn.execute(text("INSERT INTO sessions VALUES (:id,'bot-a','nick','active',CAST(:m AS jsonb))"),
                     {"id": sid, "m": json.dumps(meta)})
        for i, (mid, role, content) in enumerate(messages):
            conn.execute(text("INSERT INTO messages VALUES ('bota',:id,:role,:c,:ts,:s)"),
                         {"id": mid, "role": role, "c": content, "ts": float(i), "s": sid})


def _meta(engine, sid):
    with engine.connect() as conn:
        return conn.execute(text("SELECT session_metadata FROM sessions WHERE id=:id"), {"id": sid}).scalar_one()


class Model:
    def __init__(self, reply):
        self.reply, self.calls = reply, []

    def __call__(self, request):
        self.calls.append(request)
        return SimpleNamespace(content=self.reply, model="test")


def test_first_message_gets_generated_title(engine):
    _session(engine, "s1", {}, [("c", "user", "/new"), ("m1", "user", "how do I rebase onto master?")])
    model = Model('"Git rebase question."')
    assert generate_session_title(engine, "s1", "bot-a", "nick", "m1", "how do I rebase onto master?", model) == "Git rebase question"
    assert _meta(engine, "s1") == {"title": "Git rebase question", "title_source": "generated"}
    assert "how do I rebase onto master?" in model.calls[0].prompt


def test_later_message_and_given_names_are_left_alone(engine):
    _session(engine, "s2", {}, [("m1", "user", "first"), ("m2", "user", "second")])
    model = Model("Anything")
    assert generate_session_title(engine, "s2", "bot-a", "nick", "m2", "second", model) is None
    _session(engine, "s3", {"title": "My name", "title_source": "user"}, [("m1", "user", "hello there")])
    _session(engine, "s4", {"title": "Daily brief", "title_source": "automation"}, [("m1", "user", "brief")])
    assert generate_session_title(engine, "s3", "bot-a", "nick", "m1", "hello there", model) is None
    assert generate_session_title(engine, "s4", "bot-a", "nick", "m1", "brief", model) is None
    assert model.calls == []
    assert _meta(engine, "s3")["title"] == "My name"
    assert _meta(engine, "s4")["title"] == "Daily brief"


def test_rename_during_generation_wins(engine):
    _session(engine, "s5", {}, [("m1", "user", "plan a trip")])

    def renaming_model(request):
        with engine.begin() as conn:
            conn.execute(text("""UPDATE sessions SET session_metadata = session_metadata
                || '{"title": "Renamed", "title_source": "user"}'::jsonb WHERE id='s5'"""))
        return SimpleNamespace(content="Trip planning", model="test")

    assert generate_session_title(engine, "s5", "bot-a", "nick", "m1", "plan a trip", renaming_model) is None
    assert _meta(engine, "s5") == {"title": "Renamed", "title_source": "user"}


def test_auto_copy_is_replaced_and_refusal_stores_nothing(engine):
    _session(engine, "s6", {"title": "plan a tr…", "title_source": "auto"}, [("m1", "user", "plan a trip")])
    assert generate_session_title(engine, "s6", "bot-a", "nick", "m1", "plan a trip", Model("Trip planning")) == "Trip planning"
    _session(engine, "s7", {}, [("m1", "user", "something")])
    assert generate_session_title(engine, "s7", "bot-a", "nick", "m1", "something", Model("I'm sorry, I can't help")) is None
    assert _meta(engine, "s7") == {}


def test_wrong_user_is_not_touched(engine):
    _session(engine, "s8", {}, [("m1", "user", "hello")])
    assert generate_session_title(engine, "s8", "bot-a", "someone-else", "m1", "hello", Model("Hello")) is None
    assert _meta(engine, "s8") == {}


def test_marker_only_first_message_releases_the_attempt(engine):
    """Codex r1: ``###`` first must not consume the session's one attempt."""
    import llm_bawt.service.dependencies as deps
    from llm_bawt.sessions import generated_title as gt

    _session(engine, "s9", {}, [("m0", "user", "###")])
    released = []
    model = Model("Unused")
    assert generate_session_title(engine, "s9", "bot-a", "nick", "m0", "###", model,
                                  on_no_eligible=lambda: released.append(True)) is None
    assert released == [True] and model.calls == []

    # Real scheduler, threads run inline: ### then the first real question.
    class Inline:
        def __init__(self, target, **_):
            self.target = target

        def start(self):
            self.target()

    calls = []
    orig = gt.generate_session_title

    def spy(*a, **kw):
        calls.append(a[4])
        return orig(*a, **{**kw, "complete": Model("Git rebase question")})

    mp = pytest.MonkeyPatch()
    try:
        mp.setattr(gt.threading, "Thread", Inline)
        mp.setattr(gt, "generate_session_title", spy)
        mp.setattr(deps, "_service", object())
        mp.setattr(gt, "_attempted", set())
        _session(engine, "s10", {}, [("q0", "user", "###")])
        gt.schedule_session_title(engine, session_id="s10", bot_id="bot-a", user_id="nick",
                                  message_id="q0", content="###")
        with engine.begin() as conn:
            conn.execute(text("INSERT INTO messages VALUES ('bota','q1','user','How do I rebase?',5,'s10')"))
        gt.schedule_session_title(engine, session_id="s10", bot_id="bot-a", user_id="nick",
                                  message_id="q1", content="How do I rebase?")
    finally:
        mp.undo()
    assert calls == ["q0", "q1"]
    assert _meta(engine, "s10")["title"] == "Git rebase question"
