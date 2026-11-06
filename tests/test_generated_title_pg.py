"""TASK-2064: generated titles against real SQL in a private PostgreSQL cluster.

Run: pytest -m integration tests/test_generated_title_pg.py
"""
import json
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, text

from llm_bawt.sessions import generated_title as gt
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
    _session(engine, "s1", {}, [("c", "user", "/new"), ("m0", "user", "###"),
                                ("m1", "user", "how do I rebase onto master?")])
    model = Model('"Git rebase question."')
    assert generate_session_title(engine, "s1", "bot-a", "nick", model) == "Git rebase question"
    assert _meta(engine, "s1") == {"title": "Git rebase question", "title_source": "generated"}
    assert "how do I rebase onto master?" in model.calls[0].prompt


def test_given_names_are_left_alone(engine):
    model = Model("Anything")
    _session(engine, "s3", {"title": "My name", "title_source": "user"}, [("m1", "user", "hello there")])
    _session(engine, "s4", {"title": "Daily brief", "title_source": "automation"}, [("m1", "user", "brief")])
    assert generate_session_title(engine, "s3", "bot-a", "nick", model) is None
    assert generate_session_title(engine, "s4", "bot-a", "nick", model) is None
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

    assert generate_session_title(engine, "s5", "bot-a", "nick", renaming_model) is None
    assert _meta(engine, "s5") == {"title": "Renamed", "title_source": "user"}


def test_auto_copy_is_replaced_and_refusal_stores_nothing(engine):
    _session(engine, "s6", {"title": "plan a tr…", "title_source": "auto"}, [("m1", "user", "plan a trip")])
    assert generate_session_title(engine, "s6", "bot-a", "nick", Model("Trip planning")) == "Trip planning"
    _session(engine, "s7", {}, [("m1", "user", "something")])
    assert generate_session_title(engine, "s7", "bot-a", "nick", Model("I'm sorry, I can't help")) is None
    assert _meta(engine, "s7") == {}


def test_wrong_user_is_not_touched(engine):
    _session(engine, "s8", {}, [("m1", "user", "hello")])
    assert generate_session_title(engine, "s8", "bot-a", "someone-else", Model("Hello")) is None
    assert _meta(engine, "s8") == {}


def test_marker_only_session_reports_no_eligible(engine):
    _session(engine, "s9", {}, [("m0", "user", "###")])
    model = Model("Unused")
    assert generate_session_title(engine, "s9", "bot-a", "nick", model) is gt.NO_ELIGIBLE
    assert model.calls == []


class Deferred:
    """Threads that start only when the test runs them."""
    queue = []

    def __init__(self, target, args=(), **_):
        self.target, self.args = target, args

    def start(self):
        Deferred.queue.append(self)

    @classmethod
    def drain(cls):
        while cls.queue:
            t = cls.queue.pop(0)
            t.target(*t.args)


@pytest.fixture()
def scheduler(monkeypatch):
    import llm_bawt.service.dependencies as deps
    Deferred.queue = []
    monkeypatch.setattr(gt.threading, "Thread", Deferred)
    monkeypatch.setattr(deps, "_service", object())
    monkeypatch.setattr(gt, "_attempted", set())
    monkeypatch.setattr(gt, "_pending", set())
    real = gt.generate_session_title
    monkeypatch.setattr(gt, "generate_session_title",
                        lambda *a, **kw: real(*a, **{**kw, "complete": Model("Git rebase question")}))
    return lambda sid, mid, content: gt.schedule_session_title(
        engine_holder[0], session_id=sid, bot_id="bot-a", user_id="nick",
        message_id=mid, content=content)


engine_holder = [None]


def _add(engine, sid, mid, content, ts):
    with engine.begin() as conn:
        conn.execute(text("INSERT INTO messages VALUES ('bota',:id,'user',:c,:ts,:s)"),
                     {"id": mid, "c": content, "ts": ts, "s": sid})


def test_eligible_write_before_marker_worker_reads(engine, scheduler):
    """Codex r2: ### worker deferred until after the real question lands."""
    engine_holder[0] = engine
    _session(engine, "s10", {}, [("q0", "user", "###")])
    scheduler("s10", "q0", "###")
    _add(engine, "s10", "q1", "How do I rebase?", 5)
    scheduler("s10", "q1", "How do I rebase?")
    Deferred.drain()
    assert _meta(engine, "s10")["title"] == "Git rebase question"


def test_eligible_write_after_marker_worker_reads(engine, scheduler, monkeypatch):
    """The question lands after the worker saw no eligible message but before it released."""
    engine_holder[0] = engine
    _session(engine, "s11", {}, [("q0", "user", "###")])
    scheduler("s11", "q0", "###")
    real = gt.generate_session_title
    state = {"first": True}

    def racing(*a, **kw):
        out = real(*a, **kw)
        if state.pop("first", False):
            _add(engine, "s11", "q1", "How do I rebase?", 5)
            scheduler("s11", "q1", "How do I rebase?")
        return out

    monkeypatch.setattr(gt, "generate_session_title", racing)
    Deferred.drain()
    assert _meta(engine, "s11")["title"] == "Git rebase question"


def test_sequential_marker_then_question(engine, scheduler):
    engine_holder[0] = engine
    _session(engine, "s12", {}, [("q0", "user", "###")])
    scheduler("s12", "q0", "###")
    Deferred.drain()
    assert gt._attempted == set()
    _add(engine, "s12", "q1", "How do I rebase?", 5)
    scheduler("s12", "q1", "How do I rebase?")
    Deferred.drain()
    assert _meta(engine, "s12")["title"] == "Git rebase question"
