"""Private-PG proof of versioned user guidance (TASK-893 slice 3)."""
from __future__ import annotations

import json
import threading
import uuid

import pytest
from sqlalchemy import create_engine, text

from llm_bawt import bot_slug_aliases
from llm_bawt.inter_bot_delivery import InterBotDeliveryStore
from llm_bawt.memory import guidance as g
from llm_bawt.memory import history_read_cursor as hrc
from llm_bawt.memory.postgresql import ensure_bot_partitions
from llm_bawt.prompt_registry import PromptTemplate, PromptTemplateVersion
from llm_bawt.runtime_settings import BotProfile
from llm_bawt.utils import db
from tests.pg_isolation import require_test_db_config

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]


@pytest.fixture(scope="module")
def engine(private_app_pg):
    engine = create_engine(db.build_postgres_url(require_test_db_config()))
    InterBotDeliveryStore(require_test_db_config())
    with engine.begin() as conn:
        BotProfile.__table__.create(conn, checkfirst=True)
        PromptTemplate.__table__.create(conn, checkfirst=True)
        PromptTemplateVersion.__table__.create(conn, checkfirst=True)
        hrc.ensure_schema(conn)
        g.ensure_schema(conn)
        g.ensure_schema(conn)  # idempotent
        bot_slug_aliases.ensure_schema(conn)  # the real alias schema, present
    try:
        yield engine
    finally:
        engine.dispose()


class Rig:
    def __init__(self, engine):
        self.engine = engine
        tag = uuid.uuid4().hex[:10]
        self.user, self.other = f"gu{tag}", f"go{tag}"
        self.bot, self.prot = f"gb{tag}", f"gp{tag}"
        self.excluded = [self.prot]
        self.sessions = {}

    def session(self, user):
        if user not in self.sessions:
            sid = str(uuid.uuid4())
            with self.engine.begin() as conn:
                ensure_bot_partitions(conn, self.bot)
                conn.execute(text("INSERT INTO sessions (id, bot_id, user_id, status) "
                                  "VALUES (:id, :b, :u, 'active')"),
                             {"id": sid, "b": self.bot, "u": user})
            self.sessions[user] = sid
        return self.sessions[user]

    def msg(self, content="I prefer short answers", *, user=None, session=None,
            author=("user", None), ts=1.0, producer=None):
        user = user or self.user
        mid = str(uuid.uuid4())
        sid = session if session is not None else self.session(user)
        with self.engine.begin() as conn:
            ensure_bot_partitions(conn, self.bot)
            conn.execute(text("""
                INSERT INTO messages (bot_id, id, role, content, timestamp, session_id,
                                      author_entity_type, author_entity_id)
                VALUES (:b, :id, 'user', :c, :ts, :sid, :at, :aid)"""),
                {"b": self.bot, "id": mid, "c": content, "ts": ts, "sid": sid,
                 "at": author[0], "aid": author[1] or user})
            if producer:
                conn.execute(text("""
                    INSERT INTO inter_bot_deliveries (id, sender_bot_id, author_entity_type,
                        author_entity_id, target_bot_id, message, user_message_id, turn_id,
                        status, metadata_json)
                    VALUES (:id, :b, :at, :aid, :b, :c, :m, :turn, 'DELIVERED', :meta)"""),
                    {"id": f"d-{mid}", "m": mid, "b": self.bot, "c": content,
                     "at": author[0] or "user", "aid": author[1] or user,
                     "turn": f"turn-{uuid.uuid4().hex}",
                     "meta": json.dumps({"_request_identity_producer": producer})})
        return {"bot_id": self.bot, "message_id": mid}

    def doc(self, *prefs, mandate=""):
        return {"v": 1, "mandate": mandate, "preferences": list(prefs)}

    def pref(self, pid, ev, statement="short answers", **kw):
        return {"id": pid, "statement": statement, "status": "active", "evidence": ev, **kw}

    def propose(self, doc, user=None, by="al"):
        return g.propose(self.engine, user_id=user or self.user, new_doc=doc, reason="seen",
                         proposed_by=by, excluded_bot_ids=self.excluded)

    def approve(self, pid, by=None):
        return g.decide(self.engine, proposal_id=pid, decided_by=by or self.user, approve=True)

    def status(self, pid):
        with self.engine.connect() as c:
            return c.execute(text("SELECT status FROM guidance_proposals WHERE id=:i"),
                             {"i": pid}).scalar_one()


@pytest.fixture
def rig(engine):
    return Rig(engine)


def test_human_evidence_accepted_and_versioned(rig):
    ev = rig.msg()
    out = rig.approve(rig.propose(rig.doc(rig.pref("p1", [ev]))))
    assert out["status"] == "approved" and out["version"] == 1
    with rig.engine.connect() as c:
        cur = g.current_version(c, rig.user)
    assert json.loads(cur["body"])["preferences"][0]["evidence"] == [ev]


@pytest.mark.parametrize("kind", ["schedule", "bot", "null_author", "command", "foreign",
                                  "unmapped", "protected"])
def test_non_owner_or_non_human_evidence_refused(rig, kind):
    if kind == "schedule":
        ev = rig.msg(producer="schedule")
    elif kind == "bot":
        ev = rig.msg(author=("bot", "peer"), producer="bot")
    elif kind == "null_author":
        ev = rig.msg(author=(None, None))
        with rig.engine.begin() as c:
            c.execute(text("UPDATE messages SET author_entity_id=NULL WHERE id=:i"),
                      {"i": ev["message_id"]})
    elif kind == "command":
        ev = rig.msg("/new", producer="ordinary")
    elif kind == "foreign":
        ev = rig.msg(user=rig.other)
    elif kind == "unmapped":
        sid = str(uuid.uuid4())  # a real session, but of another bot
        with rig.engine.begin() as c:
            c.execute(text("INSERT INTO sessions (id, bot_id, user_id, status) "
                           "VALUES (:id, 'elsewhere', :u, 'active')"), {"id": sid, "u": rig.user})
        ev = rig.msg(session=sid)
    else:
        ev = {"bot_id": rig.prot, "message_id": "x"}
    with pytest.raises(ValueError):
        rig.propose(rig.doc(rig.pref("p1", [ev])))


def test_mandate_and_extra_keys_refused(rig):
    ev = rig.msg()
    with pytest.raises(ValueError):
        rig.propose(rig.doc(rig.pref("p1", [ev]), mandate="do anything"))
    bad = rig.doc(rig.pref("p1", [ev]))
    bad["authority"] = "all"
    with pytest.raises(ValueError):
        rig.propose(bad)
    with pytest.raises(ValueError):
        rig.propose(rig.doc(rig.pref("p1", [ev], scope="all")))


def test_deciders_must_be_the_owner_and_not_a_bot(rig):
    pid = rig.propose(rig.doc(rig.pref("p1", [rig.msg()])))
    with pytest.raises(PermissionError):
        rig.approve(pid, by=rig.other)
    from sqlmodel import Session
    with Session(rig.engine) as s:  # a bot profile named like the user
        s.add(BotProfile(slug=rig.user, name=rig.user, description="", system_prompt=""))
        s.commit()
    try:
        with pytest.raises(PermissionError):
            rig.approve(pid)
    finally:
        with rig.engine.begin() as c:
            c.execute(text("DELETE FROM bot_profiles WHERE slug=:s"), {"s": rig.user})
    assert rig.status(pid) == "pending"
    # A retired bot slug (alias row) named like a user cannot decide either.
    alias_user = f"ret{rig.user}"
    with Session(rig.engine) as s:
        s.add(BotProfile(slug=f"live{rig.user}", name="x", description="", system_prompt=""))
        s.commit()
    with rig.engine.begin() as c:
        c.execute(text("INSERT INTO bot_slug_aliases (old_slug, bot_slug) VALUES (:o, :b)"),
                  {"o": alias_user, "b": f"live{rig.user}"})
    apid = rig.propose(rig.doc(rig.pref("p1", [rig.msg(user=alias_user)])), user=alias_user)
    with pytest.raises(PermissionError):
        rig.approve(apid, by=alias_user)
    assert rig.status(apid) == "pending"
    # An ordinary user decides with the alias table present.
    assert rig.approve(pid)["status"] == "approved"
    with pytest.raises(ValueError):  # one-shot
        rig.approve(pid)


def test_supersession_keeps_history_and_stale_base_refused(rig):
    ev1, ev2 = rig.msg(), rig.msg("actually, detailed answers", ts=2.0)
    rig.approve(rig.propose(rig.doc(rig.pref("p1", [ev1]))))
    old = rig.pref("p1", [ev1], status="superseded", superseded_by="p2")
    new = rig.pref("p2", [ev2], statement="detailed answers", supersedes="p1")
    a = rig.propose(rig.doc(old, new))
    b = rig.propose(rig.doc(old, rig.pref("p2", [ev2], statement="other", supersedes="p1")))
    with pytest.raises(ValueError):  # deleting a preference is refused
        rig.propose(rig.doc(rig.pref("p9", [ev2])))
    assert rig.approve(a)["version"] == 2
    assert rig.approve(b)["status"] == "stale" and rig.status(b) == "stale"
    with rig.engine.connect() as c:
        prefs = json.loads(g.current_version(c, rig.user)["body"])["preferences"]
    assert [(p["id"], p["status"]) for p in prefs] == [("p1", "superseded"), ("p2", "active")]


def test_concurrent_same_base_approvals_one_wins(rig):
    ev = rig.msg()
    ids = [rig.propose(rig.doc(rig.pref("p1", [ev], statement=f"s{i}"))) for i in range(4)]
    out = {}
    threads = [threading.Thread(target=lambda i=i: out.__setitem__(i, rig.approve(i)))
               for i in ids]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(o["status"] for o in out.values()) == ["approved", "stale", "stale", "stale"]
    with rig.engine.connect() as c:
        n = c.execute(text("SELECT count(*) FROM prompt_template_versions WHERE key=:k"),
                      {"k": g.guidance_key(rig.user)}).scalar_one()
    assert n == 1


def test_failure_after_staging_rolls_back(rig, monkeypatch):
    pid = rig.propose(rig.doc(rig.pref("p1", [rig.msg()])))
    real = g.current_version
    calls = []

    def boom(conn, user_id):
        calls.append(1)
        if len(calls) == 2:  # the read after staging
            raise RuntimeError("boom")
        return real(conn, user_id)

    monkeypatch.setattr(g, "current_version", boom)
    with pytest.raises(RuntimeError):
        rig.approve(pid)
    monkeypatch.setattr(g, "current_version", real)
    assert rig.status(pid) == "pending"
    with rig.engine.connect() as c:
        assert real(c, rig.user) is None


def test_rollback_is_a_new_version_with_the_old_body(rig):
    ev1, ev2 = rig.msg(), rig.msg("terse please", ts=2.0)
    rig.approve(rig.propose(rig.doc(rig.pref("p1", [ev1]))))
    with rig.engine.connect() as c:
        v1 = g.current_version(c, rig.user)
    rig.approve(rig.propose(rig.doc(rig.pref("p1", [ev1]), rig.pref("p2", [ev2], "terse"))))
    with rig.engine.connect() as c:
        v2 = g.current_version(c, rig.user)

    def roll(row):
        return rig.approve(g.rollback_to(rig.engine, user_id=rig.user, version_row_id=row,
                                         proposed_by="al", reason="undo",
                                         excluded_bot_ids=rig.excluded))

    # Undo an added preference: exact v1 body as a fresh version 3.
    out = roll(v1["row_id"])
    with rig.engine.connect() as c:
        cur = g.current_version(c, rig.user)
    assert out["version"] == 3 and cur["body"] == v1["body"] and cur["row_id"] != v1["row_id"]
    with pytest.raises(ValueError):  # rolling back to what is already current
        g.rollback_to(rig.engine, user_id=rig.user, version_row_id=v1["row_id"],
                      proposed_by="al", reason="undo", excluded_bot_ids=rig.excluded)
    roll(v2["row_id"])  # and forward again
    # Supersede p2, then restore the pre-supersession state.
    rig.approve(rig.propose(rig.doc(
        rig.pref("p1", [ev1]),
        rig.pref("p2", [ev2], "terse", status="superseded", superseded_by="p1x"),
        rig.pref("p1x", [ev1], "short again", supersedes="p2"))))
    with rig.engine.connect() as c:
        v3 = g.current_version(c, rig.user)
    rig.approve(rig.propose(rig.doc(
        rig.pref("p1", [ev1], "shorter"),
        rig.pref("p2", [ev2], "terse", status="superseded", superseded_by="p1x"),
        rig.pref("p1x", [ev1], "short again", supersedes="p2"))))
    out = roll(v3["row_id"])
    with rig.engine.connect() as c:
        cur = g.current_version(c, rig.user)
    assert out["version"] == 7 and cur["body"] == v3["body"] and cur["row_id"] != v3["row_id"]
    out = roll(v2["row_id"])  # back before the supersession
    with rig.engine.connect() as c:
        cur = g.current_version(c, rig.user)
        n = c.execute(text("SELECT count(*) FROM prompt_template_versions WHERE key=:k"),
                      {"k": g.guidance_key(rig.user)}).scalar_one()
    assert out["version"] == 8 and cur["body"] == v2["body"] and n == 8


def test_run_receipts_pin_exact_versions_across_runs(rig):
    ev = rig.msg(ts=1.0)
    rig.approve(rig.propose(rig.doc(rig.pref("p1", [ev], "v1"))))
    run_a = g.begin_run(rig.engine, bot_id=rig.bot, user_id=rig.user,
                        excluded_bot_ids=rig.excluded)
    a = hrc.consume(rig.engine, bot_id=rig.bot, user_id=rig.user, excluded_bot_ids=rig.excluded,
                    until=100.0, dry_run=False, run=run_a)
    rig.approve(rig.propose(rig.doc(rig.pref("p1", [ev], "v2"))))
    rig.msg("more", ts=3.0)
    run_b = g.begin_run(rig.engine, bot_id=rig.bot, user_id=rig.user,
                        excluded_bot_ids=rig.excluded)
    hrc.consume(rig.engine, bot_id=rig.bot, user_id=rig.user, excluded_bot_ids=rig.excluded,
                until=100.0, dry_run=False, run=run_b)
    rig.approve(rig.propose(rig.doc(rig.pref("p1", [ev], "v3"))))
    ra, rb = g.resolve_run(rig.engine, run_a["receipt_id"]), g.resolve_run(rig.engine, run_b["receipt_id"])
    assert (ra["status"], ra["guidance_version"], rb["guidance_version"]) == ("committed", 1, 2)
    assert json.loads(ra["body"])["preferences"][0]["statement"] == "v1"
    assert json.loads(rb["body"])["preferences"][0]["statement"] == "v2"
    assert ra["cursor_after_id"] == a["cursor_after"][1] and rb["consumed_count"] == 1
    with rig.engine.begin() as c:  # terminal receipts are immutable
        with pytest.raises(ValueError):
            g.finish_run(c, handle=run_a, status="failed", before=None, after=None, consumed=0)


def test_cursor_conflict_marks_receipt_failed(rig):
    rig.msg(ts=1.0)
    run = g.begin_run(rig.engine, bot_id=rig.bot, user_id=rig.user, excluded_bot_ids=rig.excluded)
    hrc.consume(rig.engine, bot_id=rig.bot, user_id=rig.user, excluded_bot_ids=rig.excluded,
                until=100.0, dry_run=False)  # create the cursor
    rig.msg(ts=2.0)
    real_cas = hrc._CAS
    try:
        hrc._CAS = text("SELECT NULL WHERE false")  # simulate a lost CAS race
        out = hrc.consume(rig.engine, bot_id=rig.bot, user_id=rig.user,
                          excluded_bot_ids=rig.excluded, until=100.0, dry_run=False, run=run)
    finally:
        hrc._CAS = real_cas
    assert out["status"] == "conflict"
    assert g.resolve_run(rig.engine, run["receipt_id"])["status"] == "failed"
    with pytest.raises(ValueError):  # handle for another pair
        hrc.consume(rig.engine, bot_id=rig.bot, user_id=rig.other,
                    excluded_bot_ids=rig.excluded, run=run)
