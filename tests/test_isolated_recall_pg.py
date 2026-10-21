"""Recall message retrieval on real SQL in a private PostgreSQL cluster (TASK-1210).

Shapes come from two live misses:
* a stale "Messages + summaries" answer outranked the operator's term-free
  correction and the terse confirmation that followed it (R2 follow-ons, R1);
* Caid's deliveries in the caller's own history carried the answer but were
  excluded by authorship (A1), and identifier terms were dropped.

Run: pytest -m integration tests/test_isolated_recall_pg.py
"""
from __future__ import annotations

import asyncio
import time
import uuid

import pytest
from sqlalchemy import create_engine, text

from llm_bawt.memory.postgresql import ensure_bot_partitions
from llm_bawt.recall.contracts import (
    MAX_FOLLOW_ONS_PER_HIT, MAX_SOURCE_RECORDS, RECENCY_HALF_WEIGHT_DAYS, RecallScope,
)
from llm_bawt.recall.excerpts import fts_query
from llm_bawt.recall.postgres import MessageSource, RecallDatabase
from llm_bawt.utils import db
from tests.pg_isolation import require_test_db_config

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]

USER = "operator"
DAY = 86400.0
STALE = "Done. The idle context reset job continuity setting now uses **Messages + summaries**."
CORRECTION = "No summaries. I said message history"
CONFIRMATION = ("Fixed it, sorry about that. It is now **Messages only**, the reset carries "
                "`continuity: inline` and nothing else changes for this one.")
QUESTION = "What continuity setting did Nick ask for on the idle context reset job, and what did I set it to?"


@pytest.fixture(scope="module")
def engine(private_app_pg):
    engine = create_engine(db.build_postgres_url(require_test_db_config()))
    try:
        yield engine
    finally:
        engine.dispose()


class Rig:
    def __init__(self, engine):
        self.engine = engine
        self.bot = f"recall{uuid.uuid4().hex[:12]}"
        self.session = str(uuid.uuid4())
        self.now = time.time()
        with engine.begin() as conn:
            ensure_bot_partitions(conn, self.bot)
        self.add_session(self.session)

    def add_session(self, session_id, *, bot=None, user=USER, status="active"):
        with self.engine.begin() as conn:
            conn.execute(text("INSERT INTO sessions (id, bot_id, user_id, status) VALUES (:id, :bot, :user, :status)"),
                         {"id": session_id, "bot": bot or self.bot, "user": user, "status": status})
        return session_id

    def add(self, content, *, at, role="assistant", author=None, author_type=None,
            session=None, partition=None):
        if author is None:
            author, author_type = (USER, "user") if role == "user" else (self.bot, "bot")
        message_id = str(uuid.uuid4())
        partition = partition or self.bot
        with self.engine.begin() as conn:
            ensure_bot_partitions(conn, partition)
            conn.execute(text("""
                INSERT INTO messages (bot_id, id, role, content, timestamp, session_id,
                                      author_entity_type, author_entity_id)
                VALUES (:partition, :id, :role, :content, :at, :session, :type, :author)
            """), {"partition": partition, "id": message_id, "role": role, "content": content,
                   "at": at, "session": session or self.session, "type": author_type, "author": author})
        return message_id

    def recall(self, query, *, allowed=frozenset({"peer"}), excluded=frozenset({"protected"})):
        scope = RecallScope(self.bot, USER, self.session, excluded, allowed | {self.bot} | excluded)
        return asyncio.run(MessageSource(RecallDatabase(self.engine)).search(query, scope, ()))


@pytest.fixture
def rig(engine):
    return Rig(engine)


def thread_with_dense_competitors(rig):
    """Five dense older hits outrank the stale answer; the confirmation ranks below six."""
    other = rig.add_session(str(uuid.uuid4()), status="archived")
    dense = "Idle context reset job: continuity setting for the idle context reset job, Nick asked."
    for hours in range(5):
        rig.add(dense, at=rig.now - DAY - hours * 3600, session=other)
    base = rig.now - 600
    rig.add("Update the setting to enforce consistent msg history continuity", at=base - 30, role="user")
    stale = rig.add(STALE, at=base)
    # A protected bot's delivery inside the caller's own session, between the
    # stale answer and the correction: never admitted, never consumes a slot.
    protected = rig.add("idle context reset job continuity setting summaries", at=base + 5,
                        role="user", author="protected", author_type="bot")
    correction = rig.add(CORRECTION, at=base + 15, role="user")
    confirmation = rig.add(CONFIRMATION, at=base + 28)
    return stale, protected, correction, confirmation


def test_term_free_correction_and_confirmation_follow_the_stale_hit(rig):
    stale, protected, correction, confirmation = thread_with_dense_competitors(rig)
    rows = rig.recall(QUESTION)
    by_id = {row.id: row for row in rows}

    hits = [row for row in rows if row.follows is None]
    assert len(hits) == MAX_SOURCE_RECORDS
    assert by_id[stale].follows is None
    # Neither later message is a direct hit; expansion is what reaches them.
    assert by_id[correction].follows == stale
    assert by_id[confirmation].follows == stale
    assert protected not in by_id
    # Follow-ons sit right after their hit, in conversation order.
    ids = [row.id for row in rows]
    assert ids[ids.index(stale) + 1: ids.index(stale) + 3] == [correction, confirmation]
    assert by_id[correction].speaker == "user" and by_id[confirmation].speaker == "assistant"
    assert by_id[stale].at < by_id[correction].at < by_id[confirmation].at
    assert by_id[stale].at.endswith("Z") and len(by_id[stale].at) == len("2026-10-21T10:33:37.000000Z")
    assert len(rows) <= MAX_SOURCE_RECORDS * (1 + MAX_FOLLOW_ONS_PER_HIT)


def test_follow_ons_never_cross_sessions_bots_users_or_window(rig):
    base = rig.now - 3600
    hit = rig.add("deploy checklist for the frontend release", at=base)
    rig.add("other session reply", at=base + 1, session=rig.add_session(str(uuid.uuid4()), status="archived"))
    rig.add("other user's session reply", at=base + 2,
            session=rig.add_session(str(uuid.uuid4()), user="someone-else"))
    rig.add("another bot's partition, same session id", at=base + 3, partition=f"other{uuid.uuid4().hex[:8]}")
    rig.add("allowed peer, unrelated session", at=base + 4, role="user", author="peer", author_type="bot",
            session=rig.add_session(str(uuid.uuid4()), status="archived"))
    rig.add("unknown bot delivery in my session", at=base + 5, role="user", author="stranger", author_type="bot")
    rig.add("too late to be the same exchange", at=base + 3600)
    rows = rig.recall("frontend deploy checklist")
    assert [row.id for row in rows] == [hit]


def test_hit_that_is_also_a_follow_on_is_quoted_once_as_a_hit(rig):
    first = rig.add("continuity is messages plus summaries", at=rig.now - 100)
    second = rig.add("continuity is messages only", at=rig.now - 90)
    rows = rig.recall("continuity")
    assert sorted(row.id for row in rows) == sorted([first, second])
    assert all(row.follows is None for row in rows)


def test_recency_lifts_the_terse_newer_statement_over_an_older_dense_one(rig):
    """52f101ef shape: one-term confirmation now vs a denser statement weeks ago."""
    assert RECENCY_HALF_WEIGHT_DAYS == 14.0
    old = rig.add("Continuity setting: continuity uses messages and summaries for continuity.",
                  at=rig.now - 30 * DAY)
    new = rig.add(CONFIRMATION, at=rig.now - 60)
    with rig.engine.connect() as conn:
        raw = dict(conn.execute(text("""
            SELECT id, ts_rank(to_tsvector('english', content), to_tsquery('english', :q))
            FROM messages WHERE bot_id = :bot
        """), {"q": fts_query("continuity setting"), "bot": rig.bot}).all())
    assert raw[old] > raw[new]  # without recency the stale dense row wins
    assert [row.id for row in rig.recall("continuity setting")] == [new, old]


def test_received_deliveries_follow_the_current_allowed_policy(rig):
    query = "What did Caid say about the maintenance_model and query_bounded for TASK-1210?"
    body = "TASK-1210: maintenance_model stays unset; query_bounded caps the read."
    peer = rig.add(body, at=rig.now - 50, role="user", author="peer", author_type="bot")
    protected = rig.add(body, at=rig.now - 40, role="user", author="protected", author_type="bot")
    stranger = rig.add(body, at=rig.now - 30, role="user", author="stranger", author_type="bot")
    # A peer-authored assistant row is an import, not a delivery the caller received.
    imported = rig.add(body, at=rig.now - 20, role="assistant", author="peer", author_type="bot")
    rows = rig.recall(query)
    assert [row.id for row in rows] == [peer]
    assert rows[0].speaker == "bot:peer"
    # Policy is read now: once peer is excluded, its old deliveries disappear.
    assert rig.recall(query, allowed=frozenset(), excluded=frozenset({"peer", "protected"})) == []
    assert {protected, stranger, imported}.isdisjoint(row.id for row in rig.recall(query))
