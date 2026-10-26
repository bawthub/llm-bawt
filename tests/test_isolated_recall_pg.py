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
from tests.test_fts_query_pg import assert_no_stored_vector, result_columns
from tests.test_isolated_recall_sources import AL_REPLY, NICK_ASK

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


def test_recall_never_returns_the_stored_vector(rig):
    """content_tsv drives the match and rank only; no recall row carries it (TASK-1323)."""
    stale, *_rest = thread_with_dense_competitors(rig)
    with result_columns(rig.engine) as columns:
        rows = rig.recall(QUESTION)
    assert stale in {row.id for row in rows}
    assert_no_stored_vector(columns)


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
    """52f101ef shape at equal coverage: a terse statement now vs a denser one weeks ago."""
    assert RECENCY_HALF_WEIGHT_DAYS == 14.0
    old = rig.add("Continuity setting: continuity uses messages and summaries for continuity.",
                  at=rig.now - 30 * DAY)
    new = rig.add("Continuity setting is now messages only.", at=rig.now - 60)
    with rig.engine.connect() as conn:
        raw = dict(conn.execute(text("""
            SELECT id, ts_rank(to_tsvector('english', content), to_tsquery('english', :q))
            FROM messages WHERE bot_id = :bot
        """), {"q": fts_query("continuity setting"), "bot": rig.bot}).all())
    assert raw[old] > raw[new]  # without recency the stale dense row wins
    assert [row.id for row in rig.recall("continuity setting")] == [new, old]


OPENCLAW_QUESTION = ("What happened when Nick ran openclaw auth login openai-codex in March, "
                     "and what was the error?")


def test_coverage_outranks_recency_for_an_old_exact_answer(rig):
    """Live miss: a 225-day-old exact answer lost to today's 1-2 generic-word matches."""
    march = rig.add("Ran `openclaw auth login openai-codex` and it failed with: error: unknown "
                    "command 'auth'. The login subcommand moved.", at=rig.now - 225 * DAY)
    recent = [rig.add(text_, at=rig.now - 60 * (i + 1)) for i, text_ in enumerate([
        "Nick, that error is gone now.",
        "What happened with the deploy? Nick asked about it.",
        "The error in the log was a timeout.",
        "Nick ran the migration in March.",
        "Nothing happened after the restart.",
        "Error budget is fine.",
        "Nick said the error happened twice.",
    ])]
    rows = rig.recall(OPENCLAW_QUESTION)
    assert rows[0].id == march
    assert set(row.id for row in rows) - {march} <= set(recent)
    # Recency still decides between answers in the same coverage bucket (both
    # 7/10 lexemes, the newer one terser): newest first.
    newer = rig.add("Ran openclaw auth login openai-codex: error, unknown command.",
                    at=rig.now - 3 * DAY)
    assert [row.id for row in rig.recall(OPENCLAW_QUESTION)][:2] == [newer, march]


def test_literal_receipt_beats_new_generic_history_and_keeps_provenance(rig):
    receipt = rig.add("TASK-1144 app recreate 8ee7d89 succeeded; live health checks passed.",
                      at=rig.now - 80 * DAY)
    for i in range(120):
        rig.add("The app is healthy after a routine check.", at=rig.now - i * 10)
    rows = rig.recall("What was the exact tip and live health result for TASK-1144 app recreate 8ee7d89?")
    assert rows and rows[0].id == receipt
    assert rows[0].source_bot == rig.bot and rows[0].session_id == rig.session
    assert "health checks passed" in rows[0].text


def test_public_cross_bot_recall_is_positive_scoped_with_canonical_authors(rig):
    public = f"public{uuid.uuid4().hex[:10]}"
    excluded = f"excluded{uuid.uuid4().hex[:10]}"
    public_session = rig.add_session(str(uuid.uuid4()), bot=public)
    excluded_session = rig.add_session(str(uuid.uuid4()), bot=excluded)
    different_user = rig.add_session(str(uuid.uuid4()), bot=public, user="other-user")
    body = "TASK-1144 app recreate 8ee7d89; live health checks passed."
    good = rig.add(body, at=rig.now - 40, partition=public, session=public_session,
                   author=public, author_type="bot")
    denied = rig.add(body, at=rig.now - 30, partition=excluded, session=excluded_session,
                     author=excluded, author_type="bot")
    other = rig.add(body, at=rig.now - 20, partition=public, session=different_user,
                    author=public, author_type="bot")
    imported = rig.add(body, at=rig.now - 10, partition=public, session=public_session,
                       author="unknown", author_type="bot")
    rows = rig.recall("TASK-1144 app recreate 8ee7d89", allowed=frozenset({public}),
                      excluded=frozenset({excluded}))
    assert [row.id for row in rows] == [good]
    assert rows[0].source_bot == public and rows[0].session_id == public_session
    assert {denied, other, imported}.isdisjoint(row.id for row in rows)
    # A policy change is reflected on the next request, including follow-ups.
    assert not rig.recall("TASK-1144 app recreate 8ee7d89", allowed=frozenset(),
                          excluded=frozenset({public, excluded}))


def test_nonliteral_identifier_does_not_degrade_to_generic_old_fragment(rig):
    rig.add("TASK-975 was an old unrelated app health check.", at=rig.now - 10)
    assert not rig.recall("Who owns HomeBotBand WIP?")


def test_missing_literal_does_not_waive_lexical_floor_for_loose_tier(rig):
    rig.add("The app was healthy yesterday after a routine check.", at=rig.now - 10)
    assert not rig.recall("What happened to HomeBotBand yesterday?")


@pytest.mark.parametrize("weak_literal", ["e.g.", "i.e.", "3:30"])
def test_prose_abbreviation_and_clock_do_not_override_lexical_evidence(rig, weak_literal):
    unrelated = rig.add(f"Use a smaller model, {weak_literal} haiku, for unrelated chores.",
                        at=rig.now - 5)
    relevant = rig.add("Deploy order: app first, then workers; that was our decision.",
                       at=rig.now - DAY)
    rows = rig.recall(f"What did we decide about the deploy order, {weak_literal} app first?")
    assert rows and rows[0].id == relevant
    assert unrelated not in {row.id for row in rows}


def test_broad_single_word_hit_is_not_presented_as_a_relevant_answer(rig):
    rig.add("The budget was adjusted in an unrelated initiative.", at=rig.now - 10)
    assert not rig.recall("What were the migration findings and owners of the handshake budget?")


def test_review_reference_is_omitted_if_one_excerpt_mentions_two_reviews():
    from llm_bawt.recall.postgres import review_ref
    assert review_ref("r8 B5 was revised by r9 B5") is None
    assert review_ref("r9 B5 was complete") == "r9"
    assert review_ref("Review r9 confirmed the result") == "r9"
    assert review_ref("Cloudflare R2 bucket") is None
    assert review_ref("R2 storage") is None


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


@pytest.mark.parametrize("content,query", [
    (AL_REPLY, "steer"), (NICK_ASK, "hardcoded"),
    ("lead in " + "z" * 300 + "needle tail", "needle"),
    ("short needle", "needle"), ("no literal hit here", "absent"),
    ("x" * 30 + " " + "y" * 45 + "needle", "needle"),
])
def test_sql_window_matches_python_twin_and_opens_on_word_start(engine, content, query):
    from llm_bawt.recall.contracts import MAX_RECORD_CHARS
    from llm_bawt.recall.excerpts import excerpt_window, sql_window, terms_for
    columns, lateral = sql_window("m.content")
    with engine.connect() as conn:
        row = conn.execute(text(
            f"SELECT {columns} FROM (SELECT CAST(:content AS text) AS content) m {lateral}"
        ), {"content": content, "phrase": query.lower(), "terms": terms_for(query),
            "chars": MAX_RECORD_CHARS}).mappings().one()
    excerpt, start, _, kind = excerpt_window(content, query)
    # Window parity only: the twins' `truncated` flags already differ (SQL compares
    # total length to the window size; Python also counts a skipped head).
    assert (row["excerpt"], row["start_char"], row["window_kind"]) == (excerpt, start, kind)
