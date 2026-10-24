"""TASK-1301 operator search: classification, from:/in: filters, row-set parity.

Private PostgreSQL, isolated schema, synthetic bots only. The punctuation slug
``public-fixture`` maps to partition ``publicfixture`` so canonical slug and
partition are never confused.
"""
from __future__ import annotations

import asyncio
import uuid

import pytest
from sqlalchemy import create_engine, text

from llm_bawt.bot_access import BotAccessError, BotPolicyResolver, TrustedCaller
from llm_bawt.mcp_server.message_search import CrossBotMessageSearcher, OperatorMessageScope
from llm_bawt.mcp_server.storage import MemoryStorage
from llm_bawt.memory.message_tsv import ensure_message_tsv
from llm_bawt.message_authorship import AuthorReference
from llm_bawt.utils import db
from tests.bot_tag_fixtures import ensure_tag_tables
from tests.pg_isolation import require_test_db_config

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]

# (id, partition, session, role, author_type, author_id, expected author_class)
ROWS = [
    ("a_reply", "publicfixture", "pub", "assistant", "bot", "public-fixture", "bot"),
    ("a_blank", "publicfixture", "pub", "assistant", None, None, "bot"),
    ("a_orphan", "publicfixture", None, "assistant", None, None, "bot"),
    ("a_foreign", "publicfixture", "pub", "assistant", "bot", "peer", None),
    ("a_partial", "publicfixture", "pub", "assistant", "bot", None, None),
    ("u_a", "publicfixture", "pub", "user", "user", "user_a", "user"),
    ("u_b", "publicfixture", "pub_b", "user", "user", "user_b", "user"),
    ("u_blank", "publicfixture", "pub", "user", None, None, None),
    ("u_mismatch", "publicfixture", "pub", "user", "user", "user_b", None),
    ("u_foreign_session", "publicfixture", "peer1", "user", "user", "user_b", None),
    ("u_missing_session", "publicfixture", "missing", "user", "user", "user_a", None),
    ("d_peer", "publicfixture", "pub", "user", "bot", "peer", "delivery"),
    ("d_private", "publicfixture", "pub", "user", "bot", "private_fixture", None),
    ("d_unknown", "publicfixture", "pub", "user", "bot", "ghost", None),
    ("p_reply", "peer", "peer1", "assistant", "bot", "peer", "bot"),
    ("p_u_b", "peer", "peer1", "user", "user", "user_b", "user"),
    ("p_d_pub", "peer", "peer1", "user", "bot", "public-fixture", "delivery"),
]
PUBLIC_IDS = {row[0] for row in ROWS}


@pytest.fixture
def engine(private_app_pg):
    url = db.build_postgres_url(require_test_db_config())
    root = create_engine(url)
    schema = "opsearch_" + uuid.uuid4().hex
    with root.begin() as conn:
        conn.execute(text(f"CREATE SCHEMA {schema}"))
    eng = create_engine(url, connect_args={"options": f"-csearch_path={schema},public"})
    with eng.begin() as conn:
        conn.execute(text("CREATE TABLE bot_profiles (slug text, include_in_global_search boolean, "
                          "name text, color text, avatar text, avatar_render text)"))
        conn.execute(text("INSERT INTO bot_profiles (slug, include_in_global_search, name) VALUES "
                          "('public-fixture', true, 'Public Fixture'), ('peer', true, 'Peer'), "
                          "('private_fixture', false, 'Private')"))
        ensure_tag_tables(conn)  # migrated catalog; private_fixture is private by its column
        # Read by the production display hydrator (presentation + legacy delivery lookup).
        conn.execute(text("CREATE TABLE entity_profiles (entity_type text, entity_id text, "
                          "display_name text, color text, avatar text, avatar_render text)"))
        conn.execute(text("INSERT INTO entity_profiles (entity_type, entity_id, display_name) VALUES "
                          "('user', 'user_a', 'User A'), ('user', 'user_b', 'User B')"))
        conn.execute(text("CREATE TABLE inter_bot_deliveries (user_message_id text, sender_bot_id text)"))
        # Collisions (Codex r2): delivery records that legacy hydration would use to
        # reattribute rows the SQL rule already classified (a_blank → bot) or left
        # Unknown (u_blank). Display must follow author_class, not these.
        conn.execute(text("INSERT INTO inter_bot_deliveries VALUES ('a_blank', 'peer'), ('u_blank', 'peer')"))
        conn.execute(text("CREATE TABLE sessions (id text, bot_id text, user_id text, status text)"))
        conn.execute(text("""INSERT INTO sessions VALUES
            ('pub', 'public-fixture', 'user_a', 'active'),
            ('pub_b', 'public-fixture', 'user_b', 'archived'),
            ('peer1', 'peer', 'user_b', 'active'),
            ('priv', 'private_fixture', 'user_a', 'active')"""))
        conn.execute(text("""CREATE TABLE messages (
            id text, bot_id text, session_id text, role text, content text,
            timestamp double precision, author_entity_type text, author_entity_id text)"""))
        ensure_message_tsv(conn, table_was_new=True)
        rows = ROWS + [
            ("x_reply", "private_fixture", "priv", "assistant", "bot", "private_fixture", None),
            ("x_user", "private_fixture", "priv", "user", "user", "user_a", None),
            ("sys", "publicfixture", "pub", "system", None, None, None),
        ]
        for ts, (mid, part, session, role, atype, aid, _) in enumerate(rows, start=1):
            conn.execute(text("INSERT INTO messages VALUES (:id, :bot, :s, :role, :c, :ts, :t, :a)"),
                         {"id": mid, "bot": part, "s": session, "role": role,
                          "c": f"needle haystack {mid}", "ts": float(ts), "t": atype, "a": aid})
    try:
        yield eng
    finally:
        eng.dispose()
        with root.begin() as conn:
            conn.execute(text(f"DROP SCHEMA {schema} CASCADE"))
        root.dispose()


def search(engine, *, mode="trgm", query="needle", author=None, in_bot=None,
           limit=100, sort_by="relevance"):
    access = BotPolicyResolver(engine).resolve_sync(None)
    scope = OperatorMessageScope.from_access(access, author=author, in_bot=in_bot)
    searcher = CrossBotMessageSearcher(engine, lambda rows, source: rows)
    return searcher.search_operator(mode, query, scope, n_results=limit, role_filter=None,
                                    sort_by=sort_by, since=None, until=None)


@pytest.mark.parametrize("mode", ["fts", "trgm"])
def test_unfiltered_row_set_is_public_partitions_with_classes(engine, mode):
    rows = search(engine, mode=mode)
    assert {r["id"] for r in rows} == PUBLIC_IDS
    assert {r["total"] for r in rows} == {len(PUBLIC_IDS)}
    assert {r["id"]: r["author_class"] for r in rows} == {row[0]: row[6] for row in ROWS}
    # Source is the canonical slug, not the partition name.
    assert {r["source"] for r in rows} == {"public-fixture", "peer"}


@pytest.mark.parametrize("mode", ["fts", "trgm"])
@pytest.mark.parametrize(("author", "expected"), [
    (AuthorReference.user("user_a"), {"u_a"}),
    (AuthorReference.user("user_b"), {"u_b", "p_u_b"}),
    (AuthorReference.bot("public-fixture"), {"a_reply", "a_blank", "a_orphan", "p_d_pub"}),
    (AuthorReference.bot("peer"), {"p_reply", "d_peer"}),
    (AuthorReference.bot("private_fixture"), set()),
    (AuthorReference.bot("ghost"), set()),
    (AuthorReference.user("ghost"), set()),
])
def test_author_filter_uses_computed_class(engine, mode, author, expected):
    rows = search(engine, mode=mode, author=author)
    assert {r["id"] for r in rows} == expected


@pytest.mark.parametrize("in_bot", ["private_fixture", "ghost", "publicfixture"])
def test_protected_unknown_or_partition_in_bot_is_ordinary_empty(engine, in_bot):
    assert search(engine, in_bot=in_bot) == []


def test_in_bot_narrows_to_one_canonical_partition(engine):
    rows = search(engine, mode="fts", in_bot="peer")
    assert {r["id"] for r in rows} == {"p_reply", "p_u_b", "p_d_pub"}


def test_filter_applies_before_limit_and_window_count(engine):
    rows = search(engine, author=AuthorReference.bot("public-fixture"), limit=1)
    assert len(rows) == 1
    assert rows[0]["total"] == 4


def test_filter_only_query_is_newest_first(engine):
    rows = search(engine, query="  ", author=AuthorReference.user("user_b"))
    assert [r["id"] for r in rows] == ["p_u_b", "u_b"]
    assert {r["rank"] for r in rows} == {0.0}
    assert search(engine, query="") == []


def test_policy_failure_is_not_an_empty_or_unrestricted_search(engine):
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM bot_profiles"))
    with pytest.raises(BotAccessError, match="policy_unavailable"):
        search(engine)


def production_search(engine, *, mode, **kwargs):
    """Through MemoryStorage, so the real author hydrator runs (Codex r1 blocker)."""
    storage = MemoryStorage.__new__(MemoryStorage)
    storage.engine = engine
    return asyncio.run(storage.search_messages_operator("needle", mode=mode, n_results=100, **kwargs))


@pytest.mark.parametrize("mode", ["fts", "trgm"])
def test_production_hydrator_never_contradicts_author_class(engine, mode):
    rows = production_search(engine, mode=mode)
    assert {r["id"] for r in rows} == PUBLIC_IDS
    by_id = {r["id"]: r for r in rows}
    unknown = {row[0] for row in ROWS if row[6] is None}
    # Partial (a_partial), foreign and mismatched identities stay Unknown; no guessed
    # author from legacy hydration and no ValueError failing the whole result set.
    assert {i for i, r in by_id.items() if r["author"]["entity_id"] is None} == unknown
    assert {by_id[i]["author"]["display_name"] for i in unknown} == {"Unknown participant"}
    assert by_id["u_a"]["author"]["display_name"] == "User A"
    # Delivery-record collision: class wins over the legacy delivery lookup.
    assert by_id["a_blank"]["author_class"] == "bot"
    assert (by_id["a_blank"]["author"]["entity_id"], by_id["a_blank"]["author"]["status"]) == (
        "public-fixture", "legacy")
    assert by_id["a_reply"]["author"]["status"] == "canonical"
    for i, r in by_id.items():
        expected = {"bot": ("bot", r["source"]),
                    "user": ("user", r["author_entity_id"]),
                    "delivery": ("bot", r["author_entity_id"])}.get(r["author_class"])
        if expected:
            assert (r["author"]["entity_type"], r["author"]["entity_id"]) == expected, i


@pytest.mark.parametrize("mode", ["fts", "trgm"])
def test_production_hydrator_with_author_filter(engine, mode):
    rows = production_search(engine, mode=mode, author=AuthorReference.bot("public-fixture"))
    assert {r["id"] for r in rows} == {"a_reply", "a_blank", "a_orphan", "p_d_pub"}
    assert {(r["author"]["entity_id"], r["author"]["display_name"]) for r in rows} == {
        ("public-fixture", "Public Fixture")}


def test_missing_engine_is_source_unavailable_not_500():
    storage = MemoryStorage.__new__(MemoryStorage)
    storage.engine = None
    with pytest.raises(BotAccessError, match="source_unavailable"):
        asyncio.run(storage.search_messages_operator("needle"))


def test_trusted_caller_cannot_use_operator_scope(engine):
    access = BotPolicyResolver(engine).resolve_sync(TrustedCaller("peer", "user_b", "service"))
    with pytest.raises(BotAccessError, match="caller_unavailable"):
        OperatorMessageScope.from_access(access)
