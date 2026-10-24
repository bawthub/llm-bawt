"""TASK-1323 stored ``content_tsv``: bootstrap, trigger, and the online upgrade.

Private PostgreSQL. Fresh-bootstrap cases use the module's app-bootstrapped
public schema; upgrade cases build a pre-1323 partitioned parent (legacy
expression index, no stored vector) in an isolated schema.
"""
from __future__ import annotations

import logging
import uuid

import pytest
from sqlalchemy import create_engine, event, text
from sqlalchemy.exc import DBAPIError

from llm_bawt.mcp_server.message_search import CrossBotMessageSearcher
from llm_bawt.memory.message_store import ensure_message_parent_tables
from llm_bawt.memory.message_tsv import (
    INDEX,
    MessageTsvUpgradeReport,
    drop_legacy_fts_index,
    leaf_index_name,
    message_tsv_state,
    upgrade_message_tsv,
)
from llm_bawt.memory.postgresql import ensure_bot_partitions
from llm_bawt.utils import db
from tests.pg_isolation import require_test_db_config

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]

ALPHA = "messages_p_alpha"
# 63 bytes, PostgreSQL's identifier limit: "<leaf>_..._idx" would not fit.
LONG_LEAF = "messages_p_" + "x" * 52
ALPHA_ROWS = 2002  # a multiple of BATCH (last batch is empty); big enough for the planner to pick GIN
LONG_ROWS = 13
BATCH = 7  # rows per keyset batch
LEGACY_INDEX = "messages_content_fts_idx"  # the pre-1323 name (48837b5)

STALE_ROWS = ("SELECT count(*) FROM messages "
              "WHERE content_tsv IS DISTINCT FROM to_tsvector('english', content)")


@pytest.fixture
def public(private_app_pg):
    engine = create_engine(db.build_postgres_url(require_test_db_config()))
    try:
        yield engine
    finally:
        engine.dispose()


@pytest.fixture
def legacy(private_app_pg):
    """A pre-1323 ``messages``: rows, legacy expression index, no stored vector."""
    url = db.build_postgres_url(require_test_db_config())
    root = create_engine(url)
    schema = "tsvupgrade_" + uuid.uuid4().hex
    with root.begin() as conn:
        conn.execute(text(f"CREATE SCHEMA {schema}"))
    eng = create_engine(url, connect_args={"options": f"-csearch_path={schema},public"})
    with eng.begin() as conn:
        conn.execute(text("""CREATE TABLE messages (
            bot_id text NOT NULL, id text NOT NULL, role text NOT NULL,
            content text NOT NULL, timestamp double precision NOT NULL,
            session_id text, author_entity_type text, author_entity_id text,
            PRIMARY KEY (bot_id, id)) PARTITION BY LIST (bot_id)"""))
        conn.execute(text(f"CREATE INDEX {LEGACY_INDEX} ON messages "
                          "USING gin (to_tsvector('english', content))"))
        conn.execute(text(f"CREATE TABLE {ALPHA} PARTITION OF messages FOR VALUES IN ('alpha')"))
        conn.execute(text(f"CREATE TABLE {LONG_LEAF} PARTITION OF messages FOR VALUES IN ('long')"))
        for bot, count in (("alpha", ALPHA_ROWS), ("long", LONG_ROWS)):
            conn.execute(text(
                "INSERT INTO messages (bot_id, id, role, content, timestamp) "
                "SELECT :bot, 'm' || lpad(g::text, 5, '0'), 'user', "
                "'filler words number ' || g, g FROM generate_series(1, :n) g"),
                {"bot": bot, "n": count})
        conn.execute(text("INSERT INTO messages (bot_id, id, role, content, timestamp) "
                          "VALUES ('alpha', 'hit', 'user', 'Picked up TASK-1281 today', 999)"))
    try:
        yield eng
    finally:
        eng.dispose()
        with root.begin() as conn:
            conn.execute(text(f"DROP SCHEMA {schema} CASCADE"))
        root.dispose()


def state(engine):
    with engine.connect() as conn:
        return message_tsv_state(conn)


def upgrade(engine):
    return upgrade_message_tsv(engine, batch_size=BATCH, lock_pause=0)


def test_fresh_bootstrap_installs_the_whole_contract(public):
    current = state(public)
    assert current.complete, current.missing()
    assert current.index_name == INDEX
    assert not current.legacy_index


def test_trigger_fills_on_insert_and_on_content_update(public):
    bot = "tsv" + uuid.uuid4().hex[:10]
    with public.begin() as conn:
        ensure_bot_partitions(conn, bot)
        conn.execute(text("INSERT INTO messages (bot_id, id, role, content, timestamp) "
                          "VALUES (:bot, 'm1', 'user', 'Picked up TASK-1281 today', 1)"), {"bot": bot})
    with public.begin() as conn:
        assert conn.execute(text("SELECT content_tsv @@ to_tsquery('english', 'picked') "
                                 "FROM messages WHERE bot_id = :bot"), {"bot": bot}).scalar_one()
        conn.execute(text("UPDATE messages SET content = 'rewritten entirely' WHERE bot_id = :bot"),
                     {"bot": bot})
        conn.execute(text("UPDATE messages SET processed = true WHERE bot_id = :bot"), {"bot": bot})
        assert conn.execute(text(f"{STALE_ROWS} AND bot_id = :bot"), {"bot": bot}).scalar_one() == 0
        assert conn.execute(text("SELECT content_tsv @@ to_tsquery('english', 'rewritten') "
                                 "FROM messages WHERE bot_id = :bot"), {"bot": bot}).scalar_one()


def test_oversized_vector_falls_back_instead_of_failing_the_write(public):
    bot = "tsvbig" + uuid.uuid4().hex[:10]
    huge = ("SELECT string_agg(md5(g::text), ' ') FROM generate_series(1, 60000) g")
    with public.connect() as conn:
        with pytest.raises(DBAPIError, match="too long for tsvector"):
            conn.execute(text(f"SELECT to_tsvector('english', ({huge}))")).scalar_one()
    with public.begin() as conn:
        ensure_bot_partitions(conn, bot)
        conn.execute(text("INSERT INTO messages (bot_id, id, role, content, timestamp) "
                          f"VALUES (:bot, 'big', 'user', ({huge}), 1)"), {"bot": bot})
        assert conn.execute(text(
            "SELECT content_tsv = to_tsvector('english', left(content, 100000)) "
            "FROM messages WHERE bot_id = :bot"), {"bot": bot}).scalar_one()


def test_existing_parent_is_reported_not_altered_at_startup(legacy, caplog):
    with caplog.at_level(logging.ERROR, logger="llm_bawt.memory.message_tsv"):
        with legacy.begin() as conn:
            ensure_message_parent_tables(conn)
    assert any("upgrade_message_tsv.py" in record.getMessage() for record in caplog.records)
    current = state(legacy)
    assert not current.column and not current.trigger and current.index_name is None
    assert current.legacy_index
    # Writes keep working on the un-upgraded parent.
    with legacy.begin() as conn:
        conn.execute(text("INSERT INTO messages (bot_id, id, role, content, timestamp) "
                          "VALUES ('alpha', 'later', 'user', 'still writable', 1000)"))


def scoped_plan(engine):
    captured = []

    def capture(_conn, _cursor, statement, parameters, _context, _executemany):
        captured.append((statement, parameters))

    event.listen(engine, "before_cursor_execute", capture)
    try:
        rows = CrossBotMessageSearcher(engine, lambda rows, _source: rows).search_fts(
            "TASK-1281", partitions=frozenset({"alpha"}), n_results=10, role_filter=None,
            sort_by="relevance", since=None, until=None)
    finally:
        event.remove(engine, "before_cursor_execute", capture)
    statement, parameters = captured[-1]
    with engine.connect() as conn:
        conn.exec_driver_sql("ANALYZE messages")
        conn.exec_driver_sql("SET enable_seqscan = off")
        plan = "\n".join(row[0] for row in conn.exec_driver_sql(
            "EXPLAIN " + statement, parameters).fetchall())
    return [row["id"] for row in rows], plan


def test_upgrade_backfills_indexes_and_is_idempotent(legacy):
    report = upgrade(legacy)
    assert report.column_added and report.trigger_installed and report.parent_index_created
    assert report.rows_backfilled == {ALPHA: ALPHA_ROWS + 1, LONG_LEAF: LONG_ROWS}
    expected = sorted(leaf_index_name(leaf) for leaf in (ALPHA, LONG_LEAF))
    assert sorted(report.leaf_indexes_built) == expected
    assert sorted(report.leaf_indexes_attached) == expected
    assert all(len(name) <= 63 for name in expected)
    current = state(legacy)
    assert current.complete, current.missing()
    with legacy.connect() as conn:
        assert conn.execute(text(STALE_ROWS)).scalar_one() == 0

    ids, plan = scoped_plan(legacy)
    assert ids == ["hit"]
    assert f"Bitmap Index Scan on {leaf_index_name(ALPHA)}" in plan, plan

    assert upgrade(legacy) == MessageTsvUpgradeReport(
        rows_backfilled={ALPHA: 0, LONG_LEAF: 0})

    # Rows and partitions added after the upgrade are covered automatically.
    with legacy.begin() as conn:
        conn.execute(text("CREATE TABLE messages_p_gamma PARTITION OF messages FOR VALUES IN ('gamma')"))
        conn.execute(text("INSERT INTO messages (bot_id, id, role, content, timestamp) VALUES "
                          "('gamma', 'g1', 'user', 'new partition words', 1), "
                          "('alpha', 'a1', 'user', 'new row words', 1)"))
        assert conn.execute(text(STALE_ROWS)).scalar_one() == 0
    assert state(legacy).complete


def test_upgrade_rebuilds_an_invalid_leftover_and_attaches_a_valid_one(legacy):
    upgrade(legacy)
    with legacy.begin() as conn:
        conn.execute(text(f"DROP INDEX {INDEX}"))
        # A failed CREATE INDEX CONCURRENTLY leaves an INVALID index behind.
        conn.execute(text(f"CREATE INDEX mtsv_leftover ON {ALPHA} USING gin (content_tsv)"))
        conn.execute(text("UPDATE pg_index SET indisvalid = false "
                          "WHERE indexrelid = 'mtsv_leftover'::regclass"))
        conn.execute(text(f"CREATE INDEX mtsv_unattached ON {LONG_LEAF} USING gin (content_tsv)"))
    assert not state(legacy).complete

    report = upgrade(legacy)
    assert report.parent_index_created
    assert report.invalid_indexes_dropped == ["mtsv_leftover"]
    assert report.leaf_indexes_built == [leaf_index_name(ALPHA)]
    assert sorted(report.leaf_indexes_attached) == sorted([leaf_index_name(ALPHA), "mtsv_unattached"])
    assert state(legacy).complete
    with legacy.connect() as conn:
        assert conn.execute(text("SELECT to_regclass('mtsv_leftover')")).scalar_one() is None


def test_legacy_index_drop_requires_the_upgrade_and_is_idempotent(legacy):
    with pytest.raises(RuntimeError, match="stored tsvector incomplete"):
        drop_legacy_fts_index(legacy, lock_pause=0)
    assert state(legacy).legacy_index
    upgrade(legacy)
    assert drop_legacy_fts_index(legacy, lock_pause=0) is True
    assert not state(legacy).legacy_index
    assert drop_legacy_fts_index(legacy, lock_pause=0) is False
    ids, _plan = scoped_plan(legacy)
    assert ids == ["hit"]


def test_drop_finds_the_legacy_index_by_structure_not_name(legacy):
    upgrade(legacy)
    with legacy.begin() as conn:
        conn.execute(text(f"ALTER INDEX {LEGACY_INDEX} RENAME TO messages_old_fts"))
    assert state(legacy).legacy_indexes == ("messages_old_fts",)
    assert drop_legacy_fts_index(legacy, lock_pause=0) is True
    with legacy.connect() as conn:
        assert conn.execute(text("SELECT to_regclass('messages_old_fts')")).scalar_one() is None
    assert not state(legacy).legacy_index


def test_unrelated_index_with_legacy_name_is_not_dropped(legacy):
    upgrade(legacy)
    with legacy.begin() as conn:
        conn.execute(text(f"DROP INDEX {LEGACY_INDEX}"))
        conn.execute(text("CREATE TABLE auxiliary (content text)"))
        conn.execute(text(f"CREATE INDEX {LEGACY_INDEX} ON auxiliary (content)"))
    assert not state(legacy).legacy_index
    assert drop_legacy_fts_index(legacy, lock_pause=0) is False
    with legacy.connect() as conn:
        owner = conn.execute(text(
            "SELECT i.indrelid::regclass::text FROM pg_index i "
            "WHERE i.indexrelid = CAST(:name AS regclass)"
        ), {"name": LEGACY_INDEX}).scalar_one()
        assert owner == "auxiliary"


def test_invalid_attached_leaf_is_incomplete_and_rebuilt_in_place(legacy):
    upgrade(legacy)
    leaf_index = leaf_index_name(ALPHA)
    with legacy.begin() as conn:
        conn.execute(text(
            "UPDATE pg_index SET indisvalid = false "
            "WHERE indexrelid = CAST(:name AS regclass)"
        ), {"name": leaf_index})
    current = state(legacy)
    assert not current.complete
    assert current.invalid_leaf_indexes == (leaf_index,)
    with pytest.raises(RuntimeError, match="stored tsvector incomplete"):
        drop_legacy_fts_index(legacy, lock_pause=0)
    assert state(legacy).legacy_index

    report = upgrade(legacy)
    assert report.leaf_indexes_reindexed == [leaf_index]
    assert report.leaf_indexes_built == [] and report.leaf_indexes_attached == []
    current = state(legacy)
    assert current.complete, current.missing()
    with legacy.connect() as conn:
        # Same name, still the attached child of the parent index.
        assert conn.execute(text(
            "SELECT c.relname FROM pg_inherits i JOIN pg_class c ON c.oid = i.inhrelid "
            "WHERE i.inhparent = CAST(:parent AS regclass) ORDER BY 1"
        ), {"parent": INDEX}).scalars().all() == sorted([leaf_index, leaf_index_name(LONG_LEAF)])
    ids, plan = scoped_plan(legacy)
    assert ids == ["hit"]
    assert f"Bitmap Index Scan on {leaf_index}" in plan, plan
