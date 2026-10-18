"""Bounds-based cleanup proofs against an isolated, networkless PostgreSQL DB."""

from __future__ import annotations

import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest
from sqlalchemy import create_engine, event, text
from sqlalchemy.exc import OperationalError

from llm_bawt import orphan_partitions
from llm_bawt.orphan_partitions import OrphanPartitionPlanner, PartitionManifestExecutor, PartitionExecutionHalted
from tests.private_postgres import private_url

pytestmark = pytest.mark.integration


@pytest.fixture
def orphan_db(private_pg, monkeypatch):
    name = f"orphan_plan_{uuid.uuid4().hex[:12]}_test"
    admin = create_engine(private_url(private_pg, "postgres"), isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        conn.exec_driver_sql(f'CREATE DATABASE "{name}"')
    monkeypatch.setenv("LLM_BAWT_POSTGRES_DATABASE", name)
    engine = create_engine(private_url(private_pg, name))
    try:
        with engine.begin() as conn:
            conn.exec_driver_sql("CREATE TABLE bot_profiles (slug text NOT NULL)")
            for parent in ("messages", "memories", "forgotten_messages"):
                conn.exec_driver_sql(
                    f"CREATE TABLE {parent} (bot_id text NOT NULL, body text) PARTITION BY LIST (bot_id)"
                )
            conn.exec_driver_sql("INSERT INTO bot_profiles VALUES ('real')")
            for bot in ("real", "mira", "spark", "system", "testcontext", "empty"):
                conn.exec_driver_sql(
                    f"CREATE TABLE messages_p_{bot} PARTITION OF messages FOR VALUES IN ('{bot}')"
                )
            for bot in ("real", "mira", "spark", "system", "testcontext"):
                conn.execute(text("INSERT INTO messages (bot_id, body) VALUES (:bot, 'proof')"),
                             {"bot": bot})
        yield engine
    finally:
        engine.dispose()
        with admin.connect() as conn:
            conn.exec_driver_sql(f'DROP DATABASE "{name}" WITH (FORCE)')
        admin.dispose()


def test_planner_excludes_protected_before_counting_and_plans_all_unregistered(orphan_db):
    counted = []

    def observe(conn, cursor, sql, params, context, many):
        if "SELECT count(*) FROM ONLY" in sql:
            counted.append(sql)

    event.listen(orphan_db, "before_cursor_execute", observe)
    try:
        plan = OrphanPartitionPlanner(orphan_db).plan()
        again = OrphanPartitionPlanner(orphan_db).plan()
    finally:
        event.remove(orphan_db, "before_cursor_execute", observe)
    assert plan.content_digest == again.content_digest
    assert plan.manifest()["sha256"] == plan.content_digest
    assert plan.manifest()["content"]["children"] == again.manifest()["content"]["children"]
    children = {c.storage_id: c for c in plan.children}
    assert children["mira"].row_count is None
    assert children["mira"].disposition == "held"
    assert children["real"].disposition == "held"
    for bot in ("spark", "system", "testcontext"):
        assert (children[bot].row_count, children[bot].action) == (1, "truncate-then-drop-RESTRICT")
    assert (children["empty"].row_count, children["empty"].action) == (0, "drop-empty-RESTRICT")
    assert not any("mira" in sql for sql in counted)
    assert plan.manifest()["counts"] == {"proposed": 4, "held": 2, "nonempty": 3}


def test_digest_mismatch_or_content_drift_never_mutates(orphan_db, tmp_path):
    plan = OrphanPartitionPlanner(orphan_db).plan()
    executor = PartitionManifestExecutor(orphan_db)
    journal = tmp_path / "mismatch.jsonl"
    with pytest.raises(ValueError, match="digest mismatch"):
        executor.execute(plan, "0" * 64, journal_path=journal)
    assert not journal.exists()
    with orphan_db.begin() as conn:
        conn.exec_driver_sql("INSERT INTO messages VALUES ('empty', 'new row')")
    with pytest.raises(ValueError, match="digest mismatch"):
        executor.execute(plan, plan.content_digest, journal_path=journal)
    assert not journal.exists()
    with orphan_db.connect() as conn:
        assert conn.exec_driver_sql("SELECT count(*) FROM messages").scalar_one() == 6


def test_nonempty_truncate_drop_restrict_journal_and_protected_survive(orphan_db, tmp_path):
    statements = []

    def observe(conn, cursor, sql, params, context, many):
        statements.append(sql)

    event.listen(orphan_db, "before_cursor_execute", observe)
    try:
        plan = OrphanPartitionPlanner(orphan_db).plan()
        outcomes = PartitionManifestExecutor(orphan_db).execute(
            plan, plan.content_digest, journal_path=tmp_path / "outcomes.jsonl")
    finally:
        event.remove(orphan_db, "before_cursor_execute", observe)
    assert len(outcomes) == 4
    assert all(outcome["status"] == "committed" for outcome in outcomes)
    assert (tmp_path / "outcomes.jsonl").read_text().count("\n") == 8
    locks = [sql for sql in statements if sql.startswith("LOCK TABLE")]
    assert locks[:3] == [
        'LOCK TABLE "public"."bot_profiles" IN SHARE MODE NOWAIT',
        'LOCK TABLE ONLY "public"."messages" IN ACCESS EXCLUSIVE MODE NOWAIT',
        'LOCK TABLE "public"."messages_p_empty" IN ACCESS EXCLUSIVE MODE NOWAIT',
    ]
    assert sum("TRUNCATE TABLE ONLY" in sql for sql in statements) == 3
    assert sum("DROP TABLE" in sql and "RESTRICT" in sql for sql in statements) == 4
    assert not any("CASCADE" in sql for sql in statements)
    with orphan_db.connect() as conn:
        assert conn.exec_driver_sql("SELECT bot_id FROM messages ORDER BY bot_id").scalars().all() == ["mira", "real"]
    remaining = OrphanPartitionPlanner(orphan_db).plan()
    assert remaining.manifest()["counts"] == {"proposed": 0, "held": 2, "nonempty": 0}


def test_inconsistent_bound_name_rejected_without_counting_protected(orphan_db):
    with orphan_db.begin() as conn:
        conn.exec_driver_sql("ALTER TABLE messages DETACH PARTITION messages_p_mira")
        conn.exec_driver_sql("DROP TABLE messages_p_mira")
        conn.exec_driver_sql("CREATE TABLE messages_p_mira_other PARTITION OF messages FOR VALUES IN ('mira')")
    seen = []
    event.listen(orphan_db, "before_cursor_execute", lambda conn, cursor, sql, *args: seen.append(sql))
    with pytest.raises(ValueError, match="name/bound"):
        OrphanPartitionPlanner(orphan_db).plan()
    assert not any("SELECT count(*) FROM ONLY" in sql and "mira" in sql for sql in seen)


def test_cleanup_entrypoint_requires_exact_digest_before_any_mutation(orphan_db, monkeypatch, tmp_path):
    from llm_bawt import runtime_settings
    from llm_bawt.utils import db

    monkeypatch.setattr(runtime_settings, "has_database_credentials", lambda config: True)
    monkeypatch.setattr(db, "get_shared_engine", lambda config: orphan_db)
    monkeypatch.setenv("LLM_BAWT_LOG_DIR", str(tmp_path / ".logs"))
    preview = runtime_settings.cleanup_orphaned_bot_data(object(), dry_run=True)
    assert preview["manifest"]["counts"]["nonempty"] == 3
    with pytest.raises(ValueError, match="digest mismatch"):
        runtime_settings.cleanup_orphaned_bot_data(
            object(), dry_run=False, manifest_digest="0" * 64)
    assert not (tmp_path / ".logs").exists()
    executed = runtime_settings.cleanup_orphaned_bot_data(
        object(), dry_run=False, manifest_digest=preview["manifest"]["sha256"])
    assert len(executed["outcomes"]) == 4
    assert (tmp_path / executed["journal"]).is_file()
    with orphan_db.connect() as conn:
        assert conn.exec_driver_sql("SELECT count(*) FROM bot_profiles").scalar_one() == 1


def test_outgoing_parent_fk_does_not_require_cascade(orphan_db, tmp_path):
    with orphan_db.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE sessions (id text PRIMARY KEY)")
        conn.exec_driver_sql("ALTER TABLE messages ADD COLUMN session_id text")
        conn.exec_driver_sql("ALTER TABLE messages ADD CONSTRAINT fk_messages_session "
                             "FOREIGN KEY (session_id) REFERENCES sessions(id)")
    plan = OrphanPartitionPlanner(orphan_db).plan()
    outcomes = PartitionManifestExecutor(orphan_db).execute(
        plan, plan.content_digest, journal_path=tmp_path / "fk-proof.jsonl")
    assert len(outcomes) == 4
    with orphan_db.connect() as conn:
        assert conn.exec_driver_sql("SELECT count(*) FROM messages").scalar_one() == 2
        assert conn.exec_driver_sql("SELECT count(*) FROM pg_constraint WHERE conname='fk_messages_session'").scalar_one() >= 1


def test_detached_or_partition_name_collision_fails_closed(orphan_db):
    with orphan_db.begin() as conn:
        conn.exec_driver_sql("ALTER TABLE messages DETACH PARTITION messages_p_empty")
    with pytest.raises(ValueError, match="Detached"):
        OrphanPartitionPlanner(orphan_db).plan()
    with orphan_db.begin() as conn:
        conn.exec_driver_sql("DROP TABLE messages_p_empty")
        conn.execute(text("INSERT INTO bot_profiles VALUES (:slug), (:other)"), {
            "slug": "a" * 60 + "b", "other": "a" * 60 + "c",
        })
    with pytest.raises(ValueError, match="collision"):
        OrphanPartitionPlanner(orphan_db).plan()


def test_lock_contention_halts_and_journals_without_mutation(orphan_db, tmp_path):
    plan = OrphanPartitionPlanner(orphan_db).plan()
    journal = tmp_path / "contention.jsonl"
    with orphan_db.begin() as blocker:
        blocker.exec_driver_sql("LOCK TABLE ONLY messages IN ACCESS EXCLUSIVE MODE")
        with pytest.raises(RuntimeError, match="halted"):
            PartitionManifestExecutor(orphan_db).execute(
                plan, plan.content_digest, journal_path=journal)
        records = [json.loads(line) for line in journal.read_text().splitlines()]
        assert [record["status"] for record in records] == ["pending", "halted"]
    with orphan_db.connect() as conn:
        assert conn.exec_driver_sql("SELECT count(*) FROM messages").scalar_one() == 5


def test_registration_mid_run_preserves_newly_registered_partition(orphan_db, tmp_path, monkeypatch):
    """Commit an actual registration after child one commits, before child two begins."""
    plan = OrphanPartitionPlanner(orphan_db).plan()
    journal = tmp_path / "registered.jsonl"
    original_fsync = orphan_partitions.os.fsync
    registered = False

    def insert_after_first_commit(fd):
        nonlocal registered
        original_fsync(fd)
        if registered or not journal.exists():
            return
        lines = journal.read_text().splitlines()
        if lines and json.loads(lines[-1])["status"] == "committed":
            registered = True
            with orphan_db.begin() as conn:
                conn.exec_driver_sql("INSERT INTO bot_profiles VALUES ('spark')")

    monkeypatch.setattr(orphan_partitions.os, "fsync", insert_after_first_commit)
    with pytest.raises(PartitionExecutionHalted) as halted:
        PartitionManifestExecutor(orphan_db).execute(
            plan, plan.content_digest, journal_path=journal)
    receipt = halted.value.receipt()
    assert registered
    assert receipt["status"] == "partial"
    assert receipt["committed_count"] == 1
    assert receipt["failed_child"].endswith('"messages_p_spark"')
    assert receipt["journal"] == str(journal)
    with orphan_db.connect() as conn:
        assert conn.exec_driver_sql("SELECT count(*) FROM messages_p_spark").scalar_one() == 1
        assert conn.exec_driver_sql("SELECT count(*) FROM bot_profiles WHERE slug='spark'").scalar_one() == 1


def test_registry_lock_fences_concurrent_writer_after_final_check(orphan_db, tmp_path):
    """At the old SELECT→DROP race boundary, a concurrent upsert cannot commit."""
    plan = OrphanPartitionPlanner(orphan_db).plan()
    fence_seen = False
    attempted = False

    def observe(conn, cursor, sql, params, context, many):
        nonlocal fence_seen, attempted
        if sql.startswith('LOCK TABLE "public"."bot_profiles" IN SHARE MODE'):
            fence_seen = True
        if (fence_seen and not attempted and
                'SELECT count(*) FROM ONLY "public"."messages_p_spark"' in sql):
            attempted = True

            def register():
                with orphan_db.begin() as writer:
                    writer.exec_driver_sql("SET LOCAL lock_timeout = '200ms'")
                    writer.exec_driver_sql("INSERT INTO bot_profiles VALUES ('spark')")

            with ThreadPoolExecutor(max_workers=1) as pool:
                with pytest.raises(OperationalError) as blocked:
                    pool.submit(register).result(timeout=3)
                assert blocked.value.orig.pgcode == "55P03"

    event.listen(orphan_db, "before_cursor_execute", observe)
    try:
        outcomes = PartitionManifestExecutor(orphan_db).execute(
            plan, plan.content_digest, journal_path=tmp_path / "fence.jsonl")
    finally:
        event.remove(orphan_db, "before_cursor_execute", observe)
    assert attempted and len(outcomes) == 4
    with orphan_db.connect() as conn:
        assert conn.exec_driver_sql("SELECT count(*) FROM bot_profiles WHERE slug='spark'").scalar_one() == 0


def test_executor_does_not_use_manifest_identifiers(orphan_db, tmp_path):
    plan = OrphanPartitionPlanner(orphan_db).plan()
    malicious = replace(plan.children[-1], child='"public"."messages_p_real"; DROP TABLE bot_profiles;--')
    forged = replace(plan, children=(*plan.children[:-1], malicious))
    with pytest.raises(ValueError, match="digest mismatch"):
        PartitionManifestExecutor(orphan_db).execute(
            forged, forged.content_digest, journal_path=tmp_path / "forged.jsonl")
    with orphan_db.connect() as conn:
        assert conn.exec_driver_sql("SELECT count(*) FROM bot_profiles").scalar_one() == 1
