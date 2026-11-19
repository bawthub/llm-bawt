"""TASK-631 slice 3: revision 0003_working_memory against disposable PostgreSQL."""

from __future__ import annotations

import pytest
from sqlalchemy import text

from tests.private_postgres import private_module_db, private_pg  # noqa: F401  (fixtures)
from tests.test_alembic_ops_store_pg import HEAD, make_db, new_process, stamp  # noqa: F401

pytest.importorskip("alembic")

from llm_bawt.utils import alembic_state
from llm_bawt.working_memory import store as wm_store

pytestmark = [pytest.mark.integration]

OWNED = ("working_memory_namespaces", "working_memory_items", "working_memory_task_suppressions",
         "working_memory_task_purge_audit", "working_memory_changes")


def owned_schema(engine) -> dict:
    with engine.connect() as conn:
        cols = conn.execute(text(
            "SELECT table_name, column_name, data_type, character_maximum_length, is_nullable, column_default "
            "FROM information_schema.columns WHERE table_schema='public' AND table_name = ANY(:t) "
            "ORDER BY 1, 2"), {"t": list(OWNED)}).all()
        idx = conn.execute(text(
            "SELECT tablename, indexname, indexdef FROM pg_indexes WHERE schemaname='public' "
            "AND tablename = ANY(:t) ORDER BY 1, 2"), {"t": list(OWNED)}).all()
        cons = conn.execute(text(
            "SELECT c.conrelid::regclass::text, c.conname, pg_get_constraintdef(c.oid) FROM pg_constraint c "
            "WHERE c.conrelid::regclass::text = ANY(:t) ORDER BY 1, 2"), {"t": list(OWNED)}).all()
    return {"columns": [tuple(r) for r in cols], "indexes": [tuple(r) for r in idx],
            "constraints": [tuple(r) for r in cons]}


def test_alembic_path_matches_runtime_schema(make_db):
    runtime = make_db()
    wm_store.WorkingMemoryStore(runtime).ensure_schema()
    expected = owned_schema(runtime)
    assert {c[0] for c in expected["columns"]} == set(OWNED)

    migrated = make_db()
    stamp(migrated, "0001_baseline")
    assert alembic_state.upgrade_on_startup(migrated) == HEAD
    assert owned_schema(migrated) == expected


def test_upgrade_is_noop_on_runtime_bootstrapped_db(make_db):
    # The live path: runtime DDL already ran, then the DB is stamped at 0002.
    engine = make_db()
    wm_store.WorkingMemoryStore(engine).ensure_schema()
    with engine.begin() as conn:
        conn.execute(text("INSERT INTO working_memory_namespaces (bot_id, user_id, version) VALUES ('b', 'u', 7)"))
    before = owned_schema(engine)
    stamp(engine, "0002_ops_store")
    assert alembic_state.upgrade_on_startup(engine) == HEAD
    assert owned_schema(engine) == before
    with engine.connect() as conn:
        assert conn.execute(text("SELECT version FROM working_memory_namespaces")).scalar_one() == 7


def test_guard_skips_owned_ddl_only_when_stamped(make_db, monkeypatch):
    calls: list[str] = []
    real = wm_store.bootstrap_working_memory_schema

    def counting(conn):
        calls.append("run")
        real(conn)

    monkeypatch.setattr(wm_store, "bootstrap_working_memory_schema", counting)

    unstamped = make_db()
    wm_store.WorkingMemoryStore(unstamped).ensure_schema()
    assert calls == ["run"]

    stamped = make_db()
    stamp(stamped, "0001_baseline")
    alembic_state.upgrade_on_startup(stamped)  # migration calls the module function too
    calls.clear()
    wm_store.WorkingMemoryStore(new_process(stamped)).ensure_schema()
    assert calls == []


def test_partially_upgraded_db_still_runs_runtime_ddl(make_db):
    # Stamped at 0002 but not yet upgraded: 0003 does not own the DDL yet.
    engine = make_db()
    stamp(engine, "0002_ops_store")
    assert alembic_state.owns(engine, wm_store.WORKING_MEMORY_SCHEMA_REVISION) is False
