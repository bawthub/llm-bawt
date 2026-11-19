"""TASK-631 slice 4: revision 0004_gates against disposable PostgreSQL."""

from __future__ import annotations

import pytest
from sqlalchemy import text

from tests.private_postgres import private_module_db, private_pg  # noqa: F401  (fixtures)
from tests.test_alembic_ops_store_pg import HEAD, make_db, new_process, stamp  # noqa: F401

pytest.importorskip("alembic")

from llm_bawt.service import turn_gates
from llm_bawt.utils import alembic_state

pytestmark = [pytest.mark.integration]

OWNED = ("system_gates", "gate_members", "gate_escalations", "ops_resume_notices")


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
    turn_gates.ensure_gate_schema(runtime)
    expected = owned_schema(runtime)
    assert {c[0] for c in expected["columns"]} == set(OWNED)

    migrated = make_db()
    stamp(migrated, "0001_baseline")
    assert alembic_state.upgrade_on_startup(migrated) == HEAD
    assert owned_schema(migrated) == expected


def test_upgrade_is_noop_on_runtime_bootstrapped_db(make_db):
    # The live path: runtime DDL already ran, then the DB is stamped at 0003.
    engine = make_db()
    turn_gates.ensure_gate_schema(engine)
    with engine.begin() as conn:
        gid = conn.execute(text(
            "INSERT INTO system_gates (owner_kind, owner_id, mode, scope, expires_at) "
            "VALUES ('job', 'j1', 'hold', '{}'::jsonb, now() + interval '1 hour') RETURNING id")).scalar_one()
        conn.execute(text(
            "INSERT INTO gate_members (gate_id, bot_id, turn_id, turn_status, disposition) "
            "VALUES (:g, 'b', 't1', 'running', 'running')"), {"g": gid})
    before = owned_schema(engine)
    stamp(engine, "0003_working_memory")
    assert alembic_state.upgrade_on_startup(engine) == HEAD
    assert owned_schema(engine) == before
    with engine.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM gate_members WHERE gate_id = :g"), {"g": gid}).scalar_one() == 1


def test_guard_skips_owned_ddl_only_when_stamped(make_db, monkeypatch):
    calls: list[str] = []
    real = turn_gates.bootstrap_gate_schema

    def counting(conn):
        calls.append("run")
        real(conn)

    monkeypatch.setattr(turn_gates, "bootstrap_gate_schema", counting)

    unstamped = make_db()
    turn_gates.ensure_gate_schema(unstamped)
    assert calls == ["run"]

    stamped = make_db()
    stamp(stamped, "0001_baseline")
    alembic_state.upgrade_on_startup(stamped)  # migration calls the module function too
    calls.clear()
    turn_gates.ensure_gate_schema(new_process(stamped))
    assert calls == []


def test_partially_upgraded_db_still_runs_runtime_ddl(make_db):
    # Stamped at 0003 but not yet upgraded: 0004 does not own the DDL yet.
    engine = make_db()
    stamp(engine, "0003_working_memory")
    assert alembic_state.owns(engine, turn_gates.GATE_SCHEMA_REVISION) is False
