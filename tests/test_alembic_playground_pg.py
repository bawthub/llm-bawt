"""TASK-631 slice 5: Playground ownership on disposable PostgreSQL."""

# Imported pytest fixtures are intentionally shadowed by injected parameters.
# ruff: noqa: F811

from __future__ import annotations

import pytest
from sqlalchemy import event, inspect, text

from tests.private_postgres import private_module_db, private_pg  # noqa: F401
from tests.test_alembic_ops_store_pg import HEAD, make_db, new_process, stamp  # noqa: F401

pytest.importorskip("alembic")

from llm_bawt.playground import store
from llm_bawt.utils import alembic_state

pytestmark = [pytest.mark.integration]


def schema(engine):
    with engine.connect() as conn:
        catalog = inspect(conn)
        return {
            "columns": [
                (c["name"], str(c["type"]), c["nullable"], c["default"])
                for c in catalog.get_columns(store.TABLE_NAME)
            ],
            "indexes": catalog.get_indexes(store.TABLE_NAME),
            "primary_key": catalog.get_pk_constraint(store.TABLE_NAME),
            "foreign_keys": catalog.get_foreign_keys(store.TABLE_NAME),
        }


def seed(engine):
    with engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO playground_artifacts "
            "(id,user_id,sender_bot_id,title,kind,mime,size_bytes,inline_content,created_at,opened_at) "
            "VALUES ('artifact-a','sample-user','sample-bot','Sample','text','text/plain',4,'body',now(),NULL),"
            "('artifact-b','sample-user','sample-bot','Opened','text','text/plain',4,'body',now(),now())"
        ))


def rows(engine):
    with engine.connect() as conn:
        return conn.execute(text("SELECT * FROM playground_artifacts ORDER BY id")).all()


def test_alembic_path_matches_runtime_schema(make_db):
    runtime = make_db()
    store.PlaygroundStore(runtime)
    expected = schema(runtime)  # make_db resets the same physical DB on its next call.
    migrated = make_db()
    stamp(migrated, "0004_gates")
    assert alembic_state.upgrade_on_startup(migrated) == HEAD
    assert schema(migrated) == expected


def test_upgrade_preserves_bootstrapped_schema_and_rows_without_ddl(make_db):
    from alembic import command

    engine = make_db()
    store.PlaygroundStore(engine)
    seed(engine)
    before_schema, before_rows = schema(engine), rows(engine)
    stamp(engine, "0004_gates")
    statements = []

    def capture(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement.lstrip().upper())

    event.listen(engine, "before_cursor_execute", capture)
    try:
        # This no-DDL proof covers 0005 only, not later unrelated table creation.
        with engine.begin() as conn:
            command.upgrade(alembic_state.alembic_config(conn), "0005_playground")
            assert alembic_state.read_revision(conn) == "0005_playground"
    finally:
        event.remove(engine, "before_cursor_execute", capture)
    assert not any(s.startswith(("CREATE ", "ALTER ", "DROP ")) for s in statements)
    assert schema(engine) == before_schema
    assert rows(engine) == before_rows


def test_legacy_timestamp_columns_added_without_changing_existing_state(make_db):
    engine = make_db()
    store.PlaygroundStore(engine)
    seed(engine)
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE playground_artifacts DROP COLUMN archived_at"))
        conn.execute(text("ALTER TABLE playground_artifacts DROP COLUMN trashed_at"))
        conn.execute(text("ALTER TABLE playground_artifacts DROP COLUMN revision"))
        conn.execute(text("ALTER TABLE playground_artifacts DROP COLUMN updated_at"))
    stamp(engine, "0004_gates")
    assert alembic_state.upgrade_on_startup(engine) == HEAD
    result = store.PlaygroundStore(engine).list("sample-user")
    states = {r["id"]: store.public_view(r)["status"] for r in result}
    assert states == {"artifact-a": "unopened", "artifact-b": "opened"}
    assert all(r["archived_at"] is None and r["trashed_at"] is None for r in result)
    assert {store.public_view(r)["revision"] for r in result} == {1}


def test_0007_adds_revision_columns_to_a_0006_table(make_db):
    engine = make_db()
    store.PlaygroundStore(engine)
    seed(engine)
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE playground_artifacts DROP COLUMN revision"))
        conn.execute(text("ALTER TABLE playground_artifacts DROP COLUMN updated_at"))
        conn.execute(text("DROP TABLE playground_retired_blobs"))
    stamp(engine, "0006_utility_usage")
    assert alembic_state.upgrade_on_startup(engine) == HEAD
    names = {c["name"] for c in inspect(engine).get_columns(store.TABLE_NAME)}
    assert {"revision", "updated_at"} <= names
    assert inspect(engine).has_table(store.RETIRED_TABLE)


def test_runtime_bootstrap_only_runs_before_ownership(make_db, monkeypatch):
    calls = []
    real = store.bootstrap_playground_schema

    def counting(conn):
        calls.append("run")
        real(conn)

    monkeypatch.setattr(store, "bootstrap_playground_schema", counting)
    unstamped = make_db()
    store.PlaygroundStore(unstamped)
    assert calls == ["run"]

    stamped = make_db()
    stamp(stamped, "0004_gates")
    alembic_state.upgrade_on_startup(stamped)
    calls.clear()
    store.PlaygroundStore(new_process(stamped))
    assert calls == []


def test_partial_upgrade_keeps_runtime_compatibility(make_db):
    engine = make_db()
    stamp(engine, "0004_gates")
    assert not alembic_state.owns(engine, store.PLAYGROUND_SCHEMA_REVISION)
    store.PlaygroundStore(engine)
    assert inspect(engine).has_table(store.TABLE_NAME)
