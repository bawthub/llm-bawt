"""TASK-631 slice 6: utility usage ownership on disposable PostgreSQL."""

# Imported pytest fixtures are intentionally shadowed by injected parameters.
# ruff: noqa: F811

from __future__ import annotations

import pytest
from sqlalchemy import event, inspect, text

from tests.private_postgres import private_module_db, private_pg  # noqa: F401
from tests.test_alembic_ops_store_pg import HEAD, make_db, stamp  # noqa: F401

pytest.importorskip("alembic")

from llm_bawt.playground.store import bootstrap_playground_schema
from llm_bawt.service.usage import utility_usage as usage
from llm_bawt.utils import alembic_state

pytestmark = [pytest.mark.integration]


def schema(engine):
    with engine.connect() as conn:
        catalog = inspect(conn)
        return {
            "columns": [
                (c["name"], str(c["type"]), c["nullable"], c["default"])
                for c in catalog.get_columns("utility_usage")
            ],
            "indexes": catalog.get_indexes("utility_usage"),
            "primary_key": catalog.get_pk_constraint("utility_usage"),
        }


def bootstrap(engine):
    with engine.begin() as conn:
        usage.ensure_utility_usage_schema(conn)


def seed(engine):
    row = usage.build_row(
        caller="sample-caller", bot_id="sample-bot", model="sample-model",
        box={"usage": dict.fromkeys(usage.TOKEN_KEYS, 7)}, estimate=None,
        elapsed_ms=12, outcome="ok",
    )
    usage.record(row, engine=engine)
    with engine.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM utility_usage")).scalar_one() == 1


def data_and_sequence(engine):
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT * FROM utility_usage ORDER BY id")).all()
        sequence = conn.execute(text("SELECT last_value, is_called FROM utility_usage_id_seq")).one()
        return rows, sequence


def test_alembic_path_matches_runtime_schema(make_db):
    runtime = make_db()
    bootstrap(runtime)
    expected = schema(runtime)
    assert len(expected["columns"]) == 12
    migrated = make_db()
    stamp(migrated, "0005_playground")
    assert alembic_state.upgrade_on_startup(migrated) == HEAD
    assert schema(migrated) == expected


def test_complete_catalog_upgrade_preserves_rows_sequence_and_issues_no_ddl(make_db):
    engine = make_db()
    bootstrap(engine)
    # TASK-2625: a complete catalog also has the current Playground schema,
    # so 0007 (revision columns, retired-blob table) must issue no DDL either.
    with engine.begin() as conn:
        bootstrap_playground_schema(conn)
    seed(engine)
    before_schema, before_data = schema(engine), data_and_sequence(engine)
    stamp(engine, "0005_playground")
    statements = []

    def capture(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement.lstrip().upper())

    event.listen(engine, "before_cursor_execute", capture)
    try:
        assert alembic_state.upgrade_on_startup(engine) == HEAD
    finally:
        event.remove(engine, "before_cursor_execute", capture)
    assert not any(s.startswith(("CREATE ", "ALTER ", "DROP ")) for s in statements)
    assert schema(engine) == before_schema
    assert data_and_sequence(engine) == before_data
    # Subsequent writes must use the preserved sequence.
    usage.record(usage.build_row(
        caller="next", bot_id=None, model=None, box={}, estimate=None,
        elapsed_ms=0, outcome="ok",
    ), engine=engine)
    with engine.connect() as conn:
        assert conn.execute(text("SELECT id FROM utility_usage ORDER BY id")).scalars().all() == [1, 2]


def test_missing_index_is_repaired_without_changing_data(make_db):
    engine = make_db()
    bootstrap(engine)
    seed(engine)
    before = data_and_sequence(engine)
    with engine.begin() as conn:
        conn.execute(text("DROP INDEX ix_utility_usage_created"))
    stamp(engine, "0005_playground")
    assert alembic_state.upgrade_on_startup(engine) == HEAD
    assert data_and_sequence(engine) == before
    assert schema(engine)["indexes"][0]["name"] == "ix_utility_usage_created"


def test_runtime_guard_uses_same_connection_and_skips_owned_bootstrap(make_db, monkeypatch):
    engine = make_db()
    stamp(engine, "0005_playground")
    alembic_state.upgrade_on_startup(engine)

    def forbidden(*args, **kwargs):
        raise AssertionError("owned runtime must neither bootstrap nor open another connection")

    monkeypatch.setattr(usage, "bootstrap_utility_usage_schema", forbidden)
    with engine.begin() as conn:
        monkeypatch.setattr(engine, "connect", forbidden)
        usage.ensure_utility_usage_schema(conn)


@pytest.mark.parametrize("revision", [None, "0005_playground"])
def test_unowned_runtime_keeps_compatibility(make_db, revision):
    engine = make_db()
    if revision:
        stamp(engine, revision)
    bootstrap(engine)
    assert len(schema(engine)["columns"]) == 12
    seed(engine)


@pytest.mark.parametrize("revisions", [[], ["unknown_revision"], ["0004_gates", "0005_playground"]])
def test_invalid_revision_fails_runtime_before_schema_ddl(make_db, revisions):
    engine = make_db()
    stamp(engine, "0005_playground")
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM alembic_version"))
        for revision in revisions:
            conn.execute(text("INSERT INTO alembic_version VALUES (:r)"), {"r": revision})
    with pytest.raises(alembic_state.AlembicStateError):
        bootstrap(engine)
    with engine.connect() as conn:
        assert not inspect(conn).has_table("utility_usage")


def test_failed_upgrade_rolls_back_table_and_revision_then_retries(make_db, monkeypatch):
    engine = make_db()
    stamp(engine, "0005_playground")
    real = usage.bootstrap_utility_usage_schema

    def fail_after_creation(conn):
        real(conn)
        raise RuntimeError("injected usage bootstrap failure")

    monkeypatch.setattr(usage, "bootstrap_utility_usage_schema", fail_after_creation)
    with pytest.raises(RuntimeError, match="injected usage bootstrap failure"):
        alembic_state.upgrade_on_startup(engine)
    with engine.connect() as conn:
        assert alembic_state.read_revision(conn) == "0005_playground"
        assert not inspect(conn).has_table("utility_usage")
        assert conn.execute(text("SELECT to_regclass('utility_usage_id_seq')")).scalar_one() is None
    monkeypatch.setattr(usage, "bootstrap_utility_usage_schema", real)
    assert alembic_state.upgrade_on_startup(engine) == HEAD


def test_downgrade_preserves_data_and_runtime_compatibility(make_db):
    from alembic import command

    engine = make_db()
    stamp(engine, "0005_playground")
    alembic_state.upgrade_on_startup(engine)
    seed(engine)
    before_schema, before_data = schema(engine), data_and_sequence(engine)
    with engine.begin() as conn:
        command.downgrade(alembic_state.alembic_config(conn), "0005_playground")
        assert alembic_state.read_revision(conn) == "0005_playground"
        usage.ensure_utility_usage_schema(conn)
    assert schema(engine) == before_schema
    assert data_and_sequence(engine) == before_data


def test_offline_upgrade_is_rejected_without_connecting():
    from alembic import command

    with pytest.raises(RuntimeError, match="0006_utility_usage.*online"):
        command.upgrade(alembic_state.alembic_config(), "0005_playground:0006_utility_usage", sql=True)
