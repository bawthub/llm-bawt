"""TASK-631 slice 2: revision 0002_ops_store against disposable PostgreSQL.

Covers parity between the runtime bootstrap and the Alembic path, seeded legacy
data transitions, revision-state handling, concurrent starters and rollback.
"""

from __future__ import annotations

import json
import threading

import pytest
from sqlalchemy import create_engine, inspect, text
from sqlmodel import Session

from tests.private_postgres import private_module_db, private_pg, private_url  # noqa: F401  (fixtures)

pytest.importorskip("alembic")

from llm_bawt.ops import host_access, quiesce_exit, release_store
from llm_bawt.ops import store as ops_store
from llm_bawt.ops.gate_plan import GatePlan
from llm_bawt.ops.models import OpsJob, OpsOperation
from llm_bawt.utils import alembic_state

pytestmark = [pytest.mark.integration]

HEAD = "0005_playground"

OWNED_TABLES = ("ops_operations", "ops_jobs", "ops_operation_revisions", "ops_bell_outbox",
                "ops_release_runs", "ops_release_events", "quiesce_run_sightings", "ops_host_access")


@pytest.fixture
def make_db(private_pg, private_module_db):  # noqa: F811
    """Return the module's private *_test database, reset to an empty public schema.

    The PG test boundary admits only the env-selected *_test database, so each
    call wipes and reuses it rather than creating a second one. Callers capture
    what they need from one phase before starting the next.
    """
    url = private_url(private_pg, private_module_db)
    engines: list = []

    def _make():
        # A fresh engine per phase: the per-engine schema guards cache completion,
        # so reusing one engine across a schema wipe would skip the bootstraps.
        alembic_state._reset_cache_for_tests()
        for old in engines:
            old.dispose()
        engine = create_engine(url, pool_pre_ping=True)
        engines.append(engine)
        with engine.begin() as conn:
            conn.execute(text("DROP SCHEMA public CASCADE"))
            conn.execute(text("CREATE SCHEMA public"))
        return engine

    yield _make
    alembic_state._reset_cache_for_tests()
    for old in engines:
        old.dispose()


def new_process(engine):
    """A fresh engine, as a restarted process would have: per-engine schema guards start empty."""
    return create_engine(engine.url, pool_pre_ping=True)


def runtime_bootstrap(engine) -> ops_store.OpsStore:
    store = ops_store.OpsStore(None, engine=engine)
    release_store.ReleaseStore(None, engine=engine)
    quiesce_exit.ensure_sighting_schema(engine)
    host_access.ensure_schema(engine)
    return store


def stamp(engine, revision: str) -> None:
    from alembic import command

    with engine.begin() as conn:
        command.stamp(alembic_state.alembic_config(conn), revision)


def owned_schema(engine) -> dict:
    """Columns, defaults, nullability, indexes, constraints and triggers of the owned tables."""
    with engine.connect() as conn:
        cols = conn.execute(text(
            "SELECT table_name, column_name, data_type, character_maximum_length, is_nullable, column_default "
            "FROM information_schema.columns WHERE table_schema='public' AND table_name = ANY(:t) "
            "ORDER BY 1, 2"), {"t": list(OWNED_TABLES)}).all()
        idx = conn.execute(text(
            "SELECT tablename, indexname, indexdef FROM pg_indexes WHERE schemaname='public' "
            "AND tablename = ANY(:t) ORDER BY 1, 2"), {"t": list(OWNED_TABLES)}).all()
        cons = conn.execute(text(
            "SELECT c.conrelid::regclass::text, c.conname, pg_get_constraintdef(c.oid) FROM pg_constraint c "
            "WHERE c.conrelid::regclass::text = ANY(:t) ORDER BY 1, 2"), {"t": list(OWNED_TABLES)}).all()
        trg = conn.execute(text(
            "SELECT tgrelid::regclass::text, tgname, pg_get_triggerdef(oid) FROM pg_trigger "
            "WHERE NOT tgisinternal AND tgrelid::regclass::text = ANY(:t) ORDER BY 1, 2"),
            {"t": list(OWNED_TABLES)}).all()
    return {"columns": [tuple(r) for r in cols], "indexes": [tuple(r) for r in idx],
            "constraints": [tuple(r) for r in cons], "triggers": [tuple(r) for r in trg]}


# --- parity -------------------------------------------------------------------

def test_alembic_path_matches_runtime_schema(make_db):
    runtime = make_db()
    runtime_bootstrap(runtime)
    expected = owned_schema(runtime)

    migrated = make_db()
    stamp(migrated, "0001_baseline")
    assert alembic_state.upgrade_on_startup(migrated) == HEAD
    assert set(OWNED_TABLES) <= set(inspect(migrated).get_table_names())

    assert owned_schema(migrated) == expected


# --- seeded legacy data -------------------------------------------------------

def _legacy_seed(engine) -> None:
    """Build the pre-TASK-1169/1484 shape and seed rows that the transitions touch."""
    runtime_bootstrap(engine)
    gate_args = json.dumps(GatePlan("llm-bawt", "a" * 40).to_dict())
    with Session(engine) as session:
        for slug, kind in (("legacy-docker", ops_store.EXECUTOR_DOCKER),
                           ("legacy-release", ops_store.EXECUTOR_RELEASE)):
            session.add(OpsOperation(id=slug, slug=slug, title=slug, executor_kind=kind, version=3,
                                     script="true"))
        session.commit()
        for job_id, args in (("job-gate", gate_args), ("job-plain", json.dumps({"x": 1}))):
            session.add(OpsJob(id=job_id, operation_slug="legacy-docker", operation_version=3,
                               operation_script_hash="h", args_json=args, display_args_json=args,
                               idempotency_key=f"seed-{job_id}"))
        session.commit()
    with engine.begin() as conn:
        # Roll the shape back to before the columns existed, keeping the rows.
        conn.execute(text("ALTER TABLE ops_jobs DROP COLUMN gate_plan_key"))
        conn.execute(text("ALTER TABLE ops_operations DROP COLUMN execution_mode"))
        conn.execute(text("DELETE FROM ops_operation_revisions"))


def _invariants(engine) -> dict:
    with engine.connect() as conn:
        modes = dict(conn.execute(text("SELECT slug, execution_mode FROM ops_operations ORDER BY slug")).all())
        keys = dict(conn.execute(text("SELECT id, gate_plan_key FROM ops_jobs ORDER BY id")).all())
        revs = conn.execute(text(
            "SELECT o.slug, count(r.id) FROM ops_operations o LEFT JOIN ops_operation_revisions r "
            "ON r.operation_id = o.id AND r.version = o.version GROUP BY o.slug ORDER BY o.slug")).all()
        default = conn.execute(text(
            "SELECT column_default FROM information_schema.columns "
            "WHERE table_name='ops_operations' AND column_name='execution_mode'")).scalar()
    return {"modes": modes, "keys": keys, "revisions": [tuple(r) for r in revs], "default": default}


def test_legacy_data_transitions_match_runtime_path(make_db):
    via_runtime = make_db()
    _legacy_seed(via_runtime)
    restarted = new_process(via_runtime)
    runtime_bootstrap(restarted)  # unstamped: runtime path performs the transitions
    expected = _invariants(restarted)
    restarted.dispose()

    via_alembic = make_db()
    _legacy_seed(via_alembic)
    stamp(via_alembic, "0001_baseline")
    restarted = new_process(via_alembic)
    alembic_state.upgrade_on_startup(restarted)
    runtime_bootstrap(restarted)  # stamped at 0002: DDL guarded, revision repair still runs
    got = _invariants(restarted)
    restarted.dispose()

    assert got == expected
    assert got["modes"]["legacy-docker"] != got["modes"]["legacy-release"]
    assert got["keys"]["job-gate"] is not None and got["keys"]["job-plain"] is None
    assert all(count == 1 for _, count in got["revisions"])


def test_existing_admin_choice_is_not_rewritten(make_db):
    engine = make_db()
    runtime_bootstrap(engine)
    with Session(engine) as session:
        session.add(OpsOperation(id="chosen", slug="chosen", title="c", executor_kind=ops_store.EXECUTOR_DOCKER,
                                 version=1, script="true", execution_mode="process"))
        session.commit()
    stamp(engine, "0001_baseline")
    restarted = new_process(engine)
    alembic_state.upgrade_on_startup(restarted)
    runtime_bootstrap(restarted)
    with restarted.connect() as conn:
        assert conn.execute(text("SELECT execution_mode FROM ops_operations WHERE slug='chosen'")).scalar() == "process"
    restarted.dispose()


# --- guard ---------------------------------------------------------------------

def test_guard_skips_owned_ddl_only_when_stamped(make_db, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(ops_store, "bootstrap_ops_store_schema",
                        lambda conn, _orig=ops_store.bootstrap_ops_store_schema: (calls.append("ddl"), _orig(conn)))
    unstamped = make_db()
    runtime_bootstrap(unstamped)
    assert calls == ["ddl"]

    calls.clear()
    stamped = make_db()
    stamp(stamped, "0001_baseline")
    alembic_state.upgrade_on_startup(stamped)
    calls.clear()
    restarted = new_process(stamped)
    alembic_state.upgrade_on_startup(restarted)
    runtime_bootstrap(restarted)
    restarted.dispose()
    assert calls == []


# --- revision state --------------------------------------------------------------

def test_unstamped_is_compatibility_mode(make_db):
    engine = make_db()
    assert alembic_state.upgrade_on_startup(engine) is None
    assert alembic_state.owns(engine, "0002_ops_store") is False
    assert "alembic_version" not in inspect(engine).get_table_names()


@pytest.mark.parametrize("rows", [[], ["nope_rev"], ["0001_baseline", "0002_ops_store"]])
def test_invalid_states_fail_closed(make_db, rows):
    engine = make_db()
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)"))
        for rev in rows:
            conn.execute(text("INSERT INTO alembic_version VALUES (:r)"), {"r": rev})
    with pytest.raises(alembic_state.AlembicStateError):
        alembic_state.upgrade_on_startup(engine)
    with pytest.raises(alembic_state.AlembicStateError):
        alembic_state.owns(engine, "0002_ops_store")


def test_ancestry_not_string_order():
    assert alembic_state.is_at_or_after("0002_ops_store", "0001_baseline")
    assert alembic_state.is_at_or_after("0002_ops_store", "0002_ops_store")
    assert not alembic_state.is_at_or_after("0001_baseline", "0002_ops_store")
    assert not alembic_state.is_at_or_after(None, "0001_baseline")
    with pytest.raises(alembic_state.AlembicStateError):
        alembic_state.is_at_or_after("0002_ops_store", "9999_missing")


# --- concurrency and failure ----------------------------------------------------

def test_two_concurrent_starters_upgrade_once(make_db, private_pg):  # noqa: F811
    first = make_db()
    stamp(first, "0001_baseline")
    second = create_engine(first.url, pool_pre_ping=True)
    upgrades: list[int] = []
    orig = ops_store.bootstrap_ops_store_schema
    barrier = threading.Barrier(2)

    def counting(conn):
        upgrades.append(1)
        return orig(conn)

    results: list[object] = []

    def start(engine):
        barrier.wait()
        results.append(alembic_state.upgrade_on_startup(engine))

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(ops_store, "bootstrap_ops_store_schema", counting)
        threads = [threading.Thread(target=start, args=(e,)) for e in (first, second)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(60)
    second.dispose()
    assert results == [HEAD, HEAD]
    assert len(upgrades) == 1


def test_failed_upgrade_rolls_back_and_releases_lock(make_db):
    engine = make_db()
    stamp(engine, "0001_baseline")

    def boom(conn):
        conn.execute(text("CREATE TABLE half_applied (id int)"))
        raise RuntimeError("simulated migration failure")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(ops_store, "bootstrap_ops_store_schema", boom)
        with pytest.raises(RuntimeError, match="simulated"):
            alembic_state.upgrade_on_startup(engine)

    tables = set(inspect(engine).get_table_names())
    assert "half_applied" not in tables
    with engine.connect() as conn:
        assert alembic_state.read_revision(conn) == "0001_baseline"
        got = conn.execute(text("SELECT pg_try_advisory_lock(hashtext('llm-bawt-runtime-schema'))")).scalar()
        assert got is True
        conn.execute(text("SELECT pg_advisory_unlock(hashtext('llm-bawt-runtime-schema'))"))
    assert alembic_state.upgrade_on_startup(engine) == HEAD
