"""TASK-1802: schema_init loads an exported seed A into a fresh preview DB.

Runs on the acceptance fixture's disposable databases, never operational
PostgreSQL: one plays production (bootstrapped and planted), a second fresh
one plays the preview database.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import create_engine, text

from llm_bawt.preview.probes import seed_audit_queries
from llm_bawt.preview.schema_init import META_TABLE, SeedInitError, _require_empty, load_seed
from llm_bawt.preview.seed_allowlist import COPY, export_query
from llm_bawt.preview.seed_export import write_manifest
from llm_bawt.schema_bootstrap import bootstrap_full_store
from tests.full_store_fixture import full_store_db  # noqa: F401  (pytest fixture)
from tests.private_postgres import private_url
from tests.test_preview_seed_allowlist_pg import _plant

pytestmark = pytest.mark.integration


def _export(engine, out):
    raw = engine.raw_connection()
    try:
        cur = raw.cursor()
        for table in COPY:
            with (out / f"{table}.csv").open("w", newline="") as handle:
                cur.copy_expert(f"COPY ({export_query(table)}) TO STDOUT WITH (FORMAT csv, HEADER true)", handle)
    finally:
        raw.close()
    return write_manifest(out)


@pytest.fixture
def preview_db(private_pg):
    name = f"preview_init_{uuid.uuid4().hex[:12]}_test"
    admin = create_engine(private_url(private_pg, "postgres"), isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        conn.execute(text(f'CREATE DATABASE "{name}"'))
    engine = create_engine(private_url(private_pg, name))
    try:
        yield name, engine
    finally:
        engine.dispose()
        with admin.connect() as conn:
            conn.execute(text(f'DROP DATABASE "{name}" WITH (FORCE)'))
        admin.dispose()


def test_seed_round_trip_then_refuses_reseed(full_store_db, preview_db, tmp_path, monkeypatch):  # noqa: F811
    from llm_bawt.utils import db

    config, prod = full_store_db
    bootstrap_full_store(config, prod)
    with prod.begin() as conn:
        conn.exec_driver_sql("ALTER TABLE public.bot_profiles DISABLE TRIGGER bot_profiles_model_endpoint_check")
    _plant(prod)
    with prod.begin() as conn:
        # Production bots always carry an endpoint; the trigger enforces it on load.
        model_id = conn.execute(text(
            "INSERT INTO models (key, vendor, display_name, default_context_window)"
            " VALUES ('seed-model', 'test', 'seed-model', 1000) RETURNING id")).scalar_one()
        endpoint_id = conn.execute(text(
            "INSERT INTO model_endpoints (model_id, access_path_id)"
            " SELECT :m, id FROM access_paths WHERE protocol = 'chat-completions' ORDER BY key LIMIT 1 RETURNING id"), {"m": model_id}).scalar_one()
        conn.execute(text("UPDATE bot_profiles SET endpoint_id = :e"), {"e": endpoint_id})
    with prod.begin() as conn:
        conn.exec_driver_sql("ALTER TABLE public.bot_profiles ENABLE TRIGGER bot_profiles_model_endpoint_check")
    manifest = _export(prod, tmp_path)

    name, engine = preview_db
    monkeypatch.setattr(db, "build_postgres_url", lambda _config: str(engine.url.render_as_string(hide_password=False)))
    monkeypatch.setattr(config, "POSTGRES_DATABASE", name)
    monkeypatch.setenv("LLM_BAWT_POSTGRES_DATABASE", name)
    db.reset_shared_engines()
    with pytest.raises(SeedInitError, match="expected"):
        load_seed(config, engine, tmp_path, "some_other_db")

    loaded = load_seed(config, engine, tmp_path, name)
    assert loaded == {table: entry["rows"] for table, entry in manifest["tables"].items()}
    with engine.connect() as conn:
        slugs = {row[0] for row in conn.execute(text("SELECT slug FROM bot_profiles"))}
        # _plant's public bots; seed_secret (tag) and seed_hidden (column) are private.
        assert slugs == {"seed_public", "seed_jsonnull"}
        assert conn.execute(text(f"SELECT id FROM {META_TABLE}")).scalar_one() == 1
        for table in ("messages", "memories", "sessions"):
            assert conn.execute(text(f"SELECT count(*) FROM {table}")).scalar_one() == 0
        failures = [label for label, sql in seed_audit_queries() if conn.execute(text(sql)).scalar_one() != 0]
        assert failures == []
    # A row beyond the bootstrap baseline (a leaked identity) fails the audit.
    with engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO entity_profiles (entity_type, entity_id, display_name, created_at, updated_at)"
            " VALUES ('USER', 'leaked', 'Leaked', now(), now())"))
        failures = [label for label, sql in seed_audit_queries() if conn.execute(text(sql)).scalar_one() != 0]
        assert failures == ["entity_profiles bootstrap rows only"]
        conn.execute(text("DELETE FROM entity_profiles WHERE entity_id = 'leaked'"))
        assert [label for label, sql in seed_audit_queries() if conn.execute(text(sql)).scalar_one() != 0] == []
        # Same row count, different contents: a replaced bootstrap row fails too.
        assert conn.execute(text("UPDATE entity_profiles SET display_name = 'Replaced'")).rowcount >= 1
        failures = [label for label, sql in seed_audit_queries() if conn.execute(text(sql)).scalar_one() != 0]
        assert failures == ["entity_profiles bootstrap rows only"]
    with pytest.raises(SeedInitError, match="not empty"):
        load_seed(config, engine, tmp_path, name)


def test_non_empty_target_is_refused_before_any_write(preview_db, monkeypatch):
    name, engine = preview_db
    monkeypatch.setenv("LLM_BAWT_POSTGRES_DATABASE", name)
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE public.leftover (id int)"))
    with pytest.raises(SeedInitError, match="not empty"):
        _require_empty(engine, name)
