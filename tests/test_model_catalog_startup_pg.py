"""Model catalog startup against a private PostgreSQL database only."""

import pytest
from sqlalchemy import event, text

from llm_bawt.memory import model_catalog_migration as catalog
from llm_bawt.runtime_settings import BotProfileStore

pytestmark = pytest.mark.integration
_STAMP = "SELECT obj_description('public.bot_profiles_check_model_endpoint()'::regprocedure, 'pg_proc')"


@pytest.fixture
def engine(private_app_pg):
    engine = BotProfileStore(private_app_pg).engine
    assert engine is not None
    catalog.migrate_model_catalog(engine)
    return engine


@pytest.mark.parametrize("unstamped", [False, True])
def test_ready_catalog_needs_no_ddl_or_advisory_lock(engine, unstamped):
    if unstamped:
        with engine.begin() as conn:
            conn.execute(text("COMMENT ON FUNCTION public.bot_profiles_check_model_endpoint() IS NULL"))

    def read_only(conn, *_):
        conn.exec_driver_sql("SET TRANSACTION READ ONLY")
        conn.exec_driver_sql("SET LOCAL lock_timeout = '5s'")

    with engine.begin() as holder:
        holder.execute(text(catalog._CATALOG_LOCK))
        holder.execute(text("LOCK TABLE access_paths, models, model_endpoints, bot_profiles IN ACCESS SHARE MODE"))
        event.listen(engine, "begin", read_only)
        try:
            result = catalog.prepare_model_catalog_on_startup(engine)
        finally:
            event.remove(engine, "begin", read_only)
    assert result["startup_path"] == "read-only" and result["repaired_bots"] == []
    with engine.connect() as conn:
        assert conn.execute(text(_STAMP)).scalar_one() == (None if unstamped else catalog._CATALOG_VERSION)


def test_old_version_runs_locked_repair_then_uses_read_only_path(engine, monkeypatch):
    version = catalog._CATALOG_VERSION + " test-next-version"
    monkeypatch.setattr(catalog, "_CATALOG_VERSION", version)
    assert catalog.prepare_model_catalog_on_startup(engine)["startup_path"] == "repair"
    with engine.connect() as conn:
        assert conn.execute(text(_STAMP)).scalar_one() == version
    assert catalog.prepare_model_catalog_on_startup(engine)["startup_path"] == "read-only"


def test_missing_seed_and_constraint_take_repair_path(engine):
    key = "local-llamacpp"
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM access_paths WHERE key=:key"), {"key": key})
        conn.execute(text("ALTER TABLE models DROP CONSTRAINT models_pricing_threshold_nonnegative"))
    assert catalog.prepare_model_catalog_on_startup(engine)["startup_path"] == "repair"
    with engine.connect() as conn:
        assert conn.execute(text("SELECT 1 FROM access_paths WHERE key=:key"), {"key": key}).scalar_one() == 1
    assert catalog.prepare_model_catalog_on_startup(engine)["startup_path"] == "read-only"


def test_rechecks_readiness_after_taking_migration_lock(engine, monkeypatch):
    ready = catalog._catalog_ready
    ready_results = iter((False, True))
    monkeypatch.setattr(catalog, "_catalog_ready", lambda conn: next(ready_results) and ready(conn))
    assert catalog.prepare_model_catalog_on_startup(engine)["startup_path"] == "read-only-after-lock"


def test_unresolved_assignment_still_fails_on_a_current_catalog(engine):
    with engine.begin() as conn:
        model = conn.execute(text("""
            INSERT INTO models (key, vendor, display_name, default_context_window)
            VALUES ('fixture-old-model', 'test', 'Fixture', 1000) RETURNING id
        """)).scalar_one()
        endpoint = conn.execute(text("""
            INSERT INTO model_endpoints (model_id, access_path_id)
            SELECT :model, id FROM access_paths WHERE key='openai-api' RETURNING id
        """), {"model": model}).scalar_one()
        conn.execute(text("""
            INSERT INTO bot_profiles (slug, name, description, system_prompt, endpoint_id,
                requires_memory, voice_optimized, uses_tools, uses_search, uses_home_assistant,
                created_at, updated_at)
            VALUES ('fixture-unresolved-catalog', 'Fixture', '', '', :endpoint,
                true, false, false, false, false, now(), now())
        """), {"endpoint": endpoint})
        conn.execute(text("UPDATE models SET key='fixture-renamed-model' WHERE id=:id"), {"id": model})
    prior_stamp = catalog._CATALOG_VERSION
    try:
        with pytest.raises(RuntimeError, match="has no normalized endpoint"):
            catalog.prepare_model_catalog_on_startup(engine)
        with engine.connect() as conn:
            assert conn.execute(text(_STAMP)).scalar_one() == prior_stamp
    finally:
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM bot_profiles WHERE slug='fixture-unresolved-catalog'"))
