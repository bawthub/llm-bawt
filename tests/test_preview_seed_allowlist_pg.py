"""TASK-1802: seed A is deny-by-default against the real full-store schema.

Runs on the acceptance fixture's disposable database, never operational
PostgreSQL. Bootstraps every store, then proves (1) every table and every
copied column is classified, and (2) the export queries keep private bots,
credentials and identities out of the seed.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
from sqlalchemy import text

from llm_bawt.preview import seed_allowlist
from llm_bawt.preview.seed_allowlist import COPY, SEED_ACTOR, classify, export_query
from llm_bawt.runtime_settings import BotProfile
from llm_bawt.schema_bootstrap import bootstrap_full_store
from tests.full_store_fixture import full_store_db  # noqa: F401  (pytest fixture)

pytestmark = pytest.mark.integration

TABLES_SQL = """
SELECT c.relname, array_agg(a.attname::text ORDER BY a.attnum)
FROM pg_class c
JOIN pg_namespace n ON n.oid = c.relnamespace AND n.nspname = 'public'
JOIN pg_attribute a ON a.attrelid = c.oid AND a.attnum > 0 AND NOT a.attisdropped
WHERE c.relkind IN ('r', 'p') AND NOT c.relispartition
GROUP BY c.relname
"""


def _tables(engine) -> dict[str, list[str]]:
    with engine.connect() as conn:
        return {name: list(columns) for name, columns in conn.execute(text(TABLES_SQL))}


@pytest.fixture
def store(full_store_db):  # noqa: F811
    config, engine = full_store_db
    bootstrap_full_store(config, engine)
    return engine


def test_every_bootstrapped_table_and_copied_column_is_classified(store):
    tables = _tables(store)
    assert set(COPY) <= set(tables), sorted(set(COPY) - set(tables))
    result = classify(tables)
    assert result.unclassified_tables == []
    assert result.column_drift == {}


def test_classification_fails_closed():
    result = classify({"brand_new_table": ["id"], "models": ["id", "brand_new_column"]})
    assert result.unclassified_tables == ["brand_new_table"]
    missing, _extra = result.column_drift["models"]
    assert missing == ["brand_new_column"]
    assert not result.ok
    assert seed_allowlist.unclassified_global_keys(
        ["temperature", "provider_connection:new", "notify", "brand_new_key"]) == ["brand_new_key"]
    assert seed_allowlist.unclassified_bot_keys(["ui_color", "brand_new_key"]) == ["brand_new_key"]


def _rows(engine, table):
    with engine.connect() as conn:
        return [dict(row._mapping) for row in conn.execute(text(export_query(table)))]


def test_export_excludes_private_bots_secrets_and_identity(store):
    # A pristine store has no catalog endpoint to bind and every harness needs
    # one, so the synthetic bots are planted with the endpoint trigger paused
    # (same pattern as test_schema_full_store_pg). Re-enabling needs its own
    # transaction: the inserts leave deferred trigger events behind.
    with store.begin() as conn:
        conn.exec_driver_sql("ALTER TABLE public.bot_profiles DISABLE TRIGGER bot_profiles_model_endpoint_check")
    try:
        _plant(store)
    finally:
        with store.begin() as conn:
            conn.exec_driver_sql("ALTER TABLE public.bot_profiles ENABLE TRIGGER bot_profiles_model_endpoint_check")
    _assert_export(store)


def _plant(store):
    with store.begin() as conn:
        conn.execute(text("""
            INSERT INTO bot_tag_catalog (slug, name, description, kind, capabilities, system, created_at, updated_at)
            VALUES ('private', 'Private', '', 'behavior', '[]', true, now(), now())
            ON CONFLICT (slug) DO NOTHING"""))
        for slug in ("seed_public", "seed_secret", "seed_hidden"):
            conn.execute(BotProfile.__table__.insert().values(
                slug=slug, name=slug, description="", system_prompt="prompt", harness="chat",
                nextcloud_config={"password": "x"},
                agent_backend_config={"effort": "high", "session_key": "sk-1"},
                include_in_global_search=slug != "seed_hidden",
                created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc),
            ))
        # JSON null (not SQL NULL) must not break the key filter.
        conn.execute(BotProfile.__table__.insert().values(
            slug="seed_jsonnull", name="seed_jsonnull", description="", system_prompt="prompt",
            harness="chat", agent_backend_config=text("'null'::jsonb"),
            created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc),
        ))
        conn.execute(text("""
            INSERT INTO bot_profile_tags (bot_slug, tag_slug, assigned_at, assigned_by)
            VALUES ('seed_secret', 'private', now(), 'someone@example.com')"""))
        conn.execute(text("""
            INSERT INTO runtime_settings (scope_type, scope_id, key, value_json, updated_at) VALUES
              ('global', '*', 'temperature', '0.5', now()),
              ('global', '*', 'provider_connection:x', '{"api_key": "k"}', now()),
              ('global', '*', 'notify', '{"ha_services": ["phone"]}', now()),
              ('bot', 'seed_public', 'ui_color', '"red"', now()),
              ('bot', 'seed_secret', 'ui_color', '"blue"', now()),
              ('user', 'someone', 'ui_color', '"green"', now())"""))


def _assert_export(store):
    bots = {row["slug"]: row for row in _rows(store, "bot_profiles")}
    assert "seed_secret" not in bots
    assert "seed_hidden" not in bots  # column-only private bot
    assert bots["seed_jsonnull"]["agent_backend_config"] in ({}, "{}")
    public = bots["seed_public"]
    assert public["nextcloud_config"] is None and public["last_session_id"] is None
    config = public["agent_backend_config"]
    config = json.loads(config) if isinstance(config, str) else config
    assert config == {"effort": "high"}

    assert all(row["bot_slug"] != "seed_secret" for row in _rows(store, "bot_profile_tags"))
    settings = {(row["scope_type"], row["scope_id"], row["key"]) for row in _rows(store, "runtime_settings")}
    assert ("global", "*", "temperature") in settings
    assert ("bot", "seed_public", "ui_color") in settings
    assert not {key for _, _, key in settings} & {"provider_connection:x", "notify"}
    assert ("bot", "seed_secret", "ui_color") not in settings
    assert all(scope in ("global", "bot") for scope, _, _ in settings)

    with store.begin() as conn:
        conn.execute(text("""
            INSERT INTO bot_profile_tags (bot_slug, tag_slug, assigned_at, assigned_by)
            VALUES ('seed_public', 'no_global_search', now(), 'someone@example.com')"""))
    tags = _rows(store, "bot_profile_tags")
    assert [(row["bot_slug"], row["assigned_by"]) for row in tags] == [("seed_public", SEED_ACTOR)]
