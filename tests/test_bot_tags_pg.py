"""Bot tags against an isolated, networkless PostgreSQL DB (TASK-1329).

BOT_TAGS.md §10.1 (schema/migration), §10.7 (intermediate deploy through the
real upsert), §10.9 (tenant schemas), §10.10 (purge race) and §10.11 (the
real ``sql_allowed`` query). Fixture bots only, one private DB per test.
"""
from __future__ import annotations

import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine, event, text
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlmodel import SQLModel

from llm_bawt.bot_access import BotAccessError, BotPolicyResolver
from llm_bawt.bot_access.policy import TrustedCaller
from llm_bawt.bot_tags import BehaviorTag, BotTags, sql_allowed
from llm_bawt.bot_tags.vocabulary import LEGACY_FLAG_ASSIGNER, MIGRATION_ASSIGNER
from llm_bawt.orphan_partitions import protected_storage_ids, read_protected_slugs
from llm_bawt.runtime_settings import BotProfile, BotProfileStore
from tests.bot_tag_fixtures import PRIVATE_MODES, make_private, tag
from tests.private_postgres import private_url

pytestmark = pytest.mark.integration

LEGACY = sorted(["private", "no_global_search"])


class Rig:
    def __init__(self, url):
        self.url = url
        self.engines = []
        self.engine = self.connect()

    def connect(self, search_path: str | None = None):
        args = {"options": f"-csearch_path={search_path}"} if search_path else {}
        engine = create_engine(self.url, connect_args=args)
        self.engines.append(engine)
        return engine

    def store(self, search_path: str | None = None) -> BotProfileStore:
        """A real store bootstrapped on a NEW engine: each call is an app start."""
        store = BotProfileStore.__new__(BotProfileStore)
        store.config = None
        store.engine = self.connect(search_path)
        store._ensure_tables_exist()
        return store

    def tags(self, slug: str, schema: str = "public") -> list[str]:
        with self.engine.connect() as conn:
            return list(conn.execute(text(
                f"SELECT tag_slug FROM {schema}.bot_profile_tags WHERE bot_slug=:s ORDER BY tag_slug"
            ), {"s": slug}).scalars())

    def scalar(self, sql: str, **params):
        with self.engine.connect() as conn:
            return conn.execute(text(sql), params).scalar_one()


@pytest.fixture
def rig(private_pg, monkeypatch):
    name = f"bot_tags_{uuid.uuid4().hex[:12]}_test"
    monkeypatch.setenv("LLM_BAWT_POSTGRES_DATABASE", name)
    admin = create_engine(private_url(private_pg, "postgres"), isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        conn.exec_driver_sql(f'CREATE DATABASE "{name}"')
    rig = Rig(private_url(private_pg, name))
    try:
        yield rig
    finally:
        for engine in rig.engines:
            engine.dispose()
        with admin.connect() as conn:
            conn.exec_driver_sql(f'DROP DATABASE "{name}" WITH (FORCE)')
        admin.dispose()


def legacy_profiles(rig, schema: str = "public", **bots: bool):
    """A pre-TASK-1329 bot_profiles (no tag tables) holding ``bots``."""
    engine = rig.connect(schema)
    with engine.begin() as conn:
        if schema != "public":
            conn.exec_driver_sql(f"CREATE TABLE {schema}.bot_profiles "
                                 "(LIKE public.bot_profiles INCLUDING ALL)")
        else:
            SQLModel.metadata.create_all(bind=conn, tables=[BotProfile.__table__])
            BotProfileStore._migrate_add_columns(None, conn)
        now = datetime.now(timezone.utc)
        for slug, flag in bots.items():
            # Unqualified: the engine's search_path picks this schema's table.
            conn.execute(BotProfile.__table__.insert().values(
                slug=slug.replace("_", "-"), name=slug, description="", system_prompt="fixture",
                tts_mode=False, include_summaries=True, include_in_global_search=flag,
                created_at=now, updated_at=now))


# --- §10.1 schema and migration ---------------------------------------------


def test_the_migration_maps_false_to_private_and_quiet_and_true_to_nothing(rig):
    legacy_profiles(rig, fixture_public=True, fixture_private=False)
    rig.store()

    assert rig.tags("fixture-private") == LEGACY
    assert rig.tags("fixture-public") == []
    assert rig.scalar("SELECT count(DISTINCT assigned_by) FROM bot_profile_tags") == 1
    assert rig.scalar("SELECT min(assigned_by) FROM bot_profile_tags") == MIGRATION_ASSIGNER


def test_a_second_bootstrap_issues_no_tag_ddl_or_writes_and_an_untag_survives_it(rig):
    legacy_profiles(rig, fixture_private=False)
    rig.store()
    with rig.engine.begin() as conn:
        conn.exec_driver_sql("DELETE FROM bot_profile_tags WHERE bot_slug='fixture-private'")
        conn.exec_driver_sql("UPDATE bot_profiles SET include_in_global_search=true")
    seen = []
    engine = rig.connect()
    event.listen(engine, "before_cursor_execute", lambda c, cur, sql, *a: seen.append(sql))
    store = BotProfileStore.__new__(BotProfileStore)
    store.config, store.engine = None, engine
    store._ensure_tables_exist()

    assert rig.tags("fixture-private") == []
    tag_writes = [s for s in seen if "bot_profile_tags" in s and
                  s.lstrip().upper().startswith(("CREATE", "INSERT", "DELETE", "ALTER"))]
    assert tag_writes == []
    assert not any(s.lstrip().upper().startswith("CREATE") and "bot_tag_catalog" in s for s in seen)


def test_catalog_constraints_refuse_a_referenced_delete_and_a_system_role(rig):
    legacy_profiles(rig, fixture_private=False)
    rig.store()
    with pytest.raises(IntegrityError, match="foreign key"):
        with rig.engine.begin() as conn:
            conn.exec_driver_sql("DELETE FROM bot_tag_catalog WHERE slug='private'")
    with pytest.raises(IntegrityError, match="check constraint"):
        with rig.engine.begin() as conn:
            conn.exec_driver_sql("INSERT INTO bot_tag_catalog (slug, name, kind, system) "
                                 "VALUES ('fixture_role', 'Fixture', 'role', true)")
    with pytest.raises(IntegrityError, match="check constraint"):
        with rig.engine.begin() as conn:
            conn.exec_driver_sql("INSERT INTO bot_profile_tags (bot_slug, tag_slug) "
                                 "VALUES ('Fixture-Upper', 'private')")


def test_seeds_are_idempotent_and_restore_behaviour_rows_from_code(rig):
    rig.store()
    with rig.engine.begin() as conn:
        conn.exec_driver_sql("UPDATE bot_tag_catalog SET name='Wrong', description='edited' "
                             "WHERE slug='private'")
        conn.exec_driver_sql("UPDATE bot_tag_catalog SET name='Operator coder' WHERE slug='coder'")
    rig.store()

    assert rig.scalar("SELECT count(*) FROM bot_tag_catalog") == 6
    assert rig.scalar("SELECT name FROM bot_tag_catalog WHERE slug='private'") == "Private"
    # TASK-1362: the opt-in behaviour tag is seeded from code like the others.
    assert rig.scalar("SELECT kind FROM bot_tag_catalog WHERE slug='external' AND system") == "behavior"
    # Roles are operator data after their first seed.
    assert rig.scalar("SELECT name FROM bot_tag_catalog WHERE slug='coder'") == "Operator coder"


# --- §10.9 tenant schemas ---------------------------------------------------


def test_a_tenant_schema_migrates_its_own_rows_and_never_touches_public(rig):
    legacy_profiles(rig, fixture_public_private=False)
    rig.store()
    with rig.engine.begin() as conn:
        conn.exec_driver_sql("CREATE SCHEMA tenant")
    legacy_profiles(rig, "tenant", fixture_tenant=False)
    before = rig.scalar("SELECT count(*) FROM public.bot_profile_tags")

    rig.store("tenant,public")

    assert rig.tags("fixture-tenant", "tenant") == LEGACY
    assert rig.tags("fixture-public-private", "tenant") == []
    assert rig.scalar("SELECT count(*) FROM public.bot_profile_tags") == before
    assert rig.tags("fixture-tenant") == []


def test_a_tenant_seeing_only_public_bot_profiles_refuses_and_rolls_back(rig):
    legacy_profiles(rig, fixture_private=False)
    rig.store()
    with rig.engine.begin() as conn:
        conn.exec_driver_sql("CREATE SCHEMA tenant")

    with pytest.raises(RuntimeError, match="not owned by the current schema"):
        rig.store("tenant,public")
    assert rig.scalar("SELECT count(*) FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace "
                      "WHERE n.nspname='tenant'") == 0


def test_a_completely_empty_schema_gets_every_table_and_copies_nothing(rig):
    with rig.engine.begin() as conn:
        conn.exec_driver_sql("CREATE SCHEMA fresh")

    rig.store("fresh")

    tables = rig.scalar("SELECT array_agg(c.relname::text ORDER BY c.relname) FROM pg_class c "
                        "JOIN pg_namespace n ON n.oid=c.relnamespace "
                        "WHERE n.nspname='fresh' AND c.relkind='r'")
    assert {"bot_profiles", "bot_tag_catalog", "bot_profile_tags"} <= set(tables)
    assert rig.scalar("SELECT count(*) FROM fresh.bot_profile_tags") == 0
    assert rig.scalar("SELECT count(*) FROM fresh.bot_tag_catalog") == 6


# --- §10.7 the intermediate deploy (S1–S4, legacy writers only) -------------


def private_everywhere(rig, slug: str) -> bool:
    """Private in the resolver, the SQL fragment and the storage set, or in none."""
    access = BotPolicyResolver(rig.engine).resolve_sync(TrustedCaller("fixture-caller", "owner", "turn"))
    try:
        access.require_target(slug)
        readable = True
    except BotAccessError:
        readable = False
    searchable = slug in {p.bot_id for p in access.aggregate_scope().partitions}
    sql, params = sql_allowed("v.bot", [BehaviorTag.PRIVATE])
    with rig.engine.connect() as conn:
        allowed = conn.execute(text(f"SELECT count(*) FROM (VALUES (:slug)) v(bot) WHERE {sql}"),
                               {"slug": slug, **params}).scalar_one() == 1
        stored = slug in read_protected_slugs(conn)
    answers = {not readable, not searchable, not allowed, stored,
               BotTags(rig.engine).has(slug, BehaviorTag.PRIVATE)}
    assert len(answers) == 1, (slug, readable, searchable, allowed, stored)
    return answers.pop()


def upsert(store, slug, **fields):
    return store.upsert({"slug": slug, "name": slug, "system_prompt": "fixture", **fields})


def test_create_false_is_private_everywhere_and_true_is_public(rig):
    store = rig.store()
    upsert(store, "fixture-caller")
    upsert(store, "fixture-private", include_in_global_search=False)
    upsert(store, "fixture-public", include_in_global_search=True)

    assert rig.tags("fixture-private") == LEGACY
    assert private_everywhere(rig, "fixture-private")
    assert rig.tags("fixture-public") == [] and not private_everywhere(rig, "fixture-public")
    assert rig.scalar("SELECT min(assigned_by) FROM bot_profile_tags") == LEGACY_FLAG_ASSIGNER


def test_patching_the_legacy_flag_toggles_private_and_the_stored_column_reflects_it(rig):
    store = rig.store()
    upsert(store, "fixture-caller")
    upsert(store, "fixture-bot")
    upsert(store, "fixture-bot", include_in_global_search=False)
    assert rig.tags("fixture-bot") == LEGACY and private_everywhere(rig, "fixture-bot")

    upsert(store, "fixture-bot", include_in_global_search=True)
    assert rig.tags("fixture-bot") == [] and not private_everywhere(rig, "fixture-bot")
    # GET /v1/bots emits this stored column unchanged.
    assert store.get("fixture-bot").include_in_global_search is True


def test_a_migrated_private_bot_patched_true_becomes_readable(rig):
    legacy_profiles(rig, fixture_caller=True, fixture_private=False)
    store = rig.store()
    assert private_everywhere(rig, "fixture-private")

    upsert(store, "fixture-private", include_in_global_search=True)

    assert rig.tags("fixture-private") == [] and not private_everywhere(rig, "fixture-private")


def test_an_unchanged_re_save_never_touches_tags(rig):
    store = rig.store()
    upsert(store, "fixture-caller")
    upsert(store, "fixture-private", include_in_global_search=False)
    upsert(store, "fixture-quiet")
    with rig.engine.begin() as conn:
        tag(conn, "fixture-quiet", BehaviorTag.NO_GLOBAL_SEARCH)
        # A writer that bypassed the upsert: column false, only the quiet tag.
        conn.exec_driver_sql("UPDATE bot_profiles SET include_in_global_search=false "
                             "WHERE slug='fixture-quiet'")
    before = rig.scalar("SELECT array_agg(bot_slug || ':' || tag_slug || ':' || assigned_at::text "
                        "ORDER BY bot_slug, tag_slug) FROM bot_profile_tags")

    upsert(store, "fixture-private", include_in_global_search=False, description="re-save")
    upsert(store, "fixture-quiet", include_in_global_search=False, description="re-save")
    upsert(store, "fixture-private", description="partial")

    assert rig.scalar("SELECT array_agg(bot_slug || ':' || tag_slug || ':' || assigned_at::text "
                      "ORDER BY bot_slug, tag_slug) FROM bot_profile_tags") == before
    assert rig.tags("fixture-quiet") == ["no_global_search"]
    # The restrictive fallback still makes it private.
    assert private_everywhere(rig, "fixture-quiet")


@pytest.mark.parametrize("mode", ["column", "tag"])
def test_either_private_signal_alone_is_private_everywhere(rig, mode):
    store = rig.store()
    upsert(store, "fixture-caller")
    upsert(store, "fixture-bot")
    with rig.engine.begin() as conn:
        make_private(conn, "fixture-bot", mode)

    assert private_everywhere(rig, "fixture-bot")
    assert store.get("fixture-bot").include_in_global_search is (mode == "tag")


def test_a_re_created_profileless_private_slug_comes_back_private(rig):
    store = rig.store()
    upsert(store, "fixture-caller")
    upsert(store, "fixture-private", include_in_global_search=False)
    assert store.delete("fixture-private")
    with rig.engine.connect() as conn:
        assert "fixtureprivate" in protected_storage_ids(conn)

    upsert(store, "fixture-private", include_in_global_search=True)

    assert store.get("fixture-private").include_in_global_search is False
    assert rig.tags("fixture-private") == LEGACY and private_everywhere(rig, "fixture-private")


# --- §10.11 the real sql_allowed query --------------------------------------


@pytest.mark.parametrize("mode", PRIVATE_MODES)
@pytest.mark.parametrize("exclude", [[BehaviorTag.PRIVATE], [BehaviorTag.NO_GLOBAL_SEARCH]])
def test_sql_allowed_excludes_every_private_mode_and_admits_the_public_control(rig, mode, exclude):
    store = rig.store()
    for slug in ("fixture-public", "fixture-private"):
        upsert(store, slug)
    with rig.engine.begin() as conn:
        make_private(conn, "fixture-private", mode)
        conn.exec_driver_sql("CREATE TABLE fixture_rows (bot_id text)")
        conn.exec_driver_sql("INSERT INTO fixture_rows VALUES ('fixture-public'), ('Fixture-Private'), "
                             "('fixture-unknown')")
    sql, params = sql_allowed("r.bot_id", exclude, prefix="fx")
    with rig.engine.connect() as conn:
        admitted = conn.execute(text(f"SELECT r.bot_id FROM fixture_rows r WHERE {sql}"), params).scalars().all()

    assert admitted == ["fixture-public"]


def test_sql_allowed_on_inactive_keeps_private_and_unknown_bots(rig):
    store = rig.store()
    for slug in ("fixture-public", "fixture-inactive"):
        upsert(store, slug)
    with rig.engine.begin() as conn:
        tag(conn, "fixture-inactive", BehaviorTag.INACTIVE)
        conn.exec_driver_sql("CREATE TABLE fixture_rows (bot_id text)")
        conn.exec_driver_sql("INSERT INTO fixture_rows VALUES ('fixture-public'), ('fixture-inactive'), "
                             "('fixture-unknown')")
    sql, params = sql_allowed("r.bot_id", [BehaviorTag.INACTIVE])
    with rig.engine.connect() as conn:
        admitted = conn.execute(text(f"SELECT r.bot_id FROM fixture_rows r WHERE {sql} ORDER BY 1"),
                                params).scalars().all()

    assert admitted == ["fixture-public", "fixture-unknown"]


def test_sql_allowed_with_no_tag_table_errors_instead_of_admitting(rig):
    with rig.engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE fixture_rows (bot_id text)")
        conn.exec_driver_sql("INSERT INTO fixture_rows VALUES ('fixture-public')")
    sql, params = sql_allowed("r.bot_id", [BehaviorTag.INACTIVE])
    with pytest.raises(Exception, match="bot_profile_tags"):
        with rig.engine.connect() as conn:
            conn.execute(text(f"SELECT r.bot_id FROM fixture_rows r WHERE {sql}"), params).all()


# --- §10.10 purge race --------------------------------------------------------


@pytest.fixture
def purge(rig, monkeypatch):
    from llm_bawt import runtime_settings
    from llm_bawt.utils import db

    store = rig.store()
    upsert(store, "fixture-public")
    with rig.engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE turn_logs (bot_id text)")
        conn.exec_driver_sql("INSERT INTO turn_logs VALUES ('fixture-public'), ('fixture-orphan'), "
                             "('fixture-gone')")
    monkeypatch.setattr(runtime_settings, "has_database_credentials", lambda config: True)
    monkeypatch.setattr(db, "get_shared_engine", lambda config: rig.engine)
    return lambda **kw: runtime_settings.purge_orphaned_shared_rows(object(), **kw)


def remaining(rig) -> list[str]:
    with rig.engine.connect() as conn:
        return conn.exec_driver_sql("SELECT bot_id FROM turn_logs ORDER BY bot_id").scalars().all()


def test_a_committed_private_assignment_keeps_its_profileless_rows(rig, purge):
    with rig.engine.begin() as conn:
        tag(conn, "fixture-gone", BehaviorTag.PRIVATE)

    result = purge()

    assert result["orphaned_bot_ids"] == ["fixture-orphan"]
    assert remaining(rig) == ["fixture-gone", "fixture-public"]


def test_the_purge_lock_blocks_a_concurrent_assignment_until_commit(rig, purge):
    attempted = False

    def observe(conn, cursor, sql, params, context, many):
        nonlocal attempted
        if not attempted and sql.startswith("SELECT slug FROM bot_profiles"):
            attempted = True

            def assign():
                with rig.engine.begin() as writer:
                    writer.exec_driver_sql("SET LOCAL lock_timeout = '200ms'")
                    tag(writer, "fixture-orphan", BehaviorTag.PRIVATE)

            with ThreadPoolExecutor(max_workers=1) as pool:
                with pytest.raises(OperationalError) as blocked:
                    pool.submit(assign).result(timeout=3)
                assert blocked.value.orig.pgcode == "55P03"

    event.listen(rig.engine, "before_cursor_execute", observe)
    try:
        purge()
    finally:
        event.remove(rig.engine, "before_cursor_execute", observe)
    assert attempted
    # The blocked assignment never committed, so both profile-less slugs go.
    assert remaining(rig) == ["fixture-public"]


def test_an_uncommitted_assignment_makes_the_purge_raise_and_delete_nothing(rig, purge):
    with rig.engine.connect() as holder:
        holder.begin()
        tag(holder, "fixture-orphan", BehaviorTag.PRIVATE)
        with pytest.raises(OperationalError) as conflict:
            purge()
        assert conflict.value.orig.pgcode == "55P03"
        holder.rollback()
    assert remaining(rig) == ["fixture-gone", "fixture-orphan", "fixture-public"]


def test_a_dry_run_takes_no_lock(rig, purge):
    with rig.engine.connect() as holder:
        holder.begin()
        tag(holder, "fixture-gone", BehaviorTag.PRIVATE)
        result = purge(dry_run=True)
        holder.rollback()
    assert set(result["orphaned_bot_ids"]) == {"fixture-gone", "fixture-orphan"}
    assert len(remaining(rig)) == 3
