"""Read the generated Setup catalog source using synthetic profiles only."""

import json
import shutil
from functools import partial
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import event, text
from sqlmodel import Session

from llm_bawt.bot_tags.catalog import STARTER_ROLES, seed_starter_roles
from llm_bawt.seeding import bot_presets, setup_live_source
from llm_bawt.seeding.preset_source import read_preset_source
from llm_bawt.seeding.setup_role_dependencies import SetupRole, SetupRoleConflict, install_missing_roles
from llm_bawt.seeding.setup_seed_sync import SetupSeedSnapshot, live_setup_snapshot, stage_catalog
from llm_bawt.utils import db
from tests.private_postgres import PORT
from tests.test_bot_tags_pg import legacy_profiles, rig  # noqa: F401

pytestmark = pytest.mark.integration


@pytest.fixture
def live_readers(rig, private_pg, monkeypatch):  # noqa: F811
    app = partial(setup_live_source.read_live_setup_source, None)
    # Gate images without a psql binary still exercise the same SQL in the app
    # path. Run the real psql transport parity check on hosts with the client.
    if (psql := shutil.which("psql")) is None:
        return (app,)
    command = (psql, "-h", str(private_pg), "-p", PORT, "-U", "test_user", "-d", rig.url.database)
    monkeypatch.setattr(setup_live_source, "_PGQ_COMMAND", (*command, *setup_live_source._PGQ_COMMAND[2:]))
    return app, setup_live_source.read_live_setup_source_pgq


def test_live_lineup_filters_private_and_follows_edits(rig, live_readers, monkeypatch, tmp_path):  # noqa: F811
    legacy_profiles(rig, visible_bot=True, hidden_flag=False, hidden_tag=True, later_private=True, explicit=True)
    rig.store()
    with rig.engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO bot_tag_catalog (slug, name, description, kind, prompt_key, capabilities)
            VALUES ('fixture_role', 'Fixture role', 'Reviews fixture changes.', 'role', 'role.fixture',
                    '["example_grant"]'::jsonb), ('hidden_role', 'Hidden role', 'Private note', 'role', NULL, '[]'::jsonb)
        """))
        conn.execute(text("""
            INSERT INTO bot_profile_tags (bot_slug, tag_slug, assigned_by)
            VALUES ('hidden-tag', 'private', 'test'), ('visible-bot', 'fixture_role', 'test'),
                   ('visible-bot', 'inactive', 'test'),
                   ('hidden-tag', 'hidden_role', 'test'), ('explicit', 'hidden_role', 'test')
        """))
        conn.execute(text("UPDATE bot_profiles SET system_prompt='Review fixture work.', avatar='🧪', "
                          "color='#123456', description='Fixture reviewer.' WHERE slug='visible-bot'"))
    source_queries = []

    def source_engine(*_a, **_kw):
        engine = rig.connect()
        event.listen(engine, "before_cursor_execute", lambda c, cur, sql, *a: source_queries.append(sql))
        return engine

    monkeypatch.setattr(db, "build_engine_from_config", source_engine)
    with read_preset_source(None) as conn:
        assert conn.execute(text("SHOW transaction_read_only")).scalar_one() == "on"
        assert conn.execute(text("SHOW transaction_isolation")).scalar_one() == "repeatable read"

    def read():
        app, *others = (reader(excluded=("EXPLICIT",)) for reader in live_readers)
        assert all(app == result for result in others)
        return app

    source = read()
    assert [row["slug"] for row in source.profiles] == ["later-private", "visible-bot"]
    visible = source.profiles[1]
    assert visible["roles"] == ["fixture_role"] and visible["system_prompt"] == "Review fixture work."
    assert source.roles == {"fixture_role": {"slug": "fixture_role", "name": "Fixture role", "color": None,
                                             "description": "Reviews fixture changes.", "prompt_key": "role.fixture"}}
    target = tmp_path / "generated.json"
    assert stage_catalog(lambda: live_setup_snapshot(read(), {"role.fixture"}), target)
    prior = json.loads(target.read_bytes())
    assert [(p["slug"], p["icon"], p["color"], p["role"], p["summary"]) for p in prior["presets"]] == [
        ("later-private", "", None, "Assistant", "General-purpose assistant."),
        ("visible-bot", "🧪", "#123456", "Fixture role", "Fixture reviewer."),
    ]
    for marker in ("hidden", "Private note", "example_grant", "inactive", "explicit"):
        assert marker not in target.read_text()
    # A live edit flows into the next run with no approval file in the loop.
    with rig.engine.begin() as conn:
        conn.execute(text("UPDATE bot_profiles SET system_prompt='Edited fixture prompt.', updated_at=now() "
                          "WHERE slug='visible-bot'"))
    assert stage_catalog(lambda: live_setup_snapshot(read(), {"role.fixture"}), target)
    assert json.loads(target.read_bytes())["presets"][1]["system_prompt"] == "Edited fixture prompt."
    # Marking a bot private removes it on the next run.
    with rig.engine.begin() as conn:
        conn.execute(text("INSERT INTO bot_profile_tags (bot_slug, tag_slug, assigned_by) "
                          "VALUES ('later-private', 'private', 'test')"))
    assert stage_catalog(lambda: live_setup_snapshot(read(), {"role.fixture"}), target)
    assert [row["slug"] for row in json.loads(target.read_bytes())["presets"]] == ["visible-bot"]
    assert not any(field in sql.lower() for sql in source_queries for field in (
        "capabilities", "endpoint_id", "default_model", "agent_backend", "nextcloud", "last_session",
        "house", "prompt_override", "avatar_render",
    ))


def test_partial_or_failed_query_cannot_return_a_snapshot(rig, live_readers, monkeypatch):  # noqa: F811
    legacy_profiles(rig, fixture_first=True, fixture_second=True)
    rig.store()
    monkeypatch.setattr(db, "build_engine_from_config", lambda *_a, **_kw: rig.connect())
    for reader in live_readers:
        assert [row["slug"] for row in reader(excluded=()).profiles] == ["fixture-first", "fixture-second"]
    # Distinct raw profile keys with the same normalized key are ambiguous.
    with rig.engine.begin() as conn:
        conn.execute(text("UPDATE bot_profiles SET slug='FIXTURE-FIRST' WHERE slug='fixture-second'"))
    for reader in live_readers:
        with pytest.raises(bot_presets.PresetError) as error:
            reader(excluded=())
        assert str(error.value) == setup_live_source.READ_FAILURE
    # A server error (not an empty result) produces the same safe failure.
    with rig.engine.begin() as conn:
        conn.execute(text("ALTER TABLE bot_tag_catalog RENAME TO fixture_unavailable"))
    for reader in live_readers:
        with pytest.raises(bot_presets.PresetError) as error:
            reader(excluded=())
        assert str(error.value) == setup_live_source.READ_FAILURE


def test_generated_subset_creates_only_its_roles_and_rolls_back_on_conflict(rig, monkeypatch, tmp_path):  # noqa: F811
    from llm_bawt.seeding.setup_seed_sync import SetupSeedSnapshot

    target = tmp_path / "setup_catalog.json"
    roles = {s: {"slug": s, "name": s.title(), "capabilities": ["source_only"]} for s in ("audit", "coach")}
    profiles = [{"key": slug, "slug": slug, "name": slug.title(), "role": "Assistant",
                 "summary": "A fixture bot.", "system_prompt": "Be useful.", "tags": [role]}
                for slug, role in (("first", "audit"), ("second", "coach"))]
    assert stage_catalog(lambda: SetupSeedSnapshot({"first", "second"}, profiles, roles, set()), target)
    monkeypatch.setattr(bot_presets, "GENERATED_CATALOG", target)
    first, second = bot_presets.load_generated_pack(target).presets
    store = rig.store()

    def create(preset, roles, slug):
        with Session(store.engine) as session:
            store.upsert({"slug": slug, "tags": list(preset.tags)}, session=session, create_only=True,
                         before_create=partial(install_missing_roles, roles=roles))
            session.commit()

    create(first, first.role_dependencies, first.slug)
    assert rig.tags(first.slug) == ["audit"]
    assert store.get(second.slug) is None
    assert rig.scalar("SELECT array_agg(slug ORDER BY slug) FROM bot_tag_catalog WHERE kind='role'") == ["audit"]
    assert rig.scalar("SELECT capabilities FROM bot_tag_catalog WHERE slug='audit'") == []
    create(first, first.role_dependencies, "another")
    assert rig.tags("another") == ["audit"]
    with rig.engine.begin() as conn:
        conn.execute(text("UPDATE bot_tag_catalog SET capabilities='[\"tenant_grant\"]' WHERE slug='audit'"))
    with pytest.raises(SetupRoleConflict, match="conflicts"):
        create(second, (*second.role_dependencies, *first.role_dependencies), second.slug)
    assert store.get(second.slug) is None
    assert rig.scalar("SELECT array_agg(slug ORDER BY slug) FROM bot_tag_catalog WHERE kind='role'") == ["audit"]
    assert rig.scalar("SELECT capabilities FROM bot_tag_catalog WHERE slug='audit'") == ["tenant_grant"]

    # Same display name at a different slug must not be attached or replaced.
    with rig.engine.begin() as conn:
        conn.execute(text("INSERT INTO bot_tag_catalog (slug,name,kind) VALUES ('local_coach','COACH','role')"))
    with pytest.raises(SetupRoleConflict, match="conflicts"):
        create(second, (SetupRole("coach", "Coach", "", None, None),), second.slug)
    assert store.get(second.slug) is None


def _stage_neutral_catalog(target):
    """A neutral generated roster for copies without the installation's own catalog."""
    profiles = [{"key": slug, "slug": slug, "name": slug.title(), "summary": "Does a thing.",
                 "system_prompt": f"Be a useful {slug}.", "role": "Assistant", "tags": tags,
                 "recommended": slug == "alpha"}
                for slug, tags in (("alpha", ["audit"]), ("beta", ["coach"]), ("gamma", []), ("delta", []))]
    roles = {slug: {"slug": slug, "name": slug.title()} for slug in ("audit", "coach")}
    slugs = {p["slug"] for p in profiles}
    assert stage_catalog(lambda: SetupSeedSnapshot(slugs, profiles, roles, set()), target)


def _setup_client(monkeypatch, store):
    from llm_bawt.service.routes import setup_presets as routes
    from tests.test_bot_presets import _Settings, _USABLE, _client

    settings = _Settings()
    writes = []

    async def persist(payload, *, create_only, before_create=None):
        assert create_only is True
        writes.append(payload["slug"])
        return store.upsert(payload, create_only=True, before_create=before_create)

    service = SimpleNamespace(config=object())
    monkeypatch.setattr(routes, "get_service", lambda: service)
    monkeypatch.setattr(routes, "get_runtime_settings_store", lambda _: settings)
    monkeypatch.setattr(routes, "get_bot_profile_store", lambda _: store)
    monkeypatch.setattr(routes, "usable_endpoints", lambda _: _USABLE)
    monkeypatch.setattr(routes, "check_agent_runtime", AsyncMock(return_value=SimpleNamespace(ready=True)))
    monkeypatch.setattr(routes, "_persist_bot_profile", persist)
    monkeypatch.setattr(routes, "_record_first_default_bot", lambda *_: None)
    return _client(), writes


def test_shipped_catalog_import_is_create_only_on_disposable_pg(rig, monkeypatch, tmp_path):  # noqa: F811
    if not bot_presets.GENERATED_CATALOG.is_file():
        # The public source export omits the installation's catalog; import a neutral one instead.
        _stage_neutral_catalog(tmp_path / "setup_catalog.json")
        monkeypatch.setattr(bot_presets, "GENERATED_CATALOG", tmp_path / "setup_catalog.json")

    stock = bot_presets.all_packs(None)[0]
    assert stock.pack == bot_presets.GENERATED_PACK
    # Exercise a strict subset with two independent role dependencies.
    by_role = {p.tags[0]: p for p in stock.presets if len(p.tags) == 1}
    first, second = list(by_role.values())[:2]
    existing = next(p for p in stock.presets if p.slug not in {first.slug, second.slug})
    assert len(stock.presets) > 3
    store = rig.store()
    assert rig.scalar("SELECT count(*) FROM bot_tag_catalog WHERE kind='role'") == 0
    store.upsert({"slug": existing.slug, "name": "Fixture local", "system_prompt": "Private fixture body"})
    local_before = store.get(existing.slug).model_dump()
    selection = {"selection": [item.ref for item in (first, second, existing)]}
    client, writes = _setup_client(monkeypatch, store)

    def actions():
        response = client.post("/v1/setup/bot-presets", json=selection)
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["ok"]
        return {row["ref"]: row["action"] for row in result["results"]}

    assert actions() == {first.ref: "created", second.ref: "created", existing.ref: "skip"}
    assert set(writes) == {first.slug, second.slug} and len(writes) == 2
    assert {row.slug for row in store.list_all()} == {first.slug, second.slug, existing.slug}
    for item in (first, second):
        assert store.get(item.slug).system_prompt == item.system_prompt
        assert rig.tags(item.slug) == list(item.tags)
    with rig.engine.connect() as conn:
        rows = conn.execute(text("SELECT slug, prompt_key, capabilities FROM bot_tag_catalog WHERE kind='role'"))
        assert {slug: (key, grants) for slug, key, grants in rows} == {
            role.slug: (role.prompt_key, []) for item in (first, second) for role in item.role_dependencies
        }
    assert store.get(existing.slug).model_dump() == local_before
    store.upsert({"slug": first.slug, "system_prompt": "Fixture locally edited after import"})
    edited = store.get(first.slug).model_dump()
    assert actions() == {item.ref: "skip" for item in (first, second, existing)}
    assert len(writes) == 2
    assert store.get(first.slug).model_dump() == edited
    assert store.get(existing.slug).model_dump() == local_before


def test_setup_retries_stock_presets_with_seeded_roles(rig, monkeypatch, tmp_path):  # noqa: F811
    if not bot_presets.GENERATED_CATALOG.is_file():
        # Exercise real starter roles even in exports without the installation's roster.
        target = tmp_path / "setup_catalog.json"
        roles = {slug: {"slug": slug, "name": name, "description": description, "prompt_key": f"role.{slug}"}
                 for slug, name, description, _ in STARTER_ROLES}
        profiles = [{"key": f"fixture-{slug}", "slug": f"fixture-{slug}", "name": f"Fixture {slug}",
                     "summary": "Does a thing.", "system_prompt": "Be useful.",
                     "tags": [slug] if slug in roles else []} for slug in (*roles, "plain")]
        keys = {role["prompt_key"] for role in roles.values()}
        assert stage_catalog(lambda: SetupSeedSnapshot({p["slug"] for p in profiles}, profiles, roles, keys), target)
        monkeypatch.setattr(bot_presets, "GENERATED_CATALOG", target)

    stock = bot_presets.load_generated_pack(bot_presets.GENERATED_CATALOG)
    existing = next(p for p in stock.presets if not p.tags)
    store = rig.store()
    with store.engine.begin() as conn:
        seed_starter_roles(conn)

    def seeded_rows():
        with store.engine.connect() as conn:
            return {r.slug: r for r in conn.execute(text("SELECT * FROM bot_tag_catalog WHERE kind='role'"))}

    stock_coder = next(p for p in stock.presets if "coder" in p.tags)
    coder = seeded_rows()["coder"]
    assert coder.capabilities == ["implement"]
    store.upsert({"slug": existing.slug, "name": "Fixture already created"})
    existing_before = store.get(existing.slug).model_dump()
    client, writes = _setup_client(monkeypatch, store)
    for column, operator_value, shipped_value in (
        ("capabilities", '["implement", "tenant_grant"]', json.dumps(coder.capabilities)),
        ("description", "Operator instructions", coder.description),
    ):
        def set_role_field(value):
            expr = "CAST(:value AS jsonb)" if column == "capabilities" else ":value"
            with store.engine.begin() as conn:
                conn.execute(text(f"UPDATE bot_tag_catalog SET {column}={expr} WHERE slug='coder'"), {"value": value})

        set_role_field(operator_value)
        edited = seeded_rows()["coder"]
        response = client.post("/v1/setup/bot-presets", json={"selection": [stock_coder.ref, existing.ref]})
        assert response.status_code == 200, response.text
        results = {p["ref"]: p for p in response.json()["results"]}
        assert results[stock_coder.ref]["action"] == "blocked"
        assert "conflicts with an existing local tag" in results[stock_coder.ref]["reason"]
        assert results[existing.ref]["action"] == "skip"
        assert store.get(stock_coder.slug) is None
        assert seeded_rows()["coder"] == edited
        set_role_field(shipped_value)

    before = seeded_rows()
    writes.clear()
    response = client.post("/v1/setup/bot-presets", json={"selection": [p.ref for p in stock.presets]})
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["ok"], result["results"]
    assert {row["ref"]: row["action"] for row in result["results"]} == {
        p.ref: "skip" if p == existing else "created" for p in stock.presets
    }
    assert set(writes) == {p.slug for p in stock.presets if p != existing}
    assert store.get(existing.slug).model_dump() == existing_before
    for preset in stock.presets:
        assert rig.tags(preset.slug) == list(preset.tags)
    after = seeded_rows()
    assert {slug: after[slug] for slug in before} == before
