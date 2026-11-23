"""Read the generated Setup catalog source using synthetic profiles only."""

import json
import shutil
from datetime import timezone
from functools import partial
from hashlib import sha256
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import event, text
from sqlmodel import Session

from llm_bawt.seeding import bot_presets, setup_structure
from llm_bawt.seeding.preset_source import read_preset_source
from llm_bawt.seeding.setup_portable_input import PortableSetupInput
from llm_bawt.seeding.setup_role_dependencies import SetupRole, SetupRoleConflict, install_missing_roles
from llm_bawt.seeding.setup_seed_sync import approved_setup_snapshot, stage_catalog
from llm_bawt.utils import db
from tests.private_postgres import PORT
from tests.test_bot_tags_pg import legacy_profiles, rig  # noqa: F401

pytestmark = pytest.mark.integration


@pytest.fixture
def scoped_readers(rig, private_pg, monkeypatch):  # noqa: F811
    app = partial(setup_structure.read_setup_structure, None)
    # Gate images without a psql binary still exercise the same SQL in the app
    # path. Run the real psql transport parity check on hosts with the client.
    if (psql := shutil.which("psql")) is None:
        return (app,)
    command = (psql, "-h", str(private_pg), "-p", PORT, "-U", "test_user", "-d", rig.url.database)
    monkeypatch.setattr(setup_structure, "_PGQ_COMMAND", (*command, *setup_structure._PGQ_COMMAND[2:]))
    return app, setup_structure.read_setup_structure_pgq


def test_source_filters_private_and_updates_after_privacy_change(rig, scoped_readers, monkeypatch, tmp_path):  # noqa: F811
    legacy_profiles(rig, visible_bot=True, hidden_flag=False, hidden_tag=True, later_private=True, explicit=True)
    rig.store()
    with rig.engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO bot_tag_catalog (slug, name, description, kind, prompt_key, capabilities)
            VALUES ('fixture_role', 'Tenant role', 'Tenant-only notes.', 'role', 'role.unapproved',
                    '["example_grant"]'::jsonb), ('hidden_role', 'Hidden role', '', 'role', NULL, '[]'::jsonb)
        """))
        conn.execute(text("""
            INSERT INTO bot_profile_tags (bot_slug, tag_slug, assigned_by)
            VALUES ('hidden-tag', 'private', 'test'), ('visible-bot', 'fixture_role', 'test'),
                   ('hidden-tag', 'hidden_role', 'test'), ('explicit', 'hidden_role', 'test')
        """))
    source_queries = []

    def source_engine(*_a, **_kw):
        engine = rig.connect()
        event.listen(engine, "before_cursor_execute", lambda c, cur, sql, *a: source_queries.append(sql))
        return engine

    monkeypatch.setattr(db, "build_engine_from_config", source_engine)
    with read_preset_source(None) as conn:
        assert conn.execute(text("SHOW transaction_read_only")).scalar_one() == "on"
        assert conn.execute(text("SHOW transaction_isolation")).scalar_one() == "repeatable read"

    def structure():
        scope = dict(excluded=("EXPLICIT",), allowed_roles=("fixture_role",), denied_roles=("hidden_role",))
        app, *others = (reader(**scope) for reader in scoped_readers)
        assert all(app == result for result in others)
        return app

    current = structure()
    # Keep existing approval bindings valid. Verify the SQL bytes match the
    # old Python encoding with real fixture types, including subsecond UTC.
    def legacy_revision(kind, row):
        def iso(value):
            return (value.astimezone(timezone.utc) if value.tzinfo else value).isoformat()
        return sha256(json.dumps([kind, row["slug"], row.get("id"), iso(row["created_at"]),
                                  iso(row["updated_at"])]).encode()).hexdigest()

    with rig.engine.connect() as conn:
        profile = conn.execute(text("SELECT slug,id,created_at,updated_at FROM bot_profiles "
                                    "WHERE slug='visible-bot'")).mappings().one()
        role = conn.execute(text("SELECT slug,created_at,updated_at FROM bot_tag_catalog "
                                 "WHERE slug='fixture_role'")).mappings().one()
    assert current.profiles["visible-bot"] == legacy_revision("profile", profile)
    assert current.roles["fixture_role"] == legacy_revision("role", role)
    assert set(current.profiles) == {"later-private", "visible-bot"}
    assert current.memberships == {"later-private": set(), "visible-bot": {"fixture_role"}}
    assert set(current.roles) == {"fixture_role"}
    approved = PortableSetupInput({slug: {
        "revision": rev, "roles": list(current.memberships[slug]),
        "preset": {"name": "Fixture " + slug, "summary": "Portable notes.",
                   "role": "Review", "system_prompt": "Review independently."},
    } for slug, rev in current.profiles.items()}, {"fixture_role": {
        "revision": current.roles["fixture_role"], "name": "Fixture role", "color": None,
        "description": "Reviews fixture changes.", "prompt_key": "role.fixture", "prompt_sha256": "a" * 64,
    }})
    target = tmp_path / "generated.json"

    def read():
        return approved_setup_snapshot(structure(), approved, {"role.fixture": "a" * 64})

    assert stage_catalog(read, target)
    prior = target.read_bytes()
    assert json.loads(prior)["roles"] == [{"slug": "fixture_role", "name": "Fixture role", "color": None,
                                          "description": "Reviews fixture changes.", "prompt_key": "role.fixture"}]
    for marker in (b"Tenant", b"hidden_role", b"example_grant", b"role.unapproved", b"revision"):
        assert marker not in prior
    assert not stage_catalog(read, target)
    with rig.engine.begin() as conn:
        conn.execute(text("UPDATE bot_profiles SET system_prompt='private synthetic edit', updated_at=now() "
                          "WHERE slug='visible-bot'"))
    with pytest.raises(bot_presets.PresetError, match="changed revision"):
        stage_catalog(read, target)
    assert target.read_bytes() == prior
    # Refresh the binding without reading or copying the edited live prompt.
    approved.profiles["visible-bot"]["revision"] = structure().profiles["visible-bot"]
    assert not stage_catalog(read, target)
    with rig.engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO bot_profile_tags (bot_slug, tag_slug, assigned_by)
            VALUES ('later-private', 'private', 'test')
        """))
    assert stage_catalog(read, target)
    assert [row["slug"] for row in json.loads(target.read_bytes())["presets"]] == ["visible-bot"]
    assert not any(field in sql.lower() for sql in source_queries for field in (
        "system_prompt", "description", "prompt_key", "capabilities", "endpoint_id", "avatar", "overrides",
    ))


def test_unknown_absent_or_denied_shared_role_has_one_opaque_result(rig, scoped_readers, monkeypatch):  # noqa: F811
    legacy_profiles(rig, public_first=True, public_second=True, fixture_protected=True)
    rig.store()
    monkeypatch.setattr(db, "build_engine_from_config", lambda *_a, **_kw: rig.connect())
    with rig.engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO bot_tag_catalog (slug, name, description, kind)
            VALUES ('fixture_shared', 'Fixture shared', 'Tenant note', 'role'),
                   ('fixture_protected_role', 'Fixture protected', 'Private note', 'role');
            INSERT INTO bot_profile_tags (bot_slug, tag_slug)
            VALUES ('public-first', 'fixture_shared'), ('public-second', 'fixture_shared'),
                   ('fixture-protected', 'fixture_protected_role'), ('public-first', 'inactive');
        """))
    parse = setup_structure._parse_structure
    received = []

    def capture(payload):
        received.append(json.loads(payload))
        return parse(payload)

    monkeypatch.setattr(setup_structure, "_parse_structure", capture)

    def read(expected_roles, denied=()):
        for reader in scoped_readers:
            yield reader(excluded=("fixture-protected",), allowed_roles=expected_roles, denied_roles=denied)

    for result in read(("fixture_shared",), ("fixture_protected_role",)):
        assert result.memberships == {"public-first": {"fixture_shared"}, "public-second": {"fixture_shared"}}
        assert result.roles.keys() == {"fixture_shared"}
    assert len(received) == len(scoped_readers) and all(payload == received[0] for payload in received)
    for payload in received:
        assert set(payload) == {"status", "profiles", "roles"}
        assert [row[0] for row in payload["profiles"]] == ["public-first", "public-second"]
        assert all(len(row) == 3 for row in payload["profiles"])
        assert [row[0] for row in payload["roles"]] == ["fixture_shared"]
        assert all(len(row) == 2 for row in payload["roles"])
        assert "fixture_protected" not in json.dumps(payload)
    # Not approved at all, or explicitly denied despite positive approval.
    for allowed, denied in (((), ()), (("fixture_shared",), ("fixture_shared",))):
        for reader in scoped_readers:
            with pytest.raises(bot_presets.PresetError) as error:
                reader(excluded=("fixture-protected",), allowed_roles=allowed, denied_roles=denied)
            assert str(error.value) == setup_structure.ROLE_HOLD
    # The membership remains but the allowed key no longer resolves as a role.
    with rig.engine.begin() as conn:
        conn.execute(text("UPDATE bot_tag_catalog SET kind='behavior' WHERE slug='fixture_shared'"))
    for reader in scoped_readers:
        with pytest.raises(bot_presets.PresetError) as error:
            reader(excluded=("fixture-protected",), allowed_roles=("fixture_shared",), denied_roles=())
        assert str(error.value) == setup_structure.ROLE_HOLD
    assert received[len(scoped_readers):] == [{"status": "role_hold"}] * (3 * len(scoped_readers))


def test_partial_or_failed_query_cannot_return_a_snapshot(rig, scoped_readers, monkeypatch):  # noqa: F811
    legacy_profiles(rig, fixture_first=True, fixture_second=True)
    rig.store()
    monkeypatch.setattr(db, "build_engine_from_config", lambda *_a, **_kw: rig.connect())
    scope = dict(excluded=(), allowed_roles=(), denied_roles=())
    for reader in scoped_readers:
        assert set(reader(**scope).profiles) == {"fixture-first", "fixture-second"}
    # Distinct raw profile keys with the same normalized key are ambiguous.
    with rig.engine.begin() as conn:
        conn.execute(text("UPDATE bot_profiles SET slug='FIXTURE-FIRST' WHERE slug='fixture-second'"))
    for reader in scoped_readers:
        with pytest.raises(bot_presets.PresetError) as error:
            reader(**scope)
        assert str(error.value) == setup_structure.READ_FAILURE
    # A server error (not an empty result) produces the same safe failure.
    with rig.engine.begin() as conn:
        conn.execute(text("ALTER TABLE bot_tag_catalog RENAME TO fixture_unavailable"))
    for reader in scoped_readers:
        with pytest.raises(bot_presets.PresetError) as error:
            reader(**scope)
        assert str(error.value) == setup_structure.READ_FAILURE


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


def test_shipped_catalog_import_is_create_only_on_disposable_pg(rig, monkeypatch):  # noqa: F811
    from llm_bawt.service.routes import setup_presets as routes
    from tests.test_bot_presets import _Settings, _USABLE, _client

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
    selection = {"selection": [item.ref for item in (first, second, existing)]}
    client = _client()

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
