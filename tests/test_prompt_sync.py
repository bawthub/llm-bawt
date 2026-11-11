"""Prompt sync: classification, drift, export and tenants following defaults (TASK-2127)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sqlmodel import Session, SQLModel, create_engine, select

from llm_bawt import prompt_synced_defaults
from llm_bawt.prompt_definitions import PromptDefinition
from llm_bawt.prompt_registry import PromptTemplate, PromptTemplateStore, PromptTemplateVersion
from llm_bawt.prompt_sync import (
    DEFAULT_MODE_KEY,
    DEFAULT_SHA_KEY,
    INSTALL,
    PRODUCT,
    classify,
    compute_drift,
    drift_report,
    export_report,
    is_following,
)
from llm_bawt.prompt_synced_defaults import body_sha256, load_manifest


def _row(key, body, *, category="agents", scope_type="global", updated_by="nick", metadata=None):
    return PromptTemplate(
        key=key, scope_type=scope_type, scope_id="*", title=key, category=category,
        format="plain_text", body=body, updated_by=updated_by,
        metadata_json=json.dumps(metadata) if metadata else None,
    )


def _defn(key, body):
    return PromptDefinition(key=key, title=key, category="agents", loader=lambda: body)


def test_classify_defaults_and_explicit_override() -> None:
    assert classify(_row("a", "x")) == PRODUCT
    assert classify(_row("persona.x", "x", category="persona")) == INSTALL
    assert classify(_row("a", "x", scope_type="bot")) == INSTALL
    assert classify(_row("a", "x", metadata={"sync": "install"})) == INSTALL
    assert classify(_row("persona.x", "x", category="persona", metadata={"sync": "product"})) == INSTALL
    assert classify(_row("persona.x", "x", metadata={"sync": "product"})) == INSTALL
    assert classify(_row("a", "x", scope_type="bot", metadata={"sync": "product"})) == INSTALL


def test_drift_covers_changed_and_new_product_rows_only() -> None:
    defs = {"same": _defn("same", "A"), "edited": _defn("edited", "B"), "stale": _defn("stale", "new")}
    rows = [
        _row("same", "A"),
        _row("edited", "B2"),
        _row("stale", "old", updated_by="system:seed"),
        _row("brand.new", "N"),
        _row("persona.me", "P", category="persona"),
        _row("local", "L", metadata={"sync": "install"}),
    ]
    drift = {item.key: item.kind for item in compute_drift(rows, defs)}
    assert drift == {"edited": "changed", "brand.new": "new"}
    report = drift_report(rows, defs)
    assert report["in_sync"] is False and report["drift_count"] == 1
    assert report["drift"][0]["key"] == "stale"
    assert {item["key"] for item in report["overrides"]} == {"same", "edited", "brand.new", "persona.me", "local"}
    assert report["classified"] == {PRODUCT: 4, INSTALL: 2}
    assert "body" not in report["promotion_candidates"][0]


def test_is_following_requires_default_writer_and_no_override_intent() -> None:
    assert is_following(_row("k", "x", updated_by="system:seed"))
    assert not is_following(_row("k", "x", updated_by="snark"))
    assert not is_following(_row("k", "x", updated_by="snark", metadata={DEFAULT_SHA_KEY: body_sha256("x")}))
    assert is_following(_row("k", "x", updated_by="system:sync", metadata={DEFAULT_SHA_KEY: body_sha256("x")}))
    assert not is_following(_row("k", "y", updated_by="system:seed", metadata={DEFAULT_SHA_KEY: body_sha256("x")}))
    assert not is_following(_row("k", "x", updated_by="system:seed", metadata={DEFAULT_MODE_KEY: "override"}))
    assert not is_following(_row("k", "x", updated_by="system:seed", scope_type="bot"))


def test_export_writes_bodies_and_manifest(tmp_path: Path) -> None:
    defs = {"edited": _defn("edited", "B")}
    rows = [_row("edited", "B2"), _row("brand.new", "N {thing}", category="repos")]
    rows[1].required_vars_json = json.dumps(["thing"])
    result = export_report(drift_report(rows, defs, include_body=True), tmp_path, approved_keys=["edited", "brand.new"])
    assert result["written"] == ["brand.new", "edited"]
    assert (tmp_path / "edited.txt").read_text() == "B2"
    assert (tmp_path / "brand.new.txt").read_text() == "N {thing}"
    manifest = load_manifest(tmp_path)
    assert set(manifest) == {"brand.new"}
    assert manifest["brand.new"]["category"] == "repos"
    assert manifest["brand.new"]["required_vars"] == ["thing"]


def test_export_holds_bodies_naming_install_terms(tmp_path: Path) -> None:
    rows = [_row("clean", "Ask the user first."), _row("named", "Ask Pat first; never touch Zed."),
            _row("partial", "Patterns and Zedonk are fine.")]
    result = export_report(drift_report(rows, {}, include_body=True), tmp_path, ["Pat", "Zed", " "],
                           approved_keys=["clean", "named", "partial"])
    assert result["written"] == ["clean", "partial"]
    assert result["held"] == {"named": ["installation-specific term"]}
    assert not (tmp_path / "named.txt").exists()
    assert "named" not in load_manifest(tmp_path)


def test_install_notes_prompt_is_install_only() -> None:
    from llm_bawt.prompt_definitions import DEFAULT_PROMPT_DEFINITIONS

    definition = DEFAULT_PROMPT_DEFINITIONS["agents.install_notes"]
    assert definition.default_body() == ""
    assert classify(_row("agents.install_notes", "Never touch Zed.", category="install")) == INSTALL


def test_export_rejects_report_without_bodies(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        export_report(drift_report([_row("brand.new", "N")], {}), tmp_path, approved_keys=["brand.new"])


def test_body_path_rejects_traversal() -> None:
    with pytest.raises(ValueError):
        prompt_synced_defaults.body_path("../x")


# ── tenants follow code defaults ────────────────────────────────────────


@pytest.fixture
def store(monkeypatch):
    import llm_bawt.prompt_registry as registry

    defs = {"k.follow": _defn("k.follow", "v1"), "k.custom": _defn("k.custom", "v1"), "k.legacy": _defn("k.legacy", "v1")}
    monkeypatch.setattr(registry, "DEFAULT_PROMPT_DEFINITIONS", defs)
    s = PromptTemplateStore.__new__(PromptTemplateStore)
    s.config = None
    s.connection_url = "sqlite://follow"
    s.engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(s.engine, tables=[PromptTemplate.__table__, PromptTemplateVersion.__table__])
    return s, defs


def _bodies(store):
    with Session(store.engine) as session:
        return {row.key: row for row in session.exec(select(PromptTemplate)).all()}


def test_untouched_rows_follow_new_default_and_custom_rows_stay(store) -> None:
    s, defs = store
    assert s.seed_defaults()["created"] == 3
    seeded = _bodies(s)
    assert json.loads(seeded["k.follow"].metadata_json)[DEFAULT_SHA_KEY] == body_sha256("v1")

    s.upsert(key="k.custom", body="mine", metadata=json.loads(seeded["k.custom"].metadata_json), updated_by="nick")
    # A row seeded before hashes existed: no hash, still written only by the seed.
    with Session(s.engine) as session:
        legacy = session.exec(select(PromptTemplate).where(PromptTemplate.key == "k.legacy")).one()
        legacy.metadata_json = json.dumps({"seeded_from": "code_default"})
        session.add(legacy)
        session.commit()

    for key in defs:
        defs[key] = _defn(key, "v2")
    result = s.seed_defaults()
    assert sorted(result["followed_keys"]) == ["k.follow", "k.legacy"]
    rows = _bodies(s)
    assert rows["k.follow"].body == "v2" and rows["k.follow"].updated_by == "system:sync"
    assert rows["k.legacy"].body == "v2"
    assert rows["k.custom"].body == "mine"
    assert [v.change_note for v in s.list_versions("k.follow")][0] == "Followed updated code default"
    assert s.seed_defaults()["followed_keys"] == []


def test_followed_row_takes_the_new_required_vars_and_custom_rows_keep_theirs(store) -> None:
    s, defs = store
    for key in ("k.follow", "k.custom"):
        defs[key] = PromptDefinition(key=key, title=key, category="agents",
                                     loader=lambda: "{a} {b}", required_vars=("a", "b"))
    s.seed_defaults()
    seeded = _bodies(s)
    s.upsert(key="k.custom", body="mine {a} {b}", required_vars=["a", "b"],
             metadata=json.loads(seeded["k.custom"].metadata_json), updated_by="nick")
    for key in ("k.follow", "k.custom"):
        defs[key] = PromptDefinition(key=key, title=key, category="agents", loader=lambda: "{a}", required_vars=("a",))
    assert s.seed_defaults()["followed_keys"] == ["k.follow"]
    rows = _bodies(s)
    assert rows["k.follow"].body == "{a}" and json.loads(rows["k.follow"].required_vars_json) == ["a"]
    assert json.loads(rows["k.custom"].required_vars_json) == ["a", "b"]


def test_explicit_override_matching_new_default_stays_overridden_until_reset(store) -> None:
    s, defs = store
    s.seed_defaults()
    seeded_metadata = json.loads(_bodies(s)["k.custom"].metadata_json)
    saved = s.upsert(key="k.custom", body="tuned", updated_by="tenant", metadata=seeded_metadata)
    assert json.loads(saved.metadata_json)[DEFAULT_MODE_KEY] == "override"
    assert DEFAULT_SHA_KEY not in json.loads(saved.metadata_json)
    defs["k.custom"] = _defn("k.custom", "tuned")  # the sync shipped this edit
    assert s.seed_defaults()["adopted_keys"] == []
    defs["k.custom"] = _defn("k.custom", "tuned v2")
    assert "k.custom" not in s.seed_defaults()["followed_keys"]
    assert _bodies(s)["k.custom"].body == "tuned"

    # A promotion changes the product, not this tenant's explicit override.
    # Reset after publication uses the latest fallback immediately.
    from llm_bawt.prompt_registry import PromptResolver

    resolver = PromptResolver.__new__(PromptResolver)
    resolver.store = s
    assert s.reset("k.custom")
    assert resolver.resolve("k.custom").body == "tuned v2"
    assert resolver.resolve("k.custom").provenance == "product_default"
    assert resolver.resolve("k.custom").default_version == body_sha256("tuned v2")
    s.seed_defaults()
    for body in ("tuned v3", "tuned v4"):
        defs["k.custom"] = _defn("k.custom", body)
        assert "k.custom" in s.seed_defaults()["followed_keys"]
        assert resolver.resolve("k.custom").body == body
    assert s.seed_defaults()["followed_keys"] == []


def test_noop_save_cannot_forge_following_or_change_stored_portability(store) -> None:
    s, defs = store
    s.seed_defaults()
    s.set_sync_class("k.follow", INSTALL)
    metadata = json.loads(_bodies(s)["k.follow"].metadata_json)
    metadata.update({DEFAULT_MODE_KEY: "follow", "sync": PRODUCT})
    row = s.upsert(key="k.follow", body="v1", metadata=metadata, updated_by="system:seed")
    saved = json.loads(row.metadata_json)
    assert saved[DEFAULT_MODE_KEY] == "override"
    assert DEFAULT_SHA_KEY not in saved
    assert saved["sync"] == INSTALL
    defs["k.follow"] = _defn("k.follow", "v2")
    assert "k.follow" not in s.seed_defaults()["followed_keys"]
    assert _bodies(s)["k.follow"].body == "v1"


@pytest.mark.parametrize("writer", ["tenant", None])
def test_edited_and_ambiguous_legacy_equality_does_not_enable_following(store, writer) -> None:
    s, defs = store
    s.seed_defaults()
    with Session(s.engine) as session:
        row = session.exec(select(PromptTemplate).where(PromptTemplate.key == "k.legacy")).one()
        row.updated_by = writer
        # Old editors returned tracking metadata unchanged on a tenant save.
        row.metadata_json = json.dumps({DEFAULT_SHA_KEY: body_sha256("v1")})
        session.add(row)
        session.commit()
    assert s.seed_defaults()["adopted_keys"] == []
    defs["k.legacy"] = _defn("k.legacy", "v2")
    assert "k.legacy" not in s.seed_defaults()["followed_keys"]
    assert _bodies(s)["k.legacy"].body == "v1"


def test_scoped_override_survives_global_default_releases_and_reset(store) -> None:
    from llm_bawt.prompt_registry import PromptResolver

    s, defs = store
    s.seed_defaults()
    s.upsert(key="k.follow", body="bot text", scope_type="bot", scope_id="fixture-bot", updated_by="tenant")
    resolver = PromptResolver.__new__(PromptResolver)
    resolver.store = s
    for body in ("v2", "v3"):
        defs["k.follow"] = _defn("k.follow", body)
        s.seed_defaults()
        assert resolver.resolve("k.follow", "bot", "fixture-bot").body == "bot text"
    assert s.reset("k.follow", "bot", "fixture-bot")
    assert resolver.resolve("k.follow", "bot", "fixture-bot").body == "v3"


def test_synced_fixture_catalog_is_portable_and_matches_definitions() -> None:
    from llm_bawt.prompt_definitions import DEFAULT_PROMPT_DEFINITIONS
    from llm_bawt.prompt_promotion import validate_promotion_item

    for path in sorted(prompt_synced_defaults.SYNCED_DIR.glob("*.txt")):
        key = path.stem
        assert key in DEFAULT_PROMPT_DEFINITIONS, f"unregistered fixture {key}"
        definition = DEFAULT_PROMPT_DEFINITIONS[key]
        errors = validate_promotion_item({
            "key": key, "title": definition.title, "category": definition.category,
            "format": definition.format, "required_vars": list(definition.required_vars),
            "scope_type": "global", "scope_id": "*", "sync_class": PRODUCT,
            "body": path.read_text(encoding="utf-8"),
        })
        assert not errors, f"{key}: {errors}"
    # Validate the serialized manifest itself too: Python-defined keys may ignore
    # its metadata at runtime, but that must not hide malformed/private fixtures.
    for key, entry in load_manifest().items():
        assert set(entry) <= {"title", "category", "format", "required_vars"}
        path = prompt_synced_defaults.body_path(key)
        assert path.is_file(), f"manifest entry without body: {key}"
        errors = validate_promotion_item({
            **entry, "key": key, "scope_type": "global", "scope_id": "*",
            "sync_class": PRODUCT, "body": path.read_text(encoding="utf-8"),
        })
        assert not errors, f"manifest {key}: {errors}"


def test_fresh_repos_defaults_seed_render_and_fallback_use_one_body(monkeypatch) -> None:
    import llm_bawt.prompt_registry as registry

    definitions = {key: definition for key, definition in registry.DEFAULT_PROMPT_DEFINITIONS.items()
                   if key.startswith("repos.workspace.")}
    assert len(definitions) == 7
    assert sum(key.startswith("repos.workspace.commit_") for key in definitions) == 5
    monkeypatch.setattr(registry, "DEFAULT_PROMPT_DEFINITIONS", definitions)
    s = PromptTemplateStore.__new__(PromptTemplateStore)
    s.engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(s.engine, tables=[PromptTemplate.__table__, PromptTemplateVersion.__table__])
    assert s.seed_defaults()["created"] == 7
    assert s.seed_defaults()["created"] == 0
    resolver = registry.PromptResolver.__new__(registry.PromptResolver)
    resolver.store = s
    for key, definition in definitions.items():
        resolved = resolver.resolve(key)
        assert resolved.provenance == "product_default"
        assert resolved.body == prompt_synced_defaults.synced_body(key)
        variables = {name: f"fixture {name}" for name in definition.required_vars}
        assert "fixture scope" in resolver.render(key=key, variables=variables)
        assert s.reset(key)
        assert resolver.resolve(key).body == resolved.body
        assert resolver.resolve(key).default_version == resolved.default_version


def test_set_sync_class_changes_metadata_only(store) -> None:
    s, _ = store
    s.seed_defaults()
    row = s.set_sync_class("k.follow", INSTALL)
    assert classify(row) == INSTALL
    assert len(s.list_versions("k.follow")) == 1
    assert classify(s.set_sync_class("k.follow", None)) == PRODUCT
    assert s.set_sync_class("missing", INSTALL) is None


def test_health_check_degrades_only_on_stale_managed_defaults(monkeypatch) -> None:
    from datetime import datetime, timedelta, timezone
    from types import SimpleNamespace

    import llm_bawt.prompt_registry as registry
    from llm_bawt.service import health_checks

    now = datetime(2026, 11, 7, tzinfo=timezone.utc)
    rows = [_row("edited", "B2", updated_by="system:seed")]

    class FakeResolver:
        def __init__(self, _config):
            self.store = SimpleNamespace(engine=object(), list_rows=lambda: rows)

        @staticmethod
        def definitions():
            return {"edited": _defn("edited", "B")}

    monkeypatch.setattr(registry, "PromptResolver", FakeResolver)
    rows[0].updated_at = now - timedelta(hours=2)
    fresh = health_checks.check_prompt_sync(None, now)
    assert fresh["status"] == "ok"
    assert fresh["reason"] == ("1 managed prompt behind shipped defaults; "
                               "oldest default update: today. Prompts: edited")
    rows[0].updated_at = now - timedelta(hours=72)
    stale = health_checks.check_prompt_sync(None, now)
    assert stale["status"] == "degraded"
    assert "oldest default update: 3 days." in stale["reason"]
    rows[0].updated_at = now - timedelta(hours=30)
    assert "oldest default update: 1 day." in health_checks.check_prompt_sync(None, now)["reason"]
    # Old intentional edits never become unhealthy just because they differ.
    rows[:] = [_row("edited", "B2")]
    rows[0].updated_at = now - timedelta(days=90)
    assert health_checks.check_prompt_sync(None, now)["status"] == "ok"
    rows[0].body = "B"
    assert "tenant overrides preserved" in health_checks.check_prompt_sync(None, now)["reason"]


def test_save_committed_during_seed_pass_is_not_overwritten(store, monkeypatch, tmp_path) -> None:
    """Codex TASK-2127 blocker: seed read a following row, a save commits, follow must lose."""
    s, defs = store
    # File-backed SQLite so the interleaved save uses a real second connection.
    s.engine = create_engine(f"sqlite:///{tmp_path / 'race.db'}")
    SQLModel.metadata.create_all(s.engine, tables=[PromptTemplate.__table__, PromptTemplateVersion.__table__])
    s.seed_defaults()
    defs["k.follow"] = _defn("k.follow", "v2")

    original = PromptTemplateStore._follow_default

    def interleaved(session, row, body, sha, now, *rest):
        if row.key == "k.follow":
            other = PromptTemplateStore.__new__(PromptTemplateStore)
            other.engine = s.engine
            other.upsert(key="k.follow", body="TENANT CUSTOMIZATION", updated_by="tenant",
                         metadata=json.loads(row.metadata_json))
        return original(session, row, body, sha, now, *rest)

    monkeypatch.setattr(PromptTemplateStore, "_follow_default", staticmethod(interleaved))
    result = s.seed_defaults()
    assert "k.follow" not in result["followed_keys"]
    assert _bodies(s)["k.follow"].body == "TENANT CUSTOMIZATION"
    assert [v.body for v in s.list_versions("k.follow")][0] == "TENANT CUSTOMIZATION"


@pytest.mark.parametrize(("notes", "expected"), [
    ("Never touch Zed.\n", "Shared rules.\n\nNever touch Zed."),
    ("   ", "Shared rules."),
    (None, "Shared rules."),
])
def test_install_notes_follow_the_agent_global_prompt(notes, expected) -> None:
    from types import SimpleNamespace

    from llm_bawt.core.prompt_builder import PromptBuilder
    from llm_bawt.core.prompt_sections import PromptSectionsMixin

    bodies = {"agents.global_prompt": "Shared rules.", "agents.install_notes": notes}
    probe = SimpleNamespace(
        bot_id="b1",
        _resolve_setting=lambda name, default: True,
        config_resolver=SimpleNamespace(resolve_body=lambda key, **_: (
            SimpleNamespace(body=bodies[key], source="db") if bodies[key] is not None else None)),
    )
    builder = PromptBuilder()
    PromptSectionsMixin._sec_agent_global_prompt(probe, builder, "hi")
    assert builder.get_section("agent_global_prompt").content == expected
