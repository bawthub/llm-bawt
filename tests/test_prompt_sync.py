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
    assert classify(_row("persona.x", "x", category="persona", metadata={"sync": "product"})) == PRODUCT


def test_drift_covers_changed_and_new_product_rows_only() -> None:
    defs = {"same": _defn("same", "A"), "edited": _defn("edited", "B"), "stale": _defn("stale", "new")}
    rows = [
        _row("same", "A"),
        _row("edited", "B2"),
        _row("stale", "old", updated_by="system:seed"),  # untouched seed row: follows, not drift
        _row("brand.new", "N"),
        _row("persona.me", "P", category="persona"),
        _row("local", "L", metadata={"sync": "install"}),
    ]
    drift = {item.key: item.kind for item in compute_drift(rows, defs)}
    assert drift == {"edited": "changed", "brand.new": "new"}
    report = drift_report(rows, defs)
    assert report["in_sync"] is False and report["drift_count"] == 2
    assert report["classified"] == {PRODUCT: 4, INSTALL: 2}
    assert "body" not in report["drift"][0]


def test_is_following_uses_hash_then_seed_actor() -> None:
    assert is_following(_row("k", "x", updated_by="system:seed"))
    assert not is_following(_row("k", "x", updated_by="snark"))
    assert is_following(_row("k", "x", updated_by="snark", metadata={DEFAULT_SHA_KEY: body_sha256("x")}))
    assert not is_following(_row("k", "y", updated_by="system:seed", metadata={DEFAULT_SHA_KEY: body_sha256("x")}))


def test_export_writes_bodies_and_manifest(tmp_path: Path) -> None:
    defs = {"edited": _defn("edited", "B")}
    rows = [_row("edited", "B2"), _row("brand.new", "N {thing}", category="repos")]
    rows[1].required_vars_json = json.dumps(["thing"])
    result = export_report(drift_report(rows, defs, include_body=True), tmp_path)
    assert result["written"] == ["brand.new", "edited"]
    assert (tmp_path / "edited.txt").read_text() == "B2"
    assert (tmp_path / "brand.new.txt").read_text() == "N {thing}"
    manifest = load_manifest(tmp_path)
    assert set(manifest) == {"brand.new"}
    assert manifest["brand.new"]["category"] == "repos"
    assert manifest["brand.new"]["required_vars"] == ["thing"]


def test_export_rejects_report_without_bodies(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        export_report(drift_report([_row("brand.new", "N")], {}), tmp_path)


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


def test_customized_row_matching_new_default_starts_following(store) -> None:
    s, defs = store
    s.seed_defaults()
    s.upsert(key="k.custom", body="tuned", updated_by="snark")
    defs["k.custom"] = _defn("k.custom", "tuned")  # the sync shipped this edit
    assert s.seed_defaults()["adopted_keys"] == ["k.custom"]
    defs["k.custom"] = _defn("k.custom", "tuned v2")
    assert s.seed_defaults()["followed_keys"] == ["k.custom"]
    assert _bodies(s)["k.custom"].body == "tuned v2"


def test_set_sync_class_changes_metadata_only(store) -> None:
    s, _ = store
    s.seed_defaults()
    row = s.set_sync_class("k.follow", INSTALL)
    assert classify(row) == INSTALL
    assert len(s.list_versions("k.follow")) == 1
    assert classify(s.set_sync_class("k.follow", None)) == PRODUCT
    assert s.set_sync_class("missing", INSTALL) is None


def test_health_check_degrades_only_on_stale_drift(monkeypatch) -> None:
    from datetime import datetime, timedelta, timezone
    from types import SimpleNamespace

    import llm_bawt.prompt_registry as registry
    from llm_bawt.service import health_checks

    now = datetime(2026, 11, 7, tzinfo=timezone.utc)
    rows = [_row("edited", "B2")]

    class FakeResolver:
        def __init__(self, _config):
            self.store = SimpleNamespace(engine=object(), list_rows=lambda: rows)

        @staticmethod
        def definitions():
            return {"edited": _defn("edited", "B")}

    monkeypatch.setattr(registry, "PromptResolver", FakeResolver)
    rows[0].updated_at = now - timedelta(hours=2)
    fresh = health_checks.check_prompt_sync(None, now)
    assert fresh["status"] == "ok" and "edited" in fresh["reason"]
    rows[0].updated_at = now - timedelta(hours=72)
    assert health_checks.check_prompt_sync(None, now)["status"] == "degraded"
    rows[0].body = "B"
    assert health_checks.check_prompt_sync(None, now)["reason"] == "product prompts match shipped defaults"
