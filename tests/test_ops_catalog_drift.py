"""Seed-versus-live ops catalog drift report (TASK-1861): read-only, nested, classified."""
from __future__ import annotations

import json

from llm_bawt.ops import catalog_drift
from llm_bawt.ops.seeds import SEEDS, seed_all
from tests.test_ops_store import _store

SLUG = SEEDS[0]["slug"]
#: A seed whose script carries restart-health keys: TASK-1801's real gap.
HEALTH_SLUG = next(seed["slug"] for seed in SEEDS
                   if any(key.startswith("health_") for key in json.loads(seed["command_script"])))


def _script(store, slug=SLUG):
    return json.loads(store.get_operation_by_slug(slug).command_script)


def _entry(result, slug=SLUG):
    return next(e for e in result["operations"] if e["slug"] == slug)


def _catalog(store):
    return [(r.slug, r.version, r.script_hash, r.updated_by, r.enabled)
            for r in store.list_operations(include_disabled=True, include_soft_deleted=True)]


def test_a_freshly_seeded_catalog_has_no_drift():
    store = _store()
    seed_all(store)
    result = catalog_drift.report(store)
    assert result["counts"] == {"clean": len(SEEDS)}
    assert result["live_only"] == [] and result["rows_missing_seeded_keys"] == 0


def test_a_seed_key_the_live_row_lacks_is_reported_missing_and_nothing_is_written():
    store = _store()
    seed_all(store)
    script = _script(store, HEALTH_SLUG)
    dropped = sorted(key for key in script if key.startswith("health_"))
    store.update_operation(HEALTH_SLUG, {"command_script": json.dumps(
        {k: v for k, v in script.items() if k not in dropped})}, actor="nick")
    before = _catalog(store)

    entry = _entry(catalog_drift.report(store), HEALTH_SLUG)

    assert entry["status"] == "operator_override" and entry["updated_by"] == "nick"
    assert entry["missing_paths"] == [f"command_script.{key}" for key in dropped]
    assert _catalog(store) == before  # read-only: no version, hash or actor moved


def test_a_changed_nested_value_names_its_path_and_both_values():
    seed = {"slug": "x", "command_script": json.dumps({"health": {"timeout": 30, "url": "u"}})}
    live = {"slug": "x", "updated_by": "system-seed", "version": 1,
            "command_script": json.dumps({"health": {"timeout": 60, "url": "u"}, "extra": 1})}

    entry = _entry(catalog_drift.compare([seed], [live]), "x")

    assert entry["status"] == "seed_changed"
    assert entry["differences"] == [
        {"path": "command_script.extra", "kind": "added", "live": 1},
        {"path": "command_script.health.timeout", "kind": "changed", "seed": 30, "live": 60},
    ]
    assert entry["missing_paths"] == []


def test_installer_rows_absent_rows_and_unseeded_rows_are_classified_apart():
    seeds = [{"slug": "a", "title": "A"}, {"slug": "b", "title": "B"}, {"slug": "c", "title": "C"}]
    live = [{"slug": "a", "title": "A2", "updated_by": "tenant-install"},
            {"slug": "c", "title": "C", "enabled": True, "updated_by": "nick"},
            {"slug": "local.extra", "title": "E"}]

    result = catalog_drift.compare(seeds, live)

    assert [(e["slug"], e["status"]) for e in result["operations"]] == [
        ("a", "install_override"), ("b", "missing"), ("c", "clean")]  # enabled is not drift
    assert result["live_only"] == ["local.extra"]
