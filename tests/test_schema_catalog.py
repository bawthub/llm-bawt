"""Deterministic schema-drift detection without a database connection."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def catalog_tool():
    return runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/schema_catalog.py"))


def _row(kind, key, definition):
    return json.dumps({"kind": kind, "key": key, "definition": definition})


def test_duplicate_keys_fail_closed(catalog_tool):
    record = _row("relation", "messages_p_long_truncated_name", {"partition_bound": "FOR VALUES IN ('long_full_identity')"})
    with pytest.raises(ValueError, match="duplicate catalog key"):
        catalog_tool["parse_catalog"]([record, record])


def test_semantic_drift_catches_predicate_width_nullability_and_missing_index(catalog_tool):
    parse, compare = catalog_tool["parse_catalog"], catalog_tool["compare"]
    reference = parse([
        _row("column", "bot_profiles.tts_mode", {"type": "boolean", "not_null": True, "default": "false", "position": 4}),
        _row("column", "tool_call_records.tool_use_id", {"type": "character varying", "not_null": False, "default": None, "position": 5}),
        _row("column", "entity_profiles.summary", {"type": "text", "not_null": False, "position": 9}),
        _row("index", "ix_inter_bot_delivery_dispatch", {"valid": True, "ready": True, "definition": "WHERE status IN ('QUEUED','STEERING','DISPATCHING')"}),
        _row("index", "idx_sessions_one_active_per_bot_user", {"valid": True, "ready": True, "definition": "UNIQUE NULLS NOT DISTINCT WHERE status='active'"}),
    ])
    operational = parse([
        _row("column", "bot_profiles.tts_mode", {"type": "boolean", "not_null": False, "default": None, "position": 17}),
        _row("column", "tool_call_records.tool_use_id", {"type": "character varying(128)", "not_null": False, "default": None, "position": 12}),
        _row("column", "entity_profiles.summary", {"type": "character varying", "not_null": False, "position": 8}),
        _row("index", "ix_inter_bot_delivery_dispatch", {"valid": True, "ready": True, "definition": "WHERE status IN ('QUEUED','DISPATCHING')"}),
    ])
    differences = compare(reference, operational)
    assert [(row["kind"], row["key"], row["state"]) for row in differences] == [
        ("column", "bot_profiles.tts_mode", "definition_mismatch"),
        ("column", "tool_call_records.tool_use_id", "definition_mismatch"),
        ("index", "idx_sessions_one_active_per_bot_user", "reference_only"),
        ("index", "ix_inter_bot_delivery_dispatch", "definition_mismatch"),
    ]
    assert compare(reference, reference) == []


def test_partition_identity_uses_bound_and_never_row_scans(catalog_tool):
    catalog = catalog_tool["parse_catalog"]([
        _row("relation", "messages_p_truncated", {"partition_bound": "FOR VALUES IN ('full_registered_slug')", "parents": ["messages"]}),
        _row("relation", "messages_p_mira", {"partition_bound": "FOR VALUES IN ('mira')", "parents": ["messages"]}),
    ])
    assert catalog[("relation", "messages_p_truncated")]["partition_bound"] != "FOR VALUES IN ('truncated')"
    assert "count(*) FROM ONLY" not in catalog_tool["CATALOG_SQL"]
    assert catalog_tool["compare"](catalog, catalog) == []


def test_operational_capture_uses_only_pgq_and_rejects_wrong_identity(catalog_tool):
    called = []
    run = catalog_tool["capture_live"].__globals__
    original = run["_run"]
    try:
        def fake(command, *, env):
            called.append((command, env["PGOPTIONS"]))
            return ["different|askllm|160015"]
        run["_run"] = fake
        with pytest.raises(RuntimeError, match="expected operational database/user"):
            catalog_tool["capture_live"]()
        assert len(called) == 1
        assert called[0][0][:2] == ["pgq", "llm-bawt"]
        assert "default_transaction_read_only=on" in called[0][1]
    finally:
        run["_run"] = original


def test_test_capture_rejects_unsafe_database_before_import(catalog_tool, monkeypatch):
    monkeypatch.setenv("LLM_BAWT_POSTGRES_DATABASE", "askllm")
    with pytest.raises(RuntimeError, match="explicit \\*_test database"):
        catalog_tool["capture_test"]("askllm")
