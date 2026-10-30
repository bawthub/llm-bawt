"""TASK-1802: the seed exporter refuses unclassified production state (hermetic)."""

from __future__ import annotations

import json
from pathlib import Path

from llm_bawt.preview.seed_allowlist import COPY, SCHEMA_ONLY
from llm_bawt.preview.seed_export import copy_script, preflight, write_manifest


def _clean_tables() -> dict[str, list[str]]:
    tables = {table: list(rule.columns) for table, rule in COPY.items()}
    tables.update({table: ["id"] for table in SCHEMA_ONLY})
    return tables


def test_clean_production_passes():
    assert preflight(_clean_tables(), ["temperature"], ["ui_color"]) == []


def test_unclassified_state_refuses():
    tables = _clean_tables()
    tables["brand_new_table"] = ["id"]
    tables["weird_p_name"] = ["id"]  # name filtering must not hide real tables
    tables["models"] = [*tables["models"], "api_secret"]
    del tables["access_paths"]
    problems = preflight(tables, ["brand_new_global"], ["brand_new_bot"])
    assert "unclassified table: brand_new_table" in problems
    assert "unclassified table: weird_p_name" in problems
    assert "unclassified column: models.api_secret" in problems
    assert "missing copied table: access_paths" in problems
    assert "unclassified global setting: brand_new_global" in problems
    assert "unclassified bot setting: brand_new_bot" in problems


def test_copy_script_is_one_read_only_snapshot(tmp_path):
    script = copy_script(tmp_path).splitlines()
    assert script[0] == "BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY;"
    assert script[-1] == "COMMIT;"
    assert len(script) == len(COPY) + 2
    assert all(line.startswith("\\copy (SELECT ") and " TO '" in line for line in script[1:-1])
    assert not any(word in " ".join(script).upper() for word in ("INSERT ", "UPDATE ", "DELETE ", "TRUNCATE "))


def test_manifest_counts_quoted_multiline_rows(tmp_path: Path):
    for table in COPY:
        (tmp_path / f"{table}.csv").write_text('a,b\n1,"two\nlines"\n3,x\n')
    manifest = write_manifest(tmp_path)
    assert all(entry["rows"] == 2 for entry in manifest["tables"].values())
    assert json.loads((tmp_path / "manifest.json").read_text())["tables"].keys() == COPY.keys()
