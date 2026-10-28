"""The cutover probe line contract (TASK-1476 §3)."""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

from llm_bawt.env_import import SPECS
from llm_bawt.ops import probe_contract as pc

CONTRACT_FILE = Path(pc.__file__)


def _records(**overrides):
    records = {key: {"configured": True, "connected": True} for key in pc.RECORD_KEYS}
    records.update(overrides)
    return records


def _line(**fields):
    base = {"probe": "zai_completion", "ok": True, "status": 200, "code": "ok"}
    base.update(fields)
    return json.dumps(base)


def test_destination_probes_cover_exactly_the_env_import_destinations():
    assert set(pc.DESTINATION_PROBES) == {spec.destination for spec in SPECS}
    assert set(pc.DESTINATION_PROBES.values()) <= pc.PROBE_IDS
    assert pc.RECORD_KEYS == set(pc.DESTINATION_PROBES) | {"anthropic-api", "claude"}


def test_the_contract_imports_only_the_standard_library():
    tree = ast.parse(CONTRACT_FILE.read_text())
    imported = {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import)
                for alias in node.names}
    imported |= {node.module.split(".")[0] for node in ast.walk(tree)
                 if isinstance(node, ast.ImportFrom) and node.module}
    assert imported <= set(sys.stdlib_module_names) | {"__future__"}


@pytest.mark.parametrize("probe", sorted(pc.PROBE_IDS - {pc.RECORDS_PROBE}))
def test_emit_round_trips_every_probe(probe):
    line = pc.emit(probe, False, 401, pc.UNAUTHORIZED)
    assert "\n" not in line
    assert pc.parse_line(line) == {"probe": probe, "ok": False, "status": 401, "code": "unauthorized"}


def test_records_round_trip():
    records = _records(sonarr={"configured": True, "connected": False}, claude={"configured": False, "connected": False})
    parsed = pc.parse_line(pc.emit("records", True, None, pc.OK, records=records))
    assert parsed["records"] == records


def test_a_failed_records_probe_carries_no_records():
    assert pc.parse_line(pc.emit("records", False, None, pc.ERROR)) == {
        "probe": "records", "ok": False, "status": None, "code": "error"}
    with pytest.raises(ValueError):
        pc.emit("records", False, None, pc.ERROR, records=_records())
    with pytest.raises(ValueError):
        pc.emit("records", True, None, pc.OK)  # a successful one must carry them


def test_parse_line_reads_only_the_first_line():
    first = pc.emit("s3_head", True, 200, pc.OK)
    assert pc.parse_line(first + "\n" + _line(extra=1) + "\ngarbage")["probe"] == "s3_head"
    with pytest.raises(ValueError):
        pc.parse_line("garbage\n" + first)


@pytest.mark.parametrize("text", [
    pytest.param(_line(extra=1), id="extra-key"),
    pytest.param(json.dumps({"probe": "zai_completion", "ok": True, "status": 200}), id="missing-key"),
    pytest.param(_line(status=True), id="bool-status"),
    pytest.param(_line(status=99), id="status-below-100"),
    pytest.param(_line(status=600), id="status-above-599"),
    pytest.param(_line(status="200"), id="string-status"),
    pytest.param(_line(ok="true"), id="string-ok"),
    pytest.param(_line(ok=1), id="int-ok"),
    pytest.param(_line(code="okay"), id="unknown-code"),
    pytest.param(_line(code="timeout"), id="ok-with-failure-code"),
    pytest.param(_line(ok=False), id="failed-with-ok-code"),
    pytest.param(_line(probe="nope"), id="unknown-probe"),
    pytest.param(_line(probe=None), id="null-probe"),
    pytest.param(_line(records=_records()), id="records-on-non-records-probe"),
    pytest.param('{"probe":"zai_completion","probe":"zai_completion","ok":true,"status":200,"code":"ok"}',
                 id="duplicate-key"),
    pytest.param('{"probe":"zai_completion","ok":true,"status":NaN,"code":"ok"}', id="nan"),
    pytest.param("[1]", id="array"),
    pytest.param("not json", id="not-json"),
    pytest.param("", id="empty"),
    pytest.param("\n" + _line(), id="blank-first-line"),
])
def test_parse_line_rejects_off_schema_lines(text):
    with pytest.raises(ValueError):
        pc.parse_line(text)


@pytest.mark.parametrize("records", [
    pytest.param({key: value for key, value in _records().items() if key != "zai"}, id="missing-key"),
    pytest.param({**_records(), "openai": {"configured": True, "connected": True}}, id="extra-key"),
    pytest.param(_records(zai={"configured": True}), id="missing-field"),
    pytest.param(_records(zai={"configured": True, "connected": True, "value": "x"}), id="extra-field"),
    pytest.param(_records(zai={"configured": 1, "connected": True}), id="not-a-bool"),
    pytest.param(_records(zai={"configured": False, "connected": True}), id="connected-not-configured"),
    pytest.param([], id="not-a-dict"),
])
def test_records_must_match_the_record_keys_exactly(records):
    with pytest.raises(ValueError):
        pc.emit("records", True, None, pc.OK, records=records)
    line = json.dumps({"probe": "records", "ok": True, "status": None, "code": "ok", "records": records})
    with pytest.raises(ValueError):
        pc.parse_line(line)


def test_parse_line_rejects_non_text():
    with pytest.raises(ValueError):
        pc.parse_line(b"{}")  # type: ignore[arg-type]
