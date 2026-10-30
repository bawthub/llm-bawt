"""TASK-1802: schema_init refuses a seed artifact that doesn't match its manifest (hermetic)."""

from __future__ import annotations

import json

import pytest

from llm_bawt.preview.schema_init import SeedInitError, verify_artifact
from llm_bawt.preview.seed_allowlist import COPY
from llm_bawt.preview.seed_export import write_manifest


@pytest.fixture
def seed(tmp_path):
    for table in COPY:
        (tmp_path / f"{table}.csv").write_text("a\n1\n")
    write_manifest(tmp_path)
    return tmp_path


def test_clean_artifact_verifies(seed):
    manifest, sha = verify_artifact(seed)
    assert set(manifest["tables"]) == set(COPY) and len(sha) == 64


def test_tampered_file_refuses(seed):
    (seed / "models.csv").write_text("a\n2\n")
    with pytest.raises(SeedInitError, match="sha256"):
        verify_artifact(seed)


def test_extra_or_drifted_table_refuses(seed):
    manifest = json.loads((seed / "manifest.json").read_text())
    manifest["tables"]["messages"] = manifest["tables"]["models"]
    (seed / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(SeedInitError, match="allowlist"):
        verify_artifact(seed)


def test_linked_file_refuses(seed):
    target = seed / "models.csv"
    real = seed / "real.csv"
    target.rename(real)
    target.symlink_to(real)
    with pytest.raises(SeedInitError, match="linked"):
        verify_artifact(seed)
