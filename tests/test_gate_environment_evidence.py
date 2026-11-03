"""TASK-1865: the evidence producer emits a config the trust reader accepts."""
from __future__ import annotations

import json
import subprocess

import pytest

from llm_bawt.ops.gate_environment import PG_IMAGE, validate_environment
from llm_bawt.ops.gate_plan import GatePlan
from scripts import gate_environment_evidence as cli

RUNNER = "llm-bawt-gate-runner@sha256:" + "a" * 64
DOCKERFILE = b'RUN echo "' + b"b" * 64 + b'  /x" | sha256sum -c -\n'


def _repo(tmp_path, lock=b"lock-v1"):
    def git(*a):
        return subprocess.run(["git", "-C", str(tmp_path), *a], check=True, capture_output=True).stdout
    git("init", "-q")
    (tmp_path / "uv.lock").write_bytes(lock)
    git("add", "uv.lock")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "x")
    return git("rev-parse", "HEAD").decode().strip()


def _inspect(ref):
    return {"Id": "sha256:" + ("1" if ref == RUNNER else "2") * 64, "RepoDigests": [ref]}


def test_config_round_trips_through_validation(tmp_path):
    sha = _repo(tmp_path)
    raw, digest = cli.build(tmp_path, sha, RUNNER, DOCKERFILE, b"lock-v1", inspect=_inspect)
    config = json.loads(raw)
    plan = GatePlan(repository="llm-bawt", source_sha=sha)
    evidence = validate_environment(config["evidence_json"].encode(), config["evidence_sha256"], plan)
    assert digest == config["evidence_sha256"] and evidence["postgres_image"] == PG_IMAGE


def test_refuses_mismatched_lock_and_missing_digest(tmp_path):
    sha = _repo(tmp_path)
    with pytest.raises(SystemExit):
        cli.build(tmp_path, sha, RUNNER, DOCKERFILE, b"other", inspect=_inspect)
    with pytest.raises(SystemExit):
        cli.build(tmp_path, sha, RUNNER, DOCKERFILE, b"lock-v1",
                  inspect=lambda ref: {"Id": "sha256:" + "1" * 64, "RepoDigests": []})
    with pytest.raises(SystemExit):
        cli.build(tmp_path, sha, RUNNER, b"FROM scratch\n", b"lock-v1", inspect=_inspect)

