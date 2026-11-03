"""Produce batch-gate environment evidence for one source SHA (TASK-1865).

Operator tooling, run on the Docker host. It reads committed bytes with Git
and local image identities with ``docker image inspect``; it pulls, builds and
runs nothing. Output is the runtime-setting value for
``ops_gate_environment:llm-bawt:<sha>`` (stdout) and the evidence digest
(stderr). Writing that setting is a separate, approved operator step.

The closure digests are honest identities, not attestations:
- toolchain: the runner Dockerfile bytes (pinned base, uv, Node, payloads);
- dependency: the uv.lock bytes, which must equal the lock the image was
  built from (``--image-lock``), or the image cannot run that SHA offline;
- asset: the SHA256 pins of the tokenizer and MIME payloads in that Dockerfile.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

from llm_bawt.ops.gate_environment import PG_IMAGE

SCHEMA = "llm-bawt.ops.gate-environment/v1"
CONFIG_SCHEMA = "llm-bawt.ops.gate-environment-config/v1"
_PIN = re.compile(r'echo "([0-9a-f]{64})  (\S+)" \| sha256sum -c -')


def _git(repo: Path, *args: str) -> bytes:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True).stdout


def _inspect(ref: str) -> dict:
    out = subprocess.run(["docker", "image", "inspect", ref], check=True, capture_output=True).stdout
    return json.loads(out)[0]


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build(repo: Path, sha: str, runner_image: str, dockerfile: bytes, image_lock: bytes,
          inspect=_inspect) -> tuple[bytes, str]:
    sha = _git(repo, "rev-parse", "--verify", f"{sha}^{{commit}}").decode().strip()
    tree = _git(repo, "rev-parse", f"{sha}^{{tree}}").decode().strip()
    lock = _git(repo, "show", f"{sha}:uv.lock")
    if lock != image_lock:
        raise SystemExit("refused: the image's uv.lock differs from the source SHA's uv.lock")
    pins = sorted(_PIN.findall(dockerfile.decode()))
    if not pins:
        raise SystemExit("refused: no SHA256 payload pins in the runner Dockerfile")
    runner, pg = inspect(runner_image), inspect(PG_IMAGE)
    for ref, data in ((runner_image, runner), (PG_IMAGE, pg)):
        if ref not in (data.get("RepoDigests") or []):
            raise SystemExit(f"refused: {ref} is not a local repo digest")
    evidence = {
        "schema": SCHEMA, "repository": "llm-bawt", "source_sha": sha, "tree_sha": tree,
        "workload": "test-all/offline", "platform": "linux/amd64",
        "runner_image": runner_image, "runner_image_id": runner["Id"],
        "postgres_image": PG_IMAGE, "postgres_image_id": pg["Id"],
        "toolchain_sha256": _sha(dockerfile), "dependency_closure_sha256": _sha(lock),
        "asset_closure_sha256": _sha(json.dumps(pins, separators=(",", ":")).encode()),
    }
    raw = json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode()
    config = {"schema": CONFIG_SCHEMA, "evidence_json": raw.decode(), "evidence_sha256": _sha(raw)}
    return json.dumps(config, sort_keys=True, separators=(",", ":")).encode(), _sha(raw)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--sha", required=True)
    parser.add_argument("--runner-image", required=True, help="name@sha256:... local repo digest")
    parser.add_argument("--dockerfile", type=Path, required=True)
    parser.add_argument("--image-lock", type=Path, required=True, help="uv.lock the image was built from")
    args = parser.parse_args(argv)
    value, digest = build(args.repo, args.sha, args.runner_image,
                          args.dockerfile.read_bytes(), args.image_lock.read_bytes())
    sys.stdout.write(value.decode() + "\n")
    sys.stderr.write(f"evidence_sha256={digest}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
