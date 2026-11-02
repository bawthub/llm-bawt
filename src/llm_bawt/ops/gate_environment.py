"""Closed environment-evidence validation, not packaging attestation or authority."""
from __future__ import annotations

import hashlib
import json
import re

from .gate_plan import GatePlan

PG_IMAGE = "pgvector/pgvector@sha256:ccc6e83d6e35e931dc7c5def2022729d5a6c370318d099181995567ff1fb4d6b"
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_IMAGE = re.compile(r"[a-z0-9][a-z0-9./:_-]{0,200}@sha256:[0-9a-f]{64}\Z")
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_FIELDS = {"schema", "repository", "source_sha", "tree_sha", "workload", "platform",
    "runner_image", "runner_image_id", "postgres_image", "postgres_image_id",
    "toolchain_sha256", "dependency_closure_sha256", "asset_closure_sha256"}


class GateEnvironmentError(ValueError):
    """Fixed refusal codes; no submitted evidence in errors."""


def no_duplicates(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate key")
        value[key] = item
    return value


def validate_environment(raw: bytes, expected: str, plan: GatePlan, *, tree_sha: str | None = None) -> dict:
    """Bind exact bytes to an independently trusted digest and closed workload.

    Admission may freeze a configured tree before staging. The launcher MUST
    additionally pass its independently verified source tree. A matching digest
    does not authenticate the packager or establish offline/runtime parity.
    """
    if (not isinstance(raw, bytes) or len(raw) > 8192 or not isinstance(expected, str)
            or not _HEX.fullmatch(expected) or hashlib.sha256(raw).hexdigest() != expected):
        raise GateEnvironmentError("environment_evidence_unpinned")
    try:
        data = json.loads(raw, object_pairs_hook=no_duplicates)
    except (ValueError, UnicodeError, RecursionError):
        raise GateEnvironmentError("environment_evidence_invalid") from None
    if (not isinstance(data, dict) or data.keys() != _FIELDS
            or not all(isinstance(v, str) for v in data.values())
            or data["schema"] != "llm-bawt.ops.gate-environment/v1"
            or data["repository"] != plan.repository or data["source_sha"] != plan.source_sha
            or not re.fullmatch(r"[0-9a-f]{40}", data["tree_sha"])
            or (tree_sha is not None and data["tree_sha"] != tree_sha)
            or data["workload"] != "test-all/offline" or data["platform"] != "linux/amd64"
            or not _IMAGE.fullmatch(data["runner_image"]) or data["postgres_image"] != PG_IMAGE
            or any(not _DIGEST.fullmatch(data[k]) for k in ("runner_image_id", "postgres_image_id"))
            or any(not _HEX.fullmatch(data[k]) for k in
                ("toolchain_sha256", "dependency_closure_sha256", "asset_closure_sha256"))):
        raise GateEnvironmentError("environment_evidence_incompatible")
    return data
