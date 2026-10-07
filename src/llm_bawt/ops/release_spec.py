"""Operator-pinned spec for the durable BawtHub release operation (TASK-1030).

A ``release_orchestrate`` catalog row carries no command at all: every remote
identity (repositories, workflow, branches, image repository and the follow-up
deploy operation) is fixed in the reviewed spec, never in job arguments. Only
``executor_kind="release"`` may carry this spec, and that executor accepts no
other spec. Kept free of executor/coordinator imports to avoid a cycle.
"""
from __future__ import annotations

import re

RELEASE_ACTION = "release_orchestrate"
RELEASE_EXECUTOR_KIND = "release"

_REPOSITORY = r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*"
_BRANCH = r"[A-Za-z0-9][A-Za-z0-9_./-]{0,99}"
RELEASE_SPEC_PATTERNS = {
    "github_repository": _REPOSITORY,
    "workflow_path": r"\.github/workflows/[A-Za-z0-9_.-]+\.ya?ml",
    "canonical_branch": _BRANCH,
    "llm_bawt_repository": _REPOSITORY,
    "llm_bawt_branch": _BRANCH,
    "image_repository": r"ghcr\.io/[a-z0-9][a-z0-9._-]*/[a-z0-9][a-z0-9._/-]*",
    "deploy_operation": r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}",
}


def is_release_spec(spec: object) -> bool:
    return isinstance(spec, dict) and spec.get("action") == RELEASE_ACTION


def validate_release_spec(spec: dict) -> dict:
    """Exactly the pinned fields; ``..`` never allowed in a branch."""
    allowed = {"action", *RELEASE_SPEC_PATTERNS}
    if set(spec) != allowed:
        raise ValueError(f"release spec requires exactly {sorted(allowed)}")
    for key, pattern in RELEASE_SPEC_PATTERNS.items():
        value = spec[key]
        if not isinstance(value, str) or not re.fullmatch(pattern, value) or ".." in value:
            raise ValueError(f"release spec {key} is invalid")
    return spec


def workflow_file(spec_or_path: dict | str) -> str:
    """GitHub's workflow API takes the file name, not the repository path."""
    path = spec_or_path["workflow_path"] if isinstance(spec_or_path, dict) else spec_or_path
    return path.rsplit("/", 1)[-1]


__all__ = [
    "RELEASE_ACTION",
    "RELEASE_EXECUTOR_KIND",
    "RELEASE_SPEC_PATTERNS",
    "is_release_spec",
    "validate_release_spec",
    "workflow_file",
]
