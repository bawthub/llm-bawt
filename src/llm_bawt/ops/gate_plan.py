"""Closed, inert exact-commit gate workload contract (TASK-1865).

This is not an executor spec, authorization grant, retry admission or SHIP proof.
No production dispatcher currently constructs or executes a GatePlan.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

GATE_PLAN_SCHEMA = "llm-bawt.ops.gate-plan/v1"
_FIELDS = {"schema", "repository", "source_sha", "target"}


@dataclass(frozen=True)
class GatePlan:
    """Only the llm-bawt exhaustive offline workload is supported in v1.

    ``test-all`` means unit + private PostgreSQL + local integration, not a
    user-selected Make target. A future trusted adapter must pin/export source,
    toolchain, assets and sidecar separately, and publish verified host proof.
    """

    repository: Literal["llm-bawt"]
    source_sha: str
    target: Literal["test-all"] = "test-all"

    def __post_init__(self) -> None:
        if type(self.repository) is not str or self.repository != "llm-bawt":
            raise ValueError("unsupported gate repository; only llm-bawt is supported")
        if type(self.source_sha) is not str or re.fullmatch(r"[0-9a-f]{40}", self.source_sha) is None:
            raise ValueError("source_sha must be exactly 40 lowercase hexadecimal characters")
        if type(self.target) is not str or self.target != "test-all":
            raise ValueError("gate target is fixed to test-all")

    def to_dict(self) -> dict[str, str]:
        return {"schema": GATE_PLAN_SCHEMA, "repository": self.repository,
                "source_sha": self.source_sha, "target": self.target}

    @classmethod
    def from_dict(cls, value: object) -> GatePlan:
        """Decode stored JSON without coercions, extra fields or defaults."""
        if type(value) is not dict or value.keys() != _FIELDS:
            raise ValueError("gate plan must have exactly schema, repository, source_sha and target")
        if type(value["schema"]) is not str or value["schema"] != GATE_PLAN_SCHEMA:
            raise ValueError("unsupported gate plan schema")
        return cls(repository=value["repository"], source_sha=value["source_sha"], target=value["target"])
