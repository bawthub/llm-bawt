"""One Compose service identity for every ops consumer (TASK-1169 S1b).

A Compose service is named by its project and service labels. Executor spec
validation, the config reconcilers and the service-health route all need the
same grammar and the same label selector, so they share this one definition
instead of private regex copies.

Stdlib only: the no-network ops worker image copies this file next to
``worker.py`` (``docker/Dockerfile.ops-worker``), and worker-side modules
import it the same way they import their other siblings.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

PROJECT_PATTERN = r"[a-z0-9][a-z0-9_-]{0,62}"
SERVICE_PATTERN = r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,62}"
# Spec field -> grammar, for spec validators that check fields by name.
SPEC_PATTERNS = {"compose_project": PROJECT_PATTERN, "compose_service": SERVICE_PATTERN}

PROJECT_LABEL = "com.docker.compose.project"
SERVICE_LABEL = "com.docker.compose.service"


@dataclass(frozen=True)
class ComposeServiceRef:
    """A validated ``(project, service)`` pair. Construction refuses any value
    outside the grammar, so a held ref is always safe to put in a label filter
    or a URL path."""

    project: str
    service: str

    def __post_init__(self) -> None:
        if not isinstance(self.project, str) or not re.fullmatch(PROJECT_PATTERN, self.project):
            raise ValueError("compose_project is invalid")
        if not isinstance(self.service, str) or not re.fullmatch(SERVICE_PATTERN, self.service):
            raise ValueError("compose_service is invalid")

    @classmethod
    def from_spec(cls, spec: dict, args: dict | None = None) -> "ComposeServiceRef":
        """The service a project+service spec targets: a fixed
        ``compose_service`` or the job argument ``compose_service_from_arg`` names."""
        if "compose_service" in spec:
            service = spec["compose_service"]
        elif "compose_service_from_arg" in spec:
            service = (args or {}).get(spec["compose_service_from_arg"])
        else:
            raise ValueError("spec has no compose service selector")
        return cls(spec.get("compose_project"), service)

    @classmethod
    def from_id(cls, service_id: str) -> "ComposeServiceRef":
        """Inverse of :attr:`service_id`. Unambiguous because a project never
        contains a dot."""
        project, dot, service = str(service_id).partition(".")
        if not dot:
            raise ValueError("service id must be <project>.<service>")
        return cls(project, service)

    @property
    def service_id(self) -> str:
        return f"{self.project}.{self.service}"

    def labels(self) -> dict[str, str]:
        return {PROJECT_LABEL: self.project, SERVICE_LABEL: self.service}

    def label_filters(self) -> dict[str, list[str]]:
        """Docker ``containers.list`` / ``/containers/json`` label filter."""
        return {"label": [f"{PROJECT_LABEL}={self.project}", f"{SERVICE_LABEL}={self.service}"]}

    def matches(self, labels: dict | None) -> bool:
        labels = labels or {}
        return labels.get(PROJECT_LABEL) == self.project and labels.get(SERVICE_LABEL) == self.service
