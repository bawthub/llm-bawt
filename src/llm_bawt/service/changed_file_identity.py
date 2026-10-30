"""Workspace identity resolution for changed-file captures (TASK-1275 S2).

The bridge reports the raw absolute path plus git facts. BawtHub owns the
TrackedRepo/ManagedWorkspace registry, so identity is resolved here at
persist time through its internal ``/internal/repo-workspaces`` route. Any
failure yields an ``unresolved`` identity; the raw path is always kept.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

from llm_bawt.bawthub_auth import service_auth

logger = logging.getLogger(__name__)

UNRESOLVED_REPO_KEY = "workspace"


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


class WorkspaceIdentityResolver:
    """Map an absolute capture path to the contract-v3 identity fields."""

    def __init__(self, base_url: str | None = None, timeout: float = 2.0) -> None:
        self.base_url = (
            base_url
            or os.getenv("BAWTHUB_REPO_WORKSPACES_URL")
            or os.getenv("BAWTHUB_COMMIT_EVIDENCE_URL", "http://frontend-prod:3002")
        ).rstrip("/")
        self.timeout = timeout

    def resolve(self, absolute_path: str | None, bot_id: str | None) -> dict[str, Any]:
        """Return ``repo_key`` plus identity fields; never raises."""
        if not absolute_path or not absolute_path.startswith("/"):
            return self.unresolved()
        try:
            payload = self._fetch(absolute_path, bot_id)
            return self.from_payload(payload)
        except (httpx.HTTPError, ValueError, TypeError, AttributeError, KeyError) as exc:
            logger.debug("workspace identity unresolved path=%s: %s", absolute_path, exc)
            return self.unresolved()

    def _fetch(self, absolute_path: str, bot_id: str | None) -> dict[str, Any]:
        params = {"path": absolute_path}
        if bot_id:
            params["botId"] = bot_id
        with httpx.Client(timeout=self.timeout, trust_env=False, auth=service_auth()) as client:
            response = client.get(f"{self.base_url}/internal/repo-workspaces", params=params)
            response.raise_for_status()
            return response.json()

    @staticmethod
    def unresolved() -> dict[str, Any]:
        return {"repo_key": UNRESOLVED_REPO_KEY, "identity_status": "unresolved"}

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> dict[str, Any]:
        resolved = payload.get("resolved") if isinstance(payload, dict) else None
        if not isinstance(resolved, dict):
            return cls.unresolved()
        repo_id = resolved.get("repoId")
        workspace_id = resolved.get("workspaceId")
        kind = resolved.get("kind")
        relative = resolved.get("relativePath")
        # A malformed 200 is unresolved, never a half-resolved identity.
        if not (_nonempty(repo_id) and _nonempty(workspace_id) and _nonempty(relative)
                and kind in ("canonical", "isolated")):
            return cls.unresolved()
        repo_name = next(
            (r.get("name") for r in payload.get("repositories") or []
             if isinstance(r, dict) and r.get("id") == repo_id),
            None,
        )
        base_sha = None
        if kind == "isolated":
            base_sha = next(
                (w.get("baseSha") for w in payload.get("workspaces") or []
                 if isinstance(w, dict) and w.get("id") == workspace_id),
                None,
            )
        return {
            "repo_key": workspace_id,
            "identity_status": "resolved",
            "workspace_repo_id": repo_id,
            "workspace_repo_name": repo_name if isinstance(repo_name, str) else None,
            "workspace_id": workspace_id,
            "workspace_kind": kind,
            "repo_relative_path": relative,
            "git_base_sha": base_sha if isinstance(base_sha, str) else None,
        }
