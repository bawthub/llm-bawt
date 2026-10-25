"""TASK-885: read-time, content-backed commit evidence; snapshots stay immutable.

The app has no Git mount. The internal BawtHub evidence service owns read-only
repository access. Unavailable evidence is explicitly unknown, never success.
"""
from __future__ import annotations

import logging
import os
from typing import Any

import httpx

from llm_bawt.bawthub_auth import service_auth

logger = logging.getLogger(__name__)


def reconcile_files(files: list[dict[str, Any]]) -> None:
    """Annotate serialized files in-place; no database or historical blob writes."""
    descriptors = []
    for index, file in enumerate(files):
        file["commit_state"] = "not_applicable" if file.get("in_repo") is False else "unknown"
        file["commit_reason"] = "outside_git" if file.get("in_repo") is False else "verification_unavailable"
        if file.get("in_repo") is not False:
            descriptors.append({"id": str(index), **{key: file.get(key) for key in (
                "path", "old_path", "created_at", "change_kind", "before_sha256",
                "after_sha256", "binary", "truncated", "in_repo",
            )}})
    if not descriptors:
        return
    # Prefer newest snapshots when a long conversation exceeds the read budget.
    descriptors = descriptors[-200:]
    url = os.getenv("BAWTHUB_COMMIT_EVIDENCE_URL", "http://frontend-prod:3002")
    try:
        # Only bounded metadata crosses this boundary, never snapshot contents.
        with httpx.Client(timeout=15, trust_env=False, auth=service_auth()) as client:
            for offset in range(0, min(len(descriptors), 200), 200):
                response = client.post(
                    f"{url.rstrip('/')}/internal/changed-files/commit-evidence",
                    json={"files": descriptors[offset:offset + 200]},
                )
                response.raise_for_status()
                evidence = response.json().get("evidence", {})
                for descriptor in descriptors[offset:offset + 200]:
                    item = evidence.get(descriptor["id"], {})
                    state = item.get("state")
                    if state not in {"committed", "pending", "unknown", "not_applicable"}:
                        continue
                    if state == "committed" and not all(item.get(key) for key in (
                        "commit_hash", "repo_id", "repo_path", "checked_head",
                    )):
                        continue
                    file = files[int(descriptor["id"])]
                    file.update(commit_state=state, commit_reason=item.get("reason"))
                    file["commit_evidence"] = item
    except (httpx.HTTPError, ValueError, TypeError, AttributeError):
        logger.debug("changed-file commit evidence unavailable", exc_info=True)


def pending_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Keep unknown actionable; requests do not prove commit completion."""
    files = [file for file in summary["files"] if file.get("commit_state") != "committed"
             and not file.get("ignored")
             and not (file.get("in_repo") is False and file.get("commit_requested"))]
    return {**summary, "files": files,
            "total_files": len(files),
            "total_additions": sum(f.get("additions") or 0 for f in files),
            "total_deletions": sum(f.get("deletions") or 0 for f in files)}
