"""Authoritative BawtHub task identity for working-memory pointers.

This read never confers task/turn association authority. Only the signed
association call can do that; manual links use this read solely to resolve the
user-supplied TASK-N or cuid to one immutable task identity.
"""
from __future__ import annotations

from dataclasses import dataclass

import httpx

from ..bawthub_auth import service_auth
from ..mcp_server.task_api import API_PREFIX, BASE_URL
from .contracts import WorkingMemoryError, normalize_task_ref


@dataclass(frozen=True)
class TaskInfo:
    id: str
    short_id: str
    title: str
    status: str
    assignee: str | None


class TaskCatalog:
    def get(self, ref: str) -> TaskInfo:
        ref = normalize_task_ref(ref)
        try:
            with httpx.Client(base_url=BASE_URL, timeout=2, auth=service_auth()) as client:
                response = client.get(f"{API_PREFIX}/tasks/{ref}")
                response.raise_for_status()
                row = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise WorkingMemoryError("task_unavailable", "task identity could not be verified") from exc
        if not isinstance(row, dict):
            raise WorkingMemoryError("task_unavailable", "task identity could not be verified")
        task_id, short_id, title, status = (row.get(k) for k in ("id", "shortId", "title", "status"))
        if (not isinstance(task_id, str) or not task_id or not isinstance(short_id, str)
                or not short_id.startswith("TASK-") or not isinstance(title, str)
                or not isinstance(status, str) or not status):
            raise WorkingMemoryError("task_unavailable", "task identity could not be verified")
        if ref not in (task_id, short_id):
            raise WorkingMemoryError("task_unavailable", "task identity does not match the requested reference")
        assignee = row.get("agentBotId")
        if assignee is not None and not isinstance(assignee, str):
            raise WorkingMemoryError("task_unavailable", "task assignee could not be verified")
        return TaskInfo(task_id, short_id, title, status, assignee)


__all__ = ["TaskCatalog", "TaskInfo"]
