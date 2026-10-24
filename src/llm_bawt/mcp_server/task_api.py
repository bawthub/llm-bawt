"""Shared BawtHub task API client and compact response helpers.

Kept separate from MCP registration modules so task/step/project tool groups stay
small and reuse one HTTP/error contract.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

from llm_bawt.utils.config import DEFAULT_TASK_API_URL

logger = logging.getLogger(__name__)

BASE_URL = os.getenv("LLM_BAWT_TASK_API_URL", DEFAULT_TASK_API_URL)
API_PREFIX = "/api/tasks"
TIMEOUT = 30.0

_client: httpx.AsyncClient | None = None


def count_by_status(items: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        status = str(item.get("status") or "UNKNOWN")
        counts[status] = counts.get(status, 0) + 1
    return counts


def compact_project(project: dict | None) -> dict | None:
    if not isinstance(project, dict):
        return None
    count = project.get("_count")
    task_count = count.get("tasks") if isinstance(count, dict) else None
    return {
        "id": project.get("id"),
        "name": project.get("name"),
        "agentBotId": project.get("agentBotId"),
        "taskCount": task_count,
    }


def compact_task(task: dict) -> dict:
    steps = task.get("steps") if isinstance(task.get("steps"), list) else []
    dependencies = task.get("dependencies") if isinstance(task.get("dependencies"), list) else []
    dependents = task.get("dependents") if isinstance(task.get("dependents"), list) else []
    description = task.get("description") or ""
    response = task.get("response") or ""
    compact = {
        "id": task.get("id"),
        "shortId": task.get("shortId"),
        "title": task.get("title"),
        "status": task.get("status"),
        "priority": task.get("priority"),
        "planned": task.get("planned"),
        "project": compact_project(task.get("project")),
        "agentBotId": task.get("agentBotId"),
        "createdAt": task.get("createdAt"),
        "updatedAt": task.get("updatedAt"),
        "url": task.get("url"),
        "descriptionChars": len(str(description)),
        "responseChars": len(str(response)),
        "stepCount": len(steps),
        "stepStatusCounts": count_by_status(steps),
        "dependencyCount": len(dependencies),
        "dependentCount": len(dependents),
    }
    # TASK-1368: a BLOCKED row is Nick's action item; its reason is the point.
    if task.get("status") == "BLOCKED":
        compact["blockedReason"] = task.get("blockedReason")
    # TASK-1422: a REVIEW row's "Where to look" pages are what Nick opens.
    if task.get("status") == "REVIEW" and task.get("checkLinks"):
        compact["checkLinks"] = task.get("checkLinks")
    # TASK-1380: a focused task carries its place in the user's ordered focus list.
    if task.get("focusRank") is not None:
        compact["focusRank"] = task.get("focusRank")
        compact["focusedAt"] = task.get("focusedAt")
    # TASK-1380: an archived task is finished and off the board.
    if task.get("archivedAt") is not None:
        compact["archivedAt"] = task.get("archivedAt")
    return compact


def compact_task_list_payload(payload: Any) -> Any:
    if not isinstance(payload, dict) or not isinstance(payload.get("tasks"), list):
        return payload
    compact = dict(payload)
    compact["tasks"] = [
        compact_task(task) if isinstance(task, dict) else task
        for task in payload["tasks"]
    ]
    return compact


def compact_project_list_payload(payload: Any) -> Any:
    if not isinstance(payload, list):
        return payload
    compact_projects = []
    for project in payload:
        if not isinstance(project, dict):
            compact_projects.append(project)
            continue
        count = project.get("_count")
        task_count = count.get("tasks") if isinstance(count, dict) else None
        compact_projects.append({
            "id": project.get("id"),
            "name": project.get("name"),
            "descriptionChars": len(str(project.get("description") or "")),
            "color": project.get("color"),
            "icon": project.get("icon"),
            "agentBotId": project.get("agentBotId"),
            "taskCount": task_count,
            "createdAt": project.get("createdAt"),
            "updatedAt": project.get("updatedAt"),
        })
    return compact_projects


def get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            base_url=BASE_URL,
            timeout=TIMEOUT,
            headers={"Content-Type": "application/json"},
        )
    return _client


def headers(bot_id: str | None = None) -> dict[str, str]:
    return {"X-Agent-Bot-Id": bot_id} if bot_id else {}


async def api_get(path: str, params: dict | None = None) -> Any:
    url = f"{API_PREFIX}{path}"
    logger.debug("Task API GET %s params=%s", url, params)
    resp = await get_client().get(url, params=params)
    resp.raise_for_status()
    return resp.json()


async def api_post(path: str, json: dict | list, headers: dict | None = None) -> Any:
    url = f"{API_PREFIX}{path}"
    logger.debug("Task API POST %s", url)
    resp = await get_client().post(url, json=json, headers=headers or {})
    resp.raise_for_status()
    return resp.json()


async def api_patch(path: str, json: dict, headers: dict | None = None) -> Any:
    url = f"{API_PREFIX}{path}"
    logger.debug("Task API PATCH %s", url)
    resp = await get_client().patch(url, json=json, headers=headers or {})
    resp.raise_for_status()
    return resp.json()


async def api_put(path: str, json: dict | list, headers: dict | None = None) -> Any:
    url = f"{API_PREFIX}{path}"
    logger.debug("Task API PUT %s", url)
    resp = await get_client().put(url, json=json, headers=headers or {})
    resp.raise_for_status()
    return resp.json()


async def api_delete(
    path: str,
    headers: dict | None = None,
    json: dict | list | None = None,
) -> Any:
    url = f"{API_PREFIX}{path}"
    logger.debug("Task API DELETE %s", url)
    client = get_client()
    if json is None:
        resp = await client.delete(url, headers=headers or {})
    else:
        resp = await client.request("DELETE", url, json=json, headers=headers or {})
    resp.raise_for_status()
    return resp.json()


def http_error(error: httpx.HTTPStatusError) -> dict:
    message = str(error)
    try:
        body = error.response.json()
    except Exception:
        body = None
    if isinstance(body, dict):
        server_message = body.get("error") or body.get("detail")
        if server_message:
            message = server_message if isinstance(server_message, str) else str(server_message)
    return {"error": message, "status": error.response.status_code}
