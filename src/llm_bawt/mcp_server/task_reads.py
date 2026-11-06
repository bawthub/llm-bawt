"""Bounded task read projections and lossless, revision-checked detail pages.

Only task/project/review read endpoints are used. Approval continuation framing
is a separate protocol and is deliberately not involved in read pagination.
"""
from __future__ import annotations

import hashlib
import json
from typing import Annotated, Any, Awaitable, Callable
from urllib.parse import quote

import httpx
from pydantic import Field

from .task_api import NotJsonResponse

DEFAULT_BUDGET = 8000
MIN_BUDGET = 2000
MAX_BUDGET = 8000
RECENT_LIMIT = 3
ApiGet = Callable[..., Awaitable[Any]]
ReadPath = Annotated[str | None, Field(description="Detail JSON pointer under /task, /reviews or /project; omitted = compact summary.", max_length=500)]
ReadOffset = Annotated[int, Field(description="Character offset from next; continuation requires version.", ge=0)]
ReadBudget = Annotated[int, Field(description="Whole serialized reply budget; clamped to 2000-8000 characters.")]
ReadCursor = Annotated[str | None, Field(description="Opaque review-page cursor; only for /reviews paths.", max_length=512)]
ReadVersion = Annotated[str | None, Field(description="Revision from next; changed detail refuses continuation.", max_length=64)]


def _json(value: Any) -> str:
    # ASCII serialization bounds both UTF-8 bytes and escaped JSON characters.
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def _read(tool: str, task_id: str, **arguments: Any) -> dict:
    return {"tool": tool, "arguments": {"task_id": task_id, **arguments}}


def _preview(value: Any, path: str, size: int) -> Any:
    if not isinstance(value, str):
        return value
    if len(value) <= size:
        return value
    # Current response/blocker conclusions often live at the end, not the start.
    tail = size // 3
    return {"head": value[:size - tail], "tail": value[-tail:] if tail else "",
            "total_chars": len(value), "complete": False, "path": path}


def _row(row: dict, path: str, size: int, fields: tuple[str, ...]) -> dict:
    return {key: _preview(row[key], f"{path}/{key}", size)
            for key in fields if key in row}


def _steps(task: dict, size: int) -> dict:
    rows = task.get("steps") or []
    # Keep the running work and next pending action, even with a long completed
    # tail. Fill remaining slots with latest stored steps (API order is stable).
    selected: list[int] = []
    for status in ("RUNNING", "PENDING", "FAILED"):
        candidates = [i for i, row in enumerate(rows) if row.get("status") == status]
        if candidates:
            selected.append(candidates[-1] if status == "FAILED" else candidates[0])
    for index in reversed(range(len(rows))):
        if len(selected) >= RECENT_LIMIT:
            break
        if index not in selected:
            selected.append(index)
    statuses: dict[str, int] = {}
    for row in rows:
        status = row.get("status") or "UNKNOWN"
        statuses[status] = statuses.get(status, 0) + 1
    return {"total": len(rows), "shown": len(selected), "path": "/task/steps",
            "status_counts": statuses,
            "items": [{"index": i, **_row(rows[i], f"/task/steps/{i}", size,
                       ("id", "title", "status", "type", "output", "updatedAt"))}
                      for i in selected]}


def _summary(task: dict, reviews: Any, project: Any, size: int) -> dict:
    fields = ("id", "shortId", "status", "priority", "planned", "agentBotId",
              "modelId", "createdAt", "updatedAt")
    out = _row(task, "/task", 128, fields)
    out.update(_row(task, "/task", size, ("title", "blockedReason", "outcome")))
    # TASK-2115: a WAITING task's point is what it waits on and since when.
    if task.get("status") == "WAITING":
        out.update(_row(task, "/task", size, ("waitingOn", "waitingReason", "waitingSince")))
    for key in ("description", "response"):
        if key in task:
            out[key] = _preview(task[key], f"/task/{key}", size * 3)
    out["steps"] = _steps(task, max(40, size // 2))
    for key in ("dependsOn", "dependedOnBy", "checkLinks", "imageUrls"):
        rows = task.get(key) or []
        out[key] = {"total": len(rows), "path": f"/task/{key}",
                    "preview": _preview(_json(rows[:RECENT_LIMIT]), f"/task/{key}", size)}
    if isinstance(reviews, dict) and isinstance(reviews.get("reviews"), list):
        out["reviews"] = {"total": reviews.get("total"), "shown": min(len(reviews["reviews"]), RECENT_LIMIT),
                          "path": "/reviews", "next_cursor": reviews.get("nextCursor"),
                          "items": [{**_row(row, f"/reviews/reviews/{i}", max(40, size // 2),
                                         ("id", "stage", "verdict", "actorId", "signedVia", "targetRevision", "round", "body", "createdAt")),
                                     "findingCount": len(row.get("findings") or []),
                                     "openBlockers": sum(1 for finding in row.get("findings") or []
                                                         if finding.get("severity") == "BLOCKER" and finding.get("resolution") in (None, "OPEN"))}
                                    for i, row in enumerate(reviews["reviews"][:RECENT_LIMIT])]}
    else:
        out["reviews"] = {"unavailable": True, "path": "/reviews"}
    if isinstance(project, dict):
        out["project"] = _row(project, "/project", size,
                              ("id", "name", "description", "contextPrompt", "agentBotId", "unavailable"))
    elif project is not None:
        out["project"] = project
    return out


def _pointer(source: Any, parts: list[str]) -> Any:
    for part in parts:
        # RFC 6901 escaping, not arbitrary filesystem/API navigation.
        if "~" in part.replace("~0", "").replace("~1", ""):
            raise ValueError("Invalid JSON pointer escape")
        part = part.replace("~1", "/").replace("~0", "~")
        if isinstance(source, list):
            if not part.isascii() or not part.isdigit() or (part != "0" and part.startswith("0")):
                raise ValueError("Array paths require a zero-based index")
            source = source[int(part)]
        elif isinstance(source, dict):
            source = source[part]
        else:
            raise ValueError("Path does not name a task detail value")
    return source


def _page(source: Any, *, task_id: str, tool: str, path: str, offset: int,
          cursor: str | None, version: str | None, budget: int) -> dict:
    text = source if isinstance(source, str) else _json(source)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if version is not None and version != digest:
        return {"error": "detail_changed", "detail": "Restart this path at offset 0; do not join different revisions."}
    if offset > len(text):
        return {"error": "offset_out_of_range", "total_chars": len(text)}
    base = {"view": "detail", "path": path, "encoding": "text" if isinstance(source, str) else "json",
            "offset": offset, "total_chars": len(text), "version": digest}
    args = {"path": path, "max_chars": budget, "version": digest}
    if cursor is not None:
        args["cursor"] = cursor

    def build(count: int) -> dict:
        end = offset + count
        result = {**base, "text": text[offset:end], "next_offset": end if end < len(text) else None}
        if end < len(text):
            result["next"] = _read(tool, task_id, **args, offset=end)
        if path == "/reviews" and isinstance(source, dict) and source.get("nextCursor"):
            result["next_page"] = _read(tool, task_id, path=path, cursor=source["nextCursor"], max_chars=budget)
        return result

    # Budget the whole serialized envelope, not just the unescaped source text.
    low, high = 0, min(len(text) - offset, budget)
    while low < high:
        count = (low + high + 1) // 2
        if len(_json(build(count))) <= budget:
            low = count
        else:
            high = count - 1
    result = build(low)
    if len(_json(result)) > budget or low == 0 and offset < len(text):
        return {"error": "page_metadata_too_large", "detail": "Use a shorter path or cursor."}
    return result


async def read_task(api_get: ApiGet, task_id: str, *, tool: str, path: str | None = None,
                    offset: int = 0, max_chars: int = DEFAULT_BUDGET,
                    cursor: str | None = None, version: str | None = None,
                    include_project: bool = False) -> dict:
    budget = max(MIN_BUDGET, min(max_chars, MAX_BUDGET))
    if offset < 0:
        return {"error": "offset must be nonnegative"}
    if offset and version is None:
        return {"error": "offset requires the version from the first detail page"}
    if path is not None and (not path.startswith("/") or len(path) > 500):
        return {"error": "path must be a JSON pointer under /task, /reviews or /project (max 500 characters)"}
    if cursor is not None and len(cursor) > 512:
        return {"error": "cursor is too long"}
    if len(task_id) > 128:
        return {"error": "task_id is too long"}
    if path is None and (offset or cursor is not None or version is not None):
        return {"error": "offset, cursor and version require an explicit detail path"}
    parts = path.split("/")[1:] if path is not None else []
    if parts and parts[0] not in ("task", "project", "reviews"):
        return {"error": "path must start with /task, /reviews or /project"}
    if cursor is not None and parts[0] != "reviews":
        return {"error": "cursor is only valid for /reviews paths"}
    ref = quote(task_id, safe="")
    try:
        task = await api_get(f"/tasks/{ref}")
        if not isinstance(task, dict) or "error" in task:
            return {"error": "task_not_found"}
        project = task.get("project")
        reviews = None
        if path is None or parts[0] == "reviews":
            try:
                params = {"limit": str(RECENT_LIMIT)}
                if cursor is not None:
                    params["cursor"] = cursor
                reviews = await api_get(f"/tasks/{ref}/reviews", params=params)
            except (httpx.HTTPError, NotJsonResponse):
                if path is not None:
                    raise
        if (include_project and path is None or parts and parts[0] == "project") and project:
            try:
                project = await api_get(f"/projects/{quote(str(project['id']), safe='')}")
            except (httpx.HTTPError, NotJsonResponse):
                if path is not None:
                    raise
                project = {"unavailable": True, "id": project.get("id")}
        if path is not None:
            source = {"task": task, "reviews": reviews, "project": project}
            try:
                selected = _pointer(source, parts)
            except (KeyError, IndexError, ValueError, TypeError):
                return {"error": "detail_path_not_found"}
            return _page(selected, task_id=task_id, tool=tool, path=path, offset=offset,
                         cursor=cursor, version=version, budget=budget)
        # Only fixed-count projections enter the compact result. Reduce excerpts
        # to fit escaped data; omitted fields always carry their exact read path.
        for size in (500, 300, 160, 80, 40, 0):
            result = {"view": "summary", "task": _summary(task, reviews, project, size),
                      "detail": _read(tool, task_id, path="/task", max_chars=budget),
                      "paging": {"offset_unit": "characters", "max_chars": budget,
                                 "paths": ["/task", "/task/description", "/task/steps/0/output", "/reviews", "/project"]}}
            if len(_json(result)) <= budget:
                return result
        # A small requested budget still carries current owner/state/blocker and
        # response, not just a generic size error. History is explicitly omitted.
        for size in (100, 40, 20, 0):
            header = _row(task, "/task", 64, ("id", "shortId", "status", "priority", "agentBotId"))
            header.update(_row(task, "/task", size, ("title", "blockedReason", "response")))
            result = {"view": "summary", "task": header, "history_omitted": True,
                      "detail": _read(tool, task_id, path="/task", max_chars=budget),
                      "note": "Open detail paths for description, steps, reviews and project context."}
            if len(_json(result)) <= budget:
                return result
        return {"error": "summary_metadata_too_large", "detail": _read(tool, task_id, path="/task", max_chars=budget)}
    except httpx.HTTPStatusError as error:
        return {"error": "task_read_failed", "status": error.response.status_code}
    except (httpx.HTTPError, NotJsonResponse):
        return {"error": "task_read_unavailable"}
