"""TASK-1534: activity_get pages task history by cursor and returns compact rows."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from llm_bawt.mcp_server import project_tools


def _row(i: int) -> dict[str, Any]:
    return {
        "id": f"a{i}", "type": "task.updated", "actorType": "bot", "actorId": "nova",
        "taskId": "cm1", "projectId": "p1", "meta": {"changes": [{"field": "description"}]},
        "createdAt": "2026-11-11T06:00:00.000Z",
        "task": {"id": "cm1", "shortId": "TASK-7", "title": "Long title", "status": "QUEUED"},
        "project": {"id": "p1", "name": "Core", "color": "#fff", "icon": None},
    }


def test_passes_task_ref_and_cursor_and_compacts_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[str, dict[str, str]]] = []

    async def fake_get(path: str, params: dict[str, str] | None = None, **_: Any) -> dict[str, Any]:
        seen.append((path, dict(params or {})))
        return {"activities": [_row(1), _row(2)], "total": 5, "limit": 2, "offset": None, "nextCursor": "c2"}

    monkeypatch.setattr(project_tools, "_api_get", fake_get)
    out = asyncio.run(project_tools.get_activity(task_id="TASK-7", limit=2, cursor="c1"))

    assert seen == [("/activity", {"limit": "2", "taskId": "TASK-7", "cursor": "c1"})]
    assert out["next_cursor"] == "c2"
    assert out["total"] == 5
    assert out["activities"][0] == {
        "id": "a1", "type": "task.updated", "actorType": "bot", "actorId": "nova",
        "meta": {"changes": [{"field": "description"}]}, "createdAt": "2026-11-11T06:00:00.000Z",
        "task": "TASK-7", "project": "Core",
    }


def test_limit_is_clamped_and_last_page_has_no_cursor(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[dict[str, str]] = []

    async def fake_get(path: str, params: dict[str, str] | None = None, **_: Any) -> dict[str, Any]:
        seen.append(dict(params or {}))
        return {"activities": [], "total": 0, "nextCursor": None}

    monkeypatch.setattr(project_tools, "_api_get", fake_get)
    out = asyncio.run(project_tools.get_activity(limit=500))
    assert seen[0]["limit"] == "100"
    assert "cursor" not in seen[0]
    assert out == {"activities": [], "total": 0, "next_cursor": None}
