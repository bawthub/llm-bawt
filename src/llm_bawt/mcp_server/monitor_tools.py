"""Monitor tools over BawtHub's health-check API (TASK-1712).

BawtHub owns monitor definitions, verdicts and wakes; these tools only read
and edit them through ``/api/system/health*`` with the service client.
PATCH replaces ``config`` wholesale, so mute reads, merges and writes.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from .registry import mcp
from .task_api import get_client, http_error, json_body

_CHECKS = "/api/system/health/checks"


async def _checks() -> list[dict]:
    return (json_body(await get_client().get(_CHECKS)) or {}).get("checks", [])


async def _find(monitor: str) -> dict | None:
    return next((c for c in await _checks() if monitor in (c.get("id"), c.get("slug"))), None)


@mcp.tool(name="monitors_list")
async def monitors_list() -> dict:
    """List monitor definitions (slug, kind, enabled, config) with their current verdicts."""
    try:
        checks = await _checks()
        health = json_body(await get_client().get("/api/system/health"))
    except httpx.HTTPStatusError as error:
        return http_error(error)
    verdicts = {c.get("slug"): c for c in (health or {}).get("checks", [])}
    return {"monitors": [
        {**c, "status": verdicts.get(c.get("slug"), {}).get("status"),
         "reason": verdicts.get(c.get("slug"), {}).get("reason")}
        for c in checks
    ]}


@mcp.tool(name="monitors_upsert")
async def monitors_upsert(definition: dict[str, Any], monitor: str | None = None) -> dict:
    """Create a monitor, or patch the one named by id or slug. config is replaced whole."""
    try:
        if monitor is None:
            return json_body(await get_client().post(_CHECKS, json=definition))
        found = await _find(monitor)
        if not found:
            return {"error": f"no monitor {monitor!r}"}
        return json_body(await get_client().patch(f"{_CHECKS}/{found['id']}", json=definition))
    except httpx.HTTPStatusError as error:
        return http_error(error)


@mcp.tool(name="monitors_mute")
async def monitors_mute(monitor: str, minutes: int = 60) -> dict:
    """Mute a monitor's alerts for minutes (0 unmutes); recovery notices still go out."""
    try:
        found = await _find(monitor)
        if not found:
            return {"error": f"no monitor {monitor!r}"}
        config = dict(found.get("config") or {})
        if minutes > 0:
            until = datetime.now(timezone.utc) + timedelta(minutes=minutes)
            config["mutedUntil"] = until.isoformat().replace("+00:00", "Z")
        else:
            config.pop("mutedUntil", None)
        return json_body(await get_client().patch(f"{_CHECKS}/{found['id']}", json={"config": config}))
    except httpx.HTTPStatusError as error:
        return http_error(error)


@mcp.tool(name="monitors_test")
async def monitors_test(monitor: str) -> dict:
    """Probe this monitor now (even if its interval isn't due) and return its verdict."""
    try:
        found = await _find(monitor)
        if not found:
            return {"error": f"no monitor {monitor!r}"}
        health = json_body(await get_client().get("/api/system/health", params={"fresh": "1", "probe": found.get("slug")}, timeout=35.0))
    except httpx.HTTPStatusError as error:
        return http_error(error)
    row = next((c for c in (health or {}).get("checks", []) if c.get("slug") == found.get("slug")), None)
    return row or {"error": f"monitor {monitor!r} produced no verdict"}
