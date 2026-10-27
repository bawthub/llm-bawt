"""Home Assistant weather proxy route."""

import asyncio
import time
from typing import Any

import httpx
from fastapi import APIRouter, Query

from ...integrations.ha_credentials import HomeAssistantUnavailable, resolve_ha_credentials
from ..dependencies import get_service

router = APIRouter()

_cache: dict[str, Any] = {}
_cache_ts: float = 0.0
_CACHE_TTL = 900  # 15 minutes


def _ha_credentials(config: Any) -> tuple[str | None, str | None]:
    """Derive HA base URL and token from available config.

    Tries HA_NATIVE_MCP_URL first (strips /api/mcp suffix), then falls back
    to HA_MCP_URL (strips /api/mcp if present, otherwise uses as-is).
    """
    # The Integrations record first, then the deprecated native settings
    # (TASK-1476). A stored record that can't be used raises
    # HomeAssistantUnavailable and never falls through to env.
    credentials = resolve_ha_credentials(config)
    if credentials is not None:
        return credentials.base_url, credentials.token

    # Fall back to legacy MCP config
    mcp_url = getattr(config, "HA_MCP_URL", "") or ""
    mcp_token = getattr(config, "HA_MCP_AUTH_TOKEN", "") or ""
    native_token = getattr(config, "HA_NATIVE_MCP_TOKEN", "") or ""
    if mcp_url:
        base = mcp_url.rsplit("/api/mcp", 1)[0] if "/api/mcp" in mcp_url else mcp_url.rstrip("/")
        token = mcp_token or native_token
        if token:
            return base, token

    return None, None


@router.get("/v1/ha/weather", tags=["Home Assistant"])
async def get_ha_weather(entity_id: str = Query(default="weather.home")):
    """Fetch current weather state from Home Assistant."""
    global _cache, _cache_ts

    now = time.monotonic()
    cache_key = entity_id
    if cache_key in _cache and (now - _cache_ts) < _CACHE_TTL:
        return _cache[cache_key]

    service = get_service()
    try:
        base_url, token = await asyncio.to_thread(_ha_credentials, service.config)
    except HomeAssistantUnavailable as error:
        return {"error": f"Home Assistant connection unavailable: {error.code}"}

    if not base_url or not token:
        return {"error": "Home Assistant not configured"}

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                f"{base_url}/api/states/{entity_id}",
                headers={"Authorization": f"Bearer {token}"},
            )
            if resp.status_code != 200:
                return {"error": f"HA returned {resp.status_code} for {entity_id}"}
            data = resp.json()
    except Exception as exc:
        return {"error": str(exc)}

    attrs = data.get("attributes", {})
    result = {
        "state": data.get("state", "unknown"),
        "temperature": attrs.get("temperature"),
        "temperature_unit": attrs.get("temperature_unit", "°F"),
        "apparent_temperature": attrs.get("apparent_temperature"),
        "humidity": attrs.get("humidity"),
        "wind_speed": attrs.get("wind_speed"),
        "wind_speed_unit": attrs.get("wind_speed_unit"),
        "friendly_name": attrs.get("friendly_name", entity_id),
    }

    _cache[cache_key] = result
    _cache_ts = now
    return result
