"""Opt-in, owner-authorized Tasks observation without browser credentials."""

from __future__ import annotations

import asyncio
import json
import os
from typing import Literal
from urllib.parse import urlsplit

import httpx
from mcp.server.mcpserver import Image

from ..bawthub_auth import service_auth
from ..inspection_admission import inspection_origin
from ..inspection_attestation import (
    INSPECTION_MAC_HEADER,
    INSPECTION_TASKS_PATH,
    inspection_key_available,
    sign_inspection_tasks,
)
from ..task_turn_context import TaskTurnContextError
from .task_association import current_task_turn_context


def register_inspection_tool(registry) -> None:
    """Leave the default MCP catalog unchanged without the tenant opt-in."""
    if os.getenv("BAWTHUB_INSPECTION_KEY_EXPORT_FILE", "").strip():
        registry.add_tool(inspect_bawthub_tasks, structured_output=False)

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_MAX_PNG = 2 * 1024 * 1024
_MAX_STATUS = 2048
_TIMEOUT = 22
_CONSENT_STATUSES = ("authorization_required", "wrong_origin", "expired", "revoked", "not_configured")


def _internal_base() -> str | None:
    raw = os.getenv("BAWTHUB_INSPECTION_INTERNAL_URL", "").strip()
    try:
        url = urlsplit(raw)
        if (url.scheme in ("http", "https") and url.hostname and url.port is not None
                and url.username is None and url.path in ("", "/") and not url.query and not url.fragment):
            return raw.rstrip("/")
    except ValueError:
        pass
    return None


async def _bounded_body(response: httpx.Response, limit: int) -> bytes | None:
    data = bytearray()
    async for chunk in response.aiter_raw():
        if len(chunk) > limit - len(data):
            return None
        data.extend(chunk)
    return bytes(data)


async def inspect_bawthub_tasks(viewport: Literal["phone", "desktop"] = "desktop") -> Image | dict[str, str]:
    """Read-only Tasks snapshot after owner authorization. Needs a recent signed-in owner chat turn; never returns a browser login."""
    base, origin = _internal_base(), inspection_origin()
    if not base or not origin or not inspection_key_available():
        return {"status": "not_configured"}
    try:
        turn = current_task_turn_context()
    except TaskTurnContextError:
        return {"status": "authorization_required"}
    if turn.inspection is not None and turn.inspection[1] != origin:
        return {"status": "wrong_origin"}
    signed = sign_inspection_tasks(turn, viewport)
    if signed is None:
        return {"status": "authorization_required"}
    body, signature = signed
    try:
        async with asyncio.timeout(_TIMEOUT), httpx.AsyncClient(
            base_url=base, auth=service_auth(), trust_env=False, follow_redirects=False,
        ) as client, client.stream(
            "POST", INSPECTION_TASKS_PATH, content=body,
            headers={"Content-Type": "application/json", INSPECTION_MAC_HEADER: signature},
        ) as response:
            if response.headers.get("Content-Encoding") is None:
                if response.status_code == 200 and response.headers.get("Content-Type") == "image/png":
                    raw = await _bounded_body(response, _MAX_PNG)
                    if raw and raw.startswith(_PNG_MAGIC):
                        return Image(data=raw, format="png")
                elif response.status_code == 403 and response.headers.get("Content-Type") == "application/json":
                    raw = await _bounded_body(response, _MAX_STATUS)
                    if raw:
                        payload = json.loads(raw)
                        if isinstance(payload, dict) and payload.get("status") in _CONSENT_STATUSES:
                            return {"status": payload["status"]}
    except (httpx.HTTPError, TimeoutError, ValueError, RecursionError):
        pass
    # All transport/auth/protocol failures remain opaque; never return headers or response text.
    return {"status": "unavailable"}
