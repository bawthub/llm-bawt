"""``bots_grant_set``: a grant manager grants or revokes a grant tag (TASK-1609).

A thin forwarder: the app's ``PUT /v1/bot-grants/{tag}/bots/{slug}`` opens the
signed turn capability and decides everything (``bot_tags.grants``). The
caller is never an argument, so a bot can only act as itself.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any
from urllib.parse import quote

from ..task_turn_context import TASK_TURN_CONTEXT_HEADER
from .registry import mcp
from .task_association import current_task_turn_capability

logger = logging.getLogger(__name__)

_APP_BASE_URL = os.getenv("LLM_BAWT_APP_BASE_URL", "http://localhost:8642").rstrip("/")
_TIMEOUT_S = 30.0


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def _error(error: str, message: str, **extra: Any) -> str:
    return _dump({"error": error, "message": message, **extra})


async def _put(path: str, body: dict[str, Any], capability: str):
    import httpx

    async with httpx.AsyncClient() as client:
        return await client.put(f"{_APP_BASE_URL}{path}", json=body, timeout=_TIMEOUT_S,
                                headers={TASK_TURN_CONTEXT_HEADER: capability})


@mcp.tool(name="bots_grant_set")
async def bots_grant_set(bot_id: str, tag: str, granted: bool, reason: str) -> str:
    """Grant or revoke a grant tag (such as approve_ops) on another bot.

    Only bots holding the configured grant-manager role may; never your own
    grants, never a private bot. Idempotent; a change is audited with the reason.

    Args:
        bot_id: The bot to grant or revoke.
        tag: The grant tag.
        granted: true grants, false revokes.
        reason: Why; stored on the audit row.
    """
    capability = current_task_turn_capability()
    if not capability:
        return _error("unauthorized", "a trusted turn capability is required")
    if not all(isinstance(v, str) and v.strip() for v in (bot_id, tag, reason)):
        return _error("invalid_request", "bot_id, tag and reason are required")
    path = f"/v1/bot-grants/{quote(tag.strip(), safe='')}/bots/{quote(bot_id.strip().lower(), safe='')}"
    try:
        response = await _put(path, {"granted": bool(granted), "reason": reason}, capability)
    except Exception:
        logger.exception("bots_grant_set: grant write failed for %s", bot_id)
        return _error("unavailable", "the write's outcome is unknown; check GET /v1/bot-grants")
    try:
        payload = response.json()
    except ValueError:
        payload = {"detail": response.text}
    if response.status_code >= 400:
        detail = payload.get("detail") if isinstance(payload, dict) else payload
        if isinstance(detail, dict) and detail.get("code"):
            return _error(str(detail["code"]), str(detail.get("message") or ""))
        return _error("rejected", str(detail), status_code=response.status_code)
    return _dump(payload)
