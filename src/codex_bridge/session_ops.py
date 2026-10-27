from __future__ import annotations

import asyncio
import base64
import binascii
import json
import logging
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from agent_bridge.events import AgentEvent, AgentEventKind, synthesize_event_id
from agent_bridge.publisher import COMMANDS_STREAM, RedisPublisher
from agent_bridge.session_queue import SessionQueue

from .transport import CodexTransport, validate_auth_json
from ._bridge_helpers import (
    CONSUMER_GROUP,
    CONSUMER_NAME,
    MCP_TOOL_CONTEXT_KEY,
    _MCP_TOOL_CONTEXT_FALLBACK,
    _ModelInfoCache,
    _bot_slug_from_session_key,
    _is_auth_failure,
    _is_codex_session_error,
)

logger = logging.getLogger("codex_bridge.bridge")


class CodexSessionMixin:
    """Codex session persistence + MCP tool context (TASK-555).

    Split out of ``CodexBridge`` (TASK-555). Composed back on via
    inheritance, so ``self.*`` state set in ``CodexBridge.__init__`` and
    methods on sibling mixins all resolve on the assembled instance.
    """

    async def _get_mcp_tool_context(self, bot_slug: str) -> str:
        """Return the MCP tool context block for a bot (TASK-490).

        Fetches the (bot-overridable) template body from the app registry via
        GET /v1/prompts/{key}, cached per bot, with a byte-identical local
        fallback if the app is unreachable. Returns the block WITHOUT the leading
        separator; the caller prepends ``\\n\\n``.
        """
        cache = getattr(self, "_mcp_ctx_cache", None)
        if cache is None:
            cache = {}
            self._mcp_ctx_cache = cache
        if bot_slug in cache:
            return cache[bot_slug]

        body = _MCP_TOOL_CONTEXT_FALLBACK
        if self._app_api_url:
            try:
                async with httpx.AsyncClient(timeout=5) as client:
                    resp = await client.get(
                        f"{self._app_api_url}/v1/prompts/{MCP_TOOL_CONTEXT_KEY}",
                        params={"scope_type": "bot", "scope_id": bot_slug},
                    )
                    resp.raise_for_status()
                    fetched = (resp.json() or {}).get("body")
                    if fetched:
                        body = fetched
            except Exception as e:
                logger.warning(
                    "MCP tool context fetch failed for %s (%s); using local fallback",
                    bot_slug, e,
                )
        rendered = body.replace("{bot_slug}", bot_slug)
        cache[bot_slug] = rendered
        return rendered

    async def _set_thread_session(
        self, thread_session_id: str, bot_id: str, sdk_session_id: str, model: str,
        *, required: bool = False,
    ) -> None:
        """Persist an SDK session id onto its bawthub thread.

        Writes to ``sessions.session_metadata.agent_session_keys`` via the
        app's ``PUT /v1/sessions/{id}/agent-session-key`` endpoint. This is
        the ONLY session-key write path (TASK-638); the scalar
        ``agent_backend_config.session_key`` is retired.
        """
        if not self._app_api_url:
            if required:
                raise RuntimeError("Codex reset provider binding requires the app API")
            logger.warning(
                "No API URL — thread session not persisted for %s/%s",
                bot_id, thread_session_id,
            )
            return
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.put(
                    f"{self._app_api_url}/v1/sessions/{thread_session_id}/agent-session-key",
                    params={"bot_id": bot_id},
                    json={
                        "backend": "codex",
                        "session_key": sdk_session_id,
                        "model": model,
                    },
                )
                resp.raise_for_status()
            logger.info(
                "Thread session persisted: %s thread=%s -> %s",
                bot_id, thread_session_id, sdk_session_id,
            )
        except Exception as e:
            if required:
                raise RuntimeError("Codex reset provider binding failed") from e
            logger.warning(
                "Failed to persist thread session for %s/%s: %s",
                bot_id, thread_session_id, e,
            )

    async def _record_continuity_receipt(
        self, thread_session_id: str | None, bot_id: str, messages: list[dict], *, restored: bool,
    ) -> None:
        """Confirm (or fail) the app's continuity manifest for this exact seed."""
        from agent_bridge.continuity import post_continuity_receipt

        await post_continuity_receipt(
            self._app_api_url, thread_session_id, bot_id, messages,
            restored=restored, logger=logger,
        )

    async def _bot_uses_codex(self, bot_id: str) -> bool:
        """Look up a bot's agent_backend; return True only if it's codex.

        Used to defensively skip legacy RPCs (no `backend` field on the
        message) that target bots owned by other bridges. Without this
        guard, the codex bridge would happily clear session_keys on
        claude-code / openclaw bots — a cross-backend interference bug.
        """
        if not self._app_api_url or not bot_id:
            return True
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{self._app_api_url}/v1/bots")
                resp.raise_for_status()
                for bot in resp.json().get("data", []):
                    if bot.get("slug") == bot_id:
                        return (bot.get("agent_backend") or "") == self._backend_name
        except Exception as e:
            logger.debug("agent_backend lookup failed for %s: %s", bot_id, e)
            return True
        # Bot not found — definitely not ours.
        return False
