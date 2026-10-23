"""Shared codex-bridge constants + helpers (TASK-555).

Relocated from ``bridge.py`` so the split-out mixin modules
(``session_ops``, ``command_ops``, ``event_ops``) can import them without a
cycle back through ``bridge``. ``bridge.py`` re-imports every name here.
"""

from __future__ import annotations

import asyncio
import logging

import httpx

logger = logging.getLogger("codex_bridge.bridge")

CONSUMER_GROUP = "codex-bridge"
CONSUMER_NAME = "worker-0"

# TASK-490: MCP tool context body. Canonical default lives in the app registry
# (agents.mcp_tool_context_template); the bridge cannot import llm_bawt, so it
# fetches the (overridable) body via GET /v1/prompts/{key} and falls back to
# this BYTE-IDENTICAL copy. The leading "\n\n" separator is added at append time.
MCP_TOOL_CONTEXT_KEY = "agents.mcp_tool_context_template"
_MCP_TOOL_CONTEXT_FALLBACK = (
    "## MCP Tool Context\n"
    "Your bot_id is \"{bot_slug}\". When using bawthub MCP tools:\n"
    "- Memory/message tools: always pass bot_id=\"{bot_slug}\"\n"
    "- Profile tool with entity_type=\"user\": use entity_id=\"nick\" (the user)\n"
    "- Profile tool with entity_type=\"bot\": use entity_id=\"{bot_slug}\" (yourself)\n"
    "- When you actually begin/resume an existing task in ordinary chat, call "
    "tasks_associate_current(task_id=...). Do not link from TASK-N text alone.\n"
    "- tasks_create(..., associate_current_turn=true) creates and links new work; "
    "tasks_update(..., associate_current_turn=true) claims/links existing work."
)


def system_prompt_for_attempt(
    system_prompt: str | None, delivery: str | None, *, resuming: bool,
) -> str | None:
    """The prompt to paste into this attempt's user input (TASK-1318).

    The codex thread history keeps every pasted copy, so a session-pinned
    prompt is pasted only when it takes effect: on a fresh thread (including
    the bridge's own reset and fresh-session retry) or when the app moved the
    pin this turn. Without a delivery mode the app did not pin, and the
    prompt goes out every turn as before.
    """
    from agent_bridge.command_publisher import (
        SYSTEM_PROMPT_DELIVERY_COLD_START,
        SYSTEM_PROMPT_DELIVERY_MODES,
    )

    if not system_prompt:
        return None
    if delivery == SYSTEM_PROMPT_DELIVERY_COLD_START and resuming:
        return None
    if delivery is not None and delivery not in SYSTEM_PROMPT_DELIVERY_MODES:
        logger.warning("Unknown system_prompt_delivery %r; pasting the prompt", delivery)
    return system_prompt


def _bot_slug_from_session_key(session_key: str) -> str:
    sk = (session_key or "").strip()
    if not sk:
        return ""
    return sk.split(":", 1)[0]


# --- TASK-204: failure detection helpers -----------------------------------


_AUTH_MARKERS = (
    "401",
    "403",
    "unauthorized",
    "authentication failed",
    "auth failure",
    "auth_failure",
    "invalid_grant",
    "invalid token",
    "refresh_token",
    "token has expired",
    "auth_mode=null",
    "authmode=null",
    "chatgpt token expired",
    "please run codex login",
)


def _is_auth_failure(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(marker in msg for marker in _AUTH_MARKERS)


# --- TASK-206: recoverable session errors ----------------------------------


_SESSION_ERROR_MARKERS = (
    "thread not found",
    "no rollout found",
    "missing required parameter: input",
    "encrypted_content",
    "thread_id is invalid",
)


def _is_codex_session_error(exc: BaseException) -> bool:
    if isinstance(exc, TimeoutError):
        return True
    msg = str(exc).lower()
    return any(marker in msg for marker in _SESSION_ERROR_MARKERS)


# --- TASK-208: model -> context window/max_output cache --------------------


class _ModelInfoCache:
    """One-shot lookup of context window + max output tokens.

    Polls /v1/models on the main app once at startup (and again after the
    cache misses for an unknown model). Keys are model ids like ``gpt-5.4``.
    """

    def __init__(self, app_api_url: str) -> None:
        self._app_api_url = app_api_url
        self._info: dict[str, dict[str, int | None]] = {}
        self._loaded = False
        self._lock = asyncio.Lock()

    async def _load(self) -> None:
        if not self._app_api_url:
            self._loaded = True
            return
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{self._app_api_url}/v1/models")
                resp.raise_for_status()
                payload = resp.json() or {}
                for entry in payload.get("data", []) or []:
                    if not isinstance(entry, dict):
                        continue
                    mid = entry.get("id") or entry.get("model")
                    if not mid:
                        continue
                    self._info[mid] = {
                        "context_window": (
                            entry.get("context_window")
                            or entry.get("contextWindow")
                            or entry.get("max_input_tokens")
                        ),
                        "max_output_tokens": (
                            entry.get("max_output_tokens")
                            or entry.get("maxOutputTokens")
                        ),
                    }
        except Exception as e:
            logger.debug("Model info preload failed: %s", e)
        self._loaded = True

    async def get(self, model: str) -> dict[str, int | None]:
        async with self._lock:
            if not self._loaded:
                await self._load()
            cached = self._info.get(model)
            if cached is not None:
                return cached
            # Try a refetch once — model list may have grown since startup.
            self._loaded = False
            await self._load()
            return self._info.get(model, {"context_window": None, "max_output_tokens": None})
