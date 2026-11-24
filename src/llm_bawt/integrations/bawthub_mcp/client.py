"""BawtHub MCP client for regular chat bots (TASK-2599).

Regular bots call the app's own MCP server under their own per-turn identity:
every request carries the signed task-turn context minted for this bot and
user, so the server applies that bot's grants, never the service principal's.
On top of the server grants, a per-bot allowlist limits which MCP tools the
model may discover or call.

The client is shared by a cached bot instance across turns, so it holds no
per-turn state: the capability is passed in on every call, and the set of
schemas a turn has loaded lives in the turn's ToolExecutor.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from agent_bridge.mcp_call_context import (
    MCP_CALL_CONTEXT_KEY,
    McpCallContextError,
    mint_mcp_call_context,
)

from ...task_turn_context import TASK_TURN_CONTEXT_HEADER
from ..ha_mcp.client import run_bounded

logger = logging.getLogger(__name__)

# Backend label in the per-call stamp for regular (non-agent) chat turns.
CALL_CONTEXT_BACKEND = "llm-bawt-chat"
# Tools the server holds for operator approval; see approval_interceptor.
APPROVAL_GATED_TOOLS = frozenset({"ops_run"})

DEFAULT_MCP_BASE_URL = "http://127.0.0.1:8001"
CATALOG_TTL_SECONDS = 300.0

# What a regular chat bot needs for memory, history and profiles; BawtHub MCP
# is their only source. The server scopes each call to the signed bot/user and
# the bot's grants. Anything broader (tasks, messaging other bots, ops) must be
# named in the bot's allowlist.
DEFAULT_ALLOWLIST = frozenset({
    "memory_search", "memory_store", "memory_update", "memory_delete",
    "memory_list_recent", "memory_search_all", "memory_list_sources",
    "memory_search_source", "messages_find", "messages_context",
    "messages_get_by_id", "context_recall", "profile",
})


class BawtHubMcpToolError(RuntimeError):
    """A BawtHub MCP call failed; the message is safe to show the model."""


@dataclass
class McpToolDefinition:
    """One MCP tool the server offered and the allowlist permits."""

    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)


def parse_allowlist(value: Any) -> frozenset[str] | None:
    """Tool names from a comma/space separated string or a list.

    None means the setting is absent; an explicitly empty value is an empty set.
    """
    if value is None:
        return None
    if not value:
        return frozenset()
    items = value if isinstance(value, (list, tuple, set, frozenset)) else str(value).replace(",", " ").split()
    return frozenset(str(item).strip() for item in items if str(item).strip())


class BawtHubMcpClient:
    """Allowlisted, turn-authenticated access to the app's MCP tools."""

    def __init__(self, server_url: str | None, allowlist: frozenset[str], *, timeout: float = 20.0):
        base = (server_url or DEFAULT_MCP_BASE_URL).rstrip("/")
        self._url = base if base.endswith("/mcp") else f"{base}/mcp"
        self._allowlist = frozenset(allowlist)
        self._timeout = timeout
        self._catalog: list[McpToolDefinition] = []
        self._catalog_at = 0.0

    @property
    def allowlist(self) -> frozenset[str]:
        return self._allowlist

    def allows(self, name: str) -> bool:
        return name in self._allowlist

    def search_tools(self, capability: str | None, query: str = "", limit: int = 5) -> list[McpToolDefinition]:
        """Allowlisted tools matching every query word in name or description.

        An empty query lists the allowlisted catalog (bounded by ``limit``).
        """
        catalog = self._load_catalog(capability)
        words = [w for w in (query or "").lower().split() if w]

        def score(tool: McpToolDefinition) -> int:
            name, text = tool.name.lower(), f"{tool.name} {tool.description}".lower()
            if not all(w in text for w in words):
                return -1
            return sum(2 if w in name else 1 for w in words)

        ranked = sorted(
            ((score(t), t) for t in catalog), key=lambda pair: (-pair[0], pair[1].name),
        )
        return [tool for s, tool in ranked if s >= 0][: max(1, limit)]

    def call_tool(
        self, capability: str | None, name: str, arguments: dict[str, Any], *, tool_use_id: str = "",
    ) -> str:
        if not self.allows(name):
            raise BawtHubMcpToolError(f"Tool '{name}' is not enabled for this bot")
        if name in APPROVAL_GATED_TOOLS:
            # A regular turn has no way to resume a held approval, so never
            # start one from here.
            raise BawtHubMcpToolError(f"{name} needs an approval that regular chat turns can't wait for")
        headers = self._headers(capability)
        stamped = {**arguments, MCP_CALL_CONTEXT_KEY: self._call_context(capability, name, arguments, tool_use_id)}
        try:
            result = run_bounded(self._call_async(headers, name, stamped), self._timeout)
        except Exception as error:
            logger.warning("BawtHub MCP call failed: %s: %s", name, error)
            raise BawtHubMcpToolError(f"Error calling {name}: {error}") from error
        text, structured = self._result_parts(result)
        body = text or self._serialize_structured(structured)
        # Policy refusals arrive as is_error inside the structured content.
        if (getattr(result, "isError", False) is True or getattr(result, "is_error", False) is True
                or (isinstance(structured, dict) and structured.get("is_error") is True)):
            raise BawtHubMcpToolError(body or f"{name} reported an error")
        return body or f"{name} returned no content"

    # --- result formatting ---

    _ABSENT = object()

    @classmethod
    def _result_parts(cls, result: Any) -> tuple[str, Any]:
        """Nonblank joined text items, and the structured payload or _ABSENT.

        Never falls back to the SDK object's repr. An explicitly empty dict or
        list is a real (empty) payload, distinct from an absent one.
        """
        parts = []
        for item in getattr(result, "content", None) or []:
            item_text = getattr(item, "text", None)
            if isinstance(item_text, str) and item_text.strip():
                parts.append(item_text)
        text = "\n".join(parts).strip()
        structured = cls._ABSENT
        for attr in ("structured_content", "structuredContent"):
            value = getattr(result, attr, None)
            if value is not None:
                structured = value
                break
        return text, structured

    @classmethod
    def _serialize_structured(cls, structured: Any) -> str:
        if structured is cls._ABSENT:
            return ""
        import json

        try:
            return json.dumps(structured, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return ""

    # --- internals ---

    @staticmethod
    def _call_context(capability: str | None, name: str, arguments: dict[str, Any], tool_use_id: str) -> dict[str, Any]:
        """Signed stamp binding this provider tool call to its final arguments.

        The request identity comes from the server-minted turn capability, never
        from the model. The stamp travels only in the MCP request; callers keep
        the unstamped arguments for events, persistence and logs.
        """
        from ...task_turn_context import TaskTurnContextError, open_task_turn_context

        if not capability:
            raise BawtHubMcpToolError("BawtHub tools need a signed turn identity, which this turn lacks")
        if not str(tool_use_id or "").strip():
            raise BawtHubMcpToolError("BawtHub tools need the provider's tool call id, which this call lacks")
        try:
            turn = open_task_turn_context(capability)
            return mint_mcp_call_context(
                capability=capability, tool_name=name, tool_input=arguments,
                tool_use_id=tool_use_id, agent_request_id=turn.turn_id,
                session_key=turn.session_id, backend=CALL_CONTEXT_BACKEND,
            )
        except (TaskTurnContextError, McpCallContextError) as error:
            raise BawtHubMcpToolError(f"This turn's signed identity is unusable: {error}") from error

    @staticmethod
    def _headers(capability: str | None) -> dict[str, str]:
        if not capability:
            # Never fall back to an anonymous or service identity.
            raise BawtHubMcpToolError("BawtHub tools need a signed turn identity, which this turn lacks")
        # The signed turn alone authenticates at the :8001 gate, which refuses
        # a turn sent together with a service key as ambiguous.
        return {TASK_TURN_CONTEXT_HEADER: capability}

    def _load_catalog(self, capability: str | None) -> list[McpToolDefinition]:
        headers = self._headers(capability)  # identity first, even on a warm cache
        now = time.monotonic()
        if self._catalog and now - self._catalog_at < CATALOG_TTL_SECONDS:
            return self._catalog
        try:
            raw = run_bounded(self._list_async(headers), self._timeout)
        except Exception as error:
            logger.warning("BawtHub MCP discovery failed: %s", error)
            raise BawtHubMcpToolError(f"BawtHub tools are unavailable: {error}") from error
        self._catalog = [
            McpToolDefinition(
                name=t.name,
                description=t.description or "",
                # mcp SDK 2.x names it input_schema; 1.x used inputSchema.
                input_schema=(getattr(t, "input_schema", None) or getattr(t, "inputSchema", None)
                              or {"type": "object", "properties": {}}),
            )
            for t in raw
            if t.name in self._allowlist and t.name not in APPROVAL_GATED_TOOLS
        ]
        self._catalog_at = now
        return self._catalog

    @asynccontextmanager
    async def _session(self, headers: dict[str, str]):
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client
        from mcp.shared._httpx_utils import create_mcp_http_client

        async with create_mcp_http_client(headers=headers) as http_client:
            async with streamable_http_client(self._url, http_client=http_client) as (read, write, *_):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    yield session

    async def _list_async(self, headers: dict[str, str]):
        async with self._session(headers) as session:
            return (await session.list_tools()).tools

    async def _call_async(self, headers: dict[str, str], name: str, arguments: dict[str, Any]):
        async with self._session(headers) as session:
            return await session.call_tool(name, arguments)
