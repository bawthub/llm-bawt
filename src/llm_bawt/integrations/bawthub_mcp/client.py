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

from ...task_turn_context import TASK_TURN_CONTEXT_HEADER
from ..ha_mcp.client import HomeAssistantNativeClient, run_bounded

logger = logging.getLogger(__name__)

DEFAULT_MCP_BASE_URL = "http://127.0.0.1:8001"
CATALOG_TTL_SECONDS = 300.0

# Conversational recall and memory only, each scoped by the server to the
# signed bot/user. Anything broader (profile, which can target any entity;
# tasks; messaging other bots; ops) must be named in the bot's allowlist.
DEFAULT_ALLOWLIST = frozenset({
    "memory_search", "memory_store", "memory_update", "memory_list_recent",
    "messages_find", "messages_context", "context_recall",
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

    def call_tool(self, capability: str | None, name: str, arguments: dict[str, Any]) -> str:
        if not self.allows(name):
            raise BawtHubMcpToolError(f"Tool '{name}' is not enabled for this bot")
        headers = self._headers(capability)
        try:
            result = run_bounded(self._call_async(headers, name, arguments), self._timeout)
        except Exception as error:
            logger.warning("BawtHub MCP call failed: %s: %s", name, error)
            raise BawtHubMcpToolError(f"Error calling {name}: {error}") from error
        text = HomeAssistantNativeClient._extract_text(result)
        if getattr(result, "isError", False) is True or getattr(result, "is_error", False) is True:
            raise BawtHubMcpToolError(text or f"{name} reported an error")
        return text

    # --- internals ---

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
                input_schema=getattr(t, "inputSchema", None) or {"type": "object", "properties": {}},
            )
            for t in raw
            if t.name in self._allowlist
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
