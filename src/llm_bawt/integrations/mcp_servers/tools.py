"""Granted remote MCP servers as chat tools for regular bots (TASK-2610).

A tool-using regular bot reaches the remote MCP servers granted to it (or to
one of its tags) through the same deferred ``tool_search`` discovery as the
BawtHub MCP tools: a search loads matching ``<slug>__<tool>`` schemas into
the following rounds, and a call goes to that server with a fresh credential.

The bot-level :class:`RemoteMcpTools` is cached on the bot instance, so it
keeps only tool catalogs (keyed by server and URL, never credentials). Each
turn gets a :class:`RemoteMcpTurn`, which resolves the bot's servers and their
access tokens at most once, and only if the model actually uses them.
"""
from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Any

from ..bawthub_mcp.client import BawtHubMcpClient, McpToolDefinition
from ..ha_mcp.client import run_bounded
from .service import TOOL_SEPARATOR, McpServers, ResolvedServer, open_session, tool_name

logger = logging.getLogger(__name__)

CATALOG_TTL_SECONDS = 300.0
GRANTS_TTL_SECONDS = 30.0
LIST_TIMEOUT_SECONDS = 15.0
CALL_TIMEOUT_SECONDS = 60.0
_EMPTY_SCHEMA = {"type": "object", "properties": {}}


class RemoteMcpToolError(RuntimeError):
    """A remote MCP call failed; the message is safe to show the model."""


def split_tool_name(name: str) -> tuple[str, str] | None:
    """``(slug, tool)`` for a namespaced remote tool name, else None."""
    slug, sep, tool = (name or "").partition(TOOL_SEPARATOR)
    return (slug, tool) if sep and slug and tool else None


def _score(tool: McpToolDefinition, words: list[str]) -> int:
    name, text = tool.name.lower(), f"{tool.name} {tool.description}".lower()
    if not all(w in text for w in words):
        return -1
    return sum(2 if w in name else 1 for w in words)


class RemoteMcpTools:
    """One bot's granted remote MCP servers; shared by its turns."""

    def __init__(self, servers: Callable[[], McpServers], bot_slug: str):
        self._servers = servers
        self.bot_slug = bot_slug
        self._lock = threading.Lock()
        self._catalogs: dict[tuple[str, str], tuple[float, list[McpToolDefinition]]] = {}
        self._granted: tuple[float, bool] = (0.0, False)

    def has_servers(self) -> bool:
        """Whether any enabled server is granted to this bot (cached briefly).

        Deliberately ignores credentials: a server that needs sign-in still
        offers discovery, and the turn reports it as unavailable.
        """
        now = time.monotonic()
        at, granted = self._granted
        if now - at < GRANTS_TTL_SECONDS:
            return granted
        try:
            servers = self._servers()
            from ...bot_tags.store import BotTagStore
            tags = BotTagStore(servers.store.engine).tags_for(self.bot_slug)
            granted = bool(servers.store.servers_for(self.bot_slug, tags))
        except Exception as exc:  # noqa: BLE001 - storage outage: no remote tools this turn
            logger.warning("Remote MCP grants unavailable for %s: %s", self.bot_slug, type(exc).__name__)
            granted = False
        self._granted = (now, granted)
        return granted

    def for_turn(self) -> RemoteMcpTurn:
        return RemoteMcpTurn(self)

    # --- catalog cache (no credentials) -------------------------------------

    def catalog(self, server: ResolvedServer) -> list[McpToolDefinition]:
        key = (server.slug, server.url)
        now = time.monotonic()
        with self._lock:
            cached = self._catalogs.get(key)
        if cached and now - cached[0] < CATALOG_TTL_SECONDS:
            return cached[1]
        raw = run_bounded(_list(server), LIST_TIMEOUT_SECONDS)
        tools = [
            McpToolDefinition(
                name=tool_name(server.slug, t.name),
                description=f"[{server.name}] {(t.description or '').strip()}"[:1000],
                input_schema=(getattr(t, "input_schema", None) or getattr(t, "inputSchema", None)
                              or dict(_EMPTY_SCHEMA)),
            )
            for t in raw
            if t.name and TOOL_SEPARATOR not in t.name
        ]
        with self._lock:
            self._catalogs[key] = (now, tools)
        return tools


class RemoteMcpTurn:
    """The bot's remote servers for one turn, resolved lazily and at most once."""

    def __init__(self, tools: RemoteMcpTools):
        self._tools = tools
        self._resolved: dict[str, ResolvedServer] | None = None
        self.unavailable: list[str] = []

    @property
    def servers(self) -> dict[str, ResolvedServer]:
        if self._resolved is None:
            try:
                found = self._tools._servers().resolve_for_bot(self._tools.bot_slug)
            except Exception as exc:  # noqa: BLE001 - no remote tools this turn
                logger.warning("Remote MCP servers unavailable for %s: %s",
                               self._tools.bot_slug, type(exc).__name__)
                found = []
            self._resolved = {s.slug: s for s in found}
        return self._resolved

    def allows(self, name: str) -> bool:
        parts = split_tool_name(name)
        return parts is not None and parts[0] in self.servers

    def _catalog(self, server: ResolvedServer) -> list[McpToolDefinition]:
        try:
            return self._tools.catalog(server)
        except Exception as exc:  # noqa: BLE001 - one unreachable server never hides the others
            logger.warning("Remote MCP server %s tool listing failed: %s", server.slug, type(exc).__name__)
            if server.slug not in self.unavailable:
                self.unavailable.append(server.slug)
            return []

    def search_tools(self, query: str = "", limit: int = 5) -> list[McpToolDefinition]:
        words = [w for w in (query or "").lower().split() if w]
        ranked = sorted(
            ((_score(t, words), t) for s in self.servers.values() for t in self._catalog(s)),
            key=lambda pair: (-pair[0], pair[1].name),
        )
        return [tool for score, tool in ranked if score >= 0][: max(1, limit)]

    def select(self, names: list[str]) -> list[McpToolDefinition]:
        found = []
        for name in names:
            parts = split_tool_name(name)
            server = self.servers.get(parts[0]) if parts else None
            if server is None:
                continue
            found.extend(t for t in self._catalog(server) if t.name == name)
        return found

    def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        parts = split_tool_name(name)
        server = self.servers.get(parts[0]) if parts else None
        if server is None:
            raise RemoteMcpToolError(f"Tool '{name}' is not enabled for this bot")
        try:
            result = run_bounded(_call(server, parts[1], arguments), CALL_TIMEOUT_SECONDS)
        except Exception as exc:  # noqa: BLE001 - never echo transport details or credentials
            logger.warning("Remote MCP call %s failed: %s", name, type(exc).__name__)
            raise RemoteMcpToolError(f"{server.name} could not run {parts[1]} ({type(exc).__name__})") from exc
        text, structured = BawtHubMcpClient._result_parts(result)
        body = text or BawtHubMcpClient._serialize_structured(structured)
        if getattr(result, "isError", False) is True or getattr(result, "is_error", False) is True:
            raise RemoteMcpToolError(body or f"{name} reported an error")
        return body or f"{name} returned no content"


async def _list(server: ResolvedServer):
    async with open_session(server.url, server.transport, server.headers) as session:
        return (await session.list_tools()).tools


async def _call(server: ResolvedServer, tool: str, arguments: dict[str, Any]):
    async with open_session(server.url, server.transport, server.headers) as session:
        return await session.call_tool(tool, arguments)
