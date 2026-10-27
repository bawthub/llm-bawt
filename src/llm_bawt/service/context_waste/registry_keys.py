"""Per-tool argument key names declared by the in-process MCP registry.

TASK-1590 (TASK-1585 DESIGN v2 §4): when the registry can be read in this
process, a first-party tool's argument keys come from that tool's own declared
input schema. Tools the registry doesn't hold, or any failure to read it, fall
back to the static allowlist in :mod:`.report`. Only key *names* are read,
never argument values.

Both processes hold the registry: the MCP server registers every tool through
its module imports, and the app does so at startup
(``ensure_tools_registered``). A bare or partial registry only means more
tools use the fallback.
"""
from __future__ import annotations

import logging
from collections.abc import Mapping

log = logging.getLogger(__name__)

McpKeys = Mapping[str, frozenset[str]]


async def declared_mcp_keys() -> McpKeys | None:
    """``{tool name: declared property names}``, or ``None`` when unreadable."""
    try:
        from mcp.server.fastmcp import FastMCP

        from llm_bawt.mcp_server.registry import mcp

        # FastMCP's own listing: the catalog override compacts descriptions,
        # and the property names are what matters here.
        tools = await FastMCP.list_tools(mcp)
    except Exception:
        log.debug("MCP registry unreadable; using the static key allowlist", exc_info=True)
        return None
    keys: dict[str, frozenset[str]] = {}
    for tool in tools:
        properties = (tool.inputSchema or {}).get("properties")
        if isinstance(properties, Mapping):
            keys[tool.name] = frozenset(str(name) for name in properties)
    return keys
