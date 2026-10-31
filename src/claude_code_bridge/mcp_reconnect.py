"""Restore remote MCP catalogs that disconnect inside a long Claude SDK run.

The SDK opens a new CLI client per llm-bawt turn. During one active turn,
restarting the MCP service can remove its deferred tools without generating a
tool result. Use the SDK control plane to restore only configured remote
servers; never replay a model tool call (which might be a write).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)

_CHECK_INTERVAL_SECONDS = 15
_CONTROL_TIMEOUT_SECONDS = 8
_MAX_RETRY_SECONDS = 120


class McpReconnectMonitor:
    """Watch the active SDK client; stop with its owning run."""

    def __init__(self, client: Any) -> None:
        self._client = client
        configurations = getattr(getattr(client, "options", None), "mcp_servers", None)
        if not isinstance(configurations, dict):
            configurations = {}
        self._remote_names = frozenset(
            name for name, config in configurations.items()
            if isinstance(config, dict) and config.get("type") in {"http", "sse"}
        )
        self._retry_at: dict[str, float] = {}
        self._retry_delay: dict[str, int] = {}
        self._warned: set[str] = set()
        self._status_warning = False
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        if self._remote_names and self._task is None:
            self._task = asyncio.create_task(self._run(), name="mcp-reconnect")

    def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()

    async def _run(self) -> None:
        # Short turns need no extra CLI control round trip. Continue checking
        # while the model is blocked on a tool; ToolSearch need not be called.
        while True:
            await asyncio.sleep(_CHECK_INTERVAL_SECONDS)
            try:
                status = await asyncio.wait_for(
                    self._client.get_mcp_status(), timeout=_CONTROL_TIMEOUT_SECONDS,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if not self._status_warning:
                    logger.warning("MCP status check failed: %s", type(exc).__name__)
                    self._status_warning = True
                continue
            self._status_warning = False
            if not isinstance(status, dict):
                continue
            for server in status.get("mcpServers", ()):
                if not isinstance(server, dict):
                    continue
                name = server.get("name")
                if name not in self._remote_names:
                    continue
                if server.get("status") != "failed":
                    self._retry_at.pop(name, None)
                    self._retry_delay.pop(name, None)
                    self._warned.discard(name)
                    continue
                now = asyncio.get_running_loop().time()
                if now < self._retry_at.get(name, 0):
                    continue
                try:
                    await asyncio.wait_for(
                        self._client.reconnect_mcp_server(name),
                        timeout=_CONTROL_TIMEOUT_SECONDS,
                    )
                    logger.info("MCP server reconnected: %s", name)
                    self._retry_at.pop(name, None)
                    self._retry_delay.pop(name, None)
                    self._warned.discard(name)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    # App still down: don't fail the bot's turn, replay a
                    # possibly-mutating tool call, or log secrets from the CLI.
                    if name not in self._warned:
                        logger.warning(
                            "MCP server %s reconnect failed: %s", name, type(exc).__name__,
                        )
                        self._warned.add(name)
                    delay = self._retry_delay.get(name, _CHECK_INTERVAL_SECONDS)
                    self._retry_at[name] = now + delay
                    self._retry_delay[name] = min(delay * 2, _MAX_RETRY_SECONDS)
