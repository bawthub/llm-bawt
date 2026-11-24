"""Remote MCP servers for agent-bridge bots (TASK-2610).

Claude and Codex bridge bots run their own MCP clients, so they reach a granted
remote server through this gateway on the llm-bawt MCP listener:
``/remote-mcp/<slug>/mcp``. Each request proves its bot with the signed
current-turn capability header the bridges already send to the BawtHub MCP
server. The gateway checks the bot's grant on every request, attaches the
server's credential here and forwards ``tools/list`` and ``tools/call``, so
access and refresh tokens never leave llm-bawt and a revoked grant takes
effect on the next call.

Bridges label everything returned as untrusted with source ``mcp:<slug>``.
"""
from __future__ import annotations

import contextvars
import json
import logging
from collections.abc import Callable
from typing import Any

import anyio
import mcp.types as types
from mcp.server.lowlevel import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager

from ...task_turn_context import TASK_TURN_CONTEXT_HEADER, open_task_turn_context
from .service import McpServers, ResolvedServer, open_session
from .store import McpServerError, validate_slug

logger = logging.getLogger(__name__)

PREFIX = "/remote-mcp/"
CALL_TIMEOUT_SECONDS = 120.0
LIST_TIMEOUT_SECONDS = 30.0

_current: contextvars.ContextVar[ResolvedServer | None] = contextvars.ContextVar(
    "remote_mcp_gateway_server", default=None)


def gateway_path(slug: str) -> str:
    return f"{PREFIX}{slug}/mcp"


def _slug_of(path: str) -> str | None:
    rest = path[len(PREFIX):] if path.startswith(PREFIX) else ""
    slug, sep, tail = rest.partition("/")
    if not sep or tail not in ("mcp", "mcp/"):
        return None
    try:
        return validate_slug(slug)
    except McpServerError:
        return None


def _turn_bot(scope: dict) -> str | None:
    """The bot proven by exactly one valid current-turn capability header."""
    name = TASK_TURN_CONTEXT_HEADER.lower().encode("ascii")
    values = [v for k, v in scope.get("headers", []) if k.lower() == name]
    if len(values) != 1:
        return None
    try:
        return open_task_turn_context(values[0].decode("ascii")).bot_id or None
    except Exception:  # noqa: BLE001 - invalid, expired or undecodable: refuse
        return None


async def _refuse(send, status: int, code: str) -> None:
    body = json.dumps({"error": code}).encode()
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", b"application/json"), (b"cache-control", b"no-store")]})
    await send({"type": "http.response.body", "body": body})


class RemoteMcpGateway:
    """ASGI app serving ``/remote-mcp/<slug>/mcp``; mount beside the BawtHub MCP app."""

    def __init__(self, servers: Callable[[], McpServers], *, security_settings=None):
        self._servers = servers
        server = Server("bawthub-remote-mcp", on_list_tools=self._list_tools, on_call_tool=self._call_tool)
        self.manager = StreamableHTTPSessionManager(
            app=server, json_response=True, stateless=True, security_settings=security_settings)

    @staticmethod
    def handles(path: str) -> bool:
        return path.startswith(PREFIX)

    async def __call__(self, scope, receive, send) -> None:
        slug = _slug_of(scope.get("path", ""))
        if slug is None:
            await _refuse(send, 404, "not_found")
            return
        bot = _turn_bot(scope)
        if bot is None:
            await _refuse(send, 401, "turn_capability_required")
            return
        try:
            resolved = await anyio.to_thread.run_sync(lambda: self._servers().resolve_one(bot, slug))
        except Exception as exc:  # noqa: BLE001 - storage outage
            logger.warning("Remote MCP gateway unavailable for %s/%s: %s", bot, slug, type(exc).__name__)
            await _refuse(send, 503, "unavailable")
            return
        if resolved is None:
            # Unknown, not granted, turned off and signed out all look the same to the bot.
            await _refuse(send, 404, "not_available")
            return
        token = _current.set(resolved)
        try:
            await self.manager.handle_request(scope, receive, send)
        finally:
            _current.reset(token)

    @staticmethod
    def _server() -> ResolvedServer:
        server = _current.get()
        if server is None:  # only reachable by a request that skipped __call__
            raise RuntimeError("no remote MCP server for this request")
        return server

    async def _list_tools(self, _ctx, _params) -> types.ListToolsResult:
        server = self._server()
        with anyio.fail_after(LIST_TIMEOUT_SECONDS):
            async with open_session(server.url, server.transport, server.headers) as session:
                result = await session.list_tools()
        return types.ListToolsResult(tools=list(result.tools))

    async def _call_tool(self, _ctx, params: types.CallToolRequestParams) -> types.CallToolResult:
        server = self._server()
        arguments: dict[str, Any] = dict(params.arguments or {})
        try:
            with anyio.fail_after(CALL_TIMEOUT_SECONDS):
                async with open_session(server.url, server.transport, server.headers) as session:
                    return await session.call_tool(params.name, arguments)
        except Exception as exc:  # never echo transport details or credentials
            logger.warning("Remote MCP gateway call %s/%s failed: %s", server.slug, params.name,
                           type(exc).__name__)
            return types.CallToolResult(
                content=[types.TextContent(type="text",
                                           text=f"{server.name} could not run {params.name} ({type(exc).__name__})")],
                is_error=True)


class WithRemoteMcp:
    """Route ``/remote-mcp/`` to the gateway and everything else to ``inner``.

    Sits inside the :8001 caller gate, so the gateway gets the same outer
    authentication as the BawtHub MCP server; the lifespan also runs the
    gateway's session manager.
    """

    def __init__(self, inner, gateway: RemoteMcpGateway):
        self.inner, self.gateway = inner, gateway

    async def __call__(self, scope, receive, send) -> None:
        if scope.get("type") == "lifespan":
            async with self.gateway.manager.run():
                await self.inner(scope, receive, send)
            return
        if scope.get("type") == "http" and RemoteMcpGateway.handles(scope.get("path", "")):
            await self.gateway(scope, receive, send)
            return
        await self.inner(scope, receive, send)


def with_remote_mcp(inner, *, security_settings=None):
    """The :8001 app with the remote MCP gateway mounted (one shared McpServers)."""
    from ...utils.config import Config

    holder: list[McpServers] = []

    def servers() -> McpServers:
        if not holder:
            holder.append(McpServers(Config()))
        return holder[0]

    return WithRemoteMcp(inner, RemoteMcpGateway(servers, security_settings=security_settings))
