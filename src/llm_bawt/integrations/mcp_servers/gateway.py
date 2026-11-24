"""Remote MCP servers for agent-bridge bots (TASK-2610).

Claude and Codex bridge bots run their own MCP clients, so they reach a granted
remote server through this gateway on the llm-bawt MCP listener:
``/remote-mcp/<slug>/mcp``. Each request proves its bot with the signed
current-turn capability header the bridges already send to the BawtHub MCP
server. The gateway checks the bot's grant on every request, attaches the
server's credential here and forwards ``tools/list`` and ``tools/call``, so
access and refresh tokens never leave llm-bawt and a revoked grant takes
effect on the next call.

Everything upstream is untrusted. Before any of it is returned (tool
definitions included) the gateway records the verified turn's exposure, so
write admission holds later memory/trait writes whatever the bridge; call
results also come back labelled with source ``mcp:<slug>``.
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

from mcp.shared.exceptions import MCPError

from agent_bridge.content_provenance import label_mcp_output
from ...task_turn_context import TASK_TURN_CONTEXT_HEADER, open_task_turn_context
from .service import McpServers, ResolvedServer, list_all_tools, open_session
from .store import McpServerError, validate_slug

logger = logging.getLogger(__name__)

PREFIX = "/remote-mcp/"
CALL_TIMEOUT_SECONDS = 120.0
LIST_TIMEOUT_SECONDS = 30.0

_current: contextvars.ContextVar[tuple[ResolvedServer, Any] | None] = contextvars.ContextVar(
    "remote_mcp_gateway_server", default=None)
LIST_FAILED = "The remote MCP server could not list its tools"


def _expose(turn) -> None:
    from ...content_provenance import ContentTurn
    from ...service.dependencies import get_content_provenance_store
    from ...utils.config import Config

    get_content_provenance_store(Config()).expose(
        ContentTurn(turn.bot_id, turn.user_id, turn.turn_id, turn.trigger_message_id))


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


def _turn(scope: dict):
    """The verified current turn from exactly one valid capability header, else None."""
    name = TASK_TURN_CONTEXT_HEADER.lower().encode("ascii")
    values = [v for k, v in scope.get("headers", []) if k.lower() == name]
    if len(values) != 1:
        return None
    try:
        turn = open_task_turn_context(values[0].decode("ascii"))
    except Exception:  # noqa: BLE001 - invalid, expired or undecodable: refuse
        return None
    return turn if turn.bot_id else None


async def _refuse(send, status: int, code: str) -> None:
    body = json.dumps({"error": code}).encode()
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", b"application/json"), (b"cache-control", b"no-store")]})
    await send({"type": "http.response.body", "body": body})


class RemoteMcpGateway:
    """ASGI app serving ``/remote-mcp/<slug>/mcp``; mount beside the BawtHub MCP app."""

    def __init__(self, servers: Callable[[], McpServers], *, security_settings=None,
                 expose: Callable[[Any], None] = _expose):
        self._servers = servers
        self._expose = expose
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
        turn = _turn(scope)
        bot = turn.bot_id if turn is not None else None
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
        token = _current.set((resolved, turn))
        try:
            await self.manager.handle_request(scope, receive, send)
        finally:
            _current.reset(token)

    @staticmethod
    def _request() -> tuple[ResolvedServer, Any]:
        current = _current.get()
        if current is None:  # only reachable by a request that skipped __call__
            raise RuntimeError("no remote MCP server for this request")
        return current

    async def _exposed(self, turn) -> bool:
        """Record the turn's untrusted exposure before upstream data can return."""
        try:
            await anyio.to_thread.run_sync(self._expose, turn)
            return True
        except Exception as exc:  # noqa: BLE001 - fail closed: nothing upstream is returned
            logger.warning("Remote MCP gateway: content provenance unavailable (%s)", type(exc).__name__)
            return False

    async def _list_tools(self, _ctx, _params) -> types.ListToolsResult:
        # Every page is fetched here and returned at once, so the result has no cursor.
        server, turn = self._request()
        if not await self._exposed(turn):
            raise MCPError(-32603, "Content provenance is unavailable; remote tools withheld")
        try:
            with anyio.fail_after(LIST_TIMEOUT_SECONDS):
                async with open_session(server.url, server.transport, server.headers) as session:
                    tools = await list_all_tools(session)
        except Exception as exc:  # fixed error: never the upstream diagnostics or URL
            logger.warning("Remote MCP gateway listing %s failed: %s", server.slug, type(exc).__name__)
            raise MCPError(-32603, LIST_FAILED) from None
        # Results are relabelled (see labelled), so an upstream outputSchema no
        # longer describes what the client receives; clients must not validate against it.
        return types.ListToolsResult(tools=[t.model_copy(update={"output_schema": None}) for t in tools])

    async def _call_tool(self, _ctx, params: types.CallToolRequestParams) -> types.CallToolResult:
        server, turn = self._request()
        if not await self._exposed(turn):
            return _error("Content provenance is unavailable; nothing was executed.")
        arguments: dict[str, Any] = dict(params.arguments or {})
        try:
            with anyio.fail_after(CALL_TIMEOUT_SECONDS):
                async with open_session(server.url, server.transport, server.headers) as session:
                    result = await session.call_tool(params.name, arguments)
        except Exception as exc:  # never echo transport details or credentials
            logger.warning("Remote MCP gateway call %s/%s failed: %s", server.slug, params.name,
                           type(exc).__name__)
            return _error(f"{server.name} could not run {params.name} ({type(exc).__name__})")
        return labelled(result, f"mcp:{server.slug}")


def _error(text: str) -> types.CallToolResult:
    return types.CallToolResult(content=[types.TextContent(type="text", text=text)], is_error=True)


def labelled(result: types.CallToolResult, source: str) -> types.CallToolResult:
    """The upstream result labelled untrusted ``source`` data in every model-visible part.

    Text blocks are wrapped, and so is ``structuredContent``: some clients (Codex)
    show the model only the structured value when it is present, dropping both
    ``_meta`` and the content array.
    """
    data = result.model_dump(by_alias=True, mode="json", exclude_none=True)
    out = label_mcp_output(data, source)
    if "structuredContent" in data:
        out["structuredContent"] = {"untrusted_content": True, "source": source,
                                    "data": data["structuredContent"]}
    return types.CallToolResult.model_validate(out)


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
