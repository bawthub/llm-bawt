"""Remote MCP server management and per-bot resolution (TASK-2610).

The routes call :class:`McpServers` for configuration; the chat tool loop and
the agent bridges call :meth:`McpServers.resolve_for_bot` for the servers a bot
may use, each with a ready ``Authorization`` header. Refresh happens here, so
callers hold only a short-lived access token.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from ..connections import IntegrationError, IntegrationStore
from ...runtime_settings import RuntimeSettingsStore
from ...service.providers.base import ConnectionRecord
from .oauth import McpOAuth, store_id
from .store import (AUTH_KINDS, GRANT_SCOPES, TRANSPORTS, Grant, McpServer, McpServerError,
                    McpServerStore, validate_slug, validate_url)

logger = logging.getLogger(__name__)
TOOL_SEPARATOR = "__"
LIST_TIMEOUT_SECONDS = 20
MAX_GRANTS = 200


@dataclass(frozen=True)
class ResolvedServer:
    """A granted, enabled, usable server for one bot; ``headers`` carry a live credential."""
    slug: str
    name: str
    url: str
    transport: str
    headers: dict[str, str] = field(repr=False, default_factory=dict)


def tool_name(slug: str, tool: str) -> str:
    return f"{slug}{TOOL_SEPARATOR}{tool}"


class McpServers:
    def __init__(self, config, *, engine=None, credentials: IntegrationStore | None = None,
                 oauth: McpOAuth | None = None):
        self.config = config
        engine = engine if engine is not None else RuntimeSettingsStore(config).engine
        self.store = McpServerStore(engine)
        self.credentials = credentials if credentials is not None else IntegrationStore(config)
        self.oauth = oauth if oauth is not None else McpOAuth(config, store=self.credentials)

    # --- configuration ------------------------------------------------------

    def describe(self, server: McpServer) -> dict:
        return {**server.public(), **self.oauth.status(server),
                "grants": [g.__dict__ for g in self.store.grants(server.slug)]}

    def list(self) -> list[dict]:
        return [self.describe(s) for s in self.store.list()]

    def get(self, slug: str) -> McpServer:
        server = self.store.get(slug)
        if server is None:
            raise McpServerError("not_found")
        return server

    def create(self, *, slug: str, name: str, url: str, transport: str, auth_kind: str,
               bearer_token: str | None, actor: str | None) -> dict:
        slug = validate_slug(slug)
        if transport not in TRANSPORTS or auth_kind not in AUTH_KINDS:
            raise McpServerError("invalid_field")
        name = (name or "").strip()[:80] or slug
        server = McpServer(slug, name, validate_url(url), transport, auth_kind, True)
        if auth_kind == "bearer" and not (bearer_token or "").strip():
            raise McpServerError("bearer_token_required")
        if self.store.get(slug) is not None:
            raise McpServerError("slug_taken")
        # Stale secrets from an earlier server with this slug must never attach to the new one.
        self.oauth.disconnect(slug)
        self.store.create(server, actor)
        if auth_kind == "bearer":
            self.set_bearer(slug, bearer_token or "")
        return self.describe(server)

    def update(self, slug: str, *, name: str | None, enabled: bool | None, actor: str | None) -> dict:
        if name is not None:
            name = name.strip()[:80] or slug
        if not self.store.update(slug, name=name, enabled=enabled, actor=actor):
            raise McpServerError("not_found")
        return self.describe(self.get(slug))

    def delete(self, slug: str) -> bool:
        removed = self.store.delete(slug)
        self.oauth.disconnect(slug)
        return removed

    def set_bearer(self, slug: str, token: str) -> None:
        server = self.get(slug)
        token = token.strip()
        if server.auth_kind != "bearer" or not token or len(token) > 8192 or any(c in token for c in "\r\n"):
            raise McpServerError("invalid_bearer_token")
        self.credentials.save(ConnectionRecord(provider=store_id(slug), status="connected",
                                               auth_method="bearer", meta={"server_url": server.url},
                                               secret={"token": token}))

    def set_grants(self, slug: str, grants: list[dict], actor: str | None) -> dict:
        if not isinstance(grants, list) or len(grants) > MAX_GRANTS:
            raise McpServerError("invalid_field")
        parsed = []
        for g in grants:
            scope_type, scope_id = (g or {}).get("scope_type"), (g or {}).get("scope_id")
            if scope_type not in GRANT_SCOPES or not isinstance(scope_id, str) or not 0 < len(scope_id) <= 100:
                raise McpServerError("invalid_field")
            parsed.append(Grant(scope_type, scope_id))
        self.store.set_grants(slug, parsed, actor)
        return self.describe(self.get(slug))

    # --- use ----------------------------------------------------------------

    def headers(self, server: McpServer) -> dict[str, str]:
        """The credential header for one request batch; raises IntegrationError when unusable."""
        if server.auth_kind == "none":
            return {}
        if server.auth_kind == "bearer":
            record = self.credentials.load(store_id(server.slug))
            if not record or record.auth_method != "bearer" or not record.secret.get("token"):
                raise IntegrationError("not_connected")
            return {"Authorization": f"Bearer {record.secret['token']}"}
        return {"Authorization": f"Bearer {self.oauth.access_token(server)}"}

    def resolve_for_bot(self, bot_slug: str, tags: list[str] | None = None) -> list[ResolvedServer]:
        """Servers granted to the bot (directly or by tag) that have a usable credential.

        A server that can't produce a credential right now is skipped and logged,
        so one expired sign-in never takes a bot's other tools down.
        """
        if tags is None:
            from ...bot_tags.store import BotTagStore
            tags = BotTagStore(self.store.engine).tags_for(bot_slug)
        resolved = []
        for server in self.store.servers_for(bot_slug, tags):
            try:
                headers = self.headers(server)
            except IntegrationError as exc:
                logger.warning("MCP server %s unavailable for %s: %s", server.slug, bot_slug, exc)
                continue
            resolved.append(ResolvedServer(server.slug, server.name, server.url, server.transport, headers))
        return resolved

    def list_tools(self, slug: str) -> list[dict]:
        """Connect once and list the server's tools (Settings "Test"), namespaced."""
        from ..ha_mcp.client import run_bounded
        server = self.get(slug)
        headers = self.headers(server)
        try:
            tools = run_bounded(_list_tools(server, headers), LIST_TIMEOUT_SECONDS)
        except Exception as exc:  # transport, protocol or auth: bounded code only
            logger.warning("MCP server %s tool listing failed: %s", slug, type(exc).__name__)
            raise IntegrationError("server_unreachable") from exc
        return [{"name": tool_name(slug, t.name), "description": (t.description or "")[:300]} for t in tools]


@asynccontextmanager
async def open_session(server_url: str, transport: str, headers: dict[str, str]):
    """An initialized MCP client session over the server's transport."""
    from mcp import ClientSession
    if transport == "sse":
        from mcp.client.sse import sse_client
        async with sse_client(server_url, headers=headers) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session
        return
    from mcp.client.streamable_http import streamable_http_client
    from mcp.shared._httpx_utils import create_mcp_http_client
    async with create_mcp_http_client(headers=headers) as http_client:
        async with streamable_http_client(server_url, http_client=http_client) as (read, write, *_):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session


async def _list_tools(server: McpServer, headers: dict[str, str]):
    async with open_session(server.url, server.transport, headers) as session:
        return (await session.list_tools()).tools
