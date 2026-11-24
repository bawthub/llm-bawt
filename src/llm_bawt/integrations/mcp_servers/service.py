"""Remote MCP server management and per-bot resolution (TASK-2610).

The routes call :class:`McpServers` for configuration; the chat tool loop and
the agent bridges call :meth:`McpServers.resolve_for_bot` for the servers a bot
may use, each with a ready ``Authorization`` header. Refresh happens here, so
callers hold only a short-lived access token.
"""
from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from ..connections import IntegrationError, IntegrationStore
from ...runtime_settings import RuntimeSettingsStore
from ...service.providers.base import ConnectionRecord
from .oauth import McpOAuth, owned_by, store_id
from .store import (AUTH_KINDS, GRANT_SCOPES, TRANSPORTS, Grant, McpServer, McpServerError,
                    McpServerStore, validate_slug, validate_url)

logger = logging.getLogger(__name__)
TOOL_SEPARATOR = "__"
LIST_TIMEOUT_SECONDS = 20
MAX_GRANTS = 200
MAX_TOOL_PAGES = 20
MAX_TOOLS = 1000


@dataclass(frozen=True)
class ResolvedServer:
    """A granted, enabled, usable server for one bot; ``headers`` carry a live credential."""
    slug: str
    name: str
    url: str
    transport: str
    headers: dict[str, str] = field(repr=False, default_factory=dict)


def _bearer(token: str | None) -> str:
    token = (token or "").strip()
    if not token or len(token) > 8192 or any(c in token for c in "\r\n"):
        raise McpServerError("invalid_bearer_token")
    return token


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
        # Credential writes are fenced to the exact row (slug, URL, generation)
        # they began for, under the slug's lifecycle lock shared with create/remove.
        self.oauth.current = self.store.get

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
        server = McpServer(slug, name, validate_url(url), transport, auth_kind, True, uuid.uuid4().hex)
        if auth_kind != "none" and not server.url.startswith("https://"):
            raise McpServerError("https_required")  # never send a credential in plaintext
        if auth_kind == "bearer" and not (bearer_token or "").strip():
            raise McpServerError("bearer_token_required")
        token = _bearer(bearer_token) if auth_kind == "bearer" else None
        with self.oauth.fenced(server):
            if self.store.get(slug) is not None:
                raise McpServerError("slug_taken")
            # Stale secrets from an earlier server with this slug must never attach to the new one.
            self.oauth.disconnect(slug)
            self.store.create(server, actor)
            if token:
                self._write_bearer(server, token)
        return self.describe(server)

    def update(self, slug: str, *, name: str | None, enabled: bool | None, actor: str | None) -> dict:
        if name is not None:
            name = name.strip()[:80] or slug
        if not self.store.update(slug, name=name, enabled=enabled, actor=actor):
            raise McpServerError("not_found")
        return self.describe(self.get(slug))

    def delete(self, slug: str) -> bool:
        with self.credentials.lifecycle(store_id(slug)):
            removed = self.store.delete(slug)
            self.oauth.disconnect(slug)
        return removed

    def sign_out(self, slug: str) -> bool:
        """Forget the tokens and client registration of the row this request read.

        Fenced like every credential write: if the server was removed or
        replaced (same slug, new generation) meanwhile, refuse rather than
        delete the replacement's credentials.
        """
        server = self.get(slug)
        with self.oauth.fenced(server):
            current = self.store.get(slug)
            if current is None or current.generation != server.generation or current.url != server.url:
                raise McpServerError("not_found")
            return self.oauth.disconnect(slug)

    def set_bearer(self, slug: str, token: str) -> None:
        server = self.get(slug)
        if server.auth_kind != "bearer":
            raise McpServerError("invalid_bearer_token")
        token = _bearer(token)
        with self.oauth.fenced(server):
            self._write_bearer(server, token)

    def _write_bearer(self, server: McpServer, token: str) -> None:
        """Under the slug's lifecycle lock: save only for the row this request read."""
        current = self.store.get(server.slug)
        if current is None or current.generation != server.generation or current.url != server.url:
            raise McpServerError("not_found")
        self.credentials.save(ConnectionRecord(
            provider=store_id(server.slug), status="connected", auth_method="bearer",
            meta={"server_url": server.url, "generation": server.generation}, secret={"token": token}))

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
            if not (record and record.auth_method == "bearer" and record.secret.get("token")
                    and owned_by(record, server)):
                raise IntegrationError("not_connected")
            return {"Authorization": f"Bearer {record.secret['token']}"}
        return {"Authorization": f"Bearer {self.oauth.access_token(server)}"}

    def granted(self, bot_slug: str, tags: list[str] | None = None) -> list[McpServer]:
        """Enabled servers granted to the bot directly or by tag; no credentials touched."""
        if tags is None:
            from ...bot_tags.store import BotTagStore
            tags = BotTagStore(self.store.engine).tags_for(bot_slug)
        return self.store.servers_for(bot_slug, tags)

    def resolve_one(self, bot_slug: str, slug: str) -> ResolvedServer | None:
        """One granted server with a live credential, or None (ungranted, off or signed out)."""
        server = next((s for s in self.granted(bot_slug) if s.slug == slug), None)
        if server is None:
            return None
        try:
            headers = self.headers(server)
        except IntegrationError as exc:
            logger.warning("MCP server %s unavailable for %s: %s", slug, bot_slug, exc)
            return None
        return ResolvedServer(server.slug, server.name, server.url, server.transport, headers)

    def resolve_for_bot(self, bot_slug: str, tags: list[str] | None = None) -> list[ResolvedServer]:
        """Servers granted to the bot (directly or by tag) that have a usable credential.

        A server that can't produce a credential right now is skipped and logged,
        so one expired sign-in never takes a bot's other tools down.
        """
        resolved = []
        for server in self.granted(bot_slug, tags):
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


async def list_all_tools(session) -> list:
    """Every page of a server's tool catalog, bounded in pages and tools."""
    from mcp.types import PaginatedRequestParams

    tools: list = []
    cursor = None
    for _ in range(MAX_TOOL_PAGES):
        result = await (session.list_tools(params=PaginatedRequestParams(cursor=cursor)) if cursor
                        else session.list_tools())
        tools.extend(result.tools)
        cursor = getattr(result, "next_cursor", None) or getattr(result, "nextCursor", None)
        if not cursor or len(tools) >= MAX_TOOLS:
            break
    return tools[:MAX_TOOLS]


async def _list_tools(server: McpServer, headers: dict[str, str]):
    async with open_session(server.url, server.transport, headers) as session:
        return await list_all_tools(session)
