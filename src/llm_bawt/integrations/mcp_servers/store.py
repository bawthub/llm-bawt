"""Remote MCP server rows and their bot/tag grants (TASK-2610).

SQL only. Secrets (bearer tokens, OAuth client registration and tokens) never
live here: they stay in the encrypted integration store under ``mcp:<slug>``.
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from urllib.parse import urlsplit

from sqlalchemy import text

from ...utils.schema import SchemaBootstrapGuard, ensure_relations

TRANSPORTS = ("streamable_http", "sse")
AUTH_KINDS = ("oauth", "bearer", "none")
GRANT_SCOPES = ("bot", "tag")
#: Built-in MCP servers whose names a remote server may never take.
RESERVED_SLUGS = frozenset({"bawthub", "playwright", "crawl4ai", "mcp", "homeassistant", "ha"})
#: Lowercase, starts with a letter, no double underscore: tools are named ``<slug>__<tool>``.
SLUG_RE = re.compile(r"^[a-z][a-z0-9_-]{0,30}$")


class McpServerError(Exception):
    """A bounded public error code; never contains a secret or an upstream body."""


def _vocab(column: str, values: Iterable[str]) -> str:
    return f"{column} IN ({', '.join(repr(v) for v in values)})"


_RELATIONS = {
    "mcp_servers": f"""
        CREATE TABLE IF NOT EXISTS mcp_servers (
            slug TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            url TEXT NOT NULL,
            transport TEXT NOT NULL CHECK ({_vocab('transport', TRANSPORTS)}),
            auth_kind TEXT NOT NULL CHECK ({_vocab('auth_kind', AUTH_KINDS)}),
            enabled BOOLEAN NOT NULL DEFAULT true,
            updated_by TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )""",
    "mcp_server_grants": f"""
        CREATE TABLE IF NOT EXISTS mcp_server_grants (
            server_slug TEXT NOT NULL REFERENCES mcp_servers(slug) ON DELETE CASCADE,
            scope_type TEXT NOT NULL CHECK ({_vocab('scope_type', GRANT_SCOPES)}),
            scope_id TEXT NOT NULL,
            updated_by TEXT,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (server_slug, scope_type, scope_id)
        )""",
}
_schema_guard = SchemaBootstrapGuard()


@dataclass(frozen=True)
class McpServer:
    slug: str
    name: str
    url: str
    transport: str
    auth_kind: str
    enabled: bool

    def public(self) -> dict:
        return {"slug": self.slug, "name": self.name, "url": self.url, "transport": self.transport,
                "auth_kind": self.auth_kind, "enabled": self.enabled}


@dataclass(frozen=True)
class Grant:
    scope_type: str
    scope_id: str


def validate_slug(slug: str) -> str:
    if not isinstance(slug, str) or not SLUG_RE.match(slug) or "__" in slug or slug.endswith(("_", "-")):
        raise McpServerError("invalid_slug")
    if slug in RESERVED_SLUGS:
        raise McpServerError("reserved_slug")
    return slug


def validate_url(url: str) -> str:
    """An absolute http(s) URL without credentials or fragment (the MCP resource identifier)."""
    url = (url or "").strip()
    try:
        parts = urlsplit(url)
    except ValueError as exc:
        raise McpServerError("invalid_url") from exc
    if (parts.scheme not in ("https", "http") or not parts.hostname or parts.username
            or parts.password or parts.fragment or len(url) > 2048):
        raise McpServerError("invalid_url")
    return url


def _row(r) -> McpServer:
    return McpServer(r.slug, r.name, r.url, r.transport, r.auth_kind, r.enabled)


class McpServerStore:
    def __init__(self, engine) -> None:
        self.engine = engine

    def ensure_schema(self) -> None:
        _schema_guard.run(self.engine, "mcp-servers", lambda conn: ensure_relations(conn, _RELATIONS))

    def _conn(self):
        if self.engine is None:
            raise McpServerError("storage_unavailable")
        self.ensure_schema()
        return self.engine.begin()

    def list(self) -> list[McpServer]:
        with self._conn() as conn:
            rows = conn.execute(text("SELECT * FROM mcp_servers ORDER BY slug")).all()
        return [_row(r) for r in rows]

    def get(self, slug: str) -> McpServer | None:
        with self._conn() as conn:
            r = conn.execute(text("SELECT * FROM mcp_servers WHERE slug = :s"), {"s": slug}).first()
        return _row(r) if r else None

    def create(self, server: McpServer, actor: str | None) -> None:
        with self._conn() as conn:
            inserted = conn.execute(text("""
                INSERT INTO mcp_servers (slug, name, url, transport, auth_kind, enabled, updated_by)
                VALUES (:slug, :name, :url, :transport, :auth_kind, :enabled, :actor)
                ON CONFLICT (slug) DO NOTHING"""), {**server.public(), "actor": actor}).rowcount
        if inserted != 1:
            raise McpServerError("slug_taken")

    def update(self, slug: str, *, name: str | None, enabled: bool | None, actor: str | None) -> bool:
        with self._conn() as conn:
            return conn.execute(text("""
                UPDATE mcp_servers SET name = COALESCE(:name, name), enabled = COALESCE(:enabled, enabled),
                       updated_by = :actor, updated_at = now() WHERE slug = :slug"""),
                {"slug": slug, "name": name, "enabled": enabled, "actor": actor}).rowcount == 1

    def delete(self, slug: str) -> bool:
        with self._conn() as conn:
            return conn.execute(text("DELETE FROM mcp_servers WHERE slug = :s"), {"s": slug}).rowcount == 1

    def grants(self, slug: str) -> list[Grant]:
        with self._conn() as conn:
            rows = conn.execute(text("""
                SELECT scope_type, scope_id FROM mcp_server_grants WHERE server_slug = :s
                ORDER BY scope_type, scope_id"""), {"s": slug}).all()
        return [Grant(r.scope_type, r.scope_id) for r in rows]

    def set_grants(self, slug: str, grants: list[Grant], actor: str | None) -> None:
        """Replace a server's grants in one transaction."""
        with self._conn() as conn:
            if conn.execute(text("SELECT 1 FROM mcp_servers WHERE slug = :s FOR UPDATE"), {"s": slug}).first() is None:
                raise McpServerError("not_found")
            conn.execute(text("DELETE FROM mcp_server_grants WHERE server_slug = :s"), {"s": slug})
            for g in grants:
                conn.execute(text("""
                    INSERT INTO mcp_server_grants (server_slug, scope_type, scope_id, updated_by)
                    VALUES (:s, :t, :i, :a) ON CONFLICT DO NOTHING"""),
                    {"s": slug, "t": g.scope_type, "i": g.scope_id, "a": actor})

    def servers_for(self, bot_slug: str, tags: Iterable[str]) -> list[McpServer]:
        """Enabled servers granted to this bot directly or through one of its tags."""
        with self._conn() as conn:
            rows = conn.execute(text("""
                SELECT DISTINCT s.* FROM mcp_servers s JOIN mcp_server_grants g ON g.server_slug = s.slug
                WHERE s.enabled AND ((g.scope_type = 'bot' AND g.scope_id = :bot)
                                     OR (g.scope_type = 'tag' AND g.scope_id = ANY(:tags)))
                ORDER BY s.slug"""), {"bot": bot_slug, "tags": list(tags)}).all()
        return [_row(r) for r in rows]
