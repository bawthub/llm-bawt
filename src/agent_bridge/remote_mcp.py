"""Owner-added remote MCP servers in agent bridges (TASK-2610).

The app publishes the slugs of the remote servers granted to the bot (never a
URL or credential) in the ``remote_mcp`` command field. Each bridge then adds,
for that turn only, one MCP server per slug pointing at the llm-bawt gateway
``/remote-mcp/<slug>/mcp`` on the BawtHub MCP listener, authenticated by the
turn's signed capability header. The gateway re-checks the grant and holds the
server's credential, so a bridge never sees a token.
"""
from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from urllib.parse import urlsplit

from .content_provenance import remote_mcp_source

FIELD = "remote_mcp"
TURN_HEADER = "X-LLM-Bawt-Task-Turn-Context"
MAX_SERVERS = 50
# Mirrors llm_bawt.integrations.mcp_servers.store.validate_slug.
_SLUG = re.compile(r"[a-z][a-z0-9_-]{0,30}")


def _valid(slug: object) -> bool:
    return (isinstance(slug, str) and bool(_SLUG.fullmatch(slug)) and "__" not in slug
            and not slug.endswith(("_", "-")))


def remote_mcp_field(slugs: Iterable[str]) -> str | None:
    """The command field value: a JSON list of valid, unique slugs, or None."""
    out = list(dict.fromkeys(s for s in slugs if _valid(s)))[:MAX_SERVERS]
    return json.dumps(out) if out else None


def parse_remote_mcp(value: object) -> tuple[str, ...]:
    """Slugs from a command field; anything malformed is dropped, never trusted."""
    if not isinstance(value, str) or not value:
        return ()
    try:
        items = json.loads(value)
    except ValueError:
        return ()
    if not isinstance(items, list):
        return ()
    return tuple(dict.fromkeys(s for s in items if _valid(s)))[:MAX_SERVERS]


def gateway_url(bawthub_mcp_url: str, slug: str) -> str:
    """The gateway endpoint on the same listener as the BawtHub MCP server."""
    parts = urlsplit(bawthub_mcp_url)
    return f"{parts.scheme}://{parts.netloc}/remote-mcp/{slug}/mcp"


def claude_servers(slugs: Iterable[str], existing: Mapping[str, object], bawthub_url: str,
                   capability: str | None) -> dict[str, dict]:
    """Per-turn Claude SDK ``mcp_servers`` entries; built-in names always win."""
    if not capability or not bawthub_url:
        return {}
    return {
        slug: {"type": "http", "url": gateway_url(bawthub_url, slug), "headers": {TURN_HEADER: capability}}
        for slug in slugs if _valid(slug) and slug not in existing
    }


def codex_overrides(slugs: Iterable[str], existing: Iterable[str], bawthub_url: str,
                    capability_env: str) -> list[str]:
    """``codex --config`` overrides adding one MCP server per slug for this run.

    The capability travels in the run's environment (``env_http_headers``), so it
    never appears in a process argument list. Values round-trip through tomllib.
    """
    import tomllib

    taken = set(existing)
    picked = [s for s in slugs if _valid(s) and s not in taken]
    if not picked or not bawthub_url:
        return []
    out: list[str] = []
    for slug in picked:
        out.append(f"mcp_servers.{slug}.url={json.dumps(gateway_url(bawthub_url, slug))}")
        out.append(f"mcp_servers.{slug}.env_http_headers="
                   f"{{{json.dumps(TURN_HEADER)}={json.dumps(capability_env)}}}")
    parsed = tomllib.loads("\n".join(out))
    expected = {slug: {"url": gateway_url(bawthub_url, slug), "env_http_headers": {TURN_HEADER: capability_env}}
                for slug in picked}
    if parsed != {"mcp_servers": expected}:
        raise ValueError("remote MCP overrides do not round-trip")
    return out


def source_for(tool_name: str, slugs: Iterable[str]) -> str | None:
    """``mcp:<slug>`` for a tool of one of this turn's remote servers, else None."""
    source = remote_mcp_source(tool_name)
    return source if source and source.removeprefix("mcp:") in set(slugs) else None


def bridge_source(tool_name: str, slugs: Iterable[str]) -> str | None:
    """A bridge tool's untrusted source: covered externals first, then this turn's remote servers."""
    from .content_provenance import external_source
    return external_source(tool_name) or source_for(tool_name, slugs)
