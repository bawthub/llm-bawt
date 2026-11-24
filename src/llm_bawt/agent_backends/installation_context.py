"""Where this installation's web UI lives, for agent prompts (TASK-2523).

An agent on an installed tenant otherwise has to guess its own BawtHub
address (an internal service port, a localhost guess). The deployment already
knows the origin its people open (``LLM_BAWT_PUBLIC_ORIGIN``), so agent
backends state it once in the composed system prompt. The wording comes from
the prompt registry (``agents.installation_context_template`` and
``agents.inspection_hint``); only the origin and the inspection switch are
deployment facts. Pinned sessions keep the bytes they started with.
"""

from __future__ import annotations

import logging
import os
from urllib.parse import urlsplit

from ..media.serializers import public_origin

logger = logging.getLogger(__name__)


def installation_origin(value: str | None = None) -> str:
    """Return ``value`` as a bare ``http(s)://host[:port]`` origin, or ""."""
    raw = (public_origin() if value is None else value).strip()
    try:
        parts = urlsplit(raw)
        port = parts.port  # raises ValueError on a malformed port
        hostname = parts.hostname
    except ValueError:
        return ""
    if parts.scheme not in ("http", "https") or not hostname:
        return ""
    if parts.username is not None or parts.password is not None:
        return ""
    if parts.path not in ("", "/") or parts.query or parts.fragment:
        return ""
    host = f"[{hostname}]" if ":" in hostname else hostname
    return f"{parts.scheme}://{host}" + (f":{port}" if port is not None else "")


def _registry_body(key: str, fallback: str) -> str:
    try:
        from ..prompt_registry import get_prompt_resolver

        resolved = get_prompt_resolver().resolve(key)
        if resolved and resolved.body:
            return resolved.body
    except Exception:  # noqa: BLE001 — a registry outage must not break a turn
        logger.debug("prompt registry unavailable for %s; using default", key, exc_info=True)
    return fallback


def installation_context_block() -> str:
    """Return the installation block, or "" when no usable origin is set."""
    from ..chat_prompt_defaults import INSPECTION_HINT, INSTALLATION_CONTEXT_TEMPLATE

    origin = installation_origin()
    if not origin:
        return ""
    template = _registry_body("agents.installation_context_template", INSTALLATION_CONTEXT_TEMPLATE)
    try:
        block = template.format(origin=origin)
    except (KeyError, IndexError, ValueError):
        block = INSTALLATION_CONTEXT_TEMPLATE.format(origin=origin)
    # Same switch that registers the tool (mcp_server/inspection_tools.py).
    if os.getenv("BAWTHUB_INSPECTION_KEY_EXPORT_FILE", "").strip():
        hint = _registry_body("agents.inspection_hint", INSPECTION_HINT).strip()
        if hint:
            block = f"{block.rstrip()}\n{hint}"
    return block.strip()


def with_installation_context(system_prompt: str) -> str:
    """Append the installation block after the rendered prompt."""
    block = installation_context_block()
    if not block:
        return system_prompt
    sp = (system_prompt or "").rstrip()
    return f"{sp}\n\n{block}" if sp else block
