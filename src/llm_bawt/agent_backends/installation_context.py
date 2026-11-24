"""Where this installation's web UI lives, for agent prompts (TASK-2523).

An agent on an installed tenant otherwise has to guess its own BawtHub
address (an internal service port, a localhost guess). The deployment already
knows the origin its people open (``LLM_BAWT_PUBLIC_ORIGIN``), so agent
backends state it once in the composed system prompt. The block is fixed per
installation, so pinned prompts and prompt caching stay byte-stable.
"""

from __future__ import annotations

import os
from urllib.parse import urlsplit

from ..media.serializers import public_origin


def installation_context_block() -> str:
    """Return the installation block, or "" when no usable origin is set."""
    origin = public_origin()
    parts = urlsplit(origin)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        return ""
    lines = [
        "## This installation",
        f"This BawtHub's web UI is at {origin}, the address its people open. "
        "If it asks you to sign in, you have no login there; never ask anyone for a password.",
    ]
    # Same switch that registers the tool (mcp_server/inspection_tools.py).
    if os.getenv("BAWTHUB_INSPECTION_KEY_EXPORT_FILE", "").strip():
        lines.append(
            "To see your linked Tasks as the owner sees them, use inspect_bawthub_tasks "
            "once the owner authorizes it in Settings → Profile → Bot inspection."
        )
    return "\n".join(lines)


def with_installation_context(system_prompt: str) -> str:
    """Append the installation block after the rendered prompt."""
    block = installation_context_block()
    if not block:
        return system_prompt
    sp = (system_prompt or "").rstrip()
    return f"{sp}\n\n{block}" if sp else block
