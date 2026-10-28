"""Allowed Claude Code prompt-cache TTL values, shared by app and bridge.

The app resolves the ``prompt_cache_ttl`` runtime setting per turn; the bridge
re-validates what arrives on the Redis command. Both sides use this allowlist
so an unknown value is dropped (CLI default) instead of reaching the CLI.
"""

from __future__ import annotations

from typing import Any

PROMPT_CACHE_TTLS: tuple[str, ...] = ("5m", "1h")


def normalize_prompt_cache_ttl(raw: Any) -> str | None:
    """Return an allowed TTL, or ``None`` for unset/empty/unknown values."""
    if raw is None:
        return None
    value = str(raw).strip().lower()
    return value if value in PROMPT_CACHE_TTLS else None
