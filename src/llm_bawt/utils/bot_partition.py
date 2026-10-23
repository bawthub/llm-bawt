"""Canonical legacy slug-to-partition mapping, without storage dependencies."""
import re


def sanitize_table_name(bot_id: str) -> str:
    """Preserve the storage mapping; authorization separately rejects collisions."""
    sanitized = re.sub(r'[^a-z0-9_]', '', bot_id.lower())
    if not sanitized:
        sanitized = "default"
    return sanitized
