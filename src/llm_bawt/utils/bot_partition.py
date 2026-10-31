"""Canonical legacy slug-to-partition mapping, without storage dependencies."""
import re


def sanitize_table_name(bot_id: str) -> str:
    """Preserve the storage mapping; authorization separately rejects collisions."""
    sanitized = re.sub(r'[^a-z0-9_]', '', bot_id.lower())
    if not sanitized:
        sanitized = "default"
    return sanitized


def lock_bot_storage(conn, storage_id: str) -> None:
    """Serialize create/purge of any slugs that share a partition identity."""
    from sqlalchemy import text

    conn.execute(text("SELECT pg_advisory_xact_lock(hashtext('bot-storage'), hashtext(:id))"),
                 {"id": storage_id})
