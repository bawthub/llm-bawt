"""Transactional change hints for bounded bot-policy caches (no table reads)."""
from sqlalchemy import text

POLICY_CHANNEL = "llm_bot_policy_changed"


def policy_changed(conn) -> None:
    """Deliver only on commit, including borrowed sessions; rollback emits nothing.

    This SELECT calls PostgreSQL's notification writer; it reads no policy,
    ownership or revision table. PostgreSQL coalesces identical notifications
    within a transaction. The payload scopes listeners to the owned schema and
    never includes bot identities. Background reconciliation is the lost-hint
    backstop, not a request-side read.
    """
    if conn.dialect.name == "postgresql":
        conn.execute(text("SELECT pg_notify(:channel, current_schema())"),
                     {"channel": POLICY_CHANNEL})
