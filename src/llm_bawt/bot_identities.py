"""Non-profile identities used by app-owned scheduled work."""

SYSTEM_BOT_ID = "system"
WILDCARD_BOT_ID = "*"
RESERVED_JOB_BOT_IDS = frozenset({SYSTEM_BOT_ID, WILDCARD_BOT_ID})
# Job selectors protect shared-row purge, not unregistered storage partitions.


def seed_system_author(conn) -> None:
    """Seed presentation only: System is not a runnable bot or a new role.

    The typed conflict key leaves the existing USER:system identity untouched
    and preserves any deliberate presentation edits on repeat bootstrap.
    """
    from sqlalchemy import text

    conn.execute(text("""
        INSERT INTO entity_profiles
            (entity_type, entity_id, display_name, description, created_at, updated_at)
        VALUES ('BOT', :id, 'System', 'Scheduled system actions',
                CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        ON CONFLICT (entity_type, entity_id) DO NOTHING
    """), {"id": SYSTEM_BOT_ID})
