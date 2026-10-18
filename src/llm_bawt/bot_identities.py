"""Non-profile identities used by app-owned scheduled work."""

SYSTEM_BOT_ID = "system"
WILDCARD_BOT_ID = "*"
RESERVED_JOB_BOT_IDS = frozenset({SYSTEM_BOT_ID, WILDCARD_BOT_ID})
# Job selectors protect shared-row purge, not unregistered storage partitions.
