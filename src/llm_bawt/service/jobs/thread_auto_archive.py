"""Archive chat threads idle past a configured age (TASK-261).

A daily housekeeping sweep: an ``active`` thread whose newest user/assistant
message (or, with no messages, its start) is older than ``days`` becomes
``archived``. Nothing is summarized or deleted; the messages stay, and the
owner's next message opens a fresh active thread through the normal
get-or-create path. ``days=0`` disables the sweep.

Each archived row gets ``session_metadata.archived_via = "auto"`` so archive
counts can separate this job from user archives and from task threads that are
born archived.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from sqlalchemy import text

logger = logging.getLogger(__name__)

DEFAULT_DAYS = 30

_ELIGIBLE = """
    s.status = 'active'
    AND s.started_at < (NOW() AT TIME ZONE 'UTC') - make_interval(days => :days)
    AND NOT EXISTS (
        SELECT 1 FROM messages m
        WHERE m.session_id = s.id
          AND m.role IN ('user', 'assistant')
          AND m.timestamp >= EXTRACT(EPOCH FROM NOW() - make_interval(days => :days))
    )
"""

_CANDIDATES_SQL = f"SELECT s.id FROM sessions s WHERE {_ELIGIBLE} ORDER BY s.id"

# FOR UPDATE conflicts with the FOR KEY SHARE lock a message INSERT takes
# through messages.session_id -> sessions.id, so a message whose transaction is
# still open blocks the lock until it commits, and one that starts later waits
# for this archive and is ordered after it (like a manual archive). The
# eligibility recheck is a separate statement, so under READ COMMITTED it takes
# a fresh snapshot after the lock and sees every message committed before it.
_LOCK_SQL = "SELECT 1 FROM sessions WHERE id = :id AND status = 'active' FOR UPDATE"

_ARCHIVE_SQL = f"""
    UPDATE sessions s
    SET status = 'archived',
        ended_at = COALESCE(s.ended_at, CURRENT_TIMESTAMP),
        archived_at = COALESCE(s.archived_at, CURRENT_TIMESTAMP),
        session_metadata = COALESCE(s.session_metadata, '{{}}'::jsonb)
                           || CAST(:stamp AS JSONB)
    WHERE s.id = :id AND {_ELIGIBLE}
    RETURNING s.id, s.bot_id, s.user_id
"""


def run_thread_auto_archive(config: Any, *, days: int = DEFAULT_DAYS) -> dict[str, Any]:
    """Archive stale active threads; return ``{"days", "archived", "sessions"}``."""
    if days <= 0:
        return {"days": days, "archived": 0, "sessions": [], "disabled": True}
    from ...utils.db import get_shared_engine

    engine = get_shared_engine(config)
    params = {"days": int(days), "stamp": json.dumps({"archived_via": "auto"})}
    with engine.connect() as conn:
        candidates = [r[0] for r in conn.execute(text(_CANDIDATES_SQL), params)]
    sessions: list[dict[str, Any]] = []
    for sid in candidates:
        # One short transaction per thread: lock, then recheck in a new snapshot.
        with engine.begin() as conn:
            if conn.execute(text(_LOCK_SQL), {"id": sid}).first() is None:
                continue
            row = conn.execute(text(_ARCHIVE_SQL), {**params, "id": sid}).mappings().first()
            if row is not None:
                sessions.append(dict(row))
    for row in sessions:
        logger.info("thread_archived source=auto session=%s bot=%s user=%s",
                    row["id"], row["bot_id"], row["user_id"])
    return {"days": int(days), "archived": len(sessions), "sessions": sessions}
