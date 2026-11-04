"""A user's steer never overtakes their own earlier queued send (TASK-1412).

A composer send that cannot steer (a ``/new``, images the run can't take, a
send while held) queues as an ordinary delivery behind the running turn. Plain
text sent after it used to steer straight into that turn, ahead of it: a queued
``/new`` then fired after messages the user typed later and reset the session
that had just answered them.

So a composer steer is refused while the same user still has a QUEUED
ordinary chat row for that bot bound for this conversation (no explicit thread,
the steer's own thread, or a reset). The refusal is definite, before any bridge
RPC; the client then sends the message as an ordinary turn, which queues
behind that row in FIFO order. Cancelling the queued row lifts the fence.
"""
from __future__ import annotations

from typing import Any

from sqlalchemy import text

# 409 detail: definite, nothing reached the run; send it as the next turn.
QUEUED_SEND_AHEAD = "queued_send_ahead"


def queued_send_ahead(engine: Any, *, bot_id: str, user_id: str, session_id: str | None) -> str | None:
    """The id of the user's earliest queued send this steer must not pass, else None."""
    if engine is None or getattr(getattr(engine, "dialect", None), "name", None) != "postgresql":
        return None  # durable ordinary admission exists only on PostgreSQL
    with engine.connect() as conn:
        return conn.execute(text("""
            SELECT id FROM inter_bot_deliveries
            WHERE target_bot_id=:bot AND lane='normal' AND status='QUEUED'
              AND author_entity_type='user' AND author_entity_id=:user
              AND metadata_json->'ordinary_chat' IS NOT NULL
              AND (metadata_json->'new_session_intent' IS NOT NULL
                   OR metadata_json->'ordinary_chat'->>'session_id' IS NULL
                   OR metadata_json->'ordinary_chat'->>'session_id' = :session)
            ORDER BY ordinal ASC LIMIT 1
        """), {"bot": bot_id, "user": user_id, "session": session_id or ""}).scalar_one_or_none()
