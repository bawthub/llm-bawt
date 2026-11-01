"""Display-only receipts for accepted prompts resuming an existing conversation.

The admission transaction owns writes. No message/turn FK: streaming acceptance
precedes the canonical user row, and turn-log retention must not erase markers.
History reads never initialize schema, infer acceptance, or move session pointers.
"""
from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone

from sqlalchemy import Column, Float, MetaData, String, Table, bindparam, select, text

from ..utils.schema import SchemaBootstrapGuard

_metadata = MetaData()
RECEIPTS = Table(
    "session_resumption_receipts", _metadata,
    Column("bot_id", String(128), primary_key=True),
    Column("trigger_message_id", String(128), primary_key=True),
    Column("turn_id", String(128), nullable=False),
    Column("user_id", String(128), nullable=False),
    Column("session_id", String(36), nullable=False),
    Column("previous_session_id", String(36), nullable=False),
    Column("accepted_at", Float, nullable=False),
    Column("session_started_at", Float, nullable=True),
    Column("title", String(300), nullable=True),
)
_guard = SchemaBootstrapGuard()
_FIELDS = ("turn_id", "session_id", "previous_session_id", "accepted_at", "session_started_at", "title")


def ensure_schema(engine) -> None:
    _guard.run(engine, "session-resumption-receipts", lambda conn: _metadata.create_all(conn))


def _epoch(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if isinstance(value, datetime):
        value = value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value
        value = value.timestamp()
    result = float(value)
    return result if math.isfinite(result) else None


def payload(row) -> dict:
    return {key: row[key] for key in _FIELDS}


def record(conn, *, bot_id: str, user_id: str, turn_id: str,
           trigger_message_id: str | None, session_id: str) -> dict | None:
    """Called only for an explicit existing binding, under the admission lock.

    Verify real prior conversation rows, not just a selected/requested session.
    The caller commits this receipt together with BotSessionState.accept.
    """
    if not trigger_message_id:
        return None
    key = {"bot_id": bot_id, "trigger_message_id": trigger_message_id}
    existing = conn.execute(select(RECEIPTS).filter_by(**key)).mappings().one_or_none()
    if existing is not None:
        if (existing["turn_id"], existing["user_id"], existing["session_id"]) != (turn_id, user_id, session_id):
            raise ValueError("session resumption receipt identity conflict")
        return payload(existing)
    previous = conn.execute(text(
        "SELECT last_session_id FROM bot_profiles WHERE slug=:bot"
    ), {"bot": bot_id}).scalar_one_or_none()
    if not previous or previous == session_id:
        return None
    target = conn.execute(text("""
        SELECT s.started_at, s.session_metadata FROM sessions s
        WHERE s.id=:session AND s.bot_id=:bot AND s.user_id=:user AND s.status <> 'deleted'
          AND EXISTS (SELECT 1 FROM messages m
                      WHERE m.bot_id=:bot AND m.session_id=s.id
                        AND m.id <> :trigger AND m.role IN ('user', 'assistant'))
    """), {"session": session_id, "bot": bot_id, "user": user_id, "trigger": trigger_message_id}).mappings().one_or_none()
    if target is None:
        return None
    metadata = target["session_metadata"] or {}
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except (ValueError, TypeError):
            metadata = {}
    title = metadata.get("title") if isinstance(metadata, dict) else None
    title = title.strip()[:300] or None if isinstance(title, str) else None
    row = {**key, "user_id": user_id, "turn_id": turn_id, "session_id": session_id,
           "previous_session_id": previous, "accepted_at": time.time(),
           "session_started_at": _epoch(target["started_at"]), "title": title}
    conn.execute(RECEIPTS.insert().values(**row))
    return payload(row)


def event(receipt: dict, *, bot_id: str, user_id: str, trigger_message_id: str) -> dict:
    """The same committed payload for unified SSE and HTTP SSE metadata."""
    return {"_type": "session_resumed", "bot_id": bot_id, "user_id": user_id,
            "session_id": receipt["session_id"], "turn_id": receipt["turn_id"],
            "trigger_message_id": trigger_message_id, "session_resumption": receipt,
            "ts": receipt["accepted_at"]}


_PAGE = text("""
    SELECT r.* FROM session_resumption_receipts r
    JOIN messages m ON m.bot_id=r.bot_id AND m.id=r.trigger_message_id
                   AND m.session_id=r.session_id AND m.role='user'
    JOIN sessions s ON s.id=r.session_id AND s.bot_id=r.bot_id AND s.user_id=r.user_id
    WHERE r.bot_id=:bot AND r.trigger_message_id IN :ids AND s.status <> 'deleted'
      AND (:owner IS NULL OR s.user_id=:owner)
""").bindparams(bindparam("ids", expanding=True))


def for_messages(engine, *, bot_id: str, messages: list[dict], user_id: str | None = None) -> dict[str, dict]:
    """Read-only bounded-page enrichment, independent from turn-log retention."""
    ids = list(dict.fromkeys(str(m["id"]) for m in messages if m.get("id") and m.get("role") == "user"))
    result = {}
    with engine.connect() as conn:
        for start in range(0, len(ids), 500):
            for row in conn.execute(_PAGE, {"bot": bot_id, "ids": ids[start:start + 500],
                                            "owner": user_id}).mappings():
                result[str(row["trigger_message_id"])] = payload(row)
    return result
