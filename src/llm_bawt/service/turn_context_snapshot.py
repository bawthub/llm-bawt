"""Latest turn's context snapshot on its owning bot/user session.

The durable turn log remains the audit record. This session-scoped projection is
written when the trigger message identifies its owning session; a linked
admission records unknown usage, while terminal completion records measured
usage. A bot profile is shared by users and cannot represent independent contexts.
"""

from __future__ import annotations

import json

from sqlalchemy import text

from ..memory.postgresql import _sanitize_table_name


def record_turn_context(session, row) -> None:
    """Project a turn into its thread without overwriting a newer turn.

    Admission invalidates a gauge only when its message already identifies the
    owning session; otherwise the previous completed snapshot remains until
    the terminal writer can project to the linked owner. An older in-flight
    turn completing later cannot overwrite a newer admission or completion.
    """
    engine = getattr(session, "bind", None) or session.engine
    if (engine.dialect.name != "postgresql"
            or not (row.bot_id and row.user_id and row.trigger_message_id)):
        return
    usage = None
    if row.token_usage_json:
        try:
            usage = json.loads(row.token_usage_json)
        except (TypeError, ValueError):
            pass
    snapshot = {
        "turn_id": row.id,
        "created_at": row.created_at.isoformat(),
        "ended_at": row.ended_at.isoformat() if row.ended_at else None,
        "usage": usage if isinstance(usage, dict) else None,
        "agent_session_key": row.agent_session_key,
    }
    params = {
        "storage_bot": _sanitize_table_name(row.bot_id),
        "bot": row.bot_id, "user": row.user_id,
        "trigger": row.trigger_message_id,
        "created": row.created_at, "turn_id": row.id,
        "snapshot": json.dumps(snapshot),
    }
    newer_guard = """
        AND (CAST(:created AS timestamptz), CAST(:turn_id AS text)) >= (
            COALESCE((s.session_metadata->'context_stats'->>'created_at')::timestamptz,
                     '-infinity'::timestamptz),
            COALESCE(s.session_metadata->'context_stats'->>'turn_id', '')
        )
    """
    # An unpersisted trigger may belong to an explicitly selected archived task
    # session. Never guess that the active thread owns it: admission can defer
    # its unknown projection until the message exists; terminal completion then
    # stamps the exact owning session. The reset guard blocks in-flight work.
    session.execute(text("""
        UPDATE sessions s SET session_metadata=jsonb_set(
            COALESCE(s.session_metadata, '{}'::jsonb),
            '{context_stats}', CAST(:snapshot AS jsonb)
        )
        FROM messages m
        WHERE m.bot_id=:storage_bot AND m.id=:trigger
          AND s.id=m.session_id AND s.bot_id=:bot AND s.user_id=:user
    """ + newer_guard), params)
