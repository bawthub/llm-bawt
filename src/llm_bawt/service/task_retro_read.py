"""Bounded, read-only usage projection. No prompts, results or task mutations."""
from __future__ import annotations

import json
from datetime import datetime

from sqlalchemy import text

from .task_retro_accounting import usage_facts

MAX_RETRO_TURNS = 20_000
MAX_RETRO_LINKS = 20_000

_READ = text("""
    WITH links AS (
        SELECT bot_id, trigger_message_id, array_agg(DISTINCT task_ref) AS refs
        FROM jsonb_to_recordset(CAST(:links AS jsonb))
             AS l(bot_id text, trigger_message_id text, task_ref text)
        GROUP BY bot_id, trigger_message_id
    )
    SELECT t.id, t.bot_id, t.trigger_message_id, t.created_at, t.ended_at,
           t.status, t.model, t.task_ref, t.task_attribution_source,
           t.token_usage_json, d.task_id AS delivery_task, d.message_kind,
           l.refs AS linked_tasks
    FROM turn_logs t
    LEFT JOIN inter_bot_deliveries d
      ON d.turn_id=t.id AND d.target_bot_id=t.bot_id
     AND d.payload_json->>'user'=t.user_id
    LEFT JOIN links l ON l.bot_id=t.bot_id AND l.trigger_message_id=t.trigger_message_id
    WHERE t.user_id=:user AND t.bot_id=ANY(CAST(:bots AS text[]))
      AND t.status <> 'reserved'
      AND (CAST(:start AS timestamptz) IS NULL OR t.created_at>=CAST(:start AS timestamptz))
      AND (CAST(:end AS timestamptz) IS NULL OR t.created_at<CAST(:end AS timestamptz))
      AND (CAST(:refs AS text[]) IS NULL
           OR t.task_ref=ANY(CAST(:refs AS text[]))
           OR d.task_id=ANY(CAST(:refs AS text[]))
           OR l.refs && CAST(:refs AS text[]))
    ORDER BY t.created_at, t.id
    LIMIT :limit
""")


def attribution(row):
    """Strong explicit evidence first; never charge one turn to several tasks."""
    ref, source = row["task_ref"], row["task_attribution_source"]
    linked = row["linked_tasks"] or []
    if ref and source == "association":
        return ref, source
    if len(linked) > 1:
        return None, "ambiguous"
    if len(linked) == 1:
        return linked[0], "task_turn_link"
    if row["delivery_task"]:
        return row["delivery_task"], "delivery"
    return ref, source or "unattributed"


def read_retro_turns(conn, *, user, bots, links, refs=None, start=None, end=None):
    rows = conn.execute(_READ, {
        "user": user, "bots": bots, "links": json.dumps(links),
        "refs": refs, "start": start, "end": end, "limit": MAX_RETRO_TURNS + 1,
    }).mappings().all()
    truncated = len(rows) > MAX_RETRO_TURNS
    rows = rows[:MAX_RETRO_TURNS]
    turns = []
    for row in rows:
        ref, source = attribution(row)
        turns.append({
            "id": row["id"], "bot_id": row["bot_id"],
            "trigger_message_id": row["trigger_message_id"],
            "created_at": row["created_at"].isoformat(),
            "ended_at": row["ended_at"].isoformat() if row["ended_at"] else None,
            "task_ref": ref, "attribution_source": source,
            "linked_tasks": row["linked_tasks"] or [],
            "candidate_task_refs": list(dict.fromkeys(filter(None, [
                row["task_ref"], row["delivery_task"], *(row["linked_tasks"] or []),
            ]))),
            "status": row["status"], "model": row["model"],
            "message_kind": row["message_kind"], **usage_facts(row["token_usage_json"]),
        })
    return {"turns": turns, "truncated": truncated, "limit": MAX_RETRO_TURNS}


def read_retro_messages(conn, *, user, bots, refs=None, start=None, end=None):
    """Structured, task-indexed deliveries only; includes steering and queued rows."""
    rows = conn.execute(text("""
        SELECT id, task_id, sender_bot_id, target_bot_id, created_at
        FROM inter_bot_deliveries
        WHERE payload_json->>'user'=:user AND task_id IS NOT NULL
          AND sender_bot_id=ANY(CAST(:bots AS text[]))
          AND target_bot_id=ANY(CAST(:bots AS text[]))
          AND (CAST(:refs AS text[]) IS NULL OR task_id=ANY(CAST(:refs AS text[])))
          AND (CAST(:start AS timestamptz) IS NULL OR created_at>=CAST(:start AS timestamptz))
          AND (CAST(:end AS timestamptz) IS NULL OR created_at<CAST(:end AS timestamptz))
        ORDER BY created_at, id LIMIT :limit
    """), {"user": user, "bots": bots, "refs": refs, "start": start,
           "end": end, "limit": MAX_RETRO_TURNS + 1}).mappings().all()
    return {
        "messages": [{"id": row["id"], "task_ref": row["task_id"],
                      "created_at": row["created_at"].isoformat()}
                     for row in rows[:MAX_RETRO_TURNS]],
        "messages_truncated": len(rows) > MAX_RETRO_TURNS,
    }


def cached_claude_quota():
    """Only an existing snapshot; this read never refreshes provider usage."""
    from .usage import get_cached

    snapshot = get_cached("claude")
    if snapshot is None or snapshot.status != "ok" or not snapshot.fetched_at:
        return None
    for window in snapshot.limits:
        if (window.id == "weekly_all" and window.resets_at is not None
                and window.used_pct is not None
                and window.resets_at - 604800 <= snapshot.fetched_at < window.resets_at):
            return {"start": window.resets_at - 604800, "end": window.resets_at,
                    "fetched_at": snapshot.fetched_at, "utilization_pct": window.used_pct}
    return None


def window_datetime(seconds):
    from datetime import timezone
    return datetime.fromtimestamp(seconds, tz=timezone.utc)
