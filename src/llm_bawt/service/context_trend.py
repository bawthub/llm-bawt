"""Bounded first-SDK-turn context snapshots from persisted turn logs."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping

from sqlalchemy import text
from sqlmodel import Session


def _count(value: Any) -> int | None:
    return value if type(value) is int and value >= 0 else None


def _snapshot(row: Mapping[str, Any]) -> dict | None:
    try:
        usage = json.loads(row["token_usage_json"])
    except (TypeError, ValueError):
        return None
    if not isinstance(usage, dict):
        return None
    breakdown = usage.get("context_breakdown")
    if not isinstance(breakdown, dict):
        return None
    start = breakdown.get("session_start")
    if start not in {"clean", "seeded"}:
        return None
    categories = []
    raw_categories = breakdown.get("categories")
    if not isinstance(raw_categories, list):
        return None
    for item in raw_categories[:32]:
        if not isinstance(item, dict):
            continue
        name, tokens = item.get("name"), _count(item.get("tokens"))
        if isinstance(name, str) and name.strip() and tokens is not None:
            categories.append({
                "name": name[:160], "tokens": tokens,
                "is_deferred": item.get("isDeferred") is True,
            })
    if not categories:
        return None
    messages = breakdown.get("message_breakdown")
    tool_results = _count(messages.get("toolResultTokens")) if isinstance(messages, dict) else None
    created_at = row["created_at"]
    return {
        "turn_id": row["id"], "bot_id": row["bot_id"],
        "created_at": created_at.isoformat() if isinstance(created_at, datetime) else str(created_at),
        "session_start": start,
        "resident_tokens": _count(usage.get("resident_tokens")),
        "resident_source": usage.get("resident_source") if isinstance(usage.get("resident_source"), str) else None,
        "cli_total_tokens": _count(breakdown.get("cli_total_tokens")),
        "tool_result_tokens": tool_results,
        "categories": categories,
    }


def load_context_trend(engine, *, user_id: str, bot_id: str, days: int, limit: int) -> dict:
    """Return one bot's series plus each bot's latest proven first turn.

    The session table stores only the *current* SDK resume key. Earlier SDK
    boundaries cannot be reconstructed from thread timestamps, so only rows
    explicitly marked by the bridge are eligible. Both queries are bounded by
    user, date, marker and result count; prompts never leave the database.
    """
    params = {
        "user_id": user_id,
        "bot_id": bot_id,
        "since": datetime.now(timezone.utc) - timedelta(days=days),
        "marker": '%"session_start"%',
        "limit": limit,
    }
    eligible = """
        user_id = :user_id AND created_at >= :since AND ended_at IS NOT NULL
        AND bot_id IS NOT NULL AND token_usage_json LIKE :marker
    """
    with Session(engine) as session:
        latest_rows = session.execute(text(f"""
            WITH ranked AS (
                SELECT id, bot_id, created_at, token_usage_json,
                    ROW_NUMBER() OVER (PARTITION BY bot_id ORDER BY created_at DESC, id DESC) AS rn
                FROM turn_logs WHERE {eligible}
            )
            SELECT id, bot_id, created_at, token_usage_json
            FROM ranked WHERE rn = 1 ORDER BY bot_id LIMIT 200
        """), params).mappings().all()
        series_rows = session.execute(text(f"""
            SELECT id, bot_id, created_at, token_usage_json FROM turn_logs
            WHERE {eligible} AND bot_id = :bot_id
            ORDER BY created_at DESC, id DESC LIMIT :limit
        """), params).mappings().all()
    latest = [item for row in latest_rows if (item := _snapshot(row)) is not None]
    series = [item for row in reversed(series_rows) if (item := _snapshot(row)) is not None]
    return {"latest": latest, "series": series, "days": days, "bot_id": bot_id}
