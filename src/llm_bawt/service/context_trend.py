"""Bounded first-SDK-turn context snapshots from persisted turn logs."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping

from sqlalchemy import text
from sqlmodel import Session

MAX_CANDIDATES = 5000
MAX_LATEST_BOTS = 200


def _count(value: Any) -> int | None:
    return value if type(value) is int and value >= 0 else None


def _marker(raw: Any) -> tuple[dict, dict, str] | None:
    """Parse an explicit clean/seeded ``context_breakdown.session_start`` marker."""
    try:
        usage = json.loads(raw)
    except (TypeError, ValueError):
        return None
    if not isinstance(usage, dict):
        return None
    breakdown = usage.get("context_breakdown")
    if not isinstance(breakdown, dict):
        return None
    start = breakdown.get("session_start")
    if not isinstance(start, str) or start not in {"clean", "seeded"}:
        return None
    return usage, breakdown, start


def _earliest_marker(session: Session, params: dict) -> datetime | None:
    """Oldest structurally valid marker for the user, scanned oldest-first.

    The LIKE prefilter only narrows candidates; each is validated so stray
    text can't backdate the tracking boundary. Bounded by MAX_CANDIDATES.
    """
    cursor: tuple[Any, str] | None = None
    scanned = 0
    while scanned < MAX_CANDIDATES:
        after = "" if cursor is None else (
            "AND (created_at > :after_at OR (created_at = :after_at AND id > :after_id))"
        )
        batch = session.execute(text(f"""
            SELECT id, created_at, token_usage_json FROM turn_logs
            WHERE user_id = :user_id AND bot_id IS NOT NULL AND token_usage_json LIKE :marker
            {after}
            ORDER BY created_at ASC, id ASC LIMIT :batch
        """), {
            **params, "batch": min(500, MAX_CANDIDATES - scanned),
            **({} if cursor is None else {"after_at": cursor[0], "after_id": cursor[1]}),
        }).mappings().all()
        if not batch:
            return None
        for row in batch:
            if _marker(row["token_usage_json"]) is not None:
                return row["created_at"]
        scanned += len(batch)
        cursor = (batch[-1]["created_at"], batch[-1]["id"])
    return None


def _snapshot(row: Mapping[str, Any]) -> dict | None:
    parsed = _marker(row["token_usage_json"])
    if parsed is None:
        return None
    usage, breakdown, start = parsed
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
    explicitly marked by the bridge are eligible. Candidates are bounded by
    user, date, marker and a hard cap, then validated before latest/series
    selection. Prompts never leave the database.
    """
    params = {
        "user_id": user_id,
        "bot_id": bot_id,
        "since": datetime.now(timezone.utc) - timedelta(days=days),
        "marker": '%"session_start"%',
        "candidate_limit": MAX_CANDIDATES + 1,
    }
    eligible = """
        user_id = :user_id AND created_at >= :since AND ended_at IS NOT NULL
        AND bot_id IS NOT NULL AND token_usage_json LIKE :marker
    """
    with Session(engine) as session:
        rows = session.execute(text(f"""
            SELECT id, bot_id, created_at, token_usage_json FROM turn_logs
            WHERE {eligible}
            ORDER BY created_at DESC, id DESC LIMIT :candidate_limit
        """), params).mappings().all()
        # Earliest marker for this user on any bot: when the metric began to exist.
        tracking_since = _earliest_marker(session, params)
        if tracking_since is None:
            # Scan cap hit before a valid marker: fall back to the oldest valid
            # one already loaded. Later than the truth, so it over-discloses.
            valid = [r["created_at"] for r in rows[:MAX_CANDIDATES] if _marker(r["token_usage_json"])]
            tracking_since = min(valid) if valid else None
        # Completed turns of this bot inside the window that predate any marker.
        untracked = None if tracking_since is None else session.execute(text("""
            SELECT COUNT(*) FROM turn_logs
            WHERE user_id = :user_id AND bot_id = :bot_id AND ended_at IS NOT NULL
              AND created_at >= :since AND created_at < :tracked
        """), {**params, "tracked": tracking_since}).scalar() or 0
    truncated = len(rows) > MAX_CANDIDATES
    latest_by_bot: dict[str, dict] = {}
    newest_series: list[dict] = []
    eligible_for_bot = 0
    for row in rows[:MAX_CANDIDATES]:
        item = _snapshot(row)
        if item is None:
            continue
        latest_by_bot.setdefault(item["bot_id"], item)
        if item["bot_id"] == bot_id:
            eligible_for_bot += 1
            if len(newest_series) < limit:
                newest_series.append(item)
    latest = [latest_by_bot[bot] for bot in sorted(latest_by_bot)[:MAX_LATEST_BOTS]]
    if isinstance(tracking_since, datetime):
        tracking_iso = tracking_since.isoformat()
    else:
        tracking_iso = str(tracking_since) if tracking_since else None
    return {
        "latest": latest, "series": list(reversed(newest_series)),
        "days": days, "bot_id": bot_id,
        "truncated": truncated or len(latest_by_bot) > MAX_LATEST_BOTS,
        "coverage": {
            # None means no marker exists yet for this user: nothing is measurable.
            "tracking_since": tracking_iso,
            "window_start": params["since"].isoformat(),
            # Lower bound when the candidate scan was truncated.
            "eligible": eligible_for_bot,
            "shown": len(newest_series),
            "untracked_turns": None if untracked is None else int(untracked),
            "eligible_is_lower_bound": truncated,
        },
    }
