"""Read-only turn-duration percentiles, grouped by current bot/tag membership."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from sqlalchemy import text

from llm_bawt.bot_tags.facade import BotTags, TagPolicyUnavailable


_STATS_SQL = text("""
    WITH eligible AS MATERIALIZED (
        SELECT lower(bot_id) AS bot_id, latency_ms
        FROM turn_logs
        WHERE created_at >= :since AND created_at < :until
          AND ended_at IS NOT NULL AND ended_at <= :until
          AND latency_ms >= 0 AND latency_ms < 'Infinity'::double precision
          AND lower(bot_id) = ANY(CAST(:bots AS text[]))
          AND (:user_id IS NULL OR user_id = :user_id)
    ), samples AS (
        SELECT 'bot' AS kind, bot_id AS key, latency_ms FROM eligible
        UNION ALL
        SELECT 'tag', membership.tag, eligible.latency_ms
        FROM eligible
        JOIN jsonb_to_recordset(CAST(:memberships AS jsonb))
            AS membership(bot_id text, tag text) USING (bot_id)
    )
    SELECT kind, key, count(*) AS count,
           percentile_cont(0.50) WITHIN GROUP (ORDER BY latency_ms) AS p50_ms,
           percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms) AS p95_ms,
           max(latency_ms) AS max_ms
    FROM samples GROUP BY kind, key ORDER BY kind, key
""")


def load_turn_duration_stats(
    engine, *, days: int = 7, tag: str | None = None,
    bot_id: str | None = None, user_id: str | None = None,
    now: datetime | None = None,
) -> dict:
    """Aggregate retained, finished turns without loading prompts or raw samples.

    Tag filters select a cohort using one validated current snapshot, not tags
    at turn time. A tag percentile pools that cohort's turns, not bot percentiles.
    Private/unknown/no-global-search bots are never part of this aggregate.
    """
    if not 1 <= days <= 7:
        raise ValueError("days must be between 1 and 7")
    until = now or datetime.now(timezone.utc)
    since = until - timedelta(days=days)
    with engine.connect() as conn, conn.begin():
        conn.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"))
        conn.execute(text("SET LOCAL statement_timeout = '5000ms'"))
        conn.execute(text("SET LOCAL lock_timeout = '500ms'"))
        snapshot = BotTags(engine).snapshot(conn)
        if not snapshot.available:
            raise TagPolicyUnavailable("bot tags unavailable")
        if tag is not None and tag not in snapshot.catalog:
            raise ValueError("unknown bot tag")
        bots = sorted(
            bot for bot in snapshot.searchable_bots
            if (bot_id is None or bot == bot_id.strip().lower())
            and (tag is None or snapshot.has(bot, tag))
        )
        memberships = [
            {"bot_id": bot, "tag": assigned}
            for bot in bots for assigned in sorted(snapshot.tags[bot])
        ]
        rows = conn.execute(_STATS_SQL, {
            "since": since, "until": until, "bots": bots,
            "memberships": json.dumps(memberships), "user_id": user_id,
        }).mappings().all() if bots else []
    by_bot, by_tag = [], []
    for row in rows:
        item = {key: row[key] for key in ("count", "p50_ms", "p95_ms", "max_ms")}
        if row["kind"] == "bot":
            by_bot.append({
                "bot_id": row["key"], "tags": sorted(snapshot.tags[row["key"]]), **item,
            })
        else:
            by_tag.append({"tag": row["key"], **item})
    return {
        "days": days, "window_start": since.isoformat(), "window_end": until.isoformat(),
        "tag": tag, "bot_id": bot_id, "user_id": user_id,
        "tag_membership": "current_snapshot",
        "sample": "retained_finished_turns_with_finite_nonnegative_latency_ms",
        "percentile_method": "continuous",
        "by_bot": by_bot, "by_tag": by_tag,
    }
