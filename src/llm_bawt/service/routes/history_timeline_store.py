"""Bounded, owned timeline repository. All SQL for TASK-985 lives here.

Normal page: grouped segments + one first-message query + summary availability
(<=3 statements). All-history overview uses one monthly query instead. Sparse
recent ranges add a previous-active-day lookup and a grouped read (<=5). Count does not grow
with sessions or links; every message must join an owned session.
"""

from __future__ import annotations

from fastapi import HTTPException
from sqlalchemy import text

from ...sessions.first_message import fetch_first_messages, first_message_params, given_title_sql
from .history_scope import MAX_TIMELINE_ANCHORS, owned_message_scope
from .history_timeline_summaries import summary_availability

MAX_GROUPS = 2000
SEGMENT_LIMIT = 64
MAX_OVERVIEW_MONTHS = 240
VISIBLE = "m.role NOT IN ('system', 'summary')"


def timeline_rows(engine, table: str, bot_id: str, user_id: str, session_id: str | None,
                  tz: str, start: float, end: float, *, recent: bool = False,
                  overview_only: bool = False, overview_before: float | None = None):
    """Return group/meta/prompts/overview plus actual lower bound (UTC epoch)."""
    base = {"bot_id": bot_id, "user_id": user_id, "session_id": session_id, "tz": tz,
            "start": start, "end": end, "limit": MAX_GROUPS + 1,
            "segment_limit": SEGMENT_LIMIT, **first_message_params()}
    owner = owned_message_scope()
    session = " AND m.session_id = :session_id" if session_id else ""
    group_sql = text(f"""
        WITH groups AS MATERIALIZED (
            SELECT (to_timestamp(m.timestamp) AT TIME ZONE :tz)::date AS day,
                   m.session_id, count(*) AS n,
                   count(*) FILTER (WHERE m.role = 'user') AS prompts,
                   min(m.timestamp) AS first_ts, max(m.timestamp) AS last_ts,
                   (array_agg(m.id ORDER BY m.timestamp, m.id))[1] AS first_id
            FROM {table} m
            WHERE {VISIBLE} AND {owner}{session}
              AND m.timestamp >= :start AND m.timestamp < :end
            GROUP BY 1, 2
        ), ranked_days AS (
            SELECT day, sum(count(*)) OVER (ORDER BY day DESC) AS group_total,
                   first_value(count(*)) OVER (ORDER BY day DESC) AS newest_groups
            FROM groups GROUP BY day
        )
        SELECT groups.*, s.status, {given_title_sql('s')} AS title, EXISTS (
            SELECT 1 FROM {table} earlier
            WHERE earlier.session_id = groups.session_id
              AND earlier.role NOT IN ('system', 'summary')
              AND (earlier.timestamp, earlier.id) < (groups.first_ts, groups.first_id)
        ) AS continued, EXISTS (
            SELECT 1 FROM ranked_days older
            WHERE older.group_total > greatest(:segment_limit, older.newest_groups)
        ) AS has_more
        FROM groups JOIN sessions s ON s.id = groups.session_id
             AND s.bot_id = :bot_id AND s.user_id = :user_id
        JOIN ranked_days page ON page.day = groups.day
        WHERE page.group_total <= greatest(:segment_limit, page.newest_groups)
        ORDER BY groups.day, first_ts, first_id LIMIT :limit
    """)
    with engine.connect() as conn:
        groups = [] if overview_only else [dict(row) for row in conn.execute(group_sql, base).mappings()]
        if (recent and not overview_only and not any(g.get("has_more") for g in groups)
                and len({str(g['day']) for g in groups}) < 7):
            # An initial sparse preset may extend, but its returned scope must
            # remain a valid <=90-local-day pinned scope for later page requests.
            from datetime import datetime, timedelta
            from zoneinfo import ZoneInfo
            from .history_scope import local_day_bounds
            last_day = datetime.fromtimestamp(end - 0.000001, ZoneInfo(tz)).date()
            earliest, _ = local_day_bounds(last_day - timedelta(days=89), last_day, tz)
            previous = conn.execute(text(f"""
                SELECT DISTINCT (to_timestamp(m.timestamp) AT TIME ZONE :tz)::date AS day
                FROM {table} m WHERE {VISIBLE} AND {owner}{session}
                  AND m.timestamp < :start AND m.timestamp >= :earliest
                ORDER BY day DESC LIMIT 7
            """), {**base, "earliest": earliest}).mappings().all()
            if previous:
                oldest = previous[min(6 - len({str(g['day']) for g in groups}), len(previous) - 1)]['day']
                start, _ = local_day_bounds(oldest, oldest, tz)
                base['start'] = start
                groups = [dict(row) for row in conn.execute(group_sql, base).mappings()]
        if len(groups) > MAX_GROUPS:
            raise HTTPException(status_code=413, detail="Too many timeline segments; narrow the range")
        ids = sorted({str(g['session_id']) for g in groups})
        # TASK-2061: every grouped session already joined an owned session, so
        # the shared first-message rule runs for exactly those (one statement).
        prompts = fetch_first_messages(
            conn, [{"id": sid, "bot_id": bot_id, "user_id": user_id} for sid in ids],
            messages_table=table)
        months = [] if not overview_only else [dict(row) for row in conn.execute(text(f"""
            SELECT date_trunc('month', to_timestamp(m.timestamp) AT TIME ZONE :tz)::date AS month,
                   count(*) AS messages
            FROM {table} m WHERE {VISIBLE} AND {owner}{session}
              AND (:overview_before IS NULL OR m.timestamp < :overview_before)
            GROUP BY 1 ORDER BY 1 DESC LIMIT :month_limit
        """), {**base, 'month_limit': MAX_OVERVIEW_MONTHS + 1,
                 'overview_before': overview_before}).mappings()]
        availability = summary_availability(conn, table, bot_id, user_id, groups)
        for group in groups:
            summary = availability.get((str(group["day"]), str(group["session_id"])), {})
            group["summary_count"] = int(summary.get("summary_count", 0))
            group["summary_revision"] = summary.get("summary_revision", "0")
    meta = {str(g['session_id']): {'title': g['title'], 'status': g['status']} for g in groups}
    return groups, meta, prompts, months, start


def session_detail_row(conn, table: str, bot_id: str, user_id: str, session_id: str,
                       start: float | None, end: float | None) -> dict | None:
    """Full metadata only for an explicitly selected, owned, active-in-range thread."""
    bounds = ""
    if start is not None:
        bounds += " AND m.timestamp >= :start"
    if end is not None:
        bounds += " AND m.timestamp < :end"
    query = text(f"""
        SELECT active.*, s.status, {given_title_sql('s')} AS title
        FROM (
            SELECT m.session_id AS id, count(*) AS message_count,
                   min(m.timestamp) AS first_ts, max(m.timestamp) AS last_ts,
                   (array_agg(m.id ORDER BY m.timestamp, m.id))[1] AS first_message_id
            FROM {table} m
            WHERE {VISIBLE} AND {owned_message_scope()} AND m.session_id = :session_id
                  {bounds}
            GROUP BY m.session_id
        ) active JOIN sessions s ON s.id = active.id
                  AND s.bot_id = :bot_id AND s.user_id = :user_id
    """)
    rows = conn.execute(query, {"bot_id": bot_id, "user_id": user_id,
                               "session_id": session_id, "start": start, "end": end,
                               **first_message_params()}).mappings().all()
    if not rows:
        return None
    row = dict(rows[0])
    first = fetch_first_messages(
        conn, [{"id": session_id, "bot_id": bot_id, "user_id": user_id}],
        messages_table=table)
    row.update(first.get(session_id, {}))
    return row


def anchor_rows(engine, table: str, bot_id: str, user_id: str,
                session_id: str | None, message_ids: list[str]) -> list[dict]:
    """One batch read; missing, deleted, and foreign IDs return no anchor."""
    if len(message_ids) > MAX_TIMELINE_ANCHORS:
        raise HTTPException(status_code=413, detail="Too many timeline anchors")
    if not message_ids:
        return []
    scope = " AND m.session_id = :session_id" if session_id else ""
    query = text(f"SELECT m.id, m.session_id, m.timestamp AS ts FROM {table} m "
                 f"WHERE m.id = ANY(:ids) AND {VISIBLE} AND {owned_message_scope()}{scope} "
                 "ORDER BY m.timestamp, m.id")
    with engine.connect() as conn:
        return [dict(row) for row in conn.execute(query, {'bot_id': bot_id, 'user_id': user_id,
                        'session_id': session_id, 'ids': list(set(message_ids))}).mappings()]
