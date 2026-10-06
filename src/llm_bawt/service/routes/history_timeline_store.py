"""Bounded, owned timeline repository. All SQL for TASK-985 lives here.

Normal 30d: grouped segments + at most one untitled-label query + monthly
calendar overview (<=3 statements). Sparse recent ranges add a previous-active-
day lookup and a second grouped read (<=5). SQL statement count does not grow
with sessions or links; every message must join an owned session.
"""

from __future__ import annotations

from fastapi import HTTPException
from sqlalchemy import text

from .history_scope import MAX_TIMELINE_ANCHORS, owned_message_scope

MAX_GROUPS = 2000
MAX_OVERVIEW_MONTHS = 240
VISIBLE = "m.role NOT IN ('system', 'summary')"


def timeline_rows(engine, table: str, bot_id: str, user_id: str, session_id: str | None,
                  tz: str, start: float, end: float, *, recent: bool = False,
                  overview_only: bool = False, overview_before: float | None = None):
    """Return group/meta/prompts/overview plus actual lower bound (UTC epoch)."""
    base = {"bot_id": bot_id, "user_id": user_id, "session_id": session_id, "tz": tz,
            "start": start, "end": end, "limit": MAX_GROUPS + 1}
    owner = owned_message_scope()
    session = " AND m.session_id = :session_id" if session_id else ""
    group_sql = text(f"""
        WITH groups AS (
            SELECT (to_timestamp(m.timestamp) AT TIME ZONE :tz)::date AS day,
                   m.session_id, count(*) AS n,
                   count(*) FILTER (WHERE m.role = 'user') AS prompts,
                   min(m.timestamp) AS first_ts, max(m.timestamp) AS last_ts,
                   (array_agg(m.id ORDER BY m.timestamp, m.id))[1] AS first_id,
                   max(s.status) AS status,
                   max(s.session_metadata->>'title') AS title
            FROM {table} m JOIN sessions s ON s.id = m.session_id
            WHERE {VISIBLE} AND {owner}{session}
              AND m.timestamp >= :start AND m.timestamp < :end
            GROUP BY 1, 2
            ORDER BY 1, 5, 7 LIMIT :limit
        )
        SELECT groups.*, EXISTS (
            SELECT 1 FROM {table} earlier
            WHERE earlier.session_id = groups.session_id
              AND earlier.role NOT IN ('system', 'summary')
              AND (earlier.timestamp, earlier.id) < (groups.first_ts, groups.first_id)
        ) AS continued
        FROM groups ORDER BY day, first_ts, first_id
    """)
    with engine.connect() as conn:
        groups = [] if overview_only else [dict(row) for row in conn.execute(group_sql, base).mappings()]
        if recent and not overview_only and len({str(g['day']) for g in groups}) < 7:
            previous = conn.execute(text(f"""
                SELECT DISTINCT (to_timestamp(m.timestamp) AT TIME ZONE :tz)::date AS day
                FROM {table} m WHERE {VISIBLE} AND {owner}{session}
                  AND m.timestamp < :start
                ORDER BY day DESC LIMIT 7
            """), base).mappings().all()
            if previous:
                oldest = previous[min(6 - len({str(g['day']) for g in groups}), len(previous) - 1)]['day']
                from .history_scope import local_day_bounds
                start, _ = local_day_bounds(oldest, oldest, tz)
                base['start'] = start
                groups = [dict(row) for row in conn.execute(group_sql, base).mappings()]
        if len(groups) > MAX_GROUPS:
            raise HTTPException(status_code=413, detail="Too many timeline segments; narrow the range")
        ids = sorted({str(g['session_id']) for g in groups})
        missing = [str(g['session_id']) for g in groups if not (g.get('title') or '').strip()]
        prompts = {}
        if missing:
            rows = conn.execute(text(f"""
                SELECT DISTINCT ON (m.session_id) m.session_id, m.content
                FROM {table} m WHERE m.session_id = ANY(:ids)
                  AND m.role = 'user' AND ltrim(m.content) NOT LIKE '/%'
                  AND {owner}
                ORDER BY m.session_id, m.timestamp, m.id
            """), {**base, 'ids': sorted(set(missing))}).mappings()
            prompts = {str(row['session_id']): row['content'] for row in rows}
        months = [dict(row) for row in conn.execute(text(f"""
            SELECT date_trunc('month', to_timestamp(m.timestamp) AT TIME ZONE :tz)::date AS month,
                   count(*) AS messages
            FROM {table} m WHERE {VISIBLE} AND {owner}{session}
              AND (:overview_before IS NULL OR m.timestamp < :overview_before)
            GROUP BY 1 ORDER BY 1 DESC LIMIT :month_limit
        """), {**base, 'month_limit': MAX_OVERVIEW_MONTHS + 1,
                 'overview_before': overview_before}).mappings()]
    meta = {sid: {'title': next((g['title'] for g in groups if str(g['session_id']) == sid), None),
                  'status': next((g['status'] for g in groups if str(g['session_id']) == sid), None)}
            for sid in ids}
    return groups, meta, prompts, months, start


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
