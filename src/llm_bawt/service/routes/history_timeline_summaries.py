"""Read-only stored summaries; provenance is checked before prose leaves SQL.

Never import/instantiate the summarizer here. A row's session stamp alone is
not provenance (historical backfills stamped mixed-source summaries). Both
availability and excerpts use this same fail-closed source validation.
"""

from __future__ import annotations

import json

from sqlalchemy import text

from .history_scope import owned_message_scope

MAX_SUMMARY_SOURCES = 2000
MAX_SUMMARY_EXCERPTS = 8
SUMMARY_EXCERPT_CHARS = 1200


def _safe_summaries_sql(table: str, selection: str) -> str:
    """Shared CTE. ``selection`` is fixed SQL, never request text.

    Guard array functions with CASE, not WHERE evaluation order. The source
    join stays inside the bot partition and explicitly matches its bot key.
    Equal DISTINCT count rejects duplicates, deleted IDs and partial resolution.
    Missing/invalid dates do not invalidate owned prose, only day coverage.
    """
    return f"""
        selected AS ({selection}),
        candidates AS MATERIALIZED (
            SELECT s.*, CASE
                WHEN jsonb_typeof(s.summary_metadata->'message_ids') = 'array'
                THEN CASE WHEN jsonb_array_length(s.summary_metadata->'message_ids')
                                  BETWEEN 1 AND {MAX_SUMMARY_SOURCES}
                          THEN s.summary_metadata->'message_ids' ELSE '[]'::jsonb END
                ELSE '[]'::jsonb END AS source_ids
            FROM {table} s
            WHERE s.role = 'summary'
              AND EXISTS (SELECT 1 FROM selected k WHERE k.session_id = s.session_id)
              AND EXISTS (SELECT 1 FROM sessions summary_owner
                          WHERE summary_owner.id = s.session_id
                            AND summary_owner.bot_id = :bot_id
                            AND summary_owner.user_id = :user_id)
        ),
        safe AS (
            SELECT s.id, s.session_id, s.content, s.summary_metadata,
                   jsonb_array_length(s.source_ids) AS source_message_count,
                   CASE WHEN jsonb_typeof(s.summary_metadata->'session_start') = 'number'
                        THEN CASE WHEN abs((s.summary_metadata->>'session_start')::numeric) <= 1e12
                                  THEN (s.summary_metadata->>'session_start')::float8 END
                   END AS source_start_utc,
                   CASE WHEN jsonb_typeof(s.summary_metadata->'session_end') = 'number'
                        THEN CASE WHEN abs((s.summary_metadata->>'session_end')::numeric) <= 1e12
                                  THEN (s.summary_metadata->>'session_end')::float8 END
                   END AS source_end_utc
            FROM candidates s
            WHERE s.summary_metadata->'source_session_ids' = jsonb_build_array(s.session_id)
              AND jsonb_array_length(s.source_ids) > 0
              AND s.summary_metadata->'message_count' = to_jsonb(jsonb_array_length(s.source_ids))
              AND NOT EXISTS (
                  SELECT 1 FROM jsonb_array_elements(s.source_ids) item
                  WHERE jsonb_typeof(item) <> 'string'
                     OR length(item #>> '{{}}') NOT BETWEEN 1 AND 128
              )
              AND (SELECT count(DISTINCT src.id)
                   FROM jsonb_array_elements_text(s.source_ids) ids(id)
                   JOIN {table} src ON src.id = ids.id AND src.bot_id = s.bot_id
                   JOIN sessions source_owner ON source_owner.id = src.session_id
                   WHERE src.session_id = s.session_id
                     AND src.role NOT IN ('system', 'summary')
                     AND source_owner.bot_id = :bot_id AND source_owner.user_id = :user_id
                  ) = jsonb_array_length(s.source_ids)
              AND length(btrim(s.content)) > 0
        )
    """


def summary_availability(conn, table: str, bot_id: str, user_id: str,
                         groups: list[dict]) -> dict:
    """One bounded batch for <=2000 visible day/session keys; no prose returned.

    Counts are exact for safe stored excerpts applicable to active sessions,
    not claims of complete daily coverage. Fixed-state hash aggregation avoids
    accumulating summary text/IDs in the overview. Empty/all-month views skip SQL.
    """
    if not groups:
        return {}
    keys = [{"day": str(g["day"]), "session_id": g["session_id"]} for g in groups]
    selection = """
        SELECT * FROM jsonb_to_recordset(CAST(:summary_keys AS jsonb))
        AS k(day text, session_id text)
    """
    query = text(f"""
        WITH {_safe_summaries_sql(table, selection)},
        per_session AS (
            SELECT s.session_id, count(s.id) AS summary_count,
               COALESCE(bit_xor(hashtextextended(
                   concat_ws('|', s.id, s.content, s.summary_metadata::text), 0))
                   FILTER (WHERE s.id IS NOT NULL), 0)::text AS summary_revision
            FROM safe s GROUP BY s.session_id
        )
        SELECT k.day, k.session_id, COALESCE(p.summary_count, 0) AS summary_count,
               COALESCE(p.summary_revision, '0') AS summary_revision
        FROM selected k LEFT JOIN per_session p ON p.session_id = k.session_id
    """)
    return {(str(row["day"]), str(row["session_id"])): dict(row)
            for row in conn.execute(query, {"bot_id": bot_id, "user_id": user_id,
                                           "summary_keys": json.dumps(keys)}).mappings()}


def stored_summary_rows(conn, table: str, bot_id: str, user_id: str,
                        session_id: str | None, start: float | None,
                        end: float | None, *, has_day: bool) -> list[dict]:
    """One extra read, fetch nine to return eight. Independent of prompt paging.

    Select active sessions from *visible* messages, including assistant-only
    days. Session-specific requests may also show unknown/wider stored context;
    the owning session must still be active inside any supplied bounds.
    """
    clauses = [owned_message_scope(), "m.role NOT IN ('system', 'summary')"]
    if session_id:
        clauses.append("m.session_id = :session_id")
    if start is not None:
        clauses.append("m.timestamp >= :summary_start")
    if end is not None:
        clauses.append("m.timestamp < :summary_end")
    selection = f"SELECT DISTINCT m.session_id FROM {table} m WHERE {' AND '.join(clauses)}"
    anchor_bounds = ""
    if start is not None:
        anchor_bounds += " AND m.timestamp >= :summary_start"
    if end is not None:
        anchor_bounds += " AND m.timestamp < :summary_end"
    query = text(f"""
        WITH {_safe_summaries_sql(table, selection)},
        excerpts AS (
        SELECT id, session_id, left(content, :excerpt_chars) AS excerpt,
               length(content) > :excerpt_chars AS truncated,
               source_start_utc, source_end_utc, source_message_count,
               CASE WHEN :has_day AND source_start_utc >= :summary_start
                          AND source_end_utc >= source_start_utc
                          AND source_end_utc < :summary_end
                    THEN 'day' ELSE 'conversation' END AS coverage
        FROM safe
        ORDER BY source_end_utc DESC NULLS LAST, source_start_utc DESC NULLS LAST, id
        LIMIT :summary_limit
        )
        SELECT excerpts.*, anchor.source_first_message_id
        FROM excerpts
        JOIN LATERAL (
            SELECT m.id AS source_first_message_id
            FROM {table} m
            WHERE m.session_id = excerpts.session_id
              AND m.role NOT IN ('system', 'summary') AND {owned_message_scope()}
              {anchor_bounds}
            ORDER BY m.timestamp, m.id LIMIT 1
        ) anchor ON TRUE
        ORDER BY source_end_utc DESC NULLS LAST, source_start_utc DESC NULLS LAST, excerpts.id
    """)
    return [dict(row) for row in conn.execute(query, {
        "bot_id": bot_id, "user_id": user_id, "session_id": session_id,
        "summary_start": start, "summary_end": end, "has_day": has_day,
        "excerpt_chars": SUMMARY_EXCERPT_CHARS, "summary_limit": MAX_SUMMARY_EXCERPTS + 1,
    }).mappings()]
