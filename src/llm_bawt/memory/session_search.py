"""Read-only, user-scoped session discovery (TASK-1119).

Only human-facing metadata is searchable/returned. Never search the raw JSON:
SDK keys, source message IDs and continuity bodies are not a search index.
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import text
from sqlalchemy.engine import Engine


class SessionSearch:
    def __init__(self, engine: Engine):
        self.engine = engine

    def search(
        self, *, user_id: str, query: str = "", bot_id: str | None = None,
        status: str = "all", since: float | None = None, sort_by: str = "relevance",
        limit: int = 20, offset: int = 0,
    ) -> dict:
        if not user_id or not user_id.strip():
            raise ValueError("A user scope is required")
        if status not in {"all", "active", "archived"}:
            raise ValueError("Invalid session status")
        orders = {
            "relevance": "rank DESC, started_at DESC, id",
            "recent": "started_at DESC, id",
            "oldest": "started_at ASC, id",
            "title": "LOWER(title), started_at DESC, id",
        }
        if sort_by not in orders:
            raise ValueError("Invalid session ordering")
        if not 1 <= limit <= 200 or offset < 0:
            raise ValueError("Invalid search page")
        clauses = ["s.user_id = :user_id", "s.status IN ('active', 'archived', 'completed')"]
        params = {"user_id": user_id, "query": query.strip().lower(), "limit": limit, "offset": offset}
        if bot_id:
            clauses.append("s.bot_id = :bot_id")
            params["bot_id"] = bot_id
        if status != "all":
            clauses.append("s.status = 'active'" if status == "active" else "s.status IN ('archived', 'completed')")
        if since is not None:
            clauses.append("s.started_at >= :since")
            # Registry timestamps are UTC stored WITHOUT time zone. Bind the
            # same representation, independent of the DB connection timezone.
            params["since"] = datetime.fromtimestamp(since, tz=timezone.utc).replace(tzinfo=None)
        # Filtering/counting happen before LIMIT. No per-bot fanout and no scan of
        # message bodies. A lateral model projection accepts only JSON strings.
        sql = f"""
            WITH owned AS (
                SELECT s.id, s.bot_id, s.user_id, s.started_at, s.ended_at,
                    CASE WHEN s.status = 'completed' THEN 'archived' ELSE s.status END AS status,
                    CASE WHEN jsonb_typeof(s.session_metadata->'title') = 'string'
                        THEN LEFT(BTRIM(s.session_metadata->>'title'), 300) ELSE '' END AS title,
                    CASE WHEN jsonb_typeof(s.session_metadata->'provider') = 'string'
                        THEN LEFT(s.session_metadata->>'provider', 100) ELSE '' END AS provider,
                    COALESCE(models.names, '') AS models,
                    CASE WHEN jsonb_typeof(s.session_metadata->'reset_predecessor_ids') = 'array'
                        THEN jsonb_array_length(s.session_metadata->'reset_predecessor_ids') > 0
                        ELSE FALSE END AS has_predecessor
                FROM sessions s
                LEFT JOIN LATERAL (
                    SELECT LEFT(string_agg(DISTINCT value #>> '{{}}', ', ' ORDER BY value #>> '{{}}'), 300) AS names
                    FROM jsonb_each(CASE
                        WHEN jsonb_typeof(s.session_metadata->'agent_session_key_models') = 'object'
                        THEN s.session_metadata->'agent_session_key_models' ELSE '{{}}'::jsonb END)
                    WHERE jsonb_typeof(value) = 'string'
                ) models ON TRUE
                WHERE {' AND '.join(clauses)}
            ), matches AS (
                SELECT *, CASE
                    WHEN :query = '' THEN 0
                    WHEN LOWER(title) = :query THEN 4
                    WHEN strpos(LOWER(title), :query) = 1 THEN 3
                    WHEN strpos(LOWER(title), :query) > 0 THEN 2 ELSE 1 END AS rank
                FROM owned
                WHERE :query = '' OR strpos(LOWER(title), :query) > 0
                    OR strpos(LOWER(bot_id), :query) > 0
                    OR strpos(LOWER(models), :query) > 0
                    OR strpos(LOWER(provider), :query) > 0
            ), page AS (
                SELECT * FROM matches ORDER BY {orders[sort_by]} LIMIT :limit OFFSET :offset
            )
            SELECT page.*, totals.total_count
            FROM (SELECT COUNT(*) AS total_count FROM matches) totals
            LEFT JOIN page ON TRUE
            ORDER BY {orders[sort_by]}
        """
        with self.engine.connect() as conn:
            rows = conn.execute(text(sql), params).mappings().all()
        return {
            "sessions": [dict(row) for row in rows if row["id"] is not None],
            "total_count": int(rows[0]["total_count"]) if rows else 0,
        }
