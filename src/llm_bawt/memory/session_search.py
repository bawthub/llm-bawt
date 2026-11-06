"""Read-only, user-scoped session discovery (TASK-1119).

Only human-facing metadata is searchable/returned. Never search the raw JSON:
SDK keys, source message IDs and continuity bodies are not a search index.

TASK-2061: a session matches on its given name (rename or schedule name) or
its full first eligible message, never on the stored 60-character auto copy.
Each owned session's first eligible message is read once through
(session_id, timestamp) and matched; cost does not grow with how common the
query word is.
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import text
from sqlalchemy.engine import Engine

from ..sessions.first_message import (
    eligible_sql, first_message_fields, first_message_lateral, first_message_params,
    given_title_sql, INITIATOR_KINDS, initiator_kind_sql, storage_bot_sql,
)


def _like_literal(query: str) -> str:
    """``%query%`` for ILIKE with ``\\`` as the escape character."""
    escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


class SessionSearch:
    def __init__(self, engine: Engine):
        self.engine = engine

    def search(
        self, *, user_id: str, query: str = "", bot_id: str | None = None,
        status: str = "all", since: float | None = None, sort_by: str = "relevance",
        limit: int = 20, offset: int = 0, initiated_by: str | None = None,
    ) -> dict:
        if not user_id or not user_id.strip():
            raise ValueError("A user scope is required")
        if status not in {"all", "active", "archived"}:
            raise ValueError("Invalid session status")
        orders = {
            "relevance": "rank DESC, started_at DESC, id",
            "recent": "started_at DESC, id",
            "oldest": "started_at ASC, id",
            "title": "LOWER(label), started_at DESC, id",
        }
        if sort_by not in orders:
            raise ValueError("Invalid session ordering")
        if not 1 <= limit <= 200 or offset < 0:
            raise ValueError("Invalid search page")
        if initiated_by is not None and initiated_by not in INITIATOR_KINDS:
            raise ValueError("Invalid session initiator")
        needle = query.strip().lower()
        clauses = ["s.user_id = :user_id", "s.status IN ('active', 'archived', 'completed')"]
        params = {
            "user_id": user_id, "query": needle, "like": _like_literal(needle),
            "limit": limit, "offset": offset, "initiator": initiated_by,
            **first_message_params(),
        }
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
        # Filtering/counting happen before LIMIT. No per-bot fanout. A lateral
        # model projection accepts only JSON strings. ``title`` is the stored
        # legacy value for older clients; matching uses ``given_title`` and the
        # full first message instead.
        owned = f"""
            owned AS (
                SELECT s.id, s.bot_id, s.user_id, s.started_at, s.ended_at,
                    CASE WHEN s.status = 'completed' THEN 'archived' ELSE s.status END AS status,
                    CASE WHEN jsonb_typeof(s.session_metadata->'title') = 'string'
                        THEN BTRIM(s.session_metadata->>'title') ELSE '' END AS title,
                    CASE WHEN jsonb_typeof(s.session_metadata->'title_source') = 'string'
                        THEN s.session_metadata->>'title_source' ELSE '' END AS title_source,
                    {given_title_sql('s')} AS given_title,
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
            )"""
        # First-message hits: read each owned session's first eligible message
        # (index-ordered, LIMIT 1) and match it. Scanning every matching row
        # instead costs ~1.3s on common words over full history.
        hits = f"""
            fm_hits AS (
                SELECT o.id AS session_id, f.content AS hit_text
                FROM owned o
                CROSS JOIN LATERAL (
                    SELECT m.content FROM messages m
                    WHERE m.session_id = o.id AND m.bot_id = {storage_bot_sql('o.bot_id')}
                      AND {eligible_sql('m')}
                    ORDER BY m.timestamp, m.id LIMIT 1
                ) f
                WHERE :query <> '' AND f.content ILIKE :like ESCAPE '\\'
            )"""
        metadata_match = " OR ".join(
            f"strpos(LOWER(o.{column}), :query) > 0"
            for column in ("given_title", "bot_id", "models", "provider"))
        # A session without a given name is labelled by its first message; when
        # that message is not a hit it cannot contain the query, so '' ranks the
        # same as the full text would.
        matches = f"""
            matches AS (
                SELECT o.*, CASE
                    WHEN :query = '' THEN 0
                    WHEN LOWER(x.label) = :query THEN 4
                    WHEN strpos(LOWER(x.label), :query) = 1 THEN 3
                    WHEN strpos(LOWER(x.label), :query) > 0 THEN 2 ELSE 1 END AS rank
                FROM owned o
                LEFT JOIN fm_hits h ON h.session_id = o.id
                CROSS JOIN LATERAL (SELECT CASE WHEN o.given_title <> '' THEN o.given_title
                                                ELSE COALESCE(h.hit_text, '') END AS label) x
                WHERE :query = '' OR h.session_id IS NOT NULL OR {metadata_match}
            )"""

        def labelled(source: str) -> str:
            return f"""
                SELECT c.*, fm.*,
                    CASE WHEN c.given_title <> '' THEN c.given_title
                         ELSE COALESCE(fm.fm_text, '') END AS label,
                    {initiator_kind_sql('fm')} AS initiator_kind
                FROM {source} c
                {first_message_lateral('messages', 'c')}"""

        if sort_by != "title" and initiated_by is None:
            # Ordering and filtering need no first-message text: look it up
            # for the returned page only.
            sql = f"""
                WITH {owned}, {hits}, {matches}, page AS (
                    SELECT * FROM matches ORDER BY {orders[sort_by]} LIMIT :limit OFFSET :offset
                ), labelled AS ({labelled('page')})
                SELECT labelled.*, totals.total_count
                FROM (SELECT COUNT(*) AS total_count FROM matches) totals
                LEFT JOIN labelled ON TRUE
                ORDER BY {orders[sort_by]}
            """
        else:
            # Label ordering or the initiator filter read every match's first
            # message before paging.
            sql = f"""
                WITH {owned}, {hits}, {matches}, labelled AS ({labelled('matches')}),
                filtered AS (
                    SELECT * FROM labelled
                    WHERE CAST(:initiator AS text) IS NULL OR initiator_kind = :initiator
                ), page AS (
                    SELECT * FROM filtered ORDER BY {orders[sort_by]} LIMIT :limit OFFSET :offset
                )
                SELECT page.*, totals.total_count
                FROM (SELECT COUNT(*) AS total_count FROM filtered) totals
                LEFT JOIN page ON TRUE
                ORDER BY {orders[sort_by]}
            """
        with self.engine.begin() as conn:
            # Interactive and sub-second; the per-session lateral's row estimate
            # is far too high, and JIT compile alone measured ~2.5s of a 3s run.
            if conn.dialect.name == "postgresql":
                conn.execute(text("SET LOCAL jit = off"))
            rows = conn.execute(text(sql), params).mappings().all()
        return {
            "sessions": [_public(row) for row in rows if row["id"] is not None],
            "total_count": int(rows[0]["total_count"]) if rows else 0,
        }


_PUBLIC = ("id", "bot_id", "user_id", "started_at", "ended_at", "status", "title",
           "title_source", "provider", "models", "has_predecessor", "rank")


def _public(row) -> dict:
    return {**{key: row[key] for key in _PUBLIC}, **first_message_fields(row)}
