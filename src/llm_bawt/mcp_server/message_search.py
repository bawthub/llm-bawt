"""Cross-bot message search with explicit database transaction boundaries.

Search rows are fully materialized before author enrichment starts.  Keeping
those phases separate is an important lock-safety invariant: enrichment may
need additional database connections and must never run while the search
connection still owns an open read transaction.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Engine

from llm_bawt.memory.postgresql import MESSAGES_PARENT, build_fts_query

AuthorHydrator = Callable[[list[dict[str, Any]], str], list[dict[str, Any]]]


class CrossBotMessageSearcher:
    """Run aggregate searches without nesting enrichment inside a DB scope."""

    def __init__(self, engine: Engine, hydrate_authors: AuthorHydrator):
        self._engine = engine
        self._hydrate_authors = hydrate_authors

    def search_fts(
        self,
        query: str,
        *,
        n_results: int,
        role_filter: str | None,
        sort_by: str,
        since: float | None,
        until: float | None,
        bot_id: str | None,
        excluded_bot_ids: set[str],
        allowed_bot_ids: frozenset[str] | None = None,
    ) -> list[dict[str, Any]]:
        or_query = build_fts_query(query)
        if not or_query:
            return []

        params: dict[str, Any] = {"query": or_query, "limit": n_results}
        if bot_id:
            if bot_id in excluded_bot_ids:
                return []
            bot_clause = "AND bot_id = :bot_filter"
            params["bot_filter"] = bot_id
        elif excluded_bot_ids:
            bot_clause = "AND NOT (bot_id = ANY(:excluded))"
            params["excluded"] = sorted(excluded_bot_ids)
        else:
            bot_clause = ""

        if allowed_bot_ids is not None:
            bot_clause += " AND bot_id = ANY(CAST(:allowed AS text[]))"
            params["allowed"] = sorted(allowed_bot_ids)
        time_lower = "AND timestamp >= :since" if since is not None else ""
        time_upper = "AND timestamp <= :until" if until is not None else ""
        role_clause = "AND role = :role_filter" if role_filter else ""
        order_clause = (
            "ORDER BY timestamp DESC"
            if sort_by == "recent"
            else "ORDER BY rank DESC, timestamp DESC"
        )
        statement = text(f"""
            SELECT id, role, content, timestamp, session_id,
                   author_entity_type, author_entity_id,
                   bot_id AS source,
                   ts_rank(to_tsvector('english', content),
                           to_tsquery('english', :query)) AS rank,
                   COUNT(*) OVER () AS total
            FROM {MESSAGES_PARENT}
            WHERE to_tsvector('english', content) @@ to_tsquery('english', :query)
              AND role != 'system'
              {bot_clause}
              {role_clause}
              {time_lower}
              {time_upper}
            {order_clause}
            LIMIT :limit
        """)
        if role_filter:
            params["role_filter"] = role_filter
        if since is not None:
            params["since"] = since
        if until is not None:
            params["until"] = until
        return self._execute_then_hydrate(statement, params)

    def search_trgm(
        self,
        query: str,
        *,
        n_results: int,
        role_filter: str | None,
        sort_by: str,
        since: float | None,
        until: float | None,
        excluded_bot_ids: set[str],
        allowed_bot_ids: frozenset[str] | None = None,
    ) -> list[dict[str, Any]]:
        query = (query or "").strip()
        if not query:
            return []

        params: dict[str, Any] = {
            "query": query,
            "ilike": f"%{query}%",
            "limit": n_results,
        }
        if excluded_bot_ids:
            bot_clause = "AND NOT (bot_id = ANY(:excluded))"
            params["excluded"] = sorted(excluded_bot_ids)
        else:
            bot_clause = ""
        if allowed_bot_ids is not None:
            bot_clause += " AND bot_id = ANY(CAST(:allowed AS text[]))"
            params["allowed"] = sorted(allowed_bot_ids)
        time_lower = "AND timestamp >= :since" if since is not None else ""
        time_upper = "AND timestamp <= :until" if until is not None else ""
        role_clause = "AND role = :role_filter" if role_filter else ""
        order_clause = (
            "ORDER BY timestamp DESC"
            if sort_by == "recent"
            else "ORDER BY rank DESC, timestamp DESC"
        )
        statement = text(f"""
            SELECT id, role, content, timestamp, session_id,
                   author_entity_type, author_entity_id,
                   bot_id AS source,
                   similarity(content, :query) AS rank,
                   COUNT(*) OVER () AS total
            FROM {MESSAGES_PARENT}
            WHERE content ILIKE :ilike
              AND role != 'system'
              {bot_clause}
              {role_clause}
              {time_lower}
              {time_upper}
            {order_clause}
            LIMIT :limit
        """)
        if role_filter:
            params["role_filter"] = role_filter
        if since is not None:
            params["since"] = since
        if until is not None:
            params["until"] = until
        return self._execute_then_hydrate(statement, params)

    def _execute_then_hydrate(self, statement, params: dict[str, Any]) -> list[dict[str, Any]]:
        """Materialize rows, close the read transaction, then enrich them."""
        with self._engine.connect() as conn:
            rows = [
                {
                    "id": row.id,
                    "role": row.role,
                    "content": row.content,
                    "timestamp": row.timestamp,
                    "session_id": row.session_id,
                    "author_entity_type": row.author_entity_type,
                    "author_entity_id": row.author_entity_id,
                    "source": row.source,
                    "rank": row.rank,
                    "total": int(row.total),
                }
                for row in conn.execute(statement, params).fetchall()
            ]

        grouped: dict[str, list[dict[str, Any]]] = {}
        order: dict[tuple[str, str], int] = {}
        for index, row in enumerate(rows):
            source = str(row["source"])
            message_id = str(row["id"])
            order[(source, message_id)] = index
            grouped.setdefault(source, []).append(row)

        hydrated: list[dict[str, Any]] = []
        for source, source_rows in grouped.items():
            hydrated.extend(self._hydrate_authors(source_rows, source))
        hydrated.sort(key=lambda row: order[(str(row["source"]), str(row["id"]))])
        return hydrated
