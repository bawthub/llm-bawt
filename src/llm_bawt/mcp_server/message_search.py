"""Cross-bot message search with explicit database transaction boundaries.

Search rows are fully materialized before author enrichment starts.  Keeping
those phases separate is an important lock-safety invariant: enrichment may
need additional database connections and must never run while the search
connection still owns an open read transaction.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from sqlalchemy import text
from sqlalchemy.engine import Engine

from llm_bawt.bot_access import BotAccess, BotAccessError, PartitionScope
from llm_bawt.memory.postgresql import MESSAGES_PARENT
from llm_bawt.message_authorship import AuthorReference, EffectiveAuthorRule
from llm_bawt.retrieval.fts_query import FtsQuery

AuthorHydrator = Callable[[list[dict[str, Any]], str], list[dict[str, Any]]]


@dataclass(frozen=True)
class OperatorMessageScope:
    """Operator (Spotlight/REST) search over positive public partitions (TASK-1301).

    Built from a fresh, fail-closed ``BotAccess`` with no trusted caller, so
    membership is the live public set, never an exclusion list. Every row is
    classified by ``EffectiveAuthorRule.operator_case()``; an ``author``
    filter narrows on that same computed class in WHERE, before the window
    count and LIMIT. Unfiltered searches add no session, user or status
    eligibility predicates, so their row set is the public partitions' rows.
    """

    partitions: tuple[PartitionScope, ...]
    public_authors: frozenset[str]
    author: AuthorReference | None = None
    in_bot: str | None = None

    @property
    def filtered(self) -> bool:
        """Whether a filter alone may drive a search with no query text."""
        return self.author is not None or self.in_bot is not None

    @classmethod
    def from_access(
        cls,
        access: BotAccess,
        *,
        author: AuthorReference | None = None,
        in_bot: str | None = None,
    ) -> OperatorMessageScope:
        if access.caller is not None:
            # Operator search is not a trusted-caller surface; callers with an
            # identity use the scoped MCP path and its CallerScope semantics.
            raise BotAccessError("caller_unavailable")
        try:
            scope = access.aggregate_scope(in_bot)
        except BotAccessError as exc:
            if exc.code != "access_denied":
                raise
            # Unknown and protected bots read as an ordinary empty match.
            return cls((), access.public_bots, author, in_bot)
        return cls(scope.partitions, scope.public_authors, author, in_bot)

    @property
    def params(self) -> dict[str, Any]:
        params: dict[str, Any] = {
            "op_bots": [p.bot_id for p in self.partitions],
            "op_partitions": [p.partition for p in self.partitions],
            "authority_authors": sorted(self.public_authors),
        }
        if self.author is not None:
            params["author_id"] = self.author.entity_id
        return params

    @property
    def from_sql(self) -> str:
        return f"""
            FROM {MESSAGES_PARENT} m
            JOIN unnest(CAST(:op_bots AS text[]), CAST(:op_partitions AS text[]))
                 AS p(bot_id, partition) ON m.bot_id = p.partition
            LEFT JOIN sessions s ON s.id = m.session_id AND s.bot_id = p.bot_id
            CROSS JOIN LATERAL (
                SELECT {EffectiveAuthorRule.operator_case()} AS author_class
            ) c
        """

    @property
    def author_clause(self) -> str:
        if self.author is None:
            return ""
        if self.author.entity_type == "user":
            return "AND c.author_class = 'user' AND m.author_entity_id = :author_id"
        return """AND ((c.author_class = 'bot' AND p.bot_id = :author_id)
                    OR (c.author_class = 'delivery' AND m.author_entity_id = :author_id))"""


class CrossBotMessageSearcher:
    """Run aggregate searches without nesting enrichment inside a DB scope."""

    def __init__(self, engine: Engine, hydrate_authors: AuthorHydrator):
        self._engine = engine
        self._hydrate_authors = hydrate_authors

    def search_fts(
        self,
        query: str,
        *,
        partitions: frozenset[str],
        n_results: int,
        role_filter: str | None,
        sort_by: str,
        since: float | None,
        until: float | None,
    ) -> list[dict[str, Any]]:
        """Scoped FTS over exactly ``partitions`` (from a ``BotAccess`` scope).

        There is no unscoped form: an empty set searches nothing.
        """
        fts = FtsQuery.parse(query)
        if fts is None or not partitions:
            return []

        params: dict[str, Any] = {**fts.params, "limit": n_results}
        if len(partitions) == 1:
            bot_clause = "AND bot_id = :partition"
            params["partition"] = next(iter(partitions))
        else:
            bot_clause = "AND bot_id = ANY(CAST(:partitions AS text[]))"
            params["partitions"] = sorted(partitions)
        tsquery = fts.tsquery_sql()
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
                   ts_rank(content_tsv, {tsquery}) AS rank,
                   COUNT(*) OVER () AS total
            FROM {MESSAGES_PARENT}
            WHERE content_tsv @@ {tsquery}
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

    def search_operator(
        self,
        mode: Literal["fts", "trgm"],
        query: str,
        scope: OperatorMessageScope,
        *,
        n_results: int,
        role_filter: str | None,
        sort_by: str,
        since: float | None,
        until: float | None,
    ) -> list[dict[str, Any]]:
        """Operator search with per-row ``author_class`` and optional filters.

        Empty query text is valid only with a scope filter; then rows are
        ordered newest first with rank 0 and ``sort_by`` does not apply.
        """
        if not scope.partitions:
            return []
        params: dict[str, Any] = {**scope.params, "limit": n_results}
        stripped = (query or "").strip()
        if not stripped:
            if not scope.filtered:
                return []
            match_clause, rank_sql, order_clause = "", "0.0", "ORDER BY m.timestamp DESC"
        else:
            if mode == "trgm":
                params.update(query=stripped, ilike=f"%{stripped}%")
                match_clause = "AND m.content ILIKE :ilike"
                rank_sql = "similarity(m.content, :query)"
            else:
                fts = FtsQuery.parse(query)
                if fts is None:
                    return []
                params.update(fts.params)
                tsquery = fts.tsquery_sql()
                match_clause = f"AND m.content_tsv @@ {tsquery}"
                rank_sql = f"ts_rank(m.content_tsv, {tsquery})"
            order_clause = ("ORDER BY m.timestamp DESC" if sort_by == "recent"
                            else "ORDER BY rank DESC, m.timestamp DESC")
        role_clause = "AND m.role = :role_filter" if role_filter else ""
        time_lower = "AND m.timestamp >= :since" if since is not None else ""
        time_upper = "AND m.timestamp <= :until" if until is not None else ""
        if role_filter:
            params["role_filter"] = role_filter
        if since is not None:
            params["since"] = since
        if until is not None:
            params["until"] = until
        statement = text(f"""
            SELECT m.id, m.role, m.content, m.timestamp, m.session_id,
                   m.author_entity_type, m.author_entity_id,
                   p.bot_id AS source, c.author_class,
                   {rank_sql} AS rank,
                   COUNT(*) OVER () AS total
            {scope.from_sql}
            WHERE m.role != 'system'
              {match_clause}
              {scope.author_clause}
              {role_clause}
              {time_lower}
              {time_upper}
            {order_clause}
            LIMIT :limit
        """)
        return self._execute_then_hydrate(statement, params, extra=("author_class",))

    def _execute_then_hydrate(
        self, statement, params: dict[str, Any], *, extra: tuple[str, ...] = (),
    ) -> list[dict[str, Any]]:
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
                    **{name: getattr(row, name) for name in extra},
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
