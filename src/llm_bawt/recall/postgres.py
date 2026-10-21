"""Read-only bounded SQL sources, independent of mutating memory search APIs."""
from __future__ import annotations

from abc import abstractmethod

from sqlalchemy import text

from .budgets import DATABASE_WORKER, compact_json
from .contracts import (
    MAX_RECORD_CHARS, MAX_SOURCE_RECORDS, Evidence, RecallError, RecallScope, SourceProvider,
)

from .excerpts import sql_window, terms_for

# Unknown authors, legacy NULL user sessions, summaries and deleted sessions do
# not establish provenance. An imported bot message is not caller-owned evidence.
MESSAGE_AUTHORITY = """
    m.bot_id = :partition AND s.bot_id = :bot AND s.user_id = :user
    AND s.status IN ('active', 'archived')
    AND m.role IN ('user', 'assistant')
    AND ((m.author_entity_type = 'user' AND m.author_entity_id = :user)
      OR (m.author_entity_type = 'bot' AND m.author_entity_id = :bot))
"""


def partition_id(bot_id: str) -> str:
    from llm_bawt.memory.postgresql import _sanitize_table_name
    return _sanitize_table_name(bot_id)


class RecallDatabase:
    def __init__(self, engine):
        self.engine = engine

    def read(self, statement: str, params: dict) -> list[dict]:
        if self.engine is None:
            raise RecallError("source_unavailable")
        # PostgreSQL enforces read-only mode; no provisioning, access counters,
        # recall marking, embeddings, or session creation.
        with self.engine.connect() as connection:
            with connection.begin():
                connection.execute(text("SET TRANSACTION READ ONLY"))
                connection.execute(text("SET LOCAL statement_timeout = '2000ms'"))
                connection.execute(text("SET LOCAL lock_timeout = '500ms'"))
                return [dict(row) for row in connection.execute(text(statement), params).mappings()]

    async def scope(self, context) -> RecallScope:
        try:
            rows = await DATABASE_WORKER.run(self.read, """
                SELECT slug, include_in_global_search FROM bot_profiles
                ORDER BY slug LIMIT 257
            """, {})
            if not rows or len(rows) > 256:
                raise RecallError("policy_unavailable")
            if any(not isinstance(r["include_in_global_search"], bool) for r in rows):
                raise RecallError("policy_unavailable")
            allowed = frozenset(r["slug"] for r in rows if r["include_in_global_search"])
            excluded = frozenset(r["slug"] for r in rows if not r["include_in_global_search"])
            if context.bot_id not in allowed:
                raise RecallError("unauthorized")
            # Sanitation collisions must never broaden partition scope.
            identities = [partition_id(r["slug"]) for r in rows]
            if len(set(identities)) != len(identities):
                raise RecallError("policy_unavailable")
            from llm_bawt.task_turn_context import TASK_TURN_CONTEXT_TTL_SECONDS
            return RecallScope(context.bot_id, context.user_id, context.session_id, excluded, allowed,
                               context.issued_at + TASK_TURN_CONTEXT_TTL_SECONDS)
        except RecallError:
            raise
        except Exception as error:
            raise RecallError("policy_unavailable") from error


class PostgresSource(SourceProvider):
    def __init__(self, database: RecallDatabase):
        self.database = database

    @property
    @abstractmethod
    def statement(self) -> str:
        pass

    def evidence_text(self, row: dict) -> str:
        return row["excerpt"]

    async def search(self, query: str, scope: RecallScope, task_ids: tuple[str, ...]) -> list[Evidence]:
        from llm_bawt.memory.postgresql import build_fts_query
        tsquery = build_fts_query(query)
        if not tsquery:
            return []
        params = {"query": tsquery, "bot": scope.bot_id, "partition": partition_id(scope.bot_id),
                  "user": scope.user_id, "limit": MAX_SOURCE_RECORDS, "chars": MAX_RECORD_CHARS,
                  "phrase": query.lower(), "terms": terms_for(query)}
        try:
            rows = await DATABASE_WORKER.run(self.database.read, self.statement, params)
        except RecallError:
            raise
        except Exception as error:
            # Do not return [] (a false no-match) or forward a SQL error/payload.
            raise RecallError("source_unavailable") from error
        return [Evidence(self.source, str(row["id"]), scope.bot_id, scope.user_id,
                         self.evidence_text(row), row.get("start_char", 0),
                         row.get("truncated", False), row.get("window_kind", "matched")) for row in rows]


class MessageSource(PostgresSource):
    source = "messages"
    statement = f"""
        SELECT m.id, {sql_window('m.content')}
        FROM messages m JOIN sessions s ON s.id = m.session_id
        WHERE {MESSAGE_AUTHORITY}
          AND to_tsvector('english', m.content) @@ to_tsquery('english', :query)
        ORDER BY ts_rank(to_tsvector('english', m.content), to_tsquery('english', :query)) DESC,
                 m.timestamp DESC, m.id
        LIMIT :limit
    """


class MemorySource(PostgresSource):
    source = "memories"
    # Memories have NO user_id column. Every source must resolve to an allowed
    # message, not just one matching source. Unattributed/manual facts are omitted.
    # Current schema physically deletes consolidated originals; no imaginary
    # superseded column exists to filter. Read only the live memories parent.
    statement = f"""
        SELECT mem.id, {sql_window('mem.content')}
        FROM memories mem
        WHERE mem.bot_id = :partition
          AND jsonb_typeof(mem.source_message_ids) = 'array'
          AND jsonb_array_length(mem.source_message_ids) BETWEEN 1 AND 64
          AND NOT EXISTS (
            SELECT 1 FROM jsonb_array_elements_text(mem.source_message_ids) src(id)
            WHERE NOT EXISTS (
                SELECT 1 FROM messages m JOIN sessions s ON s.id = m.session_id
                WHERE m.id = src.id AND {MESSAGE_AUTHORITY}
            )
          )
          AND to_tsvector('english', mem.content) @@ to_tsquery('english', :query)
        ORDER BY ts_rank(to_tsvector('english', mem.content), to_tsquery('english', :query)) DESC,
                 mem.created_at DESC, mem.id
        LIMIT :limit
    """


class ToolEvidenceSource(PostgresSource):
    source = "tool_evidence"
    # Search only names, never arbitrary arguments/results (which may contain
    # another bot's data). Return canonical receipt metadata, not execution truth.
    statement = f"""
        SELECT t.id, left(t.tool_name, 128) AS tool_name,
               left(t.approval_status, 32) AS approval_status, t.created_at::text AS created_at,
               left(t.result_sha256, 64) AS result_sha256,
               t.result_total_bytes, t.result_total_chars
        FROM tool_call_records t
        JOIN turn_logs turns ON turns.id = t.turn_id
          AND turns.bot_id = t.bot_id AND turns.user_id = t.user_id
        JOIN messages m ON m.id = turns.trigger_message_id AND m.bot_id = :partition
        JOIN sessions s ON s.id = m.session_id
        WHERE t.bot_id = :bot AND t.user_id = :user AND {MESSAGE_AUTHORITY}
          AND to_tsvector('english', t.tool_name) @@ to_tsquery('english', :query)
        ORDER BY t.created_at DESC, t.id
        LIMIT :limit
    """

    def evidence_text(self, row: dict) -> str:
        return compact_json({key: value for key, value in row.items() if key != "id"})
