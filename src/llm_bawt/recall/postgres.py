"""Read-only bounded SQL sources, independent of mutating memory search APIs."""
from __future__ import annotations

from abc import abstractmethod
from datetime import datetime, timezone

from sqlalchemy import text

from llm_bawt.bot_tags import SNAPSHOT_SQL, TagPolicyUnavailable, TagSnapshot

from .budgets import DATABASE_WORKER, compact_json
from .contracts import (
    COVERAGE_BUCKETS, FOLLOW_ON_WINDOW_SECONDS, MAX_FOLLOW_ONS_PER_HIT, MAX_RECORD_CHARS, MAX_SOURCE_RECORDS,
    RECENCY_HALF_WEIGHT_DAYS, Evidence, RecallError, RecallScope, SourceProvider,
)

from .excerpts import fts_query, sql_window, terms_for

# Unknown authors, legacy NULL user sessions, summaries and deleted sessions do
# not establish provenance. An imported bot message is not caller-owned evidence.
MESSAGE_AUTHORITY = """
    m.bot_id = :partition AND s.bot_id = :bot AND s.user_id = :user
    AND s.status IN ('active', 'archived')
    AND m.role IN ('user', 'assistant')
    AND ((m.author_entity_type = 'user' AND m.author_entity_id = :user)
      OR (m.author_entity_type = 'bot' AND m.author_entity_id = :bot))
"""


# History rows additionally admit deliveries the caller actually received: a
# role=user row authored by another bot, inside the caller's own partition and
# session (both already pinned above), whose author is in the CURRENT allowed
# search policy. Excluded/protected bots are never in :delivery_authors, so
# their deliveries stay unreachable by construction. Memories and tool evidence
# keep the strict MESSAGE_AUTHORITY.
HISTORY_AUTHORITY = MESSAGE_AUTHORITY.rstrip().removesuffix(")") + """
      OR (m.role = 'user' AND m.author_entity_type = 'bot'
          AND m.author_entity_id = ANY(CAST(:delivery_authors AS text[]))))
"""



def delivery_authors(scope: RecallScope) -> list[str]:
    return sorted(scope.allowed_bot_ids - scope.excluded_bot_ids - {scope.bot_id})


def utc_iso(value) -> str | None:
    """Server-owned evidence time. Naive datetimes are stored UTC (utcnow)."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        value = datetime.fromtimestamp(value, timezone.utc)
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    # Fixed-width microseconds: the service sorts these strings oldest-first, so
    # same-second turns (a correction 0.8 s after the hit) must still order.
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def partition_id(bot_id: str) -> str:
    from llm_bawt.memory.postgresql import _sanitize_table_name
    return _sanitize_table_name(bot_id)


_MESSAGE_WINDOW = sql_window("m.content")
_MEMORY_WINDOW = sql_window("mem.content")


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
            rows = await DATABASE_WORKER.run(self.read, SNAPSHOT_SQL, {})
            if not rows:
                raise RecallError("policy_unavailable")
            try:
                tags = TagSnapshot.from_rows(rows)
            except TagPolicyUnavailable:
                raise RecallError("policy_unavailable") from None
            # Cross-bot recall spans searchable bots only: private implies
            # no_global_search, and the legacy column false is private.
            allowed = tags.searchable_bots
            excluded = tags.known_bots - allowed
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
        tsquery = fts_query(query)
        if not tsquery:
            return []
        params = {"query": tsquery, "bot": scope.bot_id, "partition": partition_id(scope.bot_id),
                  "user": scope.user_id, "limit": MAX_SOURCE_RECORDS, "chars": MAX_RECORD_CHARS,
                  "phrase": query.lower(), "terms": terms_for(query), **self.extra_params(scope)}
        try:
            rows = await DATABASE_WORKER.run(self.database.read, self.statement, params)
        except RecallError:
            raise
        except Exception as error:
            # Do not return [] (a false no-match) or forward a SQL error/payload.
            raise RecallError("source_unavailable") from error
        return [Evidence(self.source, str(row["id"]), scope.bot_id, scope.user_id,
                         self.evidence_text(row), row.get("start_char", 0),
                         row.get("truncated", False), row.get("window_kind", "matched"),
                         utc_iso(row.get("at")), row.get("speaker"),
                         str(row["follows"]) if row.get("follows") else None) for row in rows]

    def extra_params(self, scope: RecallScope) -> dict:
        return {}


class MessageSource(PostgresSource):
    source = "messages"
    # Hits plus their follow-on turns; the service's global record/token caps
    # still bound what reaches the model (follow-ons sit right after their hit).
    max_records = MAX_SOURCE_RECORDS * (1 + MAX_FOLLOW_ONS_PER_HIT)
    statement = f"""
        WITH query_lexemes AS (
            -- Same parser and config as the tsquery, so coverage counts exactly
            -- the lexemes the match can hit (stop words never count).
            SELECT ARRAY(SELECT DISTINCT unnest(tsvector_to_array(
                       to_tsvector('english', array_to_string(CAST(:terms AS text[]), ' '))))) AS lexemes
        ), hits AS (
            SELECT m.id, m.session_id, m.timestamp,
                   CEIL(cardinality(ARRAY(SELECT unnest(q.lexemes)
                                          INTERSECT SELECT unnest(tsvector_to_array(v.doc))))
                        * {COVERAGE_BUCKETS}::numeric / GREATEST(cardinality(q.lexemes), 1)) AS coverage,
                   ts_rank(v.doc, to_tsquery('english', :query))
                   / (1 + GREATEST(0, extract(epoch FROM now()) - m.timestamp)
                          / 86400.0 / {RECENCY_HALF_WEIGHT_DAYS}) AS score
            FROM messages m JOIN sessions s ON s.id = m.session_id
            CROSS JOIN query_lexemes q
            CROSS JOIN LATERAL (SELECT m.content_tsv AS doc) v
            WHERE {HISTORY_AUTHORITY}
              AND v.doc @@ to_tsquery('english', :query)
            ORDER BY coverage DESC, score DESC, m.timestamp DESC, m.id
            LIMIT :limit
        ), follow_ons AS (
            SELECT f.id, h.coverage, h.score, h.id AS hit_id, f.ord
            FROM hits h CROSS JOIN LATERAL (
                SELECT m.id, row_number() OVER (ORDER BY m.timestamp, m.id) AS ord
                FROM messages m JOIN sessions s ON s.id = m.session_id
                WHERE {HISTORY_AUTHORITY}
                  AND m.session_id = h.session_id
                  AND m.timestamp > h.timestamp
                  AND m.timestamp <= h.timestamp + {FOLLOW_ON_WINDOW_SECONDS}
                ORDER BY m.timestamp, m.id
                LIMIT {MAX_FOLLOW_ONS_PER_HIT}
            ) f
        ), picked AS (
            -- One record per message; a message that is itself a hit stays a hit.
            SELECT DISTINCT ON (id) id, coverage, score, hit_id, ord
            FROM (SELECT id, coverage, score, id AS hit_id, 0 AS ord FROM hits
                  UNION ALL SELECT id, coverage, score, hit_id, ord FROM follow_ons) candidates
            ORDER BY id, ord = 0 DESC, coverage DESC, score DESC, hit_id
        )
        SELECT m.id, {_MESSAGE_WINDOW[0]}, m.timestamp AS at,
               CASE WHEN p.ord > 0 THEN p.hit_id END AS follows,
               CASE WHEN m.author_entity_type = 'bot' AND m.author_entity_id <> :bot
                    THEN 'bot:' || m.author_entity_id ELSE m.role END AS speaker
        FROM picked p JOIN messages m ON m.id = p.id AND m.bot_id = :partition
        {_MESSAGE_WINDOW[1]}
        ORDER BY p.coverage DESC, p.score DESC, p.hit_id, p.ord
        LIMIT {MAX_SOURCE_RECORDS * (1 + MAX_FOLLOW_ONS_PER_HIT)}
    """

    def extra_params(self, scope: RecallScope) -> dict:
        return {"delivery_authors": delivery_authors(scope)}


class MemorySource(PostgresSource):
    source = "memories"
    # Memories have NO user_id column. Every source must resolve to an allowed
    # message, not just one matching source. Unattributed/manual facts are omitted.
    # Current schema physically deletes consolidated originals; no imaginary
    # superseded column exists to filter. Read only the live memories parent.
    statement = f"""
        SELECT mem.id, {_MEMORY_WINDOW[0]}, mem.created_at AS at
        FROM memories mem
        {_MEMORY_WINDOW[1]}
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
               t.created_at AS at,
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
        return compact_json({key: value for key, value in row.items() if key not in {"id", "at"}})
