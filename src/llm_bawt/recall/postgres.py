"""Read-only bounded SQL sources, independent of mutating memory search APIs."""
from __future__ import annotations

from abc import abstractmethod
from datetime import datetime, timezone
import re

from llm_bawt.retrieval.find_query import parse as parse_find_query
from llm_bawt.retrieval.find import like_pattern

from sqlalchemy import text

from llm_bawt.bot_tags import SNAPSHOT_SQL, TagPolicyUnavailable, TagSnapshot

from .budgets import DATABASE_WORKER, compact_json
from .contracts import (
    COVERAGE_BUCKETS, FOLLOW_ON_WINDOW_SECONDS, MAX_FOLLOW_ONS_PER_HIT, MAX_RECORD_CHARS, MAX_SOURCE_RECORDS,
    RECENCY_HALF_WEIGHT_DAYS, Evidence, RecallError, RecallScope, SourceProvider,
)

from .excerpts import fts_query, query_terms, sql_window, terms_for

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


# Message recall searches a positive public partition map, not an exclusion
# filter. Each message still needs the trusted user/session and canonical author.
# The caller's own history uses the same rule; memories/tools stay caller-only.
PUBLIC_HISTORY_AUTHORITY = """
    m.bot_id = p.partition AND s.bot_id = p.source_bot AND s.user_id = :user
    AND s.status IN ('active', 'archived')
    AND ((m.role = 'user' AND m.author_entity_type = 'user'
          AND m.author_entity_id = :user)
      OR (m.role = 'assistant' AND m.author_entity_type = 'bot'
          AND m.author_entity_id = p.source_bot)
      OR (m.role = 'user' AND m.author_entity_type = 'bot'
          AND m.author_entity_id = ANY(CAST(:delivery_authors AS text[]))))
"""


def delivery_authors(scope: RecallScope) -> list[str]:
    return sorted(scope.allowed_bot_ids - scope.excluded_bot_ids - {scope.bot_id})


def review_ref(text: str) -> str | None:
    """Only label a review when its id is present in authorized source bytes."""
    # Bare rN may name a storage product, revision, or unrelated artifact.
    # Report a review only when its marker appears with review-specific context.
    pattern = (r"\breview\s+(r[1-9]\d{0,3})\b|\b(r[1-9]\d{0,3})\s+"
               r"(?:B[1-9]\d*|APPROVE(?:_WITH_CHANGES)?|REQUEST_CHANGES)\b")
    references = {next(group for group in match.groups() if group).lower()
                  for match in re.finditer(pattern, text, re.IGNORECASE)}
    return next(iter(references)) if len(references) == 1 else None


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

    def statement_for(self, query: str) -> str:
        return self.statement

    async def search(self, query: str, scope: RecallScope, task_ids: tuple[str, ...]) -> list[Evidence]:
        tsquery = fts_query(query)
        if not tsquery:
            return []
        params = {"query": tsquery, "bot": scope.bot_id, "partition": partition_id(scope.bot_id),
                  "user": scope.user_id, "limit": MAX_SOURCE_RECORDS, "chars": MAX_RECORD_CHARS,
                  "phrase": query.lower(), "terms": terms_for(query), **self.extra_params(scope, query)}
        try:
            rows = await DATABASE_WORKER.run(self.database.read, self.statement_for(query), params)
        except RecallError:
            raise
        except Exception as error:
            # Do not return [] (a false no-match) or forward a SQL error/payload.
            raise RecallError("source_unavailable") from error
        return [Evidence(self.source, str(row["id"]), scope.bot_id, scope.user_id,
                         self.evidence_text(row), row.get("start_char", 0),
                         row.get("truncated", False), row.get("window_kind", "matched"),
                         utc_iso(row.get("at")), row.get("speaker"),
                         str(row["follows"]) if row.get("follows") else None,
                         row.get("source_bot"), row.get("session_id"),
                         review_ref(self.evidence_text(row))) for row in rows]

    def extra_params(self, scope: RecallScope, query: str) -> dict:
        return {}


class MessageSource(PostgresSource):
    source = "messages"
    max_records = MAX_SOURCE_RECORDS * (1 + MAX_FOLLOW_ONS_PER_HIT)
    # Every candidate tier is capped before coverage/rank or full content is
    # projected. Literal exact tiers use the existing trigram-backed semantics of
    # messages_find; lexical tiers use the stored indexed content_tsv. The broad
    # OR is a fallback only when stricter tiers returned nothing. No excluded
    # partition participates even in candidate limiting.
    statement = f"""
        WITH focused AS MATERIALIZED (
            SELECT m.id, m.bot_id AS partition, p.source_bot, m.session_id, m.timestamp,
                   TRUE AS exact_match
            FROM messages m
            JOIN unnest(CAST(:bots AS text[]), CAST(:parts AS text[]))
                 AS p(source_bot, partition) ON m.bot_id = p.partition
            JOIN sessions s ON s.id = m.session_id
            WHERE {PUBLIC_HISTORY_AUTHORITY} AND :focus_enabled
              AND m.content_tsv @@ plainto_tsquery('english', :focus)
              AND __FOCUS_EXACT__
            ORDER BY CASE WHEN m.author_entity_type = 'bot'
                                AND m.author_entity_id = ANY(CAST(:named_bots AS text[]))
                          THEN 1 ELSE 0 END DESC,
                     ts_rank(m.content_tsv, to_tsquery('english', :query)) DESC,
                     m.timestamp DESC, m.id LIMIT :candidate_limit
        ), exact_topic AS MATERIALIZED (
            SELECT m.id, m.bot_id AS partition, p.source_bot, m.session_id, m.timestamp,
                   TRUE AS exact_match
            FROM messages m
            JOIN unnest(CAST(:bots AS text[]), CAST(:parts AS text[]))
                 AS p(source_bot, partition) ON m.bot_id = p.partition
            JOIN sessions s ON s.id = m.session_id
            WHERE {PUBLIC_HISTORY_AUTHORITY} AND :topic_enabled AND __EXACT_ALL__
              AND m.content_tsv @@ plainto_tsquery('english', :topic)
              AND NOT EXISTS (SELECT 1 FROM focused)
            ORDER BY ts_rank(m.content_tsv, to_tsquery('english', :query)) DESC,
                     m.timestamp DESC, m.id LIMIT :candidate_limit
        ), exact_all AS MATERIALIZED (
            SELECT m.id, m.bot_id AS partition, p.source_bot, m.session_id, m.timestamp,
                   TRUE AS exact_match
            FROM messages m
            JOIN unnest(CAST(:bots AS text[]), CAST(:parts AS text[]))
                 AS p(source_bot, partition) ON m.bot_id = p.partition
            JOIN sessions s ON s.id = m.session_id
            WHERE {PUBLIC_HISTORY_AUTHORITY} AND __EXACT_ALL__
              AND NOT EXISTS (SELECT 1 FROM focused)
              AND NOT EXISTS (SELECT 1 FROM exact_topic)
            ORDER BY ts_rank(m.content_tsv, to_tsquery('english', :query)) DESC,
                     m.timestamp DESC, m.id LIMIT :candidate_limit
        ), exact_any AS MATERIALIZED (
            SELECT m.id, m.bot_id AS partition, p.source_bot, m.session_id, m.timestamp,
                   TRUE AS exact_match
            FROM messages m
            JOIN unnest(CAST(:bots AS text[]), CAST(:parts AS text[]))
                 AS p(source_bot, partition) ON m.bot_id = p.partition
            JOIN sessions s ON s.id = m.session_id
            WHERE {PUBLIC_HISTORY_AUTHORITY} AND __EXACT_ANY__
              AND NOT EXISTS (SELECT 1 FROM focused)
              AND NOT EXISTS (SELECT 1 FROM exact_topic)
              AND NOT EXISTS (SELECT 1 FROM exact_all)
            ORDER BY ts_rank(m.content_tsv, to_tsquery('english', :query)) DESC,
                     m.timestamp DESC, m.id LIMIT :candidate_limit
        ), strong AS MATERIALIZED (
            SELECT m.id, m.bot_id AS partition, p.source_bot, m.session_id, m.timestamp,
                   FALSE AS exact_match
            FROM messages m
            JOIN unnest(CAST(:bots AS text[]), CAST(:parts AS text[]))
                 AS p(source_bot, partition) ON m.bot_id = p.partition
            JOIN sessions s ON s.id = m.session_id
            WHERE {PUBLIC_HISTORY_AUTHORITY}
              AND :allow_lexical
              AND m.content_tsv @@ plainto_tsquery('english', :strong)
              AND NOT EXISTS (SELECT 1 FROM focused)
              AND NOT EXISTS (SELECT 1 FROM exact_topic)
              AND NOT EXISTS (SELECT 1 FROM exact_all)
              AND NOT EXISTS (SELECT 1 FROM exact_any)
            ORDER BY m.timestamp DESC, m.id LIMIT :candidate_limit
        ), loose AS MATERIALIZED (
            SELECT m.id, m.bot_id AS partition, p.source_bot, m.session_id, m.timestamp,
                   FALSE AS exact_match
            FROM messages m
            JOIN unnest(CAST(:bots AS text[]), CAST(:parts AS text[]))
                 AS p(source_bot, partition) ON m.bot_id = p.partition
            JOIN sessions s ON s.id = m.session_id
            WHERE {PUBLIC_HISTORY_AUTHORITY}
              AND :allow_lexical
              AND m.content_tsv @@ to_tsquery('english', :query)
              AND NOT EXISTS (SELECT 1 FROM focused)
              AND NOT EXISTS (SELECT 1 FROM exact_topic)
              AND NOT EXISTS (SELECT 1 FROM exact_all)
              AND NOT EXISTS (SELECT 1 FROM exact_any)
              AND NOT EXISTS (SELECT 1 FROM strong)
            ORDER BY m.timestamp DESC, m.id LIMIT :candidate_limit
        ), candidates AS MATERIALIZED (
            SELECT * FROM focused UNION ALL SELECT * FROM exact_topic
            UNION ALL SELECT * FROM exact_all UNION ALL SELECT * FROM exact_any
            UNION ALL SELECT * FROM strong
            UNION ALL SELECT * FROM loose
        ), query_lexemes AS (
            SELECT ARRAY(SELECT DISTINCT unnest(tsvector_to_array(
                       to_tsvector('english', array_to_string(CAST(:terms AS text[]), ' '))))) AS lexemes
        ), scored AS MATERIALIZED (
            SELECT c.id, c.partition, c.source_bot, c.session_id, c.timestamp, c.exact_match,
                   (m.author_entity_type = 'bot'
                    AND m.author_entity_id = ANY(CAST(:named_bots AS text[]))) AS named_author,
                   cardinality(ARRAY(SELECT unnest(q.lexemes)
                                     INTERSECT SELECT unnest(tsvector_to_array(m.content_tsv)))) AS matched,
                   cardinality(q.lexemes) AS possible,
                   ts_rank(m.content_tsv, to_tsquery('english', :query))
                   / (1 + GREATEST(0, extract(epoch FROM now()) - c.timestamp)
                          / 86400.0 / {RECENCY_HALF_WEIGHT_DAYS}) AS score
            FROM candidates c JOIN messages m ON m.id = c.id AND m.bot_id = c.partition
            CROSS JOIN query_lexemes q
        ), hits AS MATERIALIZED (
            SELECT id, partition, source_bot, session_id, timestamp,
                   CEIL(matched * {COVERAGE_BUCKETS}::numeric / GREATEST(possible, 1)) AS coverage,
                   score
            FROM scored
            -- Broad fallback may return one generic word from a long question;
            -- only an actual exact-tier match establishes identity without a
            -- second lexical token. Having an exact token in the query does not.
            WHERE exact_match OR matched >= LEAST(2, possible)
            ORDER BY coverage DESC, named_author DESC, score DESC, timestamp DESC, id
            LIMIT :limit
        ), follow_ons AS (
            SELECT f.id, h.partition, h.source_bot, h.coverage, h.score, h.id AS hit_id, f.ord
            FROM hits h CROSS JOIN LATERAL (
                SELECT m.id, row_number() OVER (ORDER BY m.timestamp, m.id) AS ord
                FROM messages m
                JOIN sessions s ON s.id = m.session_id
                JOIN unnest(CAST(:bots AS text[]), CAST(:parts AS text[]))
                     AS p(source_bot, partition) ON m.bot_id = p.partition
                WHERE {PUBLIC_HISTORY_AUTHORITY}
                  AND m.bot_id = h.partition AND m.session_id = h.session_id
                  AND m.timestamp > h.timestamp
                  AND m.timestamp <= h.timestamp + {FOLLOW_ON_WINDOW_SECONDS}
                ORDER BY m.timestamp, m.id LIMIT {MAX_FOLLOW_ONS_PER_HIT}
            ) f
        ), picked AS (
            SELECT DISTINCT ON (partition, id) id, partition, source_bot, coverage, score, hit_id, ord
            FROM (SELECT id, partition, source_bot, coverage, score, id AS hit_id, 0 AS ord FROM hits
                  UNION ALL SELECT id, partition, source_bot, coverage, score, hit_id, ord FROM follow_ons) choices
            ORDER BY partition, id, ord = 0 DESC, coverage DESC, score DESC, hit_id
        )
        SELECT m.id, p.source_bot, m.session_id, {_MESSAGE_WINDOW[0]}, m.timestamp AS at,
               CASE WHEN p.ord > 0 THEN p.hit_id END AS follows,
               CASE WHEN m.author_entity_type = 'bot' AND m.author_entity_id <> p.source_bot
                    THEN 'bot:' || m.author_entity_id ELSE m.role END AS speaker
        FROM picked p JOIN messages m ON m.id = p.id AND m.bot_id = p.partition
        {_MESSAGE_WINDOW[1]}
        ORDER BY p.coverage DESC, p.score DESC, p.hit_id, p.ord
        LIMIT {MAX_SOURCE_RECORDS * (1 + MAX_FOLLOW_ONS_PER_HIT)}
    """

    @staticmethod
    def exact_literals(query: str) -> tuple[str, ...]:
        # messages_find treats abbreviations and clock times as literals for
        # browsing; recall must not promote them above topical evidence.
        return tuple(token for token in parse_find_query(query[:512]).exact
                     if not re.fullmatch(
                         r"(?:[A-Za-z]\.)+[A-Za-z]?|\d{1,2}:\d{2}(?::\d{2})?|B[1-9]\d*-B[1-9]\d*",
                         token, re.IGNORECASE))

    @staticmethod
    def review_terms(query: str) -> tuple[str, ...]:
        """Index the review number and list endpoints, not the textual range.

        A literal B1-B5 selects unrelated summaries quoting that range, while
        actual reviews often say B1 through B5 individually. Unlabeled reviews
        remain searchable via the ordinary lexical fallback.
        """
        if not re.search(r"\b(?:review|blockers?)\b", query, re.IGNORECASE):
            return ()
        marker = re.search(r"(?<![A-Za-z0-9_])r[1-9]\d{0,3}(?![A-Za-z0-9_])", query, re.IGNORECASE)
        if marker is None:
            return ()
        bounds = re.search(r"\bB([1-9]\d*)\s*[-–]\s*B([1-9]\d*)\b", query, re.IGNORECASE)
        if bounds:
            endpoints = (f"b{bounds[1]}", f"b{bounds[2]}")
        else:
            # "five blockers" asks for the numbered list even when B1-B5 was
            # not spelled out. A miss falls back to ordinary lexical retrieval.
            count = re.search(r"\b(one|two|three|four|five|six|seven|eight|nine|ten)\s+blockers?\b",
                              query, re.IGNORECASE)
            numbers = ("one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten")
            endpoints = ("b1", f"b{numbers.index(count[1].lower()) + 1}") if count else ()
        return (marker.group().lower(), *endpoints)

    def statement_for(self, query: str) -> str:
        exact = self.exact_literals(query)
        clauses = [f"m.content ILIKE :exact_{i}" for i in range(len(exact))]
        # A focused review cannot suppress exact search for a different task.
        # Require every explicit identity in the query before its candidate cap.
        return (self.statement.replace("__FOCUS_EXACT__", "(" + " AND ".join(clauses) + ")" if clauses else "TRUE")
                .replace("__EXACT_ALL__", "(" + " AND ".join(clauses) + ")" if clauses else "FALSE")
                .replace("__EXACT_ANY__", "(" + " OR ".join(clauses) + ")" if clauses else "FALSE"))

    def extra_params(self, scope: RecallScope, query: str) -> dict:
        parsed = parse_find_query(query[:512])
        exact = self.exact_literals(query)
        allowed = sorted(scope.allowed_bot_ids - scope.excluded_bot_ids)
        review = self.review_terms(query)
        # The policy snapshot is the only source of bot identities. Mentioning a
        # searchable author helps distinguish their actual review from newer
        # recaps; it never grants a partition or delivery read.
        named = [bot for bot in allowed if re.search(
            rf"(?<![A-Za-z0-9_]){re.escape(bot)}(?![A-Za-z0-9_])", query, re.IGNORECASE)]
        terms = list(dict.fromkeys([*terms_for(query), *review]))
        search = " | ".join(terms)
        # A numbered review uses an indexed conjunction first; if no labeled
        # answer exists, the existing bounded exact/lexical tiers still run.
        strong = list(dict.fromkeys([*review, *terms_for(query)]))[:5]
        # Recent mentions of a task ID can crowd out its earlier decision.
        # The question's last three substantive words form an indexed topical
        # tier; an absent topic falls back to the ranked exact-ID tier.
        topical = query_terms(query)
        topic = topical[-3:] if len(exact) == 1 and len(topical) >= 5 else []
        return {"bots": allowed, "parts": [partition_id(bot) for bot in allowed],
                "delivery_authors": delivery_authors(scope), "named_bots": named,
                "focus_enabled": len(review) > 1, "focus": " ".join(review),
                "topic_enabled": bool(topic), "topic": " ".join(topic),
                "strong": " ".join(strong), "query": search, "terms": terms,
                "allow_lexical": not exact or bool(parsed.parts),
                "candidate_limit": 96,
                **{f"exact_{i}": like_pattern(token) for i, token in enumerate(exact)}}


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
