"""Bounded, private-safe repair of missing memory embeddings (TASK-1843)."""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass

from sqlalchemy import text

from ..bot_tags import BehaviorTag, BotTags, sql_allowed
from .embeddings import EMBED_DIM, embed_with_status, meaning_text
from .postgresql import PostgreSQLMemoryBackend

log = logging.getLogger(__name__)

# One session lock across app replicas and manual repair; never a transaction
# across /embed. The name is a product identifier, not an operator setting.
_LOCK_SQL = "SELECT pg_try_advisory_lock(hashtext('memory-embedding-repair'))"
_UNLOCK_SQL = "SELECT pg_advisory_unlock(hashtext('memory-embedding-repair'))"
_MAX_ROWS = 25
_MAX_TRANSIENT = 3


def _eligible(bot_id: str, engine, conn=None) -> bool:
    """Never construct a backend (or read its partition) for an excluded bot."""
    snapshot = BotTags(engine).snapshot(conn)
    if bot_id not in snapshot.known_bots or snapshot.excluded_reason(bot_id) is not None:
        return False
    from ..utils.bot_partition import sanitize_table_name
    return sum(sanitize_table_name(slug) == sanitize_table_name(bot_id)
               for slug in snapshot.known_bots) == 1


def _allowed_sql(prefix: str = "er") -> tuple[str, dict]:
    # Partition bot_id is SANITIZED; profiles/tags use the raw slug. Never
    # authorize a sanitized id against a different bot's profile.
    return sql_allowed("b.bot_id", [BehaviorTag.PRIVATE, BehaviorTag.INACTIVE], prefix=prefix)


@dataclass
class RepairResult:
    examined: int = 0
    repaired_content: int = 0
    repaired_meaning: int = 0
    permanent_failures: int = 0
    transient_failures: int = 0
    stopped: bool = False
    busy: bool = False
    dimension_mismatch: bool = False

    def as_dict(self) -> dict:
        return vars(self).copy()


class EmbeddingRepair:
    """Run a capped keyset pass; persist cursor after EACH attempted row.

    Permanently bad rows do not starve later candidates. Missing vectors remain
    visible and are retried on subsequent cursor wraps. No row is marked repaired
    unless the source fields and private-bot policy still match at UPDATE time.
    """

    def __init__(self, config, engine):
        self.config = config
        self.engine = engine

    def report(self) -> dict[str, dict]:
        """Report-only: no DDL, and never read an excluded partition."""
        snapshot = BotTags(self.engine).snapshot()
        if not snapshot.available:
            raise RuntimeError("embedding repair: bot policy unavailable")
        allowed, guard = _allowed_sql("er_report")
        totals: dict[str, dict] = {}
        for bot_id in sorted(snapshot.known_bots):
            if not _eligible(bot_id, self.engine):
                continue
            backend = PostgreSQLMemoryBackend(self.config, bot_id, provision_schema=False)
            with self.engine.connect() as conn:
                if conn.execute(text("SELECT to_regclass(:table)"),
                                {"table": backend._memories_table_name}).scalar() is None:
                    continue
                row = conn.execute(text(f"""
                    SELECT COUNT(*) FILTER (WHERE m.embedding IS NULL) AS content,
                           COUNT(*) FILTER (WHERE m.meaning_embedding IS NULL) AS meaning
                    FROM {backend._memories_table_name} AS m,
                         (SELECT CAST(:raw_bot_id AS text) AS bot_id) AS b
                    WHERE m.bot_id = :storage_id AND {allowed}
                """), {**guard, "raw_bot_id": bot_id,
                       "storage_id": backend.bot_id_sanitized}).mappings().one()
            if row["content"] or row["meaning"]:
                totals[bot_id] = {"content": row["content"], "meaning": row["meaning"]}
        return totals

    def repair_bot(self, bot_id: str, *, limit: int = _MAX_ROWS,
                   stop: threading.Event | None = None) -> dict:
        if not 1 <= limit <= _MAX_ROWS:
            raise ValueError(f"repair limit must be 1..{_MAX_ROWS}")
        result = RepairResult()
        if not _eligible(bot_id, self.engine):
            return {**result.as_dict(), "excluded": True}
        # No schema provisioning on this path. A missing partition is an error,
        # never a cue to create one for a possibly protected bot.
        backend = PostgreSQLMemoryBackend(self.config, bot_id, provision_schema=False)
        table = backend._memories_table_name
        allowed, params = _allowed_sql()
        params = {**params, "storage_id": backend.bot_id_sanitized, "raw_bot_id": bot_id}
        with self.engine.connect() as conn:
            try:
                if not conn.execute(text(_LOCK_SQL)).scalar():
                    return {**result.as_dict(), "busy": True}
                conn.commit()
                try:
                    dimensions = conn.execute(text("""
                        SELECT atttypmod FROM pg_attribute
                        WHERE attrelid = CAST(:table AS regclass)
                          AND attname IN ('embedding', 'meaning_embedding')
                          AND NOT attisdropped
                    """), {"table": table}).scalars().all()
                    conn.commit()
                    if len(dimensions) != 2 or any(dim != EMBED_DIM for dim in dimensions):
                        result.dimension_mismatch = True
                        result.stopped = True
                        log.error("embedding repair: vector dimension mismatch on bot=%s", bot_id)
                        return result.as_dict()
                    # The schema is only created in explicitly enabled repair
                    # mode; a read-only report never creates anything.
                    if conn.execute(text("SELECT to_regclass('memory_embedding_repair_progress')")).scalar() is None:
                        conn.execute(text("""
                            CREATE TABLE IF NOT EXISTS memory_embedding_repair_progress (
                                bot_id VARCHAR(64) PRIMARY KEY,
                                last_id VARCHAR(36),
                                examined_at TIMESTAMPTZ NOT NULL DEFAULT now()
                            )
                        """))
                    conn.commit()
                    stored = conn.execute(text("""
                        SELECT last_id FROM memory_embedding_repair_progress
                        WHERE bot_id = :bot_id
                    """), {"bot_id": bot_id}).scalar()
                    conn.commit()
                    cursor = stored
                    failure_streak = 0
                    for _ in range(limit):
                        if stop is not None and stop.is_set():
                            result.stopped = True
                            break
                        row = conn.execute(text(f"""
                            SELECT m.id, m.content, m.intent, m.stakes,
                                   m.emotional_charge, m.recurrence_keywords,
                                   m.embedding IS NULL AS missing_content,
                                   m.meaning_embedding IS NULL AS missing_meaning
                            FROM {table} AS m,
                                 (SELECT CAST(:raw_bot_id AS text) AS bot_id) AS b
                            WHERE m.bot_id = :storage_id
                              AND (m.embedding IS NULL OR m.meaning_embedding IS NULL)
                              AND (CAST(:cursor AS varchar) IS NULL OR m.id > :cursor)
                              AND {allowed}
                            ORDER BY m.id LIMIT 1
                        """), {**params, "cursor": cursor}).mappings().first()
                        conn.commit()
                        if row is None:
                            # Completed a cycle; next pass starts at the beginning.
                            self._stamp_cursor(conn, bot_id, None)
                            break
                        result.examined += 1
                        cursor = row["id"]
                        # Save progress BEFORE the HTTP call. A crash can skip this
                        # row until wrap, but can never strand all following rows.
                        self._stamp_cursor(conn, bot_id, cursor)
                        if not _eligible(bot_id, self.engine, conn):
                            result.stopped = True
                            break
                        for field, body in (
                            ("embedding", row["content"] if row["missing_content"] else None),
                            ("meaning_embedding", meaning_text(
                                row["content"], row["intent"], row["stakes"],
                                row["emotional_charge"], row["recurrence_keywords"],
                            ) if row["missing_meaning"] else None),
                        ):
                            if body is None:
                                continue
                            if ((stop is not None and stop.is_set())
                                    or not _eligible(bot_id, self.engine, conn)):
                                result.stopped = True
                                break
                            conn.commit()
                            outcome = embed_with_status([body], backend.embedding_model)
                            if ((stop is not None and stop.is_set())
                                    or not _eligible(bot_id, self.engine, conn)):
                                result.stopped = True
                                break
                            conn.commit()
                            if outcome.vectors is None:
                                if outcome.failure == "permanent":
                                    result.permanent_failures += 1
                                    failure_streak = 0
                                else:
                                    result.transient_failures += 1
                                    failure_streak += 1
                                    if failure_streak >= _MAX_TRANSIENT:
                                        result.stopped = True
                                        break
                                continue
                            failure_streak = 0
                            vector = outcome.vectors[0]
                            if len(vector) != EMBED_DIM:
                                result.dimension_mismatch = True
                                result.stopped = True
                                break
                            changed = self._fill(conn, table, params, row, field, vector)
                            if changed:
                                if field == "embedding":
                                    result.repaired_content += 1
                                else:
                                    result.repaired_meaning += 1
                        if result.stopped or result.dimension_mismatch:
                            break
                finally:
                    # The connection may have failed; invalidate before the pool
                    # can hand out a session still holding this advisory lock.
                    try:
                        conn.rollback()
                        conn.execute(text(_UNLOCK_SQL))
                        conn.commit()
                    except Exception:
                        conn.invalidate()
                        raise
            except Exception:
                conn.rollback()
                conn.invalidate()
                raise
        return result.as_dict()

    @staticmethod
    def _stamp_cursor(conn, bot_id: str, cursor: str | None) -> None:
        conn.execute(text("""
            INSERT INTO memory_embedding_repair_progress(bot_id, last_id, examined_at)
            VALUES (:bot_id, :cursor, now())
            ON CONFLICT (bot_id) DO UPDATE SET
                last_id = EXCLUDED.last_id, examined_at = EXCLUDED.examined_at
        """), {"bot_id": bot_id, "cursor": cursor})
        conn.commit()

    @staticmethod
    def _fill(conn, table: str, params: dict, row, field: str,
              vector: list[float]) -> bool:
        # field is selected only from our own two constants above.
        assert field in {"embedding", "meaning_embedding"}
        allowed, _ = _allowed_sql()
        audit = ", meaning_updated_at = now()" if field == "meaning_embedding" else ""
        changed = conn.execute(text(f"""
            UPDATE {table} AS m
            SET {field} = :vector, updated_at = now(){audit}
            FROM (SELECT CAST(:raw_bot_id AS text) AS bot_id) AS b
            WHERE m.bot_id = :storage_id AND m.id = :id AND m.{field} IS NULL
              AND m.content = :content
              AND m.intent IS NOT DISTINCT FROM :intent
              AND m.stakes IS NOT DISTINCT FROM :stakes
              AND m.emotional_charge IS NOT DISTINCT FROM :emotional_charge
              AND m.recurrence_keywords IS NOT DISTINCT FROM CAST(:keywords AS jsonb)
              AND {allowed}
        """), {**params, "id": row["id"], "vector": str(vector),
               "content": row["content"], "intent": row["intent"],
               "stakes": row["stakes"], "emotional_charge": row["emotional_charge"],
               "keywords": (json.dumps(row["recurrence_keywords"])
                            if row["recurrence_keywords"] is not None else None),
               }).rowcount
        conn.commit()
        return bool(changed)
