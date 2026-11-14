"""Memory/history schema migrations for llm-bawt.

Split out of ``migrations.py`` (TASK-553). These migrations touch the
per-bot memory + message tables and the shared ``sessions`` table:
tag backfill, meaning-embedding backfill, the ``recalled_history`` column,
and the ``sessions`` backfill from historical message rows.

The public ``migrations`` facade re-exports every function here, so
``from llm_bawt.memory.migrations import backfill_empty_tags`` keeps working.
"""

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def backfill_empty_tags(
    backend: Any,
    batch_size: int = 100,
    dry_run: bool = False,
) -> dict:
    """Migrate existing memories: populate empty tags with default.

    For each memory where tags IS NULL or empty, set tags = ["misc"].

    Args:
        backend: PostgreSQLMemoryBackend instance
        batch_size: Number of rows to process per commit
        dry_run: If True, only report what would be done

    Returns:
        Dict with migration statistics
    """
    from sqlalchemy import text

    updated = 0

    with backend.engine.connect() as conn:
        # Count total needing migration (NULL or empty array)
        count_sql = text(f"""
            SELECT COUNT(*) FROM {backend._memories_table_name}
            WHERE tags IS NULL OR tags = '[]'::jsonb
        """)
        total = conn.execute(count_sql).scalar() or 0
        logger.debug(f"Found {total} memories needing tag backfill")

        if total == 0:
            return {"updated": 0, "total": 0}

        if dry_run:
            logger.debug(f"[DRY RUN] Would update {total} memories")
            return {"updated": 0, "total": total, "dry_run": True}

        # Process in batches
        while True:
            fetch_sql = text(f"""
                SELECT id FROM {backend._memories_table_name}
                WHERE tags IS NULL OR tags = '[]'::jsonb
                LIMIT :limit
            """)
            rows = conn.execute(fetch_sql, {"limit": batch_size}).fetchall()
            if not rows:
                break

            for row in rows:
                update_sql = text(f"""
                    UPDATE {backend._memories_table_name}
                    SET tags = CAST(:tags AS jsonb), updated_at = CURRENT_TIMESTAMP
                    WHERE id = :id
                """)
                conn.execute(update_sql, {"id": row.id, "tags": json.dumps(["misc"])})
                updated += 1

            conn.commit()
            logger.debug(f"Migrated {updated}/{total} memories")

    logger.debug(f"Tag backfill complete: {updated} updated")
    return {"updated": updated, "total": total}


def backfill_meaning_embeddings(
    backend: Any,
    batch_size: int = 50,
    dry_run: bool = False,
) -> dict:
    """Generate meaning embeddings for memories that have meaning fields but no meaning_embedding.

    Args:
        backend: PostgreSQLMemoryBackend instance
        batch_size: Number of rows to process per commit
        dry_run: If True, only report what would be done

    Returns:
        Dict with migration statistics
    """
    from sqlalchemy import text
    from .embeddings import generate_embedding, meaning_text

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    updated = 0
    failed = 0
    skipped = 0

    with backend.engine.connect() as conn:
        # Count total needing migration
        count_sql = text(f"""
            SELECT COUNT(*) FROM {backend._memories_table_name}
            WHERE meaning_embedding IS NULL
              AND (intent IS NOT NULL OR stakes IS NOT NULL OR recurrence_keywords IS NOT NULL
                   OR emotional_charge IS NOT NULL)
        """)
        total = conn.execute(count_sql).scalar() or 0
        logger.debug(f"Found {total} memories needing meaning embedding generation")

        if total == 0:
            return {"updated": 0, "failed": 0, "total": 0}

        if dry_run:
            logger.debug(f"[DRY RUN] Would generate embeddings for {total} memories")
            return {"updated": 0, "failed": 0, "total": total, "dry_run": True}

        # Snapshot the pass boundary and advance even for empty/failed embeds.
        # Otherwise an unchanged first page can be fetched forever.
        upper_id = conn.execute(text(f"SELECT MAX(id) FROM {backend._memories_table_name}")).scalar()
        conn.commit()
        after_id = ""
        while True:
            fetch_sql = text(f"""
                SELECT id, content, intent, stakes, emotional_charge, recurrence_keywords
                FROM {backend._memories_table_name}
                WHERE meaning_embedding IS NULL
                  AND (intent IS NOT NULL OR stakes IS NOT NULL OR recurrence_keywords IS NOT NULL
                       OR emotional_charge IS NOT NULL)
                  AND id > :after_id AND id <= :upper_id
                ORDER BY id
                LIMIT :limit
            """)
            rows = conn.execute(fetch_sql, {
                "limit": batch_size, "after_id": after_id, "upper_id": upper_id,
            }).fetchall()
            conn.commit()  # No reader transaction spans the embedding request.
            if not rows:
                break
            after_id = rows[-1].id

            for row in rows:
                try:
                    keywords = row.recurrence_keywords or []
                    if isinstance(keywords, str):
                        keywords = json.loads(keywords)
                    body = meaning_text(row.content or "", row.intent, row.stakes,
                                        row.emotional_charge, keywords)
                    if not body:
                        skipped += 1
                        continue

                    emb = generate_embedding(body, backend.embedding_model)
                    if emb:
                        update_sql = text(f"""
                            UPDATE {backend._memories_table_name}
                            SET meaning_embedding = :embedding,
                                meaning_updated_at = CURRENT_TIMESTAMP,
                                updated_at = CURRENT_TIMESTAMP
                            WHERE id = :id AND meaning_embedding IS NULL
                              AND content IS NOT DISTINCT FROM :old_content
                              AND intent IS NOT DISTINCT FROM :old_intent
                              AND stakes IS NOT DISTINCT FROM :old_stakes
                              AND emotional_charge IS NOT DISTINCT FROM :old_emotion
                              AND recurrence_keywords IS NOT DISTINCT FROM CAST(:old_keywords AS jsonb)
                        """)
                        result = conn.execute(update_sql, {
                            "id": row.id,
                            "old_content": row.content, "old_intent": row.intent,
                            "old_stakes": row.stakes, "old_emotion": row.emotional_charge,
                            "old_keywords": (json.dumps(row.recurrence_keywords)
                                             if row.recurrence_keywords is not None else None),
                            "embedding": f"[{','.join(str(x) for x in emb)}]",
                        })
                        conn.commit()
                        updated += result.rowcount
                        skipped += int(result.rowcount == 0)
                    else:
                        failed += 1
                except Exception as e:
                    conn.rollback()
                    logger.warning(f"Failed to generate meaning embedding for {row.id}: {e}")
                    failed += 1

            logger.debug(f"Generated embeddings for {updated}/{total} memories")

    logger.debug(f"Meaning embedding generation complete: {updated} updated, {failed} failed")
    return {"updated": updated, "failed": failed, "skipped": skipped, "total": total}


def add_recalled_history_column(backend: Any, dry_run: bool = False) -> dict:
    """Add recalled_history boolean column to message tables if missing.

    This column tracks messages that were re-inserted into context via the
    history recall tool. Future summarization passes should skip these
    to avoid creating duplicate summaries.

    Args:
        backend: PostgreSQLMemoryBackend instance
        dry_run: If True, only report what would be done

    Returns:
        Dict with migration statistics
    """
    from sqlalchemy import text

    table_name = backend._messages_table_name

    with backend.engine.connect() as conn:
        # Check if column already exists
        check_sql = text("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = :table_name AND column_name = 'recalled_history'
        """)
        exists = conn.execute(check_sql, {"table_name": table_name}).fetchone()
        if exists:
            logger.debug(f"recalled_history column already exists on {table_name}")
            return {"added": False, "already_exists": True}

        if dry_run:
            logger.debug(f"[DRY RUN] Would add recalled_history column to {table_name}")
            return {"added": False, "dry_run": True}

        alter_sql = text(f"""
            ALTER TABLE {table_name}
            ADD COLUMN recalled_history BOOLEAN DEFAULT FALSE
        """)
        conn.execute(alter_sql)
        conn.commit()
        logger.debug(f"Added recalled_history column to {table_name}")
        return {"added": True}


def backfill_sessions(
    backend: Any,
    dry_run: bool = False,
) -> dict:
    """Backfill the shared `sessions` table from existing message history.

    Scans every `*_messages` table in the database, finds each distinct
    session_id, and inserts a corresponding session row with started_at /
    ended_at inferred from MIN/MAX message timestamps. The bot_id is
    derived from the table name prefix (`{bot_id}_messages`).

    Sessions backfilled this way are marked `status='archived'` since
    they predate the new tracking — there's no live process holding them
    open. Existing rows are left alone (ON CONFLICT DO NOTHING).

    Args:
        backend: PostgreSQLMemoryBackend instance — only used for the engine.
        dry_run: If True, only report what would be done.

    Returns:
        Dict with counts: {scanned_tables, distinct_sessions,
        inserted, skipped, dry_run}.
    """
    from sqlalchemy import text
    from datetime import datetime, timezone

    inserted = 0
    skipped = 0
    distinct_sessions = 0

    with backend.engine.connect() as conn:
        # Preview must not create/alter tables or migrate existing sessions.
        if not dry_run:
            _prepare_sessions(conn)

        # Discover all *_messages tables. Excludes *_forgotten_messages by
        # requiring the suffix to be _messages exactly.
        tables_sql = text("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
              AND table_name LIKE '%\\_messages' ESCAPE '\\'
              AND table_name NOT LIKE '%\\_forgotten\\_messages' ESCAPE '\\'
        """)
        tables = [row.table_name for row in conn.execute(tables_sql).fetchall()]
        logger.debug(f"Found {len(tables)} message tables: {tables}")

        for table_name in tables:
            # Bot id is the part before "_messages"
            if not table_name.endswith("_messages"):
                continue
            bot_id = table_name[: -len("_messages")]

            # Find distinct session_ids and their time ranges. Skip rows
            # where session_id is NULL — those messages had no session.
            sessions_sql = text(f"""
                SELECT
                    session_id,
                    MIN(timestamp) AS first_ts,
                    MAX(timestamp) AS last_ts,
                    COUNT(*) AS msg_count
                FROM {table_name}
                WHERE session_id IS NOT NULL
                GROUP BY session_id
            """)
            rows = conn.execute(sessions_sql).fetchall()
            logger.debug(
                f"Bot {bot_id}: {len(rows)} distinct session(s) in {table_name}"
            )

            for row in rows:
                distinct_sessions += 1
                if dry_run:
                    skipped += 1
                    continue

                # Convert UNIX timestamps (seconds) to datetimes.
                started_at = datetime.fromtimestamp(
                    row.first_ts, tz=timezone.utc
                )
                ended_at = datetime.fromtimestamp(
                    row.last_ts, tz=timezone.utc
                )

                # TASK-250 (review finding): an archived row must carry its
                # archival timestamp — use the inferred ended_at.
                insert_sql = text("""
                    INSERT INTO sessions
                        (id, bot_id, started_at, ended_at, archived_at, status,
                         session_metadata)
                    VALUES
                        (:id, :bot_id, :started_at, :ended_at, :ended_at,
                         'archived', CAST(:metadata AS jsonb))
                    ON CONFLICT (id) DO NOTHING
                """)
                metadata_json = json.dumps({
                    "backfilled": True,
                    "message_count": int(row.msg_count),
                })
                result = conn.execute(insert_sql, {
                    "id": row.session_id,
                    "bot_id": bot_id,
                    "started_at": started_at,
                    "ended_at": ended_at,
                    "metadata": metadata_json,
                })
                if result.rowcount and result.rowcount > 0:
                    inserted += 1
                else:
                    skipped += 1

            if not dry_run:
                conn.commit()

    return {
        "scanned_tables": len(tables) if 'tables' in locals() else 0,
        "distinct_sessions": distinct_sessions,
        "inserted": inserted,
        "skipped": skipped,
        "dry_run": dry_run,
    }


def _prepare_sessions(conn) -> None:
    """Bootstrap and migrate session schema only for an executing backfill."""
    from sqlalchemy import text

    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS sessions (
            id VARCHAR(36) PRIMARY KEY,
            bot_id VARCHAR(64) NOT NULL,
            started_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            ended_at TIMESTAMP,
            archived_at TIMESTAMP,
            status VARCHAR(16) NOT NULL DEFAULT 'active',
            session_metadata JSONB
        )
    """))
    conn.execute(text("""
        ALTER TABLE sessions ADD COLUMN IF NOT EXISTS user_id VARCHAR(64)
    """))
    conn.execute(text("""
        ALTER TABLE sessions ADD COLUMN IF NOT EXISTS archived_at TIMESTAMP
    """))
    conn.execute(text("""
        UPDATE sessions
        SET status = 'archived',
            archived_at = COALESCE(archived_at, ended_at)
        WHERE status = 'completed'
    """))
    conn.execute(text("""
        UPDATE sessions
        SET archived_at = ended_at
        WHERE status = 'archived'
          AND archived_at IS NULL
          AND ended_at IS NOT NULL
    """))
    conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_sessions_bot_started
        ON sessions(bot_id, started_at)
    """))
    conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_sessions_status
        ON sessions(status)
    """))
    conn.commit()
