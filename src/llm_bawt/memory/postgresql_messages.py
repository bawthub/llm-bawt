"""Permanent-message operations for the PostgreSQL memory backend.

Extracted from :mod:`llm_bawt.memory.postgresql` so the engine/schema composition
root does not also own message retrieval, search, and forget/restore behavior.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from sqlalchemy import delete, select, text, update
from sqlalchemy.orm import Session

from ..message_authorship import AuthorReference
from ..retrieval.fts_query import FtsQuery

logger = logging.getLogger(__name__)


class PostgreSQLMessageMixin:
    """Raw conversation-message storage and retrieval behavior."""

    def add_message(
        self,
        message_id: str,
        role: str,
        content: str,
        timestamp: float,
        session_id: str | None = None,
        attachments: list[dict] | None = None,
        reasoning: str | None = None,
        author: AuthorReference | None = None,
        extract_memory: bool = True,
    ) -> None:
        """Add a message through the extracted permanent-row store."""
        self._message_rows.upsert(
            message_id=message_id,
            role=role,
            content=content,
            timestamp=timestamp,
            session_id=session_id,
            attachments=attachments,
            reasoning=reasoning,
            author=author,
            extract_memory=extract_memory,
        )

    def get_unprocessed_messages(self, limit: int = 100) -> list[dict]:
        """Get messages that haven't been processed for memory extraction."""
        with Session(self.engine) as session:
            stmt = (
                select(self.messages_table)
                .where(self.messages_table.c.processed == False)
                .order_by(self.messages_table.c.timestamp.asc())
                .limit(limit)
            )
            rows = session.execute(stmt).fetchall()
            
            return [
                {
                    "id": row.id,
                    "role": row.role,
                    "content": row.content,
                    "timestamp": row.timestamp,
                    "session_id": row.session_id,
                }
                for row in rows
            ]

    def mark_messages_processed(self, message_ids: list[str]) -> None:
        """Mark messages as processed for memory extraction."""
        if not message_ids:
            return
        
        with Session(self.engine) as session:
            try:
                stmt = (
                    update(self.messages_table)
                    .where(self.messages_table.c.id.in_(message_ids))
                    .values(processed=True)
                )
                session.execute(stmt)
                session.commit()
            except Exception as e:
                session.rollback()
                logger.error(f"Failed to mark messages processed: {e}")

    def get_messages_by_ids(self, message_ids: list[str]) -> list[dict]:
        """Retrieve messages by their IDs (for context retrieval)."""
        if not message_ids:
            return []

        with Session(self.engine) as session:
            stmt = (
                select(self.messages_table)
                .where(self.messages_table.c.id.in_(message_ids))
                .order_by(self.messages_table.c.timestamp.asc())
            )
            rows = session.execute(stmt).fetchall()

            return [
                {
                    "id": row.id,
                    "role": row.role,
                    "content": row.content,
                    "timestamp": row.timestamp,
                    "session_id": row.session_id,
                }
                for row in rows
            ]

    def get_attachments_for_message_ids(self, message_ids: list[str]) -> dict[str, list[dict]]:
        """Return ``{message_id: attachments_list}`` for the given ids.

        TASK-226 — the canonical ``get_messages`` path drops the
        ``attachments`` JSONB column because the wider LLM-prep code
        path doesn't want it. ``/v1/history`` does, so it fetches the
        column directly via this focused query. Always returns an entry
        per requested id (missing rows or NULL attachments map to
        ``[]``) so callers can iterate without defensive checks.
        """
        if not message_ids:
            return {}

        sql = text(
            f"SELECT id, attachments FROM {self._messages_table_name} "
            "WHERE id = ANY(:ids)"
        )
        result: dict[str, list[dict]] = {mid: [] for mid in message_ids}
        with self.engine.connect() as conn:
            rows = conn.execute(sql, {"ids": list(message_ids)}).mappings().all()
            for row in rows:
                refs = row.get("attachments") or []
                if not isinstance(refs, list):
                    # Defensive: a corrupt JSONB cell shouldn't take the
                    # whole page down — log via debug and treat as empty.
                    logger.debug(
                        "Unexpected attachments shape for msg %s: %r",
                        row.get("id"),
                        type(refs).__name__,
                    )
                    refs = []
                result[row["id"]] = list(refs)
        return result

    def get_reasoning_for_message_ids(self, message_ids: list[str]) -> dict[str, str | None]:
        """Return ``{message_id: reasoning}`` for the given ids (TASK-301).

        Like ``get_attachments_for_message_ids``, this is a focused read of a
        column the canonical ``get_messages`` path deliberately drops (reasoning
        must never re-enter LLM context). ``/v1/history`` calls it to rehydrate
        the collapsed "Thought process" lane on reload. Missing rows / NULL map
        to ``None`` so callers can index unconditionally.
        """
        if not message_ids:
            return {}

        sql = text(
            f"SELECT id, reasoning FROM {self._messages_table_name} "
            "WHERE id = ANY(:ids)"
        )
        result: dict[str, str | None] = {mid: None for mid in message_ids}
        with self.engine.connect() as conn:
            rows = conn.execute(sql, {"ids": list(message_ids)}).mappings().all()
            for row in rows:
                result[row["id"]] = row.get("reasoning")
        return result

    def set_interrupt_anchor(
        self,
        message_id: str,
        source_message_id: str,
        content_offset: int,
    ) -> bool:
        """Persist the assistant-turn anchor for one accepted steer message."""
        if not message_id or not source_message_id or content_offset < 0:
            return False

        with self.engine.begin() as conn:
            result = conn.execute(
                text(
                    f"UPDATE {self._messages_table_name} "
                    "SET interrupt_source_message_id=:source_message_id, "
                    "interrupt_content_offset=:content_offset "
                    "WHERE id=:message_id"
                ),
                {
                    "message_id": message_id,
                    "source_message_id": source_message_id,
                    "content_offset": content_offset,
                },
            )
        return bool(result.rowcount)

    def get_interrupt_anchors_for_message_ids(
        self,
        message_ids: list[str],
    ) -> dict[str, tuple[str, int] | None]:
        """Return durable interrupt anchors for a history page."""
        if not message_ids:
            return {}

        result: dict[str, tuple[str, int] | None] = {mid: None for mid in message_ids}
        with self.engine.connect() as conn:
            rows = conn.execute(
                text(
                    f"SELECT id, interrupt_source_message_id, interrupt_content_offset "
                    f"FROM {self._messages_table_name} WHERE id = ANY(:ids)"
                ),
                {"ids": list(message_ids)},
            ).mappings().all()
        for row in rows:
            source_id = row.get("interrupt_source_message_id")
            offset = row.get("interrupt_content_offset")
            if source_id and isinstance(offset, int) and offset >= 0:
                result[str(row["id"])] = (str(source_id), offset)
        return result

    def add(self, message_id: str, role: str, content: str, timestamp: float) -> None:
        """Add a message to storage (implements MemoryBackend interface).
        
        For backwards compatibility, this adds to the messages table.
        """
        self.add_message(message_id, role, content, timestamp)

    def search_messages_by_text(
        self,
        query: str,
        n_results: int = 5,
        exclude_recent_seconds: float = 5.0,
        role_filter: str | None = "user",
        since: float | None = None,
        until: float | None = None,
    ) -> list[dict]:
        """Search raw messages using PostgreSQL full-text search.

        This is a fallback when no distilled memories exist yet.
        Uses OR logic so any matching word, identifier or quoted phrase
        returns results (:class:`llm_bawt.retrieval.FtsQuery`).

        Args:
            query: Search query
            n_results: Max number of results
            exclude_recent_seconds: Exclude messages from the last N seconds to avoid
                                   finding the query message itself
            role_filter: Only include messages with this role (default: "user" to avoid
                        retrieving assistant hallucinations as facts). Set to None to
                        include all roles.
            since: Unix timestamp - only include messages after this time.
            until: Unix timestamp - only include messages before this time.
        """
        if not query or query.isspace():
            return []

        fts = FtsQuery.parse(query)
        if fts is None:
            return []
        tsquery = fts.tsquery_sql()

        cutoff_time = time.time() - exclude_recent_seconds

        with self.engine.connect() as conn:
            try:
                # Build filter clauses
                role_clause = "AND role = :role" if role_filter else ""
                since_clause = "AND timestamp >= :since" if since is not None else ""
                until_clause = "AND timestamp <= :until" if until is not None else ""

                sql = text(f"""
                    SELECT id, role, content, timestamp,
                           ts_rank(content_tsv, {tsquery}) AS rank
                    FROM {self._messages_table_name}
                    WHERE content_tsv @@ {tsquery}
                    AND timestamp < :cutoff
                    {role_clause}
                    {since_clause}
                    {until_clause}
                    ORDER BY rank DESC, timestamp DESC
                    LIMIT :limit
                """)

                params: dict[str, Any] = {
                    **fts.params,
                    "limit": n_results,
                    "cutoff": cutoff_time,
                }
                if role_filter:
                    params["role"] = role_filter
                if since is not None:
                    params["since"] = since
                if until is not None:
                    params["until"] = until

                rows = conn.execute(sql, params).fetchall()

                return [
                    {
                        "id": row.id,
                        "content": row.content,
                        "role": row.role,
                        "timestamp": row.timestamp,
                        "relevance": row.rank,
                    }
                    for row in rows
                ]

            except Exception as e:
                logger.error(f"Failed to search messages: {e}")
                return []

    def ignore_recent_messages(self, count: int) -> int:
        """Move the last N messages to the forgotten table.
        
        Returns the number of messages actually forgotten.
        """
        with self.engine.connect() as conn:
            try:
                # Get IDs of last N messages
                select_sql = text(f"""
                    SELECT id, role, content, timestamp, session_id, processed, created_at,
                           author_entity_type, author_entity_id
                    FROM {self._messages_table_name}
                    ORDER BY timestamp DESC
                    LIMIT :count
                """)
                rows = conn.execute(select_sql, {"count": count}).fetchall()
                
                if not rows:
                    return 0
                
                ids_to_forget = [row.id for row in rows]
                
                # Insert into forgotten table
                for row in rows:
                    insert_sql = text(f"""
                        INSERT INTO {self._forgotten_table_name}
                        (id, role, content, timestamp, session_id, processed, created_at,
                         author_entity_type, author_entity_id, forgotten_at)
                        VALUES (:id, :role, :content, :timestamp, :session_id, :processed, :created_at,
                                :author_entity_type, :author_entity_id, CURRENT_TIMESTAMP)
                        ON CONFLICT (bot_id, id) DO NOTHING
                    """)
                    conn.execute(insert_sql, {
                        "id": row.id,
                        "role": row.role,
                        "content": row.content,
                        "timestamp": row.timestamp,
                        "session_id": row.session_id,
                        "processed": row.processed,
                        "created_at": row.created_at,
                        "author_entity_type": row.author_entity_type,
                        "author_entity_id": row.author_entity_id,
                    })
                
                # Delete from messages table
                delete_sql = text(f"""
                    DELETE FROM {self._messages_table_name}
                    WHERE id = ANY(:ids)
                """)
                conn.execute(delete_sql, {"ids": ids_to_forget})
                conn.commit()
                
                logger.debug(f"Forgot {len(ids_to_forget)} messages")
                return len(ids_to_forget)
            except Exception as e:
                conn.rollback()
                logger.error(f"Failed to forget messages: {e}")
                return 0

    def ignore_messages_since_minutes(self, minutes: int) -> int:
        """Move all messages from the last N minutes to the forgotten table.
        
        Returns the number of messages actually forgotten.
        """
        cutoff = time.time() - (minutes * 60)
        with self.engine.connect() as conn:
            try:
                # Get messages to forget
                select_sql = text(f"""
                    SELECT id, role, content, timestamp, session_id, processed, created_at,
                           author_entity_type, author_entity_id
                    FROM {self._messages_table_name}
                    WHERE timestamp >= :cutoff
                    ORDER BY timestamp ASC
                """)
                rows = conn.execute(select_sql, {"cutoff": cutoff}).fetchall()
                
                if not rows:
                    return 0
                
                ids_to_forget = [row.id for row in rows]
                
                # Insert into forgotten table
                for row in rows:
                    insert_sql = text(f"""
                        INSERT INTO {self._forgotten_table_name}
                        (id, role, content, timestamp, session_id, processed, created_at,
                         author_entity_type, author_entity_id, forgotten_at)
                        VALUES (:id, :role, :content, :timestamp, :session_id, :processed, :created_at,
                                :author_entity_type, :author_entity_id, CURRENT_TIMESTAMP)
                        ON CONFLICT (bot_id, id) DO NOTHING
                    """)
                    conn.execute(insert_sql, {
                        "id": row.id,
                        "role": row.role,
                        "content": row.content,
                        "timestamp": row.timestamp,
                        "session_id": row.session_id,
                        "processed": row.processed,
                        "created_at": row.created_at,
                        "author_entity_type": row.author_entity_type,
                        "author_entity_id": row.author_entity_id,
                    })
                
                # Delete from messages table
                delete_sql = text(f"""
                    DELETE FROM {self._messages_table_name}
                    WHERE id = ANY(:ids)
                """)
                conn.execute(delete_sql, {"ids": ids_to_forget})
                conn.commit()
                
                logger.debug(f"Forgot {len(ids_to_forget)} messages from last {minutes} minutes")
                return len(ids_to_forget)
            except Exception as e:
                conn.rollback()
                logger.error(f"Failed to forget messages: {e}")
                return 0

    def get_message_by_id(
        self,
        message_id: str,
        before: int = 0,
        after: int = 0,
    ) -> dict | None:
        """Get a specific message by its ID, optionally with surrounding context.

        Supports both full UUID and prefix matching (first 8 chars).

        Args:
            message_id: Full UUID or prefix (min 8 chars).
            before: Number of messages to include before the match (by timestamp).
            after: Number of messages to include after the match (by timestamp).

        Returns:
            If before/after are both 0: a single message dict or None.
            If either is > 0: a dict with ``message`` (the matched row),
            ``before`` (list, oldest-first), and ``after`` (list, oldest-first),
            or None if the target ID wasn't found.
        """
        with self.engine.connect() as conn:
            try:
                # Find the message (support prefix matching)
                if len(message_id) < 36:
                    select_sql = text(f"""
                        SELECT id, role, content, timestamp, session_id, processed, created_at,
                           author_entity_type, author_entity_id, summary_metadata
                        FROM {self._messages_table_name}
                        WHERE id LIKE :id_pattern
                        LIMIT 1
                    """)
                    row = conn.execute(select_sql, {"id_pattern": f"{message_id}%"}).fetchone()
                else:
                    select_sql = text(f"""
                        SELECT id, role, content, timestamp, session_id, processed, created_at,
                           author_entity_type, author_entity_id, summary_metadata
                        FROM {self._messages_table_name}
                        WHERE id = :id
                    """)
                    row = conn.execute(select_sql, {"id": message_id}).fetchone()

                if not row:
                    return None

                def _row_to_dict(r):
                    return {
                        "id": r.id,
                        "role": r.role,
                        "content": r.content,
                        "timestamp": r.timestamp,
                        "session_id": r.session_id,
                        "processed": r.processed,
                        "created_at": str(r.created_at) if r.created_at else None,
                        "author_entity_type": r.author_entity_type,
                        "author_entity_id": r.author_entity_id,
                        "summary_metadata": r.summary_metadata,
                    }

                # No surrounding context requested — return single dict (back-compat).
                if before == 0 and after == 0:
                    return _row_to_dict(row)

                target_ts = row.timestamp
                target_id = row.id
                result_before: list[dict] = []
                result_after: list[dict] = []

                if before > 0:
                    before_sql = text(f"""
                        SELECT id, role, content, timestamp, session_id, processed, created_at,
                           author_entity_type, author_entity_id, summary_metadata
                        FROM {self._messages_table_name}
                        WHERE (timestamp < :ts OR (timestamp = :ts AND id < :tid))
                          AND role != 'system'
                        ORDER BY timestamp DESC, id DESC
                        LIMIT :n
                    """)
                    before_rows = conn.execute(before_sql, {"ts": target_ts, "tid": target_id, "n": before}).fetchall()
                    result_before = [_row_to_dict(r) for r in reversed(before_rows)]

                if after > 0:
                    after_sql = text(f"""
                        SELECT id, role, content, timestamp, session_id, processed, created_at,
                           author_entity_type, author_entity_id, summary_metadata
                        FROM {self._messages_table_name}
                        WHERE (timestamp > :ts OR (timestamp = :ts AND id > :tid))
                          AND role != 'system'
                        ORDER BY timestamp ASC, id ASC
                        LIMIT :n
                    """)
                    after_rows = conn.execute(after_sql, {"ts": target_ts, "tid": target_id, "n": after}).fetchall()
                    result_after = [_row_to_dict(r) for r in after_rows]

                return {
                    "before": result_before,
                    "message": _row_to_dict(row),
                    "after": result_after,
                }
            except Exception as e:
                logger.error(f"Failed to get message {message_id}: {e}")
                return None

    def ignore_message_by_id(self, message_id: str) -> bool:
        """Move a specific message to the forgotten table by its ID.
        
        Supports both full UUID and prefix matching (first 8 chars).
        Returns True if a message was forgotten, False otherwise.
        """
        with self.engine.connect() as conn:
            try:
                # Find the message (support prefix matching)
                if len(message_id) < 36:
                    select_sql = text(f"""
                        SELECT id, role, content, timestamp, session_id, processed, created_at,
                           interrupt_source_message_id, interrupt_content_offset,
                           author_entity_type, author_entity_id
                        FROM {self._messages_table_name}
                        WHERE id LIKE :id_pattern
                        LIMIT 1
                    """)
                    row = conn.execute(select_sql, {"id_pattern": f"{message_id}%"}).fetchone()
                else:
                    select_sql = text(f"""
                        SELECT id, role, content, timestamp, session_id, processed, created_at,
                           interrupt_source_message_id, interrupt_content_offset,
                           author_entity_type, author_entity_id
                        FROM {self._messages_table_name}
                        WHERE id = :id
                    """)
                    row = conn.execute(select_sql, {"id": message_id}).fetchone()
                
                if not row:
                    logger.debug(f"Message {message_id} not found")
                    return False
                
                # Insert into forgotten table, preserving canonical authorship.
                insert_sql = text(f"""
                    INSERT INTO {self._forgotten_table_name}
                    (id, role, content, timestamp, session_id, processed, created_at,
                     interrupt_source_message_id, interrupt_content_offset,
                     author_entity_type, author_entity_id, forgotten_at)
                    VALUES (:id, :role, :content, :timestamp, :session_id, :processed, :created_at,
                            :interrupt_source_message_id, :interrupt_content_offset,
                            :author_entity_type, :author_entity_id, CURRENT_TIMESTAMP)
                    ON CONFLICT (bot_id, id) DO NOTHING
                """)
                conn.execute(insert_sql, {
                    "id": row.id,
                    "role": row.role,
                    "content": row.content,
                    "timestamp": row.timestamp,
                    "session_id": row.session_id,
                    "processed": row.processed,
                    "created_at": row.created_at,
                    "interrupt_source_message_id": row.interrupt_source_message_id,
                    "interrupt_content_offset": row.interrupt_content_offset,
                    "author_entity_type": row.author_entity_type,
                    "author_entity_id": row.author_entity_id,
                })
                
                # Delete from messages table
                delete_sql = text(f"""
                    DELETE FROM {self._messages_table_name}
                    WHERE id = :id
                """)
                conn.execute(delete_sql, {"id": row.id})
                conn.commit()
                
                logger.debug(f"Forgot message {row.id}")
                return True
            except Exception as e:
                conn.rollback()
                logger.error(f"Failed to forget message {message_id}: {e}")
                return False

    def restore_ignored_messages(self) -> int:
        """Restore all forgotten messages back to the messages table.
        
        Returns the number of messages restored.
        """
        with self.engine.connect() as conn:
            try:
                # Get all forgotten messages
                select_sql = text(f"""
                    SELECT id, role, content, timestamp, session_id, processed, created_at,
                           interrupt_source_message_id, interrupt_content_offset,
                           author_entity_type, author_entity_id
                    FROM {self._forgotten_table_name}
                    ORDER BY timestamp ASC
                """)
                rows = conn.execute(select_sql).fetchall()
                
                if not rows:
                    return 0
                
                ids_to_restore = [row.id for row in rows]
                
                # Insert back into messages table with the original author.
                for row in rows:
                    insert_sql = text(f"""
                        INSERT INTO {self._messages_table_name}
                        (id, role, content, timestamp, session_id, processed, created_at,
                         interrupt_source_message_id, interrupt_content_offset,
                         author_entity_type, author_entity_id)
                        VALUES (:id, :role, :content, :timestamp, :session_id, :processed, :created_at,
                                :interrupt_source_message_id, :interrupt_content_offset,
                                :author_entity_type, :author_entity_id)
                        ON CONFLICT (bot_id, id) DO NOTHING
                    """)
                    conn.execute(insert_sql, {
                        "id": row.id,
                        "role": row.role,
                        "content": row.content,
                        "timestamp": row.timestamp,
                        "session_id": row.session_id,
                        "processed": row.processed,
                        "created_at": row.created_at,
                        "author_entity_type": row.author_entity_type,
                        "author_entity_id": row.author_entity_id,
                    })
                
                # Delete from forgotten table
                delete_sql = text(f"""
                    DELETE FROM {self._forgotten_table_name}
                    WHERE id = ANY(:ids)
                """)
                conn.execute(delete_sql, {"ids": ids_to_restore})
                conn.commit()
                
                logger.debug(f"Restored {len(ids_to_restore)} forgotten messages")
                return len(ids_to_restore)
            except Exception as e:
                conn.rollback()
                logger.error(f"Failed to restore messages: {e}")
                return 0

    def get_ignored_count(self) -> int:
        """Get the count of currently forgotten messages."""
        with self.engine.connect() as conn:
            result = conn.execute(text(f"SELECT COUNT(*) FROM {self._forgotten_table_name}"))
            return result.scalar() or 0

    def preview_recent_messages(self, count: int) -> list[dict]:
        """Preview the last N messages (for confirmation before forget).
        
        Returns list of message dicts with id, role, content, timestamp.
        """
        with self.engine.connect() as conn:
            sql = text(f"""
                SELECT id, role, content, timestamp
                FROM {self._messages_table_name}
                ORDER BY timestamp DESC
                LIMIT :count
            """)
            rows = conn.execute(sql, {"count": count}).fetchall()
            
            return [
                {
                    "id": row.id,
                    "role": row.role,
                    "content": row.content,
                    "timestamp": row.timestamp,
                }
                for row in reversed(rows)  # Return in chronological order
            ]

    def preview_messages_since_minutes(self, minutes: int) -> list[dict]:
        """Preview messages from the last N minutes (for confirmation before forget).
        
        Returns list of message dicts with id, role, content, timestamp.
        """
        cutoff = time.time() - (minutes * 60)
        with self.engine.connect() as conn:
            sql = text(f"""
                SELECT id, role, content, timestamp
                FROM {self._messages_table_name}
                WHERE timestamp >= :cutoff
                ORDER BY timestamp ASC
            """)
            rows = conn.execute(sql, {"cutoff": cutoff}).fetchall()
            
            return [
                {
                    "id": row.id,
                    "role": row.role,
                    "content": row.content,
                    "timestamp": row.timestamp,
                }
                for row in rows
            ]

    def preview_ignored_messages(self) -> list[dict]:
        """Preview all currently forgotten messages (for confirmation before restore).
        
        Returns list of message dicts with id, role, content, timestamp.
        """
        with self.engine.connect() as conn:
            sql = text(f"""
                SELECT id, role, content, timestamp
                FROM {self._forgotten_table_name}
                ORDER BY timestamp ASC
            """)
            rows = conn.execute(sql).fetchall()
            
            return [
                {
                    "id": row.id,
                    "role": row.role,
                    "content": row.content,
                    "timestamp": row.timestamp,
                }
                for row in rows
            ]

