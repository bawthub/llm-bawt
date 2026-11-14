"""Permanent-message operations for the PostgreSQL memory backend.

Extracted from :mod:`llm_bawt.memory.postgresql` so the engine/schema composition
root does not also own message retrieval, search, and forget/restore behavior.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from sqlalchemy import func, select, text, update
from sqlalchemy.orm import Session

from ..message_authorship import AuthorReference
from ..retrieval.fts_query import FtsQuery
from .message_store import _NativeSelectedOwnerConstraint, NativeSelectedOwnerDenied

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
        *,
        _native_selected_owner: _NativeSelectedOwnerConstraint | None = None,
    ) -> None:
        """Add a message through the extracted permanent-row store."""
        if _native_selected_owner is not None:
            if type(_native_selected_owner) is not _NativeSelectedOwnerConstraint:
                raise NativeSelectedOwnerDenied()
            _native_selected_owner.validate(session_id, self.bot_id)
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
            **({"_native_selected_owner": _native_selected_owner}
               if _native_selected_owner is not None else {}),
        )

    def _unprocessed_owned(self):
        from .postgresql import sessions_table

        return (select(self.messages_table)
                .join(sessions_table, sessions_table.c.id == self.messages_table.c.session_id)
                .where(self.messages_table.c.processed.is_(False),
                       sessions_table.c.bot_id == self.bot_id,
                       sessions_table.c.status.in_(("active", "archived")),
                       sessions_table.c.user_id.is_not(None),
                       sessions_table.c.user_id.not_in(("", "system"))))

    def get_unprocessed_message_users(self, limit: int = 10) -> list[str]:
        """Discover proven owners, oldest pending batch first; never infer one."""
        from .postgresql import sessions_table

        stmt = (self._unprocessed_owned().with_only_columns(sessions_table.c.user_id)
                .group_by(sessions_table.c.user_id)
                .order_by(func.min(self.messages_table.c.timestamp), sessions_table.c.user_id)
                .limit(limit))
        with Session(self.engine) as session:
            return list(session.execute(stmt).scalars())

    def get_unprocessed_messages(self, limit: int = 100, *, user_id: str | None = None) -> list[dict]:
        """Select one proven owner's messages before ordering or limiting."""
        if not user_id or user_id == "system":
            return []
        from .postgresql import sessions_table

        with Session(self.engine) as session:
            stmt = (self._unprocessed_owned()
                    .where(sessions_table.c.user_id == user_id)
                    .order_by(self.messages_table.c.timestamp.asc(), self.messages_table.c.id)
                    .limit(limit))
            rows = session.execute(stmt).fetchall()
            
            return [
                {
                    "id": row.id,
                    "role": row.role,
                    "content": row.content,
                    "timestamp": row.timestamp,
                    "session_id": row.session_id,
                    "author_entity_type": row.author_entity_type,
                    "author_entity_id": row.author_entity_id,
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
                    "author_entity_type": row.author_entity_type,
                    "author_entity_id": row.author_entity_id,
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

    def _owned_selection(
        self,
        *,
        user_id: str,
        session_id: str | None,
        count: int | None,
        minutes: int | None,
    ) -> tuple[str, dict[str, Any]]:
        """Select the caller's newest messages (TASK-2279).

        A bot partition holds every user's threads, so count/time selection is
        bound to the caller's sessions, and to one thread when given. Returns a
        sub-select of IDs plus its bind parameters.
        """
        if not user_id or not str(user_id).strip():
            raise ValueError("user_id is required to forget messages")
        if (count is None) == (minutes is None):
            raise ValueError("pass exactly one of count or minutes")
        from .short_term_message_reads import history_owner_clause

        params: dict[str, Any] = {}
        where = [history_owner_clause(params, bot_id=self.bot_id, user_id=user_id)]
        if session_id:
            where.append("session_id = :scope_session")
            params["scope_session"] = session_id
        if minutes is not None:
            where.append("timestamp >= :cutoff")
            params["cutoff"] = time.time() - (int(minutes) * 60)
            limit = ""
        else:
            limit = " LIMIT :count"
            params["count"] = int(count)
        return (
            f"SELECT id FROM {self._messages_table_name} WHERE {' AND '.join(where)}"
            f" ORDER BY timestamp DESC, id DESC{limit}",
            params,
        )

    def ignore_recent_messages(
        self, count: int, *, user_id: str, session_id: str | None = None,
    ) -> int:
        """Move the caller's last N messages to the forgotten table."""
        selected, params = self._owned_selection(
            user_id=user_id, session_id=session_id, count=count, minutes=None)
        return self._forget_where(f"id IN ({selected})", params)

    def ignore_messages_since_minutes(
        self, minutes: int, *, user_id: str, session_id: str | None = None,
    ) -> int:
        """Move the caller's messages from the last N minutes to forgotten storage."""
        selected, params = self._owned_selection(
            user_id=user_id, session_id=session_id, count=None, minutes=minutes)
        return self._forget_where(f"id IN ({selected})", params)

    def forget_owned_messages(
        self,
        *,
        user_id: str,
        session_id: str | None = None,
        count: int | None = None,
        minutes: int | None = None,
    ) -> dict[str, Any]:
        """Forget the caller's messages and the memories sourced from them.

        One transaction selects one stable ID set, moves exactly those rows,
        and deletes only memories whose sources include a moved ID.
        """
        from .message_store import MESSAGE_STATE_COLUMNS

        selected, params = self._owned_selection(
            user_id=user_id, session_id=session_id, count=count, minutes=minutes)
        cols = ", ".join(MESSAGE_STATE_COLUMNS)
        refresh = ", ".join(
            f"{c} = EXCLUDED.{c}" for c in (*MESSAGE_STATE_COLUMNS[1:], "forgotten_at")
        )
        with self.engine.begin() as conn:
            ids = [r.id for r in conn.execute(text(f"{selected} FOR UPDATE"), params)]
            if not ids:
                return {"message_ids": [], "messages_ignored": 0, "memories_deleted": 0}
            moved = conn.execute(text(f"""
                WITH moved AS (
                    DELETE FROM {self._messages_table_name}
                    WHERE id = ANY(:ids)
                    RETURNING {cols}
                )
                INSERT INTO {self._forgotten_table_name} ({cols}, forgotten_at)
                SELECT {cols}, CURRENT_TIMESTAMP FROM moved
                ON CONFLICT (bot_id, id) DO UPDATE SET {refresh}
                RETURNING id
            """), {"ids": ids}).fetchall()
            moved_ids = sorted(r.id for r in moved)
            memories = conn.execute(text(f"""
                DELETE FROM {self._memories_table_name}
                WHERE source_message_ids::jsonb ?| CAST(:ids AS text[])
                RETURNING id
            """), {"ids": moved_ids}).fetchall() if moved_ids else []
        logger.debug("Forgot %d messages and %d memories", len(moved_ids), len(memories))
        return {
            "message_ids": moved_ids,
            "messages_ignored": len(moved_ids),
            "memories_deleted": len(memories),
        }

    def _forget_where(self, where_sql: str, params: dict[str, Any]) -> int:
        """Move matching canonical messages, with all durable state, to forgotten storage.

        One statement deletes and copies, so a failure leaves both tables as
        they were. Re-forgetting an id refreshes its forgotten copy instead of
        discarding the canonical row.
        """
        from .message_store import MESSAGE_STATE_COLUMNS

        cols = ", ".join(MESSAGE_STATE_COLUMNS)
        refresh = ", ".join(
            f"{c} = EXCLUDED.{c}" for c in (*MESSAGE_STATE_COLUMNS[1:], "forgotten_at")
        )
        sql = text(f"""
            WITH moved AS (
                DELETE FROM {self._messages_table_name}
                WHERE {where_sql}
                RETURNING {cols}
            )
            INSERT INTO {self._forgotten_table_name} ({cols}, forgotten_at)
            SELECT {cols}, CURRENT_TIMESTAMP FROM moved
            ON CONFLICT (bot_id, id) DO UPDATE SET {refresh}
            RETURNING id
        """)
        with self.engine.begin() as conn:
            return len(conn.execute(sql, params).fetchall())

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
        # Supports prefix matching; a prefix forgets its first match, as before.
        if len(message_id) < 36:
            where, params = (
                f"id = (SELECT id FROM {self._messages_table_name} "
                "WHERE id LIKE :id_pattern LIMIT 1)",
                {"id_pattern": f"{message_id}%"},
            )
        else:
            where, params = "id = :id", {"id": message_id}
        try:
            forgotten = self._forget_where(where, params)
        except Exception as e:
            logger.error(f"Failed to forget message {message_id}: {e}")
            return False
        if not forgotten:
            logger.debug(f"Message {message_id} not found")
            return False
        logger.debug(f"Forgot message {message_id}")
        return True

    def restore_ignored_messages(self) -> int:
        """Restore all forgotten messages back to the messages table.
        
        Returns the number of messages restored.
        """
        from .message_store import MESSAGE_STATE_COLUMNS

        cols = ", ".join(MESSAGE_STATE_COLUMNS)
        # One statement moves every column back. A forgotten copy whose id is
        # already canonical stays put rather than being dropped; a concurrent
        # insert of the same id fails the statement and changes nothing.
        sql = text(f"""
            WITH moved AS (
                DELETE FROM {self._forgotten_table_name} f
                WHERE NOT EXISTS (
                    SELECT 1 FROM {self._messages_table_name} m WHERE m.id = f.id
                )
                RETURNING {cols}
            )
            INSERT INTO {self._messages_table_name} ({cols})
            SELECT {cols} FROM moved
            RETURNING id
        """)
        try:
            with self.engine.begin() as conn:
                restored = len(conn.execute(sql).fetchall())
                kept = conn.execute(
                    text(f"SELECT COUNT(*) FROM {self._forgotten_table_name}")
                ).scalar() or 0
        except Exception as e:
            logger.error(f"Failed to restore messages: {e}")
            return 0
        if kept:
            logger.warning(
                f"Kept {kept} forgotten messages whose ids are already in history"
            )
        logger.debug(f"Restored {restored} forgotten messages")
        return restored

    def get_ignored_count(self) -> int:
        """Get the count of currently forgotten messages."""
        with self.engine.connect() as conn:
            result = conn.execute(text(f"SELECT COUNT(*) FROM {self._forgotten_table_name}"))
            return result.scalar() or 0

    def _preview_owned(self, **selection: Any) -> list[dict]:
        selected, params = self._owned_selection(**selection)
        with self.engine.connect() as conn:
            rows = conn.execute(text(f"""
                SELECT id, role, content, timestamp
                FROM {self._messages_table_name}
                WHERE id IN ({selected})
                ORDER BY timestamp ASC, id ASC
            """), params).fetchall()
        return [
            {"id": row.id, "role": row.role, "content": row.content, "timestamp": row.timestamp}
            for row in rows
        ]

    def preview_recent_messages(
        self, count: int, *, user_id: str, session_id: str | None = None,
    ) -> list[dict]:
        """Preview the caller's last N messages, chronological."""
        return self._preview_owned(
            user_id=user_id, session_id=session_id, count=count, minutes=None)

    def preview_messages_since_minutes(
        self, minutes: int, *, user_id: str, session_id: str | None = None,
    ) -> list[dict]:
        """Preview the caller's messages from the last N minutes, chronological."""
        return self._preview_owned(
            user_id=user_id, session_id=session_id, count=None, minutes=minutes)

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

