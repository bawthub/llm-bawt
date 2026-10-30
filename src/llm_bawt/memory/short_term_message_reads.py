"""Read side of the PostgreSQL short-term history manager.

Split out of ``postgresql_short_term.py`` (monolith gate, TASK-1774) so the
message-read queries live in one place. ``PostgreSQLShortTermManager`` mixes
this in; the mixin relies on its ``_backend``, ``bot_id``, ``user_id``,
``_current_session_id`` and ``get_active_session``.
"""

from __future__ import annotations

import time

from sqlalchemy import select, text
from sqlalchemy.orm import Session


def history_owner_clause(params: dict, *, bot_id: str, user_id: str) -> str:
    """SQL predicate limiting a ``messages_p_<bot>`` read to one user (TASK-1774).

    A bot's partition holds every user's conversations; only the durable
    ``sessions`` row records whose thread a message belongs to. Context reads
    for a request must carry this predicate, or one user's history (raw turns
    and rolling summaries) is served to another. Adds its bind parameters to
    ``params`` and returns the clause.
    """
    params["owner_bot"] = bot_id
    params["owner_user"] = user_id
    return (
        "session_id IN (SELECT id FROM sessions"
        " WHERE bot_id = :owner_bot AND user_id = :owner_user)"
    )


class ShortTermMessageReadsMixin:
    """Message reads for ``PostgreSQLShortTermManager``."""

    def load_session_scoped(
        self,
        since_minutes: int | None = None,
        session_id: str | None = None,
        *,
        user_id: str | None = None,
    ) -> list | None:
        """Session-scoped history read (TASK-251) — direct-manager twin of the
        MCP adapter's ``load_session_scoped`` so embedded mode (no MCP server)
        supports explicit-thread context too.

        One thread's raw bubbles (role <> 'summary', not yet summarized)
        composed with the bot's rolling summary husks, chronological. When
        ``session_id`` is None resolves the active thread; returns None when
        no thread can be resolved (caller falls back to the continuous load).

        ``user_id`` (TASK-1774, default: this manager's user) scopes both the
        thread and the summaries to that user's sessions, so a session id
        belonging to someone else yields no rows.
        """
        from ..models.message import Message

        owner = user_id if user_id is not None else self.user_id
        if not session_id:
            active = self.get_active_session(bot_id=self.bot_id, user_id=owner)
            if not active or not active.get("id"):
                return None
            session_id = active["id"]

        params: dict = {"session_id": session_id}
        raw_clauses = [
            "role <> 'summary'",
            "session_id = :session_id",
            "(summarized IS NULL OR summarized = FALSE)",
        ]
        if since_minutes is not None and since_minutes >= 0:
            # NOTE: legacy naming — value is seconds (matches get_messages).
            params["cutoff"] = time.time() - since_minutes
            raw_clauses.append("timestamp >= :cutoff")
        summary_params: dict = {}
        summary_clauses = ["role = 'summary'"]
        if owner is not None:
            raw_clauses.append(
                history_owner_clause(params, bot_id=self.bot_id, user_id=owner)
            )
            summary_clauses.append(
                history_owner_clause(summary_params, bot_id=self.bot_id, user_id=owner)
            )

        with self._backend.engine.connect() as conn:
            raw_rows = conn.execute(
                text(f"""
                    SELECT id, role, content, timestamp, session_id,
                           author_entity_type, author_entity_id
                    FROM {self._backend._messages_table_name}
                    WHERE {' AND '.join(raw_clauses)}
                    ORDER BY timestamp ASC
                """),
                params,
            ).fetchall()
            summary_rows = conn.execute(
                text(f"""
                    SELECT id, role, content, timestamp, session_id,
                           author_entity_type, author_entity_id
                    FROM {self._backend._messages_table_name}
                    WHERE {' AND '.join(summary_clauses)}
                    ORDER BY timestamp ASC
                """),
                summary_params,
            ).fetchall()

        merged = [
            Message(
                role=r.role,
                content=r.content,
                timestamp=r.timestamp or 0.0,
                db_id=r.id,
                session_id=r.session_id,
                author_entity_type=r.author_entity_type,
                author_entity_id=r.author_entity_id,
            )
            for r in (list(raw_rows) + list(summary_rows))
        ]
        merged.sort(key=lambda m: m.timestamp or 0.0)
        return merged

    def get_session_history(self, limit: int = 50) -> list[dict]:
        """Get messages from the current session."""
        with Session(self._backend.engine) as session:
            stmt = (
                select(self._backend.messages_table)
                .where(self._backend.messages_table.c.session_id == self._current_session_id)
                .order_by(self._backend.messages_table.c.timestamp.desc())
                .limit(limit)
            )
            rows = session.execute(stmt).fetchall()
            
            # Return in chronological order
            return [
                {
                    "id": row.id,
                    "role": row.role,
                    "content": row.content,
                    "timestamp": row.timestamp,
                }
                for row in reversed(rows)
            ]
    
    def get_recent_messages(self, limit: int = 20) -> list[dict]:
        """Get recent messages regardless of session (for context building)."""
        with Session(self._backend.engine) as session:
            stmt = (
                select(self._backend.messages_table)
                .order_by(self._backend.messages_table.c.timestamp.desc())
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
                for row in reversed(rows)
            ]
    
    def count(self) -> int:
        """Count messages in current session."""
        with Session(self._backend.engine) as session:
            from sqlalchemy import func
            stmt = (
                select(func.count())
                .select_from(self._backend.messages_table)
                .where(self._backend.messages_table.c.session_id == self._current_session_id)
            )
            return session.execute(stmt).scalar() or 0

    def get_messages(
        self,
        since_minutes: int | None = None,
        include_summaries: bool = True,
        *,
        user_id: str | None = None,
    ) -> list:
        """Get messages with smart summary filtering.

        When include_summaries is True (default), returns:
        - All summary rows (role='summary')
        - All unsummarized user/assistant/system messages
        - Excludes messages with summarized=TRUE (their content is in a summary)

        This prevents double-loading: raw messages that have been summarized
        won't appear alongside their summaries.

        Args:
            since_minutes: If provided, only return messages from the last N seconds.
                          When None, returns all messages (token budget handles windowing).
            include_summaries: If True, include summary rows and exclude summarized messages.
            user_id: TASK-1774 — when given, only that user's sessions (raw
                rows and summaries). ``None`` reads the whole bot partition;
                request context assembly never passes ``None``
                (``HistoryManager.load_history`` fails closed first).

        Returns:
            List of Message objects.
        """
        from ..models.message import Message

        params: dict = {}
        owner = (
            history_owner_clause(params, bot_id=self.bot_id, user_id=user_id)
            if user_id is not None
            else None
        )
        with self._backend.engine.connect() as conn:
            if include_summaries:
                # Smart filtering: summaries + unsummarized messages
                clauses = [
                    "((role = 'summary') OR"
                    " (role != 'summary' AND (summarized IS NULL OR summarized = FALSE)))"
                ]
                if since_minutes is not None:
                    params["cutoff"] = time.time() - since_minutes
                    clauses.append("(timestamp >= :cutoff OR role = 'summary')")
                if owner is not None:
                    clauses.append(owner)
                sql = f"""
                    SELECT id, role, content, timestamp, session_id,
                           author_entity_type, author_entity_id
                    FROM {self._backend._messages_table_name}
                    WHERE {' AND '.join(clauses)}
                    ORDER BY timestamp ASC
                """
                rows = conn.execute(text(sql), params).fetchall()

                return [
                    Message(
                        role=row.role,
                        content=row.content,
                        timestamp=row.timestamp,
                        db_id=row.id,
                        session_id=row.session_id,
                        author_entity_type=row.author_entity_type,
                        author_entity_id=row.author_entity_id,
                    )
                    for row in rows
                ]
            else:
                # Raw messages only (no summary filtering)
                stmt = (
                    select(self._backend.messages_table)
                    .order_by(self._backend.messages_table.c.timestamp.asc())
                )

                if since_minutes is not None:
                    cutoff = time.time() - since_minutes
                    stmt = stmt.where(self._backend.messages_table.c.timestamp >= cutoff)
                if owner is not None:
                    stmt = stmt.where(text(owner).bindparams(
                        owner_bot=params["owner_bot"], owner_user=params["owner_user"],
                    ))

                with Session(self._backend.engine) as session:
                    rows = session.execute(stmt).fetchall()

                    return [
                        Message(
                            role=row.role,
                            content=row.content,
                            timestamp=row.timestamp,
                            db_id=row.id,
                            session_id=row.session_id,
                            author_entity_type=row.author_entity_type,
                            author_entity_id=row.author_entity_id,
                        )
                        for row in rows
                    ]
