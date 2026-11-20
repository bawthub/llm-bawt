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

    def load_fork_scoped(
        self, *, session_id: str, parent_session_id: str, at_message_id: str,
        user_id: str | None = None,
    ) -> list | None:
        """A forked thread's pool (TASK-259): see ``memory.fork_history``.

        Both threads must belong to ``user_id``; returns None when the parent
        or the fork message can't be found.
        """
        from ..models.message import Message
        from .fork_history import compose_fork_chain, forked_from_of

        owner = user_id if user_id is not None else self.user_id
        if not (session_id and parent_session_id and at_message_id and owner):
            return None

        def _read(clauses: list[str], params: dict) -> list:
            params = dict(params)
            clauses = clauses + [
                history_owner_clause(params, bot_id=self.bot_id, user_id=owner)
            ]
            with self._backend.engine.connect() as conn:
                rows = conn.execute(
                    text(f"""
                        SELECT id, role, content, timestamp, session_id,
                               author_entity_type, author_entity_id
                        FROM {self._backend._messages_table_name}
                        WHERE {' AND '.join(clauses)}
                        ORDER BY timestamp ASC
                    """),
                    params,
                ).fetchall()
            return [
                Message(
                    role=r.role, content=r.content, timestamp=r.timestamp or 0.0,
                    db_id=r.id, session_id=r.session_id,
                    author_entity_type=r.author_entity_type,
                    author_entity_id=r.author_entity_id,
                )
                for r in rows
            ]

        def _thread(sid: str) -> list:
            return _read(["role <> 'summary'", "session_id = :sid"], {"sid": sid})

        def _fork_of(sid: str) -> dict | None:
            return forked_from_of(
                self.get_session(sid, bot_id=self.bot_id, user_id=owner)
            )

        return compose_fork_chain(
            _thread(session_id), _read(["role = 'summary'"], {}),
            session_id=session_id,
            fork_from={"session_id": parent_session_id, "at_message_id": at_message_id},
            read_rows=_thread, fork_of=_fork_of,
        )

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
    def get_context_window(
        self,
        *,
        token_budget: int,
        summary_limit: int,
        user_id: str | None = None,
        min_rows: int = 4,
        page_size: int = 50,
    ) -> list:
        """Newest slice of ``get_messages()`` that a budgeted seed can use (TASK-2395).

        Returns the same rows ``get_messages()`` would, restricted to:
        - the newest ``summary_limit`` summaries (the assembler keeps at most
          ``summary_count`` of them, newest last), and
        - unsummarized non-system rows newest-first until they exceed
          ``token_budget`` by the assembler's own estimate and the newest user
          row has been read. ``token_budget`` is the raw cap the assembler
          applies (``history_tokens`` bounded by the physical budget). The row
          that crosses it is included, and the newest complete turn is always
          read, so the newest-turn floor and degrade decisions see exactly what
          they would see in the full list.
        System rows are not read for the seed, which drops them (agents carry
        their own system prompt). At least ``min_rows`` rows are returned when
        they exist, topping up with system rows only for a tiny history, so
        cold-start priming (``len(messages) <= 3`` after the new prompt is
        appended) sees the same side of its threshold as the full read.

        Order and ties match ``get_messages()``: ``timestamp, id`` ascending.
        """
        from ..models.message import Message
        from ..utils.token_estimate import estimate_message_tokens

        params: dict = {}
        owner = (
            history_owner_clause(params, bot_id=self.bot_id, user_id=user_id)
            if user_id is not None
            else None
        )
        owner_sql = f" AND {owner}" if owner else ""
        cols = (
            "id, role, content, timestamp, session_id,"
            " author_entity_type, author_entity_id"
        )
        table = self._backend._messages_table_name
        rows: list = []
        # One REPEATABLE READ snapshot for the summary query and every raw
        # page, like the single-statement full read: a summary committed in
        # between can't drop both itself and the rows it replaces, and an
        # insert or delete can't shift later OFFSET pages.
        with self._backend.engine.connect().execution_options(
            isolation_level="REPEATABLE READ",
        ) as conn, conn.begin():
            if summary_limit > 0:
                rows.extend(conn.execute(
                    text(f"""
                        SELECT {cols} FROM {table}
                        WHERE role = 'summary'{owner_sql}
                        ORDER BY timestamp DESC, id DESC
                        LIMIT :summary_limit
                    """),
                    {**params, "summary_limit": summary_limit},
                ).fetchall())
            used = 0
            raw_count = 0
            seen_user = False
            exhausted = False
            offset = 0
            while True:
                page = conn.execute(
                    text(f"""
                        SELECT {cols} FROM {table}
                        WHERE role NOT IN ('summary', 'system')
                          AND (summarized IS NULL OR summarized = FALSE){owner_sql}
                        ORDER BY timestamp DESC, id DESC
                        LIMIT :page_size OFFSET :offset
                    """),
                    {**params, "page_size": page_size, "offset": offset},
                ).fetchall()
                for row in page:
                    rows.append(row)
                    raw_count += 1
                    used += estimate_message_tokens(row.content or "")
                    seen_user = seen_user or row.role == "user"
                    if used > token_budget and raw_count >= min_rows and seen_user:
                        break
                else:
                    if len(page) == page_size:
                        offset += page_size
                        continue
                    exhausted = True
                break
            # Cold-start priming counts every history row (``<= 3`` after the
            # new prompt is appended), system rows included. When the whole
            # unsummarized history fit and is still under ``min_rows``, add
            # enough system rows to keep that count on the same side of the
            # threshold as the full read. Seed delivery drops them anyway.
            if exhausted and len(rows) < min_rows:
                rows.extend(conn.execute(
                    text(f"""
                        SELECT {cols} FROM {table}
                        WHERE role = 'system'
                          AND (summarized IS NULL OR summarized = FALSE){owner_sql}
                        ORDER BY timestamp DESC, id DESC
                        LIMIT :need
                    """),
                    {**params, "need": min_rows - len(rows)},
                ).fetchall())

        def order(row):
            # Mirrors ORDER BY timestamp ASC, id ASC (NULL timestamps last).
            return (row.timestamp is None, row.timestamp or 0.0, row.id or "")

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
            for row in sorted(rows, key=order)
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
                    ORDER BY timestamp ASC, id ASC
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
