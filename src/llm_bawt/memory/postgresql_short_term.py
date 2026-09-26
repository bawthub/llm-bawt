"""Session registry and short-term conversation history for PostgreSQL.

The session manager is intentionally separate from the permanent message and
semantic-memory backend. ``postgresql.py`` re-exports the class for compatibility.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..message_authorship import AuthorReference

logger = logging.getLogger(__name__)


class PostgreSQLShortTermManager:
    """Manages short-term (session) conversation history using PostgreSQL.
    
    This stores messages permanently
    but provides session-scoped access for building context windows.
    """
    
    def __init__(
        self,
        config: Any,
        bot_id: str = "nova",
        user_id: str | None = None,
        *,
        provision_schema: bool = True,
    ):
        self.config = config
        self.bot_id = bot_id
        # TASK-284: sessions are namespaced (bot_id, user_id). May be None for
        # legacy/back-compat callers that don't carry a user.
        self.user_id = user_id
        from .postgresql import PostgreSQLMemoryBackend

        self._backend = PostgreSQLMemoryBackend(
            config,
            bot_id,
            provision_schema=provision_schema,
        )
        # TASK-284: do NOT mint+insert a session row per construction. That was
        # the leak — these managers are cached/evicted per (model, bot, user),
        # so eager per-construction insertion accumulated hundreds of
        # never-closed "active" rows. The active thread is resolved lazily from
        # the sessions table (the DB-derived single source of truth) on first
        # use, so every instance/worker for a (bot, user) converges on one row.
        self._session_id_cache: str | None = None

    def ensure_schema(self) -> None:
        """Promote a read-only manager before a mutating operation."""
        self._backend.ensure_schema()

    @property
    def _current_session_id(self) -> str:
        """The live thread's session id, resolved lazily from the DB.

        Resolving on first access (rather than at construction) is what stops
        the leak: a manager that is built but never writes no longer creates a
        session row.
        """
        if self._session_id_cache is None:
            try:
                self._session_id_cache = self.get_or_create_active_session(
                    bot_id=self.bot_id, user_id=self.user_id
                )
            except Exception as e:
                logger.warning(
                    f"Failed to resolve active session for bot {self.bot_id}: {e}"
                )
                # Fall back to an ephemeral id so writes still get *some*
                # attribution; it just won't be registry-backed this turn.
                self._session_id_cache = str(uuid.uuid4())
        return self._session_id_cache

    @_current_session_id.setter
    def _current_session_id(self, value: str) -> None:
        self._session_id_cache = value

    def _insert_session_row(
        self,
        session_id: str,
        user_id: str | None = None,
        bot_id: str | None = None,
        session_metadata: dict | None = None,
    ) -> None:
        """Insert a new row into the shared sessions table.

        Uses INSERT ... ON CONFLICT DO NOTHING so reusing an existing UUID
        (rare, but possible in a backfill or test scenario) is a no-op.

        ``user_id`` defaults to this manager's ``self.user_id`` so callers that
        don't pass it still tag the session with the manager's user (TASK-284).
        """
        target_user = self.user_id if user_id is None else user_id
        target_bot = self.bot_id if bot_id is None else bot_id
        import json as _json
        with self._backend.engine.connect() as conn:
            stmt = text("""
                INSERT INTO sessions (id, bot_id, user_id, started_at, status, session_metadata)
                VALUES (:id, :bot_id, :user_id, CURRENT_TIMESTAMP, 'active', CAST(:meta AS JSONB))
                ON CONFLICT (id) DO NOTHING
            """)
            conn.execute(stmt, {
                "id": session_id,
                "bot_id": target_bot,
                "user_id": target_user,
                "meta": _json.dumps(session_metadata) if session_metadata else None,
            })
            conn.commit()

    def _close_session_row(self, session_id: str) -> None:
        """Mark a session row as archived (sets ended_at/archived_at + status)."""
        with self._backend.engine.connect() as conn:
            stmt = text("""
                UPDATE sessions
                SET ended_at = CURRENT_TIMESTAMP,
                    archived_at = CURRENT_TIMESTAMP,
                    status = 'archived'
                WHERE id = :id AND ended_at IS NULL
            """)
            conn.execute(stmt, {"id": session_id})
            conn.commit()

    @staticmethod
    def _row_to_session_dict(row: Any) -> dict:
        """Coerce a `sessions` row into a JSON-friendly dict."""
        # TASK-250: status vocabulary is active|archived|deleted. Legacy rows
        # written before the migration (or by a not-yet-restarted process) may
        # still carry 'completed' — normalize on read so no consumer ever sees
        # the retired value.
        status = "archived" if row.status == "completed" else row.status
        archived_at = getattr(row, "archived_at", None)

        def iso_epoch(value: float | None) -> str | None:
            if value is None:
                return None
            return datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat()

        return {
            "id": row.id,
            "bot_id": row.bot_id,
            "user_id": getattr(row, "user_id", None),
            "started_at": row.started_at.isoformat() if row.started_at else None,
            "ended_at": row.ended_at.isoformat() if row.ended_at else None,
            "archived_at": archived_at.isoformat() if archived_at else None,
            "status": status,
            # TASK-284 step 15 fix: key must match the column name — every
            # consumer (routes/sessions.py, the provider↔thread coordinator)
            # reads "session_metadata"; the old "metadata" key had zero readers
            # and made the /v1/sessions API silently return null metadata.
            "session_metadata": row.session_metadata,
            "message_count": int(getattr(row, "message_count", 0) or 0),
            "turn_count": int(getattr(row, "turn_count", 0) or 0),
            "first_message_at": iso_epoch(getattr(row, "first_message_at", None)),
            "last_activity_at": iso_epoch(getattr(row, "last_message_at", None)),
            "first_user_message": getattr(row, "first_user_message", None),
            "last_user_message": getattr(row, "last_user_message", None),
            "summary": getattr(row, "summary", None),
        }

    # ---------- Public session query API ----------
    # The `sessions` table is shared across bots, but the manager is bot-scoped,
    # so query methods default `bot_id` to `self.bot_id` for ergonomic in-process
    # calls while still accepting an override for cross-bot lookups.

    def close_session(self, session_id: str | None = None) -> bool:
        """Close a session: set `ended_at=now()` and `status='archived'`.

        If `session_id` is None, closes this manager's current session and
        keeps `_current_session_id` pointing at it (no rotation). Use
        `new_session()` if you also want to open a fresh session.

        Returns True if a row was updated, False otherwise (already closed
        or not found).
        """
        target = session_id or self._current_session_id
        if not target:
            return False
        try:
            with self._backend.engine.connect() as conn:
                stmt = text("""
                    UPDATE sessions
                    SET ended_at = CURRENT_TIMESTAMP,
                        archived_at = CURRENT_TIMESTAMP,
                        status = 'archived'
                    WHERE id = :id AND ended_at IS NULL
                    RETURNING id
                """)
                result = conn.execute(stmt, {"id": target}).fetchone()
                conn.commit()
                return result is not None
        except Exception as e:
            logger.warning(f"close_session({target}) failed: {e}")
            return False

    def get_session(self, session_id: str) -> dict | None:
        """Return a session row as a dict, or None if not found."""
        if not session_id:
            return None
        try:
            with self._backend.engine.connect() as conn:
                stmt = text("""
                    SELECT id, bot_id, user_id, started_at, ended_at, archived_at,
                           status, session_metadata
                    FROM sessions
                    WHERE id = :id
                """)
                row = conn.execute(stmt, {"id": session_id}).fetchone()
                return self._row_to_session_dict(row) if row else None
        except Exception as e:
            logger.warning(f"get_session({session_id}) failed: {e}")
            return None

    def list_sessions(
        self,
        bot_id: str | None = None,
        since: float | str | None = None,
        status: str | None = None,
        limit: int = 50,
        user_id: str | None = None,
        include_deleted: bool = False,
    ) -> list[dict]:
        """List sessions, newest first.

        Args:
            bot_id: Restrict to this bot. Defaults to ``self.bot_id``. Pass
                an empty string to query across all bots.
            since: Only sessions with ``started_at >= since``. Accepts a
                Unix timestamp (float/int) or an ISO-8601 string.
            status: Filter by status ('active' | 'archived' | 'deleted';
                legacy 'completed' is accepted as an alias for 'archived').
            limit: Max rows to return.
            user_id: Restrict to this user (TASK-284). Pass ``None`` to leave
                the user dimension unfiltered (cross-user for this bot); pass a
                value to scope to one user's threads.
            include_deleted: TASK-250 — soft-deleted sessions are excluded by
                default; pass True (or an explicit ``status='deleted'``
                filter) to include them.

        Returns:
            List of session dicts in ``started_at DESC`` order.
        """
        target_bot = self.bot_id if bot_id is None else bot_id
        clauses: list[str] = []
        params: dict = {"limit": int(limit)}

        if target_bot:
            clauses.append("bot_id = :bot_id")
            params["bot_id"] = target_bot
        if user_id is not None:
            clauses.append("user_id = :user_id")
            params["user_id"] = user_id
        if status:
            if status == "completed":  # retired alias (TASK-250)
                status = "archived"
            if status == "archived":
                # Match legacy rows a not-yet-restarted writer may still mint.
                clauses.append("status IN ('archived', 'completed')")
            else:
                clauses.append("status = :status")
                params["status"] = status
        elif not include_deleted:
            clauses.append("status <> 'deleted'")
        if since is not None:
            if isinstance(since, (int, float)):
                params["since"] = datetime.fromtimestamp(float(since), tz=timezone.utc)
            else:
                params["since"] = since
            clauses.append("started_at >= :since")

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        messages_table = self._backend._messages_table_name
        sql = f"""
            WITH message_stats AS (
                SELECT
                    session_id,
                    COUNT(*) FILTER (WHERE role IN ('user', 'assistant')) AS message_count,
                    COUNT(*) FILTER (WHERE role = 'user') AS turn_count,
                    MIN(timestamp) FILTER (WHERE role IN ('user', 'assistant')) AS first_message_at,
                    MAX(timestamp) FILTER (WHERE role IN ('user', 'assistant')) AS last_message_at,
                    (ARRAY_AGG(
                        LEFT(REGEXP_REPLACE(content, '[[:space:]]+', ' ', 'g'), 240)
                        ORDER BY timestamp ASC
                    ) FILTER (WHERE role = 'user'))[1] AS first_user_message,
                    (ARRAY_AGG(
                        LEFT(REGEXP_REPLACE(content, '[[:space:]]+', ' ', 'g'), 240)
                        ORDER BY timestamp DESC
                    ) FILTER (WHERE role = 'user'))[1] AS last_user_message,
                    (ARRAY_AGG(
                        LEFT(REGEXP_REPLACE(content, '[[:space:]]+', ' ', 'g'), 600)
                        ORDER BY timestamp DESC
                    ) FILTER (WHERE role = 'summary'))[1] AS summary
                FROM {messages_table}
                WHERE session_id IS NOT NULL
                GROUP BY session_id
            )
            SELECT
                s.id, s.bot_id, s.user_id, s.started_at, s.ended_at,
                s.archived_at, s.status, s.session_metadata,
                COALESCE(ms.message_count, 0) AS message_count,
                COALESCE(ms.turn_count, 0) AS turn_count,
                ms.first_message_at,
                ms.last_message_at,
                ms.first_user_message,
                ms.last_user_message,
                ms.summary
            FROM sessions AS s
            LEFT JOIN message_stats AS ms ON ms.session_id = s.id
            {where}
            ORDER BY
                CASE WHEN s.status = 'active' THEN 0 ELSE 1 END,
                COALESCE(ms.last_message_at, EXTRACT(EPOCH FROM s.started_at)) DESC
            LIMIT :limit
        """
        try:
            with self._backend.engine.connect() as conn:
                rows = conn.execute(text(sql), params).fetchall()
                return [self._row_to_session_dict(r) for r in rows]
        except Exception as e:
            logger.warning(f"list_sessions failed: {e}")
            return []

    def get_active_session(
        self, bot_id: str | None = None, user_id: str | None = None
    ) -> dict | None:
        """Return the most-recent active session for a (bot, user), or None.

        An "active" session has ``status='active'`` and ``ended_at IS NULL``.
        If multiple are active (shouldn't happen but possible after a
        crashed close), returns the most recently started.

        ``user_id`` (TASK-284): when provided, scopes to that user's threads
        (legacy rows with NULL user_id won't match — that's intentional; they
        are reconciled by the migration). When ``None``, the user dimension is
        left unfiltered (legacy bot-only behavior).
        """
        target_bot = self.bot_id if bot_id is None else bot_id
        if not target_bot:
            return None
        clauses = ["bot_id = :bot_id", "status = 'active'", "ended_at IS NULL"]
        params: dict = {"bot_id": target_bot}
        if user_id is not None:
            clauses.append("user_id = :user_id")
            params["user_id"] = user_id
        sql = f"""
            SELECT id, bot_id, user_id, started_at, ended_at, archived_at,
                   status, session_metadata
            FROM sessions
            WHERE {' AND '.join(clauses)}
            ORDER BY started_at DESC
            LIMIT 1
        """
        try:
            with self._backend.engine.connect() as conn:
                row = conn.execute(text(sql), params).fetchone()
                return self._row_to_session_dict(row) if row else None
        except Exception as e:
            logger.warning(f"get_active_session({target_bot}) failed: {e}")
            return None

    def get_or_create_active_session(
        self, bot_id: str | None = None, user_id: str | None = None
    ) -> str:
        """Return the live thread's session id for (bot, user), creating one if none.

        TASK-284: this is the single, DB-derived source of truth for "where does
        the active thread begin" — deliberately NOT held in instance memory. The
        legacy per-construction ``_current_session_id`` minted a fresh session on
        every cached-instance build (instances are keyed (model, bot, user) and
        evicted on profile/settings changes), which is how the sessions table
        accumulated hundreds of never-closed "active" rows. Resolving from the DB
        instead means every instance/worker for a (bot, user) converges on the
        same active thread.

        Returns the active session id; inserts a new active row if none exists.
        """
        target_bot = self.bot_id if bot_id is None else bot_id
        target_user = self.user_id if user_id is None else user_id
        existing = self.get_active_session(bot_id=target_bot, user_id=target_user)
        if existing:
            return existing["id"]
        new_id = str(uuid.uuid4())
        try:
            self._insert_session_row(new_id, user_id=target_user, bot_id=target_bot)
        except IntegrityError:
            # TASK-284: lost a cold-start race — a concurrent first-write created
            # the active row for this (bot,user) first, and the partial unique
            # index (one active row per (bot,user)) rejected our INSERT. Re-resolve
            # and use the winner rather than minting a duplicate active thread.
            # When the index is absent this branch simply never fires, so the
            # method is safe to land before the index is deployed.
            winner = self.get_active_session(bot_id=target_bot, user_id=target_user)
            if winner:
                logger.info(
                    "get_or_create_active_session lost race for (%s,%s); using winner %s",
                    target_bot, target_user, winner["id"],
                )
                return winner["id"]
            raise
        return new_id

    def rotate_session(
        self, bot_id: str | None = None, user_id: str | None = None
    ) -> str:
        """Atomically close the active thread for (bot,user) and open a new one.

        TASK-284: this is the primitive behind a non-destructive ``/new``. The
        old thread's rows are untouched (its session row flips to
        ``status='archived'``); a fresh ``active`` thread is created and its id
        returned. Close+open run in ONE transaction so a crash cannot leave zero
        or two active rows for the pair.
        """
        target_bot = self.bot_id if bot_id is None else bot_id
        target_user = self.user_id if user_id is None else user_id
        new_id = str(uuid.uuid4())
        from .session_reset_lineage import SessionResetLineage

        SessionResetLineage.rotate(
            self._backend.engine, bot_id=target_bot, user_id=target_user, new_id=new_id,
        )
        # Point this manager at the freshly opened thread.
        self._session_id_cache = new_id
        return new_id

    def create_inactive_session(
        self,
        bot_id: str | None = None,
        user_id: str | None = None,
        session_metadata: dict | None = None,
    ) -> str:
        """Insert a new session row that is born archived (TASK-702).

        Background dispatch uses this to mint a dedicated task thread
        WITHOUT rotating (closing) the user's currently active conversation.
        The one-active-per-(bot,user) invariant is untouched — the fresh
        row is created with ``status='archived'`` and ``ended_at``/
        ``archived_at`` set, so it never competes for the active slot but
        is fully owned by (bot,user), reachable via explicit ``session_id``
        continuation, and eligible for per-thread SDK resume metadata.

        Returns the new session id.
        """
        target_bot = self.bot_id if bot_id is None else bot_id
        target_user = self.user_id if user_id is None else user_id
        with self._backend.engine.begin() as conn:
            return self.create_inactive_session_row(
                conn, bot_id=target_bot, user_id=target_user,
                session_metadata=session_metadata,
            )

    @staticmethod
    def create_inactive_session_row(
        conn, *, bot_id: str, user_id: str | None,
        session_metadata: dict | None = None, session_id: str | None = None,
    ) -> str:
        """Born-archived primitive, composable with an outbox transaction.

        A caller-supplied deterministic identity is retry-safe, but may never
        take over another owner's thread or resurrect a deleted session.
        """
        new_id = session_id or str(uuid.uuid4())
        meta = json.dumps(session_metadata) if session_metadata else None
        # SQLite is used only by hermetic tests, not as a production backend.
        meta_expr = "CAST(:meta AS JSONB)" if conn.dialect.name == "postgresql" else ":meta"
        conn.execute(text(f"""
            INSERT INTO sessions
                (id, bot_id, user_id, started_at, ended_at,
                 archived_at, status, session_metadata)
            VALUES (:id, :bot_id, :user_id, CURRENT_TIMESTAMP,
                    CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, 'archived', {meta_expr})
            ON CONFLICT (id) DO NOTHING
        """), {"id": new_id, "bot_id": bot_id, "user_id": user_id, "meta": meta})
        row = conn.execute(text(
            "SELECT bot_id, user_id, status FROM sessions WHERE id=:id"
        ), {"id": new_id}).mappings().one()
        if row["bot_id"] != bot_id or row["user_id"] != user_id or row["status"] == "deleted":
            raise ValueError("Dedicated session identity is not an available owned thread")
        return new_id

    def activate_session(
        self,
        session_id: str,
        bot_id: str | None = None,
        user_id: str | None = None,
    ) -> bool:
        """Make an existing thread the active one for (bot,user).

        TASK-284: switch primitive for the thread API. Deactivates the current
        active thread(s) for the pair, then flips ``session_id`` back to
        ``active`` — in ONE transaction, others-first so the one-active-row
        invariant is never transiently violated. Returns ``False`` if the target
        doesn't exist or belongs to a different (bot,user).
        """
        target_bot = self.bot_id if bot_id is None else bot_id
        target_user = self.user_id if user_id is None else user_id
        with self._backend.engine.begin() as conn:
            owned = conn.execute(
                text("""
                    SELECT id FROM sessions
                    WHERE id = :id AND bot_id = :bot_id
                      AND (user_id = :user_id OR (:user_id IS NULL AND user_id IS NULL))
                """),
                {"id": session_id, "bot_id": target_bot, "user_id": target_user},
            ).fetchone()
            if owned is None:
                return False
            # Deactivate the current active thread(s) BEFORE activating the
            # target, so a partial unique index never sees two active rows.
            conn.execute(
                text("""
                    UPDATE sessions
                    SET ended_at = CURRENT_TIMESTAMP,
                        archived_at = CURRENT_TIMESTAMP,
                        status = 'archived'
                    WHERE bot_id = :bot_id AND status = 'active' AND ended_at IS NULL
                      AND id <> :id
                      AND (user_id = :user_id OR (:user_id IS NULL AND user_id IS NULL))
                """),
                {"id": session_id, "bot_id": target_bot, "user_id": target_user},
            )
            conn.execute(
                text("""
                    UPDATE sessions
                    SET status = 'active', ended_at = NULL, archived_at = NULL
                    WHERE id = :id
                """),
                {"id": session_id},
            )
        self._session_id_cache = session_id
        return True

    def set_session_status(self, session_id: str, status: str) -> bool:
        """Set a session's lifecycle status to 'archived' or 'deleted' (TASK-250).

        - ``archived``: sets ``archived_at``/``ended_at`` if not already set.
          Also the restore target for a soft-deleted thread.
        - ``deleted``: soft delete — messages are retained; the row is excluded
          from default listings and its deep-links return 410.

        Reactivation is NOT handled here — that is :meth:`activate_session`,
        which owns the one-active-per-(bot,user) invariant.

        Returns ``False`` for an unknown status, a missing row, or a DB error;
        never raises.
        """
        if status not in ("archived", "deleted"):
            logger.warning(f"set_session_status: invalid status {status!r}")
            return False
        try:
            with self._backend.engine.begin() as conn:
                res = conn.execute(
                    text("""
                        UPDATE sessions
                        SET status = :status,
                            ended_at = COALESCE(ended_at, CURRENT_TIMESTAMP),
                            archived_at = COALESCE(archived_at, CURRENT_TIMESTAMP)
                        WHERE id = :id
                    """),
                    {"id": session_id, "status": status},
                )
                return (res.rowcount or 0) > 0
        except Exception as e:
            logger.warning(f"set_session_status({session_id}, {status}) failed: {e}")
            return False

    def update_session_metadata(self, session_id: str, patch: dict) -> bool:
        """Merge ``patch`` into a session row's ``session_metadata`` (jsonb).

        Backs canonical per-thread SDK identity/model maps. Shallow jsonb
        merge (``||``): existing top-level keys not in ``patch`` are preserved.
        Returns ``False`` when the row doesn't
        exist or the update fails; never raises.
        """
        if not session_id or not patch:
            return False
        try:
            with self._backend.engine.begin() as conn:
                res = conn.execute(
                    text("""
                        UPDATE sessions
                        SET session_metadata =
                            COALESCE(session_metadata, '{}'::jsonb) || CAST(:patch AS jsonb)
                        WHERE id = :id
                    """),
                    {"id": session_id, "patch": json.dumps(patch)},
                )
                return (res.rowcount or 0) > 0
        except Exception as e:
            logger.warning(f"update_session_metadata({session_id}) failed: {e}")
            return False

    @property
    def session_id(self) -> str:
        """Get the current session ID."""
        return self._current_session_id

    def new_session(self) -> str:
        """Start a new session and return its ID.

        Closes the previous session (sets ended_at, status='archived'),
        then opens a new one with a fresh UUID. Both operations are
        best-effort: failures are logged but don't block returning the
        new ID since callers depend on it for message attribution.
        """
        previous_session_id = self._current_session_id
        self._current_session_id = str(uuid.uuid4())

        try:
            self._close_session_row(previous_session_id)
        except Exception as e:
            logger.warning(
                f"Failed to close session {previous_session_id}: {e}"
            )

        try:
            self._insert_session_row(self._current_session_id)
        except Exception as e:
            logger.warning(
                f"Failed to record new session {self._current_session_id}: {e}"
            )

        return self._current_session_id
    
    def add_message(
        self,
        role: str,
        content: str,
        timestamp: float | None = None,
        message_id: str | None = None,
        attachments: list[dict] | None = None,
        reasoning: str | None = None,
        session_id: str | None = None,
        author: AuthorReference | None = None,
        extract_memory: bool = True,
    ) -> str:
        """Add a message to the current session.

        Returns the message ID.

        ``message_id``, ``attachments`` and ``reasoning`` are passed through to
        the backend so this adapter matches the MCP adapter signature used by
        ``HistoryManager`` (TASK-222 / TASK-301 plumbing).

        ``session_id`` (TASK-251): explicit thread override for
        continue-old-thread turns — when set, the row lands on THAT thread
        instead of the active one. None = active thread (unchanged default).
        """
        mid = (str(message_id).strip() if message_id else "") or str(uuid.uuid4())
        ts = timestamp or datetime.now(timezone.utc).timestamp()

        # EPIC TASK-217 / TASK-357: the backend now RAISES on a failed INSERT
        # (it no longer swallows + returns None). We deliberately do NOT wrap
        # this in try/except: the exception propagates so `mid` is returned
        # ONLY when the row actually committed. Previously this returned `mid`
        # unconditionally, handing the caller a valid-looking id for a row that
        # never persisted (the silent-swallow / orphan-turn_logs bug).
        self._backend.add_message(
            message_id=mid,
            role=role,
            content=content,
            timestamp=ts,
            session_id=session_id or self._current_session_id,
            attachments=attachments,
            reasoning=reasoning,
            author=author,
            **({"extract_memory": False} if not extract_memory else {}),
        )

        return mid
    
    def load_session_scoped(
        self, since_minutes: int | None = None, session_id: str | None = None
    ) -> list | None:
        """Session-scoped history read (TASK-251) — direct-manager twin of the
        MCP adapter's ``load_session_scoped`` so embedded mode (no MCP server)
        supports explicit-thread context too.

        One thread's raw bubbles (role <> 'summary', not yet summarized)
        composed with the bot's rolling summary husks, chronological. When
        ``session_id`` is None resolves the active thread; returns None when
        no thread can be resolved (caller falls back to the continuous load).
        """
        from ..models.message import Message

        if not session_id:
            active = self.get_active_session(bot_id=self.bot_id, user_id=self.user_id)
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
                    WHERE role = 'summary'
                    ORDER BY timestamp ASC
                """),
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

        Returns:
            List of Message objects.
        """
        from ..models.message import Message

        with self._backend.engine.connect() as conn:
            if include_summaries:
                # Smart filtering: summaries + unsummarized messages
                sql = f"""
                    SELECT id, role, content, timestamp, session_id,
                           author_entity_type, author_entity_id
                    FROM {self._backend._messages_table_name}
                    WHERE
                        (role = 'summary')
                        OR
                        (role != 'summary' AND (summarized IS NULL OR summarized = FALSE))
                """
                params: dict = {}

                if since_minutes is not None:
                    cutoff = time.time() - since_minutes
                    sql += " AND (timestamp >= :cutoff OR role = 'summary')"
                    params["cutoff"] = cutoff

                sql += " ORDER BY timestamp ASC"
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

    def clear(self) -> bool:
        """Clear all messages for this bot."""
        return self._backend.clear()

    def remove_last_message_if_partial(self, role: str) -> bool:
        """Remove the last message if it matches the specified role.
        
        Used for cleanup on error/interrupt.
        """
        with Session(self._backend.engine) as session:
            # Find the last message
            stmt = (
                select(self._backend.messages_table)
                .order_by(self._backend.messages_table.c.timestamp.desc())
                .limit(1)
            )
            row = session.execute(stmt).fetchone()
            
            if row and row.role == role:
                delete_stmt = (
                    self._backend.messages_table.delete()
                    .where(self._backend.messages_table.c.id == row.id)
                )
                session.execute(delete_stmt)
                session.commit()
                return True
            return False

    def get_messages_for_summary(self, summary_id: str) -> list:
        """Get the original raw messages that a summary covers.

        Finds the summary by ID, extracts its timestamp range from
        summary_metadata, and returns all user/assistant messages in
        that range.

        Args:
            summary_id: Database ID of the summary row.

        Returns:
            List of Message objects, or empty list if not found.
        """
        from ..models.message import Message

        with self._backend.engine.connect() as conn:
            # Get the summary row
            sql = text(f"""
                SELECT id, timestamp, summary_metadata
                FROM {self._backend._messages_table_name}
                WHERE id = :id AND role = 'summary'
            """)
            row = conn.execute(sql, {"id": summary_id}).fetchone()
            if not row:
                return []

            # Determine the time range from metadata or estimate from surrounding summaries
            meta = row.summary_metadata or {}
            start_ts = meta.get("start_timestamp", row.timestamp - 7200)
            end_ts = meta.get("end_timestamp", row.timestamp)

            # Fetch raw messages in this time range
            msgs_sql = text(f"""
                SELECT id, role, content, timestamp
                FROM {self._backend._messages_table_name}
                WHERE role IN ('user', 'assistant')
                  AND timestamp >= :start AND timestamp <= :end
                ORDER BY timestamp ASC
            """)
            rows = conn.execute(msgs_sql, {"start": start_ts, "end": end_ts}).fetchall()
            return [
                Message(role=r.role, content=r.content, timestamp=r.timestamp, db_id=r.id)
                for r in rows
            ]

    def mark_messages_recalled(self, message_ids: list[str]) -> int:
        """Mark messages as recalled (re-inserted from summary expansion).

        Args:
            message_ids: List of database IDs to mark.

        Returns:
            Number of rows updated.
        """
        if not message_ids:
            return 0

        with self._backend.engine.connect() as conn:
            sql = text(f"""
                UPDATE {self._backend._messages_table_name}
                SET recalled_history = TRUE
                WHERE id = ANY(:ids)
            """)
            result = conn.execute(sql, {"ids": message_ids})
            conn.commit()
            return result.rowcount

