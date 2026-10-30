"""Storage wrapper for llm-bawt MCP server.

Thin async wrapper around the existing PostgreSQLMemoryBackend that:
1. Provides async-compatible interface for FastMCP tools
2. Handles connection lifecycle
3. Exposes simplified CRUD operations

The actual implementation delegates to the battle-tested postgresql.py backend.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, TYPE_CHECKING

from llm_bawt.memory.postgresql import PostgreSQLMemoryBackend
from llm_bawt.memory.embeddings import generate_embedding
from llm_bawt.memory.short_term_message_reads import history_owner_clause
from llm_bawt.message_authorship import AuthorReference, normalize_author
from llm_bawt.utils.config import Config
from llm_bawt.utils.db import get_shared_engine

from .storage_admin import StorageAdminMixin
from .storage_search import StorageSearchMixin
from .storage_sessions import StorageSessionsMixin

if TYPE_CHECKING:
    from llm_bawt.memory.postgresql import PostgreSQLShortTermManager

logger = logging.getLogger(__name__)


@dataclass
class Memory:
    """A stored memory/fact."""
    id: str
    content: str
    tags: list[str] = field(default_factory=lambda: ["misc"])
    importance: float = 0.5
    source_message_ids: list[str] = field(default_factory=list)
    access_count: int = 0
    created_at: datetime | None = None
    last_accessed: datetime | None = None
    relevance: float | None = None  # Set during search results
    intent: str | None = None
    stakes: str | None = None
    emotional_charge: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "tags": self.tags,
            "importance": self.importance,
            "source_message_ids": self.source_message_ids,
            "access_count": self.access_count,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
            "relevance": self.relevance,
            "intent": self.intent,
            "stakes": self.stakes,
            "emotional_charge": self.emotional_charge,
        }


@dataclass
class Message:
    """A conversation message."""
    id: str
    role: str
    content: str
    timestamp: float
    session_id: str | None = None
    author: AuthorReference | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "author": self.author.to_dict() if self.author else None,
        }


class MemoryStorage(StorageAdminMixin, StorageSearchMixin, StorageSessionsMixin):
    """Async-friendly wrapper around PostgreSQLMemoryBackend.
    
    Provides the storage layer for MCP memory tools.
    Each bot_id gets isolated tables automatically.
    """

    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self.engine = get_shared_engine(self.config)
        self._backends: dict[str, PostgreSQLMemoryBackend] = {}
        self._short_term_managers: dict[str, "PostgreSQLShortTermManager"] = {}
        self._cache_lock = threading.Lock()

    def _require_engine(self):
        if self.engine is None:
            raise RuntimeError("MemoryStorage requires Postgres credentials")
        return self.engine

    def _get_backend(
        self,
        bot_id: str,
        *,
        provision_schema: bool = False,
    ) -> PostgreSQLMemoryBackend:
        """Get a backend, provisioning schema only for an explicit mutation."""
        cached = self._backends.get(bot_id)
        if cached is not None:
            if provision_schema:
                cached.ensure_schema()
            return cached
        with self._cache_lock:
            cached = self._backends.get(bot_id)
            if cached is not None:
                if provision_schema:
                    cached.ensure_schema()
                return cached
            self._backends[bot_id] = PostgreSQLMemoryBackend(
                config=self.config,
                bot_id=bot_id,
                provision_schema=provision_schema,
            )
            return self._backends[bot_id]

    def get_backend(
        self,
        bot_id: str,
        *,
        provision_schema: bool = False,
    ) -> PostgreSQLMemoryBackend:
        """Public access to get or create a backend for the given bot.
        
        Used by MemoryClient for raw backend access.
        """
        return self._get_backend(bot_id, provision_schema=provision_schema)

    def hydrate_message_authors(
        self,
        messages: list[dict[str, Any]],
        *,
        bot_id: str,
    ) -> list[dict[str, Any]]:
        """Attach canonical, deterministic-legacy, or unknown author envelopes."""
        from llm_bawt.message_authorship import LegacyAuthorResolver

        engine = self._require_engine()
        authors = LegacyAuthorResolver(engine).resolve(messages, bot_id=bot_id)
        references = [authors[str(message.get("id") or "")] for message in messages]
        from llm_bawt.entity_presentation import EntityPresentationResolver

        presentation = EntityPresentationResolver(engine).resolve_many_safe(
            references
        )
        return [
            {
                **message,
                "author": presentation[(author.entity_type, author.entity_id)],
            }
            for message, author in zip(messages, references, strict=True)
        ]

    def get_short_term_manager(
        self,
        bot_id: str,
        *,
        provision_schema: bool = False,
    ) -> "PostgreSQLShortTermManager":
        """Get a short-term memory manager for conversation history.
        
        The short-term manager handles session messages for HistoryManager.
        """
        cached = self._short_term_managers.get(bot_id)
        if cached is not None:
            if provision_schema:
                cached.ensure_schema()
            return cached
        with self._cache_lock:
            cached = self._short_term_managers.get(bot_id)
            if cached is not None:
                if provision_schema:
                    cached.ensure_schema()
                return cached
            from llm_bawt.memory.postgresql import PostgreSQLShortTermManager
            self._short_term_managers[bot_id] = PostgreSQLShortTermManager(
                config=self.config,
                bot_id=bot_id,
                provision_schema=provision_schema,
            )
            return self._short_term_managers[bot_id]

    # =========================================================================
    # Memory Operations
    # =========================================================================

    async def store_memory(
        self,
        content: str,
        bot_id: str = "default",
        tags: list[str] | None = None,
        importance: float = 0.5,
        source_message_ids: list[str] | None = None,
    ) -> Memory:
        """Store a new memory with auto-generated embedding."""
        memory_id = str(uuid.uuid4())
        tags = tags or ["misc"]
        
        backend = self._get_backend(bot_id, provision_schema=True)
        
        # Generate embedding
        embedding = generate_embedding(content, self.config.MEMORY_EMBEDDING_MODEL, verbose=self.config.VERBOSE)
        
        # Store via backend (handles dedup, indexing, etc.)
        backend.add_memory(
            memory_id=memory_id,
            content=content,
            tags=tags,
            importance=importance,
            source_message_ids=source_message_ids or [],
            embedding=embedding,
        )
        
        return Memory(
            id=memory_id,
            content=content,
            tags=tags,
            importance=importance,
            source_message_ids=source_message_ids or [],
            created_at=datetime.now(timezone.utc),
        )

    async def search_memories(
        self,
        query: str,
        bot_id: str = "default",
        n_results: int = 10,
        min_relevance: float = 0.0,
        tags: list[str] | None = None,
    ) -> list[Memory]:
        """Search memories using semantic similarity."""
        backend = self._get_backend(bot_id)
        
        # Generate query embedding
        query_embedding = generate_embedding(query, self.config.MEMORY_EMBEDDING_MODEL, verbose=self.config.VERBOSE)
        
        if query_embedding:
            results = backend.search_memories_by_embedding(
                embedding=query_embedding,
                n_results=n_results,
                min_importance=min_relevance,
                tags=tags,
            )
        else:
            # Fallback to text search
            results = backend.search_memories_by_text(
                query=query,
                n_results=n_results,
                min_importance=min_relevance,
                tags=tags,
            )
        
        memories = []
        for r in results:
            # Filter by min_relevance (similarity score)
            relevance = r.get("similarity") or r.get("relevance", 0.0)
            if relevance < min_relevance:
                continue
            
            memories.append(Memory(
                id=r["id"],
                content=r["content"],
                tags=r.get("tags", ["misc"]),
                importance=r.get("importance", 0.5),
                source_message_ids=r.get("source_message_ids", []),
                access_count=r.get("access_count", 0),
                created_at=r.get("created_at"),
                last_accessed=r.get("last_accessed"),
                relevance=relevance,
                intent=r.get("intent"),
                stakes=r.get("stakes"),
                emotional_charge=r.get("emotional_charge"),
            ))
        
        return memories

    async def get_memory(self, memory_id: str, bot_id: str = "default") -> Memory | None:
        """Get a specific memory by ID."""
        backend = self._get_backend(bot_id)
        
        # Use the backend's method if available, otherwise use raw SQL
        result = None
        if hasattr(backend, 'get_memory_by_id'):
            result = backend.get_memory_by_id(memory_id)  # type: ignore
        else:
            # Fallback: raw SQL lookup
            from sqlalchemy import text
            try:
                with backend.engine.connect() as conn:
                    sql = text(f"""
                        SELECT id, content, tags, importance, source_message_ids
                        FROM {backend._memories_table_name}
                        WHERE id = :id
                    """)
                    row = conn.execute(sql, {"id": memory_id}).fetchone()
                    if row:
                        result = {
                            "id": row[0],
                            "content": row[1],
                            "tags": row[2] if row[2] else ["misc"],
                            "importance": row[3] if row[3] else 0.5,
                            "source_message_ids": row[4] if row[4] else [],
                        }
            except Exception as e:
                logger.error(f"Failed to get memory {memory_id}: {e}")
        
        if result:
            return Memory(
                id=result["id"],
                content=result["content"],
                tags=result.get("tags", ["misc"]),
                importance=result.get("importance", 0.5),
                source_message_ids=result.get("source_message_ids", []),
            )
        return None

    async def update_memory(
        self,
        memory_id: str,
        bot_id: str = "default",
        content: str | None = None,
        tags: list[str] | None = None,
        importance: float | None = None,
    ) -> Memory | None:
        """Update an existing memory."""
        backend = self._get_backend(bot_id, provision_schema=True)
        
        # Get current memory
        current = await self.get_memory(memory_id, bot_id)
        if not current:
            return None
        
        # Update with new values
        new_content = content if content is not None else current.content
        new_tags = tags if tags is not None else current.tags
        new_importance = importance if importance is not None else current.importance
        
        # Re-generate embedding if content changed
        embedding = None
        if content is not None:
            embedding = generate_embedding(new_content, self.config.MEMORY_EMBEDDING_MODEL, verbose=self.config.VERBOSE)
        
        backend.add_memory(
            memory_id=memory_id,
            content=new_content,
            tags=new_tags,
            importance=new_importance,
            embedding=embedding,
        )
        
        return Memory(
            id=memory_id,
            content=new_content,
            tags=new_tags,
            importance=new_importance,
        )

    async def delete_memory(
        self,
        memory_id: str,
        bot_id: str = "default",
    ) -> bool:
        """Delete a memory."""
        backend = self._get_backend(bot_id, provision_schema=True)
        
        try:
            if hasattr(backend, 'delete_memory'):
                return backend.delete_memory(memory_id)
            else:
                # Fallback: use raw SQL
                from sqlalchemy import text
                with backend.engine.connect() as conn:
                    conn.execute(text(f"DELETE FROM {backend._memories_table_name} WHERE id = :id"), {"id": memory_id})
                    conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to delete memory {memory_id}: {e}")
            return False

    # =========================================================================
    # Message Operations
    # =========================================================================

    async def add_message(
        self,
        role: str,
        content: str,
        bot_id: str = "default",
        session_id: str | None = None,
        timestamp: float | None = None,
        message_id: str | None = None,
        attachments: list[dict] | None = None,
        reasoning: str | None = None,
        user_id: str | None = None,
        author_entity_type: str | None = None,
        author_entity_id: str | None = None,
    ) -> Message:
        """Add a message to conversation history.

        ``message_id`` allows the client to supply a stable UUID (e.g. the
        frontend-generated user-message UUID) so downstream joins on
        ``trigger_message_id`` work without remapping.

        ``attachments`` (TASK-222) is the tiny JSONB ref list persisted on the
        message row's ``attachments`` column. ``None`` leaves the column at
        its default ``[]``.

        ``session_id`` / ``user_id`` (TASK-284): this is the single live write
        chokepoint for both embedded and server mode, so it owns thread
        attribution. When the caller doesn't supply ``session_id`` (the normal
        case), the active thread for ``(bot_id, user_id)`` is resolved from the
        DB and stamped on the row. Resolution is best-effort: a failure logs and
        leaves ``session_id`` NULL rather than dropping the message.
        """
        provided = (str(message_id).strip() if message_id else "") or None
        message_id_final = provided or str(uuid.uuid4())
        ts = timestamp if timestamp is not None else time.time()
        author = normalize_author(author_entity_type, author_entity_id)

        # TASK-284: stamp the live thread's session_id on every insert. Resolve
        # from the sessions table (DB-derived source of truth) so all workers /
        # cached instances converge on the same active thread per (bot, user).
        if session_id is None:
            try:
                manager = self.get_short_term_manager(bot_id, provision_schema=True)
                session_id = manager.get_or_create_active_session(
                    bot_id=bot_id, user_id=user_id
                )
            except Exception as e:
                logger.warning(
                    "TASK-284 session resolution failed for bot=%s user=%s: %s",
                    bot_id, user_id, e,
                )

        backend = self._get_backend(bot_id, provision_schema=True)
        backend.add_message(
            message_id=message_id_final,
            role=role,
            content=content,
            timestamp=ts,
            session_id=session_id,
            attachments=attachments,
            reasoning=reasoning,
            author=author,
        )

        # TASK-256: cheap auto-title from the first user message on a thread
        # still carrying the default title. Cosmetic + best-effort — the
        # helper never raises, so it can't break the write path.
        if role == "user" and session_id:
            from ..sessions.auto_title import maybe_auto_title

            maybe_auto_title(
                self.get_short_term_manager(bot_id), session_id, content
            )

        return Message(
            id=message_id_final,
            role=role,
            content=content,
            timestamp=ts,
            session_id=session_id,
            author=author,
        )

    async def get_recent_messages(
        self,
        bot_id: str = "default",
        max_messages: int = 20,
        max_age_seconds: int = 3600,
    ) -> list[Message]:
        """Get recent conversation messages."""
        backend = self._get_backend(bot_id)
        
        # Get recent messages
        cutoff = time.time() - max_age_seconds
        
        try:
            from sqlalchemy import text
            with backend.engine.connect() as conn:
                sql = text(f"""
                    SELECT id, role, content, timestamp, session_id
                    FROM {backend._messages_table_name}
                    WHERE timestamp >= :cutoff
                    ORDER BY timestamp DESC
                    LIMIT :limit
                """)
                rows = conn.execute(sql, {"cutoff": cutoff, "limit": max_messages}).fetchall()
                
                # Reverse to chronological order
                return [
                    Message(
                        id=row.id,
                        role=row.role,
                        content=row.content,
                        timestamp=row.timestamp,
                        session_id=row.session_id,
                    )
                    for row in reversed(rows)
                ]
        except Exception as e:
            logger.error(f"Failed to get recent messages: {e}")
            return []

    async def get_unprocessed_messages(
        self,
        bot_id: str = "default",
        limit: int = 100,
    ) -> list[Message]:
        """Get messages that haven't been processed for memory extraction."""
        backend = self._get_backend(bot_id)
        results = backend.get_unprocessed_messages(limit=limit)
        
        return [
            Message(
                id=r["id"],
                role=r["role"],
                content=r["content"],
                timestamp=r["timestamp"],
                session_id=r.get("session_id"),
                author=normalize_author(
                    r.get("author_entity_type"), r.get("author_entity_id")),
            )
            for r in results
        ]

    async def mark_messages_processed(
        self,
        message_ids: list[str],
        bot_id: str = "default",
    ) -> None:
        """Mark messages as processed for memory extraction."""
        backend = self._get_backend(bot_id, provision_schema=True)
        backend.mark_messages_processed(message_ids)

    # =========================================================================
    # History / Short-term Message Operations (DB-backed, exposed via MCP tools)
    # =========================================================================

    async def get_messages(
        self,
        bot_id: str = "default",
        since_seconds: int | None = None,
        limit: int | None = None,
        raw: bool = False,
        session_id: str | None = None,
        summaries_only: bool = False,
        exclude_summarized: bool = False,
        user_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get messages for building a context window.

        Default path (``raw=False``) uses the short-term manager's
        summary-aware retrieval, so older session summaries stand in for raw
        history that has aged out of the recent window. This is the chatbot
        short-term/long-term memory model — DO NOT change its behaviour.

        When ``raw=True`` the summary-aware path is bypassed and the messages
        table is queried directly: every real bubble is returned regardless of
        its ``summarized`` flag, and ``role='summary'`` husks are excluded.
        Callers like self_recap / self_tail want the actual transcript the user
        sees in the app, not a summary of a summary.

        TASK-284: when ``session_id`` is given, the read is scoped to that one
        durable thread's transcript via the direct-table (raw) path — this is
        the primitive the thread API and session-scoped context assembly build
        on. It implies ``raw`` semantics (a specific thread's real bubbles), so
        the summary-aware window path is bypassed. When ``session_id`` is None,
        behaviour is exactly as before.

        TASK-284 step 12: ``summaries_only`` returns ONLY the ``role='summary''``
        continuity husks (across all of the bot's sessions), newest-last. The
        session-scoped v2 read composes this with the active thread's raw bubbles
        (``session_id=...``) to reproduce the legacy summary-aware view for a
        single selected thread. Cheap: it never pulls cross-session raw rows.

        TASK-1774: ``user_id`` limits every mode to that user's sessions. A
        bot's partition holds all of its users' conversations, so context
        assembly for a request must pass it; ``None`` keeps the bot-wide read.
        """
        if summaries_only:
            return await asyncio.to_thread(
                self._get_summary_husks,
                bot_id,
                limit=limit,
                user_id=user_id,
            )
        if session_id is not None:
            return await asyncio.to_thread(
                self._get_messages_raw,
                bot_id,
                since_seconds=since_seconds,
                limit=limit,
                session_id=session_id,
                exclude_summarized=exclude_summarized,
                user_id=user_id,
            )
        if raw:
            return await asyncio.to_thread(
                self._get_messages_raw,
                bot_id,
                since_seconds=since_seconds,
                limit=limit,
                user_id=user_id,
            )

        manager = self.get_short_term_manager(bot_id)

        try:
            def load_and_hydrate() -> list[dict[str, Any]]:
                messages = manager.get_messages(
                    since_minutes=since_seconds, user_id=user_id,
                )
                if limit is not None and limit >= 0:
                    messages = messages[-limit:]

                rows = [
                    {
                        "id": msg.db_id,
                        "role": msg.role,
                        "content": msg.content,
                        "timestamp": msg.timestamp,
                        # Preserve durable thread and canonical author columns.
                        "session_id": getattr(msg, "session_id", None),
                        "author_entity_type": getattr(msg, "author_entity_type", None),
                        "author_entity_id": getattr(msg, "author_entity_id", None),
                    }
                    for msg in messages
                ]
                return self.hydrate_message_authors(rows, bot_id=bot_id)

            return await asyncio.to_thread(load_and_hydrate)
        except Exception as e:
            logger.error("Failed to get messages: %s", e)
            return []

    def _get_messages_raw(
        self,
        bot_id: str = "default",
        since_seconds: int | None = None,
        limit: int | None = None,
        session_id: str | None = None,
        exclude_summarized: bool = False,
        user_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Direct-table read of real message bubbles.

        Bypasses the summary-aware manager entirely. Returns every row whose
        role is NOT ``summary`` (so prior recaps/session summaries are excluded)
        and — by default — ignores the ``summarized`` flag (so bubbles already
        rolled into a summary are still returned; self_recap/self_tail want the
        literal transcript). Ordered oldest-first; ``limit`` keeps the most
        recent N.

        TASK-284: when ``session_id`` is given, restrict to that one durable
        thread's rows. ``exclude_summarized`` (step 12) additionally drops rows
        whose content is already captured in a ``role='summary'`` husk — the
        session-scoped context read sets it so summarized bubbles are NEVER
        double-loaded alongside their summary (the raw/summary partition stays
        disjoint, exactly as the legacy summary-aware path).
        """
        from sqlalchemy import text

        backend = self._get_backend(bot_id)
        clauses = ["role <> 'summary'"]
        params: dict[str, Any] = {}
        if session_id is not None:
            clauses.append("session_id = :session_id")
            params["session_id"] = session_id
        if exclude_summarized:
            clauses.append("(summarized IS NULL OR summarized = FALSE)")
        if since_seconds is not None and since_seconds >= 0:
            params["cutoff"] = time.time() - since_seconds
            clauses.append("timestamp >= :cutoff")
        if user_id is not None:  # TASK-1774: only this user's sessions
            clauses.append(history_owner_clause(params, bot_id=bot_id, user_id=user_id))
        where = " AND ".join(clauses)
        sql = f"""
            SELECT id, role, content, timestamp, session_id,
                   author_entity_type, author_entity_id
            FROM {backend._messages_table_name}
            WHERE {where}
            ORDER BY timestamp ASC
        """
        try:
            with backend.engine.connect() as conn:
                rows = conn.execute(text(sql), params).fetchall()
            out = [
                {
                    "id": r.id,
                    "role": r.role,
                    "content": r.content,
                    "timestamp": r.timestamp,
                    "session_id": r.session_id,
                    "author_entity_type": r.author_entity_type,
                    "author_entity_id": r.author_entity_id,
                }
                for r in rows
            ]
            if limit is not None and limit >= 0:
                out = out[-limit:]
            return self.hydrate_message_authors(out, bot_id=bot_id)
        except Exception as e:
            logger.error("Failed to get raw messages: %s", e)
            return []

    def _get_summary_husks(
        self,
        bot_id: str = "default",
        limit: int | None = None,
        user_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Direct-table read of the ``role='summary'`` continuity husks only.

        TASK-284 step 12: the session-scoped v2 read composes the active thread's
        raw bubbles with these rolling summaries so a `/new`'d thread still shows
        the gist of prior threads. Ordered oldest-first; ``limit`` keeps the most
        recent N (the assembler's ``summary_count`` still bounds what's injected).
        """
        from sqlalchemy import text

        backend = self._get_backend(bot_id)
        clauses = ["role = 'summary'"]
        params: dict[str, Any] = {}
        if user_id is not None:  # TASK-1774: only this user's summaries
            clauses.append(history_owner_clause(params, bot_id=bot_id, user_id=user_id))
        sql = f"""
            SELECT id, role, content, timestamp, session_id,
                   author_entity_type, author_entity_id
            FROM {backend._messages_table_name}
            WHERE {" AND ".join(clauses)}
            ORDER BY timestamp ASC
        """
        try:
            with backend.engine.connect() as conn:
                rows = conn.execute(text(sql), params).fetchall()
            out = [
                {
                    "id": r.id,
                    "role": r.role,
                    "content": r.content,
                    "timestamp": r.timestamp,
                    "session_id": r.session_id,
                    "author_entity_type": r.author_entity_type,
                    "author_entity_id": r.author_entity_id,
                }
                for r in rows
            ]
            if limit is not None and limit >= 0:
                out = out[-limit:]
            return self.hydrate_message_authors(out, bot_id=bot_id)
        except Exception as e:
            logger.error("Failed to get summary husks: %s", e)
            return []

    async def clear_messages(self, bot_id: str = "default") -> int:
        """Delete all messages for a bot."""
        backend = self._get_backend(bot_id, provision_schema=True)
        from sqlalchemy import text

        try:
            with backend.engine.connect() as conn:
                res = conn.execute(text(f"DELETE FROM {backend._messages_table_name}"))
                conn.commit()
                return int(getattr(res, "rowcount", 0) or 0)
        except Exception as e:
            logger.error("Failed to clear messages: %s", e)
            raise

    async def clear_memories(self, bot_id: str = "default") -> bool:
        """Delete all distilled memories for a bot through its cached backend."""
        return bool(self._get_backend(bot_id, provision_schema=True).clear())

    async def remove_last_message_if_partial(self, bot_id: str = "default", role: str = "assistant") -> bool:
        """Remove the most recent message if it matches role."""
        backend = self._get_backend(bot_id, provision_schema=True)
        from sqlalchemy import text

        try:
            with backend.engine.connect() as conn:
                row = conn.execute(
                    text(
                        f"""
                        SELECT id, role
                        FROM {backend._messages_table_name}
                        ORDER BY timestamp DESC
                        LIMIT 1
                        """
                    )
                ).fetchone()
                if not row or row.role != role:
                    return False
                conn.execute(
                    text(f"DELETE FROM {backend._messages_table_name} WHERE id = :id"),
                    {"id": row.id},
                )
                conn.commit()
                return True
        except Exception as e:
            logger.error("Failed to remove last message: %s", e)
            return False

    async def store_recap_summary(
        self,
        bot_id: str,
        content: str,
        window_start: float,
        window_end: float,
        message_ids: list[str],
        model: str,
    ) -> str | None:
        """Persist a self-recap as a ``role='summary'`` row.

        Unlike the compaction summarizer (``HistorySummarizer.summarize_session``)
        this deliberately does NOT mark the source messages ``summarized=TRUE``.
        A recap is a handoff artifact layered on top of history, not a
        compaction that should evict raw messages from the live context window.
        Mirrors the proven INSERT in ``summarization.py`` (json.dumps into the
        ``summary_metadata`` JSON column, no ``::jsonb`` cast needed).
        """
        import json

        from sqlalchemy import text

        from llm_bawt.memory.summarization import resolve_source_session_ids

        backend = self._get_backend(bot_id, provision_schema=True)
        summary_id = str(uuid.uuid4())
        metadata = {
            "summary_type": "self_recap",
            "summarization_method": "self-recap",
            "model": model,
            "session_start": window_start,
            "session_end": window_end,
            "message_ids": message_ids,
            "message_count": len(message_ids),
            "created_at": time.time(),
        }
        try:
            with backend.engine.connect() as conn:
                # TASK-284 step 13: stamp summary provenance (source threads).
                source_session_ids = resolve_source_session_ids(
                    conn, backend._messages_table_name, message_ids
                )
                metadata["source_session_ids"] = source_session_ids
                conn.execute(
                    text(
                        f"""
                        INSERT INTO {backend._messages_table_name}
                        (id, role, content, timestamp, summary_metadata, session_id, created_at)
                        VALUES (:id, 'summary', :content, :ts, :meta, :session_id, CURRENT_TIMESTAMP)
                        """
                    ),
                    {
                        "id": summary_id,
                        "session_id": source_session_ids[0] if len(source_session_ids) == 1 else None,
                        "content": content,
                        "ts": window_end,
                        "meta": json.dumps(metadata),
                    },
                )
                conn.commit()
            logger.info(
                "Stored self-recap summary %s for bot=%s (%d source messages)",
                summary_id[:8], bot_id, len(message_ids),
            )
            return summary_id
        except Exception as e:
            logger.error("Failed to store recap summary: %s", e)
            return None



# Singleton instance
_storage: MemoryStorage | None = None
_storage_lock = threading.Lock()


def get_storage() -> MemoryStorage:
    """Get the singleton storage instance."""
    global _storage
    cached = _storage
    if cached is not None:
        return cached
    with _storage_lock:
        if _storage is None:
            _storage = MemoryStorage()
        return _storage
