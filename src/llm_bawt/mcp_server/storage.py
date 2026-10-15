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
from llm_bawt.message_authorship import AuthorReference, normalize_author
from llm_bawt.utils.config import Config
from llm_bawt.utils.db import get_shared_engine

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


class MemoryStorage:
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
    # Admin / Maintenance Operations (used by service UX endpoints)
    # =========================================================================

    async def stats(self, bot_id: str = "default") -> dict[str, Any]:
        """Return memory/message statistics."""
        backend = self._get_backend(bot_id)
        return backend.stats()

    async def get_high_importance_memories(
        self,
        bot_id: str = "default",
        n_results: int = 20,
        min_importance: float = 0.0,
    ) -> list[dict[str, Any]]:
        backend = self._get_backend(bot_id)
        return backend.get_high_importance_memories(n_results=n_results, min_importance=min_importance)

    async def list_recent_memories(self, bot_id: str = "default", n: int = 10) -> list[dict[str, Any]]:
        backend = self._get_backend(bot_id)
        if hasattr(backend, "list_recent"):
            return backend.list_recent(n=n)
        return []

    async def preview_recent_messages(self, bot_id: str = "default", count: int = 10) -> list[dict[str, Any]]:
        backend = self._get_backend(bot_id)
        return backend.preview_recent_messages(count)

    async def preview_messages_since_minutes(self, bot_id: str = "default", minutes: int = 60) -> list[dict[str, Any]]:
        backend = self._get_backend(bot_id)
        return backend.preview_messages_since_minutes(minutes)

    async def preview_ignored_messages(self, bot_id: str = "default") -> list[dict[str, Any]]:
        backend = self._get_backend(bot_id)
        return backend.preview_ignored_messages()

    async def ignore_recent_messages(self, bot_id: str = "default", count: int = 10) -> int:
        backend = self._get_backend(bot_id, provision_schema=True)
        return backend.ignore_recent_messages(count)

    async def ignore_messages_since_minutes(self, bot_id: str = "default", minutes: int = 60) -> int:
        backend = self._get_backend(bot_id, provision_schema=True)
        return backend.ignore_messages_since_minutes(minutes)

    async def get_message_by_id(
        self, bot_id: str = "default", message_id: str = "",
        before: int = 0, after: int = 0,
    ) -> dict | None:
        backend = self._get_backend(bot_id)
        result = backend.get_message_by_id(message_id, before=before, after=after)
        if result is None:
            return None
        if "message" in result:
            combined = [*result.get("before", []), result["message"], *result.get("after", [])]
            hydrated = self.hydrate_message_authors(combined, bot_id=bot_id)
            offset = len(result.get("before", []))
            return {
                "before": hydrated[:offset],
                "message": hydrated[offset],
                "after": hydrated[offset + 1 :],
            }
        return self.hydrate_message_authors([result], bot_id=bot_id)[0]

    async def ignore_message_by_id(self, bot_id: str = "default", message_id: str = "") -> bool:
        backend = self._get_backend(bot_id, provision_schema=True)
        return backend.ignore_message_by_id(message_id)

    async def restore_ignored_messages(self, bot_id: str = "default") -> int:
        backend = self._get_backend(bot_id, provision_schema=True)
        return backend.restore_ignored_messages()

    async def get_messages_for_summary(self, bot_id: str = "default", summary_id: str = "") -> list[dict[str, Any]]:
        manager = self.get_short_term_manager(bot_id)
        rows = manager.get_messages_for_summary(summary_id)
        results: list[dict[str, Any]] = []
        for row in rows:
            results.append(
                {
                    "id": getattr(row, "db_id", None),
                    "role": getattr(row, "role", ""),
                    "content": getattr(row, "content", ""),
                    "timestamp": getattr(row, "timestamp", 0.0),
                }
            )
        return results

    async def mark_messages_recalled(self, bot_id: str = "default", message_ids: list[str] | None = None) -> int:
        manager = self.get_short_term_manager(bot_id, provision_schema=True)
        return manager.mark_messages_recalled(message_ids or [])

    async def delete_memories_by_source_message_ids(self, bot_id: str = "default", message_ids: list[str] | None = None) -> int:
        backend = self._get_backend(bot_id, provision_schema=True)
        return backend.delete_memories_by_source_message_ids(message_ids or [])

    async def regenerate_embeddings(self, bot_id: str = "default", batch_size: int = 50) -> dict[str, Any]:
        backend = self._get_backend(bot_id, provision_schema=True)
        return backend.regenerate_embeddings(batch_size=batch_size)

    async def consolidate_memories(
        self,
        bot_id: str = "default",
        dry_run: bool = True,
        similarity_threshold: float | None = None,
    ) -> dict[str, Any]:
        """Run consolidation using the existing backend implementation.

        Note: This may use a heuristic merge when no local LLM is available.
        """
        from llm_bawt.memory.consolidation import MemoryConsolidator, get_local_llm_client

        backend = self._get_backend(bot_id, provision_schema=True)
        llm_client = None if dry_run else get_local_llm_client(self.config)

        threshold = similarity_threshold or getattr(self.config, "MEMORY_CONSOLIDATION_THRESHOLD", 0.92)
        consolidator = MemoryConsolidator(
            backend=backend,
            llm_client=llm_client,
            similarity_threshold=threshold,
            config=self.config,
        )

        result = consolidator.consolidate(dry_run=dry_run)
        # Result is a dataclass-like object; make it JSON-friendly.
        return {
            "clusters_found": result.clusters_found,
            "clusters_merged": result.clusters_merged,
            "memories_consolidated": result.memories_consolidated,
            "new_memories_created": result.new_memories_created,
            "errors": result.errors,
            "dry_run": dry_run,
            "similarity_threshold": threshold,
        }

    async def update_memory_meaning(
        self,
        bot_id: str = "default",
        memory_id: str = "",
        intent: str | None = None,
        stakes: str | None = None,
        emotional_charge: float | None = None,
        recurrence_keywords: list[str] | None = None,
        updated_tags: list[str] | None = None,
    ) -> bool:
        """Update meaning metadata on an existing memory."""
        if not memory_id:
            return False
        from llm_bawt.memory.maintenance import update_memory_meaning

        backend = self._get_backend(bot_id, provision_schema=True)
        return bool(
            update_memory_meaning(
                backend=backend,
                memory_id=memory_id,
                intent=intent,
                stakes=stakes,
                emotional_charge=emotional_charge,
                recurrence_keywords=recurrence_keywords,
                updated_tags=updated_tags,
            )
        )

    async def run_maintenance(
        self,
        bot_id: str = "default",
        run_consolidation: bool = True,
        run_recurrence_detection: bool = True,
        run_decay_pruning: bool = False,
        run_orphan_cleanup: bool = False,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Run unified memory maintenance inside the memory service."""
        from llm_bawt.memory.maintenance import MemoryMaintenance

        backend = self._get_backend(bot_id, provision_schema=True)
        maint = MemoryMaintenance(
            backend=backend,
            llm_client=None,
            config=self.config,
        )
        result = maint.run(
            run_consolidation=run_consolidation,
            run_recurrence_detection=run_recurrence_detection,
            run_decay_pruning=run_decay_pruning,
            run_orphan_cleanup=run_orphan_cleanup,
            dry_run=dry_run,
        )
        return result.to_dict()

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
        """
        if summaries_only:
            return await asyncio.to_thread(
                self._get_summary_husks,
                bot_id,
                limit=limit,
            )
        if session_id is not None:
            return await asyncio.to_thread(
                self._get_messages_raw,
                bot_id,
                since_seconds=since_seconds,
                limit=limit,
                session_id=session_id,
                exclude_summarized=exclude_summarized,
            )
        if raw:
            return await asyncio.to_thread(
                self._get_messages_raw,
                bot_id,
                since_seconds=since_seconds,
                limit=limit,
            )

        manager = self.get_short_term_manager(bot_id)

        try:
            def load_and_hydrate() -> list[dict[str, Any]]:
                messages = manager.get_messages(since_minutes=since_seconds)
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
    ) -> list[dict[str, Any]]:
        """Direct-table read of the ``role='summary'`` continuity husks only.

        TASK-284 step 12: the session-scoped v2 read composes the active thread's
        raw bubbles with these rolling summaries so a `/new`'d thread still shows
        the gist of prior threads. Ordered oldest-first; ``limit`` keeps the most
        recent N (the assembler's ``summary_count`` still bounds what's injected).
        """
        from sqlalchemy import text

        backend = self._get_backend(bot_id)
        sql = f"""
            SELECT id, role, content, timestamp, session_id,
                   author_entity_type, author_entity_id
            FROM {backend._messages_table_name}
            WHERE role = 'summary'
            ORDER BY timestamp ASC
        """
        try:
            with backend.engine.connect() as conn:
                rows = conn.execute(text(sql)).fetchall()
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

    # =========================================================================
    # Sessions (shared `sessions` table)
    # =========================================================================

    async def close_session(
        self,
        session_id: str,
        bot_id: str = "default",
    ) -> bool:
        """Mark a session row archived (sets ended_at + status='archived').

        Returns True iff a row was updated. Idempotent: closing an already-
        closed session returns False without error.
        """
        manager = self.get_short_term_manager(bot_id, provision_schema=True)
        return manager.close_session(session_id)

    async def get_session(
        self,
        session_id: str,
        bot_id: str = "default",
    ) -> dict | None:
        """Return a session row by id (or None). The `sessions` table is
        shared, so `bot_id` only selects which engine to use — any bot
        works.
        """
        manager = self.get_short_term_manager(bot_id)
        return manager.get_session(session_id)

    async def list_sessions(
        self,
        bot_id: str = "default",
        since: float | str | None = None,
        status: str | None = None,
        limit: int = 50,
        user_id: str | None = None,
        include_deleted: bool = False,
    ) -> list[dict]:
        """List sessions for a bot, newest first.

        Pass `bot_id=""` to query across all bots. Pass ``user_id`` (TASK-284)
        to scope to one user's threads; ``None`` leaves the user dimension
        unfiltered. Soft-deleted sessions are excluded unless
        ``include_deleted=True`` or an explicit ``status='deleted'`` filter is
        given (TASK-250).
        """
        # Use a default manager engine, but pass the requested bot through
        # to the query (manager is bot-scoped, the table is not).
        engine_manager = self.get_short_term_manager(bot_id or "default")
        return engine_manager.list_sessions(
            bot_id=bot_id,
            since=since,
            status=status,
            limit=limit,
            user_id=user_id,
            include_deleted=include_deleted,
        )

    async def get_active_session(
        self,
        bot_id: str = "default",
        user_id: str | None = None,
    ) -> dict | None:
        """Return the most-recent active session for a (bot, user), or None.

        TASK-284: ``user_id`` scopes to that user's active thread; ``None``
        preserves the legacy bot-only lookup.
        """
        manager = self.get_short_term_manager(bot_id)
        return manager.get_active_session(bot_id=bot_id, user_id=user_id)

    async def rotate_session(
        self,
        bot_id: str = "default",
        user_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> str:
        """Close the active thread for (bot, user) and open a new one.

        TASK-284: the non-destructive ``/new`` primitive — old rows untouched,
        the previous session row flips to ``status='archived'``. Returns the
        new active session id. Close+open run in one transaction.
        """
        manager = self.get_short_term_manager(bot_id, provision_schema=True)
        return manager.rotate_session(
            bot_id=bot_id, user_id=user_id,
            **({"idempotency_key": idempotency_key} if idempotency_key else {}),
        )

    async def create_inactive_session(
        self,
        bot_id: str = "default",
        user_id: str | None = None,
        session_metadata: dict | None = None,
    ) -> str:
        """Mint a session row that is born archived (TASK-702).

        Background dispatch uses this to open a dedicated task thread
        without rotating (closing) the user's currently active
        conversation. The one-active-per-(bot,user) invariant is untouched.
        The returned session is fully owned and reachable via explicit
        ``session_id`` continuation.
        """
        manager = self.get_short_term_manager(bot_id, provision_schema=True)
        return manager.create_inactive_session(
            bot_id=bot_id,
            user_id=user_id,
            session_metadata=session_metadata,
        )

    async def activate_session(
        self,
        session_id: str,
        bot_id: str = "default",
        user_id: str | None = None,
    ) -> bool:
        """Make an existing thread the active one for (bot, user).

        TASK-284: the thread-switch primitive. Returns ``False`` if the target
        doesn't exist or belongs to a different (bot, user).
        """
        manager = self.get_short_term_manager(bot_id, provision_schema=True)
        return manager.activate_session(session_id, bot_id=bot_id, user_id=user_id)

    async def get_or_create_active_session(
        self,
        bot_id: str = "default",
        user_id: str | None = None,
    ) -> str:
        """Return the active thread id for (bot, user), creating one if none.

        TASK-284 step 15: used by the provider↔thread mirror so a freshly
        minted provider session always has a durable DB thread to map onto.
        """
        manager = self.get_short_term_manager(bot_id, provision_schema=True)
        return manager.get_or_create_active_session(bot_id=bot_id, user_id=user_id)

    async def set_session_status(
        self,
        session_id: str,
        status: str,
        bot_id: str = "default",
    ) -> bool:
        """Set a session's lifecycle status: 'archived' or 'deleted' (TASK-250).

        Soft-delete keeps the messages; the row just leaves default listings
        and its deep-links return 410. Reactivation goes through
        :meth:`activate_session`, never here.
        """
        manager = self.get_short_term_manager(bot_id, provision_schema=True)
        return manager.set_session_status(session_id, status)

    async def update_session_metadata(
        self,
        session_id: str,
        patch: dict,
        bot_id: str = "default",
    ) -> bool:
        """Shallow-merge ``patch`` into a session row's ``session_metadata``.

        Records canonical per-backend SDK key/model maps on the durable thread.
        """
        manager = self.get_short_term_manager(bot_id, provision_schema=True)
        return manager.update_session_metadata(session_id, patch)


    # =========================================================================
    # Cross-bot / Source Discovery & Search
    # =========================================================================

    def _list_partition_bots(self, parent: str) -> list[str]:
        """Enumerate bot ids that have a partition of ``parent`` (TASK-571).

        Replaces the legacy ``information_schema`` shard-table discovery.
        Partitions are named ``<parent>_p_<bot>`` by
        :func:`llm_bawt.memory.postgresql.partition_name`, and the partition
        bound value equals the suffix, so the name is authoritative. Bots
        with EMPTY partitions are included — matching the old behavior where
        an existing-but-empty shard table still surfaced as a source.
        """
        from sqlalchemy import text

        prefix = f"{parent}_p_"
        try:
            with self._require_engine().connect() as conn:
                rows = conn.execute(text("""
                    SELECT c.relname
                    FROM pg_inherits i
                    JOIN pg_class c ON c.oid = i.inhrelid
                    WHERE i.inhparent = CAST(:parent AS regclass)
                    ORDER BY c.relname
                """), {"parent": parent}).fetchall()
            return [
                relname.removeprefix(prefix)
                for (relname,) in rows
                if relname.startswith(prefix)
            ]
        except Exception as e:
            logger.error("Failed to list %s partitions: %s", parent, e)
            return []

    def _global_search_excluded_bot_ids(self) -> set[str]:
        """Return sanitized bot IDs opted out of aggregate search.

        Sanitized on purpose: the ``bot_id`` partition-key values are the
        sanitized bot identity (same identity the legacy shard-table
        prefixes carried), so exclusion compares like with like.
        """
        try:
            from llm_bawt.bots import BotManager
            from llm_bawt.memory.postgresql import _sanitize_table_name

            return {
                _sanitize_table_name(bot.slug)
                for bot in BotManager(self.config).list_bots()
                if getattr(bot, "include_in_global_search", True) is False
            }
        except Exception as e:
            logger.warning("Failed to load global search bot policy: %s", e)
            return set()

    def _global_search_bots(self, parent: str) -> list[str]:
        """Bot ids eligible for aggregate search over ``parent``."""
        bots = self._list_partition_bots(parent)
        if not bots:
            return []

        excluded = self._global_search_excluded_bot_ids()
        if not excluded:
            return bots

        return [bot_id for bot_id in bots if bot_id not in excluded]

    async def list_memory_sources(self) -> list[dict[str, Any]]:
        """Discover all available memory sources (bot_ids with memory partitions).

        One GROUP BY over the partitioned ``memories`` parent replaces the
        legacy per-shard COUNT loop (TASK-571). Bots with empty partitions
        still appear with ``memory_count`` 0.
        """
        from sqlalchemy import text

        from llm_bawt.memory.postgresql import MEMORIES_PARENT

        bots = self._global_search_bots(MEMORIES_PARENT)
        if not bots:
            return []

        try:
            with self._require_engine().connect() as conn:
                rows = conn.execute(text(
                    f"SELECT bot_id, COUNT(*) FROM {MEMORIES_PARENT} GROUP BY bot_id"
                )).fetchall()
            counts = {bot_id: int(n) for bot_id, n in rows}
            return [
                {"source": bot_id, "memory_count": counts.get(bot_id, 0)}
                for bot_id in bots
            ]
        except Exception as e:
            logger.error("Failed to list memory sources: %s", e)
            return []

    async def search_all_messages(
        self,
        query: str,
        n_results: int = 10,
        role_filter: str | None = None,
        sort_by: str = "relevance",
        since: float | None = None,
        until: float | None = None,
        bot_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Full-text search across ALL bots' message histories.

        One query against the partitioned ``messages`` parent (TASK-571) so a
        single question like *"who was working on the stop button?"* returns
        ranked results from all bots in one call. A ``bot_id`` filter prunes
        to that bot's partition; the unfiltered form scans all partitions.

        Parameters drive the Spotlight modal's facet controls:

        * ``sort_by`` — ``"relevance"`` (default; rank then timestamp) or
          ``"recent"`` (timestamp only, ignores rank). The Spotlight UI
          exposes both for cases where the most-relevant doc is two months
          old and the user actually wants the latest match.
        * ``since`` / ``until`` — Unix seconds. Inclusive bounds; either can
          be omitted. Used by the time-range chips (24h / 7d / 30d / all).
        * ``bot_id`` — restrict to a single bot's history.  Omit to search
          all bots.

        Each returned row also carries a ``total`` field (window COUNT(*)
        over the unbounded result set) so the UI can render "showing N of M
        matches" without a second roundtrip. ``total`` is identical on
        every row; callers can read it from the first row and ignore the
        rest. Empty result → no rows → caller should treat as ``total=0``.
        """
        from llm_bawt.mcp_server.message_search import CrossBotMessageSearcher

        searcher = CrossBotMessageSearcher(
            self._require_engine(),
            lambda rows, source: self.hydrate_message_authors(rows, bot_id=source),
        )
        try:
            return await asyncio.to_thread(
                searcher.search_fts,
                query,
                n_results=n_results,
                role_filter=role_filter,
                sort_by=sort_by,
                since=since,
                until=until,
                bot_id=bot_id,
                excluded_bot_ids=self._global_search_excluded_bot_ids(),
            )
        except Exception as e:
            logger.error("search_all_messages failed: %s", e)
            return []

    async def search_all_messages_trgm(
        self,
        query: str,
        n_results: int = 10,
        role_filter: str | None = None,
        sort_by: str = "relevance",
        since: float | None = None,
        until: float | None = None,
    ) -> list[dict[str, Any]]:
        """Cross-bot substring/fuzzy search via ``pg_trgm``.

        Companion to :meth:`search_all_messages` (which uses Postgres FTS via
        ``to_tsquery``). FTS is great for natural-language queries — it
        stems, drops stop-words, and ranks by lexeme density — but it
        actively breaks on identifiers: ``to_tsquery('english', 'TASK-241')``
        tokenizes to ``task`` (the digit half is dropped by our tokenizer
        and the english stemmer doesn't know what to do with ``241`` anyway),
        so a search for ``TASK-241`` returns every message that ever said
        the word "task." That's wrong for the case where the user is
        looking up an ID.

        This method goes the other way: treat ``content`` as a flat
        character sequence, find substring matches with ``ILIKE``, and rank
        by trigram similarity. Backed by a per-table
        ``USING gin (content gin_trgm_ops)`` index so the leading-wildcard
        ILIKE stays fast. No tokenizer, no stemming, no stop-words —
        ``TASK-241`` matches the literal substring or it doesn't.

        UI dispatches between the two via the ``mode`` param on the route:
        ``mode=fts`` (default) keeps prior behaviour; ``mode=trgm`` calls
        this method. The Spotlight modal exposes the choice as a toggle.
        """
        from llm_bawt.mcp_server.message_search import CrossBotMessageSearcher

        searcher = CrossBotMessageSearcher(
            self._require_engine(),
            lambda rows, source: self.hydrate_message_authors(rows, bot_id=source),
        )
        try:
            return await asyncio.to_thread(
                searcher.search_trgm,
                query,
                n_results=n_results,
                role_filter=role_filter,
                sort_by=sort_by,
                since=since,
                until=until,
                excluded_bot_ids=self._global_search_excluded_bot_ids(),
            )
        except Exception as e:
            logger.error("search_all_messages_trgm failed: %s", e)
            return []

    async def search_all_memories(
        self,
        query: str,
        n_results: int = 10,
        min_relevance: float = 0.0,
        since: float | None = None,
        until: float | None = None,
        bot_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Semantic search across ALL bots' memory stores.

        Generates one embedding, runs the existing per-bot search (with
        temporal decay, access boost, diversity sampling), then merges and
        ranks across all sources.

        * ``since`` / ``until`` — Unix seconds.  Post-filters results by
          ``created_at``.  Either can be omitted.
        * ``bot_id`` — restrict to a single bot's memory store.
        """
        from llm_bawt.memory.embeddings import generate_embedding

        query_embedding = generate_embedding(
            query, self.config.MEMORY_EMBEDDING_MODEL, verbose=self.config.VERBOSE,
        )
        if not query_embedding:
            return []

        from llm_bawt.memory.postgresql import MEMORIES_PARENT

        bots = self._global_search_bots(MEMORIES_PARENT)
        if not bots:
            return []

        # Filter to a single bot if requested.
        if bot_id:
            bots = [bid for bid in bots if bid == bot_id]
            if not bots:
                return []

        all_results: list[dict[str, Any]] = []
        for bot_id in bots:
            try:
                backend = self._get_backend(bot_id)
                results = backend.search_memories_by_embedding(
                    embedding=query_embedding,
                    n_results=n_results,
                    min_importance=0.0,
                )
                for r in results:
                    relevance = r.get("similarity") or r.get("relevance", 0.0)
                    if relevance < min_relevance:
                        continue
                    r["source"] = bot_id
                    r["relevance"] = relevance
                    all_results.append(r)
            except Exception as e:
                logger.warning("search_all_memories skipped %s: %s", bot_id, e)

        # Time-range post-filter on created_at (ISO string or Unix float).
        if since is not None or until is not None:
            def _to_epoch(val: Any) -> float | None:
                if val is None:
                    return None
                if isinstance(val, (int, float)):
                    return float(val)
                try:
                    from datetime import datetime, timezone
                    return datetime.fromisoformat(str(val)).replace(
                        tzinfo=timezone.utc,
                    ).timestamp()
                except Exception:
                    return None

            filtered: list[dict[str, Any]] = []
            for r in all_results:
                ts = _to_epoch(r.get("created_at"))
                if ts is None:
                    filtered.append(r)  # keep if no timestamp
                    continue
                if since is not None and ts < since:
                    continue
                if until is not None and ts > until:
                    continue
                filtered.append(r)
            all_results = filtered

        # Sort by effective_score (from the backend's scoring) then relevance
        all_results.sort(
            key=lambda r: r.get("effective_score", r.get("relevance", 0.0)),
            reverse=True,
        )
        return all_results[:n_results]


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
