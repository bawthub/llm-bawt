"""MCP Memory Client - unified interface to memory operations.

This client provides a unified interface for memory operations that can work in two modes:
1. Embedded mode (default): Direct calls to storage layer (in-process, no IPC overhead)
2. Server mode: JSON-RPC calls to a running llm-bawt MCP server

The embedded mode is preferred for single-process usage as it avoids IPC overhead
while maintaining the same API surface.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from llm_bawt.utils.config import Config

if TYPE_CHECKING:
    from llm_bawt.mcp_server.storage import MemoryStorage

logger = logging.getLogger(__name__)


def _run_async(coro):
    """Run an async coroutine from sync context."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    
    if loop and loop.is_running():
        # We're in an async context, create a new thread to run
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
    else:
        return asyncio.run(coro)


@dataclass
class MemoryResult:
    """Result from a memory operation."""
    id: str
    content: str
    tags: list[str] = field(default_factory=list)
    importance: float = 0.5
    relevance: float = 0.0  # Only set for search results
    created_at: str | None = None
    last_accessed: str | None = None
    source_message_ids: list[str] = field(default_factory=list)
    intent: str | None = None
    stakes: str | None = None
    emotional_charge: float | None = None

    @classmethod
    def from_dict(cls, data: dict) -> "MemoryResult":
        return cls(
            id=data.get("id", ""),
            content=data.get("content", ""),
            tags=data.get("tags", []),
            importance=data.get("importance", 0.5),
            relevance=data.get("relevance", 0.0),
            created_at=data.get("created_at"),
            last_accessed=data.get("last_accessed"),
            source_message_ids=data.get("source_message_ids", []),
            intent=data.get("intent"),
            stakes=data.get("stakes"),
            emotional_charge=data.get("emotional_charge"),
        )


@dataclass
class MessageResult:
    """Result from a message operation."""
    id: str
    role: str
    content: str
    bot_id: str
    session_id: str | None = None
    created_at: str | None = None
    
    @classmethod
    def from_dict(cls, data: dict) -> "MessageResult":
        return cls(
            id=data.get("id", ""),
            role=data.get("role", ""),
            content=data.get("content", ""),
            bot_id=data.get("bot_id", ""),
            session_id=data.get("session_id"),
            created_at=data.get("created_at"),
        )


class MemoryClient:
    """Unified memory client supporting embedded and server modes.
    
    Usage:
        # Embedded mode (default, in-process)
        client = MemoryClient(config, bot_id="nova")
        
        # Server mode (connects to running MCP server)
        client = MemoryClient(config, bot_id="nova", server_url="http://localhost:8001")
    """
    
    def __init__(
        self,
        config: Config,
        bot_id: str = "",  # Required - must be passed explicitly
        user_id: str | None = None,
        server_url: str | None = None,
    ):
        """Initialize memory client.
        
        Args:
            config: Application config.
            bot_id: Bot namespace for memory isolation (required).
            user_id: User ID for profile attribute extraction (optional; required for extract_facts).
            server_url: If provided, use server mode; otherwise use embedded mode.
        """
        if not bot_id:
            raise ValueError("bot_id is required for MemoryClient")
        self.config = config
        self.bot_id = bot_id
        self.user_id = user_id
        self.server_url = server_url
        self._storage: MemoryStorage | None = None
        self._initialized = False
        
    def _ensure_initialized(self) -> None:
        """Lazy initialization of storage layer."""
        if self._initialized:
            return
            
        if self.server_url:
            # Server mode - will use HTTP/JSON-RPC
            logger.debug(f"Memory client using server mode: {self.server_url}")
            self._initialized = True
        else:
            # Embedded mode - direct storage access
            from llm_bawt.mcp_server.storage import get_storage
            self._storage = get_storage()
            logger.debug("Memory client using embedded mode")
            self._initialized = True
    
    def _get_storage(self) -> "MemoryStorage":
        """Get storage with type narrowing."""
        self._ensure_initialized()
        if self._storage is None:
            raise RuntimeError("Storage not initialized - are you in server mode?")
        return self._storage

    def get_short_term_manager(self) -> Any:
        """Get the short-term memory manager for conversation history.
        
        Used by HistoryManager for session message persistence.
        Returns a PostgreSQLShortTermManager in embedded mode.
        """
        if self.server_url:
            return _MCPShortTermManager(self)
        return self._get_storage().get_short_term_manager(self.bot_id)
    
    # -------------------------------------------------------------------------
    # Memory Operations
    # -------------------------------------------------------------------------
    
    def store_memory(
        self,
        content: str,
        tags: list[str] | None = None,
        importance: float = 0.5,
        source_message_ids: list[str] | None = None,
    ) -> MemoryResult:
        """Store a new memory.
        
        Args:
            content: The memory content.
            tags: Optional list of tags.
            importance: Importance score (0.0-1.0).
            source_message_ids: IDs of messages this memory was extracted from.
            
        Returns:
            MemoryResult with the stored memory.
        """
        self._ensure_initialized()
        
        if self.server_url:
            result = self._call_server("memory_store", {
                "content": content,
                "bot_id": self.bot_id,
                "tags": tags or ["misc"],
                "importance": importance,
                "source_message_ids": source_message_ids or [],
            })
            return MemoryResult.from_dict(result)
        
        # Embedded mode - call storage directly
        storage = self._get_storage()
        result = _run_async(
            storage.store_memory(
                content=content,
                bot_id=self.bot_id,
                tags=tags or ["misc"],
                importance=importance,
                source_message_ids=source_message_ids or [],
            )
        )
        return MemoryResult.from_dict(result.to_dict())
    
    def search(
        self,
        query: str,
        n_results: int | None = None,
        min_relevance: float | None = None,
    ) -> list[MemoryResult]:
        """Search memories by semantic similarity.
        
        Args:
            query: Search query text.
            n_results: Maximum results to return.
            min_relevance: Minimum relevance threshold.
            
        Returns:
            List of MemoryResult sorted by relevance.
        """
        self._ensure_initialized()
        
        n_results = n_results or self.config.MEMORY_N_RESULTS
        min_relevance = min_relevance or self.config.MEMORY_MIN_RELEVANCE
        
        if self.server_url:
            results = self._call_server("memory_search", {
                "query": query,
                "bot_id": self.bot_id,
                "n_results": n_results,
                "min_relevance": min_relevance,
            })
            return [MemoryResult.from_dict(r) for r in results]
        
        # Embedded mode
        storage = self._get_storage()
        results = _run_async(
            storage.search_memories(
                query=query,
                bot_id=self.bot_id,
                n_results=n_results,
                min_relevance=min_relevance,
            )
        )
        return [MemoryResult.from_dict(r.to_dict()) for r in results]
    
    def get_recent_context(
        self,
        n_messages: int = 10,
        n_memories: int = 5,
        query: str | None = None,
    ) -> dict[str, Any]:
        """Get recent conversation context including messages and memories.
        
        Args:
            n_messages: Number of recent messages.
            n_memories: Number of relevant memories.
            query: Optional query for memory relevance.
            
        Returns:
            Dict with 'messages' and 'memories' keys.
        """
        self._ensure_initialized()
        
        if self.server_url:
            return self._call_server("context_get_recent", {
                "bot_id": self.bot_id,
                "n_messages": n_messages,
                "n_memories": n_memories,
                "query": query,
            })
        
        # Embedded mode - compose from messages + memories
        storage = self._get_storage()
        
        # Get recent messages
        messages = _run_async(
            storage.get_recent_messages(
                bot_id=self.bot_id,
                max_messages=n_messages,
            )
        )
        
        # Get relevant memories
        memories: list = []
        if n_memories > 0:
            search_query = query or (messages[-1].content if messages else "")
            if search_query:
                memories = _run_async(
                    storage.search_memories(
                        query=search_query,
                        bot_id=self.bot_id,
                        n_results=n_memories,
                    )
                )
        
        return {
            "messages": [m.to_dict() for m in messages],
            "memories": [m.to_dict() for m in memories],
        }

    # -------------------------------------------------------------------------
    # Admin / UX Operations (mirrors llm-service /v1/memory endpoints)
    # -------------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        self._ensure_initialized()
        if self.server_url:
            return self._call_server("system_stats", {"bot_id": self.bot_id})
        storage = self._get_storage()
        backend = storage.get_backend(self.bot_id)
        return backend.stats()

    def list_memories(self, limit: int = 20, min_importance: float = 0.0) -> list[dict[str, Any]]:
        self._ensure_initialized()
        if self.server_url:
            return self._call_server(
                "list_memories",
                {"bot_id": self.bot_id, "limit": limit, "min_importance": min_importance},
            )
        storage = self._get_storage()
        backend = storage.get_backend(self.bot_id)
        return backend.get_high_importance_memories(n_results=limit, min_importance=min_importance)

    def preview_recent_messages(self, count: int = 10) -> list[dict[str, Any]]:
        self._ensure_initialized()
        if self.server_url:
            return self._call_server("messages_preview_recent", {"bot_id": self.bot_id, "count": count})
        storage = self._get_storage()
        backend = storage.get_backend(self.bot_id)
        return backend.preview_recent_messages(count)

    def preview_messages_since_minutes(self, minutes: int) -> list[dict[str, Any]]:
        self._ensure_initialized()
        if self.server_url:
            return self._call_server(
                "preview_messages_since_minutes",
                {"bot_id": self.bot_id, "minutes": minutes},
            )
        storage = self._get_storage()
        backend = storage.get_backend(self.bot_id)
        return backend.preview_messages_since_minutes(minutes)

    def preview_ignored_messages(self) -> list[dict[str, Any]]:
        self._ensure_initialized()
        if self.server_url:
            return self._call_server("messages_preview_ignored", {"bot_id": self.bot_id})
        storage = self._get_storage()
        backend = storage.get_backend(self.bot_id)
        return backend.preview_ignored_messages()

    def forget_recent_messages(self, count: int) -> dict[str, int]:
        """Forget last N messages and delete related memories (best-effort)."""
        preview = self.preview_recent_messages(count)
        message_ids = [m.get("id") for m in preview if m.get("id")]

        memories_deleted = 0
        if message_ids:
            memories_deleted = self.delete_memories_by_source_message_ids(message_ids)

        ignored = self.ignore_recent_messages(count)
        return {"messages_ignored": ignored, "memories_deleted": memories_deleted}

    def forget_messages_since_minutes(self, minutes: int) -> dict[str, int]:
        preview = self.preview_messages_since_minutes(minutes)
        message_ids = [m.get("id") for m in preview if m.get("id")]

        memories_deleted = 0
        if message_ids:
            memories_deleted = self.delete_memories_by_source_message_ids(message_ids)

        ignored = self.ignore_messages_since_minutes(minutes)
        return {"messages_ignored": ignored, "memories_deleted": memories_deleted}

    def ignore_recent_messages(self, count: int) -> int:
        self._ensure_initialized()
        if self.server_url:
            return int(self._call_server("messages_ignore_recent", {"bot_id": self.bot_id, "count": count}))
        storage = self._get_storage()
        backend = storage.get_backend(self.bot_id, provision_schema=True)
        return int(backend.ignore_recent_messages(count))

    def ignore_messages_since_minutes(self, minutes: int) -> int:
        self._ensure_initialized()
        if self.server_url:
            return int(
                self._call_server("messages_ignore_since_minutes", {"bot_id": self.bot_id, "minutes": minutes})
            )
        storage = self._get_storage()
        backend = storage.get_backend(self.bot_id, provision_schema=True)
        return int(backend.ignore_messages_since_minutes(minutes))

    def get_message_by_id(self, message_id: str) -> dict | None:
        """Get a specific message by ID (supports prefix match)."""
        self._ensure_initialized()
        if self.server_url:
            result = self._call_server("messages_get_by_id", {"bot_id": self.bot_id, "message_id": message_id})
            return result if result else None
        storage = self._get_storage()
        backend = storage.get_backend(self.bot_id)
        return backend.get_message_by_id(message_id)

    def ignore_message_by_id(self, message_id: str) -> bool:
        """Move a specific message to the forgotten table by ID."""
        self._ensure_initialized()
        if self.server_url:
            return bool(
                self._call_server("messages_ignore_by_id", {"bot_id": self.bot_id, "message_id": message_id})
            )
        storage = self._get_storage()
        backend = storage.get_backend(self.bot_id, provision_schema=True)
        return bool(backend.ignore_message_by_id(message_id))

    def restore_ignored_messages(self) -> int:
        self._ensure_initialized()
        if self.server_url:
            return int(self._call_server("messages_restore_ignored", {"bot_id": self.bot_id}))
        storage = self._get_storage()
        backend = storage.get_backend(self.bot_id, provision_schema=True)
        return int(backend.restore_ignored_messages())

    def delete_memories_by_source_message_ids(self, message_ids: list[str]) -> int:
        self._ensure_initialized()
        if self.server_url:
            return int(
                self._call_server(
                    "memory_delete_by_source_messages",
                    {"bot_id": self.bot_id, "message_ids": message_ids},
                )
            )
        storage = self._get_storage()
        backend = storage.get_backend(self.bot_id, provision_schema=True)
        return int(backend.delete_memories_by_source_message_ids(message_ids))

    def regenerate_embeddings(self, batch_size: int = 50) -> dict[str, Any]:
        self._ensure_initialized()
        if self.server_url:
            return self._call_server(
                "memory_regenerate_embeddings",
                {"bot_id": self.bot_id, "batch_size": batch_size},
            )
        storage = self._get_storage()
        backend = storage.get_backend(self.bot_id, provision_schema=True)
        return backend.regenerate_embeddings(batch_size=batch_size)

    def consolidate_memories(self, dry_run: bool = True, similarity_threshold: float | None = None) -> dict[str, Any]:
        self._ensure_initialized()
        if self.server_url:
            return self._call_server(
                "memory_consolidate",
                {"bot_id": self.bot_id, "dry_run": dry_run, "similarity_threshold": similarity_threshold},
            )
        storage = self._get_storage()
        backend = storage.get_backend(self.bot_id, provision_schema=True)
        from llm_bawt.memory.consolidation import MemoryConsolidator, get_local_llm_client

        llm_client = None if dry_run else get_local_llm_client(self.config)
        threshold = similarity_threshold or getattr(self.config, "MEMORY_CONSOLIDATION_THRESHOLD", 0.92)
        consolidator = MemoryConsolidator(
            backend=backend,
            llm_client=llm_client,
            similarity_threshold=threshold,
            config=self.config,
        )
        result = consolidator.consolidate(dry_run=dry_run)
        return {
            "clusters_found": result.clusters_found,
            "clusters_merged": result.clusters_merged,
            "memories_consolidated": result.memories_consolidated,
            "new_memories_created": result.new_memories_created,
            "errors": result.errors,
            "dry_run": dry_run,
            "similarity_threshold": threshold,
        }

    def update_memory_meaning(
        self,
        memory_id: str,
        intent: str | None = None,
        stakes: str | None = None,
        emotional_charge: float | None = None,
        recurrence_keywords: list[str] | None = None,
        updated_tags: list[str] | None = None,
    ) -> bool:
        self._ensure_initialized()
        if self.server_url:
            return bool(
                self._call_server(
                    "memory_update_meaning",
                    {
                        "bot_id": self.bot_id,
                        "memory_id": memory_id,
                        "intent": intent,
                        "stakes": stakes,
                        "emotional_charge": emotional_charge,
                        "recurrence_keywords": recurrence_keywords,
                        "updated_tags": updated_tags,
                    },
                )
            )
        storage = self._get_storage()
        return bool(
            _run_async(
                storage.update_memory_meaning(
                    bot_id=self.bot_id,
                    memory_id=memory_id,
                    intent=intent,
                    stakes=stakes,
                    emotional_charge=emotional_charge,
                    recurrence_keywords=recurrence_keywords,
                    updated_tags=updated_tags,
                )
            )
        )

    def run_maintenance(
        self,
        run_consolidation: bool = True,
        run_recurrence_detection: bool = True,
        run_decay_pruning: bool = False,
        run_orphan_cleanup: bool = False,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        self._ensure_initialized()
        if self.server_url:
            return self._call_server(
                "system_run_maintenance",
                {
                    "bot_id": self.bot_id,
                    "run_consolidation": run_consolidation,
                    "run_recurrence_detection": run_recurrence_detection,
                    "run_decay_pruning": run_decay_pruning,
                    "run_orphan_cleanup": run_orphan_cleanup,
                    "dry_run": dry_run,
                },
            )
        storage = self._get_storage()
        return _run_async(
            storage.run_maintenance(
                bot_id=self.bot_id,
                run_consolidation=run_consolidation,
                run_recurrence_detection=run_recurrence_detection,
                run_decay_pruning=run_decay_pruning,
                run_orphan_cleanup=run_orphan_cleanup,
                dry_run=dry_run,
            )
        )

    # -------------------------------------------------------------------------
    # History Operations (MCP-backed)
    # -------------------------------------------------------------------------

    def get_messages(
        self,
        since_seconds: int | None = None,
        limit: int | None = None,
        since: float | None = None,
        until: float | None = None,
        session_id: str | None = None,
        summaries_only: bool = False,
        exclude_summarized: bool = False,
    ) -> list[dict[str, Any]]:
        """Get conversation messages with optional filtering.

        Args:
            since_seconds: Only messages from last N seconds (legacy param).
            limit: Maximum messages to return.
            since: Unix timestamp - only include messages after this time.
            until: Unix timestamp - only include messages before this time.
            session_id: TASK-284 — scope the read to one durable thread's
                transcript. When None, existing behaviour is unchanged.
            summaries_only: TASK-284 step 12 — return only the rolling summary
                husks (continuity), no raw bubbles.

        Returns:
            List of message dicts with role, content, timestamp, session_id.
        """
        self._ensure_initialized()
        if self.server_url:
            result = self._call_server(
                "messages_get",
                {
                    "bot_id": self.bot_id, "since_seconds": since_seconds,
                    "limit": limit, "session_id": session_id,
                    "summaries_only": summaries_only,
                    "exclude_summarized": exclude_summarized,
                },
            )
            # Apply timestamp filtering (server doesn't support since/until yet)
            if since is not None or until is not None:
                filtered = []
                for msg in result:
                    ts = msg.get("timestamp", 0)
                    if since is not None and ts < since:
                        continue
                    if until is not None and ts > until:
                        continue
                    filtered.append(msg)
                return filtered
            return result
        storage = self._get_storage()
        # Use storage API directly (async) via helper
        messages = _run_async(storage.get_messages(bot_id=self.bot_id, since_seconds=since_seconds, limit=limit, session_id=session_id, summaries_only=summaries_only, exclude_summarized=exclude_summarized))

        # Apply timestamp filtering if specified (post-filter for now)
        if since is not None or until is not None:
            filtered = []
            for msg in messages:
                ts = msg.get("timestamp", 0)
                if since is not None and ts < since:
                    continue
                if until is not None and ts > until:
                    continue
                filtered.append(msg)
            return filtered

        return messages

    def get_active_session(self) -> dict[str, Any] | None:
        """Return this (bot, user)'s active durable session, or None.

        TASK-284 step 12: the read-path resolver for session-scoped history.
        Scopes by ``self.user_id`` so it targets the same thread the write path
        attributes messages to. Returns None when no active thread exists yet
        (callers fall back to the legacy whole-history load).
        """
        self._ensure_initialized()
        if self.server_url:
            return self._call_server(
                "sessions_get_active",
                {"bot_id": self.bot_id, "user_id": self.user_id},
            )
        storage = self._get_storage()
        return _run_async(
            storage.get_active_session(bot_id=self.bot_id, user_id=self.user_id)
        )

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        """Return one session row by id (TASK-252), or None.

        Backs the per-thread SDK-key resolution at agent dispatch: the app
        reads the target thread's ``session_metadata`` synchronously through
        the same client chain the rotate/activate helpers use.
        """
        self._ensure_initialized()
        if self.server_url:
            return self._call_server(
                "sessions_get",
                {"bot_id": self.bot_id, "session_id": session_id},
            )
        storage = self._get_storage()
        return _run_async(storage.get_session(session_id, bot_id=self.bot_id))

    def get_or_create_active_session(self) -> str:
        """Return this (bot, user)'s active thread id, creating one if absent."""
        self._ensure_initialized()
        if self.server_url:
            return str(self._call_server(
                "sessions_get_or_create_active",
                {"bot_id": self.bot_id, "user_id": self.user_id},
            ))
        storage = self._get_storage()
        return str(_run_async(storage.get_or_create_active_session(
            bot_id=self.bot_id, user_id=self.user_id,
        )))

    def rotate_session(
        self, idempotency_key: str | None = None, *, accept_pointer: bool = False,
    ) -> str | None:
        """Rotate the thread; an accepted /new stamps its pointer atomically."""
        self._ensure_initialized()
        key_args = {"idempotency_key": idempotency_key} if idempotency_key else {}
        if self.server_url:
            return self._call_server(
                "sessions_rotate", {"bot_id": self.bot_id, "user_id": self.user_id,
                                    "accept_pointer": accept_pointer,
                                    **({"key": idempotency_key} if idempotency_key else {})}
            )
        storage = self._get_storage()
        return _run_async(
            storage.rotate_session(bot_id=self.bot_id, user_id=self.user_id,
                                   accept_pointer=accept_pointer, **key_args)
        )

    def clear_messages(self) -> int:
        self._ensure_initialized()
        if self.server_url:
            return int(self._call_server("messages_clear", {"bot_id": self.bot_id}))
        storage = self._get_storage()
        return int(_run_async(storage.clear_messages(bot_id=self.bot_id)))

    def clear_memories(self) -> bool:
        """Delete all distilled memories for this bot."""
        self._ensure_initialized()
        if self.server_url:
            return bool(self._call_server("memory_clear", {"bot_id": self.bot_id}))
        storage = self._get_storage()
        return bool(_run_async(storage.clear_memories(bot_id=self.bot_id)))

    def remove_last_message_if_partial(self, role: str) -> bool:
        self._ensure_initialized()
        if self.server_url:
            return bool(self._call_server("messages_remove_last_partial", {"bot_id": self.bot_id, "role": role}))
        storage = self._get_storage()
        return bool(_run_async(storage.remove_last_message_if_partial(bot_id=self.bot_id, role=role)))

    def get_messages_for_summary(self, summary_id: str) -> list[dict[str, Any]]:
        """Get raw messages referenced by a summary row."""
        self._ensure_initialized()
        if self.server_url:
            result = self._call_server(
                "get_messages_for_summary",
                {"bot_id": self.bot_id, "summary_id": summary_id},
            )
            return result if isinstance(result, list) else []

        storage = self._get_storage()
        return _run_async(storage.get_messages_for_summary(bot_id=self.bot_id, summary_id=summary_id))

    def mark_messages_recalled(self, message_ids: list[str]) -> int:
        """Mark summary-recalled messages in storage."""
        self._ensure_initialized()
        if self.server_url:
            return int(
                self._call_server(
                    "mark_messages_recalled",
                    {"bot_id": self.bot_id, "message_ids": message_ids},
                )
            )

        storage = self._get_storage()
        return int(
            _run_async(
                storage.mark_messages_recalled(bot_id=self.bot_id, message_ids=message_ids)
            )
        )
    
    def update_memory(
        self,
        memory_id: str,
        content: str | None = None,
        importance: float | None = None,
        tags: list[str] | None = None,
    ) -> MemoryResult | None:
        """Update an existing memory.
        
        Args:
            memory_id: ID of memory to update.
            content: New content (if updating).
            importance: New importance score.
            tags: New tags list.
            
        Returns:
            Updated MemoryResult or None if not found.
        """
        self._ensure_initialized()
        
        if self.server_url:
            result = self._call_server("memory_update", {
                "memory_id": memory_id,
                "bot_id": self.bot_id,
                "content": content,
                "importance": importance,
                "tags": tags,
            })
            return MemoryResult.from_dict(result) if result else None
        
        # Embedded mode
        storage = self._get_storage()
        result = _run_async(
            storage.update_memory(
                memory_id=memory_id,
                bot_id=self.bot_id,
                content=content,
                importance=importance,
                tags=tags,
            )
        )
        return MemoryResult.from_dict(result.to_dict()) if result else None
    
    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory by ID.
        
        Args:
            memory_id: ID of memory to delete.
            
        Returns:
            True if deleted, False if not found.
        """
        self._ensure_initialized()
        
        if self.server_url:
            return self._call_server("memory_delete", {
                "memory_id": memory_id,
                "bot_id": self.bot_id,
            })
        
        # Embedded mode
        storage = self._get_storage()
        return _run_async(
            storage.delete_memory(
                memory_id=memory_id,
                bot_id=self.bot_id,
            )
        )
    
    # -------------------------------------------------------------------------
    # Message Operations
    # -------------------------------------------------------------------------
    
    def add_message(
        self,
        role: str,
        content: str,
        session_id: str | None = None,
        timestamp: float | None = None,
        message_id: str | None = None,
        attachments: list[dict] | None = None,
        reasoning: str | None = None,
        author_entity_type: str | None = None,
        author_entity_id: str | None = None,
    ) -> MessageResult:
        """Add a message to conversation history.

        Args:
            role: Message role (user/assistant/system).
            content: Message content.
            session_id: Optional session ID.
            message_id: Optional client-supplied message UUID.  When set, used
                as the persistent ID so frontend/backend share the same key.
            attachments: TASK-222 — optional tiny JSONB ref list to persist on
                the message row's ``attachments`` column, e.g.
                ``[{"asset_id": "ma_xxx", "kind": "image"}]``. ``None`` leaves
                the column at its default ``[]``.

        Returns:
            MessageResult with the stored message.
        """
        self._ensure_initialized()

        if self.server_url:
            result = self._call_server("messages_add", {
                "role": role,
                "content": content,
                "bot_id": self.bot_id,
                "session_id": session_id,
                "timestamp": timestamp,
                "message_id": message_id,
                "attachments": attachments,
                "reasoning": reasoning,
                "author_entity_type": author_entity_type,
                "author_entity_id": author_entity_id,
                # TASK-284: carry the user dimension so the server resolves the
                # active thread per (bot_id, user_id) when session_id is None.
                "user_id": self.user_id,
            })
            return MessageResult.from_dict(result)

        # Embedded mode
        storage = self._get_storage()
        result = _run_async(
            storage.add_message(
                role=role,
                content=content,
                bot_id=self.bot_id,
                session_id=session_id,
                timestamp=timestamp,
                message_id=message_id,
                attachments=attachments,
                reasoning=reasoning,
                user_id=self.user_id,
                author_entity_type=author_entity_type,
                author_entity_id=author_entity_id,
            )
        )
        return MessageResult.from_dict(result.to_dict())
    
    def search_messages(
        self,
        query: str,
        n_results: int = 10,
        role_filter: str | None = None,
        since: float | None = None,
        until: float | None = None,
    ) -> list[MessageResult]:
        """Search ALL conversation history using full-text search.

        Uses PostgreSQL full-text search for efficient keyword matching
        across the entire message history.

        Args:
            query: Search query (keywords, phrases).
            n_results: Maximum results to return.
            role_filter: Only include messages with this role (user/assistant/None for all).
            since: Unix timestamp - only include messages after this time.
            until: Unix timestamp - only include messages before this time.

        Returns:
            List of MessageResult sorted by relevance.
        """
        self._ensure_initialized()

        # For message search, we need direct database access
        # Create a backend connection directly since this isn't routed through MCP
        from ..memory.postgresql import PostgreSQLMemoryBackend

        backend = PostgreSQLMemoryBackend(
            self.config,
            bot_id=self.bot_id,
            provision_schema=False,
        )

        # Use the existing search_messages_by_text method
        results = backend.search_messages_by_text(
            query=query,
            n_results=n_results,
            exclude_recent_seconds=0,  # Don't exclude recent for explicit searches
            role_filter=role_filter,
            since=since,
            until=until,
        )

        return [
            MessageResult(
                id=r.get("id", ""),
                role=r.get("role", ""),
                content=r.get("content", ""),
                bot_id=self.bot_id,
                created_at=str(r.get("timestamp", "")),
            )
            for r in results
        ]

    # -------------------------------------------------------------------------
    # Extraction Operations
    # -------------------------------------------------------------------------
    
    def extract_facts(
        self,
        messages: list[dict],
        store: bool = True,
        use_llm: bool = True,
    ) -> list[MemoryResult]:
        """Extract facts from conversation messages.
        
        Args:
            messages: List of message dicts with role/content.
            store: Whether to persist extracted facts as memories.
            use_llm: Whether to use LLM extraction (vs heuristics).
            
        Returns:
            List of extracted MemoryResult objects.
        """
        self._ensure_initialized()
        
        if self.server_url:
            if not self.user_id:
                raise ValueError("user_id is required for extract_facts")
            results = self._call_server("facts_extract", {
                "messages": messages,
                "bot_id": self.bot_id,
                "user_id": self.user_id,
                "store": store,
                "use_llm": use_llm,
            })
            return [MemoryResult.from_dict(r) for r in results]
        
        # Embedded mode - call extraction directly
        from llm_bawt.mcp_server.extraction import extract_facts_from_messages
        
        if not self.user_id:
            raise ValueError("user_id is required for extract_facts")
        facts = _run_async(
            extract_facts_from_messages(
                messages=messages,
                config=self.config,
                use_llm=use_llm,
                user_id=self.user_id,
            )
        )
        
        if not facts:
            return []
        
        # Store facts if requested
        if store:
            stored = []
            for fact in facts:
                memory = self.store_memory(
                    content=fact["content"],
                    tags=fact.get("tags", ["misc"]),
                    importance=fact.get("importance", 0.5),
                    source_message_ids=fact.get("source_message_ids", []),
                )
                stored.append(memory)
            return stored
        
        return [MemoryResult.from_dict(f) for f in facts]
    
    # -------------------------------------------------------------------------
    # Cross-bot Source Discovery & Search
    # -------------------------------------------------------------------------

    def list_memory_sources(self) -> list[dict[str, Any]]:
        """List available memory sources (bot namespaces with stored memories).

        Returns:
            List of dicts with 'source' (bot_id) and 'memory_count'.
        """
        self._ensure_initialized()

        if self.server_url:
            return self._call_server("memory_list_sources", {})

        storage = self._get_storage()
        return _run_async(storage.list_memory_sources())

    def search_memory_source(
        self,
        source: str,
        query: str,
        n_results: int = 10,
        min_relevance: float = 0.0,
        tags: list[str] | None = None,
    ) -> list[MemoryResult]:
        """Search another bot's memories (read-only cross-bot search).

        Args:
            source: The bot_id whose memories to search.
            query: Natural language search query.
            n_results: Maximum results to return.
            min_relevance: Minimum similarity threshold 0-1.
            tags: Optional tag filter.

        Returns:
            List of MemoryResult from the target source.
        """
        self._ensure_initialized()

        if self.server_url:
            results = self._call_server("memory_search_source", {
                "source": source,
                "query": query,
                "n_results": n_results,
                "min_relevance": min_relevance,
                "tags": tags,
            })
            return [MemoryResult.from_dict(r) for r in results]

        storage = self._get_storage()
        results = _run_async(
            storage.search_memories(
                query=query,
                bot_id=source,
                n_results=n_results,
                min_relevance=min_relevance,
                tags=tags,
            )
        )
        return [MemoryResult.from_dict(r.to_dict()) for r in results]

    # -------------------------------------------------------------------------
    # Cross-bot Aggregate Search
    # -------------------------------------------------------------------------

    def search_all_messages(
        self,
        query: str,
        n_results: int = 10,
        role_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Full-text search across ALL bots' message histories.

        Returns results tagged with ``source`` (bot_id).
        """
        self._ensure_initialized()

        if self.server_url:
            return self._call_server("messages_search_all", {
                "query": query,
                "n_results": n_results,
                "role_filter": role_filter,
            })

        storage = self._get_storage()
        return _run_async(storage.search_all_messages(
            query=query,
            n_results=n_results,
            role_filter=role_filter,
        ))

    def search_all_memories(
        self,
        query: str,
        n_results: int = 10,
        min_relevance: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Semantic search across ALL bots' memory stores.

        Returns results tagged with ``source`` (bot_id).
        """
        self._ensure_initialized()

        if self.server_url:
            return self._call_server("memory_search_all", {
                "query": query,
                "n_results": n_results,
                "min_relevance": min_relevance,
            })

        storage = self._get_storage()
        return _run_async(storage.search_all_memories(
            query=query,
            n_results=n_results,
            min_relevance=min_relevance,
        ))

    # -------------------------------------------------------------------------
    # Additional Memory Operations
    # -------------------------------------------------------------------------
    
    def add_memory(
        self,
        content: str,
        tags: list[str] | None = None,
        importance: float = 0.5,
        source_message_ids: list[str] | None = None,
    ) -> str:
        """Add a memory and return its ID.
        
        Convenience wrapper around store_memory that returns just the ID.
        """
        result = self.store_memory(
            content=content,
            tags=tags,
            importance=importance,
            source_message_ids=source_message_ids,
        )
        return result.id
    
    def list_recent(self, n: int = 50) -> list[dict]:
        """List recent memories.
        
        Args:
            n: Maximum number of memories to return.
            
        Returns:
            List of memory dicts ordered by recency.
        """
        self._ensure_initialized()
        
        if self.server_url:
            return self._call_server("list_recent", {"bot_id": self.bot_id, "n": n})
        
        # Embedded mode
        storage = self._get_storage()
        backend = storage.get_backend(self.bot_id)
        if hasattr(backend, 'list_recent'):
            return backend.list_recent(n=n)
        return []
    
    def supersede_memory(self, old_memory_id: str, new_memory_id: str) -> bool:
        """Mark a memory as superseded by another.
        
        Used for memory evolution - when a fact is updated, the old version
        is superseded rather than deleted to preserve history.
        
        Args:
            old_memory_id: ID of the memory being replaced.
            new_memory_id: ID of the new memory (or "DELETED" for soft delete).
            
        Returns:
            True if successful.
        """
        self._ensure_initialized()
        
        if self.server_url:
            return self._call_server("supersede_memory", {
                "bot_id": self.bot_id,
                "old_memory_id": old_memory_id,
                "new_memory_id": new_memory_id,
            })
        
        # Embedded mode
        storage = self._get_storage()
        backend = storage.get_backend(self.bot_id, provision_schema=True)
        if hasattr(backend, 'supersede_memory'):
            return backend.supersede_memory(old_memory_id, new_memory_id)  # type: ignore
        return False
    
    # -------------------------------------------------------------------------
    # Server Mode Helpers
    # -------------------------------------------------------------------------
    
    def _call_server(self, method: str, params: dict) -> Any:
        """Call an MCP server method via HTTP.
        
        This is a simple HTTP client for the MCP server. For production use,
        consider using the official MCP client library.
        """
        import json
        import time
        import urllib.request
        import urllib.error
        from uuid import uuid4
        
        url = f"{self.server_url}/mcp"
        call_id = uuid4().hex[:8]
        payload = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": method,
                "arguments": params,
            },
            "id": call_id,
        }

        # If we're running inside llm-service, propagate its per-request id into logs.
        request_id = None
        try:
            from llm_bawt.service.logging import get_request_id  # type: ignore

            request_id = get_request_id()
        except Exception:
            request_id = None

        # Use ServiceLogger for human-friendly output if available, otherwise standard logging
        try:
            from llm_bawt.service.logging import get_service_logger
            slog = get_service_logger(__name__)
            # Log a human-friendly summary (call + result combined after the fact)
            use_service_logger = True
        except Exception:
            slog = None
            use_service_logger = False
        
        # For DEBUG mode, log technical details
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("MCP request -> tools/%s params=%s", method, params)
        
        data = json.dumps(payload).encode('utf-8')
        headers = {
            "Content-Type": "application/json",
            # MCP streamable-http transport requires Accept header to include text/event-stream
            "Accept": "application/json, text/event-stream",
        }
        
        start = time.perf_counter()
        try:
            req = urllib.request.Request(url, data=data, headers=headers, method='POST')
            with urllib.request.urlopen(req, timeout=30) as response:
                result = json.loads(response.read().decode('utf-8'))
                duration_ms = (time.perf_counter() - start) * 1000
                
                if "error" in result:
                    logger.error(f"MCP server error: {result['error']}")
                    if use_service_logger and slog:
                        slog.mcp_operation(method, bot_id=self.bot_id, duration_ms=duration_ms, success=False)
                    raise RuntimeError(result['error'].get('message', 'Unknown error'))
                    
                # FastMCP returns JSON-RPC result of the form:
                # {"result": {"content": [...], "structuredContent": {"result": <actual_data>}}}
                rpc_result = result.get("result")
                # An explicit JSON null is a decoded domain value (notably an
                # absent sessions_get row), not a signal to return the MCP wrapper.
                undecoded = object()
                parsed_result = undecoded
                if isinstance(rpc_result, dict):
                    if rpc_result.get("isError"):
                        # MCP tool failures are valid JSON-RPC responses, not
                        # successful domain values. Never pass their text to
                        # MessageResult.from_dict (or other typed consumers).
                        details = "; ".join(
                            item["text"] for item in (rpc_result.get("content") or [])
                            if isinstance(item, dict) and isinstance(item.get("text"), str)
                        )
                        raise RuntimeError(f"MCP tool {method} failed: {details or 'unknown tool error'}")
                    # Prefer structuredContent which has the actual typed data
                    structured = rpc_result.get("structuredContent")
                    if isinstance(structured, dict) and "result" in structured:
                        parsed_result = structured["result"]
                    # Fallback to parsing content array
                    elif "content" in rpc_result:
                        content_items = rpc_result.get("content") or []
                        if content_items and isinstance(content_items, list):
                            first = content_items[0]
                            if isinstance(first, dict):
                                if first.get("type") == "json" and "json" in first:
                                    parsed_result = first.get("json")
                                elif first.get("type") == "text" and "text" in first:
                                    text = first.get("text")
                                    if isinstance(text, str):
                                        stripped = text.lstrip()
                                        if stripped.startswith("{") or stripped.startswith("["):
                                            try:
                                                parsed_result = json.loads(text)
                                            except Exception:
                                                parsed_result = text
                                        else:
                                            parsed_result = text
                
                if parsed_result is undecoded:
                    parsed_result = rpc_result
                
                # Human-friendly log via ServiceLogger (shows "Fetching history" instead of raw tool names)
                if use_service_logger and slog:
                    # Count items if result is a list
                    count = len(parsed_result) if isinstance(parsed_result, list) else None
                    # Pass params and result for verbose mode detail display
                    slog.mcp_operation(
                        method, 
                        bot_id=self.bot_id, 
                        duration_ms=duration_ms, 
                        count=count,
                        params=params,
                        result=parsed_result,
                    )
                elif logger.isEnabledFor(logging.DEBUG):
                    # Fallback to debug logging
                    logger.debug("MCP result <- tools/%s %.1fms", method, duration_ms)
                
                # Verbose logging - show response data (DEBUG level)
                if logger.isEnabledFor(logging.DEBUG):
                    self._log_response_debug(method, parsed_result)
                
                return parsed_result
        except urllib.error.URLError as e:
            logger.error(f"Failed to connect to MCP server: {e}")
            raise ConnectionError(f"Failed to connect to MCP server at {self.server_url}: {e}")

    def _log_response_debug(self, method: str, result: Any) -> None:
        """Log full response data at DEBUG level without truncation."""
        import json
        
        if result is None:
            logger.debug("MCP response: None")
            return
        
        try:
            logger.debug("MCP response: %s", json.dumps(result, default=str, indent=2))
        except Exception:
            logger.debug("MCP response: %s", result)


def get_memory_client(
    config: Config,
    bot_id: str = "",  # Required - must be passed explicitly
    user_id: str | None = None,
    server_url: str | None = None,
) -> MemoryClient:
    """Factory function to create a memory client.
    
    Args:
        config: Application config.
        bot_id: Bot namespace (required).
        user_id: User ID for profile attribute extraction (optional).
        server_url: Optional MCP server URL for server mode.
        
    Returns:
        Configured MemoryClient instance.
    """
    return MemoryClient(config=config, bot_id=bot_id, user_id=user_id, server_url=server_url)


class _MCPShortTermManager:
    """Adapter matching PostgreSQLShortTermManager but routed via MCP tools."""

    def __init__(self, memory_client: MemoryClient):
        self._memory_client = memory_client

    def add_message(
        self,
        role: str,
        content: str,
        timestamp: float | None = None,
        message_id: str | None = None,
        attachments: list[dict] | None = None,
        reasoning: str | None = None,
        session_id: str | None = None,
        author=None,
    ) -> str:
        # ``reasoning`` (TASK-301) is forwarded through the MCP write chain so
        # server-mode deployments persist it too (display-only on the row).
        # ``session_id`` (TASK-251): explicit thread override — None keeps the
        # server-side active-thread resolution (the primary mode).
        msg = self._memory_client.add_message(
            role=role,
            content=content,
            timestamp=timestamp,
            message_id=message_id,
            attachments=attachments,
            reasoning=reasoning,
            session_id=session_id,
            author_entity_type=(author.entity_type if author else None),
            author_entity_id=(author.entity_id if author else None),
        )
        return msg.id

    def get_messages(
        self,
        since_minutes: int | None = None,
    ) -> list:
        # NOTE: despite name, since_minutes is seconds for backward compatibility.
        from llm_bawt.models.message import Message

        rows = self._memory_client.get_messages(since_seconds=since_minutes)
        return [
            Message(
                role=r.get("role", ""),
                content=r.get("content", ""),
                timestamp=r.get("timestamp", 0.0),
                db_id=r.get("id"),
                session_id=r.get("session_id"),  # TASK-284: preserve thread id
                author_entity_type=(r.get("author") or {}).get("entity_type"),
                author_entity_id=(r.get("author") or {}).get("entity_id"),
            )
            for r in rows
        ]

    def load_session_scoped(
        self, since_minutes: int | None = None, session_id: str | None = None
    ) -> list | None:
        """Session-scoped history read (TASK-284 step 12 / TASK-251).

        Loads ONE durable thread's raw bubbles composed with the bot's rolling
        summary husks (continuity) — the same content shape the legacy
        summary-aware path produced, but with raw scoped to a single thread.

        ``session_id`` (TASK-251): the EXPLICITLY selected thread — this is the
        only caller-facing mode now; a request carrying session_id views or
        continues that thread. When None, falls back to the active thread
        (legacy step-12 behaviour; no longer used as a default context source
        anywhere — sessions are metadata, not a context filter).

        Returns a chronologically-ordered ``list[Message]``, or ``None`` when
        no thread can be resolved (the caller falls back to the continuous
        load — never a silent empty context).
        """
        from llm_bawt.models.message import Message

        if not session_id:
            active = self._memory_client.get_active_session()
            if not active or not active.get("id"):
                return None
            session_id = active["id"]

        raw_rows = self._memory_client.get_messages(
            session_id=session_id, since_seconds=since_minutes,
            exclude_summarized=True,
        )
        summary_rows = self._memory_client.get_messages(summaries_only=True)

        merged = [
            Message(
                role=r.get("role", ""),
                content=r.get("content", ""),
                timestamp=r.get("timestamp", 0.0) or 0.0,
                db_id=r.get("id"),
                session_id=r.get("session_id"),
                author_entity_type=(r.get("author") or {}).get("entity_type"),
                author_entity_id=(r.get("author") or {}).get("entity_id"),
            )
            for r in (list(raw_rows) + list(summary_rows))
        ]
        merged.sort(key=lambda m: m.timestamp or 0.0)
        return merged

    def get_session(self, session_id: str) -> dict | None:
        """Return one session row by id for per-thread SDK-key resolution."""
        return self._memory_client.get_session(session_id)

    def get_active_session(
        self, bot_id: str | None = None, user_id: str | None = None,
    ) -> dict | None:
        """Return this adapter's active thread.

        Optional arguments keep parity with the embedded manager; ownership is
        already bound into ``MemoryClient`` and mismatched overrides are not
        used by the app's request-local history manager.
        """
        return self._memory_client.get_active_session()

    def get_or_create_active_session(
        self, bot_id: str | None = None, user_id: str | None = None,
    ) -> str:
        """Return/create this adapter's active thread for SDK write-back."""
        return self._memory_client.get_or_create_active_session()

    def rotate_session(
        self, idempotency_key: str | None = None, *, accept_pointer: bool = False,
    ) -> str | None:
        """Rotate the active thread; accepted /new stamps the bot-wide pointer."""
        return self._memory_client.rotate_session(
            accept_pointer=accept_pointer,
            **({"idempotency_key": idempotency_key} if idempotency_key else {})
        )

    def clear(self) -> bool:
        deleted = self._memory_client.clear_messages()
        return deleted >= 0

    def remove_last_message_if_partial(self, role: str) -> bool:
        return bool(self._memory_client.remove_last_message_if_partial(role=role))

    def get_messages_for_summary(self, summary_id: str) -> list:
        from llm_bawt.models.message import Message

        rows = self._memory_client.get_messages_for_summary(summary_id)
        return [
            Message(
                role=row.get("role", ""),
                content=row.get("content", ""),
                timestamp=row.get("timestamp", 0.0),
                db_id=row.get("id"),
                session_id=row.get("session_id"),  # TASK-284: preserve thread id
            )
            for row in rows
        ]

    def mark_messages_recalled(self, message_ids: list[str]) -> int:
        return int(self._memory_client.mark_messages_recalled(message_ids))
