"""MCP Memory Client - unified interface to memory operations.

This client provides a unified interface for memory operations that can work in two modes:
1. Embedded mode (default): Direct calls to storage layer (in-process, no IPC overhead)
2. Server mode: JSON-RPC calls to a running llm-bawt MCP server

The embedded mode is preferred for single-process usage as it avoids IPC overhead
while maintaining the same API surface.
"""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

from llm_bawt.utils.config import Config

# Responsibility split (TASK-1222): the facade composes these mixins. The
# result types, ``_run_async`` and ``_MCPShortTermManager`` are re-exported so
# existing ``from llm_bawt.mcp_server.client import ...`` callers are unchanged.
from .client_admin import MemoryClientAdminMixin
from .client_crossbot import MemoryClientCrossBotMixin
from .client_results import MemoryResult, MessageResult
from .client_runtime import _run_async
from .client_short_term import _MCPShortTermManager
from .client_transport import MemoryClientTransportMixin

if TYPE_CHECKING:
    from llm_bawt.mcp_server.storage import MemoryStorage

logger = logging.getLogger(__name__)

__all__ = [
    "MemoryClient",
    "MemoryResult",
    "MessageResult",
    "_MCPShortTermManager",
    "_run_async",
    "get_memory_client",
]


class MemoryClient(
    MemoryClientAdminMixin,
    MemoryClientCrossBotMixin,
    MemoryClientTransportMixin,
):
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

    # Admin / UX operations live in client_admin.MemoryClientAdminMixin.

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
    
    # Cross-bot discovery/search lives in client_crossbot.MemoryClientCrossBotMixin.

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

    # Server-mode transport (_call_server / _log_response_debug) lives in
    # client_transport.MemoryClientTransportMixin.


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
