"""Cross-bot source discovery and aggregate search for the memory client (TASK-1222)."""

from __future__ import annotations

from typing import Any

from .client_results import MemoryResult
from .client_runtime import _run_async


class MemoryClientCrossBotMixin:
    """Read-only memory/message search across bot namespaces."""

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
