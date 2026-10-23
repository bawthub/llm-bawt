"""Cross-bot discovery and search for the storage facade.

Mechanical extraction only: existing policy and error semantics are unchanged.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

# Preserve the original logging namespace for existing filters and diagnostics.
logger = logging.getLogger("llm_bawt.mcp_server.storage")


class StorageSearchMixin:
    """Aggregate operations using the facade's engine, backends and hydration."""

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
