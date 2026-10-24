"""Cross-bot discovery and search for the storage facade.

Every aggregate call runs under a ``BotAccess``: the caller's bound access, or
a fresh no-caller policy read (TASK-1323). There is no exclusion-list path, so
a policy or backend failure raises :class:`BotAccessError` instead of
returning ``[]`` or an unrestricted result.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from llm_bawt.bot_access import BotAccess, BotAccessError, BotPolicyResolver
from llm_bawt.bot_access.context import current_bot_access
from llm_bawt.bot_access.provenance import MemoryProvenance, ScopedMessageAuthority
from llm_bawt.message_authorship import AuthorReference

# Preserve the original logging namespace for existing filters and diagnostics.
logger = logging.getLogger("llm_bawt.mcp_server.storage")


def _operator_author(row: dict[str, Any], source: str) -> AuthorReference:
    """Map an operator row's authoritative ``author_class`` to its author.

    ``bot``: the conversation's own bot (``source``, canonical slug); legacy
    when the row carries no author pair. ``user`` / ``delivery``: the stored
    id, which the rule already matched. Anything else is Unknown.
    """
    author_class = row.get("author_class")
    stored_id = row.get("author_entity_id")
    if author_class == "bot":
        reference = AuthorReference.bot(source)
        return reference if stored_id else AuthorReference("bot", reference.entity_id, "legacy")
    if author_class == "user" and stored_id:
        return AuthorReference.user(stored_id)
    if author_class == "delivery" and stored_id:
        return AuthorReference.bot(stored_id)
    return AuthorReference.unknown()


def _with_row_status(envelope: dict[str, Any], reference: AuthorReference) -> dict[str, Any]:
    """Presentation is keyed by identity, so one bot's canonical and legacy
    rows share an envelope; keep each row's own status. A missing entity
    stays ``unknown``."""
    if envelope.get("status") == "unknown":
        return dict(envelope)
    return {**envelope, "status": reference.status}


class StorageSearchMixin:
    """Aggregate operations using the facade's engine, backends and hydration."""

    async def _aggregate_access(self) -> BotAccess:
        """The bound caller's access, else a fresh no-caller policy read.

        Guarded MCP tools and the embedded client always bind one; the
        fallback serves direct in-process callers. Never a cached or
        best-effort list: a failed read raises ``policy_unavailable``.
        """
        access = current_bot_access()
        if access is not None:
            return access
        try:
            engine = self._require_engine()
        except Exception:
            raise BotAccessError("source_unavailable") from None
        return await BotPolicyResolver(engine).resolve(None)

    async def list_memory_sources(self) -> list[dict[str, Any]]:
        """List the memory sources in scope with their memory counts.

        One GROUP BY over the partitioned ``memories`` parent (TASK-571) for
        the bots in scope. Bots with no memories appear with
        ``memory_count`` 0.
        """
        from sqlalchemy import text

        from llm_bawt.memory.postgresql import MEMORIES_PARENT

        scope = (await self._aggregate_access()).aggregate_scope()
        if not scope.partitions:
            return []

        try:
            with self._require_engine().connect() as conn:
                rows = conn.execute(text(
                    f"SELECT bot_id, COUNT(*) FROM {MEMORIES_PARENT} "
                    "WHERE bot_id = ANY(CAST(:partitions AS text[])) GROUP BY bot_id"
                ), {"partitions": [p.partition for p in scope.partitions]}).fetchall()
        except Exception as e:
            logger.error("Failed to list memory sources: %s", e)
            raise BotAccessError("source_unavailable") from None
        counts = {partition: int(n) for partition, n in rows}
        return [
            {"source": p.bot_id, "memory_count": counts.get(p.partition, 0)}
            for p in scope.partitions
        ]

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
          all bots in scope. A bot outside the scope raises ``access_denied``.

        Each returned row also carries a ``total`` field (window COUNT(*)
        over the unbounded result set) so the UI can render "showing N of M
        matches" without a second roundtrip. ``total`` is identical on
        every row; callers can read it from the first row and ignore the
        rest. Empty result → no rows → caller should treat as ``total=0``.
        A failed search raises ``source_unavailable``; ``[]`` always means
        no match.
        """
        from llm_bawt.mcp_server.message_search import CrossBotMessageSearcher

        scope = (await self._aggregate_access()).aggregate_scope(bot_id or None)
        names = {p.partition: p.bot_id for p in scope.partitions}
        try:
            searcher = CrossBotMessageSearcher(
                self._require_engine(),
                lambda rows, source: self.hydrate_message_authors(rows, bot_id=source),
            )
            rows = await asyncio.to_thread(
                searcher.search_fts,
                query,
                partitions=frozenset(names),
                n_results=n_results,
                role_filter=role_filter,
                sort_by=sort_by,
                since=since,
                until=until,
            )
        except Exception as e:
            logger.error("search_all_messages failed: %s", e)
            raise BotAccessError("source_unavailable") from None
        for row in rows:
            row["source"] = names[row["source"]]
        return rows

    async def search_messages_operator(
        self,
        query: str,
        *,
        mode: str = "fts",
        author: AuthorReference | None = None,
        in_bot: str | None = None,
        n_results: int = 10,
        role_filter: str | None = None,
        sort_by: str = "relevance",
        since: float | None = None,
        until: float | None = None,
    ) -> list[dict[str, Any]]:
        """Operator (Spotlight/REST) cross-bot search with ``from:``/``in:`` (TASK-1301).

        Membership comes from a fresh positive policy read; a policy or
        backend failure raises :class:`BotAccessError` instead of returning an
        empty or unrestricted result. Rows carry ``author_class``.

        ``mode="fts"`` uses Postgres FTS (stems, drops stop-words; breaks on
        identifiers: ``TASK-241`` tokenizes to ``task``). ``mode="trgm"``
        treats content as a flat character sequence: ``ILIKE`` substring
        match ranked by trigram similarity, backed by the per-partition
        ``gin_trgm_ops`` indexes, so ``TASK-241`` matches literally or not.
        """
        from llm_bawt.bot_access import BotPolicyResolver
        from llm_bawt.mcp_server.message_search import (
            CrossBotMessageSearcher,
            OperatorMessageScope,
        )

        try:
            engine = self._require_engine()
            access = await BotPolicyResolver(engine).resolve(None)
            scope = OperatorMessageScope.from_access(access, author=author, in_bot=in_bot)
            searcher = CrossBotMessageSearcher(engine, self._hydrate_operator_authors)
            return await asyncio.to_thread(
                searcher.search_operator,
                "trgm" if mode == "trgm" else "fts",
                query,
                scope,
                n_results=n_results,
                role_filter=role_filter,
                sort_by=sort_by,
                since=since,
                until=until,
            )
        except BotAccessError:
            raise
        except Exception as e:
            logger.error("search_messages_operator failed: %s", e)
            raise BotAccessError("source_unavailable") from None

    def _hydrate_operator_authors(
        self, rows: list[dict[str, Any]], source: str
    ) -> list[dict[str, Any]]:
        """Display envelopes derived only from the SQL ``author_class``.

        :class:`EffectiveAuthorRule` already decided who wrote each row, so
        the identity comes from that decision, not from LegacyAuthorResolver:
        legacy hydration would re-guess (a delivery record can reattribute a
        blank assistant row) or raise on a partial pair and fail the whole
        result set. Presentation only adds name, color and avatar.
        """
        from llm_bawt.entity_presentation import EntityPresentationResolver

        references = [_operator_author(row, source) for row in rows]
        presentation = EntityPresentationResolver(
            self._require_engine()
        ).resolve_many_safe(references)
        return [
            {**row, "author": _with_row_status(presentation[(ref.entity_type, ref.entity_id)], ref)}
            for row, ref in zip(rows, references, strict=True)
        ]

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

        scope = (await self._aggregate_access()).aggregate_scope(
            bot_id or None, require_user=True,
        )
        try:
            query_embedding = generate_embedding(
                query, self.config.MEMORY_EMBEDDING_MODEL, verbose=self.config.VERBOSE,
            )
        except Exception:
            raise BotAccessError("source_unavailable") from None
        if not query_embedding:
            raise BotAccessError("source_unavailable")

        all_results: list[dict[str, Any]] = []
        for partition in scope.partitions:
            bot_id = partition.bot_id
            try:
                backend = self._get_backend(bot_id)
                results = backend.search_memories_by_embedding(
                    embedding=query_embedding,
                    n_results=n_results,
                    min_importance=0.0,
                    eligibility=MemoryProvenance(
                        ScopedMessageAuthority.for_partition(scope, bot_id)
                    ),
                )
                for r in results:
                    relevance = r.get("similarity") or r.get("relevance", 0.0)
                    if relevance < min_relevance:
                        continue
                    r["source"] = bot_id
                    r["relevance"] = relevance
                    all_results.append(r)
            except BotAccessError:
                raise
            except Exception as e:
                logger.error("search_all_memories failed on %s: %s", bot_id, e)
                raise BotAccessError("source_unavailable") from None

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
