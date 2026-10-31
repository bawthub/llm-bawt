"""Administrative memory/history operations for the storage facade."""

from __future__ import annotations

from typing import Any


class StorageAdminMixin:
    """Maintenance and UX operations sharing the facade's cached backends."""

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
        from llm_bawt.memory.embedding_repair import _eligible

        if not _eligible(bot_id, self.engine):
            return {"error": "bot excluded from embedding repair", "updated": 0, "failed": 0}
        backend = self._get_backend(bot_id, provision_schema=False)
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
