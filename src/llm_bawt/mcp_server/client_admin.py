"""Admin / UX operations for the memory client facade (TASK-1222).

Mirrors llm-service ``/v1/memory`` endpoints. Server mode calls MCP tools;
embedded mode uses the facade's storage backends.
"""

from __future__ import annotations

from typing import Any

from .client_runtime import _run_async


class MemoryClientAdminMixin:
    """Stats, previews, forget/ignore, consolidation and maintenance."""

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
                "memory_list_high_importance",
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
                "messages_preview_since_minutes",
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
