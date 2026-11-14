"""Admin / UX operations for the memory client facade (TASK-1222).

Mirrors llm-service ``/v1/memory`` endpoints. Server mode calls MCP tools;
embedded mode uses the facade's storage backends.
"""

from __future__ import annotations

from typing import Any

from .client_runtime import _run_async


def _wire(scope: dict[str, Any]) -> dict[str, Any]:
    """Server mode: the signed principal carries the user, not an argument."""
    return {k: v for k, v in scope.items() if k != "user_id"}


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

    def _forget_scope(self, session_id: str | None) -> dict[str, Any]:
        """Owner and optional thread for count/time selection (TASK-2279).

        The client's bound user owns the selection; a bot partition holds every
        user's threads, so a user-less client cannot select by count or time.
        """
        user_id = self._service_principal.user_id
        if not user_id:
            raise ValueError("a user-bound memory client is required to select messages by count or time")
        scope: dict[str, Any] = {"user_id": user_id}
        if session_id:
            scope["session_id"] = session_id
        return scope


    def _owned_backend(self, *, provision: bool = False):
        storage = self._get_storage()
        if provision:
            return storage.get_backend(self.bot_id, provision_schema=True)
        return storage.get_backend(self.bot_id)

    def preview_recent_messages(self, count: int = 10) -> list[dict[str, Any]]:
        self._ensure_initialized()
        scope = self._forget_scope(None)
        if self.server_url:
            return self._call_server("messages_preview_recent", {"bot_id": self.bot_id, "count": count, **_wire(scope)})
        return self._owned_backend().preview_recent_messages(count=count, **scope)

    def preview_messages_since_minutes(self, minutes: int) -> list[dict[str, Any]]:
        self._ensure_initialized()
        scope = self._forget_scope(None)
        if self.server_url:
            return self._call_server(
                "messages_preview_since_minutes", {"bot_id": self.bot_id, "minutes": minutes, **_wire(scope)})
        return self._owned_backend().preview_messages_since_minutes(minutes=minutes, **scope)

    def preview_ignored_messages(self) -> list[dict[str, Any]]:
        self._ensure_initialized()
        if self.server_url:
            return self._call_server("messages_preview_ignored", {"bot_id": self.bot_id})
        storage = self._get_storage()
        backend = storage.get_backend(self.bot_id)
        return backend.preview_ignored_messages()

    def _forget(self, session_id: str | None, **selection: int) -> dict[str, Any]:
        """Ignore the caller's selection and delete memories sourced from it.

        One storage transaction selects, moves and deletes the same ID set.
        """
        self._ensure_initialized()
        scope = self._forget_scope(session_id)
        if self.server_url:
            return self._call_server("messages_forget", {"bot_id": self.bot_id, **selection, **_wire(scope)})
        return self._owned_backend(provision=True).forget_owned_messages(**selection, **scope)

    def forget_recent_messages(self, count: int, *, session_id: str | None = None) -> dict[str, Any]:
        return self._forget(session_id, count=int(count))

    def forget_messages_since_minutes(self, minutes: int, *, session_id: str | None = None) -> dict[str, Any]:
        return self._forget(session_id, minutes=int(minutes))

    def ignore_recent_messages(self, count: int) -> int:
        self._ensure_initialized()
        scope = self._forget_scope(None)
        if self.server_url:
            return int(self._call_server("messages_ignore_recent", {"bot_id": self.bot_id, "count": count, **_wire(scope)}))
        return int(self._owned_backend(provision=True).ignore_recent_messages(count=count, **scope))

    def ignore_messages_since_minutes(self, minutes: int) -> int:
        self._ensure_initialized()
        scope = self._forget_scope(None)
        if self.server_url:
            return int(self._call_server(
                "messages_ignore_since_minutes", {"bot_id": self.bot_id, "minutes": minutes, **_wire(scope)}))
        return int(self._owned_backend(provision=True).ignore_messages_since_minutes(minutes=minutes, **scope))

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
        from llm_bawt.memory.embedding_repair import _eligible

        if not _eligible(self.bot_id, storage.engine):
            return {"error": "bot excluded from embedding repair", "updated": 0, "failed": 0}
        backend = storage.get_backend(self.bot_id, provision_schema=False)
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
