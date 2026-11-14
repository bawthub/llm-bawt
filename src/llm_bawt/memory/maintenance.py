"""Unified memory maintenance orchestrator.

Runs periodic maintenance tasks on the memory system:
1. Consolidation: merge semantically similar memories
2. Recurrence detection: identify repeating themes, add 'recurring' tag
3. Decay pruning: archive stale low-importance memories

The legacy orphan-cleanup flag is accepted for compatibility but performs no work.

Usage (background service):
    from llm_bawt.memory.maintenance import MemoryMaintenance
    maint = MemoryMaintenance(backend, llm_client, config)
    result = maint.run(dry_run=False)

Or via task:
    from llm_bawt.service.tasks import create_maintenance_task
    task = create_maintenance_task(bot_id="helper")
"""

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from llm_bawt.memory.ownership import MemoryAuthority, MemoryOwnership, NO_MEMORY_AUTHORITY
from llm_bawt.utils.error_text import safe_error_text

logger = logging.getLogger(__name__)


INTENT_PROMPT_WITH_CONTEXT = """Given this conversation excerpt and the fact that was extracted from it, determine WHY the user shared this information.

Conversation:
{conversation}

Extracted fact: {fact}

What was the user's intent in sharing this? Describe it in a brief phrase (3-6 words) that captures their underlying motivation or goal. Be specific to this situation rather than generic.

Respond with ONLY the intent phrase, nothing else."""


INTENT_PROMPT_CONTENT_ONLY = """Given this extracted memory/fact about a user, determine WHY they likely shared this information originally.

Memory: {fact}

What was the user's probable intent in sharing this? Describe it in a brief phrase (3-6 words) that captures their underlying motivation or goal. Be specific to this situation rather than generic.

Respond with ONLY the intent phrase, nothing else."""

INTENT_SYSTEM_PROMPT = (
    "You are a helpful assistant that analyzes conversation intent. "
    "Respond with only a short intent phrase."
)


@dataclass
class MaintenanceResult:
    """Result of a maintenance run."""
    consolidation_clusters_merged: int = 0
    consolidation_memories_consolidated: int = 0
    recurrence_themes_detected: int = 0
    recurrence_memories_tagged: int = 0
    decay_memories_archived: int = 0
    orphans_cleaned: int = 0
    errors: list[str] = field(default_factory=list)
    dry_run: bool = False

    def to_dict(self) -> dict:
        return {
            "consolidation_clusters_merged": self.consolidation_clusters_merged,
            "consolidation_memories_consolidated": self.consolidation_memories_consolidated,
            "recurrence_themes_detected": self.recurrence_themes_detected,
            "recurrence_memories_tagged": self.recurrence_memories_tagged,
            "decay_memories_archived": self.decay_memories_archived,
            "orphans_cleaned": self.orphans_cleaned,
            "errors": self.errors,
            "dry_run": self.dry_run,
        }


class MemoryMaintenance:
    """Orchestrates memory maintenance tasks."""

    def __init__(
        self,
        backend: Any,  # PostgreSQLMemoryBackend
        llm_client: Any | None = None,
        config: Any = None,
        *, authority: MemoryAuthority | None = None,
    ):
        self.backend = backend
        self.llm_client = llm_client
        self.config = config
        self.authority = authority or NO_MEMORY_AUTHORITY

    @property
    def ownership(self) -> MemoryOwnership:
        # Resolve only when an operation needs the backend, inside run's error boundary.
        return self.authority.scope(self.backend.bot_id)

    def run(
        self,
        run_consolidation: bool = True,
        run_recurrence_detection: bool = True,
        run_decay_pruning: bool = False,
        run_orphan_cleanup: bool = False,
        dry_run: bool = False,
    ) -> MaintenanceResult:
        """Execute selected operations; run_orphan_cleanup is a compatibility no-op."""
        result = MaintenanceResult(dry_run=dry_run)

        if run_consolidation:
            try:
                cons_result = self._run_consolidation(dry_run)
                result.consolidation_clusters_merged = cons_result.get("clusters_merged", 0)
                result.consolidation_memories_consolidated = cons_result.get("memories_consolidated", 0)
            except Exception as e:
                logger.exception("Consolidation failed")
                result.errors.append(f"consolidation: {safe_error_text(e)}")

        if run_recurrence_detection:
            try:
                rec_result = self._run_recurrence_detection(dry_run)
                result.recurrence_themes_detected = rec_result.get("themes_detected", 0)
                result.recurrence_memories_tagged = rec_result.get("memories_tagged", 0)
            except Exception as e:
                logger.exception("Recurrence detection failed")
                result.errors.append(f"recurrence: {safe_error_text(e)}")

        if run_decay_pruning:
            try:
                decay_result = self._run_decay_pruning(dry_run)
                result.decay_memories_archived = decay_result.get("archived", 0)
            except Exception as e:
                logger.exception("Decay pruning failed")
                result.errors.append(f"decay: {safe_error_text(e)}")

        if run_orphan_cleanup:
            try:
                orphan_result = self._run_orphan_cleanup(dry_run)
                result.orphans_cleaned = orphan_result.get("cleaned", 0)
            except Exception as e:
                logger.exception("Orphan cleanup failed")
                result.errors.append(f"orphan: {safe_error_text(e)}")

        return result

    # ------------------------------------------------------------------
    # Consolidation (delegates to existing consolidation module)
    # ------------------------------------------------------------------
    def _run_consolidation(self, dry_run: bool) -> dict:
        from .consolidation import MemoryConsolidator

        consolidator = MemoryConsolidator(
            backend=self.backend,
            llm_client=self.llm_client,
            config=self.config,
            authority=self.authority,
        )
        cons = consolidator.consolidate(dry_run=dry_run)
        return {
            "clusters_merged": cons.clusters_merged,
            "memories_consolidated": cons.memories_consolidated,
        }

    # ------------------------------------------------------------------
    # Recurrence detection: cluster by recurrence_keywords overlap
    # ------------------------------------------------------------------
    def _run_recurrence_detection(self, dry_run: bool) -> dict:
        from sqlalchemy import text

        themes_detected = 0
        memories_tagged = 0

        ownership = self.ownership
        with self.backend._memory_write_transaction() as conn:
            # Lock candidates so the evidence/group cannot change during tagging.
            sql = text(f"""
                SELECT id, tags, recurrence_keywords, owner_user_id
                FROM {self.backend._memories_table_name} mem
                WHERE recurrence_keywords IS NOT NULL AND {ownership.mutation_predicate}
                FOR UPDATE
            """)
            rows = conn.execute(sql, ownership.params).fetchall()

            # Identical keywords from distinct owners are not a recurring theme.
            keyword_to_ids: dict[tuple[str | None, str], list[str]] = defaultdict(list)
            id_to_tags: dict[str, list[str]] = {}
            for row in rows:
                keywords = row.recurrence_keywords or []
                if isinstance(keywords, str):
                    try:
                        keywords = json.loads(keywords)
                    except Exception:
                        keywords = []
                tags = row.tags or []
                if isinstance(tags, str):
                    try:
                        tags = json.loads(tags)
                    except Exception:
                        tags = []
                id_to_tags[row.id] = tags
                for kw in keywords:
                    keyword_to_ids[(row.owner_user_id, kw.lower())].append(row.id)

            # Identify recurring themes (keyword appearing in 3+ memories)
            recurring_threshold = getattr(self.config, "MEMORY_RECURRENCE_THRESHOLD", 3)
            recurring_keywords = {kw for kw, ids in keyword_to_ids.items() if len(ids) >= recurring_threshold}
            themes_detected = len(recurring_keywords)

            # Tag memories with 'recurring' if any of their keywords is recurring
            for mem_id, tags in id_to_tags.items():
                mem_keywords = set()
                for kw, ids in keyword_to_ids.items():
                    if mem_id in ids:
                        mem_keywords.add(kw)
                if mem_keywords & recurring_keywords:
                    if "recurring" not in tags:
                        if dry_run:
                            logger.debug(f"[DRY RUN] Would tag {mem_id} as recurring")
                        else:
                            new_tags = tags + ["recurring"]
                            update_sql = text(f"""
                                UPDATE {self.backend._memories_table_name} mem
                                SET tags = CAST(:tags AS jsonb), updated_at = CURRENT_TIMESTAMP
                                WHERE id = :id AND {ownership.mutation_predicate}
                            """)
                            changed = conn.execute(update_sql, {
                                **ownership.params, "id": mem_id, "tags": json.dumps(new_tags),
                            })
                            memories_tagged += changed.rowcount
                            continue
                        memories_tagged += 1

        return {"themes_detected": themes_detected, "memories_tagged": memories_tagged}

    # ------------------------------------------------------------------
    # Decay pruning: archive stale low-importance memories
    # ------------------------------------------------------------------
    def _run_decay_pruning(self, dry_run: bool) -> dict:
        import math
        from sqlalchemy import text

        archived = 0
        half_life = getattr(self.config, "MEMORY_DECAY_HALF_LIFE_DAYS", 90.0)
        prune_threshold = getattr(self.config, "MEMORY_PRUNE_THRESHOLD", 0.05)

        ownership = self.ownership
        with self.backend._memory_write_transaction() as conn:
            sql = text(f"""
                SELECT id, importance, created_at
                FROM {self.backend._memories_table_name} mem
                WHERE {ownership.mutation_predicate}
                FOR UPDATE
            """)
            rows = conn.execute(sql, ownership.params).fetchall()

            now = datetime.now(timezone.utc)
            to_archive: list[str] = []
            for row in rows:
                created_at = row.created_at
                if created_at.tzinfo is None:
                    created_at = created_at.replace(tzinfo=timezone.utc)
                age_days = (now - created_at).total_seconds() / 86400.0
                decay = math.exp(-age_days * math.log(2) / half_life)
                effective_importance = float(row.importance) * decay
                if effective_importance < prune_threshold:
                    to_archive.append(row.id)

            if to_archive:
                if dry_run:
                    logger.debug(f"[DRY RUN] Would delete {len(to_archive)} decayed memories")
                else:
                    # Delete stale low-importance memories
                    delete_sql = text(f"""
                        DELETE FROM {self.backend._memories_table_name} mem
                        WHERE id = ANY(:ids) AND {ownership.mutation_predicate}
                    """)
                    changed = conn.execute(delete_sql, {**ownership.params, "ids": to_archive})
                    archived = changed.rowcount
                if dry_run:
                    archived = len(to_archive)

        return {"archived": archived}

    # ------------------------------------------------------------------
    # Orphan cleanup: retained compatibility no-op, not a supported operation
    # ------------------------------------------------------------------
    def _run_orphan_cleanup(self, dry_run: bool) -> dict:
        # Keep the public flag and zero-count result stable for existing callers.
        return {"cleaned": 0}


def update_memory_meaning(
    backend: Any,
    memory_id: str,
    intent: str | None = None,
    stakes: str | None = None,
    emotional_charge: float | None = None,
    recurrence_keywords: list[str] | None = None,
    updated_tags: list[str] | None = None,
    *, authority: MemoryAuthority | None = None,
) -> bool:
    """Update meaning metadata on an existing memory.
    
    This is the async-safe function called by the background service
    when processing MEANING_UPDATE tasks.
    """
    from sqlalchemy import text
    from .embeddings import generate_embedding, meaning_text

    ownership = (authority or NO_MEMORY_AUTHORITY).scope(backend.bot_id)
    with backend.engine.connect() as conn:
        updates = ["updated_at = CURRENT_TIMESTAMP"]
        params: dict[str, Any] = {**ownership.params, "id": memory_id}
        meaning_changed = any(
            value is not None for value in (intent, stakes, emotional_charge, recurrence_keywords)
        )
        current = conn.execute(text(f"""
            SELECT content, intent, stakes, emotional_charge, recurrence_keywords, owner_user_id
            FROM {backend._memories_table_name} mem
            WHERE id = :id AND {ownership.mutation_predicate}
        """), params).mappings().first()
        conn.commit()  # No transaction/row lock spans the embedding request.
        if current is None:
            return False
        params["old_owner"] = current["owner_user_id"]
        guard = "AND owner_user_id IS NOT DISTINCT FROM CAST(:old_owner AS text)"
        if meaning_changed:
            guard += """
                AND content = :old_content
                AND intent IS NOT DISTINCT FROM :old_intent
                AND stakes IS NOT DISTINCT FROM :old_stakes
                AND emotional_charge IS NOT DISTINCT FROM :old_emotional_charge
                AND recurrence_keywords IS NOT DISTINCT FROM CAST(:old_keywords AS jsonb)
            """
            params.update({
                "old_content": current["content"], "old_intent": current["intent"],
                "old_stakes": current["stakes"],
                "old_emotional_charge": current["emotional_charge"],
                "old_keywords": (json.dumps(current["recurrence_keywords"])
                                 if current["recurrence_keywords"] is not None else None),
            })

        if intent is not None:
            updates.append("intent = :intent")
            params["intent"] = intent
        if stakes is not None:
            updates.append("stakes = :stakes")
            params["stakes"] = stakes
        if emotional_charge is not None:
            updates.append("emotional_charge = :emotional_charge")
            params["emotional_charge"] = emotional_charge
        if recurrence_keywords is not None:
            updates.append("recurrence_keywords = CAST(:recurrence_keywords AS jsonb)")
            params["recurrence_keywords"] = json.dumps(recurrence_keywords)
        if updated_tags is not None:
            updates.append("tags = CAST(:tags AS jsonb)")
            params["tags"] = json.dumps(updated_tags)

        if meaning_changed:
            body = meaning_text(
                current["content"],
                intent if intent is not None else current["intent"],
                stakes if stakes is not None else current["stakes"],
                emotional_charge if emotional_charge is not None else current["emotional_charge"],
                recurrence_keywords if recurrence_keywords is not None else current["recurrence_keywords"],
            )
            try:
                emb = generate_embedding(body, backend.embedding_model)
            except Exception as exc:
                logger.warning("Could not regenerate meaning embedding: %s", exc)
                emb = None
            # Never retain a vector for fields changed during an outage.
            updates.extend([
                "meaning_embedding = :meaning_embedding",
                "meaning_updated_at = CASE WHEN :meaning_embedding IS NULL "
                "THEN NULL ELSE CURRENT_TIMESTAMP END",
            ])
            params["meaning_embedding"] = f"[{','.join(str(x) for x in emb)}]" if emb else None

    with backend._memory_write_transaction() as conn:
        result = conn.execute(text(f"""
            UPDATE {backend._memories_table_name} mem
            SET {', '.join(updates)}
            WHERE id = :id {guard} AND {ownership.mutation_predicate}
        """), params)
        return result.rowcount > 0
