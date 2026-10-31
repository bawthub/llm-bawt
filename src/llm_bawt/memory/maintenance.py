"""Unified memory maintenance orchestrator.

Runs periodic maintenance tasks on the memory system:
1. Consolidation: merge semantically similar memories
2. Recurrence detection: identify repeating themes, add 'recurring' tag
3. Decay pruning: archive stale low-importance memories
4. Orphan cleanup: remove orphaned embeddings or metadata

Usage (background service):
    from llm_bawt.memory.maintenance import MemoryMaintenance
    maint = MemoryMaintenance(backend, llm_client, config)
    result = maint.run(dry_run=False)

Or via task:
    from llm_bawt.service.tasks import create_maintenance_task
    task = create_maintenance_task(bot_id="nova")
"""

import json
import logging
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from ..prompt_registry import get_prompt_resolver

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
    ):
        self.backend = backend
        self.llm_client = llm_client
        self.config = config

    def run(
        self,
        run_consolidation: bool = True,
        run_recurrence_detection: bool = True,
        run_decay_pruning: bool = False,
        run_orphan_cleanup: bool = False,
        dry_run: bool = False,
    ) -> MaintenanceResult:
        """Execute selected maintenance operations."""
        result = MaintenanceResult(dry_run=dry_run)

        if run_consolidation:
            try:
                cons_result = self._run_consolidation(dry_run)
                result.consolidation_clusters_merged = cons_result.get("clusters_merged", 0)
                result.consolidation_memories_consolidated = cons_result.get("memories_consolidated", 0)
            except Exception as e:
                logger.exception("Consolidation failed")
                result.errors.append(f"consolidation: {e}")

        if run_recurrence_detection:
            try:
                rec_result = self._run_recurrence_detection(dry_run)
                result.recurrence_themes_detected = rec_result.get("themes_detected", 0)
                result.recurrence_memories_tagged = rec_result.get("memories_tagged", 0)
            except Exception as e:
                logger.exception("Recurrence detection failed")
                result.errors.append(f"recurrence: {e}")

        if run_decay_pruning:
            try:
                decay_result = self._run_decay_pruning(dry_run)
                result.decay_memories_archived = decay_result.get("archived", 0)
            except Exception as e:
                logger.exception("Decay pruning failed")
                result.errors.append(f"decay: {e}")

        if run_orphan_cleanup:
            try:
                orphan_result = self._run_orphan_cleanup(dry_run)
                result.orphans_cleaned = orphan_result.get("cleaned", 0)
            except Exception as e:
                logger.exception("Orphan cleanup failed")
                result.errors.append(f"orphan: {e}")

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

        with self.backend.engine.connect() as conn:
            # Fetch all memories with recurrence_keywords
            sql = text(f"""
                SELECT id, tags, recurrence_keywords
                FROM {self.backend._memories_table_name}
                WHERE recurrence_keywords IS NOT NULL
            """)
            rows = conn.execute(sql).fetchall()

            # Build keyword -> memory_ids mapping
            keyword_to_ids: dict[str, list[str]] = defaultdict(list)
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
                    keyword_to_ids[kw.lower()].append(row.id)

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
                                UPDATE {self.backend._memories_table_name}
                                SET tags = CAST(:tags AS jsonb), updated_at = CURRENT_TIMESTAMP
                                WHERE id = :id
                            """)
                            conn.execute(update_sql, {"id": mem_id, "tags": json.dumps(new_tags)})
                        memories_tagged += 1

            if not dry_run:
                conn.commit()

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

        with self.backend.engine.connect() as conn:
            sql = text(f"""
                SELECT id, importance, created_at
                FROM {self.backend._memories_table_name}
            """)
            rows = conn.execute(sql).fetchall()

            now = datetime.now(timezone.utc)
            to_archive: list[str] = []
            for row in rows:
                age_days = (now - row.created_at).total_seconds() / 86400.0
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
                        DELETE FROM {self.backend._memories_table_name}
                        WHERE id = ANY(:ids)
                    """)
                    conn.execute(delete_sql, {"ids": to_archive})
                    conn.commit()
                archived = len(to_archive)

        return {"archived": archived}

    # ------------------------------------------------------------------
    # Orphan cleanup: placeholder for future implementation
    # ------------------------------------------------------------------
    def _run_orphan_cleanup(self, dry_run: bool) -> dict:
        # Currently no orphan cleanup implemented
        return {"cleaned": 0}


def update_memory_meaning(
    backend: Any,
    memory_id: str,
    intent: str | None = None,
    stakes: str | None = None,
    emotional_charge: float | None = None,
    recurrence_keywords: list[str] | None = None,
    updated_tags: list[str] | None = None,
) -> bool:
    """Update meaning metadata on an existing memory.
    
    This is the async-safe function called by the background service
    when processing MEANING_UPDATE tasks.
    """
    from sqlalchemy import text
    from .embeddings import generate_embedding, meaning_text

    with backend.engine.connect() as conn:
        updates = ["updated_at = CURRENT_TIMESTAMP"]
        params: dict[str, Any] = {"id": memory_id}
        meaning_changed = any(
            value is not None for value in (intent, stakes, emotional_charge, recurrence_keywords)
        )
        guard = ""
        if meaning_changed:
            current = conn.execute(text(f"""
                SELECT content, intent, stakes, emotional_charge, recurrence_keywords
                FROM {backend._memories_table_name} WHERE id = :id
            """), {"id": memory_id}).mappings().first()
            conn.commit()  # No transaction/row lock spans the embedding request.
            if current is None:
                return False
            guard = """
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

        result = conn.execute(text(f"""
            UPDATE {backend._memories_table_name}
            SET {', '.join(updates)}
            WHERE id = :id {guard}
        """), params)
        conn.commit()
        return result.rowcount > 0


def backfill_meaning_fields(backend: Any, batch_size: int = 100) -> dict:
    """Backfill heuristic meaning fields for existing memories.
    
    Only populates stakes, emotional_charge, recurrence_keywords, and meaning_embedding.
    Does NOT populate intent - that requires conversation context and LLM inference.
    
    Returns:
        dict with 'updated' count and any 'errors'
    """
    import re
    from sqlalchemy import text
    from .embeddings import generate_embedding, meaning_text

    updated = 0
    errors = []
    
    def infer_stakes(importance: float) -> str:
        if importance >= 0.8:
            return "critical to remember; forgetting would harm trust or outcomes"
        if importance >= 0.6:
            return "important context for smooth interactions"
        return "nice-to-know; low stakes"
    
    def infer_emotion(content: str, importance: float) -> float:
        text_lower = content.lower()
        charge = 0.2 + 0.6 * importance
        keywords_high = ["love", "hate", "angry", "excited", "anxious", "worried", "sad"]
        if any(k in text_lower for k in keywords_high):
            charge = max(charge, 0.8)
        return min(1.0, max(0.0, charge))
    
    def infer_recurrence(content: str, tags: list[str]) -> list[str]:
        tag_list = [t for t in tags if t]
        words = re.findall(r"[a-zA-Z]{4,}", content.lower())
        return list(set(tag_list + words[:3]))

    with backend.engine.connect() as conn:
        # Fetch memories missing meaning fields (check stakes as proxy)
        sql = text(f"""
            SELECT id, content, tags, importance, intent
            FROM {backend._memories_table_name}
            WHERE stakes IS NULL
            LIMIT :limit
        """)
        rows = conn.execute(sql, {"limit": batch_size}).fetchall()
        conn.commit()  # Do not hold a transaction across embedding calls.

        for row in rows:
            try:
                tags = row.tags if isinstance(row.tags, list) else (json.loads(row.tags) if row.tags else ["misc"])
                importance = float(row.importance) if row.importance else 0.5
                content = row.content or ""
                existing_intent = row.intent  # Preserve any existing LLM-inferred intent
                
                # Only heuristic fields - intent stays as-is (NULL or LLM-provided)
                stakes = infer_stakes(importance)
                emotional_charge = infer_emotion(content, importance)
                recurrence_keywords = infer_recurrence(content, tags)
                
                body = meaning_text(
                    content, existing_intent, stakes, emotional_charge, recurrence_keywords,
                )
                try:
                    meaning_emb = generate_embedding(body, backend.embedding_model)
                except Exception as exc:
                    logger.warning("Could not backfill meaning embedding: %s", exc)
                    meaning_emb = None

                # A failed embed must leave NULL, not the vector for old fields.
                result = conn.execute(text(f"""
                    UPDATE {backend._memories_table_name}
                    SET stakes = :stakes,
                        emotional_charge = :emotional_charge,
                        recurrence_keywords = CAST(:recurrence_keywords AS jsonb),
                        meaning_embedding = :meaning_embedding,
                        meaning_updated_at = CASE WHEN :meaning_embedding IS NULL
                            THEN NULL ELSE CURRENT_TIMESTAMP END,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = :id AND content = :old_content
                      AND intent IS NOT DISTINCT FROM :old_intent AND stakes IS NULL
                """), {
                    "id": row.id, "old_content": row.content,
                    "old_intent": existing_intent, "stakes": stakes,
                    "emotional_charge": emotional_charge,
                    "recurrence_keywords": json.dumps(recurrence_keywords),
                    "meaning_embedding": (f"[{','.join(str(x) for x in meaning_emb)}]"
                                          if meaning_emb else None),
                })
                updated += result.rowcount
                conn.commit()
            except Exception as e:
                conn.rollback()
                errors.append(f"{row.id}: {e}")
    
    return {"updated": updated, "errors": errors, "remaining": len(rows) == batch_size}


def backfill_intent_with_llm(
    backend: Any,
    llm_client: Any,
    batch_size: int = 10,
) -> dict:
    """Backfill intent for memories by fetching original conversation and using LLM.
    
    This is an expensive operation - requires LLM calls for each memory.
    Uses source_message_ids to reconstruct the original conversation context.
    Memories whose source messages no longer exist are deleted (orphaned).
    
    Args:
        backend: PostgreSQLMemoryBackend instance
        llm_client: LLM client for intent inference
        batch_size: Number of memories to process per batch (keep small due to LLM cost)
    
    Returns:
        dict with 'updated', 'deleted', 'skipped' counts, and any 'errors'
    """
    from sqlalchemy import text
    
    updated = 0
    deleted = 0
    skipped = 0
    errors = []
    
    messages_table = backend._messages_table_name
    
    with backend.engine.connect() as conn:
        # First, delete orphaned memories (source messages no longer exist)
        orphan_sql = text(f"""
            WITH orphaned AS (
                SELECT m.id
                FROM {backend._memories_table_name} m
                WHERE m.intent IS NULL
                  AND m.source_message_ids IS NOT NULL
                  AND jsonb_array_length(m.source_message_ids) > 0
                  AND NOT EXISTS (
                      SELECT 1 FROM {messages_table} msg
                      WHERE msg.id = ANY(
                          SELECT jsonb_array_elements_text(m.source_message_ids)
                      )
                  )
            )
            DELETE FROM {backend._memories_table_name}
            WHERE id IN (SELECT id FROM orphaned)
            RETURNING id
        """)
        deleted_rows = conn.execute(orphan_sql).fetchall()
        deleted = len(deleted_rows)
        if deleted > 0:
            logger.debug(f"Deleted {deleted} orphaned memories (source messages no longer exist)")
        
        # Fetch all memories missing intent (with or without source_message_ids)
        sql = text(f"""
            SELECT m.id, m.content, m.source_message_ids
            FROM {backend._memories_table_name} m
            WHERE m.intent IS NULL
            LIMIT :limit
        """)
        rows = conn.execute(sql, {"limit": batch_size}).fetchall()
        
        for row in rows:
            try:
                # Parse source message IDs if present
                source_ids = row.source_message_ids
                if source_ids and isinstance(source_ids, str):
                    source_ids = json.loads(source_ids)
                
                # Fetch the original messages if we have source IDs
                messages = []
                if source_ids:
                    msg_sql = text(f"""
                        SELECT role, content
                        FROM {messages_table}
                        WHERE id = ANY(:ids)
                        ORDER BY timestamp ASC
                    """)
                    messages = conn.execute(msg_sql, {"ids": source_ids}).fetchall()
                
                # Format conversation if messages exist, otherwise use content-only
                if messages:
                    conversation = "\n".join([
                        f"{msg.role.capitalize()}: {msg.content}"
                        for msg in messages
                    ])
                    prompt = get_prompt_resolver().render(
                        key="memory.maintenance.intent_with_context",
                        variables={
                            "conversation": conversation[:2000],
                            "fact": row.content,
                        },
                    )
                else:
                    # Fallback to content-only inference
                    prompt = get_prompt_resolver().render(
                        key="memory.maintenance.intent_content_only",
                        variables={"fact": row.content},
                    )
                
                # Query LLM for intent
                from ..models.message import Message
                
                system_msg = Message(
                    role="system",
                    content=get_prompt_resolver().render(
                        key="memory.maintenance.intent_system", variables={},
                    ),
                )
                user_msg = Message(role="user", content=prompt)
                
                response = llm_client.query(
                    messages=[system_msg, user_msg],
                    plaintext_output=True,
                    stream=False,
                )
                
                # Clean up response
                intent = response.strip().strip('"').strip("'").lower()
                if len(intent) > 50:  # Sanity check
                    intent = intent[:50]
                
                # The old meaning vector describes the old intent. Leave it
                # NULL for the bounded repair, never searchable as stale data.
                update_sql = text(f"""
                    UPDATE {backend._memories_table_name}
                    SET intent = :intent, meaning_embedding = NULL,
                        meaning_updated_at = NULL, updated_at = CURRENT_TIMESTAMP
                    WHERE id = :id AND intent IS NULL AND content = :old_content
                """)
                result = conn.execute(update_sql, {
                    "id": row.id, "intent": intent, "old_content": row.content,
                })
                updated += result.rowcount
                
            except Exception as e:
                errors.append(f"{row.id}: {e}")
        
        conn.commit()
    
    remaining = len(rows) == batch_size
    return {"updated": updated, "deleted": deleted, "skipped": skipped, "errors": errors, "remaining": remaining}
