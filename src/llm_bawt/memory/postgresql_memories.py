"""Semantic-memory operations for the PostgreSQL memory backend.

This module owns distilled-memory CRUD, retrieval, ranking, statistics, and
embedding maintenance. Engine and schema construction remain in postgresql.py.
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone

from sqlalchemy import delete, select, text
from sqlalchemy.orm import Session

from llm_bawt.bot_access import BotAccessError
from llm_bawt.bot_access.provenance import MemoryProvenance
from llm_bawt.memory.errors import MemoryUpdateConflictError, MemoryUpdateDuplicateError

logger = logging.getLogger(__name__)


class PostgreSQLSemanticMemoryMixin:
    """Distilled semantic-memory behavior for PostgreSQLMemoryBackend."""

    def add_memory(
        self,
        memory_id: str,
        content: str,
        tags: list[str] | None = None,
        importance: float = 0.5,
        source_message_ids: list[str] | None = None,
        embedding: list[float] | None = None,
        intent: str | None = None,
        stakes: str | None = None,
        emotional_charge: float | None = None,
        recurrence_keywords: list[str] | None = None,
        meaning_embedding: list[float] | None = None,
        meaning_updated_at: datetime | None = None,
    ) -> None:
        """Add a distilled memory to storage.
        
        If no embedding is provided, one will be generated automatically
        using the configured local embedding model.
        """
        if not content or content.isspace():
            logger.warning(f"Skipping empty content for memory ID: {memory_id}")
            return
        
        # Normalize tags
        tags = tags or ["misc"]
        tags = [t.strip().lower() for t in tags if isinstance(t, str) and t.strip()]
        if not tags:
            tags = ["misc"]
        
        # Generate embeddings if not provided
        if embedding is None:
            try:
                from .embeddings import generate_embedding
                embedding = generate_embedding(content, self.embedding_model, verbose=getattr(self.config, 'VERBOSE', False))
                if embedding:
                    logger.debug(f"Generated embedding for memory: {content[:50]}...")
            except Exception as e:
                logger.debug(f"Could not generate embedding: {e}")
        if meaning_embedding is None:
            try:
                from .embeddings import generate_embedding, meaning_text
                meaning_embedding = generate_embedding(
                    meaning_text(content, intent, stakes, emotional_charge, recurrence_keywords),
                    self.embedding_model, verbose=getattr(self.config, 'VERBOSE', False),
                )
            except Exception as e:
                logger.debug(f"Could not generate meaning embedding: {e}")
        if meaning_embedding:
            meaning_updated_at = meaning_updated_at or datetime.now(timezone.utc)
        
        with self.engine.connect() as conn:
            try:
                # Check for duplicate/similar content BEFORE inserting
                # Use embedding similarity if available, otherwise exact content match
                # Threshold from config (default 0.85) - catches semantic duplicates
                dedup_threshold = getattr(self.config, 'MEMORY_DEDUP_SIMILARITY', 0.85)
                
                if embedding:
                    # Vector similarity check - find most similar existing memory
                    dedup_sql = text(f"""
                        SELECT id, content, 1 - (embedding <=> :embedding) AS similarity
                        FROM {self._memories_table_name}
                        WHERE embedding IS NOT NULL
                        ORDER BY embedding <=> :embedding
                        LIMIT 1
                    """)
                    similar = conn.execute(dedup_sql, {
                        "embedding": f"[{','.join(str(x) for x in embedding)}]",
                    }).first()
                    
                    if similar and similar.similarity > dedup_threshold:
                        logger.debug(f"Skipping duplicate memory (similarity={similar.similarity:.2%} > {dedup_threshold:.2%}): '{content[:50]}...' similar to '{similar.content[:50]}...'")
                        return
                else:
                    # Exact content match fallback
                    exact_sql = text(f"""
                        SELECT id FROM {self._memories_table_name} 
                        WHERE LOWER(content) = LOWER(:content)
                        LIMIT 1
                    """)
                    exact_match = conn.execute(exact_sql, {"content": content}).first()
                    if exact_match:
                        logger.debug(f"Skipping exact duplicate memory: '{content[:50]}...'")
                        return
                
                # Check if this specific ID exists (for updates)
                check_sql = text(f"""
                    SELECT id FROM {self._memories_table_name} WHERE id = :id
                """)
                existing = conn.execute(check_sql, {"id": memory_id}).first()
                
                if existing:
                    # Update existing
                    update_sql = text(f"""
                        UPDATE {self._memories_table_name}
                        SET content = :content,
                            tags = CAST(:tags AS jsonb),
                            importance = :importance,
                            source_message_ids = :source_ids,
                            embedding = CASE WHEN content IS DISTINCT FROM :content
                                THEN :embedding ELSE COALESCE(:embedding, embedding) END,
                            intent = :intent,
                            stakes = :stakes,
                            emotional_charge = :emotional_charge,
                            recurrence_keywords = CAST(:recurrence_keywords AS jsonb),
                            meaning_embedding = CASE WHEN
                                content IS DISTINCT FROM :content
                                OR intent IS DISTINCT FROM :intent
                                OR stakes IS DISTINCT FROM :stakes
                                OR emotional_charge IS DISTINCT FROM :emotional_charge
                                OR recurrence_keywords IS DISTINCT FROM CAST(:recurrence_keywords AS jsonb)
                                THEN :meaning_embedding
                                ELSE COALESCE(:meaning_embedding, meaning_embedding) END,
                            meaning_updated_at = CASE WHEN
                                content IS DISTINCT FROM :content
                                OR intent IS DISTINCT FROM :intent
                                OR stakes IS DISTINCT FROM :stakes
                                OR emotional_charge IS DISTINCT FROM :emotional_charge
                                OR recurrence_keywords IS DISTINCT FROM CAST(:recurrence_keywords AS jsonb)
                                THEN :meaning_updated_at
                                ELSE COALESCE(:meaning_updated_at, meaning_updated_at) END,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE id = :id
                    """)
                    conn.execute(update_sql, {
                        "id": memory_id,
                        "content": content,
                        "tags": json.dumps(tags),
                        "importance": importance,
                        "source_ids": json.dumps(source_message_ids or []),
                        "embedding": f"[{','.join(str(x) for x in embedding)}]" if embedding else None,
                        "intent": intent,
                        "stakes": stakes,
                        "emotional_charge": emotional_charge,
                        "recurrence_keywords": json.dumps(recurrence_keywords or []),
                        "meaning_embedding": f"[{','.join(str(x) for x in meaning_embedding)}]" if meaning_embedding else None,
                        "meaning_updated_at": meaning_updated_at,
                    })
                else:
                    # Insert new
                    insert_sql = text(f"""
                        INSERT INTO {self._memories_table_name}
                        (id, content, tags, importance, source_message_ids, embedding,
                         intent, stakes, emotional_charge, recurrence_keywords, meaning_embedding, meaning_updated_at, created_at)
                        VALUES (:id, :content, CAST(:tags AS jsonb), :importance, :source_ids, :embedding,
                                :intent, :stakes, :emotional_charge, CAST(:recurrence_keywords AS jsonb), :meaning_embedding, :meaning_updated_at, CURRENT_TIMESTAMP)
                    """)
                    conn.execute(insert_sql, {
                        "id": memory_id,
                        "content": content,
                        "tags": json.dumps(tags),
                        "importance": importance,
                        "source_ids": json.dumps(source_message_ids or []),
                        "embedding": f"[{','.join(str(x) for x in embedding)}]" if embedding else None,
                        "intent": intent,
                        "stakes": stakes,
                        "emotional_charge": emotional_charge,
                        "recurrence_keywords": json.dumps(recurrence_keywords or []),
                        "meaning_embedding": f"[{','.join(str(x) for x in meaning_embedding)}]" if meaning_embedding else None,
                        "meaning_updated_at": meaning_updated_at,
                    })
                
                conn.commit()
                logger.debug(f"Added memory {memory_id} to {self._memories_table_name}")
            except Exception as e:
                conn.rollback()
                logger.error(f"Failed to add memory {memory_id}: {e}")

    def update_memory(
        self,
        memory_id: str,
        *,
        content: str | None = None,
        tags: list[str] | None = None,
        importance: float | None = None,
        source_message_ids: list[str] | None = None,
    ) -> dict | None:
        """Patch an existing row, excluding itself from duplicate detection.

        Do not hold a database lock across embedding HTTP calls. A stale snapshot
        fails visibly instead of overwriting a newer edit or returning invented
        state. An unavailable embedder clears vectors only when content changes.
        """
        with self.engine.connect() as conn:
            original = conn.execute(text(f"""
                SELECT content, intent, stakes, emotional_charge,
                       recurrence_keywords, updated_at
                FROM {self._memories_table_name} WHERE id = :id
            """), {"id": memory_id}).mappings().first()
        if original is None:
            return None

        changed = content is not None and content != original["content"]
        if content is not None and not content.strip():
            raise ValueError("memory content cannot be blank")
        if tags is not None:
            tags = [tag.strip().lower() for tag in tags
                    if isinstance(tag, str) and tag.strip()] or ["misc"]
        embedding = meaning_embedding = None
        if changed:
            from .embeddings import generate_embedding, meaning_text

            try:
                embedding = generate_embedding(content, self.embedding_model)
            except Exception as exc:
                logger.warning("Could not regenerate content embedding: %s", exc)
            try:
                meaning_embedding = generate_embedding(
                    meaning_text(content, original["intent"], original["stakes"],
                                 original["emotional_charge"],
                                 original["recurrence_keywords"]),
                    self.embedding_model,
                )
            except Exception as exc:
                logger.warning("Could not regenerate meaning embedding: %s", exc)

        params = {
            "id": memory_id, "old_content": original["content"],
            "old_updated_at": original["updated_at"],
            "content": content, "changed": changed,
            "tags_provided": tags is not None,
            "tags": json.dumps(tags) if tags is not None else None,
            "importance": importance,
            "sources_provided": source_message_ids is not None,
            "sources": (json.dumps(source_message_ids)
                        if source_message_ids is not None else None),
            "embedding": (f"[{','.join(str(x) for x in embedding)}]"
                          if embedding else None),
            "meaning_embedding": (f"[{','.join(str(x) for x in meaning_embedding)}]"
                                  if meaning_embedding else None),
        }
        with self.engine.begin() as conn:
            if changed:
                threshold = getattr(self.config, "MEMORY_DEDUP_SIMILARITY", 0.85)
                # A prior embedding outage leaves NULL vectors. Exact matches
                # still count even when this edit has a valid new embedding.
                exact = conn.execute(text(f"""
                    SELECT id FROM {self._memories_table_name}
                    WHERE id <> :id AND LOWER(content) = LOWER(:content) LIMIT 1
                """), params).first()
                is_duplicate = exact is not None
                if not is_duplicate and embedding:
                    similar = conn.execute(text(f"""
                        SELECT id, 1 - (embedding <=> :embedding) AS similarity
                        FROM {self._memories_table_name}
                        WHERE id <> :id AND embedding IS NOT NULL
                        ORDER BY embedding <=> :embedding LIMIT 1
                    """), params).first()
                    is_duplicate = similar is not None and similar.similarity > threshold
                if is_duplicate:
                    raise MemoryUpdateDuplicateError("memory update conflicts with another duplicate")

            updated = conn.execute(text(f"""
                UPDATE {self._memories_table_name}
                SET content = COALESCE(:content, content),
                    tags = CASE WHEN :tags_provided THEN CAST(:tags AS jsonb) ELSE tags END,
                    importance = COALESCE(:importance, importance),
                    source_message_ids = CASE WHEN :sources_provided
                        THEN CAST(:sources AS jsonb) ELSE source_message_ids END,
                    embedding = CASE WHEN :changed THEN :embedding ELSE embedding END,
                    meaning_embedding = CASE WHEN :changed
                        THEN :meaning_embedding ELSE meaning_embedding END,
                    meaning_updated_at = CASE WHEN :changed
                        THEN CASE WHEN :meaning_embedding IS NULL THEN NULL
                                  ELSE CURRENT_TIMESTAMP END
                        ELSE meaning_updated_at END,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = :id AND content = :old_content
                  AND updated_at IS NOT DISTINCT FROM :old_updated_at
                RETURNING id, content, tags, importance, source_message_ids,
                          access_count, created_at, last_accessed,
                          intent, stakes, emotional_charge
            """), params).mappings().first()
            if updated is None:
                exists = conn.execute(text(f"""
                    SELECT 1 FROM {self._memories_table_name} WHERE id = :id
                """), params).first()
                if exists is not None:
                    raise MemoryUpdateConflictError("memory changed concurrently; retry the update")
                return None
            return dict(updated)

    def delete_memory(self, memory_id: str) -> bool:
        """Delete a specific memory.
        
        Supports both full UUID and prefix matching (first 8 chars).
        """
        with Session(self.engine) as session:
            try:
                # If it's a short ID (prefix), use LIKE matching
                if len(memory_id) < 36:  # Full UUID is 36 chars
                    stmt = delete(self.memories_table).where(
                        self.memories_table.c.id.like(f"{memory_id}%")
                    )
                else:
                    stmt = delete(self.memories_table).where(
                        self.memories_table.c.id == memory_id
                    )
                result = session.execute(stmt)
                session.commit()
                return result.rowcount > 0
            except Exception as e:
                session.rollback()
                logger.error(f"Failed to delete memory {memory_id}: {e}")
                return False

    def delete_memories_by_source_message_ids(self, message_ids: list[str]) -> int:
        """Delete all memories whose source_message_ids contain any of the given message IDs.
        
        Args:
            message_ids: List of message UUIDs to match against source_message_ids.
            
        Returns:
            Number of memories deleted.
        """
        if not message_ids:
            return 0
            
        with self.engine.connect() as conn:
            try:
                # Use JSONB containment: source_message_ids ?| array[...] checks if any element matches
                sql = text(f"""
                    DELETE FROM {self._memories_table_name}
                    WHERE source_message_ids ?| :ids
                    RETURNING id
                """)
                result = conn.execute(sql, {"ids": message_ids})
                deleted_ids = [row.id for row in result.fetchall()]
                conn.commit()
                
                if deleted_ids:
                    logger.debug(f"Deleted {len(deleted_ids)} memories associated with forgotten messages")
                return len(deleted_ids)
            except Exception as e:
                conn.rollback()
                logger.error(f"Failed to delete memories by source message IDs: {e}")
                return 0

    def update_memory_access(self, memory_id: str) -> None:
        """Update access tracking for a memory (reinforcement)."""
        with self.engine.connect() as conn:
            try:
                sql = text(f"""
                    UPDATE {self._memories_table_name}
                    SET access_count = access_count + 1,
                        last_accessed = CURRENT_TIMESTAMP
                    WHERE id = :id
                """)
                conn.execute(sql, {"id": memory_id})
                conn.commit()
            except Exception as e:
                logger.error(f"Failed to update memory access: {e}")

    def search_memories_by_text(
        self,
        query: str,
        n_results: int = 5,
        min_importance: float = 0.0,
        tags: list[str] | None = None,
    ) -> list[dict]:
        """Search memories using PostgreSQL full-text search."""
        if not query or query.isspace():
            return []
        
        with self.engine.connect() as conn:
            try:
                # Build the query with optional filters
                tag_filter = ""
                if tags:
                    tag_filter = "AND tags ?| CAST(:filter_tags AS text[])"
                
                # Use PostgreSQL full-text search
                sql = text(f"""
                          SELECT id, content, tags, importance, source_message_ids,
                           access_count, last_accessed, created_at,
                           intent, stakes, emotional_charge,
                           ts_rank(to_tsvector('english', content), plainto_tsquery('english', :query)) AS rank
                    FROM {self._memories_table_name}
                    WHERE to_tsvector('english', content) @@ plainto_tsquery('english', :query)
                    AND importance >= :min_importance
                    {tag_filter}
                    ORDER BY rank DESC, importance DESC
                    LIMIT :limit
                """)
                
                rows = conn.execute(sql, {
                    "query": query,
                    "min_importance": min_importance,
                    "limit": n_results,
                    "filter_tags": tags,
                }).fetchall()
                
                results = []
                for row in rows:
                    # Update access tracking
                    self.update_memory_access(row.id)
                    
                    row_tags = row.tags if isinstance(row.tags, list) else (json.loads(row.tags) if row.tags else ["misc"])
                    results.append({
                        "id": row.id,
                        "content": row.content,
                        "tags": row_tags,
                        "importance": row.importance,
                        "source_message_ids": row.source_message_ids or [],
                        "access_count": row.access_count,
                        "relevance": row.rank,
                        "intent": row.intent,
                        "stakes": row.stakes,
                        "emotional_charge": row.emotional_charge,
                    })
                
                return results
                
            except Exception as e:
                logger.error(f"Failed to search memories: {e}")
                return []

    def search_memories_by_embedding(
        self,
        embedding: list[float],
        n_results: int = 5,
        min_importance: float = 0.0,
        tags: list[str] | None = None,
        meaning_embedding: list[float] | None = None,
        meaning_weight: float | None = None,
        *,
        eligibility: MemoryProvenance | None = None,
    ) -> list[dict]:
        """Search memories using vector similarity with temporal decay and diversity.
        
        The effective score combines:
        - Semantic similarity (cosine distance)
        - Base importance
        - Temporal decay (memories fade over time)
        - Access boost (frequently accessed memories get reinforced)
        - Diversity sampling (avoid echo chambers by sampling across time/types)
        """
        if not embedding:
            return []
        
        # Get decay settings from config
        decay_enabled = getattr(self.config, 'MEMORY_DECAY_ENABLED', True)
        half_life_days = getattr(self.config, 'MEMORY_DECAY_HALF_LIFE_DAYS', 90.0)
        access_boost_factor = getattr(self.config, 'MEMORY_ACCESS_BOOST_FACTOR', 0.15)
        recency_weight = getattr(self.config, 'MEMORY_RECENCY_WEIGHT', 0.3)
        diversity_enabled = getattr(self.config, 'MEMORY_DIVERSITY_ENABLED', True)
        
        # Different decay rates per tag (multiplier on half_life)
        # Higher = slower decay (more persistent)
        tag_decay_multipliers = {
            'fact': 2.0,          # Core facts persist longer
            'professional': 1.5,  # Career info moderately persistent
            'preference': 0.8,    # Preferences change
            'health': 1.2,        # Health info somewhat persistent
            'relationship': 1.0,  # Relationships change at normal rate
            'event': 0.5,         # Events become less relevant quickly
            'plan': 0.3,          # Plans/goals are very temporal
        }
        
        with self.engine.connect() as conn:
            try:
                tag_filter = ""
                if tags:
                    tag_filter = "AND tags ?| CAST(:filter_tags AS text[])"
                
                if decay_enabled:
                    # Fetch more candidates for post-processing with decay + diversity
                    fetch_limit = n_results * 4 if diversity_enabled else n_results * 2
                    
                    sql = text(f"""
                           SELECT id, content, tags, importance, source_message_ids,
                               access_count, last_accessed, created_at,
                               intent, stakes, emotional_charge,
                               1 - (embedding <=> :embedding) AS similarity,
                               CASE WHEN :use_meaning THEN 1 - (meaning_embedding <=> :meaning_embedding) ELSE NULL END AS meaning_similarity,
                               EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - created_at)) / 86400.0 AS age_days
                        FROM {self._memories_table_name} mem
                        WHERE embedding IS NOT NULL
                        AND importance >= :min_importance
                        {f'AND ({eligibility.predicate})' if eligibility is not None else ''}
                        {tag_filter}
                        ORDER BY embedding <=> :embedding
                        LIMIT :limit
                    """)
                else:
                    # Simple similarity-based search without decay
                    sql = text(f"""
                           SELECT id, content, tags, importance, source_message_ids,
                               access_count, last_accessed, created_at,
                               intent, stakes, emotional_charge,
                               1 - (embedding <=> :embedding) AS similarity,
                               CASE WHEN :use_meaning THEN 1 - (meaning_embedding <=> :meaning_embedding) ELSE NULL END AS meaning_similarity,
                               0 AS age_days
                        FROM {self._memories_table_name} mem
                        WHERE embedding IS NOT NULL
                        AND importance >= :min_importance
                        {f'AND ({eligibility.predicate})' if eligibility is not None else ''}
                        {tag_filter}
                        ORDER BY embedding <=> :embedding
                        LIMIT :limit
                    """)
                    fetch_limit = n_results
                
                rows = conn.execute(sql, {
                    "embedding": str(embedding),
                    "meaning_embedding": str(meaning_embedding) if meaning_embedding else str(embedding),
                    "use_meaning": bool(meaning_embedding),
                    "min_importance": min_importance,
                    "limit": fetch_limit,
                    "filter_tags": tags,
                    **(eligibility.params if eligibility is not None else {}),
                }).fetchall()
                
                if not rows:
                    return []
                
                # Calculate effective scores with decay
                import math
                scored_results = []
                
                meaning_w = meaning_weight if meaning_weight is not None else getattr(self.config, 'MEMORY_MEANING_WEIGHT', 0.3)
                meaning_w = max(0.0, min(1.0, meaning_w))
                for row in rows:
                    # Convert Decimal types to float for calculations
                    similarity = float(row.similarity) if row.similarity else 0.0
                    meaning_similarity = float(row.meaning_similarity) if meaning_embedding and row.meaning_similarity is not None else None
                    importance = float(row.importance) if row.importance else 0.5
                    age_days = float(row.age_days) if row.age_days else 0.0
                    access_count = int(row.access_count) if row.access_count else 0
                    
                    # Parse tags
                    row_tags = row.tags if isinstance(row.tags, list) else (json.loads(row.tags) if row.tags else ["misc"])
                    
                    # Get tag-specific half-life (use first tag for decay rate)
                    primary_tag = row_tags[0] if row_tags else "misc"
                    tag_multiplier = tag_decay_multipliers.get(primary_tag, 1.0)
                    effective_half_life = half_life_days * tag_multiplier
                    
                    # Temporal decay: exp(-age * ln(2) / half_life)
                    if decay_enabled and age_days > 0:
                        decay_factor = math.exp(-age_days * math.log(2) / effective_half_life)
                    else:
                        decay_factor = 1.0
                    
                    # Access boost: 1 + factor * log(access_count + 1)
                    access_boost = 1.0 + access_boost_factor * math.log(access_count + 1)
                    
                    # Combine semantic similarities
                    sim_combined = similarity
                    if meaning_similarity is not None:
                        sim_combined = (1 - meaning_w) * similarity + meaning_w * meaning_similarity
                    
                    # Combined score:
                    # similarity provides base relevance
                    # importance is the extracted importance
                    # decay_factor reduces old memories
                    # access_boost reinforces frequently used ones
                    # recency_weight balances recency vs base importance
                    base_score = sim_combined * importance
                    recency_score = sim_combined * decay_factor
                    effective_score = (
                        (1 - recency_weight) * base_score + 
                        recency_weight * recency_score
                    ) * access_boost
                    
                    scored_results.append({
                        "id": row.id,
                        "content": row.content,
                        "tags": row_tags,
                        "importance": importance,
                        "source_message_ids": row.source_message_ids or [],
                        "access_count": access_count,
                        "created_at": row.created_at,
                        "last_accessed": row.last_accessed,
                        "similarity": similarity,
                        "meaning_similarity": meaning_similarity,
                        "age_days": age_days,
                        "decay_factor": decay_factor,
                        "effective_score": effective_score,
                        "intent": row.intent,
                        "stakes": row.stakes,
                        "emotional_charge": float(row.emotional_charge) if row.emotional_charge else None,
                    })
                
                # Sort by effective score
                scored_results.sort(key=lambda x: x["effective_score"], reverse=True)
                
                # Apply diversity sampling if enabled
                if diversity_enabled and len(scored_results) > n_results:
                    results = self._diversity_sample(scored_results, n_results)
                else:
                    results = scored_results[:n_results]
                
                # Update access counts for retrieved memories
                for r in results:
                    self.update_memory_access(r["id"])
                
                return results
                
            except Exception as e:
                if eligibility is not None:
                    raise BotAccessError("source_unavailable") from None
                logger.error(f"Failed to search by embedding: {e}")
                return []

    def _diversity_sample(self, candidates: list[dict], n_results: int) -> list[dict]:
        """Sample diverse memories to avoid echo chambers.
        
        Strategy:
        1. Always include top-scoring result
        2. Ensure representation from different memory types
        3. Ensure representation from different time periods
        4. Fill remaining slots by score
        """
        if len(candidates) <= n_results:
            return candidates
        
        selected = []
        used_ids = set()
        
        # 1. Always take the top result
        selected.append(candidates[0])
        used_ids.add(candidates[0]["id"])
        
        # 2. Ensure tag diversity - try to get one from each represented tag set
        tags_seen = set(candidates[0].get("tags", ["misc"]))
        for candidate in candidates[1:]:
            if len(selected) >= n_results:
                break
            candidate_tags = set(candidate.get("tags", ["misc"]))
            if not candidate_tags.issubset(tags_seen) and candidate["id"] not in used_ids:
                selected.append(candidate)
                used_ids.add(candidate["id"])
                tags_seen.update(candidate_tags)
        
        # 3. Ensure temporal diversity - split into time buckets
        if len(selected) < n_results:
            # Recent (< 7 days), Medium (7-30 days), Old (30+ days)
            buckets = {"recent": [], "medium": [], "old": []}
            for candidate in candidates:
                if candidate["id"] in used_ids:
                    continue
                age = candidate.get("age_days", 0)
                if age < 7:
                    buckets["recent"].append(candidate)
                elif age < 30:
                    buckets["medium"].append(candidate)
                else:
                    buckets["old"].append(candidate)
            
            # Try to get one from each bucket we haven't covered
            for bucket_name in ["medium", "old", "recent"]:  # Prioritize less-recent
                if len(selected) >= n_results:
                    break
                bucket = buckets[bucket_name]
                if bucket:
                    candidate = bucket[0]
                    selected.append(candidate)
                    used_ids.add(candidate["id"])
        
        # 4. Fill remaining by score
        for candidate in candidates:
            if len(selected) >= n_results:
                break
            if candidate["id"] not in used_ids:
                selected.append(candidate)
                used_ids.add(candidate["id"])
        
        # Re-sort by effective_score for consistent ordering
        selected.sort(key=lambda x: x["effective_score"], reverse=True)
        
        return selected

    def search(self, query: str, n_results: int = 5, min_relevance: float = 0.0) -> list[dict] | None:
        """Search for relevant memories (implements MemoryBackend interface).
        
        Search strategy:
        1. Text search on memories (fast, keyword matching) - if good matches found, use them
        2. Embedding search + high-importance blend - semantic search combined with important facts
        3. Raw messages fallback (last resort)
        
        For most queries, we blend embedding search results with high-importance core facts
        (like user's name) to ensure both relevance and foundational context.
        """
        # Minimum relevance threshold for text search to be considered "good"
        TEXT_SEARCH_MIN_RELEVANCE = 0.1
        
        def format_memory_result(r: dict) -> dict:
            return {
                "id": r["id"],
                "document": r["content"],
                "content": r["content"],  # Keep both for compatibility
                "metadata": {
                    "tags": r.get("tags", ["misc"]),
                    "importance": r["importance"],
                    "source_message_ids": r.get("source_message_ids", []),
                },
                "relevance": r.get("similarity", r.get("relevance", r["importance"])),
                # Meaning fields for structured context
                "intent": r.get("intent"),
                "stakes": r.get("stakes"),
                "emotional_charge": r.get("emotional_charge"),
                "tags": r.get("tags", ["misc"]),
                "importance": r.get("importance", 0.5),
            }
        
        # 1. Try text search on memories - but only accept high-quality matches
        text_results = self.search_memories_by_text(query, n_results, min_importance=min_relevance)
        if text_results:
            # Check if any result has good relevance (not just keyword matching noise)
            best_relevance = max(r.get("relevance", 0) for r in text_results)
            if best_relevance >= TEXT_SEARCH_MIN_RELEVANCE:
                logger.debug(f"Found {len(text_results)} memories via text search (best relevance: {best_relevance:.3f})")
                return [format_memory_result(r) for r in text_results]
            else:
                logger.debug(f"Text search results too weak (best: {best_relevance:.3f}), trying other methods")
        
        # 2. Blend embedding search with high-importance core facts
        combined_results = []
        seen_ids = set()
        
        # Get embedding search results (query-relevant)
        if self.embedding_model:
            try:
                from .embeddings import generate_embedding
                query_embedding = generate_embedding(query, self.embedding_model, verbose=getattr(self.config, 'VERBOSE', False))
                if query_embedding:
                    embedding_results = self.search_memories_by_embedding(query_embedding, n_results, min_importance=min_relevance)
                    if embedding_results:
                        logger.debug(f"Found {len(embedding_results)} memories via embedding search")
                        for r in embedding_results:
                            if r["id"] not in seen_ids:
                                combined_results.append(format_memory_result(r))
                                seen_ids.add(r["id"])
            except Exception as e:
                logger.debug(f"Embedding search failed: {e}")
        
        # Add high-importance core facts (limit to a few to avoid overwhelming)
        # These are foundational facts like name, profession that should always be included
        high_importance_results = self.get_high_importance_memories(3, min_importance=0.9)
        for r in high_importance_results:
            if r["id"] not in seen_ids:
                combined_results.append(format_memory_result(r))
                seen_ids.add(r["id"])
        
        if combined_results:
            logger.debug(f"Returning {len(combined_results)} blended results (embedding + core facts)")
            return combined_results[:n_results]
        
        # 3. Last resort: search raw messages
        message_results = self.search_messages_by_text(query, n_results, exclude_recent_seconds=10.0)
        
        if not message_results:
            return None
        
        logger.debug(f"Falling back to {len(message_results)} message results")
        
        # Format message results like memory results
        return [
            {
                "id": r["id"],
                "document": r["content"],
                "metadata": {
                    "role": r["role"],
                    "timestamp": r["timestamp"],
                },
                "relevance": r.get("relevance", 0.5),
            }
            for r in message_results
        ]

    def get_high_importance_memories(self, n_results: int = 10, min_importance: float = 0.7) -> list[dict]:
        """Get the most important memories, regardless of query matching.
        
        Useful as a fallback for general queries like "what do you know about me?"
        """
        with self.engine.connect() as conn:
            try:
                sql = text(f"""
                    SELECT id, content, tags, importance, source_message_ids,
                           access_count, last_accessed, created_at,
                           intent, stakes, emotional_charge
                    FROM {self._memories_table_name}
                    WHERE importance >= :min_importance
                    ORDER BY importance DESC, access_count DESC, created_at DESC
                    LIMIT :limit
                """)
                
                rows = conn.execute(sql, {
                    "min_importance": min_importance,
                    "limit": n_results,
                }).fetchall()
                
                return [
                    {
                        "id": row.id,
                        "content": row.content,
                        "tags": row.tags if isinstance(row.tags, list) else (json.loads(row.tags) if row.tags else ["misc"]),
                        "importance": row.importance,
                        "source_message_ids": row.source_message_ids or [],
                        "relevance": row.importance,  # Use importance as relevance for fallback
                        "intent": row.intent,
                        "stakes": row.stakes,
                        "emotional_charge": float(row.emotional_charge) if row.emotional_charge else None,
                    }
                    for row in rows
                ]
                
            except Exception as e:
                logger.error(f"Failed to get high-importance memories: {e}")
                return []

    def clear(self) -> bool:
        """Clear all memories (NOT messages - those are permanent)."""
        with Session(self.engine) as session:
            try:
                stmt = delete(self.memories_table)
                session.execute(stmt)
                session.commit()
                logger.debug(f"Cleared all memories from {self._memories_table_name}")
                return True
            except Exception as e:
                session.rollback()
                logger.error(f"Failed to clear memories: {e}")
                return False

    def list_recent(self, n: int = 10) -> list[dict]:
        """List the most recent memories."""
        with Session(self.engine) as session:
            stmt = (
                select(self.memories_table)
                .order_by(self.memories_table.c.created_at.desc())
                .limit(n)
            )
            rows = session.execute(stmt).fetchall()
            
            return [
                {
                    "id": row.id,
                    "document": row.content,
                    "metadata": {
                        "tags": row.tags if isinstance(row.tags, list) else (json.loads(row.tags) if row.tags else ["misc"]),
                        "importance": row.importance,
                        "source_message_ids": row.source_message_ids or [],
                    },
                }
                for row in rows
            ]

    def stats(self) -> dict:
        """Get statistics about memory storage."""
        with self.engine.connect() as conn:
            try:
                # Memory stats
                mem_stats_sql = text(f"""
                    SELECT 
                        COUNT(*) as total,
                        MIN(created_at) as oldest,
                        MAX(created_at) as newest,
                        AVG(importance) as avg_importance,
                        COUNT(*) FILTER (WHERE embedding IS NULL) AS missing_content_embeddings,
                        COUNT(*) FILTER (WHERE meaning_embedding IS NULL) AS missing_meaning_embeddings,
                        COUNT(*) FILTER (WHERE embedding IS NULL OR meaning_embedding IS NULL) AS missing_any_embeddings,
                        pg_total_relation_size('{self._memories_table_name}') as size_bytes
                    FROM {self._memories_table_name}
                """)
                mem_row = conn.execute(mem_stats_sql).first()
                
                # Message stats
                msg_stats_sql = text(f"""
                    SELECT 
                        COUNT(*) as total,
                        SUM(CASE WHEN processed THEN 1 ELSE 0 END) as processed,
                        MIN(created_at) as oldest,
                        MAX(created_at) as newest
                    FROM {self._messages_table_name}
                """)
                msg_row = conn.execute(msg_stats_sql).first()
                
                # Forgotten message count
                forgotten_count = 0
                try:
                    forgotten_sql = text(f"SELECT COUNT(*) FROM {self._forgotten_table_name}")
                    forgotten_count = conn.execute(forgotten_sql).scalar() or 0
                except Exception:
                    pass  # Table may not exist yet
                
                return {
                    "memories": {
                        "total_count": mem_row.total if mem_row else 0,
                        "oldest_timestamp": mem_row.oldest.isoformat() if mem_row and mem_row.oldest else None,
                        "newest_timestamp": mem_row.newest.isoformat() if mem_row and mem_row.newest else None,
                        "avg_importance": float(mem_row.avg_importance) if mem_row and mem_row.avg_importance else 0.0,
                        "storage_size_bytes": mem_row.size_bytes if mem_row else 0,
                        "missing_content_embeddings": mem_row.missing_content_embeddings if mem_row else 0,
                        "missing_meaning_embeddings": mem_row.missing_meaning_embeddings if mem_row else 0,
                        "missing_any_embeddings": mem_row.missing_any_embeddings if mem_row else 0,
                    },
                    "messages": {
                        "total_count": msg_row.total if msg_row else 0,
                        "processed_count": msg_row.processed if msg_row else 0,
                        "forgotten_count": forgotten_count,
                        "oldest_timestamp": msg_row.oldest.isoformat() if msg_row and msg_row.oldest else None,
                        "newest_timestamp": msg_row.newest.isoformat() if msg_row and msg_row.newest else None,
                    },
                }
            except Exception as e:
                logger.error(f"Failed to get stats: {e}")
                return {"error": str(e)}

    def regenerate_embeddings(self, batch_size: int = 50) -> dict:
        """Repair both missing vectors in a bounded pass; never alter schema.

        This manual entry point deliberately refuses private and inactive bots,
        just like the recurring repair. A subsequent call continues at the saved
        cursor; no permanently bad row can stall the rest of the queue.
        """
        from .embedding_repair import EmbeddingRepair

        repaired = EmbeddingRepair(self.config, self.engine).repair_bot(
            self.bot_id, limit=min(batch_size, 25),
        )
        if repaired.get("excluded") or repaired["busy"] or repaired["dimension_mismatch"]:
            reason = ("bot excluded from embedding repair" if repaired.get("excluded") else
                      "embedding repair already running" if repaired["busy"] else
                      "embedding dimension mismatch; explicit schema migration required")
            return {"error": reason, "updated": 0, "failed": 0}
        return {**repaired, "embedding_dim": self.embedding_dim,
                "updated": repaired["repaired_content"] + repaired["repaired_meaning"],
                "failed": repaired["permanent_failures"] + repaired["transient_failures"]}

