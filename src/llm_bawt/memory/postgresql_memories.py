"""Semantic-memory operations for the PostgreSQL memory backend.

This module owns distilled-memory CRUD, retrieval, ranking, statistics, and
embedding maintenance. Engine and schema construction remain in postgresql.py.
"""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from datetime import datetime, timezone

from sqlalchemy import text

from llm_bawt.bot_access import BotAccessError
from llm_bawt.bot_access.provenance import MemoryProvenance
from llm_bawt.memory.ownership import (
    MemoryAuthority, NO_MEMORY_AUTHORITY, OPERATOR_MEMORY_AUTHORITY,
)
from llm_bawt.memory.errors import (
    MemoryUpdateConflictError,
    MemoryUpdateDuplicateError,
    MemoryWriteOutcome,
)

logger = logging.getLogger(__name__)


class PostgreSQLSemanticMemoryMixin:
    """Distilled semantic-memory behavior for PostgreSQLMemoryBackend."""

    @contextmanager
    def _memory_write_transaction(self):
        """Serialize dedup + write by storage partition, after remote embedding.

        READ COMMITTED gives each dedup query a fresh snapshot after the lock
        wait. The transaction releases the lock on commit, rollback or error.
        """
        with self.engine.begin() as conn:
            # Exclude activation before observing its marker. A later INSERT's
            # implicit table lock would be too late to close the cutover race.
            conn.execute(text("LOCK TABLE sessions, messages, memories IN ROW EXCLUSIVE MODE"))
            conn.execute(text("""
                SELECT pg_advisory_xact_lock(
                    hashtext('memory-dedup'), hashtext(:partition))
            """), {"partition": self._memories_table_name})
            yield conn

    def _ownership(self, authority=None, eligibility=None):
        if authority is None and eligibility is not None:
            authority = MemoryAuthority(user_id=eligibility.authority.user_id)
        return (authority or NO_MEMORY_AUTHORITY).scope(self.bot_id)

    @staticmethod
    def _validate_extraction_sources(conn, eligibility, source_message_ids):
        if eligibility is None:
            return
        valid = conn.execute(text(f"""
            SELECT 1 FROM (SELECT CAST(:source_ids AS jsonb) AS source_message_ids,
                                  CAST(:authority_partition AS text) AS bot_id) mem
            WHERE {eligibility.predicate}
        """), {**eligibility.params, "source_ids": json.dumps(source_message_ids or [])}).first()
        if valid is None:
            raise ValueError("extraction memory sources do not belong to the owner")

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
        *,
        eligibility: MemoryProvenance | None = None,
        authority: MemoryAuthority | None = None,
        _merge_originals: list[dict] | None = None,
    ) -> MemoryWriteOutcome:
        """Add a distilled memory to storage.

        If no embedding is provided, one will be generated automatically
        using the configured local embedding model.

        Returns what was actually written. Callers that hand out IDs or delete
        other rows must check ``outcome.persisted`` and use ``outcome.memory_id``.
        """
        if not content or content.isspace():
            logger.warning(f"Skipping empty content for memory ID: {memory_id}")
            return MemoryWriteOutcome("skipped")
        
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

        ownership = self._ownership(authority, eligibility)
        guard = f"({eligibility.predicate if eligibility else 'TRUE'}) AND {ownership.mutation_predicate}"
        scope_params = {**(eligibility.params if eligibility else {}), **ownership.params}
        try:
            with self._memory_write_transaction() as conn:
                ownership.validate_write(conn, source_message_ids)
                self._validate_extraction_sources(conn, eligibility, source_message_ids)
                existing = conn.execute(text(f"""
                    SELECT owner_user_id FROM {self._memories_table_name} WHERE id = :id
                """), {"id": memory_id}).first()
                if existing and conn.execute(text(f"""
                    SELECT 1 FROM {self._memories_table_name} mem
                    WHERE id = :id AND {guard}
                """), {**scope_params, "id": memory_id}).first() is None:
                    raise ValueError("cannot overwrite another owner's memory")
                insert_owner = ownership.authority.user_id
                if _merge_originals is not None:
                    # Internal consolidation only: retain the originals' owner,
                    # including NULL. Operator visibility must never launder a
                    # merge into another owner; personal legacy merges stay NULL.
                    if existing or not _merge_originals:
                        raise ValueError("merge requires originals and a new survivor ID")
                    owners = {row.get("owner_user_id") for row in _merge_originals}
                    if len(owners) != 1:
                        raise ValueError("cannot consolidate distinct memory owners")
                    insert_owner = owners.pop()
                    for original in _merge_originals:
                        valid = conn.execute(text(f"""
                            SELECT id FROM {self._memories_table_name} mem
                            WHERE id=:id AND {guard}
                              AND owner_user_id IS NOT DISTINCT FROM CAST(:merge_owner AS text)
                              AND updated_at IS NOT DISTINCT FROM :merge_updated_at
                            FOR SHARE
                        """), {**scope_params, "id": original["id"],
                               "merge_owner": insert_owner,
                               "merge_updated_at": original.get("updated_at")}).first()
                        if valid is None:
                            raise ValueError("consolidation original changed or is not owned")
                scope_params["memory_dedup_owner"] = (
                    existing.owner_user_id if existing else insert_owner
                )
                dedup_guard = f"({guard}) AND {ownership.dedup_group_predicate}"
                if _merge_originals is not None:
                    dedup_guard += " AND mem.owner_user_id IS NOT DISTINCT FROM CAST(:memory_dedup_owner AS text)"
                # Check for duplicate/similar content BEFORE inserting
                # Use embedding similarity if available, otherwise exact content match
                # Threshold from config (default 0.85) - catches semantic duplicates
                dedup_threshold = getattr(self.config, 'MEMORY_DEDUP_SIMILARITY', 0.85)
                
                if embedding:
                    # Vector similarity check - find most similar existing memory
                    dedup_sql = text(f"""
                        SELECT id, content, 1 - (embedding <=> :embedding) AS similarity
                        FROM {self._memories_table_name} mem
                        WHERE embedding IS NOT NULL AND {dedup_guard}
                        ORDER BY embedding <=> :embedding
                        LIMIT 1
                    """)
                    similar = conn.execute(dedup_sql, {
                        **scope_params,
                        "embedding": f"[{','.join(str(x) for x in embedding)}]",
                    }).first()
                    
                    if similar and similar.similarity > dedup_threshold:
                        logger.debug(f"Skipping duplicate memory (similarity={similar.similarity:.2%} > {dedup_threshold:.2%}): '{content[:50]}...' similar to '{similar.content[:50]}...'")
                        return MemoryWriteOutcome("duplicate", str(similar.id))
                else:
                    # Exact content match fallback
                    exact_sql = text(f"""
                        SELECT id FROM {self._memories_table_name} mem
                        WHERE LOWER(content) = LOWER(:content) AND {dedup_guard}
                        LIMIT 1
                    """)
                    exact_match = conn.execute(exact_sql, {**scope_params, "content": content}).first()
                    if exact_match:
                        logger.debug(f"Skipping exact duplicate memory: '{content[:50]}...'")
                        return MemoryWriteOutcome("duplicate", str(exact_match.id))
                
                if existing:
                    # Update existing
                    update_sql = text(f"""
                        UPDATE {self._memories_table_name} mem
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
                        WHERE id = :id AND {guard}
                    """)
                    conn.execute(update_sql, {
                        **scope_params,
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
                        (id, content, tags, importance, source_message_ids, embedding, owner_user_id,
                         intent, stakes, emotional_charge, recurrence_keywords, meaning_embedding, meaning_updated_at, created_at)
                        VALUES (:id, :content, CAST(:tags AS jsonb), :importance, :source_ids, :embedding, :owner_user_id,
                                :intent, :stakes, :emotional_charge, CAST(:recurrence_keywords AS jsonb), :meaning_embedding, :meaning_updated_at, CURRENT_TIMESTAMP)
                    """)
                    conn.execute(insert_sql, {
                        "owner_user_id": insert_owner,
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

            logger.debug(f"Added memory {memory_id} to {self._memories_table_name}")
            return MemoryWriteOutcome("updated" if existing else "inserted", memory_id)
        except Exception as e:
            logger.error(f"Failed to add memory {memory_id}: {e}")
            return MemoryWriteOutcome("failed", error=str(e))

    def update_memory(
        self,
        memory_id: str,
        *,
        content: str | None = None,
        tags: list[str] | None = None,
        importance: float | None = None,
        source_message_ids: list[str] | None = None,
        eligibility: MemoryProvenance | None = None,
        authority: MemoryAuthority | None = None,
    ) -> dict | None:
        """Patch an existing row, excluding itself from duplicate detection.

        Do not hold a database lock across embedding HTTP calls. A stale snapshot
        fails visibly instead of overwriting a newer edit or returning invented
        state. An unavailable embedder clears vectors only when content changes.
        """
        ownership = self._ownership(authority, eligibility)
        guard = f"({eligibility.predicate if eligibility else 'TRUE'}) AND {ownership.mutation_predicate}"
        scope_params = {**(eligibility.params if eligibility else {}), **ownership.params}
        with self.engine.connect() as conn:
            original = conn.execute(text(f"""
                SELECT content, intent, stakes, emotional_charge,
                       recurrence_keywords, updated_at, owner_user_id
                FROM {self._memories_table_name} mem WHERE id = :id AND {guard}
            """), {**scope_params, "id": memory_id}).mappings().first()
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
            **scope_params,
            "id": memory_id, "old_content": original["content"],
            "old_updated_at": original["updated_at"],
            "memory_dedup_owner": original["owner_user_id"],
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
        with self._memory_write_transaction() as conn:
            ownership.validate_write(conn, source_message_ids)
            if source_message_ids is not None:
                self._validate_extraction_sources(conn, eligibility, source_message_ids)
            if changed:
                threshold = getattr(self.config, "MEMORY_DEDUP_SIMILARITY", 0.85)
                # A prior embedding outage leaves NULL vectors. Exact matches
                # still count even when this edit has a valid new embedding.
                exact = conn.execute(text(f"""
                    SELECT id FROM {self._memories_table_name} mem
                    WHERE id <> :id AND LOWER(content) = LOWER(:content) AND {guard}
                      AND {ownership.dedup_group_predicate} LIMIT 1
                """), params).first()
                is_duplicate = exact is not None
                if not is_duplicate and embedding:
                    similar = conn.execute(text(f"""
                        SELECT id, 1 - (embedding <=> :embedding) AS similarity
                        FROM {self._memories_table_name} mem
                        WHERE id <> :id AND embedding IS NOT NULL AND {guard}
                          AND {ownership.dedup_group_predicate}
                        ORDER BY embedding <=> :embedding LIMIT 1
                    """), params).first()
                    is_duplicate = similar is not None and similar.similarity > threshold
                if is_duplicate:
                    raise MemoryUpdateDuplicateError("memory update conflicts with another duplicate")

            updated = conn.execute(text(f"""
                UPDATE {self._memories_table_name} mem
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
                WHERE id = :id AND content = :old_content AND {guard}
                  AND updated_at IS NOT DISTINCT FROM :old_updated_at
                  AND owner_user_id IS NOT DISTINCT FROM CAST(:memory_dedup_owner AS text)
                RETURNING id, content, tags, importance, source_message_ids,
                          access_count, created_at, last_accessed,
                          intent, stakes, emotional_charge
            """), params).mappings().first()
            if updated is None:
                exists = conn.execute(text(f"""
                    SELECT 1 FROM {self._memories_table_name} mem WHERE id = :id AND {guard}
                """), params).first()
                if exists is not None:
                    raise MemoryUpdateConflictError("memory changed concurrently; retry the update")
                return None
            return dict(updated)

    def delete_memory(self, memory_id: str, *, user_id: str | None = None,
                      authority: MemoryAuthority | None = None) -> bool:
        """Delete by ID/prefix, preserving the optional scoped-delete API.

        The existing internal API defines user_id=None as operator. Unbound
        entry points must pass NO_MEMORY_AUTHORITY rather than use that legacy
        operator shorthand; MCP resolves the signed caller before reaching here.
        """
        if authority is None and user_id is None:
            authority = OPERATOR_MEMORY_AUTHORITY
        if user_id is not None:
            personal = MemoryAuthority(user_id=user_id)
            if authority is not None and authority != personal:
                raise ValueError("conflicting memory authorities")
            authority = personal
        if authority is not None and authority.user_id is not None:
            return self._delete_owned_memory(memory_id, authority.user_id)
        ownership = self._ownership(authority)
        match = "id LIKE :mid" if len(memory_id) < 36 else "id = :mid"
        params = {**ownership.params, "mid": f"{memory_id}%" if len(memory_id) < 36 else memory_id}
        try:
            with self._memory_write_transaction() as conn:
                return conn.execute(text(f"""
                    DELETE FROM {self._memories_table_name} mem
                    WHERE {match} AND {ownership.mutation_predicate}
                """), params).rowcount > 0
        except Exception as e:
            logger.error(f"Failed to delete memory {memory_id}: {e}")
            return False

    def _delete_owned_memory(self, memory_id: str, user_id: str) -> bool:
        if not str(user_id).strip():
            raise ValueError("user_id must be nonblank")
        from .short_term_message_reads import history_owner_clause

        ownership = self._ownership(MemoryAuthority(user_id=user_id))
        params = ownership.params
        owner = history_owner_clause(params, bot_id=self.bot_id, user_id=user_id)
        if len(memory_id) < 36:
            match, params["mid"] = "id LIKE :mid", f"{memory_id}%"
        else:
            match, params["mid"] = "id = :mid", memory_id
        # Owned: at least one source, and every source is one of the user's
        # messages, live or forgotten.
        sql = text(f"""
            DELETE FROM {self._memories_table_name} mem
            WHERE {match} AND (
              ({ownership.active} AND {ownership.owned_predicate}) OR
              (NOT ({ownership.active})
              AND jsonb_typeof(mem.source_message_ids::jsonb) = 'array'
              AND jsonb_array_length(mem.source_message_ids::jsonb) > 0
              AND NOT EXISTS (
                SELECT 1 FROM jsonb_array_elements_text(mem.source_message_ids::jsonb) src(source_id)
                WHERE src.source_id NOT IN (
                    SELECT id FROM {self._messages_table_name} WHERE {owner}
                    UNION ALL
                    SELECT id FROM {self._forgotten_table_name} WHERE {owner}
                )
              ))
            )
        """)
        try:
            with self._memory_write_transaction() as conn:
                return conn.execute(sql, params).rowcount > 0
        except Exception as e:
            logger.error(f"Failed to delete memory {memory_id}: {e}")
            return False

    def delete_memories_by_source_message_ids(self, message_ids: list[str], *,
                                             authority: MemoryAuthority | None = None) -> int:
        """Forget any memory citing a validated owned message, including mixed rows.

        This intentionally differs from by-ID deletion: demanding ownership of
        every source would retain private content the user asked to forget.
        """
        if not message_ids:
            return 0
        ownership = self._ownership(authority)
        try:
            with self._memory_write_transaction() as conn:
                ownership.validate_forget(conn, message_ids)
                return conn.execute(text(f"""
                    DELETE FROM {self._memories_table_name}
                    WHERE source_message_ids ?| CAST(:ids AS text[])
                """), {"ids": message_ids}).rowcount
        except Exception as e:
            logger.error(f"Failed to delete memories by source message IDs: {e}")
            return 0

    def update_memory_access(self, memory_id: str, *,
                             authority: MemoryAuthority | None = None,
                             eligibility: MemoryProvenance | None = None) -> None:
        """Reinforce only rows eligible for this read (including legacy reads)."""
        ownership = self._ownership(authority, eligibility)
        try:
            with self._memory_write_transaction() as conn:
                conn.execute(text(f"""
                    UPDATE {self._memories_table_name} mem
                    SET access_count = access_count + 1,
                        last_accessed = CURRENT_TIMESTAMP
                    WHERE id = :id AND {ownership.read_predicate}
                      AND ({eligibility.predicate if eligibility else 'TRUE'})
                """), {"id": memory_id, **ownership.params,
                       **(eligibility.params if eligibility else {})})
        except Exception as e:
            logger.error(f"Failed to update memory access: {e}")

    def search_memories_by_text(
        self,
        query: str,
        n_results: int = 5,
        min_importance: float = 0.0,
        tags: list[str] | None = None,
        *, authority: MemoryAuthority | None = None,
    ) -> list[dict]:
        """Search memories using PostgreSQL full-text search."""
        ownership = self._ownership(authority)
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
                    FROM {self._memories_table_name} mem
                    WHERE to_tsvector('english', content) @@ plainto_tsquery('english', :query)
                    AND {ownership.read_predicate}
                    AND importance >= :min_importance
                    {tag_filter}
                    ORDER BY rank DESC, importance DESC
                    LIMIT :limit
                """)
                
                rows = conn.execute(sql, {
                    **ownership.params,
                    "query": query,
                    "min_importance": min_importance,
                    "limit": n_results,
                    "filter_tags": tags,
                }).fetchall()
                # Access reinforcement uses another transaction. Release read
                # locks first so activation cannot deadlock that writer.
                conn.commit()

                results = []
                for row in rows:
                    # Update access tracking
                    self.update_memory_access(row.id, authority=authority)
                    
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
        authority: MemoryAuthority | None = None,
        created_since: float | None = None,
        created_until: float | None = None,
    ) -> list[dict]:
        """Search memories using vector similarity with temporal decay and diversity.

        ``created_since``/``created_until`` (Unix seconds, inclusive) bound
        ``created_at`` before ranking and the result limit, so a window never
        loses an eligible match to an older, higher-ranked one (TASK-2238).
        Rows without ``created_at`` stay eligible.

        The effective score combines:
        - Semantic similarity (cosine distance)
        - Base importance
        - Temporal decay (memories fade over time)
        - Access boost (frequently accessed memories get reinforced)
        - Diversity sampling (avoid echo chambers by sampling across time/types)
        """
        if not embedding:
            return []
        
        ownership = self._ownership(authority, eligibility)
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
                # created_at is naive UTC; convert the bound explicitly so the
                # session time zone never shifts the window.
                if created_since is not None:
                    tag_filter += (" AND (created_at IS NULL OR created_at >= "
                                   "(to_timestamp(:created_since) AT TIME ZONE 'UTC'))")
                if created_until is not None:
                    tag_filter += (" AND (created_at IS NULL OR created_at <= "
                                   "(to_timestamp(:created_until) AT TIME ZONE 'UTC'))")

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
                        AND {ownership.read_predicate}
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
                        AND {ownership.read_predicate}
                        {f'AND ({eligibility.predicate})' if eligibility is not None else ''}
                        {tag_filter}
                        ORDER BY embedding <=> :embedding
                        LIMIT :limit
                    """)
                    fetch_limit = n_results
                
                rows = conn.execute(sql, {
                    **ownership.params,
                    "embedding": str(embedding),
                    "meaning_embedding": str(meaning_embedding) if meaning_embedding else str(embedding),
                    "use_meaning": bool(meaning_embedding),
                    "min_importance": min_importance,
                    "limit": fetch_limit,
                    "filter_tags": tags,
                    "created_since": created_since,
                    "created_until": created_until,
                    **(eligibility.params if eligibility is not None else {}),
                }).fetchall()
                # Do not retain the reader's locks while reinforcing via a
                # second transaction (activation acquires locks in writer order).
                conn.commit()

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
                    importance = float(row.importance) if row.importance is not None else 0.5
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
                    self.update_memory_access(r["id"], authority=authority, eligibility=eligibility)
                
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

    def search(self, query: str, n_results: int = 5, min_relevance: float = 0.0, *,
               authority: MemoryAuthority | None = None) -> list[dict] | None:
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
        text_results = self.search_memories_by_text(
            query, n_results, min_importance=min_relevance, authority=authority)
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
                    embedding_results = self.search_memories_by_embedding(
                        query_embedding, n_results, min_importance=min_relevance, authority=authority)
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
        high_importance_results = self.get_high_importance_memories(
            3, min_importance=0.9, authority=authority)
        for r in high_importance_results:
            if r["id"] not in seen_ids:
                combined_results.append(format_memory_result(r))
                seen_ids.add(r["id"])
        
        if combined_results:
            logger.debug(f"Returning {len(combined_results)} blended results (embedding + core facts)")
            return combined_results[:n_results]
        
        # 3. Last resort: search raw messages
        message_results = self.search_messages_by_text(
            query, n_results, exclude_recent_seconds=10.0,
            authority=authority or NO_MEMORY_AUTHORITY)
        
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

    def get_high_importance_memories(self, n_results: int = 10, min_importance: float = 0.7, *,
                                     authority: MemoryAuthority | None = None) -> list[dict]:
        """Get the most important eligible memories, regardless of query matching."""
        ownership = self._ownership(authority)
        with self.engine.connect() as conn:
            try:
                sql = text(f"""
                    SELECT id, content, tags, importance, source_message_ids,
                           access_count, last_accessed, created_at,
                           intent, stakes, emotional_charge
                    FROM {self._memories_table_name} mem
                    WHERE importance >= :min_importance AND {ownership.read_predicate}
                    ORDER BY importance DESC, access_count DESC, created_at DESC
                    LIMIT :limit
                """)
                
                rows = conn.execute(sql, {
                    **ownership.params,
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

    def clear(self, *, authority: MemoryAuthority | None = None) -> bool:
        """Clear mutable memories in the caller's scope, never messages."""
        ownership = self._ownership(authority)
        try:
            with self._memory_write_transaction() as conn:
                ownership.validate_write(conn, None)
                conn.execute(text(f"""
                    DELETE FROM {self._memories_table_name} mem
                    WHERE {ownership.mutation_predicate}
                """), ownership.params)
            return True
        except Exception as e:
            logger.error(f"Failed to clear memories: {e}")
            return False

    def get_memory_by_id(self, memory_id: str, *, authority: MemoryAuthority | None = None) -> dict | None:
        ownership = self._ownership(authority)
        with self.engine.connect() as conn:
            row = conn.execute(text(f"""
                SELECT * FROM {self._memories_table_name} mem
                WHERE id=:id AND {ownership.read_predicate}
            """), {**ownership.params, "id": memory_id}).mappings().first()
        return dict(row) if row is not None else None

    def list_recent(self, n: int = 10, *, authority: MemoryAuthority | None = None) -> list[dict]:
        """Apply caller ownership before ordering and LIMIT."""
        ownership = self._ownership(authority)
        with self.engine.connect() as conn:
            rows = conn.execute(text(f"""
                SELECT * FROM {self._memories_table_name} mem
                WHERE {ownership.read_predicate}
                ORDER BY created_at DESC LIMIT :limit
            """), {**ownership.params, "limit": n}).fetchall()
            
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

    def stats(self, *, authority: MemoryAuthority | None = None) -> dict:
        """Get caller-scoped statistics, without exposing a whole-partition size."""
        ownership = self._ownership(authority)
        size = f"pg_total_relation_size('{self._memories_table_name}')"
        if not ownership.authority.operator:
            size = f"CASE WHEN {ownership.active} THEN NULL ELSE {size} END"
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
                        {size} as size_bytes
                    FROM {self._memories_table_name} mem
                    WHERE {ownership.read_predicate}
                """)
                mem_row = conn.execute(mem_stats_sql, ownership.params).first()
                
                # Message stats
                msg_stats_sql = text(f"""
                    SELECT 
                        COUNT(*) as total,
                        SUM(CASE WHEN processed THEN 1 ELSE 0 END) as processed,
                        MIN(created_at) as oldest,
                        MAX(created_at) as newest
                    FROM {self._messages_table_name}
                    WHERE {ownership.message_read_predicate}
                """)
                msg_row = conn.execute(msg_stats_sql, ownership.params).first()
                
                # Forgotten message count
                forgotten_count = 0
                try:
                    forgotten_sql = text(f"""SELECT COUNT(*) FROM {self._forgotten_table_name}
                        WHERE {ownership.message_read_predicate}""")
                    forgotten_count = conn.execute(forgotten_sql, ownership.params).scalar() or 0
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

    def regenerate_embeddings(self, batch_size: int = 50, *,
                              authority: MemoryAuthority | None = None) -> dict:
        """Repair both missing vectors in a bounded pass; never alter schema.

        This manual entry point deliberately refuses private and inactive bots,
        just like the recurring repair. A subsequent call continues at the saved
        cursor; no permanently bad row can stall the rest of the queue.
        """
        from .embedding_repair import EmbeddingRepair

        repaired = EmbeddingRepair(self.config, self.engine).repair_bot(
            self.bot_id, limit=min(batch_size, 25), authority=authority or NO_MEMORY_AUTHORITY,
        )
        if repaired.get("excluded") or repaired["busy"] or repaired["dimension_mismatch"]:
            reason = ("bot excluded from embedding repair" if repaired.get("excluded") else
                      "embedding repair already running" if repaired["busy"] else
                      "embedding dimension mismatch; explicit schema migration required")
            return {"error": reason, "updated": 0, "failed": 0}
        return {**repaired, "embedding_dim": self.embedding_dim,
                "updated": repaired["repaired_content"] + repaired["repaired_meaning"],
                "failed": repaired["permanent_failures"] + repaired["transient_failures"]}

