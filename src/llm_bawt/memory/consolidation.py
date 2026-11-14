"""Memory consolidation utility for merging redundant memories.

This module provides functionality to:
1. Find clusters of similar memories using embedding similarity
2. Merge redundant memories into consolidated facts using a local LLM
3. Update the database with merged memories (superseding originals)

IMPORTANT: Only uses local LLMs (gguf, ollama) to avoid sending personal data externally.
"""

import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .errors import MemoryWriteOutcome
from .ownership import MemoryAuthority, NO_MEMORY_AUTHORITY

logger = logging.getLogger(__name__)
_DEFAULT_USER_ALIAS = os.getenv("LLM_BAWT_DEFAULT_USER", "").strip().lower()


@dataclass
class MemoryCluster:
    """A cluster of similar memories that could be merged."""
    memories: list[dict] = field(default_factory=list)
    centroid: np.ndarray | None = None
    avg_similarity: float = 0.0
    
    @property
    def memory_ids(self) -> list[str]:
        return [m["id"] for m in self.memories]
    
    @property
    def combined_importance(self) -> float:
        """Combined importance: max + bonus for reinforcement."""
        if not self.memories:
            return 0.0
        importances = [m.get("importance", 0.5) for m in self.memories]
        # Max importance + small bonus for each additional memory
        return min(1.0, max(importances) + 0.05 * (len(self.memories) - 1))
    
    @property
    def total_access_count(self) -> int:
        return sum(m.get("access_count", 0) for m in self.memories)
    
    @property
    def all_source_message_ids(self) -> list[str]:
        """Union of all source message IDs."""
        ids = set()
        for m in self.memories:
            sources = m.get("source_message_ids") or []
            if isinstance(sources, list):
                ids.update(sources)
        return list(ids)

    @property
    def combined_tags(self) -> list[str]:
        """Union of all tags from memories in the cluster."""
        tags = set()
        for m in self.memories:
            mem_tags = m.get("tags") or []
            if isinstance(mem_tags, list):
                tags.update(mem_tags)
        return list(tags) or ["misc"]
    
    @property
    def best_intent(self) -> str | None:
        """Get intent from highest-importance memory that has one."""
        sorted_mems = sorted(self.memories, key=lambda m: m.get("importance", 0), reverse=True)
        for m in sorted_mems:
            if m.get("intent"):
                return m["intent"]
        return None
    
    def __len__(self) -> int:
        return len(self.memories)


class MemoryConsolidationWriteError(RuntimeError):
    """The merged memory could not be verified, so originals were kept."""


@dataclass
class ConsolidationResult:
    """Result of a consolidation operation."""
    clusters_found: int = 0
    clusters_merged: int = 0
    memories_consolidated: int = 0
    new_memories_created: int = 0
    errors: list[str] = field(default_factory=list)
    dry_run: bool = False


def _normalize_text(text: str) -> str:
    """Normalize text for comparison (lowercase, strip, normalize whitespace)."""
    import re
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    # Normalize common variations
    text = re.sub(r'\bthe user\b', 'user', text)
    if _DEFAULT_USER_ALIAS:
        text = re.sub(rf'\b{re.escape(_DEFAULT_USER_ALIAS)}\b', 'user', text)
    text = re.sub(r"user's", 'user', text)
    return text


def _text_similarity(text1: str, text2: str) -> float:
    """Calculate similarity ratio between two normalized strings."""
    from difflib import SequenceMatcher
    norm1 = _normalize_text(text1)
    norm2 = _normalize_text(text2)
    return SequenceMatcher(None, norm1, norm2).ratio()


class MemoryConsolidator:
    """Consolidates redundant memories using embedding similarity and LLM merging."""
    
    # Default similarity threshold for clustering (0.75 = fairly similar)
    DEFAULT_SIMILARITY_THRESHOLD = 0.75
    # High similarity threshold for cross-type clustering
    CROSS_TYPE_SIMILARITY_THRESHOLD = 0.90
    # Text similarity threshold for near-identical memories
    TEXT_SIMILARITY_THRESHOLD = 0.85
    # Minimum cluster size to consider for merging
    MIN_CLUSTER_SIZE = 2
    # Maximum memories to merge in one LLM call
    MAX_CLUSTER_SIZE = 10
    
    def __init__(
        self,
        backend: Any,  # PostgreSQLMemoryBackend
        llm_client: Any | None = None,
        similarity_threshold: float | None = None,
        config: Any = None,
        *, authority: MemoryAuthority | None = None,
    ):
        """Initialize the consolidator.
        
        Args:
            backend: PostgreSQLMemoryBackend instance
            llm_client: Optional LLM client for intelligent merging (must be local)
            similarity_threshold: Cosine similarity threshold for clustering
            config: Config object for settings
        """
        self.backend = backend
        self.llm_client = llm_client
        self.config = config
        self.threshold = similarity_threshold or self.DEFAULT_SIMILARITY_THRESHOLD
        self.authority = authority or NO_MEMORY_AUTHORITY
    
    def get_all_active_memories_with_embeddings(self) -> list[dict]:
        """Fetch all memories that have embeddings."""
        from sqlalchemy import text
        
        ownership = self.authority.scope(self.backend.bot_id)
        with self.backend.engine.connect() as conn:
            sql = text(f"""
                SELECT id, content, tags, importance, source_message_ids,
                       access_count, last_accessed, created_at, updated_at, embedding, intent,
                       owner_user_id
                FROM {self.backend._memories_table_name} mem
                WHERE embedding IS NOT NULL AND {ownership.mutation_predicate}
                ORDER BY created_at ASC
            """)
            rows = conn.execute(sql, ownership.params).fetchall()
            
            memories = []
            for row in rows:
                # Parse embedding from pgvector format
                embedding = None
                if row.embedding:
                    if isinstance(row.embedding, str):
                        # Parse "[0.1,0.2,...]" format
                        embedding = np.array([float(x) for x in row.embedding.strip("[]").split(",")])
                    elif isinstance(row.embedding, (list, np.ndarray)):
                        embedding = np.array(row.embedding)
                
                memories.append({
                    "id": row.id,
                    "content": row.content,
                    "tags": row.tags if isinstance(row.tags, list) else (json.loads(row.tags) if row.tags else ["misc"]),
                    "importance": float(row.importance) if row.importance else 0.5,
                    "source_message_ids": row.source_message_ids,
                    "access_count": row.access_count or 0,
                    "intent": row.intent,
                    "last_accessed": row.last_accessed,
                    "created_at": row.created_at,
                    "updated_at": getattr(row, "updated_at", None),
                    "owner_user_id": row.owner_user_id,
                    "embedding": embedding,
                })
            
            return memories
    
    def compute_similarity_matrix(self, memories: list[dict]) -> np.ndarray:
        """Compute pairwise cosine similarity matrix."""
        embeddings = np.array([m["embedding"] for m in memories])
        
        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        normalized = embeddings / norms
        
        # Cosine similarity = dot product of normalized vectors
        similarity_matrix = np.dot(normalized, normalized.T)
        
        return similarity_matrix
    
    def find_clusters(self, memories: list[dict]) -> list[MemoryCluster]:
        """Find clusters of similar memories using greedy clustering.
        
        Uses a simple greedy approach:
        1. For each memory, find all others above similarity threshold
        2. Group connected memories (prefers same type, but allows cross-type for very similar)
        3. Filter to clusters with 2+ memories
        
        Clustering rules:
        - Same type + embedding similarity >= threshold: cluster together
        - Different type + embedding similarity >= CROSS_TYPE threshold: cluster together
        - Text similarity >= TEXT_SIMILARITY threshold (after normalization): cluster together
        """
        if len(memories) < 2:
            return []
        
        similarity_matrix = self.compute_similarity_matrix(memories)
        n = len(memories)
        
        # Track which memories have been assigned to a cluster
        assigned = set()
        clusters = []
        
        for i in range(n):
            if i in assigned:
                continue
            
            mem_i = memories[i]
            cluster_indices = [i]
            
            # Find all similar memories
            for j in range(i + 1, n):
                if j in assigned:
                    continue
                
                mem_j = memories[j]
                if mem_i.get("owner_user_id") != mem_j.get("owner_user_id"):
                    continue  # NULL is a separate group, not a wildcard.
                tags_i = set(mem_i.get("tags") or [])
                tags_j = set(mem_j.get("tags") or [])
                tag_overlap = bool(tags_i & tags_j)
                embedding_sim = similarity_matrix[i, j]
                
                # Determine if should cluster (ordered by speed - fastest checks first)
                should_cluster = False
                
                if tag_overlap and embedding_sim >= self.threshold:
                    # Same type OR overlapping tags, meets basic threshold
                    should_cluster = True
                elif embedding_sim >= self.CROSS_TYPE_SIMILARITY_THRESHOLD:
                    # Very high embedding similarity, allow cross-type
                    should_cluster = True
                elif embedding_sim >= 0.4:
                    # Only check expensive text similarity if embeddings are somewhat similar
                    # This avoids O(n²) string comparisons for unrelated memories
                    text_sim = _text_similarity(mem_i["content"], mem_j["content"])
                    if text_sim >= self.TEXT_SIMILARITY_THRESHOLD:
                        # Text is nearly identical (after normalization)
                        should_cluster = True
                
                if should_cluster:
                    cluster_indices.append(j)
            
            # Only create cluster if we have multiple memories
            if len(cluster_indices) >= self.MIN_CLUSTER_SIZE:
                # Limit cluster size
                if len(cluster_indices) > self.MAX_CLUSTER_SIZE:
                    cluster_indices = cluster_indices[:self.MAX_CLUSTER_SIZE]
                
                cluster_memories = [memories[idx] for idx in cluster_indices]
                
                # Compute average similarity within cluster
                cluster_sims = []
                for a in range(len(cluster_indices)):
                    for b in range(a + 1, len(cluster_indices)):
                        cluster_sims.append(similarity_matrix[cluster_indices[a], cluster_indices[b]])
                avg_sim = np.mean(cluster_sims) if cluster_sims else 0.0
                
                # Compute centroid
                embeddings = np.array([memories[idx]["embedding"] for idx in cluster_indices])
                centroid = np.mean(embeddings, axis=0)
                
                cluster = MemoryCluster(
                    memories=cluster_memories,
                    centroid=centroid,
                    avg_similarity=float(avg_sim),
                )
                clusters.append(cluster)
                
                # Mark all as assigned
                assigned.update(cluster_indices)
        
        # Sort by size (largest first)
        clusters.sort(key=lambda c: len(c), reverse=True)
        
        return clusters
    
    def merge_cluster_with_llm(self, cluster: MemoryCluster) -> str | None:
        """Use LLM to intelligently merge a cluster of memories.
        
        Returns the merged memory content, or None on failure.
        """
        if not self.llm_client:
            return None
        
        # Build the prompt
        memory_texts = []
        for i, mem in enumerate(cluster.memories, 1):
            memory_texts.append(f"{i}. {mem['content']}")
        
        memories_list = "\n".join(memory_texts)
        tags = cluster.combined_tags
        
        prompt = f"""You are a memory consolidation system. Your task is to merge these similar memories into a single, comprehensive fact.

Tags: {', '.join(tags)}

Memories to merge:
{memories_list}

Instructions:
- Combine the information into ONE clear, factual statement
- Preserve all unique details from each memory
- Remove redundant information
- Keep it concise but complete
- Do NOT add information not present in the originals
- Output ONLY the merged memory text, nothing else

Merged memory:"""

        try:
            # Use the LLM client to generate the merged memory
            from ..models.message import Message
            
            messages = [Message(role="user", content=prompt)]
            
            response = ""
            for chunk in self.llm_client.stream_raw(messages):
                response += chunk
            
            merged = response.strip()
            
            # Basic validation
            if not merged or len(merged) < 10:
                logger.warning(f"LLM returned invalid merged memory: {merged}")
                return None
            
            return merged
            
        except Exception as e:
            logger.error(f"LLM merge failed: {e}")
            return None
    
    def merge_cluster_heuristic(self, cluster: MemoryCluster) -> str:
        """Fallback heuristic merge: pick the longest/most detailed memory.
        
        If we can't use an LLM, we just pick the best representative.
        """
        # Score by length * importance
        best_mem = max(
            cluster.memories,
            key=lambda m: len(m["content"]) * m.get("importance", 0.5)
        )
        return best_mem["content"]
    
    def create_merged_memory(
        self,
        content: str,
        cluster: MemoryCluster,
    ) -> MemoryWriteOutcome:
        """Write the merged memory and return what was actually persisted.

        Uses the backend's add_memory method which handles embedding generation
        and meaning field heuristics. The backend may fold the content into an
        existing row (deduplication), which can be one of the cluster's own
        originals; ``outcome.memory_id`` names the surviving row either way.
        """
        memory_id = str(uuid.uuid4())
        tags = cluster.combined_tags
        importance = cluster.combined_importance
        source_message_ids = cluster.all_source_message_ids
        
        # Use intent from highest-importance memory (LLM-inferred, if available)
        intent = cluster.best_intent
        # Other meaning fields use heuristics
        stakes = self._infer_stakes(importance)
        emotional_charge = self._infer_emotion(content, importance)
        recurrence_keywords = self._infer_recurrence(content, tags)
        
        # Use backend's add_memory which handles embeddings and schema properly
        outcome = self.backend.add_memory(
            memory_id=memory_id,
            content=content,
            tags=tags,
            importance=importance,
            source_message_ids=source_message_ids,
            intent=intent,
            stakes=stakes,
            emotional_charge=emotional_charge,
            recurrence_keywords=recurrence_keywords,
            authority=self.authority,
            _merge_originals=cluster.memories,
        )
        
        if not isinstance(outcome, MemoryWriteOutcome) or not outcome.persisted:
            detail = getattr(outcome, "error", None) or getattr(outcome, "status", "no outcome")
            raise MemoryConsolidationWriteError(
                f"merged memory was not persisted ({detail}); originals kept"
            )

        # Access counts are carried forward only inside verified replacement.
        return outcome
    
    def _infer_stakes(self, importance: float) -> str:
        """Infer stakes from importance score."""
        if importance >= 0.8:
            return "critical to remember; forgetting would harm trust or outcomes"
        if importance >= 0.6:
            return "important context for smooth interactions"
        return "nice-to-know; low stakes"
    
    def _infer_emotion(self, content: str, importance: float) -> float:
        """Infer emotional charge from content and importance."""
        text_lower = content.lower()
        charge = 0.2 + 0.6 * importance
        keywords_high = ["love", "hate", "angry", "excited", "anxious", "worried", "sad"]
        if any(k in text_lower for k in keywords_high):
            charge = max(charge, 0.8)
        return min(1.0, max(0.0, charge))
    
    def _infer_recurrence(self, content: str, tags: list[str]) -> list[str]:
        """Extract recurrence keywords from content and tags."""
        import re
        tag_list = [t for t in tags if t]
        words = re.findall(r"[a-zA-Z]{4,}", content.lower())
        return list(set(tag_list + words[:3]))
    
    def replace_originals(
        self,
        survivor_id: str,
        cluster: MemoryCluster,
        merged_content: str,
    ) -> tuple[int, list[str]]:
        """Delete a cluster's originals only while a verified survivor exists.

        Runs in one transaction: the survivor row is locked FOR SHARE first,
        and the survivor itself is never deleted even when deduplication folded
        the merge into one of the originals. An original is deleted only if its
        ``updated_at`` still matches what consolidation read, so a concurrent
        edit keeps that row. Returns (deleted_count, kept_changed_ids).
        """
        from sqlalchemy import text

        table = self.backend._memories_table_name
        deleted = 0
        kept: list[str] = []
        ownership = self.authority.scope(self.backend.bot_id)
        owners = {mem.get("owner_user_id") for mem in cluster.memories}
        if len(owners) != 1:
            raise MemoryConsolidationWriteError("distinct memory owners; originals kept")
        params = {**ownership.params, "merge_owner": owners.pop()}
        group_guard = "mem.owner_user_id IS NOT DISTINCT FROM CAST(:merge_owner AS text)"
        with self.backend._memory_write_transaction() as conn:
            survivor = conn.execute(
                text(f"""SELECT id, content FROM {table} mem WHERE id = :id
                    AND {ownership.mutation_predicate} AND {group_guard} FOR UPDATE"""),
                {**params, "id": survivor_id},
            ).first()
            if survivor is None:
                raise MemoryConsolidationWriteError(
                    f"merged memory {survivor_id} is not present; originals kept"
                )
            # The survivor must hold exactly the merged text: a semantic dedup
            # onto a different row, or an edit since the write, keeps all.
            if survivor.content != merged_content:
                raise MemoryConsolidationWriteError(
                    f"memory {survivor_id} does not hold the merged content; originals kept"
                )
            for mem in cluster.memories:
                mem_id = str(mem["id"])
                if mem_id == str(survivor_id):
                    continue
                row = conn.execute(
                    text(f"""
                        DELETE FROM {table} mem
                        WHERE id = :id
                          AND updated_at IS NOT DISTINCT FROM :updated_at
                          AND {ownership.mutation_predicate} AND {group_guard}
                        RETURNING id
                    """),
                    {**params, "id": mem_id, "updated_at": mem.get("updated_at")},
                ).first()
                if row is None:
                    kept.append(mem_id)
                else:
                    deleted += 1
            if str(survivor_id) not in {str(mem["id"]) for mem in cluster.memories}:
                conn.execute(text(f"""
                    UPDATE {table} mem SET access_count = GREATEST(access_count, :count)
                    WHERE id=:id AND {ownership.mutation_predicate} AND {group_guard}
                """), {**params, "id": survivor_id, "count": cluster.total_access_count})
        return deleted, kept

    def consolidate(self, dry_run: bool = False) -> ConsolidationResult:
        """Run the full consolidation process.
        
        Args:
            dry_run: If True, only report what would be done without making changes
            
        Returns:
            ConsolidationResult with statistics
        """
        result = ConsolidationResult(dry_run=dry_run)
        
        # Fetch all active memories with embeddings
        logger.debug("Fetching memories with embeddings...")
        memories = self.get_all_active_memories_with_embeddings()
        
        if len(memories) < 2:
            logger.debug(f"Only {len(memories)} memories found, nothing to consolidate")
            return result
        
        # Find clusters
        logger.debug(f"Finding clusters among {len(memories)} memories (threshold={self.threshold})...")
        clusters = self.find_clusters(memories)
        result.clusters_found = len(clusters)
        
        if not clusters:
            logger.debug("No clusters found above similarity threshold")
            return result
        
        logger.debug(f"Found {len(clusters)} clusters to process")
        
        # Process each cluster
        for cluster in clusters:
            try:
                logger.debug(f"Processing cluster of {len(cluster)} memories (tags={cluster.combined_tags})")
                
                # Try LLM merge first, fall back to heuristic
                merged_content = None
                if self.llm_client:
                    merged_content = self.merge_cluster_with_llm(cluster)
                
                if not merged_content:
                    merged_content = self.merge_cluster_heuristic(cluster)
                
                if dry_run:
                    logger.debug(f"[DRY RUN] Would merge {len(cluster)} memories into: {merged_content[:100]}...")
                    result.clusters_merged += 1
                    result.memories_consolidated += len(cluster)
                else:
                    # Write the merge first; raises (originals kept) unless a
                    # row verifiably holds it.
                    outcome = self.create_merged_memory(merged_content, cluster)
                    survivor_id = str(outcome.memory_id)

                    deleted, kept = self.replace_originals(survivor_id, cluster, merged_content)
                    if kept:
                        result.errors.append(
                            f"Kept {len(kept)} memories changed during consolidation: {', '.join(kept)}"
                        )

                    result.clusters_merged += 1
                    result.memories_consolidated += deleted
                    if outcome.status == "inserted":
                        result.new_memories_created += 1

                    logger.debug(
                        f"Merged into {outcome.status} memory {survivor_id}, deleted {deleted} originals"
                    )
                    
            except Exception as e:
                error_msg = f"Failed to process cluster: {e}"
                logger.error(error_msg)
                result.errors.append(error_msg)
        
        
        return result


class LLMServiceClient:
    """Simple client that proxies LLM requests to the llm-service.
    
    This allows consolidation (running in the MCP server) to use the
    already-loaded model in the llm-service without loading another copy.
    """
    
    def __init__(self, service_url: str):
        self.service_url = service_url.rstrip("/")
    
    def query(self, messages: list[dict], max_tokens: int = 500, temperature: float = 0.7) -> str:
        """Query the LLM via the service's raw completion endpoint."""
        import urllib.request
        import json
        
        # Extract system and user messages
        system = None
        prompt = ""
        for msg in messages:
            if msg.get("role") == "system":
                system = msg.get("content", "")
            elif msg.get("role") == "user":
                prompt = msg.get("content", "")
        
        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "caller": "consolidation",
        }
        if system:
            payload["system"] = system
        
        url = f"{self.service_url}/v1/llm/complete"
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        
        try:
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=60) as response:
                result = json.loads(response.read().decode("utf-8"))
                return result.get("content", "")
        except urllib.error.HTTPError as e:
            if e.code == 503:
                logger.debug("LLM service has no model loaded")
                return ""
            raise
        except Exception as e:
            logger.debug(f"LLM service request failed: {e}")
            return ""


def get_local_llm_client(config: Any) -> Any | None:
    """Get an LLM client for consolidation.
    
    First tries to use the llm-service (which has the model already loaded).
    Falls back to heuristic merging if service is unavailable.
    """
    # Try the llm-service first - it has the model already loaded
    service_host = getattr(config, "SERVICE_HOST", "localhost")
    service_port = getattr(config, "SERVICE_PORT", 8642)
    service_url = f"http://{service_host}:{service_port}"
    
    # Quick health check
    import urllib.request
    try:
        req = urllib.request.Request(f"{service_url}/health", method="GET")
        with urllib.request.urlopen(req, timeout=2) as response:
            if response.status == 200:
                logger.debug("Using llm-service for consolidation")
                return LLMServiceClient(service_url)
    except Exception as e:
        logger.debug(f"llm-service not available: {e}")
    
    logger.debug("No LLM available for consolidation, using heuristic merging")
    return None
