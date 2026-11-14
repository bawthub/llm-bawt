"""Owner-scoped memory operations for extraction, not a global CRUD policy.

Ownership is proven by every source message's session. Unsourced/manual and
mixed-owner memories are invisible to this writer, including during dedup.
"""
from uuid import uuid4

from sqlalchemy import text

from llm_bawt.bot_access.policy import BotPolicyResolver, PartitionScope
from llm_bawt.bot_access.provenance import MemoryProvenance, ScopedMessageAuthority


class ExtractionMemoryStore:
    def __init__(self, backend, bot_id: str, user_id: str):
        self.backend = backend
        self.policy = BotPolicyResolver(backend.engine).resolve_sync()
        self.eligibility = MemoryProvenance(ScopedMessageAuthority(
            PartitionScope(bot_id, self.policy.partitions[bot_id], "global"),
            user_id, self.policy.readable_bots,
        ))

    def list_memories(self, *, limit: int = 100, min_importance: float = 0.0) -> list[dict]:
        self.policy.assert_current()
        with self.backend.engine.connect() as conn:
            return [dict(row) for row in conn.execute(text(f"""
                SELECT id, content, tags, importance, source_message_ids
                FROM {self.backend._memories_table_name} mem
                WHERE {self.eligibility.predicate} AND importance >= :importance
                ORDER BY created_at DESC LIMIT :limit
            """), {**self.eligibility.params, "importance": min_importance,
                   "limit": limit}).mappings()]

    def add_memory(self, **kwargs):
        self.policy.assert_current()
        outcome = self.backend.add_memory(str(uuid4()), eligibility=self.eligibility, **kwargs)
        if outcome.status not in ("inserted", "updated", "duplicate"):
            raise RuntimeError(outcome.error or "extraction memory was not stored")
        return outcome.memory_id

    def update_memory(self, memory_id: str, **kwargs):
        self.policy.assert_current()
        return self.backend.update_memory(memory_id, eligibility=self.eligibility, **kwargs)

    def delete_memory(self, memory_id: str) -> bool:
        self.policy.assert_current()
        with self.backend._memory_write_transaction() as conn:
            return bool(conn.execute(text(f"""
                DELETE FROM {self.backend._memories_table_name} mem
                WHERE id = :id AND {self.eligibility.predicate}
            """), {**self.eligibility.params, "id": memory_id}).rowcount)
