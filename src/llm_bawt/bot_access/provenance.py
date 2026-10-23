"""Parameterized SQL eligibility, applied before ranking, LIMIT, or access writes."""
from __future__ import annotations

from dataclasses import dataclass

from llm_bawt.message_authorship import EffectiveAuthorRule

from .policy import AggregateScope, BotAccessError, PartitionScope


@dataclass(frozen=True)
class ScopedMessageAuthority:
    partition: PartitionScope
    user_id: str
    public_authors: frozenset[str]

    @classmethod
    def for_partition(cls, scope: AggregateScope, bot_id: str) -> ScopedMessageAuthority:
        if not scope.user_id:
            raise BotAccessError("caller_unavailable")
        for partition in scope.partitions:
            if partition.bot_id == bot_id:
                return cls(partition, scope.user_id, scope.public_authors)
        raise BotAccessError("access_denied")

    @property
    def params(self) -> dict:
        return {
            "authority_bot": self.partition.bot_id,
            "authority_partition": self.partition.partition,
            "authority_user": self.user_id,
            "authority_authors": sorted(self.public_authors),
        }

    def predicate(self, *, history: bool = False) -> str:
        base = """
            m.bot_id = :authority_partition AND s.id = m.session_id
            AND s.bot_id = :authority_bot AND s.user_id = :authority_user
            AND s.status IN ('active', 'archived')
        """
        if self.partition.authority == "global":
            return base + f" AND ({EffectiveAuthorRule.sql_case}) IS NOT NULL"
        # Exact MESSAGE_AUTHORITY semantics for self memory: no blank-author
        # exception and no imported-delivery admission. History alone widens.
        authors = """
            (m.author_entity_type = 'user' AND m.author_entity_id = :authority_user)
            OR (m.author_entity_type = 'bot' AND m.author_entity_id = :authority_bot)
        """
        if history:
            authors += """
                OR (m.role = 'user' AND m.author_entity_type = 'bot'
                    AND m.author_entity_id = ANY(CAST(:authority_authors AS text[])))
            """
        return base + f" AND m.role IN ('user', 'assistant') AND ({authors})"


@dataclass(frozen=True)
class MemoryProvenance:
    authority: ScopedMessageAuthority

    @property
    def params(self) -> dict:
        return self.authority.params

    @property
    def predicate(self) -> str:
        # CASE, not predicate ordering, makes malformed JSON safe regardless of
        # PostgreSQL's evaluation order. Alias mem belongs to the candidate row.
        sources = """CASE WHEN jsonb_typeof(mem.source_message_ids) = 'array'
                          THEN mem.source_message_ids ELSE '[]'::jsonb END"""
        return f"""
            mem.bot_id = :authority_partition
            AND jsonb_array_length({sources}) BETWEEN 1 AND 64
            AND NOT EXISTS (
                SELECT 1 FROM jsonb_array_elements({sources}) AS src(value)
                WHERE jsonb_typeof(src.value) <> 'string' OR NOT EXISTS (
                    SELECT 1 FROM messages m JOIN sessions s ON s.id = m.session_id
                    WHERE m.id = (src.value #>> '{{}}')
                      AND {self.authority.predicate()}
                )
            )
        """
