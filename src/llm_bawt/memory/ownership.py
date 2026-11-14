"""Immutable per-call memory authority; never attach request users to a backend.

These predicates supplement (never replace) bot-access/author eligibility. The
activation lookup stays in each SQL statement, including the final mutation:
activating a bot must not leave a cached backend with legacy permissions.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from sqlalchemy import text

from llm_bawt.memory.message_store import FORGOTTEN_PARENT
from llm_bawt.memory.ownership_migration import ACTIVATIONS
from llm_bawt.utils.bot_partition import sanitize_table_name


@dataclass(frozen=True)
class MemoryAuthority:
    user_id: str | None = None
    operator: bool = False

    def __post_init__(self) -> None:
        if self.user_id is not None and (
            not isinstance(self.user_id, str) or not self.user_id.strip()
            or self.user_id != self.user_id.strip()
        ):
            raise ValueError("memory owner must be a nonblank user identity")
        if self.operator and self.user_id is not None:
            raise ValueError("operator and personal memory authority are exclusive")

    def scope(self, bot_id: str) -> MemoryOwnership:
        return MemoryOwnership(bot_id, self)


# Omission is not operator authority. Unmarked bots still retain legacy behavior.
NO_MEMORY_AUTHORITY = MemoryAuthority()
OPERATOR_MEMORY_AUTHORITY = MemoryAuthority(operator=True)


@dataclass(frozen=True)
class MemoryOwnership:
    bot_id: str
    authority: MemoryAuthority

    @property
    def params(self) -> dict:
        return {
            "memory_owner_bot": self.bot_id,
            "memory_owner_partition": sanitize_table_name(self.bot_id),
            "memory_owner_user": self.authority.user_id,
        }

    @property
    def active(self) -> str:
        return f"""EXISTS (SELECT 1 FROM {ACTIVATIONS} ownership_activation
            WHERE ownership_activation.bot_id=:memory_owner_partition)"""

    @property
    def read_predicate(self) -> str:
        if self.authority.operator:
            return "TRUE"
        # Existing ambiguous NULL rows keep the caller's previous read scope.
        # No-user callers cannot read any row once strict mode is active.
        return f"""(NOT ({self.active}) OR
            (CAST(:memory_owner_user AS text) IS NOT NULL AND
             (mem.owner_user_id IS NULL OR mem.owner_user_id=:memory_owner_user)))"""

    @property
    def message_read_predicate(self) -> str:
        """Keep convenience-memory search/stat fallbacks in the same user scope."""
        if self.authority.operator:
            return "TRUE"
        return f"""(NOT ({self.active}) OR session_id IN (
            SELECT s.id FROM sessions s
            WHERE s.bot_id=:memory_owner_bot AND s.user_id=:memory_owner_user))"""

    def sources_predicate(self, *, include_forgotten: bool = False,
                          deletion_only: bool = False) -> str:
        # Forgetting follows session ownership, including guessed legacy sessions.
        # Those sources must never become proof for writes, dedup or by-ID CRUD.
        proof = "" if deletion_only else """
                      AND COALESCE(s.session_metadata->>'legacy_backfill', 'false')<>'true'
                      AND EXISTS (SELECT 1 FROM entity_profiles p
                          WHERE lower(p.entity_type::text)='user' AND p.entity_id=s.user_id
                            AND NOT p.is_service)
                      AND NOT EXISTS (SELECT 1 FROM entity_profiles p
                          WHERE lower(p.entity_type::text)='user' AND p.entity_id=s.user_id
                            AND p.is_service)"""
        sources = """CASE WHEN jsonb_typeof(mem.source_message_ids::jsonb)='array'
            THEN mem.source_message_ids::jsonb ELSE '[]'::jsonb END"""
        messages = "messages"
        if include_forgotten:
            messages = f"""(SELECT id, bot_id, session_id FROM messages UNION ALL
                SELECT id, bot_id, session_id FROM {FORGOTTEN_PARENT})"""
        return f"""(
            mem.bot_id=:memory_owner_partition
            AND CAST(:memory_owner_user AS text) IS NOT NULL
            AND jsonb_array_length({sources}) > 0
            AND NOT EXISTS (
                SELECT 1 FROM jsonb_array_elements({sources}) src(value)
                WHERE jsonb_typeof(src.value)<>'string' OR NOT EXISTS (
                    SELECT 1 FROM {messages} m JOIN sessions s ON s.id=m.session_id
                    WHERE m.id=(src.value #>> '{{}}')
                      AND m.bot_id=:memory_owner_partition
                      AND s.bot_id=:memory_owner_bot AND s.user_id=:memory_owner_user
                      AND s.status IN ('active', 'archived')
                      {proof}
                )
            )
        )"""

    @property
    def owned_predicate(self) -> str:
        return f"""(CAST(:memory_owner_user AS text) IS NOT NULL AND
            (mem.owner_user_id=:memory_owner_user OR
             (mem.owner_user_id IS NULL AND {self.sources_predicate(include_forgotten=True)})))"""

    @property
    def mutation_predicate(self) -> str:
        if self.authority.operator:
            return "TRUE"
        return f"(NOT ({self.active}) OR {self.owned_predicate})"

    @property
    def dedup_group_predicate(self) -> str:
        """Operator visibility is not permission to merge distinct owners.

        NULL is its own group, including for personal edits of proven legacy
        rows. The target owner is pinned by the caller's write transaction.
        """
        return f"""(NOT ({self.active}) OR
            mem.owner_user_id IS NOT DISTINCT FROM CAST(:memory_dedup_owner AS text))"""

    def validate_write(self, conn, source_message_ids: list[str] | None) -> None:
        """Require strict human provenance for new supplied live sources."""
        self._validate_sources(conn, source_message_ids, deletion_only=False)

    def validate_forget(self, conn, source_message_ids: list[str]) -> None:
        """Deletion-only permission, not ownership proof for other mutations."""
        self._validate_sources(conn, source_message_ids, deletion_only=True)

    def _validate_sources(self, conn, source_message_ids: list[str] | None, *,
                          deletion_only: bool) -> None:
        if self.authority.operator:
            return
        active = conn.execute(text(f"SELECT {self.active}"), self.params).scalar_one()
        if not active:
            return
        if self.authority.user_id is None:
            raise ValueError("strict memory writes require personal or explicit operator authority")
        if source_message_ids is None or source_message_ids == []:
            return  # A new manual memory is owned by the trusted caller.
        if not isinstance(source_message_ids, list) or any(
            not isinstance(source, str) or not source for source in source_message_ids
        ):
            raise ValueError("memory sources must be a list of nonblank message IDs")
        # Retain proven provenance until commit, even if another transaction
        # tries to delete a message or reassign its session after validation.
        if not deletion_only:
            conn.execute(text("""
                SELECT p.entity_id FROM entity_profiles p
                WHERE lower(p.entity_type::text)='user' AND p.entity_id=:memory_owner_user
                FOR SHARE OF p
            """), self.params).all()
        tables = ["messages", FORGOTTEN_PARENT] if deletion_only else ["messages"]
        for table in tables:
            conn.execute(text(f"""
                SELECT m.id FROM {table} m JOIN sessions s ON s.id=m.session_id
                WHERE m.bot_id=:memory_owner_partition
                  AND m.id=ANY(CAST(:memory_source_ids AS text[]))
                FOR SHARE OF m, s
            """), {**self.params, "memory_source_ids": source_message_ids}).all()
        valid = conn.execute(text(f"""
            SELECT 1 FROM (SELECT CAST(:memory_source_ids AS jsonb) AS source_message_ids,
                                  CAST(:memory_owner_partition AS text) AS bot_id) mem
            WHERE {self.sources_predicate(include_forgotten=deletion_only, deletion_only=deletion_only)}
        """), {**self.params, "memory_source_ids": json.dumps(source_message_ids)}).first()
        if valid is None:
            raise ValueError("memory sources do not belong to the authenticated owner")
