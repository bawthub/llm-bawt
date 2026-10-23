"""Canonical message-author references and deterministic legacy resolution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Literal

from sqlalchemy import text
from sqlalchemy.engine import Engine

EntityType = Literal["user", "bot"]
AuthorStatus = Literal["canonical", "legacy", "unknown"]


@dataclass(frozen=True)
class AuthorReference:
    """Who produced a message, independent from its protocol role."""

    entity_type: EntityType | None
    entity_id: str | None
    status: AuthorStatus = "canonical"

    @classmethod
    def user(cls, entity_id: str) -> "AuthorReference":
        return cls("user", normalize_entity_id(entity_id))

    @classmethod
    def bot(cls, entity_id: str) -> "AuthorReference":
        return cls("bot", normalize_entity_id(entity_id))

    @classmethod
    def unknown(cls) -> "AuthorReference":
        return cls(None, None, "unknown")

    def to_dict(self) -> dict[str, str | None]:
        return {
            "entity_type": self.entity_type,
            "entity_id": self.entity_id,
            "status": self.status,
        }


def normalize_entity_id(value: str) -> str:
    normalized = (value or "").strip().lower()
    if not normalized:
        raise ValueError("author entity id is required")
    return normalized


def normalize_author(
    entity_type: str | None,
    entity_id: str | None,
    *,
    status: AuthorStatus = "canonical",
) -> AuthorReference | None:
    """Validate a nullable persisted author pair."""
    if entity_type is None and entity_id is None:
        return None
    normalized_type = (entity_type or "").strip().lower()
    if normalized_type not in {"user", "bot"}:
        raise ValueError("author entity type must be 'user' or 'bot'")
    return AuthorReference(
        normalized_type,  # type: ignore[arg-type]
        normalize_entity_id(entity_id or ""),
        status,
    )


class EffectiveAuthorRule:
    """Retrieval authority, deliberately separate from legacy display hydration.

    Callers bind the canonical partition bot, trusted session user, and CURRENT
    public author set. SQL aliases are fixed so no model input is SQL syntax.
    The surrounding scope compiler supplies the canonical session/user join.
    """

    sql_case = """
        CASE
          WHEN m.role = 'assistant' AND (
            (COALESCE(m.author_entity_type, '') = ''
             AND COALESCE(m.author_entity_id, '') = '')
            OR (m.author_entity_type = 'bot' AND m.author_entity_id = :authority_bot)
          ) THEN 'bot'
          WHEN m.role = 'user' AND m.author_entity_type = 'user'
            AND m.author_entity_id = s.user_id
            AND m.author_entity_id = :authority_user THEN 'user'
          WHEN m.role = 'user' AND m.author_entity_type = 'bot'
            AND m.author_entity_id = ANY(CAST(:authority_authors AS text[])) THEN 'delivery'
        END
    """


class LegacyAuthorResolver:
    """Batch-resolve missing authors without parsing message content."""

    def __init__(self, engine: Engine):
        self.engine = engine

    def resolve(
        self,
        messages: Iterable[dict[str, Any]],
        *,
        bot_id: str,
    ) -> dict[str, AuthorReference]:
        rows = list(messages)
        result: dict[str, AuthorReference] = {}
        unresolved: list[dict[str, Any]] = []

        for row in rows:
            message_id = str(row.get("id") or "")
            explicit = normalize_author(
                row.get("author_entity_type"),
                row.get("author_entity_id"),
            )
            if explicit is not None:
                result[message_id] = explicit
            else:
                unresolved.append(row)

        if not unresolved:
            return result

        unresolved_ids = [str(row.get("id") or "") for row in unresolved if row.get("id")]
        delivery_authors = self._delivery_authors(unresolved_ids)
        session_users = self._session_users(
            [str(row.get("session_id")) for row in unresolved if row.get("session_id")]
        )
        normalized_bot = normalize_entity_id(bot_id)

        for row in unresolved:
            message_id = str(row.get("id") or "")
            if message_id in delivery_authors:
                result[message_id] = AuthorReference(
                    "bot", delivery_authors[message_id], "legacy"
                )
            elif row.get("role") == "assistant":
                result[message_id] = AuthorReference("bot", normalized_bot, "legacy")
            elif row.get("role") == "user" and row.get("session_id") in session_users:
                result[message_id] = AuthorReference(
                    "user", session_users[str(row["session_id"])], "legacy"
                )
            else:
                result[message_id] = AuthorReference.unknown()
        return result

    def _delivery_authors(self, message_ids: list[str]) -> dict[str, str]:
        if not message_ids:
            return {}
        try:
            with self.engine.connect() as conn:
                rows = conn.execute(
                    text(
                        "SELECT user_message_id, sender_bot_id "
                        "FROM inter_bot_deliveries "
                        "WHERE user_message_id = ANY(:ids)"
                    ),
                    {"ids": message_ids},
                ).fetchall()
            return {
                str(row.user_message_id): normalize_entity_id(row.sender_bot_id)
                for row in rows
                if row.sender_bot_id
            }
        except Exception:
            return {}

    def _session_users(self, session_ids: list[str]) -> dict[str, str]:
        unique_ids = sorted(set(session_ids))
        if not unique_ids:
            return {}
        with self.engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT id, user_id FROM sessions "
                    "WHERE id = ANY(:ids) AND user_id IS NOT NULL"
                ),
                {"ids": unique_ids},
            ).fetchall()
        return {
            str(row.id): normalize_entity_id(row.user_id)
            for row in rows
            if row.user_id
        }
