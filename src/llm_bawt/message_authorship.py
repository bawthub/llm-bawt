"""Canonical message-author references and deterministic legacy resolution."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Iterable, Literal, Mapping

from sqlalchemy import text
from sqlalchemy.engine import Engine

EntityType = Literal["user", "bot", "system"]
AuthorStatus = Literal["canonical", "legacy", "unknown"]

# TASK-1807: first-party system authors. These are product identifiers, not
# deployment data: the app persists its own approval-continuation turns (MCP
# result envelopes, harness approval outcomes) under them so the rows are not
# attributed to the human user. Only the app may write them (trusted storage
# parameter in-process; signed service capability over MCP); caller-facing
# inputs keep ``normalize_author`` (user|bot only). The value is the
# presentation label.
SYSTEM_AUTHORS: Mapping[str, str] = MappingProxyType({
    "approval-continuation:mcp": "Tool result",
    "approval-continuation:harness": "Approval result",
})


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
    def system(cls, entity_id: str) -> "AuthorReference":
        normalized = normalize_entity_id(entity_id)
        if normalized not in SYSTEM_AUTHORS:
            raise ValueError("unknown system author")
        return cls("system", normalized)

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


def normalize_persisted_author(
    entity_type: str | None,
    entity_id: str | None,
    *,
    status: AuthorStatus = "canonical",
) -> AuthorReference | None:
    """Validate an author pair read back from a stored row (TASK-1807).

    Unlike :func:`normalize_author`, this also accepts the allowlisted
    first-party system authors the app persists. Never use it on caller
    input: it is the read-side counterpart, not a write authority.
    """
    if (entity_type or "").strip().lower() == "system":
        reference = AuthorReference.system(entity_id or "")
        return AuthorReference(reference.entity_type, reference.entity_id, status)
    return normalize_author(entity_type, entity_id, status=status)


def require_system_author(author: object) -> AuthorReference:
    """Accept only an allowlisted system ``AuthorReference`` (TASK-1807).

    The trusted write route's gate: anything else (user/bot authors, strings,
    unknown system ids) is a ValueError, so a trusted parameter can never be
    used to forge a human or bot author.
    """
    if not isinstance(author, AuthorReference) or author.entity_type != "system":
        raise ValueError("trusted author must be a system author")
    return AuthorReference.system(author.entity_id or "")


class EffectiveAuthorRule:
    """Retrieval authority, deliberately separate from legacy display hydration.

    One CASE template renders two fixed, trusted modes; neither accepts caller
    SQL. Outputs are ``'bot'``, ``'user'``, ``'delivery'`` or NULL (unknown).

    ``sql_case`` (scoped): callers bind the canonical partition bot, trusted
    session user, and CURRENT public author set. The surrounding scope
    compiler supplies the canonical session/user join.

    ``operator_case()`` (TASK-1301, operator search): the bot branch compares
    against fixed alias ``p.bot_id``, the canonical slug of the message's
    partition from the positive policy map (``m.bot_id = p.partition``), so
    session-less rows still classify. The human branch compares the explicit
    user author to ``s.user_id`` from a LEFT session join bound to
    ``s.id = m.session_id AND s.bot_id = p.bot_id``, with no trusted-user pin;
    a missing or foreign session yields NULL rather than human authority.
    The delivery branch keeps the bound ``:authority_authors`` set.
    """

    _CASE_TEMPLATE = """
        CASE
          WHEN m.role = 'assistant' AND (
            (COALESCE(m.author_entity_type, '') = ''
             AND COALESCE(m.author_entity_id, '') = '')
            OR (m.author_entity_type = 'bot' AND m.author_entity_id = {bot})
          ) THEN 'bot'
          WHEN m.role = 'user' AND m.author_entity_type = 'user'
            AND m.author_entity_id = s.user_id{user_pin} THEN 'user'
          WHEN m.role = 'user' AND m.author_entity_type = 'bot'
            AND m.author_entity_id = ANY(CAST(:authority_authors AS text[])) THEN 'delivery'
        END
    """

    sql_case = _CASE_TEMPLATE.format(
        bot=":authority_bot",
        user_pin="\n            AND m.author_entity_id = :authority_user",
    )
    _operator_case = _CASE_TEMPLATE.format(bot="p.bot_id", user_pin="")
    _aggregate_case = _CASE_TEMPLATE.format(
        bot="p.bot_id",
        user_pin="\n            AND m.author_entity_id = :authority_user",
    )

    @classmethod
    def operator_case(cls) -> str:
        """The operator-mode rendering of the same template (see class doc)."""
        return cls._operator_case

    @classmethod
    def aggregate_case(cls) -> str:
        """``sql_case`` over many partitions at once (TASK-1323 ``messages_find``).

        The bot branch compares against ``p.bot_id`` (the canonical slug the
        positive policy maps to ``m.bot_id = p.partition``); the human branch
        keeps the trusted ``:authority_user`` pin; deliveries keep the bound
        ``:authority_authors`` set.
        """
        return cls._aggregate_case


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
            explicit = normalize_persisted_author(
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
