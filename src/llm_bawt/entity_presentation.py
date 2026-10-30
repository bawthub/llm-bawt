"""Batched presentation resolution for user and bot message authors."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Iterable

from sqlalchemy import text
from sqlalchemy.engine import Engine

from .message_authorship import SYSTEM_AUTHORS, AuthorReference, normalize_entity_id

logger = logging.getLogger(__name__)

_DEFAULT_COLORS = (
    "cyan",
    "emerald",
    "violet",
    "amber",
    "rose",
    "sky",
    "teal",
    "indigo",
)


@dataclass(frozen=True)
class EntityPresentation:
    entity_type: str | None
    entity_id: str | None
    display_name: str
    color: str
    avatar: str | None
    avatar_render: str | None
    status: str

    def to_dict(self) -> dict[str, str | None]:
        return {
            "entity_type": self.entity_type,
            "entity_id": self.entity_id,
            "display_name": self.display_name,
            "color": self.color,
            "avatar": self.avatar,
            "avatar_render": self.avatar_render,
            "status": self.status,
        }


class EntityPresentationResolver:
    """Resolve author references with two batched queries, never per message."""

    def __init__(self, engine: Engine):
        self.engine = engine

    def resolve_many(
        self, authors: Iterable[AuthorReference]
    ) -> dict[tuple[str | None, str | None], EntityPresentation]:
        unique = {
            (author.entity_type, author.entity_id): author
            for author in authors
        }
        user_ids = sorted({entity_id for entity_type, entity_id in unique if entity_type == "user" and entity_id})
        bot_ids = sorted({entity_id for entity_type, entity_id in unique if entity_type == "bot" and entity_id})
        profiles = self._profiles(user_ids + bot_ids)
        bots = self._bots(bot_ids)
        result: dict[tuple[str | None, str | None], EntityPresentation] = {}

        for key, author in unique.items():
            entity_type, entity_id = key
            if entity_type is None or entity_id is None:
                result[key] = self._unknown(author)
                continue
            if entity_type == "system":
                result[key] = self._system(author)
                continue

            profile = profiles.get((entity_type, entity_id), {})
            bot = bots.get(entity_id, {}) if entity_type == "bot" else {}
            exists = bool(profile or bot)
            display_name = str(
                profile.get("display_name")
                or bot.get("name")
                or entity_id.replace("-", " ").replace("_", " ").title()
            )
            result[key] = EntityPresentation(
                entity_type=entity_type,
                entity_id=entity_id,
                display_name=display_name if exists else f"Unknown {entity_type}",
                color=str(profile.get("color") or bot.get("color") or self._stable_color(entity_type, entity_id)),
                avatar=profile.get("avatar") or bot.get("avatar"),
                avatar_render=profile.get("avatar_render") or bot.get("avatar_render"),
                status=author.status if exists else "unknown",
            )
        return result

    def hydrate(self, author_payloads: Iterable[dict]) -> list[dict]:
        references = [
            AuthorReference(
                payload.get("entity_type"),
                payload.get("entity_id"),
                payload.get("status") or "canonical",
            )
            for payload in author_payloads
        ]
        resolved = self.resolve_many(references)
        return [
            resolved[(reference.entity_type, reference.entity_id)].to_dict()
            for reference in references
        ]

    def resolve_many_safe(
        self, authors: Iterable[AuthorReference]
    ) -> dict[tuple[str | None, str | None], dict]:
        """Resolve live-event authors with truthful neutral fallbacks."""
        references = list(authors)
        try:
            return {
                key: value.to_dict()
                for key, value in self.resolve_many(references).items()
            }
        except Exception as exc:
            logger.warning("Message author presentation lookup failed: %s", exc)
            result = {}
            for author in references:
                fallback = self._unknown(author).to_dict()
                fallback["entity_type"] = author.entity_type
                fallback["entity_id"] = author.entity_id
                result[(author.entity_type, author.entity_id)] = fallback
            return result

    def resolve_one(self, author: AuthorReference) -> dict:
        return self.resolve_many_safe([author])[(author.entity_type, author.entity_id)]

    def _profiles(self, entity_ids: list[str]) -> dict[tuple[str, str], dict]:
        if not entity_ids:
            return {}
        with self.engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT entity_type::text AS entity_type, entity_id, display_name, "
                    "color, avatar, avatar_render FROM entity_profiles "
                    "WHERE entity_id = ANY(:ids)"
                ),
                {"ids": entity_ids},
            ).mappings().all()
        return {
            (
                str(row["entity_type"]).strip().lower(),
                normalize_entity_id(row["entity_id"]),
            ): dict(row)
            for row in rows
        }

    def _bots(self, bot_ids: list[str]) -> dict[str, dict]:
        if not bot_ids:
            return {}
        with self.engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT slug, name, color, avatar, avatar_render "
                    "FROM bot_profiles WHERE slug = ANY(:ids)"
                ),
                {"ids": bot_ids},
            ).mappings().all()
        return {normalize_entity_id(row["slug"]): dict(row) for row in rows}

    @staticmethod
    def _stable_color(entity_type: str, entity_id: str) -> str:
        digest = hashlib.sha256(f"{entity_type}:{entity_id}".encode()).digest()
        return _DEFAULT_COLORS[digest[0] % len(_DEFAULT_COLORS)]

    def _system(self, author: AuthorReference) -> EntityPresentation:
        """TASK-1807: fixed first-party label; no profile lookup."""
        label = SYSTEM_AUTHORS.get(author.entity_id or "")
        if label is None:
            return self._unknown(author)
        return EntityPresentation(
            entity_type="system",
            entity_id=author.entity_id,
            display_name=label,
            color="slate",
            avatar=None,
            avatar_render=None,
            status=author.status,
        )

    def _unknown(self, author: AuthorReference) -> EntityPresentation:
        return EntityPresentation(
            entity_type=author.entity_type,
            entity_id=author.entity_id,
            display_name="Unknown participant",
            color="slate",
            avatar=None,
            avatar_render=None,
            status="unknown",
        )
