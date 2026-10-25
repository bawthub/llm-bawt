"""Operator CRUD over the tag catalog and assignments (TASK-1329 S5).

Role tags are operator data: created, edited and deleted here. Behaviour tags
are code vocabulary (``system`` rows seeded from ``vocabulary.py``) and are
read-only. Assignments are written by the profile upsert
(:func:`catalog.profile_tag_write`) and removed one at a time here, which also
reaches the assignments of a bot whose profile row is gone (BOT_TAGS.md §6).
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from . import catalog
from .errors import TagConflict, TagNotFound, TagValidationError
from .vocabulary import TAG_SLUG, BehaviorTag, is_canonical_bot_slug

ROLE_FIELDS = frozenset({"name", "description", "prompt_key", "capabilities"})
# ``slug`` is editable too, as an atomic rename (TASK-1249); it is not a column write.
EDITABLE_FIELDS = ROLE_FIELDS | {"slug"}
MAX_NAME = 128
MAX_DESCRIPTION = 2000
MAX_PROMPT_KEY = 160
MAX_CAPABILITY = 64
MAX_INSTRUCTIONS = 20000
ROLE_PROMPT_PREFIX = "role."
# Registry category for tag instruction prompts (operator data, global scope).
INSTRUCTIONS_CATEGORY = "bot_tag"

_CATALOG_SQL = """
    SELECT c.slug, c.name, c.description, c.kind, c.prompt_key, c.capabilities, c.system,
           c.created_at, c.updated_at,
           COALESCE(json_agg(json_build_object('bot', t.bot_slug,
                                               'has_profile', p.slug IS NOT NULL)
                             ORDER BY t.bot_slug)
                    FILTER (WHERE t.bot_slug IS NOT NULL), '[]'::json) AS used_by
    FROM bot_tag_catalog c
    LEFT JOIN bot_profile_tags t ON t.tag_slug = c.slug
    LEFT JOIN bot_profiles p ON p.slug = t.bot_slug
    {where}
    GROUP BY c.slug
    ORDER BY c.kind, c.slug
"""


@dataclass(frozen=True)
class TagUse:
    bot: str
    has_profile: bool


@dataclass(frozen=True)
class CatalogTag:
    slug: str
    name: str
    description: str
    kind: str
    prompt_key: str | None
    capabilities: tuple[str, ...]
    system: bool
    created_at: datetime | None
    updated_at: datetime | None
    used_by: tuple[TagUse, ...]

    @classmethod
    def from_row(cls, row: Mapping) -> CatalogTag:
        used = row["used_by"]
        if isinstance(used, str):
            used = json.loads(used)
        return cls(
            slug=row["slug"], name=row["name"], description=row["description"], kind=row["kind"],
            prompt_key=row["prompt_key"], capabilities=tuple(row["capabilities"] or ()),
            system=bool(row["system"]), created_at=row["created_at"],
            updated_at=row["updated_at"],
            used_by=tuple(TagUse(u["bot"], bool(u["has_profile"])) for u in used))


@dataclass(frozen=True)
class AssignmentRemoved:
    tag: str
    bot: str
    has_profile: bool


class RoleFields:
    """Validation for the operator-editable fields of a role tag.

    Shape only: what a ``prompt_key`` or capability *means* belongs to
    TASK-1249/1250. Slugs are exact (no normalization).
    """

    @staticmethod
    def slug(value: object) -> str:
        if not isinstance(value, str) or not TAG_SLUG.fullmatch(value):
            raise TagValidationError(
                "tag slug must match ^[a-z][a-z0-9_]{0,63}$ exactly (no normalization)")
        return value

    @staticmethod
    def name(value: object) -> str:
        if not isinstance(value, str) or not value.strip() or len(value.strip()) > MAX_NAME:
            raise TagValidationError(f"name must be 1-{MAX_NAME} non-blank characters")
        return value.strip()

    @staticmethod
    def description(value: object) -> str:
        if not isinstance(value, str) or len(value) > MAX_DESCRIPTION:
            raise TagValidationError(f"description must be at most {MAX_DESCRIPTION} characters")
        return value

    @staticmethod
    def prompt_key(value: object) -> str | None:
        if value is None:
            return None
        if (not isinstance(value, str) or not value.startswith(ROLE_PROMPT_PREFIX)
                or len(value) <= len(ROLE_PROMPT_PREFIX) or len(value) > MAX_PROMPT_KEY
                or any(ch.isspace() for ch in value)):
            raise TagValidationError(
                f"prompt_key must be null or a '{ROLE_PROMPT_PREFIX}<name>' prompt key "
                f"of at most {MAX_PROMPT_KEY} characters")
        return value

    @staticmethod
    def capabilities(value: object) -> list[str]:
        if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
            raise TagValidationError("capabilities must be a list of strings")
        if not all(isinstance(c, str) and c and len(c) <= MAX_CAPABILITY
                   and c == c.strip() for c in value):
            raise TagValidationError(
                f"each capability must be a non-blank string of at most {MAX_CAPABILITY} "
                "characters with no surrounding whitespace")
        if len(set(value)) != len(value):
            raise TagValidationError("capabilities must not repeat")
        return list(value)

    @staticmethod
    def instructions(value: object) -> str:
        if not isinstance(value, str) or not value.strip() or len(value) > MAX_INSTRUCTIONS:
            raise TagValidationError(
                f"instructions must be 1-{MAX_INSTRUCTIONS} characters and not blank")
        return value


class BotTagStore:
    """Catalog and assignment writes on one engine; every write is one transaction."""

    def __init__(self, engine):
        if engine is None:
            raise RuntimeError("Bot tags DB unavailable")
        self.engine = engine

    # --- reads ------------------------------------------------------------

    def catalog(self) -> list[CatalogTag]:
        with self.engine.connect() as conn:
            rows = conn.execute(text(_CATALOG_SQL.format(where=""))).mappings()
            return [CatalogTag.from_row(row) for row in rows]

    def get(self, slug: str) -> CatalogTag:
        with self.engine.connect() as conn:
            return self._get(conn, slug)

    def tags_by_bot(self) -> dict[str, list[str]]:
        """Every assignment, including bots whose profile row is gone."""
        with self.engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT bot_slug, array_agg(tag_slug ORDER BY tag_slug) AS tags
                FROM bot_profile_tags GROUP BY bot_slug
            """)).mappings()
            return {row["bot_slug"]: list(row["tags"]) for row in rows}

    def tags_for(self, bot: str) -> list[str]:
        with self.engine.connect() as conn:
            return list(conn.execute(text("""
                SELECT tag_slug FROM bot_profile_tags WHERE bot_slug = :bot ORDER BY tag_slug
            """), {"bot": bot}).scalars())

    def capabilities_by_tag(self) -> dict[str, tuple[str, ...]]:
        """Each role tag's capability tokens. Code never interprets a token."""
        with self.engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT slug, capabilities FROM bot_tag_catalog WHERE kind = 'role' ORDER BY slug
            """)).mappings()
            return {row["slug"]: tuple(row["capabilities"] or ()) for row in rows}

    # --- role catalog writes ---------------------------------------------

    def create_role(self, slug: object, name: object, description: object = "",
                    prompt_key: object = None, capabilities: object = ()) -> CatalogTag:
        values = {
            "slug": RoleFields.slug(slug), "name": RoleFields.name(name),
            "description": RoleFields.description(description),
            "prompt_key": RoleFields.prompt_key(prompt_key),
            "capabilities": json.dumps(RoleFields.capabilities(capabilities)),
        }
        with self.engine.begin() as conn:
            created = conn.execute(text("""
                INSERT INTO bot_tag_catalog (slug, name, description, kind, prompt_key,
                                             capabilities, system)
                VALUES (:slug, :name, :description, 'role', :prompt_key,
                        CAST(:capabilities AS jsonb), FALSE)
                ON CONFLICT (slug) DO NOTHING
                RETURNING slug
            """), values).scalar_one_or_none()
            if created is None:
                raise TagConflict(f"bot tag {values['slug']!r} already exists")
            return self._get(conn, created)

    def update_role(self, slug: str, changes: Mapping[str, object]) -> CatalogTag:
        """Edit a role tag in one transaction; a ``slug`` change renames it first."""
        unknown = sorted(set(changes) - EDITABLE_FIELDS)
        if unknown:
            raise TagValidationError(f"not editable on a tag: {', '.join(unknown)}")
        values = {key: getattr(RoleFields, key)(value) for key, value in changes.items()}
        if "capabilities" in values:
            values["capabilities"] = json.dumps(values["capabilities"])
        new_slug = values.pop("slug", slug)
        with self.engine.begin() as conn:
            self._lock_role(conn, slug)
            if new_slug != slug:
                self._rename(conn, slug, new_slug)
                slug = new_slug
            if values:
                sets = ", ".join(
                    "capabilities = CAST(:capabilities AS jsonb)" if key == "capabilities"
                    else f"{key} = :{key}" for key in sorted(values))
                conn.execute(text(f"UPDATE bot_tag_catalog SET {sets}, updated_at = now() "
                                  "WHERE slug = :slug"), {**values, "slug": slug})
            return self._get(conn, slug)

    def rename_role(self, old: str, new: object) -> CatalogTag:
        """Atomic slug rename: assignments, ``prompt_key`` and capabilities move with it."""
        return self.update_role(old, {"slug": new})

    def save_instructions(self, slug: str, body: object, updated_by: str | None) -> CatalogTag:
        """Write a role tag's instructions to the prompt registry, in one transaction.

        The first save allocates an opaque ``prompt_key`` (``role.<slug>``, or the
        first free ``_N`` suffix). It is never re-derived from the slug, so a
        renamed tag keeps its key and its prompt's version history.
        """
        from sqlmodel import Session

        from ..prompt_registry import PromptKeyTaken, stage_prompt_upsert

        body = RoleFields.instructions(body)
        try:
            with Session(self.engine) as session:
                conn = session.connection()
                self._lock_role(conn, slug)
                row = conn.execute(text(
                    "SELECT name, prompt_key FROM bot_tag_catalog WHERE slug = :slug"
                ), {"slug": slug}).mappings().one()
                fresh = row["prompt_key"] is None
                key = self._allocate_prompt_key(conn, slug) if fresh else row["prompt_key"]
                stage_prompt_upsert(
                    session, key=key, body=body, scope_type="global",
                    title=f"Bot tag: {row['name']}", category=INSTRUCTIONS_CATEGORY,
                    updated_by=updated_by, change_note=f"Instructions for bot tag {slug}",
                    create_only=fresh)
                if fresh:
                    conn.execute(text("""
                        UPDATE bot_tag_catalog SET prompt_key = :key, updated_at = now()
                        WHERE slug = :slug
                    """), {"key": key, "slug": slug})
                session.commit()
        except (IntegrityError, PromptKeyTaken) as exc:
            # Another writer took the allocated key: refuse rather than overwrite its prompt.
            raise TagConflict(f"prompt key for bot tag {slug!r} was taken concurrently; "
                              "retry") from exc
        return self.get(slug)

    def clear_instructions(self, slug: str) -> CatalogTag:
        """Detach the tag's instructions; the registry prompt and its history stay."""
        with self.engine.begin() as conn:
            self._lock_role(conn, slug)
            conn.execute(text("""
                UPDATE bot_tag_catalog SET prompt_key = NULL, updated_at = now()
                WHERE slug = :slug AND prompt_key IS NOT NULL
            """), {"slug": slug})
            return self._get(conn, slug)

    def delete_role(self, slug: str) -> None:
        try:
            with self.engine.begin() as conn:
                self._lock_role(conn, slug)
                users = conn.execute(text("""
                    SELECT bot_slug FROM bot_profile_tags WHERE tag_slug = :slug
                    ORDER BY bot_slug
                """), {"slug": slug}).scalars().all()
                if users:
                    raise TagConflict(f"bot tag {slug!r} is assigned to {', '.join(users)}; "
                                      "remove those assignments first")
                conn.execute(text("DELETE FROM bot_tag_catalog WHERE slug = :slug"),
                             {"slug": slug})
        except IntegrityError as exc:
            # An assignment committed between the check and the delete: the FK
            # (ON DELETE RESTRICT) is the guarantee, this is its message.
            raise TagConflict(f"bot tag {slug!r} is assigned; remove those assignments "
                              "first") from exc

    # --- assignment writes -------------------------------------------------

    def unassign(self, tag: str, bot: str) -> AssignmentRemoved:
        """Remove one assignment, with or without a profile row.

        Removing ``private`` sets the legacy column to true (``not private``)
        in the same transaction. Any other untag leaves the column alone, so it
        can never loosen a column written false by a bypassing writer.
        """
        RoleFields.slug(tag)
        if not is_canonical_bot_slug(bot):
            raise TagValidationError("bot must be a canonical bot slug (lower-case, trimmed)")
        with self.engine.begin() as conn:
            catalog.lock_bot(conn, bot)
            removed = conn.execute(text("""
                DELETE FROM bot_profile_tags WHERE bot_slug = :bot AND tag_slug = :tag
                RETURNING tag_slug
            """), {"bot": bot, "tag": tag}).scalar_one_or_none()
            if removed is None:
                self._get(conn, tag)  # an unknown tag reports itself first
                raise TagNotFound(f"bot {bot!r} has no {tag!r} tag")
            has_profile = bool(conn.execute(text(
                "SELECT EXISTS (SELECT 1 FROM bot_profiles WHERE slug = :bot)"
            ), {"bot": bot}).scalar_one())
            if has_profile and tag == BehaviorTag.PRIVATE:
                conn.execute(text("""
                    UPDATE bot_profiles SET include_in_global_search = TRUE, updated_at = now()
                    WHERE slug = :bot
                """), {"bot": bot})
            return AssignmentRemoved(tag, bot, has_profile)

    # --- helpers -----------------------------------------------------------

    def _get(self, conn, slug: str) -> CatalogTag:
        row = conn.execute(text(_CATALOG_SQL.format(where="WHERE c.slug = :slug")),
                           {"slug": slug}).mappings().first()
        if row is None:
            raise TagNotFound(f"bot tag {slug!r} not found")
        return CatalogTag.from_row(row)

    @staticmethod
    def _rename(conn, old: str, new: str) -> None:
        """Insert the new row, move assignments onto it, delete the old row.

        The caller holds the old row ``FOR UPDATE``. Assignments move with an
        UPDATE that the FK accepts because the new row already exists.
        """
        created = conn.execute(text("""
            INSERT INTO bot_tag_catalog (slug, name, description, kind, prompt_key,
                                         capabilities, system, created_at, updated_at)
            SELECT :new, name, description, kind, prompt_key, capabilities, FALSE,
                   created_at, now()
            FROM bot_tag_catalog WHERE slug = :old
            ON CONFLICT (slug) DO NOTHING
            RETURNING slug
        """), {"old": old, "new": new}).scalar_one_or_none()
        if created is None:
            raise TagConflict(f"bot tag {new!r} already exists")
        conn.execute(text("UPDATE bot_profile_tags SET tag_slug = :new WHERE tag_slug = :old"),
                     {"old": old, "new": new})
        conn.execute(text("DELETE FROM bot_tag_catalog WHERE slug = :old"), {"old": old})

    @staticmethod
    def _allocate_prompt_key(conn, slug: str) -> str:
        """``role.<slug>`` unless a prompt or another tag holds it; then ``_2``, ``_3``…"""
        taken = set(conn.execute(text("""
            SELECT key FROM prompt_templates WHERE key LIKE 'role.%'
            UNION SELECT prompt_key FROM bot_tag_catalog WHERE prompt_key IS NOT NULL
        """)).scalars())
        base = f"{ROLE_PROMPT_PREFIX}{slug}"
        key, suffix = base, 2
        while key in taken:
            key, suffix = f"{base}_{suffix}", suffix + 1
        return key

    @staticmethod
    def _lock_role(conn, slug: str) -> None:
        """Lock the catalog row; refuse anything that is not an operator role."""
        row = conn.execute(text("""
            SELECT kind, system FROM bot_tag_catalog WHERE slug = :slug FOR UPDATE
        """), {"slug": slug}).mappings().first()
        if row is None:
            raise TagNotFound(f"bot tag {slug!r} not found")
        if row["system"] or row["kind"] != "role":
            raise TagConflict(f"bot tag {slug!r} is a behaviour tag defined by code; "
                              "it cannot be edited or deleted")
