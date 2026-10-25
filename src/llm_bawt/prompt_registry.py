"""DB-backed prompt templates with code-default fallback."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from string import Formatter
from typing import Any

from sqlalchemy import Column, DateTime, String, Text, UniqueConstraint
from sqlmodel import Field, SQLModel, Session, select

# Re-exported for existing importers; the catalog lives in prompt_definitions.
from .chat_prompt_defaults import (  # noqa: F401
    AGENT_GLOBAL_PROMPT,
    RUNTIME_CONTEXT_TEMPLATE,
)
from .prompt_definitions import (  # noqa: F401
    COMPACT_SUMMARY_KEY,
    DEFAULT_PROMPT_DEFINITIONS,
    PromptDefinition,
)
from .utils.config import Config, has_database_credentials
from .utils.schema import SchemaBootstrapGuard

logger = logging.getLogger(__name__)

#: Category used to group switchable persona prompts in the registry, keeping
#: them separate from summarization/memory/agent/chat prompts. See TASK-477.
PERSONA_CATEGORY = "persona"
#: Key prefix for persona rows (e.g. ``persona.grumpy-editor``).
PERSONA_KEY_PREFIX = "persona."


def _slugify(text: str) -> str:
    """Lowercase, hyphenated slug from arbitrary text (ASCII word chars only)."""
    import re

    slug = re.sub(r"[^a-z0-9]+", "-", (text or "").strip().lower()).strip("-")
    return slug or "persona"


class PromptTemplate(SQLModel, table=True):
    """Stored active prompt template."""

    __tablename__ = "prompt_templates"
    __table_args__ = (
        UniqueConstraint("key", "scope_type", "scope_id", name="uq_prompt_templates_scope_key"),
    )

    id: int | None = Field(default=None, primary_key=True)
    key: str = Field(sa_column=Column(String(255), nullable=False, index=True))
    scope_type: str = Field(sa_column=Column(String(16), nullable=False, index=True))
    scope_id: str = Field(sa_column=Column(String(128), nullable=False, index=True))
    title: str | None = Field(default=None, sa_column=Column(String(255), nullable=True))
    category: str | None = Field(default=None, sa_column=Column(String(64), nullable=True, index=True))
    format: str = Field(default="plain_text", sa_column=Column(String(32), nullable=False))
    body: str = Field(sa_column=Column(Text, nullable=False))
    required_vars_json: str | None = Field(default=None, sa_column=Column(Text, nullable=True))
    metadata_json: str | None = Field(default=None, sa_column=Column(Text, nullable=True))
    updated_by: str | None = Field(default=None, sa_column=Column(String(255), nullable=True))
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )


class PromptTemplateVersion(SQLModel, table=True):
    """Append-only prompt version history."""

    __tablename__ = "prompt_template_versions"

    id: int | None = Field(default=None, primary_key=True)
    key: str = Field(sa_column=Column(String(255), nullable=False, index=True))
    scope_type: str = Field(sa_column=Column(String(16), nullable=False, index=True))
    scope_id: str = Field(sa_column=Column(String(128), nullable=False, index=True))
    version: int = Field(default=1)
    body: str = Field(sa_column=Column(Text, nullable=False))
    change_note: str | None = Field(default=None, sa_column=Column(Text, nullable=True))
    created_by: str | None = Field(default=None, sa_column=Column(String(255), nullable=True))
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )


@dataclass(frozen=True)
class ResolvedPrompt:
    """Resolved prompt template payload."""

    key: str
    title: str
    category: str
    format: str
    body: str
    required_vars: list[str]
    metadata: dict[str, Any]
    scope_type: str
    scope_id: str
    source: str
    updated_at: datetime | None = None




def _normalize_scope(scope_type: str = "global", scope_id: str | None = None) -> tuple[str, str]:
    normalized_type = (scope_type or "global").strip().lower() or "global"
    if normalized_type not in {"global", "bot"}:
        raise ValueError("scope_type must be 'global' or 'bot'")
    if normalized_type == "global":
        return ("global", "*")
    normalized_id = (scope_id or "").strip().lower()
    if not normalized_id:
        raise ValueError("scope_id is required for bot-scoped prompts")
    return ("bot", normalized_id)


def _parse_json_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    return [str(item) for item in data if str(item).strip()]


def _parse_json_dict(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def extract_placeholders(body: str) -> list[str]:
    """Extract format placeholders from a prompt template."""
    names: list[str] = []
    for _, field_name, _, _ in Formatter().parse(body or ""):
        if not field_name:
            continue
        base_name = field_name.split("!", 1)[0].split(":", 1)[0]
        if base_name and base_name not in names:
            names.append(base_name)
    return names


class PromptKeyTaken(Exception):
    """A create-only prompt write found its key already taken."""


def stage_prompt_upsert(
    session: Session,
    *,
    key: str,
    body: str,
    scope_type: str = "global",
    scope_id: str | None = None,
    title: str | None = None,
    category: str | None = None,
    format: str = "plain_text",
    required_vars: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    updated_by: str | None = None,
    change_note: str | None = None,
    create_only: bool = False,
) -> PromptTemplate:
    """Stage one prompt write and its version row in ``session``; never commits.

    :meth:`PromptTemplateStore.upsert` commits it alone. A caller that must
    write a prompt together with its own rows (TASK-1249: a bot tag's
    instructions and its ``prompt_key``) stages it in that caller's transaction.
    ``create_only`` raises :class:`PromptKeyTaken` instead of overwriting a row
    that appeared after the caller chose the key; a concurrent insert of the
    same key still fails on the unique constraint at commit.
    """
    normalized_type, normalized_id = _normalize_scope(scope_type, scope_id)
    required_vars = [str(item) for item in (required_vars or []) if str(item).strip()]
    metadata = metadata or {}
    now = datetime.now(timezone.utc)
    row = session.exec(
        select(PromptTemplate).where(
            PromptTemplate.key == key,
            PromptTemplate.scope_type == normalized_type,
            PromptTemplate.scope_id == normalized_id,
        )
    ).first()
    if create_only and row is not None:
        raise PromptKeyTaken(key)
    version_statement = select(PromptTemplateVersion).where(
        PromptTemplateVersion.key == key,
        PromptTemplateVersion.scope_type == normalized_type,
        PromptTemplateVersion.scope_id == normalized_id,
    )
    existing_versions = list(session.exec(version_statement).all())
    next_version = (max((v.version for v in existing_versions), default=0) + 1)

    if row is None:
        row = PromptTemplate(
            key=key,
            scope_type=normalized_type,
            scope_id=normalized_id,
            title=title,
            category=category,
            format=format,
            body=body,
            required_vars_json=json.dumps(required_vars) if required_vars else None,
            metadata_json=json.dumps(metadata) if metadata else None,
            updated_by=updated_by,
            created_at=now,
            updated_at=now,
        )
    else:
        row.title = title
        row.category = category
        row.format = format
        row.body = body
        row.required_vars_json = json.dumps(required_vars) if required_vars else None
        row.metadata_json = json.dumps(metadata) if metadata else None
        row.updated_by = updated_by
        row.updated_at = now

    session.add(row)
    session.add(
        PromptTemplateVersion(
            key=key,
            scope_type=normalized_type,
            scope_id=normalized_id,
            version=next_version,
            body=body,
            change_note=change_note,
            created_by=updated_by,
            created_at=now,
        )
    )
    return row


class PromptTemplateStore:
    """Database access for prompt templates."""

    _seeded_urls: set[str] = set()
    _schema_guard = SchemaBootstrapGuard()

    def __init__(self, config: Config):
        self.config = config
        self.engine = None
        self.connection_url: str | None = None
        if not has_database_credentials(config):
            return
        try:
            from .utils.db import build_postgres_url, get_shared_engine
            self.connection_url = build_postgres_url(config)
            # Process-wide shared engine (TASK-202).
            self.engine = get_shared_engine(config)
            if self.engine is None:
                return
            self._ensure_tables_exist()
        except Exception as e:
            self.engine = None
            logger.warning("Prompt template DB unavailable: %s", e)
            return

        try:
            self._seed_defaults_once()
        except Exception as e:
            logger.warning("Prompt template default seeding skipped: %s", e)

    def _ensure_tables_exist(self) -> None:
        if self.engine is None:
            return
        def bootstrap(conn) -> None:
            SQLModel.metadata.create_all(
                bind=conn,
                tables=[
                    PromptTemplate.__table__,
                    PromptTemplateVersion.__table__,
                ],
            )

        self._schema_guard.run(self.engine, "prompt-template-store", bootstrap)

    def _seed_defaults_once(self) -> None:
        if self.engine is None or not self.connection_url:
            return
        if self.connection_url in self._seeded_urls:
            return
        result = self.seed_defaults()
        self._seeded_urls.add(self.connection_url)
        if result["created"] > 0:
            logger.info("Seeded %s default prompt templates into DB", result["created"])

    def seed_defaults(self) -> dict[str, Any]:
        """Insert built-in prompt defaults into DB when missing."""
        if self.engine is None:
            raise RuntimeError("Prompt template DB unavailable")

        definitions = DEFAULT_PROMPT_DEFINITIONS
        global_scope_type = "global"
        global_scope_id = "*"
        existing_keys: set[str] = set()

        with Session(self.engine) as session:
            statement = select(PromptTemplate.key).where(
                PromptTemplate.scope_type == global_scope_type,
                PromptTemplate.scope_id == global_scope_id,
            )
            existing_keys = {str(key) for key in session.exec(statement).all()}

            now = datetime.now(timezone.utc)
            created = 0
            skipped = 0
            seeded_keys: list[str] = []

            for key, definition in definitions.items():
                if key in existing_keys:
                    skipped += 1
                    continue

                body = definition.default_body()
                metadata = {"seeded_from": "code_default"}

                session.add(
                    PromptTemplate(
                        key=key,
                        scope_type=global_scope_type,
                        scope_id=global_scope_id,
                        title=definition.title,
                        category=definition.category,
                        format=definition.format,
                        body=body,
                        required_vars_json=json.dumps(list(definition.required_vars))
                        if definition.required_vars
                        else None,
                        metadata_json=json.dumps(metadata),
                        updated_by="system:seed",
                        created_at=now,
                        updated_at=now,
                    )
                )
                session.add(
                    PromptTemplateVersion(
                        key=key,
                        scope_type=global_scope_type,
                        scope_id=global_scope_id,
                        version=1,
                        body=body,
                        change_note="Seeded from code default",
                        created_by="system:seed",
                        created_at=now,
                    )
                )
                created += 1
                seeded_keys.append(key)

            if created > 0:
                session.commit()

        total = len(definitions)
        return {
            "created": created,
            "skipped": skipped,
            "total": total,
            "seeded_keys": seeded_keys,
        }

    def get_exact(self, key: str, scope_type: str = "global", scope_id: str | None = None) -> PromptTemplate | None:
        if self.engine is None:
            return None
        normalized_type, normalized_id = _normalize_scope(scope_type, scope_id)
        with Session(self.engine) as session:
            return session.exec(
                select(PromptTemplate).where(
                    PromptTemplate.key == key,
                    PromptTemplate.scope_type == normalized_type,
                    PromptTemplate.scope_id == normalized_id,
                )
            ).first()

    def list_rows(
        self,
        *,
        category: str | None = None,
        scope_type: str | None = None,
        scope_id: str | None = None,
    ) -> list[PromptTemplate]:
        if self.engine is None:
            return []
        statement = select(PromptTemplate)
        if category:
            statement = statement.where(PromptTemplate.category == category.strip().lower())
        if scope_type:
            normalized_type, normalized_id = _normalize_scope(scope_type, scope_id)
            statement = statement.where(
                PromptTemplate.scope_type == normalized_type,
                PromptTemplate.scope_id == normalized_id,
            )
        statement = statement.order_by(PromptTemplate.category, PromptTemplate.key, PromptTemplate.scope_type, PromptTemplate.scope_id)
        with Session(self.engine) as session:
            return list(session.exec(statement).all())

    def upsert(
        self,
        *,
        key: str,
        body: str,
        scope_type: str = "global",
        scope_id: str | None = None,
        title: str | None = None,
        category: str | None = None,
        format: str = "plain_text",
        required_vars: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        updated_by: str | None = None,
        change_note: str | None = None,
    ) -> PromptTemplate:
        if self.engine is None:
            raise RuntimeError("Prompt template DB unavailable")
        with Session(self.engine) as session:
            row = stage_prompt_upsert(
                session, key=key, body=body, scope_type=scope_type, scope_id=scope_id,
                title=title, category=category, format=format, required_vars=required_vars,
                metadata=metadata, updated_by=updated_by, change_note=change_note,
            )
            session.commit()
            session.refresh(row)
            return row

    def reset(self, key: str, scope_type: str = "global", scope_id: str | None = None) -> bool:
        if self.engine is None:
            raise RuntimeError("Prompt template DB unavailable")
        normalized_type, normalized_id = _normalize_scope(scope_type, scope_id)
        with Session(self.engine) as session:
            row = session.exec(
                select(PromptTemplate).where(
                    PromptTemplate.key == key,
                    PromptTemplate.scope_type == normalized_type,
                    PromptTemplate.scope_id == normalized_id,
                )
            ).first()
            if row is None:
                return False
            session.delete(row)
            session.commit()
            return True

    def list_versions(self, key: str, scope_type: str = "global", scope_id: str | None = None) -> list[PromptTemplateVersion]:
        if self.engine is None:
            return []
        normalized_type, normalized_id = _normalize_scope(scope_type, scope_id)
        with Session(self.engine) as session:
            statement = (
                select(PromptTemplateVersion)
                .where(
                    PromptTemplateVersion.key == key,
                    PromptTemplateVersion.scope_type == normalized_type,
                    PromptTemplateVersion.scope_id == normalized_id,
                )
                .order_by(PromptTemplateVersion.version.desc())
            )
            return list(session.exec(statement).all())

    # ── Persona helpers (TASK-477) ─────────────────────────────────────
    # Personas are ordinary prompt_templates rows with category='persona'
    # and global scope, so they inherit versioning + resolution for free.
    # They are shareable across bots; a bot points at one via
    # bot_profiles.prompt_override_id (== PromptTemplate.id).

    def get_by_id(self, template_id: int) -> PromptTemplate | None:
        if self.engine is None or template_id is None:
            return None
        with Session(self.engine) as session:
            return session.get(PromptTemplate, template_id)

    def list_personas(self) -> list[PromptTemplate]:
        """All global persona prompts, ordered by title/key."""
        rows = self.list_rows(category=PERSONA_CATEGORY, scope_type="global")
        return sorted(rows, key=lambda r: ((r.title or r.key or "").lower(), r.id or 0))

    def create_persona(
        self,
        *,
        title: str,
        body: str,
        updated_by: str | None = None,
    ) -> PromptTemplate:
        """Create a new global persona, generating a unique persona.<slug> key."""
        if self.engine is None:
            raise RuntimeError("Prompt template DB unavailable")
        base_slug = _slugify(title)
        existing_keys = {r.key for r in self.list_rows(category=PERSONA_CATEGORY, scope_type="global")}
        key = f"{PERSONA_KEY_PREFIX}{base_slug}"
        suffix = 2
        while key in existing_keys:
            key = f"{PERSONA_KEY_PREFIX}{base_slug}-{suffix}"
            suffix += 1
        return self.upsert(
            key=key,
            body=body,
            scope_type="global",
            title=title.strip() or key,
            category=PERSONA_CATEGORY,
            updated_by=updated_by,
            change_note="Persona created",
        )

    def update_persona(
        self,
        template_id: int,
        *,
        body: str | None = None,
        title: str | None = None,
        updated_by: str | None = None,
    ) -> PromptTemplate | None:
        """Edit an existing persona by id; bumps a new version row."""
        if self.engine is None:
            raise RuntimeError("Prompt template DB unavailable")
        row = self.get_by_id(template_id)
        if row is None or row.category != PERSONA_CATEGORY:
            return None
        return self.upsert(
            key=row.key,
            body=body if body is not None else row.body,
            scope_type=row.scope_type,
            scope_id=row.scope_id,
            title=title.strip() if title is not None else row.title,
            category=PERSONA_CATEGORY,
            format=row.format,
            updated_by=updated_by,
            change_note="Persona edited",
        )

    def delete_persona(self, template_id: int) -> bool:
        """Delete a persona by id (its version history is retained)."""
        if self.engine is None:
            raise RuntimeError("Prompt template DB unavailable")
        row = self.get_by_id(template_id)
        if row is None or row.category != PERSONA_CATEGORY:
            return False
        return self.reset(row.key, row.scope_type, row.scope_id)


class PromptResolver:
    """Resolve prompt templates from DB with code-default fallback."""

    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self.store = PromptTemplateStore(self.config)

    @staticmethod
    def definitions() -> dict[str, PromptDefinition]:
        return DEFAULT_PROMPT_DEFINITIONS.copy()

    @staticmethod
    def definition_for(key: str) -> PromptDefinition | None:
        return DEFAULT_PROMPT_DEFINITIONS.get(key)

    def resolve(self, key: str, scope_type: str = "global", scope_id: str | None = None) -> ResolvedPrompt | None:
        normalized_type, normalized_id = _normalize_scope(scope_type, scope_id)

        if self.store.engine is not None:
            row = self.store.get_exact(key, normalized_type, normalized_id)
            if row is None and normalized_type != "global":
                row = self.store.get_exact(key, "global", "*")
            if row is not None:
                definition = self.definition_for(key)
                return ResolvedPrompt(
                    key=key,
                    title=row.title or (definition.title if definition else key),
                    category=row.category or (definition.category if definition else "custom"),
                    format=row.format or (definition.format if definition else "plain_text"),
                    body=row.body,
                    required_vars=_parse_json_list(row.required_vars_json) or list(definition.required_vars if definition else ()),
                    metadata=_parse_json_dict(row.metadata_json) or dict(definition.metadata if definition else {}),
                    scope_type=row.scope_type,
                    scope_id=row.scope_id,
                    source="db_override",
                    updated_at=row.updated_at,
                )

        definition = self.definition_for(key)
        if definition is None:
            return None
        return ResolvedPrompt(
            key=key,
            title=definition.title,
            category=definition.category,
            format=definition.format,
            body=definition.default_body(),
            required_vars=list(definition.required_vars),
            metadata=dict(definition.metadata),
            scope_type="global",
            scope_id="*",
            source="code_default",
            updated_at=None,
        )

    def validate(
        self,
        *,
        key: str,
        body: str,
        required_vars: list[str] | None = None,
    ) -> dict[str, Any]:
        definition = self.definition_for(key)
        normalized_required = [str(item) for item in (required_vars or []) if str(item).strip()]
        if not normalized_required and definition is not None:
            normalized_required = list(definition.required_vars)

        placeholders = extract_placeholders(body)
        missing_required = [name for name in normalized_required if name not in placeholders]
        unknown_placeholders = [name for name in placeholders if normalized_required and name not in normalized_required]

        errors: list[str] = []
        if not (body or "").strip():
            errors.append("Prompt body cannot be empty")
        if missing_required:
            errors.append(f"Missing required placeholders: {', '.join(missing_required)}")
        if unknown_placeholders and definition is not None:
            errors.append(f"Unknown placeholders for '{key}': {', '.join(unknown_placeholders)}")

        return {
            "valid": not errors,
            "placeholders": placeholders,
            "required_vars": normalized_required,
            "missing_required": missing_required,
            "unknown_placeholders": unknown_placeholders,
            "errors": errors,
        }

    def render(
        self,
        *,
        key: str,
        variables: dict[str, Any],
        scope_type: str = "global",
        scope_id: str | None = None,
        body_override: str | None = None,
    ) -> str:
        definition = self.definition_for(key)
        body = body_override
        source = "override"
        if body is None:
            resolved = self.resolve(key, scope_type, scope_id)
            if resolved is None:
                raise KeyError(f"Unknown prompt key '{key}'")
            body = resolved.body
            source = resolved.source

        try:
            return body.format(**variables)
        except KeyError:
            if source != "code_default" and definition is not None and body_override is None:
                logger.warning("Prompt '%s' DB override failed to render; falling back to code default", key)
                return definition.default_body().format(**variables)
            raise


_DEFAULT_PROMPT_RESOLVER: PromptResolver | None = None


def get_prompt_resolver(config: Config | None = None) -> PromptResolver:
    """Return a prompt resolver, caching the configless default path."""
    global _DEFAULT_PROMPT_RESOLVER
    if config is None:
        if _DEFAULT_PROMPT_RESOLVER is None:
            _DEFAULT_PROMPT_RESOLVER = PromptResolver()
        return _DEFAULT_PROMPT_RESOLVER
    return PromptResolver(config)
