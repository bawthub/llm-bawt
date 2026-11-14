"""Unified profile system for llm-bawt.

Profiles store attributes, preferences, and learned facts about entities (users or bots).
This enables:
- Users: Personalized responses based on learned preferences
- Bots: Evolving personality/preferences that develop through conversations

Uses SQLModel ORM with PostgreSQL for storage.
"""

import json
import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import ConfigDict
from sqlalchemy import Column, Index, JSON, String, Text, text
from sqlmodel import Field, Session, SQLModel, select
from .utils.schema import SchemaBootstrapGuard

logger = logging.getLogger(__name__)


STATUS_LIVE = "live"
STATUS_CANDIDATE = "candidate"
STATUS_RETIRED = "retired"


class CurationConflict(RuntimeError):
    """A row changed between the curation model's snapshot and apply."""


class EntityType(str, Enum):
    """Type of entity a profile belongs to."""
    USER = "user"
    BOT = "bot"


class ProfileAttribute(SQLModel, table=True):
    """Individual attribute/preference for an entity.
    
    Allows vertical growth - each learned fact is a separate row.
    This enables:
    - Tracking when each attribute was learned
    - Confidence scores for learned attributes
    - Easy querying by category
    - History of attribute changes
    """
    
    __tablename__ = "entity_profile_attributes"  # type: ignore[assignment]
    
    id: int | None = Field(default=None, primary_key=True)
    entity_type: EntityType = Field(index=True)
    entity_id: str = Field(max_length=100, index=True)  # user_id or bot_id
    
    # Attribute categorization
    category: str = Field(max_length=50, index=True)  # e.g., "preference", "fact", "personality"
    key: str = Field(max_length=100, index=True)  # e.g., "favorite_color", "occupation"
    value: Any = Field(sa_column=Column(JSON))  # The actual value (can be any JSON type)
    
    # Metadata
    confidence: float = Field(default=1.0)  # 0.0-1.0, how confident we are in this attribute
    source: str | None = Field(default=None, max_length=50)  # "explicit", "inferred", "system"
    source_message_id: str | None = Field(default=None, max_length=100)  # Message that led to this

    # TASK-1720: extracted facts land as ``candidate`` and are not rendered
    # until curation promotes them to ``live``; curation retires the rest.
    status: str = Field(default=STATUS_LIVE, max_length=16,
                        sa_column_kwargs={"server_default": STATUS_LIVE})
    retire_reason: str | None = Field(default=None, sa_column=Column(Text))

    # Timestamps
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    model_config = ConfigDict(use_enum_values=True)


class EntityProfile(SQLModel, table=True):
    """Core profile record for an entity (user or bot).
    
    Contains basic identity info. Detailed attributes are in ProfileAttribute table.
    """
    
    __tablename__ = "entity_profiles"  # type: ignore[assignment]
    __table_args__ = (Index("idx_entity_profile_unique", "entity_type", "entity_id", unique=True),)
    
    id: int | None = Field(default=None, primary_key=True)
    entity_type: EntityType = Field(index=True)
    entity_id: str = Field(max_length=100, index=True)  # user_id or bot_id
    email: str | None = Field(default=None, max_length=255, index=True)  # optional email for user lookup
    # Service authors retain their user identity/history, but are not people.
    is_service: bool = Field(default=False, nullable=False, sa_column_kwargs={"server_default": text("false")})

    # Basic identity
    display_name: str | None = Field(default=None, max_length=100)
    description: str | None = Field(default=None, max_length=500)
    color: str | None = Field(
        default=None, sa_column=Column(String(64), nullable=True)
    )
    avatar: str | None = Field(
        default=None, sa_column=Column(String(512), nullable=True)
    )
    avatar_render: str | None = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )
    
    # Pre-computed summary for system prompt injection (set by maintenance job)
    summary: str | None = Field(default=None, sa_column=Column(Text, nullable=True))
    summary_updated_at: datetime | None = Field(default=None)  # stamped each time summary is regenerated
    
    # Timestamps
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    model_config = ConfigDict(use_enum_values=True)


# Predefined attribute categories
class AttributeCategory:
    """Standard categories for profile attributes."""
    PREFERENCE = "preference"      # Likes/dislikes, style preferences
    FACT = "fact"                  # Factual info (occupation, location, etc.)
    PERSONALITY = "personality"    # For bots: personality traits that develop
    INTEREST = "interest"          # Topics they're interested in
    CONTEXT = "context"            # Contextual info (current project, etc.)
    COMMUNICATION = "communication"  # Communication style preferences


class ProfileManager:
    """Manages entity profiles (users and bots) in PostgreSQL.

    NOTE on connection pooling: callers should treat this object as a
    long-lived singleton (one per process). It owns a SQLAlchemy engine
    with its own connection pool — instantiating it per HTTP request
    leaks connections until GC collects the engine. Routes that need a
    ProfileManager should resolve a shared instance (see
    ``llm_bawt.service.dependencies.get_profile_manager``).
    """

    _schema_guard = SchemaBootstrapGuard()

    def __init__(self, config: Any):
        self.config = config
        from .utils.db import get_shared_engine

        host = getattr(config, 'POSTGRES_HOST', 'localhost')
        port = int(getattr(config, 'POSTGRES_PORT', 5432))
        database = getattr(config, 'POSTGRES_DATABASE', 'llm_bawt')

        # Process-wide shared engine (TASK-202): every Store talks to the
        # same DB, so they all share one connection pool.
        engine = get_shared_engine(config)
        if engine is None:
            raise RuntimeError("ProfileManager requires Postgres credentials")
        self.engine = engine
        self._ensure_tables_exist()
        logger.debug(f"ProfileManager connected to {host}:{port}/{database}")
    
    def _ensure_tables_exist(self) -> None:
        """Create tables if they don't exist, and run incremental column migrations."""
        def bootstrap(conn) -> None:
            SQLModel.metadata.create_all(
                bind=conn,
                tables=[EntityProfile.__table__, ProfileAttribute.__table__],
            )
            # ---- column migrations (idempotent) ----
            conn.execute(text("""
                ALTER TABLE entity_profiles
                ADD COLUMN IF NOT EXISTS email VARCHAR(255)
            """))
            conn.execute(text("""
                ALTER TABLE entity_profiles
                ADD COLUMN IF NOT EXISTS summary_updated_at TIMESTAMP
            """))
            conn.execute(text("""
                ALTER TABLE entity_profiles
                ADD COLUMN IF NOT EXISTS color VARCHAR(64)
            """))
            conn.execute(text("""
                ALTER TABLE entity_profiles
                ADD COLUMN IF NOT EXISTS avatar VARCHAR(512)
            """))
            conn.execute(text("""
                ALTER TABLE entity_profiles
                ADD COLUMN IF NOT EXISTS avatar_render TEXT
            """))
            conn.execute(text("""
                ALTER TABLE entity_profiles
                ADD COLUMN IF NOT EXISTS is_service BOOLEAN NOT NULL DEFAULT false
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_entity_profiles_email
                ON entity_profiles (email)
                WHERE email IS NOT NULL
            """))

            conn.execute(text("""
                ALTER TABLE entity_profile_attributes
                ADD COLUMN IF NOT EXISTS status VARCHAR(16) NOT NULL DEFAULT 'live'
            """))
            conn.execute(text("""
                ALTER TABLE entity_profile_attributes
                ADD COLUMN IF NOT EXISTS retire_reason TEXT
            """))
            # ---- structural indexes ----
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_entity_profile_attr_entity 
                ON entity_profile_attributes (entity_type, entity_id, category)
            """))
            conn.execute(text("""
                CREATE UNIQUE INDEX IF NOT EXISTS idx_entity_profile_unique 
                ON entity_profiles (entity_type, entity_id)
            """))

            from .bot_identities import seed_system_author
            seed_system_author(conn)

        self._schema_guard.run(self.engine, "profile-manager", bootstrap)
    
    # =========================================================================
    # Profile CRUD
    # =========================================================================
    
    def list_profiles(self, entity_type: EntityType) -> list[EntityProfile]:
        """List all profiles of a given type.
        
        Args:
            entity_type: USER or BOT
            
        Returns:
            List of profiles
        """
        with Session(self.engine) as session:
            statement = select(EntityProfile).where(
                EntityProfile.entity_type == entity_type
            ).order_by(EntityProfile.entity_id)
            return list(session.exec(statement).all())
    
    def get_profile(
        self,
        entity_type: EntityType,
        entity_id: str
    ) -> EntityProfile | None:
        """Get a profile by type and ID without creating it.

        Returns:
            Profile if found, None otherwise
        """
        entity_id = entity_id.lower().strip()

        with Session(self.engine) as session:
            statement = select(EntityProfile).where(
                EntityProfile.entity_type == entity_type,
                EntityProfile.entity_id == entity_id
            )
            return session.exec(statement).first()

    def get_profile_by_email(self, email: str) -> EntityProfile | None:
        """Look up a user profile by email address.

        Returns:
            Profile if found, None otherwise
        """
        email = email.lower().strip()

        with Session(self.engine) as session:
            statement = select(EntityProfile).where(
                EntityProfile.entity_type == EntityType.USER,
                EntityProfile.email == email
            )
            return session.exec(statement).first()
    
    def get_or_create_profile(
        self, 
        entity_type: EntityType, 
        entity_id: str
    ) -> tuple[EntityProfile, bool]:
        """Get existing profile or create a new one.
        
        Returns:
            Tuple of (profile, is_new)
        """
        entity_id = entity_id.lower().strip()
        
        with Session(self.engine) as session:
            statement = select(EntityProfile).where(
                EntityProfile.entity_type == entity_type,
                EntityProfile.entity_id == entity_id
            )
            if self.engine.dialect.name == "sqlite":
                from sqlalchemy.dialects.sqlite import insert
            else:
                from sqlalchemy.dialects.postgresql import insert
            inserted_id = session.execute(
                insert(EntityProfile).values(entity_type=entity_type, entity_id=entity_id)
                .on_conflict_do_nothing(index_elements=["entity_type", "entity_id"])
                .returning(EntityProfile.id)
            ).scalar_one_or_none()
            profile = session.exec(statement).one()
            session.commit()
            session.refresh(profile)
            return profile, inserted_id is not None
    
    def update_profile(
        self,
        entity_type: EntityType,
        entity_id: str,
        display_name: str | None = None,
        description: str | None = None,
        summary: str | None = None,
        email: str | None = None,
        color: str | None = None,
        avatar: str | None = None,
        avatar_render: str | None = None,
        is_service: bool | None = None,
    ) -> EntityProfile | None:
        """Update basic profile info."""
        entity_id = entity_id.lower().strip()

        with Session(self.engine) as session:
            statement = select(EntityProfile).where(
                EntityProfile.entity_type == entity_type,
                EntityProfile.entity_id == entity_id
            )
            profile = session.exec(statement).first()

            if not profile:
                return None

            if display_name is not None:
                profile.display_name = display_name
            if description is not None:
                profile.description = description
            if summary is not None:
                profile.summary = summary
                profile.summary_updated_at = datetime.now(timezone.utc)
            if email is not None:
                profile.email = email.lower().strip() if email else None
            if color is not None:
                profile.color = color.strip() or None
            if avatar is not None:
                profile.avatar = avatar.strip() or None
            if avatar_render is not None:
                profile.avatar_render = avatar_render or None
            if is_service is not None:
                profile.is_service = is_service
            profile.updated_at = datetime.now(timezone.utc)

            session.add(profile)
            session.commit()
            session.refresh(profile)

            return profile
    
    def set_display_name(
        self,
        entity_type: EntityType,
        entity_id: str,
        display_name: str,
    ) -> EntityProfile | None:
        """Set the display name for an entity's profile.
        
        Convenience method that ensures profile exists before updating.
        """
        # Ensure profile exists
        self.get_or_create_profile(entity_type, entity_id)
        return self.update_profile(entity_type, entity_id, display_name=display_name)
    
    def set_profile_summary(
        self,
        entity_type: EntityType,
        entity_id: str,
        summary: str,
    ) -> EntityProfile | None:
        """Set the pre-computed summary for an entity's profile.
        
        This summary is used directly in system prompt injection.
        """
        # Ensure profile exists
        self.get_or_create_profile(entity_type, entity_id)
        return self.update_profile(entity_type, entity_id, summary=summary)
    
    # =========================================================================
    # Attribute CRUD
    # =========================================================================
    
    def set_attribute(
        self,
        entity_type: EntityType,
        entity_id: str,
        category: str,
        key: str,
        value: Any,
        confidence: float = 1.0,
        source: str = "explicit",
        source_message_id: str | None = None,
    ) -> ProfileAttribute:
        """Set or update an attribute for an entity.
        
        If the attribute already exists, it will be updated.
        """
        entity_id = entity_id.lower().strip()
        key = key.lower().strip()
        category = category.lower().strip()
        
        # Ensure profile exists
        self.get_or_create_profile(entity_type, entity_id)
        
        with Session(self.engine) as session:
            # Check for existing live attribute (candidates/retired rows are separate)
            statement = select(ProfileAttribute).where(
                ProfileAttribute.entity_type == entity_type,
                ProfileAttribute.entity_id == entity_id,
                ProfileAttribute.category == category,
                ProfileAttribute.key == key,
                ProfileAttribute.status == STATUS_LIVE,
            )
            attr = session.exec(statement).first()
            
            if attr:
                # Update existing
                attr.value = value
                attr.confidence = confidence
                attr.source = source
                attr.source_message_id = source_message_id
                attr.updated_at = datetime.now(timezone.utc)
            else:
                # Create new
                attr = ProfileAttribute(
                    entity_type=entity_type,
                    entity_id=entity_id,
                    category=category,
                    key=key,
                    value=value,
                    confidence=confidence,
                    source=source,
                    source_message_id=source_message_id,
                )
            
            session.add(attr)
            session.commit()
            session.refresh(attr)
            
            logger.info(f"Set attribute {entity_type}/{entity_id}: {category}.{key} = {value} (id={attr.id})")
            return attr

    def add_candidate(
        self,
        entity_type: EntityType,
        entity_id: str,
        category: str,
        key: str,
        value: Any,
        confidence: float,
        source: str,
        source_message_id: str | None = None,
        *,
        idempotent: bool = False,
    ) -> ProfileAttribute:
        """Record an unreviewed fact; it isn't rendered until curation promotes it.

        ``idempotent`` returns an existing row with the same key, source
        message and value instead of adding another, so a retried extraction
        batch doesn't duplicate a fact; distinct facts from one message stay
        separate.
        """
        entity_id = entity_id.lower().strip()
        category = category.lower().strip()
        key = key.lower().strip()
        self.get_or_create_profile(entity_type, entity_id)
        with Session(self.engine) as session:
            if idempotent:
                existing = session.exec(select(ProfileAttribute).where(
                    ProfileAttribute.entity_type == entity_type,
                    ProfileAttribute.entity_id == entity_id,
                    ProfileAttribute.category == category,
                    ProfileAttribute.key == key,
                    ProfileAttribute.source == source,
                    ProfileAttribute.source_message_id == source_message_id,
                )).all()
                for row in existing:
                    if row.value == value:
                        return row
            attr = ProfileAttribute(
                entity_type=entity_type, entity_id=entity_id,
                category=category, key=key, value=value,
                confidence=confidence, source=source, source_message_id=source_message_id,
                status=STATUS_CANDIDATE,
            )
            session.add(attr)
            session.commit()
            session.refresh(attr)
            return attr

    def attributes_with_status(
        self, entity_type: EntityType, entity_id: str, statuses: tuple[str, ...],
    ) -> list[ProfileAttribute]:
        entity_id = entity_id.lower().strip()
        with Session(self.engine) as session:
            statement = select(ProfileAttribute).where(
                ProfileAttribute.entity_type == entity_type,
                ProfileAttribute.entity_id == entity_id,
                ProfileAttribute.status.in_(statuses),  # type: ignore[attr-defined]
            ).order_by(ProfileAttribute.id)
            return list(session.exec(statement).all())

    def entities_with_candidates(self, entity_type: EntityType) -> list[str]:
        with Session(self.engine) as session:
            statement = select(ProfileAttribute.entity_id).where(
                ProfileAttribute.entity_type == entity_type,
                ProfileAttribute.status == STATUS_CANDIDATE,
            ).distinct()
            return sorted(session.exec(statement).all())

    @staticmethod
    def curation_snapshot(attr: ProfileAttribute) -> tuple[Any, ...]:
        """Row state a curation plan was based on; apply refuses if it changed."""
        return (attr.status, attr.category, attr.key, json.dumps(attr.value, sort_keys=True,
                default=str), attr.updated_at)

    def apply_curation(self, entity_type: EntityType, entity_id: str,
                       changes: list[dict[str, Any]],
                       expected: dict[int, tuple[Any, ...]]) -> None:
        """Apply validated curation changes in one transaction.

        Each change: ``{"id", "status", "value"?, "category"?, "key"?,
        "source_message_id"?, "retire_reason"?}`` for an existing row of this entity.
        ``expected`` maps every changed id to its ``curation_snapshot`` when the
        model saw it. If any row changed since (a manual edit or another run),
        nothing is applied and ``CurationConflict`` is raised.
        """
        entity_id = entity_id.lower().strip()
        now = datetime.now(timezone.utc)
        ids = sorted(change["id"] for change in changes)
        with Session(self.engine) as session:
            rows = {a.id: a for a in session.exec(
                select(ProfileAttribute).where(ProfileAttribute.id.in_(ids))  # type: ignore[union-attr]
                .order_by(ProfileAttribute.id).with_for_update()).all()}
            for change in changes:
                attr = rows.get(change["id"])
                if attr is None or attr.entity_type != entity_type or attr.entity_id != entity_id:
                    raise ValueError(f"attribute {change['id']} is not part of {entity_id}")
                if change["id"] not in expected or self.curation_snapshot(attr) != expected[change["id"]]:
                    raise CurationConflict(f"attribute {change['id']} changed during curation")
            for change in changes:
                attr = rows[change["id"]]
                for field in ("status", "value", "category", "key", "source_message_id",
                              "retire_reason"):
                    if field in change:
                        setattr(attr, field, change[field])
                attr.updated_at = now
                session.add(attr)
            session.commit()

    def get_attribute(
        self,
        entity_type: EntityType,
        entity_id: str,
        category: str,
        key: str,
    ) -> ProfileAttribute | None:
        """Get a specific attribute."""
        entity_id = entity_id.lower().strip()
        key = key.lower().strip()
        category = category.lower().strip()
        
        with Session(self.engine) as session:
            statement = select(ProfileAttribute).where(
                ProfileAttribute.entity_type == entity_type,
                ProfileAttribute.entity_id == entity_id,
                ProfileAttribute.category == category,
                ProfileAttribute.key == key,
                ProfileAttribute.status == STATUS_LIVE,
            )
            return session.exec(statement).first()
    
    def get_attributes_by_category(
        self,
        entity_type: EntityType,
        entity_id: str,
        category: str,
    ) -> list[ProfileAttribute]:
        """Get all attributes in a category for an entity."""
        entity_id = entity_id.lower().strip()
        category = category.lower().strip()
        
        with Session(self.engine) as session:
            statement = select(ProfileAttribute).where(
                ProfileAttribute.entity_type == entity_type,
                ProfileAttribute.entity_id == entity_id,
                ProfileAttribute.category == category,
                ProfileAttribute.status == STATUS_LIVE,
            )
            return list(session.exec(statement).all())

    def get_all_attributes(
        self,
        entity_type: EntityType,
        entity_id: str,
    ) -> list[ProfileAttribute]:
        """Get all live attributes for an entity."""
        entity_id = entity_id.lower().strip()

        with Session(self.engine) as session:
            statement = select(ProfileAttribute).where(
                ProfileAttribute.entity_type == entity_type,
                ProfileAttribute.entity_id == entity_id,
                ProfileAttribute.status == STATUS_LIVE,
            ).order_by(ProfileAttribute.category, ProfileAttribute.key)
            return list(session.exec(statement).all())
    
    @staticmethod
    def _invalidate_summary(session, entity_type, entity_id):
        profile = session.exec(select(EntityProfile).where(
            EntityProfile.entity_type == entity_type,
            EntityProfile.entity_id == entity_id,
        )).first()
        if profile and profile.summary:
            profile.summary = None
            session.add(profile)

    def search_and_delete_attributes(
        self,
        entity_type: EntityType,
        entity_id: str,
        query: str,
        category: str | None = None,
    ) -> tuple[int, list[str]]:
        """Search for attributes matching query and delete them.
        
        Args:
            entity_type: USER or BOT
            entity_id: The entity ID
            query: Search term to match against key or value
            category: Optional category to filter by
            
        Returns:
            Tuple of (count_deleted, list of deleted descriptions)
        """
        entity_id = entity_id.lower().strip()
        query_lower = query.lower().strip()
        
        with Session(self.engine) as session:
            # Only facts visible to callers count as deletable matches.
            statement = select(ProfileAttribute).where(
                ProfileAttribute.entity_type == entity_type,
                ProfileAttribute.entity_id == entity_id,
                ProfileAttribute.status == STATUS_LIVE,
            )
            if category:
                statement = statement.where(ProfileAttribute.category == category.lower().strip())
            
            attributes = list(session.exec(statement).all())
            
            # Find matches (search in key and value)
            deleted = []
            for attr in attributes:
                key_match = query_lower in attr.key.lower()
                value_match = query_lower in str(attr.value).lower()
                if key_match or value_match:
                    desc = f"{attr.category}.{attr.key}: {str(attr.value)[:50]}"
                    session.delete(attr)
                    deleted.append(desc)
                    logger.debug(f"Deleted attribute {entity_type}/{entity_id}: {desc}")
            
            if deleted:
                self._invalidate_summary(session, entity_type, entity_id)
                session.commit()
            
            return len(deleted), deleted
    
    def delete_attribute(
        self,
        entity_type: EntityType,
        entity_id: str,
        category: str,
        key: str,
    ) -> bool:
        """Delete a specific attribute."""
        entity_id = entity_id.lower().strip()
        key = key.lower().strip()
        category = category.lower().strip()
        
        with Session(self.engine) as session:
            statement = select(ProfileAttribute).where(
                ProfileAttribute.entity_type == entity_type,
                ProfileAttribute.entity_id == entity_id,
                ProfileAttribute.category == category,
                ProfileAttribute.key == key,
                ProfileAttribute.status == STATUS_LIVE,
            )
            attr = session.exec(statement).first()

            if attr:
                session.delete(attr)
                self._invalidate_summary(session, entity_type, entity_id)
                session.commit()
                logger.debug(f"Deleted attribute {entity_type}/{entity_id}: {category}.{key}")
                return True
            return False
    
    def delete_attribute_by_id(self, attribute_id: int) -> bool:
        """Delete a specific attribute by its database ID.
        
        Args:
            attribute_id: The primary key ID of the attribute
            
        Returns:
            True if deleted, False if not found
        """
        with Session(self.engine) as session:
            statement = select(ProfileAttribute).where(
                ProfileAttribute.id == attribute_id,
            )
            attr = session.exec(statement).first()
            
            if attr:
                desc = f"{attr.entity_type}/{attr.entity_id}: {attr.category}.{attr.key}"
                session.delete(attr)
                self._invalidate_summary(session, attr.entity_type, attr.entity_id)
                session.commit()
                logger.debug(f"Deleted attribute by ID {attribute_id}: {desc}")
                return True
            return False

    def update_attribute_by_id(
        self,
        attribute_id: int,
        value: Any | None = None,
        confidence: float | None = None,
        source: str | None = None,
    ) -> ProfileAttribute | None:
        """Update a specific attribute by its database ID.

        Returns updated attribute, or None if not found.
        """
        with Session(self.engine) as session:
            statement = select(ProfileAttribute).where(ProfileAttribute.id == attribute_id)
            attr = session.exec(statement).first()
            if not attr:
                return None

            changed = False
            if value is not None:
                attr.value = value
                changed = True
            if confidence is not None:
                attr.confidence = confidence
                changed = True
            if source is not None:
                attr.source = source
                changed = True

            if not changed:
                return attr

            attr.updated_at = datetime.now(timezone.utc)
            session.add(attr)

            profile_stmt = select(EntityProfile).where(
                EntityProfile.entity_type == attr.entity_type,
                EntityProfile.entity_id == attr.entity_id,
            )
            profile = session.exec(profile_stmt).first()
            if profile and profile.summary:
                profile.summary = None
                session.add(profile)

            session.commit()
            session.refresh(attr)
            return attr
    
    def delete_attributes_by_category(
        self,
        entity_type: EntityType,
        entity_id: str,
        category: str,
    ) -> int:
        """Delete all attributes in a category for an entity. Returns count deleted."""
        entity_id = entity_id.lower().strip()
        category = category.lower().strip()
        
        with Session(self.engine) as session:
            statement = select(ProfileAttribute).where(
                ProfileAttribute.entity_type == entity_type,
                ProfileAttribute.entity_id == entity_id,
                ProfileAttribute.category == category,
            )
            attributes = session.exec(statement).all()
            count = len(attributes)
            
            for attr in attributes:
                session.delete(attr)
            
            session.commit()
            logger.debug(f"Deleted {count} attributes from {entity_type}/{entity_id} in category {category}")
            return count
    
    def get_attribute_by_id(self, attribute_id: int) -> ProfileAttribute | None:
        """Get a specific attribute by its database ID.
        
        Args:
            attribute_id: The primary key ID of the attribute
            
        Returns:
            The attribute if found, None otherwise
        """
        with Session(self.engine) as session:
            statement = select(ProfileAttribute).where(
                ProfileAttribute.id == attribute_id,
            )
            return session.exec(statement).first()
    
    # =========================================================================
    # Profile Summary (for system prompts)
    # =========================================================================
    
    def get_profile_summary(
        self,
        entity_type: EntityType,
        entity_id: str,
        include_categories: list[str] | None = None,
        min_confidence: float = 0.5,
        use_summary: bool = True,
    ) -> str:
        """Generate a summary of an entity's profile for system prompt injection.
        
        Args:
            entity_type: USER or BOT
            entity_id: The entity's ID
            include_categories: Categories to include (None = all)
            min_confidence: Minimum confidence threshold for attributes
            
        Returns:
            Formatted string suitable for system prompt injection.
        """
        entity_id = entity_id.lower().strip()
        
        with Session(self.engine) as session:
            # Get profile
            profile_stmt = select(EntityProfile).where(
                EntityProfile.entity_type == entity_type,
                EntityProfile.entity_id == entity_id
            )
            profile = session.exec(profile_stmt).first()
            
            # Legacy: a pre-computed summary wins (the retired maintenance job wrote it).
            if use_summary and profile and profile.summary:
                lines = []
                # Always start with name
                if profile.display_name:
                    prefix = "User's name" if entity_type == EntityType.USER else "Your name"
                    lines.append(f"{prefix}: {profile.display_name}")
                lines.append(profile.summary)
                return "\n".join(lines)
            
            # Fallback: build from individual attributes
            attr_stmt = select(ProfileAttribute).where(
                ProfileAttribute.entity_type == entity_type,
                ProfileAttribute.entity_id == entity_id,
                ProfileAttribute.confidence >= min_confidence,
                ProfileAttribute.status == STATUS_LIVE,
            )
            if include_categories:
                # Filter by categories in Python since SQLModel typing is tricky
                pass  # Will filter after query
            attr_stmt = attr_stmt.order_by(ProfileAttribute.category, ProfileAttribute.key)
            attributes = list(session.exec(attr_stmt).all())
            
            # Filter by category if specified
            if include_categories:
                lower_cats = [c.lower() for c in include_categories]
                attributes = [a for a in attributes if a.category in lower_cats]
        
        if not profile and not attributes:
            return ""
        
        lines = []
        
        # Add display name if present
        if profile and profile.display_name:
            prefix = "User's name" if entity_type == EntityType.USER else "Your name"
            lines.append(f"{prefix}: {profile.display_name}")
        
        # Group attributes by category
        by_category: dict[str, list[ProfileAttribute]] = {}
        for attr in attributes:
            by_category.setdefault(attr.category, []).append(attr)
        
        # Format each category
        category_labels = {
            AttributeCategory.PREFERENCE: "Preferences",
            AttributeCategory.FACT: "Facts",
            AttributeCategory.PERSONALITY: "Personality traits",
            AttributeCategory.INTEREST: "Interests",
            AttributeCategory.CONTEXT: "Current context",
            AttributeCategory.COMMUNICATION: "Communication style",
        }
        
        for category, attrs in by_category.items():
            label = category_labels.get(category, category.title())
            if entity_type == EntityType.USER:
                label = f"User's {label.lower()}"
            else:
                label = f"Your {label.lower()}"
            
            # Check if this category has a single "summary" entry (from maintenance)
            # If so, output the summary directly without key prefix
            if len(attrs) == 1 and attrs[0].key == "summary":
                summary_value = str(attrs[0].value).strip()
                if summary_value:
                    lines.append(f"{label}: {summary_value}")
                continue
            
            items = []
            for attr in attrs:
                # Skip summary entries when mixed with other attributes
                if attr.key == "summary":
                    continue

                # Do not inject learned tool-list traits for bots; these can become stale
                # and conflict with the authoritative runtime tool definitions.
                if (
                    entity_type == EntityType.BOT
                    and category == AttributeCategory.PERSONALITY
                    and attr.key in {"tools", "tool_list", "available_tools"}
                ):
                    continue
                    
                # Format value
                if isinstance(attr.value, list):
                    val_str = ", ".join(str(v) for v in attr.value)
                elif isinstance(attr.value, bool):
                    val_str = "yes" if attr.value else "no"
                else:
                    val_str = str(attr.value)
                
                # Format key (convert snake_case to readable)
                key_str = attr.key.replace("_", " ")
                items.append(f"{key_str}: {val_str}")
            
            if items:
                lines.append(f"{label}: {'; '.join(items)}")
        
        return "\n".join(lines)
    
    # =========================================================================
    # Convenience methods
    # =========================================================================
    
    def get_user_profile_summary(self, user_id: str, *, curated: bool = False) -> str:
        """Profile text for a user's prompt section.

        ``curated`` renders the live (curated) attributes directly; otherwise the
        legacy stored summary wins when present (TASK-1720 render mode).
        """
        return self.get_profile_summary(EntityType.USER, user_id, use_summary=not curated)
    
    def get_bot_profile_summary(self, bot_id: str) -> str:
        """Get profile summary for a bot (personality traits, preferences)."""
        return self.get_profile_summary(
            EntityType.BOT, 
            bot_id,
            include_categories=[
                AttributeCategory.PERSONALITY,
                AttributeCategory.PREFERENCE,
                AttributeCategory.INTEREST,
            ]
        )
    
    def set_user_preference(
        self, 
        user_id: str, 
        key: str, 
        value: Any,
        confidence: float = 1.0,
        source: str = "explicit",
    ) -> ProfileAttribute:
        """Convenience method to set a user preference."""
        return self.set_attribute(
            EntityType.USER,
            user_id,
            AttributeCategory.PREFERENCE,
            key,
            value,
            confidence=confidence,
            source=source,
        )
    
    def set_user_fact(
        self, 
        user_id: str, 
        key: str, 
        value: Any,
        confidence: float = 1.0,
        source: str = "explicit",
    ) -> ProfileAttribute:
        """Convenience method to set a user fact."""
        return self.set_attribute(
            EntityType.USER,
            user_id,
            AttributeCategory.FACT,
            key,
            value,
            confidence=confidence,
            source=source,
        )
    
    def set_bot_personality(
        self, 
        bot_id: str, 
        key: str, 
        value: Any,
        confidence: float = 1.0,
    ) -> ProfileAttribute:
        """Convenience method to set a bot personality trait."""
        return self.set_attribute(
            EntityType.BOT,
            bot_id,
            AttributeCategory.PERSONALITY,
            key,
            value,
            confidence=confidence,
            source="learned",
        )


# Backwards compatibility
def load_user_profile_summary(config: Any, user_id: str) -> str:
    """Load and format user profile for system prompt injection.
    
    Backwards compatible function that works with both old and new systems.
    
    Args:
        config: Config object.
        user_id: User ID (required).
    """
    if not user_id:
        raise ValueError("user_id is required for load_user_profile_summary")
    try:
        manager = ProfileManager(config)
        return manager.get_user_profile_summary(user_id)
    except Exception as e:
        logger.warning(f"Could not load user profile from DB: {e}")
        return ""
