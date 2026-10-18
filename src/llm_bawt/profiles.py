"""Unified profile system for llm-bawt.

Profiles store attributes, preferences, and learned facts about entities (users or bots).
This enables:
- Users: Personalized responses based on learned preferences
- Bots: Evolving personality/preferences that develop through conversations

Uses SQLModel ORM with PostgreSQL for storage.
"""

import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import ConfigDict
from sqlalchemy import Column, JSON, String, Text, text
from sqlmodel import Field, Session, SQLModel, select
from .utils.schema import SchemaBootstrapGuard

logger = logging.getLogger(__name__)


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
    
    # Timestamps
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    model_config = ConfigDict(use_enum_values=True)


class EntityProfile(SQLModel, table=True):
    """Core profile record for an entity (user or bot).
    
    Contains basic identity info. Detailed attributes are in ProfileAttribute table.
    """
    
    __tablename__ = "entity_profiles"  # type: ignore[assignment]
    
    id: int | None = Field(default=None, primary_key=True)
    entity_type: EntityType = Field(index=True)
    entity_id: str = Field(max_length=100, index=True)  # user_id or bot_id
    email: str | None = Field(default=None, max_length=255, index=True)  # optional email for user lookup

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
    summary: str | None = Field(default=None)
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
                CREATE INDEX IF NOT EXISTS idx_entity_profiles_email
                ON entity_profiles (email)
                WHERE email IS NOT NULL
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
            profile = session.exec(statement).first()
            
            if profile:
                return profile, False
            
            # Create new profile
            profile = EntityProfile(
                entity_type=entity_type,
                entity_id=entity_id,
            )
            session.add(profile)
            session.commit()
            session.refresh(profile)
            logger.debug(f"Created new {entity_type} profile: {entity_id}")
            
            return profile, True
    
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
            # Check for existing attribute
            statement = select(ProfileAttribute).where(
                ProfileAttribute.entity_type == entity_type,
                ProfileAttribute.entity_id == entity_id,
                ProfileAttribute.category == category,
                ProfileAttribute.key == key,
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
            )
            return list(session.exec(statement).all())
    
    def get_all_attributes(
        self,
        entity_type: EntityType,
        entity_id: str,
    ) -> list[ProfileAttribute]:
        """Get all attributes for an entity."""
        entity_id = entity_id.lower().strip()
        
        with Session(self.engine) as session:
            statement = select(ProfileAttribute).where(
                ProfileAttribute.entity_type == entity_type,
                ProfileAttribute.entity_id == entity_id,
            ).order_by(ProfileAttribute.category, ProfileAttribute.key)
            return list(session.exec(statement).all())
    
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
            # Get all attributes for this entity
            statement = select(ProfileAttribute).where(
                ProfileAttribute.entity_type == entity_type,
                ProfileAttribute.entity_id == entity_id,
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
                # Invalidate cached summary so profile regenerates on next request
                profile_stmt = select(EntityProfile).where(
                    EntityProfile.entity_type == entity_type,
                    EntityProfile.entity_id == entity_id
                )
                profile = session.exec(profile_stmt).first()
                if profile and profile.summary:
                    profile.summary = None
                    session.add(profile)
                    logger.debug(f"Invalidated profile summary for {entity_type}/{entity_id}")
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
            )
            attr = session.exec(statement).first()
            
            if attr:
                session.delete(attr)
                # Invalidate cached summary
                profile_stmt = select(EntityProfile).where(
                    EntityProfile.entity_type == entity_type,
                    EntityProfile.entity_id == entity_id
                )
                profile = session.exec(profile_stmt).first()
                if profile and profile.summary:
                    profile.summary = None
                    session.add(profile)
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
            
            # If profile has a pre-computed summary, use it (from maintenance job)
            if profile and profile.summary:
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
    
    def get_user_profile_summary(self, user_id: str) -> str:
        """Get profile summary for a user."""
        return self.get_profile_summary(EntityType.USER, user_id)
    
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
