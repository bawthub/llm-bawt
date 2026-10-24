"""Bot list and bot-profile request/response schemas.

Split out of ``service/schemas.py`` (TASK-1329, monolith gate). ``schemas.py``
re-imports every name here so ``from ..schemas import X`` is unchanged.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

BotKind = Literal["chat", "agent"]

_TAGS_DOC = ("Assigned bot tags (TASK-1329), sorted. null means the tags could not be "
             "read; it never means 'no tags'.")


class BotInfo(BaseModel):
    """Bot information for /v1/bots endpoint."""
    slug: str
    name: str
    description: str | None = None
    system_prompt: str = ""
    prompt_override_id: int | None = None
    requires_memory: bool = True
    voice_optimized: bool = False
    tts_mode: bool = False
    include_summaries: bool = True
    include_in_global_search: bool = True
    tags: list[str] | None = Field(default=None, description=_TAGS_DOC)
    default_voice: str | None = None
    uses_tools: bool = False
    uses_search: bool = False
    uses_home_assistant: bool = False
    default_model: str | None = None
    color: str | None = None
    avatar: str | None = None
    avatar_render: str | None = None
    bot_type: BotKind = "chat"
    harness: str | None = None
    endpoint_id: int | None = None
    agent_backend: str | None = None
    agent_backend_config: dict[str, Any] = Field(default_factory=dict)
    settings: dict[str, Any] = Field(default_factory=dict)


class BotsResponse(BaseModel):
    """Response for /v1/bots endpoint."""
    object: str = "list"
    data: list[BotInfo]


class BotProfileResponse(BaseModel):
    """Response payload for a bot profile."""

    slug: str
    name: str
    description: str
    system_prompt: str
    # Active persona override (prompt_templates.id). None => use system_prompt.
    prompt_override_id: int | None = None
    requires_memory: bool = True
    voice_optimized: bool = False
    tts_mode: bool = False
    include_summaries: bool = True
    include_in_global_search: bool = True
    tags: list[str] | None = Field(default=None, description=_TAGS_DOC)
    uses_tools: bool = False
    uses_search: bool = False
    uses_home_assistant: bool = False
    default_model: str | None = None
    color: str | None = None
    avatar: str | None = None
    # Server-derived, self-hosted render of ``avatar`` as a data: URL
    # (Twemoji SVG for emoji, small WebP for images). Read-only for clients.
    avatar_render: str | None = None
    default_voice: str | None = None
    nextcloud_config: dict[str, Any] | None = None
    bot_type: BotKind = "chat"
    harness: str | None = None
    endpoint_id: int | None = None
    agent_backend: str | None = None
    agent_backend_config: dict[str, Any] | None = None
    settings: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None


class BotProfileUpsertRequest(BaseModel):
    """Request payload for upserting a bot profile."""

    name: str
    description: str = ""
    system_prompt: str
    requires_memory: bool = True
    voice_optimized: bool = False
    tts_mode: bool = False
    include_summaries: bool = True
    include_in_global_search: bool = True
    uses_tools: bool = False
    uses_search: bool = False
    uses_home_assistant: bool = False
    default_model: str | None = None
    color: str | None = None
    avatar: str | None = None
    default_voice: str | None = None
    nextcloud_config: dict[str, Any] | None = None
    bot_type: BotKind | None = None
    harness: str | None = None
    endpoint_id: int | None = None
    agent_backend: str | None = None
    agent_backend_config: dict[str, Any] | None = None


class BotProfilePatchRequest(BaseModel):
    """Partial update — only provided fields are changed."""

    name: str | None = None
    description: str | None = None
    system_prompt: str | None = None
    prompt_override_id: int | None = None
    requires_memory: bool | None = None
    voice_optimized: bool | None = None
    tts_mode: bool | None = None
    include_summaries: bool | None = None
    include_in_global_search: bool | None = None
    uses_tools: bool | None = None
    uses_search: bool | None = None
    uses_home_assistant: bool | None = None
    default_model: str | None = None
    color: str | None = None
    avatar: str | None = None
    default_voice: str | None = None
    nextcloud_config: dict[str, Any] | None = None
    bot_type: BotKind | None = None
    harness: str | None = None
    endpoint_id: int | None = None
    agent_backend: str | None = None
    agent_backend_config: dict[str, Any] | None = None
    tags: list[str] | None = Field(
        default=None,
        description="Full-set replace of the bot's tags, validated against the catalog "
                    "(TASK-1329). [] clears every tag; null is refused.")
    updated_by: str | None = Field(
        default=None, min_length=1, max_length=128,
        description="Who made this change; recorded as assigned_by on new tag assignments.")
    expected_tags: list[str] | None = Field(
        default=None,
        description="Optional compare-and-set for tags (TASK-1398): the set the caller "
                    "read. If the bot's tags differ, the PATCH returns 409 "
                    "{code: 'tags_changed', current_tags} and writes nothing. Absent "
                    "keeps the unconditional replace.")

    @field_validator("expected_tags")
    @classmethod
    def _expected_not_null(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            raise ValueError("expected_tags must be a list; omit it for an unconditional replace")
        return value

    @field_validator("tags")
    @classmethod
    def _tags_not_null(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            raise ValueError("tags must be a list; send [] to clear every tag")
        return value


class BotCreateRequest(BotProfileUpsertRequest):
    """Request payload for creating a new bot profile."""

    slug: str


class BotProfileListResponse(BaseModel):
    """Response for listing bot profiles."""
    profiles: list[BotProfileResponse]
    total_count: int
    filters: dict[str, Any] = Field(default_factory=dict)


class BotTagUse(BaseModel):
    """One assignment of a catalog tag; ``has_profile`` is false for a deleted bot."""
    bot: str
    has_profile: bool


class BotTagResponse(BaseModel):
    """One bot tag catalog row (TASK-1329)."""
    slug: str
    name: str
    description: str
    kind: Literal["behavior", "role"]
    prompt_key: str | None = None
    capabilities: list[str] = Field(default_factory=list)
    system: bool
    used_by: list[BotTagUse] = Field(default_factory=list)
    created_at: datetime | None = None
    updated_at: datetime | None = None


class BotTagCatalogResponse(BaseModel):
    object: str = "list"
    data: list[BotTagResponse]


class BotTagCreateRequest(BaseModel):
    """Create a role tag; behaviour tags come from code only."""
    model_config = ConfigDict(extra="forbid")  # kind/system are code-owned, never silently dropped

    slug: str
    name: str
    description: str = ""
    prompt_key: str | None = None
    capabilities: list[str] = Field(default_factory=list)


class BotTagPatchRequest(BaseModel):
    """Edit a role tag; only provided fields change. ``prompt_key: null`` clears it."""
    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    description: str | None = None
    prompt_key: str | None = None
    capabilities: list[str] | None = None


class BotTagAssignmentRemovedResponse(BaseModel):
    tag: str
    bot: str
    has_profile: bool
