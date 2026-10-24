"""Bot list and bot-profile request/response schemas.

Split out of ``service/schemas.py`` (TASK-1329, monolith gate). ``schemas.py``
re-imports every name here so ``from ..schemas import X`` is unchanged.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

BotKind = Literal["chat", "agent"]


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


class BotCreateRequest(BotProfileUpsertRequest):
    """Request payload for creating a new bot profile."""

    slug: str


class BotProfileListResponse(BaseModel):
    """Response for listing bot profiles."""
    profiles: list[BotProfileResponse]
    total_count: int
    filters: dict[str, Any] = Field(default_factory=dict)
