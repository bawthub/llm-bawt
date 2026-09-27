"""Pydantic schemas for the llm-bawt service API."""

import time
import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, PrivateAttr

SERVICE_VERSION = "0.1.0"
BotKind = Literal["chat", "agent"]

# =============================================================================
# OpenAI-Compatible Schemas
# =============================================================================

class ChatMessage(BaseModel):
    """OpenAI-compatible chat message.

    TASK-225: extended with ``attachment_ids`` so callers can attach
    images by reference to a ``media_assets`` row instead of inlining
    base64 in the content array. The server resolves each id at the
    LLM-call boundary via :class:`~llm_bawt.media.store.MediaStore` and
    persists a tiny ``{asset_id, kind}`` ref on the chat-history row.

    Backwards compatibility: the legacy multimodal shape
    ``content: [{type:"text"}, {type:"image_url"}]`` keeps working —
    Claude Code and other OpenAI-multimodal clients still send inline
    base64 image_url parts, and the server auto-uploads those to
    MediaStore so they too get a persistent asset_id and show up in
    history just like new-style uploads.
    """
    role: Literal["system", "user", "assistant", "function", "tool"]
    # list[dict] before str so Pydantic v2 left-to-right union matching
    # tries the list variant first for OpenAI content-array payloads.
    content: list[dict] | str | None = None
    name: str | None = None
    # TASK-225: optional list of ``ma_<ulid>`` ids referencing rows in the
    # ``media_assets`` table. Only meaningful on ``role=="user"`` — other
    # roles are ignored. Resolved server-side; clients never send base64.
    attachment_ids: list[str] | None = Field(
        default=None,
        description="Optional media_assets ids to attach (resolved server-side).",
    )


class ChatRequestAnimation(BaseModel):
    """One animation entry passed in by the caller (TASK-214).

    Mirrors the bawthub Prisma `AvatarAnimation` shape but only carries the
    fields llm-bawt needs at request time: name + description. Caller is
    expected to filter to enabled rows before sending.
    """
    name: str
    description: str | None = None


class McpToolResultContinuation(BaseModel):
    """Server-owned actual result of an already-executed approved MCP call."""

    kind: Literal["mcp_tool_result"] = "mcp_tool_result"
    approval_request_id: str
    original_tool_use_id: str | None = None
    tool_name: str
    result: Any
    is_error: bool = False


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""
    model: str | None = None  # Optional, will use service default if not specified
    messages: list[ChatMessage]
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    max_tokens: int | None = None
    stream: bool = False
    stop: str | list[str] | None = None
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    user: str | None = None
    # llm-bawt extensions
    bot_id: str | None = Field(default=None, description="Bot personality to use")
    session_id: str | None = Field(
        default=None,
        description="TASK-251: converse in an explicitly selected thread — the "
        "turn persists to AND assembles context from this session. Absent "
        "(the primary mode): continuous history, byte-identical to before "
        "threads existed. Requested session must belong to (bot, user); "
        "deleted threads answer 410.",
    )
    augment_memory: bool = Field(default=True, description="Whether to augment with memory context")
    extract_memory: bool = Field(default=True, description="Whether to extract memories from response")
    include_summaries: bool = Field(default=True, description="Whether to inject conversation summary records into context")
    tts_mode: bool = Field(default=False, description="Whether to append TTS output formatting instructions to the system prompt")
    inject_user_prefix: bool = Field(default=False, description="For agent backends only: prepend chat.agent_user_prefix body to every user message. Survives agent-SDK session resume (unlike system prompt edits). Stacks independently with the voice-mode prefix.")
    client_system_context: str | None = Field(default=None, description="System context extracted from client messages (set by routes, not by callers)", exclude=True)
    ha_mode: bool = Field(default=False, description="HA-mode: cap history, force tool_choice=required on first call (set by routes)", exclude=True)
    user_message_id: str | None = Field(default=None, description="Frontend-generated UUID for the user message (used as trigger_message_id in turn logs)")
    assistant_message_id: str | None = Field(default=None, description="Frontend-generated UUID for the ASSISTANT reply row. Persisted as the assistant message id so the live streaming bubble and the reloaded history row share one id (single bubble). None → server mints one. Closes the EPIC TASK-217 assistant-identity gap.")
    # Durable inter-bot delivery correlation. These are server/MCP-generated,
    # optional for every ordinary chat caller, and deliberately ride the normal
    # chat path rather than creating a second execution mechanism.
    inter_bot_delivery_id: str | None = Field(default=None, description="Durable inter-bot delivery id (server-generated and claim-validated).")
    _internal_inter_bot_sender_id: str | None = PrivateAttr(default=None)
    # Python-only authority: never accepted from public request JSON.
    _internal_approval_claim: tuple[str, str] | None = PrivateAttr(default=None)
    inter_bot_turn_id: str | None = Field(default=None, description="Deterministic reserved turn id for a durable delivery.")
    inter_bot_bridge_request_id: str | None = Field(default=None, description="Deterministic bridge run id used to deduplicate transport retries.")
    inter_bot_claim_token: str | None = Field(default=None, description="Private dispatcher claim token; rejected unless it matches the durable delivery row.")
    inter_bot_timeout_seconds: float | None = Field(default=None, ge=1.0, le=86400.0, description="Request-local bridge inactivity timeout for a durable delivery.")
    inter_bot_session_policy: Literal["continue", "reset_retain_history", "reset_without_history"] | None = Field(
        default=None,
        description="Validated dispatcher-owned pre-delivery session policy.",
    )
    inter_bot_seed_session_id: str | None = Field(
        default=None,
        description="Dispatcher-resolved archived thread used only for retained-history reset seeding.",
    )
    # TASK-269: continuation-turn linkage.  Set when this turn is answering a
    # prior deferred AskUserQuestion.  parent_turn_id threads the chain (and
    # drives cross-tab resolution via turn_start{parent_turn_id}); when
    # answered_question_id is set, the new turn is recorded as the one that
    # carried the answer back to the agent.
    parent_turn_id: str | None = Field(default=None, description="TASK-269: the awaiting turn this continuation answers")
    answered_question_id: str | None = Field(default=None, description="TASK-269: tool_use_id of the question this continuation answers")
    continuation_payload: McpToolResultContinuation | None = Field(
        default=None,
        exclude=True,
        description="Server-owned persisted MCP result delivered on a continuation turn.",
    )
    # TASK-214: animations + avatar visibility now flow on each request.
    # The avatar catalog is owned by the bawthub frontend; llm-bawt is stateless
    # w.r.t. animations. `avatar_visible` is currently informational (consumed
    # by the embedding classifier added in TASK-215).
    animations: list[ChatRequestAnimation] | None = Field(
        default=None,
        description="Available avatar animations the model can pick from when tts_mode=true. "
                    "When None or empty, no animation tool is injected. "
                    "Owned by bawthub frontend (Prisma).",
    )
    avatar_visible: bool | None = Field(
        default=None,
        description="Whether an avatar is currently being rendered on the client. "
                    "TASK-215 will gate animation work on this so we don't waste "
                    "compute when no one can see the animation.",
    )


class ChatCompletionChoice(BaseModel):
    """OpenAI-compatible chat completion choice."""
    index: int
    message: ChatMessage
    finish_reason: str | None = "stop"


class UsageInfo(BaseModel):
    """Token usage information."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionChoice]
    usage: UsageInfo | None = None


class ChatCompletionChunk(BaseModel):
    """OpenAI-compatible streaming chunk."""
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[dict]














# =============================================================================
# Model Definition CRUD Schemas
# =============================================================================













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


class PersonaResponse(BaseModel):
    """A switchable persona prompt (TASK-477). Global / shareable across bots."""

    id: int
    key: str
    title: str
    body: str
    updated_at: datetime | None = None
    created_at: datetime | None = None


class PersonaCreateRequest(BaseModel):
    """Create a new global persona."""

    title: str
    body: str


class PersonaUpdateRequest(BaseModel):
    """Edit an existing persona (creates a new version)."""

    title: str | None = None
    body: str | None = None




















class TurnLogListItem(BaseModel):
    """Summary row for one persisted turn log."""
    id: str
    created_at: datetime
    request_id: str | None = None
    path: str
    stream: bool = False
    model: str | None = None
    bot_id: str | None = None
    user_id: str | None = None
    status: str
    latency_ms: float | None = None
    user_prompt: str | None = None
    response_preview: str | None = None
    response_tail: str | None = None
    response_chars: int = 0
    response_preview_truncated: bool = False
    tool_call_count: int = 0
    error_text: str | None = None
    animation: str | None = None
    agent_session_key: str | None = None
    agent_request_id: str | None = None
    trigger_message_id: str | None = None
    assistant_message_id: str | None = None
    # Per-turn token accounting from the upstream SDK (claude_code, etc.).
    # Shape: {input_tokens, cache_read_tokens, cache_creation_tokens,
    #         output_tokens, context_window, max_output_tokens, total_cost_usd}.
    # None when the upstream backend doesn't expose usage info.
    token_usage: dict[str, Any] | None = None
    # TASK-269 turn lifecycle / continuation chain.
    end_reason: str | None = None
    question_id: str | None = None
    parent_turn_id: str | None = None
    # Wall-clock completion time. None = turn still in progress (the single
    # path-agnostic in-flight signal; supersedes the status="streaming" proxy).
    ended_at: datetime | None = None
    # Embedded question row (chat_pending_questions.row_to_dict) when this turn
    # ended with end_reason="question" — lets the UI render a QuestionMessage
    # straight from history hydration without a second fetch.
    question: dict[str, Any] | None = None
    # TASK-661: per-turn changed-file summary (same shape the live
    # turn_complete.changed_files event carries). None/omitted when the turn
    # changed no files. Lets the chat UI hydrate file links on refresh via the
    # existing turn-log backfill graft.
    changed_files: dict[str, Any] | None = None


class TurnLogListResponse(BaseModel):
    """Response for listing persisted turn logs."""
    turns: list[TurnLogListItem]
    total_count: int
    filters: dict[str, Any] = Field(default_factory=dict)


class TurnLogDetail(BaseModel):
    """Detailed persisted turn log."""
    id: str
    created_at: datetime
    request_id: str | None = None
    path: str
    stream: bool = False
    model: str | None = None
    bot_id: str | None = None
    user_id: str | None = None
    status: str
    latency_ms: float | None = None
    user_prompt: str | None = None
    request: Any | None = None
    response: str | None = None
    # TASK-360 (P4): partial/persisted model reasoning ("thinking") so a cold
    # reload mid-turn can recover already-produced reasoning.
    reasoning: str | None = None
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    error_text: str | None = None
    animation: str | None = None
    agent_session_key: str | None = None
    agent_request_id: str | None = None
    token_usage: dict[str, Any] | None = None
    # TASK-269 turn lifecycle / continuation chain.
    end_reason: str | None = None
    question_id: str | None = None
    parent_turn_id: str | None = None
    question: dict[str, Any] | None = None


class RecentBotTurn(BaseModel):
    """Compact summary of the most recent turn for one bot.

    Returned by ``GET /v1/turn-logs/recent-by-bot``. Designed for dashboard
    "last activity per bot" views: enough information to render a tile
    (who, when, what was said, how many tools fired, did it succeed) without
    a follow-up fetch.
    """
    bot_id: str
    turn_id: str
    created_at: datetime
    model: str | None = None
    status: str
    latency_ms: float | None = None
    user_prompt_preview: str | None = None
    response_preview: str | None = None
    response_tail: str | None = None
    response_chars: int = 0
    response_preview_truncated: bool = False
    tool_call_count: int = 0
    token_usage: dict[str, Any] | None = None
    trigger_message_id: str | None = None


class RecentByBotsResponse(BaseModel):
    """Single response carrying one ``RecentBotTurn`` per bot.

    Sorted by ``created_at`` descending so the most recently active bot is
    first. Bots in the request's ``bot_ids`` filter that have no turn in the
    window are simply absent from ``turns`` — the caller can detect "no
    activity" by checking which slugs are missing.
    """
    turns: list[RecentBotTurn]


class ToolCallEvent(BaseModel):
    """Tool-call event linked to one trigger history message."""
    turn_id: str
    turn_ended_at: datetime | None = None
    created_at: datetime
    request_id: str | None = None
    model: str | None = None
    bot_id: str | None = None
    user_id: str | None = None
    message_id: str
    message_role: str = "user"
    message_timestamp: float | None = None
    tool_call_count: int = 0
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)


class ToolCallEventsResponse(BaseModel):
    """Tool-call events for history annotation in UI/CLI."""
    events: list[ToolCallEvent]
    total_count: int
    filters: dict[str, Any] = Field(default_factory=dict)


class ServiceStatusResponse(BaseModel):
    """Service health and status."""
    status: str = "ok"
    version: str = SERVICE_VERSION
    healthy: bool = True
    uptime_seconds: float
    tasks_processed: int
    tasks_pending: int
    worker_running: bool = False
    models_loaded: list[str] = []
    current_model: str | None = None
    default_model: str | None = None
    default_bot: str | None = None
    available_models: list[str] = []
    # Health checks
    checks: dict[str, str] = Field(default_factory=dict)
    # Database / memory
    database_connected: bool = False
    database_host: str | None = None
    database_error: str | None = None
    messages_count: int = 0
    memories_count: int = 0
    pgvector_available: bool = False
    embeddings_available: bool = False
    # llm-bawt MCP server
    mcp_mode: str = "embedded"
    mcp_status: str = "up"
    mcp_url: str | None = None
    mcp_http_status: int | None = None


# =============================================================================
# Full System Status (mirrors core.status.SystemStatus)
# =============================================================================































# Memory Management Models




























class RawCompletionRequest(BaseModel):
    """Request for raw LLM completion without bot/memory overhead.

    Use this for utility tasks like memory consolidation, summarization, etc.
    """
    prompt: str
    system: str | None = None
    model: str | None = None  # Inherits global maintenance_model when omitted
    max_tokens: int = 500
    temperature: float = 0.7


class RawCompletionResponse(BaseModel):
    """Response from raw LLM completion."""
    content: str
    model: str
    tokens: int | None = None
    elapsed_ms: float


# History Summarization Models












# Profile Attribute Models
class UserProfileAttribute(BaseModel):
    """A single profile attribute."""
    id: int | None = None
    category: str
    key: str
    value: Any
    confidence: float = 1.0
    source: str | None = None
    created_at: str | None = None
    updated_at: str | None = None


class ProfileAttributeUpdateRequest(BaseModel):
    """Request payload for updating an existing profile attribute."""
    value: Any | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    source: str | None = None


class ProfileAttributeUpsertRequest(BaseModel):
    """Request payload for creating/updating a profile attribute by identity."""

    entity_type: Literal["user", "bot"]
    entity_id: str
    category: str
    key: str
    value: Any
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source: str = "explicit"


# Unified Profile Models
class ProfileDetail(BaseModel):
    """Generic profile detail with attributes for any entity type (user or bot)."""
    entity_type: str  # "user" or "bot"
    entity_id: str
    email: str | None = None
    display_name: str | None = None
    description: str | None = None
    color: str | None = None
    avatar: str | None = None
    avatar_render: str | None = None
    summary: str | None = None
    summary_updated_at: str | None = None
    attributes: list[UserProfileAttribute] = []
    created_at: str | None = None


class ProfileUpdateRequest(BaseModel):
    """Request payload for updating core profile fields."""
    display_name: str | None = None
    description: str | None = None
    summary: str | None = None
    email: str | None = None
    color: str | None = None
    avatar: str | None = None


class ProfileListResponse(BaseModel):
    """Response for listing profiles of a given type."""
    profiles: list[ProfileDetail]
    total_count: int


class ProfileAttributeListResponse(BaseModel):
    """Response for listing profile attributes."""
    attributes: list[UserProfileAttribute]
    total_count: int
    filters: dict[str, Any] = Field(default_factory=dict)

# =============================================================================
# Nextcloud Admin Schemas
# =============================================================================

class NextcloudProvisionRequest(BaseModel):
    """Request to provision a Nextcloud Talk room and bot."""
    bot_id: str = Field(description="llm-bawt bot ID (e.g., nova, monika)")
    room_name: str | None = Field(default=None, description="Room name (default: bot name)")
    bot_name: str | None = Field(default=None, description="Bot display name (default: bot ID)")
    owner_user_id: str = Field(default="user", description="Room owner")

class NextcloudProvisionResponse(BaseModel):
    """Response from provisioning."""
    bot_id: str
    room_token: str
    room_url: str
    nextcloud_bot_id: int
    nextcloud_bot_name: str


# =============================================================================
# Runtime Settings Schemas
# =============================================================================

class RuntimeSettingItem(BaseModel):
    """Runtime setting key/value item."""
    key: str
    value: Any


class RuntimeSettingsResponse(BaseModel):
    """List runtime settings for a scope."""
    scope_type: str
    scope_id: str
    settings: list[RuntimeSettingItem]


class RuntimeSettingRecord(BaseModel):
    """One runtime setting row with scope metadata."""
    scope_type: str
    scope_id: str
    key: str
    value: Any
    updated_at: datetime | None = None


class RuntimeSettingsListResponse(BaseModel):
    """List runtime settings across scopes."""
    settings: list[RuntimeSettingRecord]
    total_count: int
    filters: dict[str, Any] = Field(default_factory=dict)


class BotProfileListResponse(BaseModel):
    """Response for listing bot profiles."""
    profiles: list[BotProfileResponse]
    total_count: int
    filters: dict[str, Any] = Field(default_factory=dict)


class RuntimeSettingUpsertRequest(BaseModel):
    """Upsert one runtime setting."""
    scope_type: Literal["global", "bot"]
    scope_id: str | None = None
    key: str
    value: Any


class RuntimeSettingBatchItem(BaseModel):
    """One runtime setting upsert in a batch request."""
    scope_type: Literal["global", "bot"]
    scope_id: str | None = None
    key: str
    value: Any


class RuntimeSettingBatchUpsertRequest(BaseModel):
    """Batch upsert runtime settings."""
    items: list[RuntimeSettingBatchItem]


class PromptTemplateResponse(BaseModel):
    """Resolved or exact prompt template payload."""

    key: str
    title: str
    category: str
    format: str = "plain_text"
    body: str
    scope_type: str
    scope_id: str
    source: str
    required_vars: list[str] = Field(default_factory=list)
    placeholders: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    updated_at: datetime | None = None


class PromptTemplateListResponse(BaseModel):
    """Response for listing prompt templates.

    Returns the FULL body for every prompt. For agent-context-safe lookups
    (just names, scope, source) use PromptTemplateIndexResponse via
    `GET /v1/prompts/index` — that route omits body/placeholders/metadata.
    """

    prompts: list[PromptTemplateResponse]
    total_count: int
    filters: dict[str, Any] = Field(default_factory=dict)


class PromptTemplateSummary(BaseModel):
    """Lightweight prompt-template entry — names and locators, no body.

    Use this when an agent (or a UI) needs to enumerate available prompts to
    decide which one to fetch in full. `body_length` is the only signal about
    the body itself — useful for "is this empty" / "is this huge" decisions
    without paying the context cost of the actual text.
    """

    key: str
    title: str
    category: str
    format: str = "plain_text"
    scope_type: str
    scope_id: str
    source: str
    body_length: int = 0
    updated_at: datetime | None = None


class PromptTemplateIndexResponse(BaseModel):
    """Compact prompt listing — names and metadata only, no bodies.

    Same filters as PromptTemplateListResponse, but each entry is a
    PromptTemplateSummary. Safe for agents to enumerate without dragging
    every prompt body into their context window.
    """

    prompts: list[PromptTemplateSummary]
    total_count: int
    filters: dict[str, Any] = Field(default_factory=dict)


class PromptTemplateUpsertRequest(BaseModel):
    """Create or update one prompt template."""

    body: str
    title: str | None = None
    category: str | None = None
    format: str = "plain_text"
    scope_type: Literal["global", "bot"] = "global"
    scope_id: str | None = None
    required_vars: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    updated_by: str | None = None
    change_note: str | None = None


class PromptTemplateValidateRequest(BaseModel):
    """Validate a candidate prompt body."""

    body: str | None = None
    required_vars: list[str] | None = None
    variables: dict[str, Any] = Field(default_factory=dict)


class PromptTemplateValidateResponse(BaseModel):
    """Validation result for a prompt template."""

    valid: bool
    required_vars: list[str] = Field(default_factory=list)
    placeholders: list[str] = Field(default_factory=list)
    missing_required: list[str] = Field(default_factory=list)
    unknown_placeholders: list[str] = Field(default_factory=list)
    rendered_preview: str | None = None
    errors: list[str] = Field(default_factory=list)


class PromptTemplateVersionResponse(BaseModel):
    """One stored prompt template version."""

    version: int
    body: str
    change_note: str | None = None
    created_by: str | None = None
    created_at: datetime


class PromptTemplateVersionsResponse(BaseModel):
    """Version history for a prompt key/scope."""

    key: str
    scope_type: str
    scope_id: str
    versions: list[PromptTemplateVersionResponse]
    total_count: int


class PromptTemplateSeedResponse(BaseModel):
    """Result of seeding built-in prompt templates into the DB."""

    created: int
    skipped: int
    total: int
    seeded_keys: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Avatar animation schemas
# ---------------------------------------------------------------------------
# TASK-214: the CRUD schemas (AvatarAnimationBase / Create / Update / Response)
# were used only by the deleted /v1/avatar/animations routes. The catalog now
# lives in bawthub Prisma. Per-request animation entries use the lighter
# ChatRequestAnimation defined near the top of this module.


# TASK-557: model/bot-catalog + history/memory schema domains were split
# into sibling modules. Re-imported here so `from ..schemas import X` is unchanged.
from .schemas_tasks import (  # noqa: E402,F401
    TaskSubmitRequest,
    TaskSubmitResponse,
    TaskStatusResponse,
    TaskListItem,
    TaskListResponse,
    ScheduledJobInfo,
    ScheduledJobsResponse,
    JobRunInfo,
    JobRunsResponse,
)
from .schemas_status import (  # noqa: E402,F401
    ServiceInfoSchema,
    MemoryInfoSchema,
    ModelStatusInfoSchema,
    DependencyInfoSchema,
    McpInfoSchema,
    BotSummarySchema,
    ConfigInfoSchema,
    SystemStatusResponse,
    HealthResponse,
)
from .schemas_models import (  # noqa: E402,F401
    ModelPricing,
    ModelInfo,
    ModelsResponse,
    ModelSwitchRequest,
    ModelSwitchResponse,
    ModelDetail,
)
from .schemas_history_memory import (  # noqa: E402,F401
    HistoryMessage,
    HistoryResponse,
    HistorySearchResponse,
    HistorySearchAllMessage,
    HistorySearchAllResponse,
    HistoryClearResponse,
    MemoryItem,
    MemorySearchRequest,
    MemorySearchResponse,
    MemoryStatsResponse,
    MemoryForgetRequest,
    MemoryForgetResponse,
    MemoryRestoreResponse,
    MemoryDeleteResponse,
    MemoryUpdateRequest,
    MessagePreview,
    MessagesPreviewResponse,
    RegenerateEmbeddingsResponse,
    ConsolidateRequest,
    ConsolidateResponse,
    SummarizableSession,
    SummarizePreviewResponse,
    SummarizeResponse,
    SummaryInfo,
    ListSummariesResponse,
    DeleteSummaryResponse,
)
