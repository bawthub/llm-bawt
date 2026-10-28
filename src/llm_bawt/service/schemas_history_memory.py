"""History, memory, and summary request/response schemas.

Split out of ``service/schemas.py`` (TASK-557). ``schemas.py`` re-imports every
name here so ``from ..schemas import X`` across the service is unchanged.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class MessageAuthor(BaseModel):
    """Canonical message author, independent from protocol role."""

    entity_type: Literal["user", "bot"] | None = None
    entity_id: str | None = None
    status: Literal["canonical", "legacy", "unknown"] = "canonical"
    display_name: str = "Unknown participant"
    color: str = "slate"
    avatar: str | None = None
    avatar_render: str | None = None


class ApproverNotice(BaseModel):
    """An approval notice in an approver bot's chat (TASK-1661).

    Re-checked at read time: ``available`` is False (and ``detail`` None) once
    the bot may no longer see the card. ``detail`` is the masked approver view.
    """

    request_id: str
    kind: str
    available: bool
    detail: dict | None = None


class ScheduledMessageOrigin(BaseModel):
    """Trusted display-only provenance from a durable prompt occurrence."""

    schedule_id: str
    occurrence_id: str
    scheduled_for: str


class HistoryMessage(BaseModel):
    """A message in the conversation history.

    ``attachments`` (TASK-226) carries the resolved media-asset envelopes
    described in :mod:`llm_bawt.media.serializers`. Always present so
    frontend renderers can iterate unconditionally; an empty list means
    the row has no media. The list is hydrated by the
    ``/v1/history`` route only — DB layers strip it from the canonical
    ``get_messages`` path because LLM-prep code doesn't want it.
    """
    id: str | None = None
    role: str
    content: str
    timestamp: float
    attachments: list[dict] = []
    # TASK-301: persisted model reasoning ("thinking"), hydrated by the
    # /v1/history route only (display-only; never in LLM context). None when the
    # row carried no reasoning (user rows, pre-feature assistant rows).
    reasoning: str | None = None
    # Canonical reply edge for assistant rows. This is the originating user
    # message UUID from turn_logs, not an adjacency guess.
    reply_to_message_id: str | None = None
    # Turn stats for assistant rows, from the same turn_logs join: the model
    # that ran it, its token accounting, and real completion time (epoch
    # seconds; ``timestamp`` may be an ordering key). None when unknown.
    model: str | None = None
    token_usage: dict | None = None
    completed_at: float | None = None
    # Mid-turn steering anchor for user rows. Both fields are populated together.
    interrupt_source_message_id: str | None = None
    interrupt_content_offset: int | None = None
    author: MessageAuthor
    scheduler: ScheduledMessageOrigin | None = None
    approval_notice: ApproverNotice | None = None

class HistoryResponse(BaseModel):
    """Response for conversation history.

    Pagination flags describe the loaded window's boundaries against the
    bot's full timeline:

    - ``has_more`` / ``has_older``: more messages exist *before* the
      oldest row returned. The two aliases are kept in sync — ``has_more``
      pre-dates the deep-link work and is the field every legacy reader
      checks; ``has_older`` is the explicit name used by the deep-link
      ``/v1/history/around`` window endpoint and `?after=` forward paging.
    - ``has_newer``: more messages exist *after* the newest row returned.
      Newly added for deep-link windows and forward pagination; defaults
      ``False`` so legacy backward-only pagination behaves unchanged.
    - ``oldest_timestamp`` / ``newest_timestamp``: window boundaries used
      by the frontend as cursors for paginating in either direction.
    """
    bot_id: str
    messages: list[HistoryMessage]
    total_count: int
    has_more: bool = False
    has_older: bool = False
    has_newer: bool = False
    oldest_timestamp: float | None = None
    newest_timestamp: float | None = None
    anchor_id: str | None = None

class HistorySearchResponse(BaseModel):
    """Response for per-bot history search."""
    bot_id: str
    query: str
    messages: list[HistoryMessage]
    total_count: int
    has_more: bool = False
    has_older: bool = False
    oldest_timestamp: float | None = None
    newest_timestamp: float | None = None

class HistorySearchAllMessage(BaseModel):
    """A single hit from a cross-bot full-text message search.

    Mirrors :class:`HistoryMessage` plus the ``bot_id`` source attribution
    (so the frontend knows which bot's chat to deep-link into) and the FTS
    ``rank`` score (so the dropdown can re-rank or filter low-confidence
    matches). Carries no attachments — cross-bot search is content-only by
    design; the attachment hydration round-trip only fires after the user
    follows the link into the per-bot chat surface.
    """
    id: str
    role: str
    content: str
    timestamp: float
    bot_id: str
    rank: float
    author: MessageAuthor
    # EffectiveAuthorRule class (TASK-1301): who the row authoritatively is,
    # independent of display hydration. None = unknown (legacy/partial/foreign).
    author_class: Literal["bot", "user", "delivery"] | None = None

class HistorySearchAllResponse(BaseModel):
    """Response for cross-bot history search.

    ``messages`` are pre-sorted by FTS rank descending then timestamp
    descending (most recent breaks ties). ``total_count`` matches the
    length of the returned list; pagination cursors are not used because
    the storage layer applies the limit before merging across bots.
    """
    query: str
    messages: list[HistorySearchAllMessage]
    total_count: int

class HistoryClearResponse(BaseModel):
    """Response for clearing history."""
    success: bool
    message: str
    deleted_count: int = 0

class MemoryItem(BaseModel):
    """A memory item."""
    id: str | None = None
    content: str
    importance: float = 0.5
    relevance: float | None = None
    tags: list[str] = []
    created_at: float | str | None = None
    last_accessed: float | str | None = None
    access_count: int = 0
    source_message_ids: list[str] = []

class MemorySearchRequest(BaseModel):
    """Request for memory search."""
    query: str
    method: str = "all"  # text, embedding, high-importance, all
    limit: int = 10
    min_importance: float = 0.0
    bot_id: str | None = None

class MemorySearchResponse(BaseModel):
    """Response for memory search."""
    bot_id: str
    method: str
    query: str
    results: list[MemoryItem]
    total_count: int

class MemoryStatsResponse(BaseModel):
    """Memory statistics."""
    bot_id: str
    messages: dict
    memories: dict

class MemoryForgetRequest(BaseModel):
    """Request to forget messages."""
    count: int | None = None  # forget recent N
    minutes: int | None = None  # forget last N minutes
    message_id: str | None = None  # forget specific message by ID

class MemoryForgetResponse(BaseModel):
    """Response for forget operation."""
    success: bool
    messages_ignored: int
    memories_deleted: int
    message: str

class MemoryRestoreResponse(BaseModel):
    """Response for restore operation."""
    success: bool
    messages_restored: int
    message: str

class MemoryDeleteResponse(BaseModel):
    """Response for deleting a specific memory."""
    success: bool
    memory_id: str
    message: str

class MemoryUpdateRequest(BaseModel):
    """Request payload for updating a memory."""
    content: str | None = None
    importance: float | None = Field(default=None, ge=0.0, le=1.0)
    tags: list[str] | None = None

class MessagePreview(BaseModel):
    """Preview of a message for confirmation."""
    id: str  # UUID or int, stored as string
    role: str
    content: str
    timestamp: float | None = None

class MessagesPreviewResponse(BaseModel):
    """Response with message previews."""
    bot_id: str
    messages: list[MessagePreview]
    total_count: int

class RegenerateEmbeddingsResponse(BaseModel):
    """Response for regenerate embeddings operation."""
    success: bool
    updated: int
    failed: int
    embedding_dim: int | None = None
    message: str

class ConsolidateRequest(BaseModel):
    """Request for memory consolidation."""
    dry_run: bool = False
    similarity_threshold: float | None = None

class ConsolidateResponse(BaseModel):
    """Response for consolidation operation."""
    success: bool
    dry_run: bool
    clusters_found: int
    clusters_merged: int
    memories_consolidated: int
    new_memories_created: int
    errors: list[str] = []
    message: str

class SummarizableSession(BaseModel):
    """A session eligible for summarization."""
    start_timestamp: float
    end_timestamp: float
    start_time: str
    end_time: str
    message_count: int
    first_message: str
    last_message: str

class SummarizePreviewResponse(BaseModel):
    """Response for summarization preview."""
    bot_id: str
    sessions: list[SummarizableSession]
    total_messages: int

class SummarizeResponse(BaseModel):
    """Response for summarization operation."""
    success: bool
    sessions_summarized: int
    messages_summarized: int
    sessions_targeted: int | None = None
    summaries_replaced: int | None = None
    summaries_purged: int | None = None
    errors: list[str] = []

class SummaryInfo(BaseModel):
    """Information about a single summary."""
    id: str
    content: str
    timestamp: float
    session_start_time: str | None
    session_end_time: str | None
    message_count: int
    method: str

class ListSummariesResponse(BaseModel):
    """Response for listing summaries."""
    bot_id: str
    summaries: list[SummaryInfo]
    total_count: int

class DeleteSummaryResponse(BaseModel):
    """Response for deleting a summary."""
    success: bool
    summary_id: str | None = None
    messages_restored: int = 0
    detail: str | None = None
