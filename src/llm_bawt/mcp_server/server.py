"""FastMCP server exposing the BawtHub MCP toolset (formerly 'llm-bawt-memory').

This server exposes grouped tools for memory, messages, conversation
context, fact extraction, inter-bot messaging, system maintenance, and
the agent task system (tasks/steps/projects/activity).

Tools are namespaced by prefix:
    memory_*    — bot memory CRUD + search + maintenance
    messages_*  — conversation history CRUD + search + ignore/restore
    context_*   — combined recent message + memory context
    facts_*     — LLM-based fact extraction
    system_*    — service-wide stats and maintenance
    bots_*      — inter-bot messaging
    tasks_*, steps_*, projects_*, activity_*  — agent task system

Run standalone:
    uv run python -m llm_bawt.mcp_server
Or via entry point (after install):
    llm-memory
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from llm_bawt.shared.logging import LogConfig

# TASK-639 slice A: the FastMCP instance + host allowlist moved to
# ``mcp_server/registry.py`` so the approval-aware interceptor (Slice E) can
# wrap it without importing this 900+ line tool-registration module. We
# re-export ``mcp`` here so existing ``from .server import mcp`` imports
# across the tool_modules keep working byte-compatibly.
from .registry import mcp  # noqa: F401 — re-exported for downstream tool modules

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Storage accessor (lazy load to avoid import-time DB connection)
# ---------------------------------------------------------------------------

def _get_storage():
    from llm_bawt.mcp_server.storage import get_storage
    return get_storage()


def _parse_timestamp(value: str | float | int | None) -> float | None:
    """Convert a flexible date/time input to a Unix timestamp (float).

    Accepts:
      - None → None
      - float/int → returned as-is (assumed Unix seconds)
      - ISO-8601 date string: "2026-06-01" → start of that day UTC
      - ISO-8601 datetime: "2026-06-01T14:30:00" → that moment UTC
      - ISO-8601 with tz: "2026-06-01T14:30:00-04:00" → converted to UTC

    Raises ValueError on unparseable strings.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None

    s = value.strip()
    if not s:
        return None

    # Try as a bare number first (Unix seconds passed as a string).
    try:
        return float(s)
    except ValueError:
        pass

    # Try ISO-8601 datetime formats
    for fmt in (
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
    ):
        try:
            dt = datetime.strptime(s, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except ValueError:
            continue

    raise ValueError(f"Cannot parse timestamp: {value!r}")


# ---------------------------------------------------------------------------
# Memory Tools
# ---------------------------------------------------------------------------


@mcp.tool(name="memory_store")
async def store_memory(
    content: str,
    tags: list[str] | None = None,
    importance: float = 0.5,
    bot_id: str = "default",
    source_message_ids: list[str] | None = None,
) -> dict:
    """Store a new memory/fact in the knowledge base.

    Args:
        content: The memory content to store.
        tags: Categorization tags (identity, preference, etc.).
        importance: Importance score 0.0-1.0.
        bot_id: Bot namespace for isolation.
        source_message_ids: Message IDs this memory was derived from.

    Returns:
        Stored memory dict with generated id.
    """
    logger.debug("MCP tool invoked: tools/store_memory bot_id=%s", bot_id)
    storage = _get_storage()
    memory = await storage.store_memory(
        content=content,
        bot_id=bot_id,
        tags=tags,
        importance=importance,
        source_message_ids=source_message_ids,
    )
    return memory.to_dict()


@mcp.tool(name="memory_search")
async def search_memories(
    query: str,
    bot_id: str = "default",
    n_results: int = 10,
    min_relevance: float = 0.0,
    tags: list[str] | None = None,
) -> list[dict]:
    """Search memories using semantic similarity.

    Args:
        query: Natural language search query.
        bot_id: Bot namespace for isolation.
        n_results: Maximum results to return.
        min_relevance: Minimum similarity threshold 0-1.
        tags: Optional tag filter.

    Returns:
        List of memory dicts with relevance scores.
    """
    logger.debug("MCP tool invoked: tools/search_memories bot_id=%s", bot_id)
    storage = _get_storage()
    memories = await storage.search_memories(
        query=query,
        bot_id=bot_id,
        n_results=n_results,
        min_relevance=min_relevance,
        tags=tags,
    )
    return [m.to_dict() for m in memories]


@mcp.tool(name="memory_list_sources")
async def list_memory_sources() -> list[dict]:
    """List available memory sources (bot namespaces that have stored memories).

    Use this to discover which bots have memories you can search with
    ``search_memory_source``.

    Returns:
        List of dicts with 'source' (bot_id) and 'memory_count'.
    """
    logger.debug("MCP tool invoked: tools/list_memory_sources")
    storage = _get_storage()
    return await storage.list_memory_sources()


@mcp.tool(name="memory_search_source")
async def search_memory_source(
    source: str,
    query: str,
    n_results: int = 10,
    min_relevance: float = 0.0,
    tags: list[str] | None = None,
) -> list[dict]:
    """Search another bot's memories by source (read-only cross-bot search).

    This lets any bot look into a specific bot's memory store without
    modifying it.  Use ``list_memory_sources`` first to discover available
    sources.

    Args:
        source: The bot_id whose memories to search (e.g. "nova", "mira").
        query: Natural language search query.
        n_results: Maximum results to return.
        min_relevance: Minimum similarity threshold 0-1.
        tags: Optional tag filter.

    Returns:
        List of memory dicts with relevance scores.  Each dict also
        includes a 'source' key indicating which bot the memory belongs to.
    """
    logger.debug("MCP tool invoked: tools/search_memory_source source=%s", source)
    storage = _get_storage()
    memories = await storage.search_memories(
        query=query,
        bot_id=source,
        n_results=n_results,
        min_relevance=min_relevance,
        tags=tags,
    )
    results = []
    for m in memories:
        d = m.to_dict()
        d["source"] = source
        results.append(d)
    return results


@mcp.tool(name="messages_search_all")
async def search_all_messages(
    query: str,
    n_results: int = 10,
    role_filter: str | None = None,
    since: str | float | None = None,
    until: str | float | None = None,
    sort_by: str = "relevance",
    bot_id: str | None = None,
) -> list[dict]:
    """Full-text search across ALL bots' message histories at once.

    Use this when you need to find who was talking about a topic, without
    knowing which bot to search.  Much faster than searching each bot
    individually.

    Args:
        query: Search keywords or phrase.
        n_results: Maximum total results across all bots.
        role_filter: Only include messages with this role (user/assistant).
                     System messages are always excluded.
        since: Only include messages at or after this time.
              Accepts ISO date ("2026-06-01"), ISO datetime
              ("2026-06-01T14:30:00"), or Unix timestamp (1782857172.0).
        until: Only include messages at or before this time.
              Same formats as ``since``.
        sort_by: "relevance" (default; rank then recency) or "recent"
                 (recency only, ignores rank).
        bot_id: Restrict search to a single bot's history (e.g. "snark").
                Omit to search all bots.

    Returns:
        List of message dicts with 'source' (bot_id), role, content,
        timestamp, and full-text rank.
    """
    logger.debug("MCP tool invoked: tools/search_all_messages")
    storage = _get_storage()
    return await storage.search_all_messages(
        query=query,
        n_results=n_results,
        role_filter=role_filter,
        since=_parse_timestamp(since),
        until=_parse_timestamp(until),
        sort_by=sort_by,
        bot_id=bot_id,
    )


@mcp.tool(name="memory_search_all")
async def search_all_memories(
    query: str,
    n_results: int = 10,
    min_relevance: float = 0.0,
    since: float | None = None,
    until: float | None = None,
    bot_id: str | None = None,
) -> list[dict]:
    """Semantic search across ALL bots' memory stores at once.

    Use this when you need to find remembered facts without knowing which
    bot stored the memory.

    Args:
        query: Natural language search query.
        n_results: Maximum total results across all bots.
        min_relevance: Minimum similarity threshold 0-1.
        since: Only include memories created at or after this Unix timestamp.
        until: Only include memories created at or before this Unix timestamp.
        bot_id: Restrict search to a single bot's memory store (e.g. "snark").
                Omit to search all bots.

    Returns:
        List of memory dicts with 'source' (bot_id) and relevance scores.
    """
    logger.debug("MCP tool invoked: tools/search_all_memories")
    storage = _get_storage()
    return await storage.search_all_memories(
        query=query,
        n_results=n_results,
        min_relevance=min_relevance,
        since=since,
        until=until,
        bot_id=bot_id,
    )


@mcp.tool(name="context_get_recent")
async def get_recent_context(
    bot_id: str = "default",
    n_messages: int = 10,
    n_memories: int = 5,
    query: str | None = None,
) -> dict:
    """Get recent conversation context (messages + relevant memories).

    Args:
        bot_id: Bot namespace.
        user_id: User for profile context.
        max_messages: Maximum messages to return.
        max_age_seconds: Time window.

    Returns:
        Dict with keys: messages, memories.
    """
    logger.debug("MCP tool invoked: tools/get_recent_context bot_id=%s", bot_id)
    storage = _get_storage()
    messages = await storage.get_recent_messages(
        bot_id=bot_id,
        max_messages=n_messages,
        max_age_seconds=3600,
    )

    memories: list[dict] = []
    if n_memories > 0:
        search_query = query
        if not search_query and messages:
            search_query = messages[-1].content
        if search_query:
            mems = await storage.search_memories(
                query=search_query,
                bot_id=bot_id,
                n_results=n_memories,
                min_relevance=0.0,
            )
            memories = [m.to_dict() for m in mems]

    return {"messages": [m.to_dict() for m in messages], "memories": memories}


@mcp.tool(name="messages_add")
async def add_message(
    role: str,
    content: str,
    bot_id: str = "default",
    session_id: str | None = None,
    timestamp: float | None = None,
    message_id: str | None = None,
    attachments: list[dict] | None = None,
    reasoning: str | None = None,
    user_id: str | None = None,
    author_entity_type: str | None = None,
    author_entity_id: str | None = None,
) -> dict:
    """Add a message to conversation history.

    Args:
        role: user, assistant, or system.
        content: Message content.
        bot_id: Bot namespace.
        session_id: Optional session grouping.
        message_id: Optional client-supplied UUID for the message.
        attachments: TASK-222 — optional tiny JSONB ref list persisted on the
            message row's ``attachments`` column, e.g.
            ``[{"asset_id": "ma_xxx", "kind": "image"}]``. ``None`` leaves the
            column at its default ``[]``.
        user_id: TASK-284 — user namespace for resolving the active thread when
            ``session_id`` is not supplied. Sessions are keyed (bot_id, user_id).

    Returns:
        Stored message dict.
    """
    logger.debug("MCP tool invoked: tools/add_message bot_id=%s role=%s", bot_id, role)
    storage = _get_storage()
    message = await storage.add_message(
        role=role,
        content=content,
        bot_id=bot_id,
        session_id=session_id,
        timestamp=timestamp,
        message_id=message_id,
        attachments=attachments,
        reasoning=reasoning,
        user_id=user_id,
        author_entity_type=author_entity_type,
        author_entity_id=author_entity_id,
    )
    return message.to_dict()


@mcp.tool(name="facts_extract")
async def extract_facts(
    messages: list[dict],
    bot_id: str,  # Required - must be passed explicitly
    user_id: str,  # Required - must be passed explicitly
    store: bool = True,
    use_llm: bool = True,
) -> list[dict]:
    """Extract facts from conversation messages (LLM-based).

    Args:
        messages: List of message dicts with role/content.
        bot_id: Bot namespace (required).
        user_id: User ID for profile attribute extraction (required).
        store: Whether to persist extracted facts.
        use_llm: Whether to use LLM extraction (falls back to heuristics if False).

    Returns:
        List of extracted memory dicts.
    """
    if not bot_id:
        raise ValueError("bot_id is required for extract_facts")
    if not user_id:
        raise ValueError("user_id is required for extract_facts")
    from llm_bawt.mcp_server.extraction import extract_facts_from_messages
    
    facts = await extract_facts_from_messages(
        messages=messages,
        use_llm=use_llm,
        user_id=user_id,
    )
    
    if not facts:
        return []
    
    logger.debug(f"Extracted {len(facts)} facts from {len(messages)} messages")
    
    # Store facts as memories if requested
    if store:
        storage = _get_storage()
        stored_facts = []
        for fact in facts:
            memory = await storage.store_memory(
                content=fact["content"],
                bot_id=bot_id,
                tags=fact.get("tags", ["misc"]),
                importance=fact.get("importance", 0.5),
                source_message_ids=fact.get("source_message_ids", []),
            )
            stored_facts.append(memory.to_dict())
        return stored_facts
    
    return facts


@mcp.tool(name="memory_update")
async def update_memory(
    memory_id: str,
    bot_id: str = "default",
    content: str | None = None,
    tags: list[str] | None = None,
    importance: float | None = None,
) -> dict | None:
    """Update an existing memory.

    Args:
        memory_id: ID of memory to update.
        content: New content (optional).
        tags: New tags (optional).
        importance: New importance score (optional).
        bot_id: Bot namespace.

    Returns:
        Updated memory dict, or None if not found.
    """
    logger.debug("MCP tool invoked: tools/update_memory bot_id=%s memory_id=%s", bot_id, memory_id)
    storage = _get_storage()
    memory = await storage.update_memory(
        memory_id=memory_id,
        bot_id=bot_id,
        content=content,
        tags=tags,
        importance=importance,
    )
    return memory.to_dict() if memory else None


@mcp.tool(name="memory_delete")
async def delete_memory(
    memory_id: str,
    bot_id: str = "default",
) -> bool:
    """Delete a memory.

    Args:
        memory_id: ID of memory to delete.
        bot_id: Bot namespace.

    Returns:
        True if deleted successfully.
    """
    logger.debug("MCP tool invoked: tools/delete_memory bot_id=%s memory_id=%s", bot_id, memory_id)
    storage = _get_storage()
    result = await storage.delete_memory(
        memory_id=memory_id,
        bot_id=bot_id,
    )
    return result


@mcp.tool(name="memory_clear")
async def clear_memories(bot_id: str = "default") -> bool:
    """Delete all distilled memories for a bot."""
    logger.debug("MCP tool invoked: tools/clear_memories bot_id=%s", bot_id)
    storage = _get_storage()
    return await storage.clear_memories(bot_id=bot_id)


@mcp.tool(name="memory_supersede")
async def supersede_memory(
    old_memory_id: str,
    new_memory_id: str,
    bot_id: str = "default",
) -> bool:
    """Mark a memory as superseded by another (or DELETED)."""
    storage = _get_storage()
    backend = storage.get_backend(bot_id, provision_schema=True)
    if hasattr(backend, "supersede_memory"):
        return bool(backend.supersede_memory(old_memory_id, new_memory_id))  # type: ignore[attr-defined]
    return False


@mcp.tool(name="memory_list_recent")
async def list_recent(
    bot_id: str = "default",
    n: int = 50,
) -> list[dict]:
    """List recent memories (backend-native shape)."""
    storage = _get_storage()
    return await storage.list_recent_memories(bot_id=bot_id, n=n)


@mcp.tool(name="system_stats")
async def stats(bot_id: str = "default") -> dict:
    """Get memory/message stats."""
    logger.debug("MCP tool invoked: tools/stats bot_id=%s", bot_id)
    storage = _get_storage()
    return await storage.stats(bot_id=bot_id)


@mcp.tool(name="memory_list_high_importance")
async def list_memories(
    bot_id: str = "default",
    limit: int = 20,
    min_importance: float = 0.0,
) -> list[dict]:
    """List memories ordered by importance."""
    storage = _get_storage()
    return await storage.get_high_importance_memories(
        bot_id=bot_id,
        n_results=limit,
        min_importance=min_importance,
    )


@mcp.tool(name="messages_preview_recent")
async def preview_recent_messages(bot_id: str = "default", count: int = 10) -> list[dict]:
    storage = _get_storage()
    return await storage.preview_recent_messages(bot_id=bot_id, count=count)


@mcp.tool(name="messages_preview_since_minutes")
async def preview_messages_since_minutes(bot_id: str = "default", minutes: int = 60) -> list[dict]:
    storage = _get_storage()
    return await storage.preview_messages_since_minutes(bot_id=bot_id, minutes=minutes)


@mcp.tool(name="messages_preview_ignored")
async def preview_ignored_messages(bot_id: str = "default") -> list[dict]:
    storage = _get_storage()
    return await storage.preview_ignored_messages(bot_id=bot_id)


@mcp.tool(name="messages_ignore_recent")
async def ignore_recent_messages(bot_id: str = "default", count: int = 10) -> int:
    storage = _get_storage()
    return await storage.ignore_recent_messages(bot_id=bot_id, count=count)


@mcp.tool(name="messages_ignore_since_minutes")
async def ignore_messages_since_minutes(bot_id: str = "default", minutes: int = 60) -> int:
    storage = _get_storage()
    return await storage.ignore_messages_since_minutes(bot_id=bot_id, minutes=minutes)


@mcp.tool(name="messages_get_by_id")
async def get_message_by_id(
    bot_id: str = "default",
    message_id: str = "",
    before: int = 0,
    after: int = 0,
) -> dict | None:
    """Get a specific message by ID (supports prefix match), with optional
    surrounding conversation context.

    Args:
        bot_id: Bot whose history to search.
        message_id: Full UUID or prefix (min 8 chars).
        before: Number of messages to include before the match (by timestamp).
        after: Number of messages to include after the match (by timestamp).

    Returns:
        If before/after are both 0: a single message dict.
        If either is > 0: {message, before: [...], after: [...]}.
        None if not found.
    """
    storage = _get_storage()
    return await storage.get_message_by_id(
        bot_id=bot_id, message_id=message_id, before=before, after=after,
    )


@mcp.tool(name="messages_ignore_by_id")
async def ignore_message_by_id(bot_id: str = "default", message_id: str = "") -> bool:
    """Move a specific message to the forgotten table by ID (soft delete)."""
    storage = _get_storage()
    return await storage.ignore_message_by_id(bot_id=bot_id, message_id=message_id)


@mcp.tool(name="messages_restore_ignored")
async def restore_ignored_messages(bot_id: str = "default") -> int:
    storage = _get_storage()
    return await storage.restore_ignored_messages(bot_id=bot_id)


@mcp.tool(name="messages_get_for_summary")
async def get_messages_for_summary(bot_id: str = "default", summary_id: str = "") -> list[dict]:
    """Get raw user/assistant messages referenced by a summary row."""
    storage = _get_storage()
    return await storage.get_messages_for_summary(bot_id=bot_id, summary_id=summary_id)


@mcp.tool(name="messages_mark_recalled")
async def mark_messages_recalled(bot_id: str = "default", message_ids: list[str] | None = None) -> int:
    """Mark messages as recalled from summary expansion."""
    storage = _get_storage()
    return await storage.mark_messages_recalled(bot_id=bot_id, message_ids=message_ids)


@mcp.tool(name="memory_delete_by_source_messages")
async def delete_memories_by_source_message_ids(bot_id: str = "default", message_ids: list[str] | None = None) -> int:
    storage = _get_storage()
    return await storage.delete_memories_by_source_message_ids(bot_id=bot_id, message_ids=message_ids)


@mcp.tool(name="memory_regenerate_embeddings")
async def regenerate_embeddings(bot_id: str = "default", batch_size: int = 50) -> dict:
    storage = _get_storage()
    return await storage.regenerate_embeddings(bot_id=bot_id, batch_size=batch_size)


@mcp.tool(name="memory_consolidate")
async def consolidate_memories(
    bot_id: str = "default",
    dry_run: bool = True,
    similarity_threshold: float | None = None,
) -> dict:
    storage = _get_storage()
    return await storage.consolidate_memories(
        bot_id=bot_id,
        dry_run=dry_run,
        similarity_threshold=similarity_threshold,
    )


@mcp.tool(name="memory_update_meaning")
async def update_memory_meaning(
    bot_id: str = "default",
    memory_id: str = "",
    intent: str | None = None,
    stakes: str | None = None,
    emotional_charge: float | None = None,
    recurrence_keywords: list[str] | None = None,
    updated_tags: list[str] | None = None,
) -> bool:
    logger.debug("MCP tool invoked: tools/update_memory_meaning bot_id=%s memory_id=%s", bot_id, memory_id)
    storage = _get_storage()
    return await storage.update_memory_meaning(
        bot_id=bot_id,
        memory_id=memory_id,
        intent=intent,
        stakes=stakes,
        emotional_charge=emotional_charge,
        recurrence_keywords=recurrence_keywords,
        updated_tags=updated_tags,
    )


@mcp.tool(name="system_run_maintenance")
async def run_maintenance(
    bot_id: str = "default",
    run_consolidation: bool = True,
    run_recurrence_detection: bool = True,
    run_decay_pruning: bool = False,
    run_orphan_cleanup: bool = False,
    dry_run: bool = False,
) -> dict:
    logger.debug("MCP tool invoked: tools/run_maintenance bot_id=%s", bot_id)
    storage = _get_storage()
    return await storage.run_maintenance(
        bot_id=bot_id,
        run_consolidation=run_consolidation,
        run_recurrence_detection=run_recurrence_detection,
        run_decay_pruning=run_decay_pruning,
        run_orphan_cleanup=run_orphan_cleanup,
        dry_run=dry_run,
    )


@mcp.tool(name="messages_get")
async def get_messages(
    bot_id: str = "default",
    since_seconds: int | None = None,
    limit: int | None = None,
    session_id: str | None = None,
    summaries_only: bool = False,
    exclude_summarized: bool = False,
) -> list[dict]:
    """Get messages for building context windows.

    TASK-284: pass ``session_id`` to scope the read to one durable thread's
    transcript (direct-table read). Omit it for the existing summary-aware
    behaviour. ``summaries_only`` returns only the rolling summary husks — the
    session-scoped v2 read (step 12) composes it with a thread's raw bubbles.
    ``exclude_summarized`` drops already-summarized bubbles from a session-scoped
    read so raw and summary rows stay disjoint (no double-loading).
    """
    logger.debug(
        "MCP tool invoked: tools/get_messages bot_id=%s session_id=%s summaries_only=%s excl_summ=%s",
        bot_id, session_id, summaries_only, exclude_summarized,
    )
    storage = _get_storage()
    return await storage.get_messages(
        bot_id=bot_id, since_seconds=since_seconds, limit=limit,
        session_id=session_id, summaries_only=summaries_only,
        exclude_summarized=exclude_summarized,
    )


@mcp.tool(name="messages_clear")
async def clear_messages(bot_id: str = "default") -> int:
    """Delete all messages for a bot."""
    logger.debug("MCP tool invoked: tools/clear_messages bot_id=%s", bot_id)
    storage = _get_storage()
    return await storage.clear_messages(bot_id=bot_id)


@mcp.tool(name="messages_remove_last_partial")
async def remove_last_message_if_partial(bot_id: str = "default", role: str = "assistant") -> bool:
    logger.debug("MCP tool invoked: tools/remove_last_message_if_partial bot_id=%s role=%s", bot_id, role)
    storage = _get_storage()
    return await storage.remove_last_message_if_partial(bot_id=bot_id, role=role)


# ---------------------------------------------------------------------------
# Focused tool modules (registered via import side-effect)
# ---------------------------------------------------------------------------

from .session_tools import (  # noqa: E402,F401
    close_session,
    get_active_session,
    get_or_create_active_session,
    get_session,
    list_sessions,
    rotate_session,
)
from .inter_bot_tools import (  # noqa: E402,F401
    _await_delivery,
    _bot_send_wait_ceiling_seconds,
    _check_bot_in_turn,
    _enqueue_durable,
    _find_delivery_by_key,
    _get_json,
    cancel_delivery,
    get_delivery,
    list_available_bots,
    list_deliveries,
    send_message_to_bot,
)
from .profile_tools import _get_profile_manager, profile  # noqa: E402,F401

# ---------------------------------------------------------------------------
# Task system tools (registered via import side-effect)
# ---------------------------------------------------------------------------

from . import initiative_tools as _initiative_tools  # noqa: F401, E402
from . import project_tools as _project_tools  # noqa: F401, E402
from . import step_tools as _step_tools  # noqa: F401, E402
from . import task_tools as _task_tools  # noqa: F401, E402

# ---------------------------------------------------------------------------
# Media library tools (Sonarr / Radarr / SABnzbd)
# ---------------------------------------------------------------------------

from . import media_tools as _media_tools  # noqa: F401, E402

# ---------------------------------------------------------------------------
# Image generation tool (Grok Imagine — text->image + iterate)
# ---------------------------------------------------------------------------

from . import media_generation_tools as _media_generation_tools  # noqa: F401, E402
from . import home_audio_tools as _home_audio_tools  # noqa: F401, E402

# ---------------------------------------------------------------------------
# Self-recap tool (agent continuation briefing via Grok)
# ---------------------------------------------------------------------------

from . import self_tools as _self_tools  # noqa: F401, E402

# ---------------------------------------------------------------------------
# Ops tools (TASK-639 — DB-configured operations catalog)
# ---------------------------------------------------------------------------

from . import ops_tools as _ops_tools  # noqa: F401, E402

# ---------------------------------------------------------------------------
# Web search tool (local Brave/Reddit/Tavily fan-out — replaces the Anthropic
# server-side WebSearch that hangs on the proxy path)
# ---------------------------------------------------------------------------

from . import search_tools as _search_tools  # noqa: F401, E402

# ---------------------------------------------------------------------------
# Run helpers
# ---------------------------------------------------------------------------


def run_server(
    transport: str | None = None,
    host: str = "0.0.0.0",
    port: int = 8001,
) -> None:
    """Run the MCP server.

    When called as CLI entry point, parses command-line arguments.
    When called programmatically, uses provided arguments.

    Args:
        transport: 'stdio' or 'http'. Defaults to 'http' for service mode.
        host: Bind host for HTTP transport.
        port: Bind port for HTTP transport.
    """
    import argparse

    # Parse CLI arguments when called as entry point
    parser = argparse.ArgumentParser(description="Run the MCP mcp server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default=None,
        help="Transport protocol: 'stdio' for local tools, 'http' for web services (default: http)",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Bind host for HTTP transport (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8001,
        help="Bind port for HTTP transport (default: 8001)",
    )

    args, _ = parser.parse_known_args()

    # CLI args override function arguments
    final_transport = args.transport or transport or "http"
    final_host = args.host if args.host != "0.0.0.0" else host
    final_port = args.port if args.port != 8001 else port

    # Default to verbose logging for the MCP server so tool invocations
    # are visible when running standalone. Can be disabled via env.
    verbose_env = os.getenv("LLM_BAWT_MCP_SERVER_VERBOSE", "1").lower()
    debug_env = os.getenv("LLM_BAWT_MCP_SERVER_DEBUG", "0").lower()
    verbose = verbose_env not in {"0", "false", "no"}
    debug = debug_env in {"1", "true", "yes"}
    
    # Only configure logging if running standalone (not in-process within llm-service)
    # Check if we're in the main thread - if not, we're likely in a daemon thread
    import threading
    is_standalone = threading.current_thread() is threading.main_thread()
    
    if is_standalone:
        LogConfig.configure(verbose=verbose or debug, debug=debug)
    
    # Suppress uvicorn access logs - FastMCP starts uvicorn internally,
    # and we log our own MCP operation summaries
    if not debug:
        logging.getLogger("uvicorn.access").setLevel(logging.ERROR)
        logging.getLogger("uvicorn.error").setLevel(logging.WARNING)

    # Only log startup message when running standalone
    if is_standalone:
        logger.info("Starting MCP server: transport=%s host=%s port=%s", final_transport, final_host, final_port)

    if final_transport == "stdio":
        mcp.run(transport="stdio")
    else:
        # Use streamable-http which exposes /mcp endpoint for JSON-RPC calls.
        # TASK-701 wraps only this generated ASGI app so an opaque, per-request
        # task-turn capability can reach task tools without becoming a model
        # argument or leaking through process-global state.
        import uvicorn
        from .task_association import TaskTurnCapabilityMiddleware

        mcp.settings.host = final_host
        mcp.settings.port = final_port
        app = TaskTurnCapabilityMiddleware(mcp.streamable_http_app())
        config = uvicorn.Config(
            app,
            host=final_host,
            port=final_port,
            log_level=mcp.settings.log_level.lower(),
        )
        uvicorn.Server(config).run()


if __name__ == "__main__":
    run_server()
