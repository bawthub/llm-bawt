"""Per-bot and cross-bot history search routes."""

from fastapi import APIRouter, HTTPException, Query

from ..dependencies import get_service
from ..logging import get_service_logger
from ..schemas import (
    HistoryMessage,
    HistorySearchAllMessage,
    HistorySearchAllResponse,
    HistorySearchResponse,
)
from .history_pages import (
    _hydrate_attachments_for_page,
    _hydrate_reasoning_for_page,
    _hydrate_turn_meta_for_page,
    _load_sorted_visible_messages,
    _message_author_payload,
    _resolve_cursor,
)

router = APIRouter()
log = get_service_logger(__name__)

def _filter_messages_by_time_range(
    messages: list[dict],
    *,
    since: float | None = None,
    until: float | None = None,
) -> list[dict]:
    """Apply inclusive timestamp bounds to a sorted message list."""
    if since is None and until is None:
        return messages

    filtered: list[dict] = []
    for msg in messages:
        ts = float(msg.get("timestamp") or 0.0)
        if since is not None and ts < since:
            continue
        if until is not None and ts > until:
            continue
        filtered.append(msg)
    return filtered


def _search_visible_messages(
    messages: list[dict],
    query: str,
) -> list[dict]:
    """Case-insensitive substring search over visible message content."""
    query_lower = query.lower()
    if not query_lower:
        return messages
    return [
        msg for msg in messages
        if query_lower in str(msg.get("content", "")).lower()
    ]



def _build_history_search_response(
    service,
    effective_bot_id: str,
    query: str,
    matches: list[dict],
    page_messages: list[dict],
    *,
    has_older: bool,
) -> HistorySearchResponse:
    """Hydrate attachments and assemble the paginated search response."""
    attachments_by_id = _hydrate_attachments_for_page(
        service, effective_bot_id, page_messages
    )
    reasoning_by_id = _hydrate_reasoning_for_page(
        service, effective_bot_id, page_messages
    )
    turn_meta_by_id = _hydrate_turn_meta_for_page(
        service, effective_bot_id, page_messages
    )
    history_messages = [
        HistoryMessage(
            id=msg.get("id"),
            role=msg.get("role", ""),
            content=msg.get("content", ""),
            timestamp=msg.get("timestamp", 0.0),
            attachments=attachments_by_id.get(str(msg.get("id") or ""), []),
            reasoning=reasoning_by_id.get(str(msg.get("id") or "")),
            reply_to_message_id=(turn_meta_by_id.get(str(msg.get("id") or "")) or {}).get("reply_to_message_id"),
            author=_message_author_payload(msg),
        )
        for msg in page_messages
    ]

    oldest_timestamp = history_messages[0].timestamp if history_messages else None
    newest_timestamp = history_messages[-1].timestamp if history_messages else None

    return HistorySearchResponse(
        bot_id=effective_bot_id,
        query=query,
        messages=history_messages,
        total_count=len(matches),
        has_more=has_older,
        has_older=has_older,
        oldest_timestamp=oldest_timestamp,
        newest_timestamp=newest_timestamp,
    )


def _page_messages_before_cursor(
    messages: list[dict],
    *,
    before_ts: float | None,
    limit: int,
) -> tuple[list[dict], bool]:
    """Return the newest page of messages older than an optional cursor."""
    candidate_messages = [
        m for m in messages
        if before_ts is None or float(m.get("timestamp") or 0.0) < before_ts
    ]
    if limit > 0 and len(candidate_messages) > limit:
        return candidate_messages[-limit:], True
    return candidate_messages, False


def _resolve_cursor_against_messages(
    before: str | None,
    messages: list[dict],
) -> float | None:
    """Resolve an optional cursor against an already-filtered message set."""
    if not before:
        return None
    return _resolve_cursor(before, messages)


def _slice_history_search_matches(
    matches: list[dict],
    *,
    before_ts: float | None,
    limit: int,
) -> tuple[list[dict], bool]:
    """Return a paged history-search slice plus whether older matches remain."""
    return _page_messages_before_cursor(matches, before_ts=before_ts, limit=limit)


def _search_history_messages(
    service,
    effective_bot_id: str,
    query: str,
    *,
    limit: int,
    before: str | None,
    since: float | None,
    until: float | None,
) -> HistorySearchResponse:
    """Run per-bot history search with time-bounded, cursor-based paging."""
    visible_messages = _load_sorted_visible_messages(service, effective_bot_id)
    bounded_messages = _filter_messages_by_time_range(
        visible_messages,
        since=since,
        until=until,
    )
    matches = _search_visible_messages(bounded_messages, query)
    before_ts = _resolve_cursor_against_messages(before, matches)
    page_messages, has_older = _slice_history_search_matches(
        matches,
        before_ts=before_ts,
        limit=limit,
    )
    return _build_history_search_response(
        service,
        effective_bot_id,
        query,
        matches,
        page_messages,
        has_older=has_older,
    )


def _search_history_messages_legacy(
    service,
    effective_bot_id: str,
    query: str,
    *,
    limit: int,
) -> HistorySearchResponse:
    """Compatibility wrapper for callers expecting the old search behavior."""
    return _search_history_messages(
        service,
        effective_bot_id,
        query,
        limit=limit,
        before=None,
        since=None,
        until=None,
    )


def _search_history_time_bounds_description() -> str:
    """Human-readable docs fragment for time-bounded per-bot history search."""
    return "Inclusive unix-second timestamp bounds for constraining search results."


def _search_history_before_description() -> str:
    """Human-readable docs fragment for paged per-bot history search."""
    return "Cursor for older search-result pages (ISO timestamp, unix timestamp, or message ID)."


def _search_history_limit_description() -> str:
    """Human-readable docs fragment for search page size."""
    return "Maximum number of matching messages to return in this page."


def _search_history_query_description() -> str:
    """Human-readable docs fragment for the per-bot search query."""
    return "Case-insensitive substring query over visible bot history."


def _search_history_bot_description() -> str:
    """Human-readable docs fragment for per-bot search bot selection."""
    return "Bot ID (uses default if not specified)."


def _search_history_docstring() -> str:
    """Return the route docstring text for per-bot history search."""
    return "Search conversation history for a bot with cursor paging and optional time bounds."


def _search_history_error_log_message(query: str, error: Exception) -> str:
    """Format the per-bot history search error log."""
    return f"Failed to search history for query {query!r}: {error}"


def _resolve_search_limit(limit: int) -> int:
    """Normalize per-bot history search page size."""
    return max(0, limit)


def _resolve_search_bounds(
    since: float | None,
    until: float | None,
) -> tuple[float | None, float | None]:
    """Validate and normalize per-bot search time bounds."""
    if since is not None and until is not None and since > until:
        raise HTTPException(status_code=400, detail="`since` cannot be greater than `until`")
    return since, until


def _resolve_search_query(query: str) -> str:
    """Validate and normalize the incoming per-bot search query."""
    trimmed = query.strip()
    if not trimmed:
        raise HTTPException(status_code=400, detail="`query` cannot be empty")
    return trimmed


def _resolve_effective_bot_id(service, bot_id: str | None) -> str:
    """Apply default bot fallback shared by history routes."""
    return bot_id or service._default_bot


def _search_history_route_impl(
    service,
    query: str,
    bot_id: str | None,
    limit: int,
    before: str | None,
    since: float | None,
    until: float | None,
) -> HistorySearchResponse:
    """Shared implementation for the per-bot search route."""
    effective_bot_id = _resolve_effective_bot_id(service, bot_id)
    resolved_query = _resolve_search_query(query)
    resolved_limit = _resolve_search_limit(limit)
    resolved_since, resolved_until = _resolve_search_bounds(since, until)
    return _search_history_messages(
        service,
        effective_bot_id,
        resolved_query,
        limit=resolved_limit,
        before=before,
        since=resolved_since,
        until=resolved_until,
    )


def _handle_history_search_exception(query: str, error: Exception) -> None:
    """Log and raise a consistent per-bot search failure."""
    log.error(_search_history_error_log_message(query, error))
    raise HTTPException(status_code=500, detail=str(error))


def _run_history_search_route(
    query: str,
    bot_id: str | None,
    limit: int,
    before: str | None,
    since: float | None,
    until: float | None,
) -> HistorySearchResponse:
    """Thin wrapper so the route body stays compact and readable."""
    service = get_service()
    return _search_history_route_impl(
        service,
        query,
        bot_id,
        limit,
        before,
        since,
        until,
    )


def _history_search_route_doc() -> str:
    """Provide stable docs text without repeating long literals inline."""
    return _search_history_docstring()


def _history_search_limit_query() -> str:
    """Stable docs helper for per-bot search page size."""
    return _search_history_limit_description()


def _history_search_before_query() -> str:
    """Stable docs helper for per-bot search paging cursor."""
    return _search_history_before_description()


def _history_search_since_query() -> str:
    """Stable docs helper for lower time bound."""
    return _search_history_time_bounds_description()


def _history_search_until_query() -> str:
    """Stable docs helper for upper time bound."""
    return _search_history_time_bounds_description()


def _history_search_bot_query() -> str:
    """Stable docs helper for bot selection."""
    return _search_history_bot_description()


def _history_search_query_query() -> str:
    """Stable docs helper for query description."""
    return _search_history_query_description()


def _history_search_route(
    query: str,
    bot_id: str | None,
    limit: int,
    before: str | None,
    since: float | None,
    until: float | None,
) -> HistorySearchResponse:
    """Call the actual shared search implementation."""
    return _run_history_search_route(query, bot_id, limit, before, since, until)


def _history_search_legacy(
    query: str,
    bot_id: str | None,
    limit: int,
) -> HistorySearchResponse:
    """Legacy hook kept for any future callers needing pre-paging semantics."""
    service = get_service()
    effective_bot_id = _resolve_effective_bot_id(service, bot_id)
    resolved_query = _resolve_search_query(query)
    resolved_limit = _resolve_search_limit(limit)
    return _search_history_messages_legacy(
        service,
        effective_bot_id,
        resolved_query,
        limit=resolved_limit,
    )


def _history_search_route_body(
    query: str,
    bot_id: str | None,
    limit: int,
    before: str | None,
    since: float | None,
    until: float | None,
) -> HistorySearchResponse:
    """Route-body indirection so docs/logic stay separated."""
    return _history_search_route(query, bot_id, limit, before, since, until)


def _history_search_route_error(query: str, error: Exception) -> None:
    """Dedicated error path for the route body."""
    _handle_history_search_exception(query, error)


def _history_search_route_execute(
    query: str,
    bot_id: str | None,
    limit: int,
    before: str | None,
    since: float | None,
    until: float | None,
) -> HistorySearchResponse:
    """Route executor shared by POST surface(s)."""
    return _history_search_route_body(query, bot_id, limit, before, since, until)

@router.post("/v1/history/search", response_model=HistorySearchResponse, tags=["History"])
def search_history(
    query: str = Query(..., description="Case-insensitive substring query over visible bot history."),
    bot_id: str = Query(None, description="Bot ID (uses default if not specified)."),
    limit: int = Query(50, description="Maximum number of matching messages to return in this page."),
    before: str | None = Query(
        None,
        description="Cursor for older search-result pages (ISO timestamp, unix timestamp, or message ID).",
    ),
    since: float | None = Query(
        None,
        description="Inclusive unix-second lower bound for constraining search results.",
    ),
    until: float | None = Query(
        None,
        description="Inclusive unix-second upper bound for constraining search results.",
    ),
):
    """Search conversation history for a bot with cursor paging and optional time bounds."""
    try:
        return _run_history_search_route(
            query=query,
            bot_id=bot_id,
            limit=limit,
            before=before,
            since=since,
            until=until,
        )
    except HTTPException:
        raise
    except Exception as e:
        _handle_history_search_exception(query, e)
        raise  # unreachable, keeps type-checkers happy


@router.get("/v1/history/search", response_model=HistorySearchResponse, tags=["History"])
def search_history_get(
    query: str = Query(..., description="Case-insensitive substring query over visible bot history."),
    bot_id: str = Query(None, description="Bot ID (uses default if not specified)."),
    limit: int = Query(50, description="Maximum number of matching messages to return in this page."),
    before: str | None = Query(
        None,
        description="Cursor for older search-result pages (ISO timestamp, unix timestamp, or message ID).",
    ),
    since: float | None = Query(
        None,
        description="Inclusive unix-second lower bound for constraining search results.",
    ),
    until: float | None = Query(
        None,
        description="Inclusive unix-second upper bound for constraining search results.",
    ),
):
    """GET variant of per-bot history search for proxy routes that prefer query params."""
    try:
        return _run_history_search_route(
            query=query,
            bot_id=bot_id,
            limit=limit,
            before=before,
            since=since,
            until=until,
        )
    except HTTPException:
        raise
    except Exception as e:
        _handle_history_search_exception(query, e)
        raise  # unreachable, keeps type-checkers happy


def _parse_author_filter(author: str | None):
    """Split the ``type:id`` wire token once and validate it (TASK-1301 M3).

    A well-formed bot id is a filter, never authorization: membership is
    decided by the live policy inside the search.
    """
    from llm_bawt.message_authorship import normalize_author

    if author is None:
        return None
    entity_type, sep, entity_id = author.partition(":")
    if not sep:
        raise HTTPException(status_code=400, detail="author must be user:<id> or bot:<id>")
    try:
        return normalize_author(entity_type, entity_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None


@router.post(
    "/v1/history/search_all",
    response_model=HistorySearchAllResponse,
    tags=["History"],
)
async def search_all_history(
    query: str = Query(
        "",
        description=(
            "Search query (FTS keywords or substring/literal). May be empty "
            "only when ``author`` or ``in_bot`` is set (newest first)."
        ),
    ),
    limit: int = Query(50, description="Maximum total results across all bots"),
    role_filter: str | None = Query(
        None,
        description="Only include messages with this role (user|assistant). System messages are always excluded.",
    ),
    mode: str = Query(
        "fts",
        description=(
            "Search mode. 'fts' (default) uses Postgres full-text search "
            "with the english config — words stem + drop stop-words; "
            "identifiers (TASK-241, snake_case, paths) and quoted text match "
            "as phrases, hex runs as commit-SHA prefixes; ranked by ts_rank. "
            "'trgm' uses the pg_trgm extension for a literal substring match "
            "against the content, ranked by trigram similarity — best for "
            "fragments FTS can't tokenize (partial words, punctuation runs)."
        ),
    ),
    sort_by: str = Query(
        "relevance",
        description=(
            "Result ordering. 'relevance' (default) sorts by rank (ts_rank "
            "for FTS, similarity for trgm) then timestamp; lets the densest "
            "match win even if it's old. 'recent' sorts by timestamp only — "
            "best when the query is a common token and you want the latest "
            "hit (e.g. '/new' in a chat history)."
        ),
    ),
    since: str | None = Query(
        None,
        description=(
            "Lower-bound (inclusive). Accepts ISO date ('2026-06-01'), "
            "ISO datetime ('2026-06-01T14:30:00'), or Unix seconds (1782857172.0). "
            "Omit for unbounded history."
        ),
    ),
    until: str | None = Query(
        None,
        description=(
            "Upper-bound (inclusive). Same formats as ``since``. Omit for now."
        ),
    ),
    author: str | None = Query(
        None,
        description=(
            "Spotlight ``from:`` filter, ``user:<id>`` or ``bot:<id>``. "
            "Matches the EffectiveAuthorRule class: a user's own messages, or "
            "a public bot's replies and deliveries. Unknown and protected "
            "bots return an ordinary empty result."
        ),
    ),
    in_bot: str | None = Query(
        None,
        description="Spotlight ``in:`` filter: one public bot's history.",
    ),
):
    """Cross-bot message search.

    Both modes run through
    :meth:`llm_bawt.mcp_server.storage.MemoryStorage.search_messages_operator`
    over the live public partitions (TASK-1301):

    * ``mode=fts`` (default) — Postgres FTS ranked by ``ts_rank``. Words
      match by stem; identifiers (``TASK-241``, ``snake_case``, paths) and
      quoted text match as phrases, hex runs as commit-SHA prefixes
      (:class:`llm_bawt.retrieval.FtsQuery`).
    * ``mode=trgm`` — pg_trgm substring match (``content ILIKE`` +
      ``similarity()`` rank). Matches the literal query string verbatim.

    ``author`` / ``in_bot`` drive Spotlight's ``from:`` / ``in:`` filters and
    apply before LIMIT and ``total_count``. Each row carries ``author_class``.

    Returns ranked rows with bot attribution so a global search UI can
    render "who said this and when" without N+1 fans-out. First
    HTTP-exposed endpoint of the Spotlight Search project's Messages
    provider.
    """
    from llm_bawt.bot_access import BotAccessError
    from llm_bawt.mcp_server.server import _parse_timestamp
    from llm_bawt.mcp_server.storage import get_storage

    try:
        since_ts = _parse_timestamp(since)
        until_ts = _parse_timestamp(until)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    author_ref = _parse_author_filter(author)
    in_bot = (in_bot or "").strip().lower() or None
    if not query.strip() and author_ref is None and in_bot is None:
        # Deliberate (TASK-1301): an empty query used to return [] silently.
        raise HTTPException(status_code=400, detail="query is required unless author or in_bot is set")

    try:
        # Operator membership is a fresh positive policy read (TASK-1301 M1):
        # public partitions only, fail closed, every row author-classified.
        rows = await get_storage().search_messages_operator(
            query,
            mode=mode,
            author=author_ref,
            in_bot=in_bot,
            n_results=limit,
            role_filter=role_filter,
            sort_by=sort_by,
            since=since_ts,
            until=until_ts,
        )
    except BotAccessError as exc:
        log.error(f"Cross-bot history search unavailable: {exc.code}")
        raise HTTPException(status_code=503, detail=exc.code) from None
    except Exception as e:
        log.error(f"Failed cross-bot history search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    try:
        messages = [
            HistorySearchAllMessage(
                id=str(row.get("id", "")),
                role=str(row.get("role", "")),
                content=str(row.get("content", "")),
                timestamp=float(row.get("timestamp", 0.0) or 0.0),
                bot_id=str(row.get("source", "")),
                rank=float(row.get("rank", 0.0) or 0.0),
                author=_message_author_payload(row),
                author_class=row.get("author_class"),
            )
            for row in rows
        ]
        # The storage methods now stamp the unbounded total onto every row
        # via a window COUNT(*). Surface it as `total_count` so the UI can
        # render "showing N of M". Empty result → 0. Read from the first
        # row since the value is the same on all rows.
        unbounded_total = int(rows[0].get("total", 0)) if rows else 0
        return HistorySearchAllResponse(
            query=query,
            messages=messages,
            total_count=unbounded_total,
        )
    except Exception as e:
        log.error(f"Failed cross-bot history search: {e}")
        raise HTTPException(status_code=500, detail=str(e))
