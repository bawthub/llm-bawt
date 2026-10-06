"""Message paging, hydration, and mutation routes for conversation history."""

from fastapi import APIRouter, HTTPException, Query

from ..dependencies import get_media_asset_store, get_service
from ..logging import get_service_logger
from ..schemas import HistoryClearResponse, HistoryMessage, HistoryResponse
from .history_scheduler import hydrate_scheduler_for_page

read_router = APIRouter()
mutation_router = APIRouter()
log = get_service_logger(__name__)

def _fetch_attachments_via_shared_engine(
    config,
    bot_id: str,
    message_ids: list[str],
) -> dict[str, list[dict]]:
    """Direct-SQL read of the bot's messages-partition ``attachments`` column.

    The route normally calls into ``PostgreSQLMemoryBackend.get_attachments_for_message_ids``
    via the embedded short-term manager, but MCP-server-mode deployments
    have no backend handle. This helper bypasses MCP and reads through
    the process-wide shared engine instead — same data, one round-trip,
    no new RPC. Returns the same ``{message_id: [refs]}`` shape.
    """
    from sqlalchemy import text
    from ...media.assets import _build_engine
    from ...memory.postgresql import MESSAGES_PARENT, partition_name

    engine = _build_engine(config)
    if engine is None:
        return {mid: [] for mid in message_ids}

    table = partition_name(MESSAGES_PARENT, bot_id)
    sql = text(f"SELECT id, attachments FROM {table} WHERE id = ANY(:ids)")
    result: dict[str, list[dict]] = {mid: [] for mid in message_ids}
    with engine.connect() as conn:
        rows = conn.execute(sql, {"ids": list(message_ids)}).mappings().all()
        for row in rows:
            refs = row.get("attachments") or []
            if isinstance(refs, list):
                result[row["id"]] = list(refs)
    return result


def _hydrate_attachments_for_page(
    service,
    bot_id: str,
    page_messages: list[dict],
) -> dict[str, list[dict]]:
    """Resolve per-message attachments for the given page (TASK-226).

    Returns ``{message_id: [resolved_attachment_dicts]}``. Messages with no
    media map to ``[]`` so callers can index unconditionally. The route
    drops orphan ``asset_id`` refs silently (already logged inside the
    serializer) — partial DB deletes never take the whole page down.

    Two DB round-trips per page regardless of message count:

    1. ``SELECT id, attachments FROM {bot}_messages WHERE id = ANY(...)``
    2. ``SELECT * FROM media_assets WHERE id = ANY(...)`` (only if the
       page references any assets at all).
    """
    message_ids = [str(m.get("id")) for m in page_messages if m.get("id")]
    if not message_ids:
        return {}

    # In embedded mode the short-term manager wraps a
    # PostgreSQLMemoryBackend we can call directly. In MCP-server mode
    # the manager has no ``_backend`` handle, so we fall back to a thin
    # direct SQL read against ``{bot}_messages.attachments`` using the
    # process-wide shared engine — same DB, just bypassing the MCP RPC
    # layer for this one read-only lookup. Both paths return the same
    # ``{message_id: [refs]}`` shape so downstream enrichment is unchanged.
    try:
        client = service.get_memory_client(bot_id)
        if not client:
            return {mid: [] for mid in message_ids}
        manager = client.get_short_term_manager()
        backend = getattr(manager, "_backend", None)

        if backend is not None:
            raw_refs_by_msg = backend.get_attachments_for_message_ids(message_ids)
        else:
            raw_refs_by_msg = _fetch_attachments_via_shared_engine(
                service.config, bot_id, message_ids
            )
    except Exception as e:
        log.warning("Failed to load attachment refs for history page: %s", e)
        return {mid: [] for mid in message_ids}

    # Run the cross-row enrichment through the canonical serializer so
    # the shape stays in lockstep with /v1/uploads and the chat-streaming
    # persistence path (TASK-225). The serializer mutates the wrapper
    # dicts in place; we use throwaway shells then extract.
    from ...media.serializers import enrich_attachments_for_messages

    shells: list[dict] = [
        {"_mid": mid, "attachments": raw_refs_by_msg.get(mid, [])}
        for mid in message_ids
    ]
    try:
        asset_store = get_media_asset_store(service.config)
        enrich_attachments_for_messages(shells, asset_store)
    except Exception as e:
        log.warning("Attachment enrichment failed: %s", e)
        return {mid: [] for mid in message_ids}

    return {s["_mid"]: s.get("attachments") or [] for s in shells}


def _fetch_reasoning_via_shared_engine(
    config,
    bot_id: str,
    message_ids: list[str],
) -> dict[str, str | None]:
    """Direct-SQL read of the bot's messages-partition ``reasoning`` column (TASK-301).

    Mirror of ``_fetch_attachments_via_shared_engine`` for the reasoning column —
    used when the short-term manager has no embedded backend handle. Returns
    ``{message_id: reasoning_or_None}``.
    """
    from sqlalchemy import text
    from ...media.assets import _build_engine
    from ...memory.postgresql import MESSAGES_PARENT, partition_name

    engine = _build_engine(config)
    if engine is None:
        return {mid: None for mid in message_ids}

    table = partition_name(MESSAGES_PARENT, bot_id)
    sql = text(f"SELECT id, reasoning FROM {table} WHERE id = ANY(:ids)")
    result: dict[str, str | None] = {mid: None for mid in message_ids}
    with engine.connect() as conn:
        rows = conn.execute(sql, {"ids": list(message_ids)}).mappings().all()
        for row in rows:
            result[row["id"]] = row.get("reasoning")
    return result


def _hydrate_reply_links_for_page(
    service,
    bot_id: str,
    page_messages: list[dict],
) -> dict[str, str]:
    """Map assistant message UUIDs to their canonical triggering user UUIDs."""
    assistant_ids = [
        str(message.get("id"))
        for message in page_messages
        if message.get("role") == "assistant" and message.get("id")
    ]
    if not assistant_ids:
        return {}

    store = getattr(service, "_turn_log_store", None)
    engine = getattr(store, "engine", None)
    if engine is None:
        return {}

    try:
        from sqlalchemy import text

        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT assistant_message_id, trigger_message_id FROM turn_logs "
                    "WHERE bot_id=:bot_id AND assistant_message_id = ANY(:assistant_ids) "
                    "AND trigger_message_id IS NOT NULL"
                ),
                {"bot_id": bot_id, "assistant_ids": assistant_ids},
            ).mappings().all()
        return {
            str(row["assistant_message_id"]): str(row["trigger_message_id"])
            for row in rows
            if row.get("assistant_message_id") and row.get("trigger_message_id")
        }
    except Exception as exc:
        log.warning("Failed to load history reply links: %s", exc)
        return {}


def _hydrate_interrupt_anchors_for_page(
    service,
    bot_id: str,
    page_messages: list[dict],
) -> dict[str, tuple[str, int] | None]:
    """Resolve durable mid-turn interrupt anchors for the given history page."""
    message_ids = [
        str(message.get("id"))
        for message in page_messages
        if message.get("role") == "user" and message.get("id")
    ]
    if not message_ids:
        return {}

    try:
        client = service.get_memory_client(bot_id)
        if not client:
            return {mid: None for mid in message_ids}
        manager = client.get_short_term_manager()
        backend = getattr(manager, "_backend", None)
        if backend is not None:
            return backend.get_interrupt_anchors_for_message_ids(message_ids)

        from sqlalchemy import text
        from ...media.assets import _build_engine
        from ...memory.postgresql import MESSAGES_PARENT, partition_name

        engine = _build_engine(service.config)
        if engine is None:
            return {mid: None for mid in message_ids}
        table = partition_name(MESSAGES_PARENT, bot_id)
        result: dict[str, tuple[str, int] | None] = {mid: None for mid in message_ids}
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    f"SELECT id, interrupt_source_message_id, interrupt_content_offset "
                    f"FROM {table} WHERE id = ANY(:ids)"
                ),
                {"ids": message_ids},
            ).mappings().all()
        for row in rows:
            source_id = row.get("interrupt_source_message_id")
            offset = row.get("interrupt_content_offset")
            if source_id and isinstance(offset, int) and offset >= 0:
                result[str(row["id"])] = (str(source_id), offset)
        return result
    except Exception as exc:
        log.warning("Failed to load interrupt anchors for history page: %s", exc)
        return {mid: None for mid in message_ids}


def _hydrate_reasoning_for_page(
    service,
    bot_id: str,
    page_messages: list[dict],
) -> dict[str, str | None]:
    """Resolve per-message reasoning for the given page (TASK-301).

    Returns ``{message_id: reasoning_or_None}``. One focused read of a column
    the canonical ``get_messages`` path drops so reasoning never re-enters LLM
    context; the chat UI uses it to restore the collapsed "Thought process" lane
    on reload. Embedded mode reads via the backend; server mode falls back to the
    shared engine — same shape either way.
    """
    message_ids = [str(m.get("id")) for m in page_messages if m.get("id")]
    if not message_ids:
        return {}

    try:
        client = service.get_memory_client(bot_id)
        if not client:
            return {mid: None for mid in message_ids}
        manager = client.get_short_term_manager()
        backend = getattr(manager, "_backend", None)
        if backend is not None:
            return backend.get_reasoning_for_message_ids(message_ids)
        return _fetch_reasoning_via_shared_engine(service.config, bot_id, message_ids)
    except Exception as e:
        log.warning("Failed to load reasoning for history page: %s", e)
        return {mid: None for mid in message_ids}

def _resolve_cursor(
    raw: str,
    visible_messages: list[dict],
) -> float | None:
    """Resolve a `before`/`after` cursor to a unix-timestamp cutoff.

    Accepts:
      1) a numeric unix timestamp (``"1717891234.567"``)
      2) an ISO-8601 timestamp (``"2026-06-06T14:00:00Z"``)
      3) a message ID — looked up in ``visible_messages`` for its timestamp

    Returns ``None`` if the raw string is empty after stripping. Raises
    ``HTTPException(400)`` only for the message-ID branch when the ID
    isn't found; numeric/ISO failures fall through silently because the
    caller may have legitimately passed empty/garbage.
    """
    from datetime import datetime

    trimmed = raw.strip()
    if not trimmed:
        return None

    # 1) Numeric unix timestamp
    try:
        return float(trimmed)
    except ValueError:
        pass

    # 2) ISO timestamp
    iso_candidate = trimmed.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(iso_candidate).timestamp()
    except ValueError:
        pass

    # 3) Message ID cursor
    cursor_msg = next(
        (m for m in visible_messages if str(m.get("id") or "") == trimmed),
        None,
    )
    if cursor_msg is None:
        raise HTTPException(status_code=400, detail="Invalid cursor")
    return float(cursor_msg.get("timestamp") or 0.0)


def _load_sorted_visible_messages(
    service,
    effective_bot_id: str,
    session_id: str | None = None,
    user_id: str | None = None,
) -> list[dict]:
    """Pull all visible messages for a bot, sorted chronologically.

    Shared by the per-bot history routes (``/v1/history``,
    ``/v1/history/around``). Filters out ``system`` / ``summary`` rows
    so summaries — which have their own surface — don't bleed into chat
    history paging.

    ``session_id`` (TASK-251): scope to one thread's transcript for the UI
    thread viewer. ``None`` (the default) = continuous scroll-back across
    all threads, unchanged.
    """
    client = service.get_memory_client(effective_bot_id)
    if not client:
        raise HTTPException(status_code=503, detail="Memory service unavailable")

    # TASK-251: forward the thread filter ONLY when set, so client doubles /
    # adapters without the parameter keep working on the continuous path.
    if session_id:
        messages = client.get_messages(since_seconds=None, session_id=session_id)
    else:
        messages = client.get_messages(since_seconds=None)
    visible = [m for m in messages if m.get("role") not in ("system", "summary")]
    if user_id:
        # Match the timeline/around owner rule without changing the legacy
        # unscoped history contract. Do not infer ownership from message author.
        from sqlalchemy import text
        from ...media.assets import _build_engine

        engine = _build_engine(service.config)
        if engine is None:
            raise HTTPException(status_code=503, detail="Session ownership unavailable")
        with engine.connect() as conn:
            owned = set(conn.execute(
                text("SELECT id FROM sessions WHERE bot_id=:bot_id AND user_id=:user_id"),
                {"bot_id": effective_bot_id, "user_id": user_id},
            ).scalars().all())
        visible = [m for m in visible if m.get("session_id") in owned]
    visible.sort(key=lambda m: (float(m.get("timestamp") or 0.0), str(m.get("id") or "")))
    return visible



def _message_author_payload(message: dict) -> dict:
    """Return the hydrated author or an explicit unresolved fallback."""
    author = message.get("author")
    if isinstance(author, dict):
        return author
    return {"entity_type": None, "entity_id": None, "status": "unknown"}

class _WindowAnchorNotFound(LookupError):
    """The anchor id is absent from the (optionally session-scoped) timeline."""


def _hydrate_window_authors(rows: list[dict], bot_id: str) -> list[dict]:
    from ...mcp_server.storage import get_storage

    return get_storage().hydrate_message_authors(rows, bot_id=bot_id)


def _load_window_via_sql(
    service,
    bot_id: str,
    message_id: str,
    before: int,
    after: int,
    session_id: str | None = None,
    user_id: str | None = None,
) -> tuple[list[dict], bool, bool] | None:
    """Indexed direct-SQL window around one anchor message (TASK-980).

    Returns ``(page_messages, has_older, has_newer)`` in ``(timestamp, id)``
    order, or ``None`` when the backend is unavailable. Raises
    :class:`_WindowAnchorNotFound` if the anchor isn't in scope.

    Reads the bot's partition directly rather than
    :meth:`PostgreSQLShortTermManager.get_messages`, whose summarization
    filter hides folded rows: a deep link (Spotlight, timeline rail,
    external URL) must land on ANY stored non-system/summary message.

    Cost is O(before + after), not O(history): one pkey lookup for the
    anchor, then two ``LIMIT`` range scans on ``(timestamp)`` (or
    ``(session_id, timestamp)`` when scoped). Each side fetches one extra
    row so the boundary flags need no count query. Ordering matches the
    previous full-table loader exactly: ``timestamp`` then ``id`` in the
    database collation. Authors are hydrated for the returned page only.
    """
    from sqlalchemy import text
    from ...media.assets import _build_engine
    from ...memory.postgresql import MESSAGES_PARENT, partition_name

    engine = _build_engine(service.config)
    if engine is None:
        return None

    from .history_scope import owned_message_scope

    table = partition_name(MESSAGES_PARENT, bot_id)
    columns = (
        "m.id, m.role, m.content, m.timestamp, m.session_id, "
        "m.author_entity_type, m.author_entity_id"
    )
    scope = "m.role NOT IN ('system', 'summary')"
    if session_id:
        scope += " AND m.session_id = :session_id"
    if user_id:
        scope += f" AND {owned_message_scope()}"
    older_sql = text(
        f"SELECT {columns} FROM {table} m WHERE {scope} "
        "AND (m.timestamp < :ts OR (m.timestamp = :ts AND m.id < :id)) "
        "ORDER BY m.timestamp DESC, m.id DESC LIMIT :lim"
    )
    newer_sql = text(
        f"SELECT {columns} FROM {table} m WHERE {scope} "
        "AND (m.timestamp > :ts OR (m.timestamp = :ts AND m.id > :id)) "
        "ORDER BY m.timestamp ASC, m.id ASC LIMIT :lim"
    )
    base = {"bot_id": bot_id, "user_id": user_id, "session_id": session_id}

    try:
        with engine.connect() as conn:
            anchor = conn.execute(
                text(f"SELECT {columns} FROM {table} m WHERE m.id = :id AND {scope}"),
                {**base, "id": message_id},
            ).mappings().first()
            if anchor is None:
                raise _WindowAnchorNotFound(message_id)
            cursor = {**base, "ts": anchor["timestamp"], "id": anchor["id"]}
            older = conn.execute(older_sql, {**cursor, "lim": before + 1}).mappings().all()
            newer = conn.execute(newer_sql, {**cursor, "lim": after + 1}).mappings().all()
    except _WindowAnchorNotFound:
        raise
    except Exception as e:
        log.warning(f"_load_window_via_sql failed for {bot_id}: {e}")
        return None

    has_older = len(older) > before
    has_newer = len(newer) > after
    ordered = [*reversed(older[:before]), anchor, *newer[:after]]
    raw_rows = [
        {
            "id": str(row["id"] or ""),
            "role": str(row["role"] or ""),
            "content": str(row["content"] or ""),
            "timestamp": float(row["timestamp"] or 0.0),
            "session_id": row.get("session_id"),
            "author_entity_type": row.get("author_entity_type"),
            "author_entity_id": row.get("author_entity_id"),
        }
        for row in ordered
    ]
    return _hydrate_window_authors(raw_rows, bot_id), has_older, has_newer


def _build_history_response(
    service,
    effective_bot_id: str,
    visible_messages: list[dict],
    page_messages: list[dict],
    candidate_count: int | None,
    *,
    has_older: bool | None = None,
    has_newer: bool | None = None,
    anchor_id: str | None = None,
    origin_user_id: str | None = None,
) -> HistoryResponse:
    """Hydrate attachments and assemble a HistoryResponse for a slice.

    Centralises the attachment hydration + boundary-flag work so the three
    history endpoints (legacy ``/v1/history`` backward, the new ``/v1/history``
    forward, and ``/v1/history/around``) all produce identically-shaped
    responses without copy-pasted glue.
    """
    attachments_by_id = _hydrate_attachments_for_page(
        service, effective_bot_id, page_messages
    )
    reasoning_by_id = _hydrate_reasoning_for_page(
        service, effective_bot_id, page_messages
    )
    reply_links_by_id = _hydrate_reply_links_for_page(
        service, effective_bot_id, page_messages
    )
    interrupt_anchors_by_id = _hydrate_interrupt_anchors_for_page(
        service, effective_bot_id, page_messages
    )
    scheduler_by_id = hydrate_scheduler_for_page(
        service, effective_bot_id, page_messages, origin_user_id
    )
    history_messages = []
    for msg in page_messages:
        message_id = str(msg.get("id") or "")
        interrupt_anchor = interrupt_anchors_by_id.get(message_id)
        history_messages.append(HistoryMessage(
            id=msg.get("id"),
            role=msg.get("role", ""),
            content=msg.get("content", ""),
            timestamp=msg.get("timestamp", 0.0),
            attachments=attachments_by_id.get(message_id, []),
            reasoning=reasoning_by_id.get(message_id),
            reply_to_message_id=reply_links_by_id.get(message_id),
            interrupt_source_message_id=(interrupt_anchor[0] if interrupt_anchor else None),
            interrupt_content_offset=(interrupt_anchor[1] if interrupt_anchor else None),
            author=_message_author_payload(msg),
            scheduler=scheduler_by_id.get(message_id) if msg.get("role") == "user" else None,
        ))

    oldest_timestamp = history_messages[0].timestamp if history_messages else None
    newest_timestamp = history_messages[-1].timestamp if history_messages else None

    # If callers passed has_older / has_newer explicitly, trust them. Otherwise
    # infer from the candidate vs page sizes (legacy single-direction path).
    resolved_has_older = (
        has_older
        if has_older is not None
        else bool(candidate_count is not None and candidate_count > len(page_messages))
    )
    resolved_has_newer = has_newer if has_newer is not None else False

    return HistoryResponse(
        bot_id=effective_bot_id,
        messages=history_messages,
        total_count=len(history_messages),
        has_more=resolved_has_older,
        has_older=resolved_has_older,
        has_newer=resolved_has_newer,
        oldest_timestamp=oldest_timestamp,
        newest_timestamp=newest_timestamp,
        anchor_id=anchor_id,
    )


@read_router.get("/v1/history", response_model=HistoryResponse, tags=["History"])
def get_history(
    bot_id: str = Query(None, description="Bot ID (uses default if not specified)"),
    limit: int = Query(50, description="Maximum number of messages to return"),
    user_id: str | None = Query(None, description="Optional owner scope for history and scheduling"),
    before: str | None = Query(
        None,
        description="Cursor for older history pages (ISO timestamp, unix timestamp, or message ID)",
    ),
    after: str | None = Query(
        None,
        description=(
            "Cursor for forward pagination — returns the OLDEST `limit` messages strictly newer "
            "than this cursor. Used after a deep-link landing (`/v1/history/around`) when the "
            "user scrolls down past the loaded window. Mutually exclusive with `before`."
        ),
    ),
    session_id: str | None = Query(
        None,
        description=(
            "TASK-251: scope the page to one thread's transcript (UI thread "
            "viewer). Absent = continuous scroll-back across all threads — "
            "the primary mode, unchanged."
        ),
    ),
):
    """Get conversation history for a bot.

    Two-direction pagination:

    - ``?before=<cursor>`` (default direction) — load older messages. Takes
      the NEWEST ``limit`` messages strictly older than the cursor.
    - ``?after=<cursor>`` — load newer messages. Takes the OLDEST ``limit``
      messages strictly newer than the cursor. Used to extend a deep-link
      window forward as the user scrolls down past it.

    Cursors accept unix timestamps, ISO-8601 strings, or message IDs.
    Passing both ``before`` and ``after`` is rejected (400).

    ``session_id`` (TASK-251) scopes the page to one thread; cursors work
    identically within the scoped transcript.
    """
    if before and after:
        raise HTTPException(
            status_code=400,
            detail="`before` and `after` are mutually exclusive",
        )

    service = get_service()
    effective_bot_id = bot_id or service._default_bot

    try:
        visible_messages = _load_sorted_visible_messages(
            service, effective_bot_id, session_id=session_id, user_id=user_id
        )

        before_ts = _resolve_cursor(before, visible_messages) if before else None
        after_ts = _resolve_cursor(after, visible_messages) if after else None

        if after_ts is not None:
            # Forward page: oldest `limit` messages strictly newer than the cursor.
            candidate_messages = [
                m for m in visible_messages
                if float(m.get("timestamp") or 0.0) > after_ts
            ]
            page_messages = candidate_messages[:limit] if limit > 0 else candidate_messages
            # In the forward direction `has_more` semantically means
            # "more *newer* messages exist beyond the returned page".
            has_older = False  # forward queries don't tell us about the older side
            has_newer = bool(candidate_messages and len(candidate_messages) > len(page_messages))
        else:
            # Backward page (default): newest `limit` strictly older than the cursor.
            if before_ts is not None:
                candidate_messages = [
                    m for m in visible_messages
                    if float(m.get("timestamp") or 0.0) < before_ts
                ]
            else:
                candidate_messages = visible_messages
            page_messages = (
                candidate_messages[-limit:]
                if limit > 0 and len(candidate_messages) > limit
                else candidate_messages
            )
            has_older = len(candidate_messages) > len(page_messages)
            has_newer = False  # legacy direction never reports the newer side

        return _build_history_response(
            service,
            effective_bot_id,
            visible_messages,
            page_messages,
            candidate_count=None,
            has_older=has_older,
            has_newer=has_newer,
            origin_user_id=user_id,
        )
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Failed to get history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@read_router.get("/v1/history/around", response_model=HistoryResponse, tags=["History"])
def get_history_around(
    bot_id: str = Query(..., description="Bot ID"),
    message_id: str = Query(..., description="Anchor message ID — window is centered on this row"),
    before: int = Query(30, ge=0, le=200, description="Number of older messages to include"),
    after: int = Query(10, ge=0, le=200, description="Number of newer messages to include"),
    user_id: str | None = Query(None, description="Owner scope for scheduling origin enrichment only"),
    session_id: str | None = Query(
        None,
        description=(
            "TASK-980: scope the window to one thread's transcript (thread "
            "viewer / timeline jumps). Absent = continuous timeline."
        ),
    ),
):
    """Return a window of messages around an anchor.

    Powers deep-link entry into chat surfaces (``/chat/<bot>?message=<id>``).
    Returns the anchor row plus ``before`` older + ``after`` newer rows,
    with ``has_older`` / ``has_newer`` flags so the frontend knows whether
    further pagination in either direction is possible.

    The returned ``oldest_timestamp`` and ``newest_timestamp`` are valid
    cursors for ``/v1/history?before=...`` and ``/v1/history?after=...``
    respectively, so continued scrolling stays on the standard paging
    surface — no separate "extend window" endpoint needed.
    """
    service = get_service()

    try:
        try:
            window = _load_window_via_sql(
                service, bot_id, message_id, before, after, session_id=session_id,
                user_id=user_id,
            )
        except _WindowAnchorNotFound:
            raise HTTPException(status_code=404, detail=f"Message {message_id!r} not found")
        if window is None:
            raise HTTPException(status_code=503, detail="Memory service unavailable")
        page_messages, has_older, has_newer = window

        return _build_history_response(
            service,
            bot_id,
            page_messages,
            page_messages,
            candidate_count=None,
            has_older=has_older,
            has_newer=has_newer,
            anchor_id=message_id,
            origin_user_id=user_id,
        )
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Failed to load history window around {message_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@mutation_router.delete("/v1/history", response_model=HistoryClearResponse, tags=["History"])
def clear_history(
    bot_id: str = Query(None, description="Bot ID (uses default if not specified)"),
):
    """Clear conversation history for a bot."""
    service = get_service()

    effective_bot_id = bot_id or service._default_bot

    try:
        # Try the fast path through the service-owned memory client (no model
        # loading and no request-local backend/schema bootstrap).
        client = service.get_memory_client(effective_bot_id)
        cleared = _clear_history_direct(client, effective_bot_id)

        if not cleared:
            # Fallback: use the full LLMBawt instance
            model_alias = list(service._available_models)[0] if service._available_models else None
            if not model_alias:
                raise HTTPException(status_code=500, detail="No models available")
            llm_bawt = service._get_llm_bawt(model_alias, effective_bot_id, service.config.DEFAULT_USER)
            llm_bawt.history_manager.clear_history()

        # Evict any cached LLMBawt instances for this bot
        stale_keys = [k for k in service._llm_bawt_cache if k[1] == effective_bot_id]
        for k in stale_keys:
            del service._llm_bawt_cache[k]

        return HistoryClearResponse(
            success=True,
            message=f"History cleared for bot '{effective_bot_id}'"
        )
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Failed to clear history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _clear_history_direct(client, bot_id: str) -> bool:
    """Clear messages and distilled memories without loading a model.

    Returns True only when both MemoryClient operations satisfy their
    contracts. A zero message count is still a successful idempotent clear.
    """
    if client is None:
        return False
    try:
        deleted_messages = client.clear_messages()
        messages_cleared = type(deleted_messages) is int and deleted_messages >= 0
        memories_cleared = client.clear_memories()
        if messages_cleared and memories_cleared is True:
            log.info("History cleared through cached client for bot '%s'", bot_id)
            return True
        log.warning(
            "Cached history clear incomplete for '%s': messages=%r memories=%r",
            bot_id,
            deleted_messages,
            memories_cleared,
        )
        return False
    except Exception as e:
        log.warning("Cached history clear failed for '%s': %s", bot_id, e)
        return False


@mutation_router.delete("/v1/history/{message_id}", response_model=HistoryClearResponse, tags=["History"])
def delete_message(
    message_id: str,
    bot_id: str = Query(None, description="Bot ID (uses default if not specified)"),
):
    """Delete (forget) a single message by its ID.

    Moves the message to the bot's ``*_forgotten_messages`` archive table so it
    stops appearing in history but stays recoverable. Backs the chat UI delete
    button (``DELETE /api/chat/history/{id}``); ``message_id`` is the DB UUID the
    ``GET /v1/history`` response carries, and a leading prefix (>= 8 chars) also
    matches. Returns 404 when no such message exists for the bot.

    NOTE: single-segment path, so it does not collide with the more specific
    ``DELETE /v1/history/summary/{summary_id}`` route (two segments) or the
    no-param ``DELETE /v1/history`` clear-all route.
    """
    service = get_service()
    effective_bot_id = bot_id or service._default_bot

    try:
        client = service.get_memory_client(effective_bot_id)
        if client is None:
            raise RuntimeError("Memory service unavailable")
        forgotten = client.ignore_message_by_id(message_id)

        if not forgotten:
            raise HTTPException(
                status_code=404,
                detail=f"Message '{message_id}' not found for bot '{effective_bot_id}'",
            )

        # Evict cached LLMBawt instances so their in-memory history_manager
        # doesn't keep serving the just-forgotten message on the next turn.
        stale_keys = [k for k in service._llm_bawt_cache if k[1] == effective_bot_id]
        for k in stale_keys:
            del service._llm_bawt_cache[k]

        return HistoryClearResponse(
            success=True,
            message=f"Message '{message_id}' deleted for bot '{effective_bot_id}'",
            deleted_count=1,
        )
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Failed to delete message {message_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
