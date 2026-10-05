"""Turn log retrieval routes."""

import json
import logging

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field
from fastapi.responses import Response
from sqlalchemy import func
from sqlmodel import Session, select

from ..dependencies import get_service, get_turn_log_store
from ..schemas import (
    RecentBotTurn,
    RecentByBotsResponse,
    ToolCallEvent,
    ToolCallEventsResponse,
    TurnLogDetail,
    TurnLogListItem,
    TurnLogListResponse,
)
from ..tool_call_events import extract_trigger_message, message_id_matches, parse_message_filters
from ..tool_call_store import ToolCallRecord, ToolCallStore
from ..turn_logs import TurnLogStore

log = logging.getLogger(__name__)

router = APIRouter()


def _parse_json(value: str | None):
    if not value:
        return None
    try:
        return json.loads(value)
    except Exception:
        return value


def _parse_token_usage(value: str | None) -> dict | None:
    """Decode token_usage_json from a turn-log row, returning None on bad JSON.

    The column may be missing on rows persisted before this field was added —
    `getattr(row, "token_usage_json", None)` already handles that case; here we
    just guard against malformed JSON so a single bad row can't 500 the route.
    """
    if not value:
        return None
    try:
        parsed = json.loads(value)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def _records_to_calls(records: list[ToolCallRecord], *, turn_ended_at=None) -> list[dict]:
    """Shape ``tool_call_records`` rows into the frontend tool-card dict list.

    TASK-364: ``ToolCallRecord`` is the single canonical source of tool-call
    data — written incrementally as tool_start/tool_end fire (via the Redis
    sink), it survives mid-stream reloads and terminated turns. The former
    ``turn_logs.tool_calls_json`` blob was a vestigial finalize-time copy and
    is no longer read. Rows are ordered by ``created_at`` (call order); each
    dict carries per-call timing and the SDK ids that let reload nest
    sub-agent children under their parent.
    """
    ordered = sorted(records, key=lambda r: (r.created_at is None, r.created_at))
    out: list[dict] = []
    for row in ordered:
        args = _parse_json(row.arguments_json) or {}
        if not isinstance(args, dict):
            args = {"raw": args}
        # In-flight calls (no ended_at) MUST have result=None so the frontend
        # renders "running…".  Empty string would be treated as completed.
        interrupted = row.result_text is None and (
            turn_ended_at is not None or (row.ended_at is not None and row.result_complete is False)
        )
        is_finished = row.ended_at is not None or (row.result_text is not None and row.result_text != "")
        result_value = row.result_text if is_finished else None
        out.append({
            "iteration": row.iteration or 1,
            "index": len(out) + 1,
            "tool": row.tool_name,
            "name": row.tool_name,
            "arguments": args,
            "parameters": args,
            "result": result_value,
            "status": "interrupted" if interrupted else ("completed" if is_finished else "pending"),
            "call_id": row.call_id,
            "started_at": row.started_at,
            "ended_at": row.ended_at,
            "duration_ms": row.duration_ms,
            "text_offset": row.text_offset,
            "is_error": row.is_error,
            # TASK-344: SDK ids so sub-agent nesting survives reload.
            "tool_use_id": row.tool_use_id,
            "parent_tool_use_id": row.parent_tool_use_id,
            # TASK-305: approval gate data for history-recalled tool cards.
            "approval_request_id": row.approval_request_id,
            "approval_status": row.approval_status,
            "preapproved": row.preapproved,
            # TASK-977: raw {asset_id, kind} refs; _enrich_call_attachments
            # turns them into URL envelopes before the response leaves.
            "attachments": _parse_json(row.attachments_json) or [],
            "result_meta": {
                "record_id": row.id,
                "preview_chars": len(row.result_text or ""),
                "total_chars": row.result_total_chars,
                "total_bytes": row.result_total_bytes,
                "total_tokens": row.result_total_tokens,
                "sha256": row.result_sha256,
                "content_type": row.result_content_type or "text/plain",
                "complete": row.result_complete,
                "available": bool(row.result_payload_available),
                "legacy": row.result_complete is None,
            },
        })
    return out


def _enrich_call_attachments(calls: list[dict]) -> None:
    """Resolve per-call attachment refs into envelopes in place (TASK-977).

    One batched media_assets lookup for every call, through the same
    serializer /v1/history uses, so tool-card previews are wire-identical to
    the live tool_end envelopes. Unresolvable refs drop out; any failure
    degrades to no inline media rather than failing the tool-call listing.
    """
    carriers = [c for c in calls if c.get("attachments")]
    if not carriers:
        return
    try:
        from ...media.serializers import enrich_attachments_for_messages
        from ..dependencies import get_media_asset_store

        enrich_attachments_for_messages(carriers, get_media_asset_store(get_service().config))
    except Exception as exc:
        log.warning("Tool-call attachment enrichment failed: %s", exc)
        for call in carriers:
            call["attachments"] = []


def _live_tool_calls(store: TurnLogStore, turn_id: str) -> list[dict]:
    """Read tool-call rows from `tool_call_records` for a single turn.

    TASK-364: this is now the canonical read path for every turn (not just
    in-flight ones) — the `tool_calls_json` blob is no longer written or read.
    """
    if store.engine is None or not turn_id:
        return []
    try:
        with Session(store.engine) as session:
            rows = list(
                session.exec(
                    select(ToolCallRecord)
                    .where(ToolCallRecord.turn_id == turn_id)
                    .order_by(ToolCallRecord.created_at)
                ).all()
            )
    except Exception:
        return []
    parent = store.get_turn(turn_id)
    calls = _records_to_calls(rows, turn_ended_at=parent.ended_at if parent else None)
    _enrich_call_attachments(calls)
    return calls


def _live_tool_call_counts(store: TurnLogStore, turn_ids: list[str]) -> dict[str, int]:
    """Count realtime tool-call rows for multiple turns in one query."""
    if store.engine is None or not turn_ids:
        return {}
    try:
        with Session(store.engine) as session:
            rows = session.exec(
                select(ToolCallRecord.turn_id, func.count())
                .where(ToolCallRecord.turn_id.in_(turn_ids))
                .group_by(ToolCallRecord.turn_id)
            ).all()
    except Exception:
        return {}

    counts: dict[str, int] = {}
    for turn_id, count in rows:
        if turn_id:
            counts[str(turn_id)] = int(count or 0)
    return counts


_TAIL_MAX = 1500
_TAIL_MIN = 200


def _response_tail(response_text: str) -> str | None:
    """Extract the bot's final output from a (potentially long) response.

    Strategy (in priority order):
    1. If the response is short enough, return it whole.
    2. Find the last markdown heading (## or #) — the final summary section.
       If the section is too short (< _TAIL_MIN), include the preceding heading
       section too so the reader gets enough context.
    3. Fallback: last _TAIL_MAX chars trimmed to the nearest paragraph break.
    """
    if not response_text:
        return None
    text = response_text.rstrip()
    if len(text) <= _TAIL_MAX:
        return text

    # Strategy 2: find the last markdown heading
    # Search from the end for lines starting with # or ##
    lines = text.split("\n")
    last_heading_idx = -1
    prev_heading_idx = -1
    for i in range(len(lines) - 1, -1, -1):
        stripped = lines[i].lstrip()
        if stripped.startswith("#"):
            if last_heading_idx == -1:
                last_heading_idx = i
            elif prev_heading_idx == -1:
                prev_heading_idx = i
                break

    if last_heading_idx >= 0:
        section = "\n".join(lines[last_heading_idx:]).strip()
        # If too short, pull in the previous heading section
        if len(section) < _TAIL_MIN and prev_heading_idx >= 0:
            section = "\n".join(lines[prev_heading_idx:]).strip()
        if section and len(section) <= _TAIL_MAX:
            return section

    # Strategy 3: fallback — last _TAIL_MAX chars, trimmed to paragraph break
    tail = text[-_TAIL_MAX:]
    pp = tail.find("\n\n")
    if 0 < pp < len(tail) - _TAIL_MIN:
        tail = tail[pp + 2:]
    return tail


@router.get("/v1/turn-logs", response_model=TurnLogListResponse, tags=["Debug"])
def list_turn_logs(
    bot_id: str | None = Query(None, description="Filter by bot ID"),
    user_id: str | None = Query(None, description="Filter by user ID"),
    model: str | None = Query(None, description="Filter by model alias"),
    request_id: str | None = Query(None, description="Filter by request ID"),
    status: str | None = Query(None, description="Filter by status"),
    active_only: bool = Query(False, description="Only in-progress turns (ended_at IS NULL), path-agnostic"),
    stream: bool | None = Query(None, description="Filter by streaming mode"),
    has_tools: bool | None = Query(None, description="Filter by presence of tool calls"),
    since_hours: int = Query(168, ge=1, le=168, description="Recent N hours for historical turns; ignored with active_only"),
    limit: int = Query(100, ge=1, le=200, description="Max rows to return"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
):
    """List persisted turn logs (24h TTL by default)."""
    service = get_service()
    store = get_turn_log_store()
    if store.engine is None:
        raise HTTPException(status_code=503, detail="Turn logs DB unavailable")

    rows, total_count = store.list_turns(
        bot_id=bot_id,
        user_id=user_id,
        model=model,
        request_id=request_id,
        status=status,
        active_only=active_only,
        stream=stream,
        has_tools=has_tools,
        since_hours=since_hours,
        limit=limit,
        offset=offset,
    )

    # TASK-364: tool-call counts come solely from tool_call_records (the
    # canonical store) for every turn — the tool_calls_json blob is retired.
    live_tool_counts = _live_tool_call_counts(store, [row.id for row in rows])

    # TASK-661: batch-fetch per-turn changed-file summaries so history
    # hydration can graft file links via the existing turn-log backfill.
    changed_files_by_turn: dict[str, dict] = {}
    try:
        changed_files_by_turn = store.changed_files_summaries_for_turns(
            [row.id for row in rows]
        )
    except Exception:
        changed_files_by_turn = {}

    # TASK-269: embed the persisted question for turns that ended on one, so the
    # UI can render a QuestionMessage straight from history hydration.
    pq_store = getattr(service, "_pending_question_store", None)
    questions_by_turn: dict[str, dict] = {}
    if pq_store is not None:
        for row in rows:
            if getattr(row, "end_reason", None) != "question":
                continue
            qrow = None
            qid = getattr(row, "question_id", None)
            if qid:
                qrow = pq_store.get(qid)
            if qrow is None:
                qrow = pq_store.get_by_turn(row.id)
            if qrow is not None:
                questions_by_turn[row.id] = pq_store.row_to_dict(qrow)

    items = []
    for row in rows:
        tool_call_count = live_tool_counts.get(row.id, 0)
        response_text = row.response_text or ""
        response_preview = response_text[:300] if response_text else None
        items.append(
            TurnLogListItem(
                id=row.id,
                created_at=row.created_at,
                request_id=row.request_id,
                path=row.path,
                stream=row.stream,
                model=row.model,
                bot_id=row.bot_id,
                user_id=row.user_id,
                status=row.status,
                latency_ms=row.latency_ms,
                user_prompt=row.user_prompt,
                response_preview=response_preview,
                response_tail=_response_tail(response_text),
                response_chars=len(response_text),
                response_preview_truncated=bool(response_text and len(response_text) > len(response_preview or "")),
                tool_call_count=tool_call_count,
                error_text=row.error_text,
                animation=getattr(row, "animation", None),
                agent_session_key=getattr(row, "agent_session_key", None),
                agent_request_id=getattr(row, "agent_request_id", None),
                trigger_message_id=getattr(row, "trigger_message_id", None),
                assistant_message_id=getattr(row, "assistant_message_id", None),
                token_usage=_parse_token_usage(getattr(row, "token_usage_json", None)),
                end_reason=getattr(row, "end_reason", None),
                question_id=getattr(row, "question_id", None),
                parent_turn_id=getattr(row, "parent_turn_id", None),
                ended_at=getattr(row, "ended_at", None),
                question=questions_by_turn.get(row.id),
                changed_files=changed_files_by_turn.get(row.id),
            )
        )

    return TurnLogListResponse(
        turns=items,
        total_count=total_count,
        filters={
            "bot_id": bot_id,
            "user_id": user_id,
            "model": model,
            "request_id": request_id,
            "status": status,
            "active_only": active_only,
            "stream": stream,
            "has_tools": has_tools,
            "since_hours": since_hours,
            "limit": limit,
            "offset": offset,
        },
    )


@router.get(
    "/v1/turn-logs/recent-by-bot",
    response_model=RecentByBotsResponse,
    tags=["Debug", "History"],
)
def recent_turns_by_bot(
    user_id: str = Query(..., description="User to scope by (required)"),
    bot_ids: str | None = Query(
        None,
        description=(
            "Comma-separated bot slugs to restrict to. Omit to return the latest "
            "turn for every bot this user has talked to in the window."
        ),
    ),
    since_hours: int = Query(
        168, ge=1, le=168, description="Only consider turns from the recent N hours (max 7d)."
    ),
    preview_chars: int = Query(
        180,
        ge=20,
        le=600,
        description="Trim user_prompt / response_text to this many characters.",
    ),
):
    """Return one "most recent turn" summary per bot in a single round trip.

    Replaces the per-bot dashboard fan-out (one ``/v1/history`` call per bot)
    with a single ``DISTINCT ON (bot_id)`` query. Useful for "who did I talk
    to recently, and what about?" UIs where each tile needs:

      - last activity timestamp + latency
      - last user prompt + assistant response preview
      - tool call count for the turn (incl. in-flight)
      - model used + token usage

    Bots in ``bot_ids`` that have no turn in the window are simply absent
    from the response — callers detect "no activity" by checking which slugs
    are missing from the returned ``turns`` list.
    """
    store = get_turn_log_store()
    if store.engine is None:
        raise HTTPException(status_code=503, detail="Turn logs DB unavailable")

    bot_list: list[str] | None = None
    if bot_ids:
        bot_list = [b.strip() for b in bot_ids.split(",") if b.strip()]
        # Empty after parsing → treat as "no filter" instead of "match nothing".
        if not bot_list:
            bot_list = None

    rows = store.recent_turns_by_bot(
        user_id=user_id,
        bot_ids=bot_list,
        since_hours=since_hours,
    )

    # TASK-364: counts come solely from tool_call_records (canonical store)
    # for every turn — finalized and in-flight alike; the blob is retired.
    live_counts = _live_tool_call_counts(store, [row.id for row in rows])

    items: list[RecentBotTurn] = []
    for row in rows:
        tool_call_count = live_counts.get(row.id, 0)
        user_prompt = (row.user_prompt or "")[:preview_chars] or None
        response_text = row.response_text or ""
        response_preview = response_text[:preview_chars] if response_text else None
        items.append(
            RecentBotTurn(
                bot_id=row.bot_id or "",
                turn_id=row.id,
                created_at=row.created_at,
                model=row.model,
                status=row.status,
                latency_ms=row.latency_ms,
                user_prompt_preview=user_prompt,
                response_preview=response_preview,
                response_tail=_response_tail(response_text),
                response_chars=len(response_text),
                response_preview_truncated=bool(
                    response_text and response_preview and len(response_text) > len(response_preview)
                ),
                tool_call_count=tool_call_count,
                token_usage=_parse_token_usage(getattr(row, "token_usage_json", None)),
                trigger_message_id=getattr(row, "trigger_message_id", None),
            )
        )

    # Sort newest-first so the caller doesn't have to.
    items.sort(key=lambda r: r.created_at, reverse=True)
    return RecentByBotsResponse(turns=items)


@router.get("/v1/bots/{bot_id}/in-turn", tags=["Bots"])
def bot_in_turn(
    bot_id: str,
    within_seconds: int = Query(
        1800, ge=1, le=86400,
        description="Only count in-flight turns started within this window (zombie guard).",
    ),
):
    """Report whether a bot is currently in a turn (path-agnostic).

    ``in_turn`` is true iff the bot has a turn-log row with ``ended_at IS NULL``
    inside the window — the same single source of truth for streaming and
    non-streaming, local and agent-backend turns.
    """
    store = get_turn_log_store()
    if store.engine is None:
        raise HTTPException(status_code=503, detail="Turn logs DB unavailable")

    turn = store.active_turn_for_bot(bot_id, within_seconds=within_seconds)
    if turn is None:
        return {"bot_id": bot_id, "in_turn": False}
    return {
        "bot_id": bot_id,
        "in_turn": True,
        "turn_id": turn.id,
        "status": turn.status,
        "model": turn.model,
        "started_at": turn.created_at.isoformat() if turn.created_at else None,
    }


@router.get("/v1/bots/{bot_id}/latest", tags=["Bots"])
async def bot_latest_activity(
    bot_id: str,
    user_id: str = Query(
        ..., description="User scope — the event stream is per (bot, user)."
    ),
):
    """Latest realtime activity for an AGENT bot, reconstructed from the Redis
    event stream.

    Returns the last user prompt and the streamed assistant text (``full_text``
    plus ``last_bubble`` — the segment since the most recent tool boundary),
    with ``streaming`` telling you whether the turn is still in flight. This
    reflects an in-flight turn AS IT STREAMS — unlike ``/v1/bots/{id}/in-turn``
    (turn-log DB, lands at turn end). Agent bots only: chat bots don't emit
    ``text_delta``, so their assistant text can't be reconstructed here.
    """
    service = get_service()
    subscriber = getattr(service, "_redis_subscriber", None)
    if subscriber is None:
        raise HTTPException(
            status_code=503, detail="Redis event stream unavailable"
        )
    snap = await subscriber.latest_activity(bot_id, user_id)
    if snap is None:
        return {"bot_id": bot_id, "user_id": user_id, "active": False}
    return {"active": True, **snap}


@router.get("/v1/turn-logs/{turn_id}", response_model=TurnLogDetail, tags=["Debug"])
def get_turn_log(turn_id: str):
    """Get one persisted turn log by ID."""
    service = get_service()
    store = get_turn_log_store()
    if store.engine is None:
        raise HTTPException(status_code=503, detail="Turn logs DB unavailable")

    row = store.get_turn(turn_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Turn log not found")

    parsed_request = _parse_json(row.request_json)
    # TASK-364: tool cards render from the canonical tool_call_records rows.
    parsed_tools = _live_tool_calls(store, row.id)

    return TurnLogDetail(
        id=row.id,
        created_at=row.created_at,
        request_id=row.request_id,
        path=row.path,
        stream=row.stream,
        model=row.model,
        bot_id=row.bot_id,
        user_id=row.user_id,
        status=row.status,
        latency_ms=row.latency_ms,
        user_prompt=row.user_prompt,
        request=parsed_request,
        response=row.response_text,
        reasoning=getattr(row, "reasoning", None),
        tool_calls=parsed_tools,
        error_text=row.error_text,
        animation=getattr(row, "animation", None),
        agent_session_key=getattr(row, "agent_session_key", None),
        agent_request_id=getattr(row, "agent_request_id", None),
        token_usage=_parse_token_usage(getattr(row, "token_usage_json", None)),
        end_reason=getattr(row, "end_reason", None),
        question_id=getattr(row, "question_id", None),
        parent_turn_id=getattr(row, "parent_turn_id", None),
        question=_question_for_turn(service, row),
    )


def _question_for_turn(service, row) -> dict | None:
    """Embedded question dict for a turn that ended on one (TASK-269)."""
    if getattr(row, "end_reason", None) != "question":
        return None
    pq_store = getattr(service, "_pending_question_store", None)
    if pq_store is None:
        return None
    qrow = None
    qid = getattr(row, "question_id", None)
    if qid:
        qrow = pq_store.get(qid)
    if qrow is None:
        qrow = pq_store.get_by_turn(row.id)
    return pq_store.row_to_dict(qrow) if qrow is not None else None


@router.get("/v1/tool-calls", response_model=ToolCallEventsResponse, tags=["Debug"])
def get_tool_call_events(
    bot_id: str | None = Query(None, description="Filter by bot ID"),
    user_id: str | None = Query(None, description="Filter by user ID"),
    message_id: str | None = Query(None, description="Single message ID to match"),
    message_ids: list[str] | None = Query(None, description="Message IDs (repeat param or CSV)"),
    after: float | None = Query(None, description="Only include turns created after this unix timestamp"),
    before: float | None = Query(None, description="Only include turns created before this unix timestamp"),
    since_hours: int = Query(168, ge=1, le=168, description="Only include turns from recent N hours (ignored if after/before set)"),
    limit: int = Query(200, ge=1, le=1000, description="Max tool-call events to return"),
):
    """List tool-call events keyed by trigger message ID for history annotation."""
    store = get_turn_log_store()
    if store.engine is None:
        raise HTTPException(status_code=503, detail="Turn logs DB unavailable")

    target_ids = parse_message_filters(message_id, message_ids)

    rows, _ = store.list_turns(
        bot_id=bot_id,
        user_id=user_id,
        # Don't filter by has_tools at the DB level: streaming/in-flight turns
        # have empty tool_calls_json but live rows in tool_call_records.  We
        # filter rows that produce zero events in the loop below.
        has_tools=None,
        trigger_message_ids=target_ids or None,
        after=after,
        before=before,
        since_hours=since_hours,
        limit=min(limit, 1000),
        offset=0,
    )

    # Batch-fetch real per-call timing for every turn in one query so each
    # tool call can carry its source event timestamp (started_at/ended_at)
    # instead of all sharing the turn-level created_at.
    records_by_turn: dict[str, list[ToolCallRecord]] = {}
    turn_ids = [r.id for r in rows if r.id]
    if store.engine is not None and turn_ids:
        try:
            with Session(store.engine) as session:
                rec_rows = list(
                    session.exec(
                        select(ToolCallRecord).where(ToolCallRecord.turn_id.in_(turn_ids))
                    ).all()
                )
            for rec in rec_rows:
                if rec.turn_id:
                    records_by_turn.setdefault(rec.turn_id, []).append(rec)
        except Exception:
            pass

    events: list[ToolCallEvent] = []
    for row in rows:
        # TASK-364: tool cards render straight from the canonical
        # tool_call_records rows (batch-fetched above), which already carry
        # per-call timing — no blob, no separate timing-enrich pass.
        parsed_tools = _records_to_calls(records_by_turn.get(row.id, []), turn_ended_at=row.ended_at)
        if not parsed_tools:
            continue

        trigger_id = row.trigger_message_id
        trigger_role = "user"
        trigger_timestamp: float | None = None
        if not trigger_id:
            parsed_request = _parse_json(row.request_json)
            trigger = extract_trigger_message(parsed_request)
            if trigger is not None:
                trigger_id, trigger_role, trigger_timestamp = trigger
                if target_ids and not message_id_matches(trigger_id, target_ids):
                    continue
            else:
                # Agent-backend turns store trigger_message_id directly.
                # If missing (legacy rows), fall back to the turn's own ID
                # so tool calls are still returned — the frontend maps them
                # via turn_id prefix matching.
                trigger_id = row.id
                trigger_timestamp = row.created_at.timestamp() if row.created_at else None

        events.append(
            ToolCallEvent(
                turn_id=row.id,
                turn_ended_at=row.ended_at,
                created_at=row.created_at,
                request_id=row.request_id,
                model=row.model,
                bot_id=row.bot_id,
                user_id=row.user_id,
                message_id=trigger_id,
                message_role=trigger_role,
                message_timestamp=trigger_timestamp,
                tool_call_count=len(parsed_tools),
                tool_calls=parsed_tools,
            )
        )
        if len(events) >= limit:
            break

    _enrich_call_attachments([call for event in events for call in event.tool_calls])

    events.sort(key=lambda event: event.created_at)
    return ToolCallEventsResponse(
        events=events,
        total_count=len(events),
        filters={
            "bot_id": bot_id,
            "user_id": user_id,
            "after": after,
            "before": before,
            "limit": limit,
        },
    )


@router.get("/v1/tool-calls/{record_id}/result", tags=["Debug"])
def get_tool_call_result(
    record_id: int,
    bot_id: str = Query(...),
    user_id: str = Query(...),
    offset: int = Query(0, ge=0),
    limit: int = Query(65_536, ge=1, le=262_144),
):
    service = get_service()
    store = ToolCallStore(service._turn_log_store.engine)
    page = store.page(
        record_id,
        bot_id=bot_id.strip().lower(),
        user_id=user_id.strip().lower(),
        offset=offset,
        limit=limit,
    )
    if page is None:
        raise HTTPException(status_code=404, detail="Tool result not found")
    return {
        "record_id": page.record_id,
        "content_type": page.content_type,
        "offset": page.offset,
        "content": page.content,
        "total_chars": page.total_chars,
        "total_bytes": page.total_bytes,
        "sha256": page.sha256,
        "complete": page.complete,
        "next_offset": page.next_offset,
    }


@router.get("/v1/tool-calls/{record_id}/result/raw", tags=["Debug"])
def download_tool_call_result(
    record_id: int,
    request: Request,
    bot_id: str = Query(...),
    user_id: str = Query(...),
):
    service = get_service()
    store = ToolCallStore(service._turn_log_store.engine)
    range_header = request.headers.get("range")
    try:
        blob = store.raw_range(
            record_id,
            bot_id=bot_id.strip().lower(),
            user_id=user_id.strip().lower(),
            range_header=range_header,
        )
    except ValueError:
        # Malformed / unsatisfiable Range header.
        raise HTTPException(status_code=416, detail="Requested Range Not Satisfiable")
    if blob is None:
        raise HTTPException(status_code=404, detail="Tool result not found")
    media_type = (
        "application/json"
        if blob.content_type == "application/x-tool-blocks+json"
        else "text/plain"
    )
    extension = "json" if media_type == "application/json" else "txt"
    headers = {
        "Content-Disposition": f'attachment; filename="tool-result-{record_id}.{extension}"',
        "X-Content-Type-Options": "nosniff",
        "X-Tool-Result-SHA256": blob.sha256,
        "Accept-Ranges": "bytes",
    }
    status_code = 200
    if blob.partial:
        headers["Content-Range"] = f"bytes {blob.start}-{blob.end}/{blob.total_bytes}"
        status_code = 206
    return Response(
        content=blob.data,
        media_type=media_type,
        status_code=status_code,
        headers=headers,
    )


# ---------------------------------------------------------------------------
# TASK-661: per-turn changed files
# ---------------------------------------------------------------------------


class CommitChangedFilesRequest(BaseModel):
    session_id: str | None = None
    bot_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    turn_ids: list[str] = Field(min_length=1)


@router.get("/v1/turn-changed-files", tags=["Debug"])
def get_turn_changed_files(
    message_id: str | None = Query(None, description="Single trigger message ID"),
    message_ids: list[str] | None = Query(None, description="Trigger message IDs (repeat or CSV)"),
    turn_id: str | None = Query(None, description="Resolve a single turn directly"),
):
    """Changed-file summaries keyed by trigger message ID for history hydration.

    Mirrors ``/v1/tool-calls``: the chat UI keys per-turn changed files by the
    user message that triggered the turn, so live-stream and hard-refresh render
    the identical summary shape.
    """
    service = get_service()
    store = service._turn_log_store
    if store.engine is None:
        raise HTTPException(status_code=503, detail="Turn logs DB unavailable")
    if turn_id:
        return {"by_message": {}, "turn": store.changed_files_summary(turn_id)}
    target_ids = parse_message_filters(message_id, message_ids)
    by_message = store.changed_files_summaries_for_triggers(list(target_ids)) if target_ids else {}
    return {"by_message": by_message}


@router.get("/v1/turn-changed-files/uncommitted", tags=["History"])
def get_uncommitted_changed_files(
    bot_id: str = Query(...),
    user_id: str = Query(...),
    session_id: str | None = Query(None),
    anchor_turn_id: str | None = Query(None),
):
    """Aggregate unacknowledged changed files inside one canonical session."""
    store = get_turn_log_store()
    if store.engine is None:
        raise HTTPException(status_code=503, detail="Turn logs DB unavailable")
    resolved, summary, other_summary = store.uncommitted_changed_files_summaries(
        session_id=session_id,
        anchor_turn_id=anchor_turn_id,
        bot_id=bot_id.strip().lower(),
        user_id=user_id.strip().lower(),
    )
    if session_id and not resolved:
        raise HTTPException(status_code=404, detail="Conversation not found")
    for scope in (summary, other_summary):
        scope.update(bot_id=bot_id.strip().lower(), user_id=user_id.strip().lower())
    return {
        "session_id": resolved,
        "summary": summary,
        "other_summary": other_summary,
    }


@router.post("/v1/turn-changed-files/commit-requested", tags=["History"])
def mark_changed_files_commit_requested(payload: CommitChangedFilesRequest):
    """Advance the durable uncommitted boundary for selected conversation turns."""
    store = get_turn_log_store()
    if store.engine is None:
        raise HTTPException(status_code=503, detail="Turn logs DB unavailable")
    resolved, marked = store.mark_changed_files_commit_requested(
        session_id=payload.session_id,
        bot_id=payload.bot_id.strip().lower(),
        user_id=payload.user_id.strip().lower(),
        turn_ids=payload.turn_ids,
    )
    if payload.session_id and not resolved:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"session_id": resolved, "marked_files": marked}


class ChangedFileDecisionTarget(BaseModel):
    turn_id: str = Field(min_length=1)
    repo_key: str = Field(min_length=1)
    path: str = Field(min_length=1)
    snapshot_version: str = Field(pattern=r"^[0-9a-f]{64}$")


class ChangedFileDecisionRequest(BaseModel):
    session_id: str | None = None
    bot_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    files: list[ChangedFileDecisionTarget] = Field(min_length=1, max_length=2000)
    ignored: bool


@router.post("/v1/turn-changed-files/decision", tags=["History"])
def set_changed_file_decision(payload: ChangedFileDecisionRequest):
    from ..changed_file_decisions import ChangedFileDecisionConflict, ChangedFileDecisionsStore

    engine = get_turn_log_store().engine
    if engine is None:
        raise HTTPException(status_code=503, detail="Turn logs DB unavailable")
    store = ChangedFileDecisionsStore(engine)
    store.ensure_schema()
    try:
        resolved, marked = store.set_decision(
            session_id=payload.session_id, bot_id=payload.bot_id.strip().lower(),
            user_id=payload.user_id.strip().lower(), ignored=payload.ignored,
            files=[file.model_dump() for file in payload.files],
        )
    except ChangedFileDecisionConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if not resolved:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"session_id": resolved, "marked_files": marked}


@router.get("/v1/turn-changed-files/{turn_id}/content", tags=["Debug"])
def get_turn_changed_file_content(
    turn_id: str,
    repo_key: str = Query(..., description="Workspace identity of the changed file"),
    path: str = Query(..., description="Repo-relative path of the changed file"),
):
    """Persisted before/after bytes for one changed file, as they existed for
    this turn. Returns the FileDiffContent shape the shared viewer consumes.

    This is deliberately NOT the live repos diff endpoint — it serves the bytes
    frozen at turn time, which stay correct after the worktree changes again.
    """
    service = get_service()
    store = service._turn_log_store
    if store.engine is None:
        raise HTTPException(status_code=503, detail="Turn logs DB unavailable")
    resolved = store.changed_file_content(turn_id=turn_id, repo_key=repo_key, path=path)
    if resolved is None:
        raise HTTPException(status_code=404, detail="Changed file not found for turn")
    before, after = resolved
    binary = before.binary or after.binary
    return {
        "oldText": before.text,
        "newText": after.text,
        "binary": binary,
        "truncated": before.truncated or after.truncated,
    }
