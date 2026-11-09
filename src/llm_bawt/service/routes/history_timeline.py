"""Chat timeline index routes (TASK-980, backend for the TASK-971 rail).

``GET /v1/history/timeline`` returns a compact day + session index for one
bot's continuous timeline (or one thread), built from SQL aggregates on the
bot's messages partition. ``GET /v1/history/timeline/prompts`` is the lazy
user-prompt tier for a single day and/or session.

Timezone rule: day buckets are derived from message epoch timestamps in the
viewer's IANA timezone, inside PostgreSQL (``LOCAL_DATE_SQL``). Never bucket
from ``sessions.started_at`` (naive local) — late-evening sessions would land
on the wrong day. Both endpoints use the same PostgreSQL tz conversion, so a
day's prompts are exactly the rows counted in that day's bucket.

Jump targets (``first_message_id``) are meant for the existing
``/v1/history/around`` deep-link contract; there is no separate loader.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import date as date_cls, datetime, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import APIRouter, HTTPException, Query

from .history_scope import local_clock_bounds, local_day_bounds, owned_message_scope
from .history_timeline_store import SEGMENT_LIMIT, anchor_rows, session_detail_row, timeline_rows
from .history_timeline_summaries import MAX_SUMMARY_EXCERPTS, stored_summary_rows

from ..dependencies import get_service
from ..logging import get_service_logger
from ..http_errors import internal_error
from ..schemas_history_timeline import (
    TimelineAnchor,
    TimelineAnchorsRequest,
    TimelineAnchorsResponse,
    TimelineDay,
    TimelinePrompt,
    TimelinePromptsResponse,
    TimelineResponse,
    TimelineSegment,
    TimelineSession,
    TimelineStoredSummary,
)

router = APIRouter()
log = get_service_logger(__name__)

DEFAULT_TZ = "America/New_York"
SNIPPET_CHARS = 120
INDEX_LABEL_CHARS = 160
VISIBLE_ROLES_SQL = "role NOT IN ('system', 'summary')"
# Single authority for "which local day does this message belong to".
LOCAL_DATE_SQL = "(to_timestamp(timestamp) AT TIME ZONE :tz)::date"
# Local-midnight epoch for a YYYY-MM-DD in :tz; same conversion as above.
LOCAL_DAY_START_SQL = "extract(epoch FROM (CAST(:day AS timestamp) AT TIME ZONE :tz))"
LOCAL_DAY_END_SQL = (
    "extract(epoch FROM ((CAST(:day AS timestamp) + interval '1 day') AT TIME ZONE :tz))"
)

_WS = re.compile(r"\s+")


# ── pure helpers (hermetically tested) ────────────────────────────────────


def validate_tz(tz: str) -> str:
    try:
        ZoneInfo(tz)
    except (ZoneInfoNotFoundError, ValueError):
        raise HTTPException(status_code=400, detail=f"Unknown timezone {tz!r}")
    return tz


def snippet(content: str, limit: int = SNIPPET_CHARS) -> str:
    """Collapse whitespace and cap length for rail labels / detail cards."""
    flat = _WS.sub(" ", content or "").strip()
    return flat if len(flat) <= limit else flat[: limit - 1].rstrip() + "…"


def is_slash_command(content: str | None) -> bool:
    return (content or "").lstrip().startswith("/")


def resolve_session_label(
    title: str | None, first_prompt: str | None, *,
    first_prompt_limit: int = SNIPPET_CHARS,
) -> tuple[str | None, str]:
    """Label precedence: given name → first eligible user message → none.

    ``title`` is a given name only (rename or schedule name), never the stored
    auto copy. A given name is returned whole; the first-message label stays
    a bounded snippet for older clients, which read ``title``. Current
    clients label from the uncut ``first_message_text`` instead.
    """
    if title and title.strip():
        return _WS.sub(" ", title).strip(), "title"
    if first_prompt and first_prompt.strip() and not is_slash_command(first_prompt):
        return snippet(first_prompt, first_prompt_limit), "first_prompt"
    return None, "none"


def _earliest(a: tuple[float, str], b: tuple[float, str]) -> tuple[float, str]:
    return a if a <= b else b


def build_timeline(
    groups: list[dict],
    session_meta: dict[str, dict],
    first_prompts: dict[str, dict],
) -> tuple[list[TimelineDay], list[TimelineSession], str]:
    """Fold ``(day, session_id)`` aggregate rows into days, sessions, version.

    ``groups`` rows carry: day (date or str), session_id (nullable), n,
    prompts, first_ts, last_ts, first_id. Messages without a session count
    toward their day but produce no session entry.
    """
    days: dict[str, dict] = {}
    sessions: dict[str, dict] = {}
    total = 0
    newest = 0.0

    for g in groups:
        day = str(g["day"])
        sid = g.get("session_id")
        n = int(g["n"])
        first = (float(g["first_ts"]), str(g["first_id"]))
        last_ts = float(g["last_ts"])
        total += n
        newest = max(newest, last_ts)

        d = days.setdefault(
            day,
            {"first": first, "last_ts": last_ts, "n": 0, "prompts": 0,
             "summaries": 0, "sessions": {}},
        )
        d["first"] = _earliest(d["first"], first)
        d["last_ts"] = max(d["last_ts"], last_ts)
        d["n"] += n
        d["prompts"] += int(g["prompts"])
        d["summaries"] += int(g.get("summary_count", 0))
        if sid:
            # One group row per (day, session), so this is that day's segment.
            d["sessions"][sid] = TimelineSegment(
                session_id=sid,
                first_message_id=first[1],
                first_ts=first[0],
                last_ts=last_ts,
                message_count=n,
                user_prompt_count=int(g["prompts"]),
                continued=bool(g.get("continued", False)),
            )

            s = sessions.setdefault(sid, {"first": first, "last_ts": last_ts, "n": 0})
            s["first"] = _earliest(s["first"], first)
            s["last_ts"] = max(s["last_ts"], last_ts)
            s["n"] += n

    day_models = []
    for day, d in sorted(days.items()):
        segments = sorted(
            d["sessions"].values(), key=lambda seg: (seg.first_ts, seg.first_message_id)
        )
        day_models.append(
            TimelineDay(
                date=day,
                first_message_id=d["first"][1],
                first_ts=d["first"][0],
                last_ts=d["last_ts"],
                message_count=d["n"],
                user_prompt_count=d["prompts"],
                session_ids=[seg.session_id for seg in segments],
                segments=segments,
                summary_count=d["summaries"],
            )
        )

    session_models = []
    for sid, s in sorted(sessions.items(), key=lambda kv: kv[1]["first"]):
        meta = session_meta.get(sid) or {}
        first = first_prompts.get(sid) or {}
        text = first.get("first_message_text")
        title, source = resolve_session_label(meta.get("title"), text,
                                             first_prompt_limit=INDEX_LABEL_CHARS)
        label_truncated = bool(
            (title and len(title) > INDEX_LABEL_CHARS)
            or (text and len(_WS.sub(" ", text).strip()) > INDEX_LABEL_CHARS))
        session_models.append(
            TimelineSession(
                id=sid,
                title=snippet(title, INDEX_LABEL_CHARS) if title else title,
                label_source=source,
                first_message_text=snippet(text, INDEX_LABEL_CHARS) if text else text,
                initiated_by=first.get("initiated_by"),
                label_truncated=label_truncated,
                first_message_id=s["first"][1],
                first_ts=s["first"][0],
                last_ts=s["last_ts"],
                message_count=s["n"],
                status=meta.get("status"),
            )
        )

    return day_models, session_models, f"{total}:{newest!r}"


# ── SQL access ────────────────────────────────────────────────────────────


def _engine_and_table(bot_id: str):
    from ...media.assets import _build_engine
    from ...memory.postgresql import MESSAGES_PARENT, partition_name

    engine = _build_engine(get_service().config)
    if engine is None:
        raise HTTPException(status_code=503, detail="Memory service unavailable")
    return engine, partition_name(MESSAGES_PARENT, bot_id)


def _load_timeline_rows(bot_id: str, user_id: str, session_id: str | None,
                        tz: str, start: float, end: float, recent: bool,
                        overview_only: bool, overview_before: float | None):
    engine, table = _engine_and_table(bot_id)
    return timeline_rows(engine, table, bot_id, user_id, session_id, tz, start, end,
                         recent=recent, overview_only=overview_only,
                         overview_before=overview_before)


def _load_timeline_session(conn, table: str, bot_id: str, user_id: str,
                           session_id: str, start: float | None,
                           end: float | None) -> TimelineSession | None:
    row = session_detail_row(conn, table, bot_id, user_id, session_id, start, end)
    if row is None:
        return None
    title, source = resolve_session_label(row.get("title"), row.get("first_message_text"),
                                         first_prompt_limit=INDEX_LABEL_CHARS)
    return TimelineSession(**{**row, "title": title, "label_source": source})


def _exclusive_end_date(end: float, tz: str) -> date_cls:
    zone = ZoneInfo(tz)
    day = datetime.fromtimestamp(end, zone).date()
    # Avoid subtracting a float epsilon which may round back to midnight when
    # datetime converts epochs to microseconds.
    midnight = datetime.combine(day, datetime.min.time(), zone).timestamp()
    return day - timedelta(days=1) if end <= midnight else day


def _pinned_timeline_bounds(start: float | None, end: float | None, tz: str,
                            today: date_cls) -> tuple[float, float]:
    """Validate pinned selection without rounding clocks or recomputing dates."""
    if (start is None or end is None or not math.isfinite(start)
            or not math.isfinite(end) or end <= start):
        raise HTTPException(status_code=400, detail="Provide finite UTC bounds with to_utc > from_utc")
    try:
        zone = ZoneInfo(tz)
        first = datetime.fromtimestamp(start, zone).date()
        last = _exclusive_end_date(end, tz)
    except (OverflowError, OSError, ValueError):
        raise HTTPException(status_code=400, detail="Timeline UTC bounds are out of range")
    if (last - first).days >= 90 or last > today:
        raise HTTPException(status_code=400, detail="Timeline scope must span at most 90 local days and not end in the future")
    return start, end


# ── routes ────────────────────────────────────────────────────────────────


@router.get("/v1/history/timeline", response_model=TimelineResponse, tags=["History"])
def get_history_timeline(
    bot_id: str = Query(...),
    user_id: str = Query(..., min_length=1),
    session_id: str | None = Query(None),
    tz: str = Query(DEFAULT_TZ),
    from_date: date_cls | None = Query(None),
    to_date: date_cls | None = Query(None),
    mode: str = Query("recent", pattern="^(recent|custom|all)$"),
    overview_before: date_cls | None = Query(None, description="Older-month cursor for all-history overview"),
    from_time: str | None = Query(None, description="Custom-only HH:MM on from_date in tz; repeated hour uses earlier occurrence"),
    to_time: str | None = Query(None, description="Custom-only exclusive HH:MM on to_date in tz; repeated hour uses later occurrence"),
    before_date: date_cls | None = Query(None, description="Exclusive viewer-local whole-day page cursor; requires pinned UTC scope"),
    from_utc: float | None = Query(None, description="Pinned inclusive selected-scope epoch; requires to_utc"),
    to_utc: float | None = Query(None, description="Pinned exclusive selected-scope epoch; requires from_utc"),
):
    """Owned [start, end) segments; custom clocks default to whole local days.

    Nonexistent DST clocks are invalid. Repeated-hour starts use the earlier
    occurrence, ends the later. `all` returns coarse months until explicit zoom.
    """
    validate_tz(tz)
    today = datetime.now(ZoneInfo(tz)).date()
    pinned = from_utc is not None or to_utc is not None
    if before_date and not pinned:
        raise HTTPException(status_code=400, detail="Older timeline pages require pinned UTC bounds")
    if mode == "all" and (pinned or before_date):
        raise HTTPException(status_code=400, detail="All-history overview does not accept segment page cursors")
    if mode != "custom" and (from_time is not None or to_time is not None):
        raise HTTPException(status_code=400, detail="Timeline clocks require custom mode")
    if mode == "custom" and not pinned and (from_date is None or to_date is None):
        raise HTTPException(status_code=400, detail="Custom range needs both dates")
    if overview_before and (mode != "all" or overview_before.day != 1):
        raise HTTPException(status_code=400, detail="Older-month cursor requires all mode and a month's first day")
    if mode != "all" and to_date and to_date > today:
        raise HTTPException(status_code=400, detail="Timeline cannot end in the future")
    start_date = from_date or today - timedelta(days=6)
    end_date = to_date or today
    if pinned:
        start, end = _pinned_timeline_bounds(from_utc, to_utc, tz, today)
        start_date = datetime.fromtimestamp(start, ZoneInfo(tz)).date()
        end_date = _exclusive_end_date(end, tz)
    else:
        start, end = local_clock_bounds(start_date, end_date, tz, from_time, to_time)
    page_end = end
    if before_date:
        page_end = max(start, min(end, local_day_bounds(before_date, before_date, tz)[0]))
    try:
        groups, session_meta, first_prompts, months, start = _load_timeline_rows(
            bot_id, user_id, session_id, tz, start, page_end,
            not pinned and mode == "recent" and (end_date - start_date).days >= 7,
            mode == "all",
            local_day_bounds(overview_before, overview_before, tz)[0] if overview_before else None,
        )
        days, sessions, version = build_timeline(groups, session_meta, first_prompts)
        has_more = any(bool(g.get("has_more")) for g in groups)
        older_before_date = days[0].date if has_more and days else None
        loaded_start = (max(start, local_day_bounds(
            date_cls.fromisoformat(days[0].date), date_cls.fromisoformat(days[0].date), tz)[0])
                        if has_more and days else start)
        revision = hashlib.sha256(json.dumps([
            start, end, loaded_start, page_end, has_more,
            version, [(s.id, s.title, s.status, s.first_message_text,
                       s.initiated_by, s.label_truncated) for s in sessions],
            [(g['first_id'], g['last_ts'], g['continued']) for g in groups],
            [(str(m['month']), m['messages']) for m in months],
            [(str(g['day']), g['session_id'], g.get('summary_count', 0),
              g.get('summary_revision', '0')) for g in groups],
        ], sort_keys=True).encode()).hexdigest()[:24]
    except HTTPException:
        raise
    except Exception as e:
        log.error("Failed to build owned timeline for %s: %s", bot_id, e)
        raise HTTPException(status_code=500, detail="Timeline unavailable")
    from datetime import timezone
    from_day = datetime.fromtimestamp(start, timezone.utc).astimezone(ZoneInfo(tz)).date().isoformat()
    return TimelineResponse(
        bot_id=bot_id, user_id=user_id, session_id=session_id, tz=tz,
        from_utc=start, to_utc=end, version=revision, days=days, sessions=sessions,
        coverage={
            'from_date': from_day, 'to_date': end_date.isoformat(),
            'mode': mode, 'segments_complete': mode != 'all' and not has_more and page_end == end,
            'loaded_from_utc': loaded_start, 'loaded_to_utc': page_end,
            'has_more': has_more, 'older_before_date': older_before_date,
            'segment_limit': SEGMENT_LIMIT,
            'overview': [{'month': str(m['month']), 'messages': int(m['messages'])}
                         for m in months[:240]],
            'overview_has_older': len(months) > 240,
            'older_month': str(months[239]['month']) if len(months) > 240 else None,
            'legacy_unowned_excluded': True,
        },
    )


@router.post("/v1/history/timeline/anchors", response_model=TimelineAnchorsResponse, tags=["History"])
def get_history_timeline_anchors(body: TimelineAnchorsRequest):
    """One bounded read of canonical message timestamps for exact task turns."""
    try:
        engine, table = _engine_and_table(body.bot_id)
        rows = anchor_rows(engine, table, body.bot_id, body.user_id,
                           body.session_id, body.message_ids)
    except HTTPException:
        raise
    except Exception as e:
        log.error("Failed to load timeline anchors: %s", e)
        raise HTTPException(status_code=500, detail="Timeline anchors unavailable")
    return TimelineAnchorsResponse(anchors=[TimelineAnchor(id=str(r['id']),
        session_id=str(r['session_id']), ts=float(r['ts'])) for r in rows])


@router.get(
    "/v1/history/timeline/prompts",
    response_model=TimelinePromptsResponse,
    tags=["History"],
)
def get_history_timeline_prompts(
    bot_id: str = Query(..., description="Bot ID"),
    user_id: str = Query(..., min_length=1),
    date: str | None = Query(None, description="Local day YYYY-MM-DD in `tz`"),
    session_id: str | None = Query(None, description="Thread to list prompts for"),
    tz: str = Query(DEFAULT_TZ, description="Viewer IANA timezone"),
    limit: int = Query(200, ge=1, le=1000),
    from_utc: float | None = Query(None, description="Inclusive selected-index epoch; requires to_utc"),
    to_utc: float | None = Query(None, description="Exclusive selected-index epoch; requires from_utc"),
    include_summaries: bool = Query(False, description="Read bounded, provenance-checked stored excerpts; never generate"),
):
    """Owned day/thread prompts, intersected with optional UTC [from_utc, to_utc).

    UTC bounds must be a finite ordered pair, and never replace day/thread scope.
    """
    from sqlalchemy import text

    if not date and not session_id:
        raise HTTPException(status_code=400, detail="Provide `date` and/or `session_id`")
    validate_tz(tz)
    if from_utc is not None or to_utc is not None:
        if (from_utc is None or to_utc is None
                or not math.isfinite(from_utc) or not math.isfinite(to_utc)
                or to_utc <= from_utc):
            raise HTTPException(status_code=400, detail="Provide finite UTC bounds with to_utc > from_utc")
    if date:
        try:
            day = date_cls.fromisoformat(date)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid date {date!r}")

    clauses = ["m.role = 'user'", owned_message_scope()]
    params: dict = {"tz": tz, "lim": limit + 1, "user_id": user_id, "bot_id": bot_id}
    if session_id:
        clauses.append("m.session_id = :session_id")
        params["session_id"] = session_id
    if date:
        clauses.extend(["m.timestamp >= :day_start", "m.timestamp < :day_end"])
        params["day_start"], params["day_end"] = local_day_bounds(day, day, tz)
    if from_utc is not None:
        clauses.extend(["m.timestamp >= :from_utc", "m.timestamp < :to_utc"])
        params.update(from_utc=from_utc, to_utc=to_utc)

    summary_start = max([params[k] for k in ("day_start", "from_utc") if k in params], default=None)
    summary_end = min([params[k] for k in ("day_end", "to_utc") if k in params], default=None)
    summaries = []
    session = None
    try:
        engine, table = _engine_and_table(bot_id)
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    f"SELECT m.id, m.timestamp, m.session_id, m.content FROM {table} m "
                    f"WHERE {' AND '.join(clauses)} "
                    "ORDER BY m.timestamp, m.id LIMIT :lim"
                ),
                params,
            ).mappings().all()
            if include_summaries:
                summaries = stored_summary_rows(conn, table, bot_id, user_id, session_id,
                                                summary_start, summary_end, has_day=bool(date))
            if session_id:
                session = _load_timeline_session(conn, table, bot_id, user_id, session_id,
                                                 summary_start, summary_end)
    except HTTPException:
        raise
    except Exception as e:
        raise internal_error(e, f"Failed to load timeline prompts for {bot_id}") from e

    return TimelinePromptsResponse(
        bot_id=bot_id,
        user_id=user_id,
        date=date,
        session_id=session_id,
        tz=tz,
        prompts=[
            TimelinePrompt(
                id=str(r["id"]),
                ts=float(r["timestamp"]),
                session_id=r.get("session_id"),
                snippet=snippet(r["content"]),
            )
            for r in rows[:limit]
        ],
        truncated=len(rows) > limit,
        summaries=[TimelineStoredSummary(**r) for r in summaries[:MAX_SUMMARY_EXCERPTS]],
        summaries_truncated=len(summaries) > MAX_SUMMARY_EXCERPTS,
        session=session,
    )
