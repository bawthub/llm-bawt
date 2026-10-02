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

import re
from datetime import date as date_cls
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import APIRouter, HTTPException, Query

from ..dependencies import get_service
from ..logging import get_service_logger
from ..schemas_history_timeline import (
    TimelineDay,
    TimelinePrompt,
    TimelinePromptsResponse,
    TimelineResponse,
    TimelineSegment,
    TimelineSession,
)

router = APIRouter()
log = get_service_logger(__name__)

DEFAULT_TZ = "America/New_York"
SNIPPET_CHARS = 120
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
    title: str | None, first_prompt: str | None
) -> tuple[str | None, str]:
    """Label precedence: session title → first non-slash user prompt → none."""
    if title and title.strip():
        return snippet(title), "title"
    if first_prompt and first_prompt.strip() and not is_slash_command(first_prompt):
        return snippet(first_prompt), "first_prompt"
    return None, "none"


def _earliest(a: tuple[float, str], b: tuple[float, str]) -> tuple[float, str]:
    return a if a <= b else b


def build_timeline(
    groups: list[dict],
    session_meta: dict[str, dict],
    first_prompts: dict[str, str],
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
            {"first": first, "last_ts": last_ts, "n": 0, "prompts": 0, "sessions": {}},
        )
        d["first"] = _earliest(d["first"], first)
        d["last_ts"] = max(d["last_ts"], last_ts)
        d["n"] += n
        d["prompts"] += int(g["prompts"])
        if sid:
            # One group row per (day, session), so this is that day's segment.
            d["sessions"][sid] = TimelineSegment(
                session_id=sid,
                first_message_id=first[1],
                first_ts=first[0],
                last_ts=last_ts,
                message_count=n,
                user_prompt_count=int(g["prompts"]),
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
            )
        )

    session_models = []
    for sid, s in sorted(sessions.items(), key=lambda kv: kv[1]["first"]):
        meta = session_meta.get(sid) or {}
        title, source = resolve_session_label(meta.get("title"), first_prompts.get(sid))
        session_models.append(
            TimelineSession(
                id=sid,
                title=title,
                label_source=source,
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


def _load_timeline_rows(
    bot_id: str, session_id: str | None, tz: str
) -> tuple[list[dict], dict[str, dict], dict[str, str]]:
    from sqlalchemy import text

    engine, table = _engine_and_table(bot_id)
    scope = VISIBLE_ROLES_SQL
    params: dict = {"tz": tz}
    if session_id:
        scope += " AND session_id = :session_id"
        params["session_id"] = session_id

    groups_sql = text(
        f"""
        SELECT {LOCAL_DATE_SQL} AS day, session_id,
               count(*) AS n,
               count(*) FILTER (WHERE role = 'user') AS prompts,
               min(timestamp) AS first_ts, max(timestamp) AS last_ts,
               (array_agg(id ORDER BY timestamp, id))[1] AS first_id
        FROM {table}
        WHERE {scope}
        GROUP BY 1, 2
        """
    )
    with engine.connect() as conn:
        groups = [dict(r) for r in conn.execute(groups_sql, params).mappings().all()]
        session_ids = sorted({g["session_id"] for g in groups if g.get("session_id")})
        session_meta: dict[str, dict] = {}
        first_prompts: dict[str, str] = {}
        if session_ids:
            meta_rows = conn.execute(
                text(
                    "SELECT id, status, session_metadata->>'title' AS title "
                    "FROM sessions WHERE id = ANY(:ids)"
                ),
                {"ids": session_ids},
            ).mappings().all()
            session_meta = {str(r["id"]): dict(r) for r in meta_rows}
            untitled = [
                sid for sid in session_ids
                if not ((session_meta.get(sid) or {}).get("title") or "").strip()
            ]
            if untitled:
                prompt_rows = conn.execute(
                    text(
                        f"""
                        SELECT DISTINCT ON (session_id) session_id, content
                        FROM {table}
                        WHERE role = 'user' AND session_id = ANY(:ids)
                          AND ltrim(content) NOT LIKE :slash
                        ORDER BY session_id, timestamp, id
                        """
                    ),
                    {"ids": untitled, "slash": "/%"},
                ).mappings().all()
                first_prompts = {str(r["session_id"]): r["content"] for r in prompt_rows}
    return groups, session_meta, first_prompts


# ── routes ────────────────────────────────────────────────────────────────


@router.get("/v1/history/timeline", response_model=TimelineResponse, tags=["History"])
def get_history_timeline(
    bot_id: str = Query(..., description="Bot ID"),
    session_id: str | None = Query(
        None, description="Scope the index to one thread (thread viewer)."
    ),
    tz: str = Query(DEFAULT_TZ, description="Viewer IANA timezone for day buckets"),
):
    """Compact day + session index powering the chat timeline rail."""
    validate_tz(tz)
    try:
        groups, session_meta, first_prompts = _load_timeline_rows(bot_id, session_id, tz)
        days, sessions, version = build_timeline(groups, session_meta, first_prompts)
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Failed to build history timeline for {bot_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    return TimelineResponse(
        bot_id=bot_id,
        session_id=session_id,
        tz=tz,
        version=version,
        days=days,
        sessions=sessions,
    )


@router.get(
    "/v1/history/timeline/prompts",
    response_model=TimelinePromptsResponse,
    tags=["History"],
)
def get_history_timeline_prompts(
    bot_id: str = Query(..., description="Bot ID"),
    date: str | None = Query(None, description="Local day YYYY-MM-DD in `tz`"),
    session_id: str | None = Query(None, description="Thread to list prompts for"),
    tz: str = Query(DEFAULT_TZ, description="Viewer IANA timezone"),
    limit: int = Query(200, ge=1, le=1000),
):
    """Lazy user-prompt tier: prompts for one day and/or one session."""
    from sqlalchemy import text

    if not date and not session_id:
        raise HTTPException(status_code=400, detail="Provide `date` and/or `session_id`")
    validate_tz(tz)
    if date:
        try:
            date_cls.fromisoformat(date)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid date {date!r}")

    clauses = ["role = 'user'"]
    params: dict = {"tz": tz, "lim": limit + 1}
    if session_id:
        clauses.append("session_id = :session_id")
        params["session_id"] = session_id
    if date:
        clauses.append(f"timestamp >= {LOCAL_DAY_START_SQL}")
        clauses.append(f"timestamp < {LOCAL_DAY_END_SQL}")
        params["day"] = date

    try:
        engine, table = _engine_and_table(bot_id)
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    f"SELECT id, timestamp, session_id, content FROM {table} "
                    f"WHERE {' AND '.join(clauses)} "
                    "ORDER BY timestamp, id LIMIT :lim"
                ),
                params,
            ).mappings().all()
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Failed to load timeline prompts for {bot_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return TimelinePromptsResponse(
        bot_id=bot_id,
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
    )
