"""Owner and range rules shared by timeline, prompt, anchor and jump reads.

Messages carry no user_id. A message is owned only if its session exists and
that session has the exact (bot_id, non-null user_id). Legacy null/orphan rows
are not assigned to a viewer by guesswork; they remain in legacy history until
an explicit ownership migration. Never use message author as session ownership.
"""

import re
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

from fastapi import HTTPException

MAX_TIMELINE_DAYS = 90
MAX_TIMELINE_ANCHORS = 2000


def owned_message_scope(alias: str = "m") -> str:
    """Correlated predicate; caller binds :bot_id and :user_id."""
    if alias != "m":
        raise ValueError("Only fixed SQL alias 'm' is supported")
    return (
        "EXISTS (SELECT 1 FROM sessions owner "
        "WHERE owner.id = m.session_id AND owner.bot_id = :bot_id "
        "AND owner.user_id = :user_id)"
    )


def local_day_bounds(from_date: date, to_date: date, tz: str) -> tuple[float, float]:
    """Inclusive local dates -> exclusive UTC epoch endpoints; each handles DST."""
    if to_date < from_date or (to_date - from_date).days >= MAX_TIMELINE_DAYS:
        raise HTTPException(status_code=400, detail="Timeline range must span 1 to 90 local days")
    zone = ZoneInfo(tz)
    start = datetime.combine(from_date, datetime.min.time(), zone)
    end = datetime.combine(to_date + timedelta(days=1), datetime.min.time(), zone)
    return start.astimezone(timezone.utc).timestamp(), end.astimezone(timezone.utc).timestamp()


def local_clock_bounds(
    from_date: date, to_date: date, tz: str,
    from_time: str | None = None, to_time: str | None = None,
) -> tuple[float, float]:
    """Inclusive dates with optional HH:MM endpoints -> UTC [start, end).

    Omitted clocks preserve whole-day bounds. An explicit end clock is on
    to_date, not the following day. Nonexistent DST wall times are rejected;
    repeated hours use the earlier occurrence for start and later for end.
    """
    start, end = local_day_bounds(from_date, to_date, tz)
    zone = ZoneInfo(tz)

    def clock_epoch(day: date, clock: str, *, is_end: bool) -> float:
        if not re.fullmatch(r"(?:[01][0-9]|2[0-3]):[0-5][0-9]", clock):
            raise HTTPException(status_code=400, detail="Timeline clocks must use HH:MM (00:00 to 23:59)")
        wall = datetime.combine(day, time.fromisoformat(clock))
        # ZoneInfo permits imaginary times: only accept folds that round-trip.
        candidates = []
        for fold in (0, 1):
            aware = wall.replace(tzinfo=zone, fold=fold)
            utc = aware.astimezone(timezone.utc)
            if utc.astimezone(zone).replace(tzinfo=None) == wall:
                candidates.append(utc.timestamp())
        if not candidates:
            raise HTTPException(status_code=400, detail="Timeline clock does not exist in viewer timezone")
        return max(candidates) if is_end else min(candidates)

    if from_time is not None:
        start = clock_epoch(from_date, from_time, is_end=False)
    if to_time is not None:
        end = clock_epoch(to_date, to_time, is_end=True)
    if end <= start:
        raise HTTPException(status_code=400, detail="Timeline end must be after start")
    return start, end
