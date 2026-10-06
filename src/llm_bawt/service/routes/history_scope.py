"""Owner and range rules shared by timeline, prompt, anchor and jump reads.

Messages carry no user_id. A message is owned only if its session exists and
that session has the exact (bot_id, non-null user_id). Legacy null/orphan rows
are not assigned to a viewer by guesswork; they remain in legacy history until
an explicit ownership migration. Never use message author as session ownership.
"""

from datetime import date, datetime, timedelta, timezone
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
