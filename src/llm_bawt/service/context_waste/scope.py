"""Who a context-waste report may cover (TASK-1585 DESIGN v2 §4).

Private bots are never in a report, for any caller, including a private bot
asking about itself. Both entry points fail closed with a sanitized
:class:`BotAccessError`; neither returns an empty or partial report on a
policy failure.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from llm_bawt.bot_access import BotAccess, BotAccessError

from .queries import Scope

MAX_DAYS = 90


def _window(days: int, now: datetime | None) -> tuple[datetime, datetime]:
    if type(days) is not int or not 1 <= days <= MAX_DAYS:
        raise BotAccessError("invalid_request")
    until = now or datetime.now(timezone.utc)
    return until - timedelta(days=days), until


def _narrow(bots: set[str], bot: str | None) -> tuple[str, ...]:
    if bot is None:
        return tuple(sorted(bots))
    target = bot.strip().lower() if isinstance(bot, str) else None
    if target not in bots:
        # Unknown, private and out-of-scope are indistinguishable.
        raise BotAccessError("access_denied")
    return (target,)


def operator_scope(access: BotAccess, *, user_id: str, bot: str | None = None,
                   days: int = 7, now: datetime | None = None) -> Scope:
    """BawtHub operator UI: every known bot except private ones, for one user."""
    if not isinstance(user_id, str) or not user_id.strip():
        raise BotAccessError("caller_unavailable")
    since, until = _window(days, now)
    return Scope(user_id, _narrow(set(access.readable_bots), bot), since, until)


def caller_scope(access: BotAccess, *, bot: str | None = None, days: int = 7,
                 now: datetime | None = None) -> Scope:
    """MCP caller: its aggregate scope (searchable bots plus itself), minus private."""
    since, until = _window(days, now)
    aggregate = access.aggregate_scope(require_user=True)
    bots = {part.bot_id for part in aggregate.partitions} - access.private_bots
    return Scope(aggregate.user_id, _narrow(bots, bot), since, until)
