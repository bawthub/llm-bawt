"""``messages_find``: thin trusted-caller adapter over :mod:`llm_bawt.retrieval.find`.

Deliberately outside the generic bot-access guard (``SCOPED_RETRIEVAL_TOOLS``):
that guard answers ``access_denied`` for an excluded ``bot`` filter, which
would tell the caller the bot exists. Here access is resolved per call and an
excluded or unknown bot narrows the scope to nothing, so it answers with the
same ``no_match`` as an empty in-scope bot.
"""
from __future__ import annotations

import asyncio
import math
import re
import time
from functools import lru_cache
from typing import Literal

from llm_bawt.bot_access import BotAccessError, BotPolicyResolver
from llm_bawt.retrieval.find import FindRequest, MessageFinder, failure, serialize

from .bot_access_identity import trusted_caller
from .registry import mcp
from .task_association import current_task_turn_context

TOOL_SECONDS = 8


@lru_cache(maxsize=1)
def _runtime() -> tuple[BotPolicyResolver, MessageFinder]:
    from llm_bawt.utils.config import Config
    from llm_bawt.utils.db import get_shared_engine
    engine = get_shared_engine(Config())
    return BotPolicyResolver(engine), MessageFinder(engine)


#: What ``since``/``until`` accept, reported back verbatim on a bad value.
ACCEPTED_TIME_FORMATS = (
    "relative: Nh or Nd back from now (1h-8760h, 1d-365d), e.g. 24h, 7d",
    "ISO date or datetime: 2026-06-01, 2026-06-01T14:30:00, 2026-06-01T14:30:00-04:00",
    "Unix seconds: 1782857172",
)
_RELATIVE = re.compile(r"(\d{1,4})([hd])", re.IGNORECASE)
_RELATIVE_MAX = {"h": 8760, "d": 365}
_UNIT_SECONDS = {"h": 3600, "d": 86400}


class InvalidTime(ValueError):
    """A ``since``/``until`` value none of :data:`ACCEPTED_TIME_FORMATS` covers."""

    def __init__(self, field: str):
        self.field = field
        super().__init__(field)


def parse_find_time(field: str, value: str | float | None, *, now: float | None = None) -> float | None:
    """Unix seconds for a ``since``/``until`` value, or ``None`` when omitted.

    Relative ``Nh``/``Nd`` counts back from ``now``; out-of-range counts and
    anything unparseable raise :class:`InvalidTime`.
    """
    if isinstance(value, str):
        match = _RELATIVE.fullmatch(value.strip())
        if match:
            count, unit = int(match.group(1)), match.group(2).lower()
            if not 1 <= count <= _RELATIVE_MAX[unit]:
                raise InvalidTime(field)
            return (time.time() if now is None else now) - count * _UNIT_SECONDS[unit]
    if value is None:
        return None
    from .server import _parse_timestamp
    try:
        parsed = _parse_timestamp(value)
    except (ValueError, OverflowError):
        raise InvalidTime(field) from None
    # A value was given: blank, an unsupported type or a NaN/inf bound would
    # silently widen the search, so each is a bad request, not "omitted".
    if parsed is None or not math.isfinite(parsed):
        raise InvalidTime(field)
    return parsed


def invalid_time(field: str) -> str:
    """The ``unavailable`` answer for a bad time value, naming what is accepted."""
    return serialize({"status": "unavailable", "reason": "invalid_request", "field": field,
                      "accepted": list(ACCEPTED_TIME_FORMATS)})


@mcp.tool(name="messages_find")
async def messages_find(
    query: str | None = None,
    bot: str | None = None,
    speaker: Literal["user", "bot", "delivery"] | None = None,
    since: str | float | None = None,
    until: str | float | None = None,
    handle: str | None = None,
) -> str:
    """Locate messages across every bot you may read; returns compact cards, not transcripts.

    Best matches plus the latest matches, session-collapsed, with facets
    (bot/age/speaker) for narrowing. Identifiers (TASK-N, SHAs, paths,
    snake_case) match exactly; words match by stem. Omit query to browse
    recent messages. ``handle`` alone fetches the next page. ``since`` and
    ``until`` take 24h/7d (back from now), an ISO date/datetime or Unix
    seconds. Status is ok, no_match or unavailable(reason).
    """
    arguments = {"query": query, "bot": bot, "speaker": speaker,
                 "since": since, "until": until, "handle": handle}
    try:
        async with asyncio.timeout(TOOL_SECONDS):
            caller = trusted_caller("messages_find", arguments)
            if caller is None:
                return failure("caller_unavailable")
            trigger_id = None
            if caller.kind == "turn":
                try:
                    turn = current_task_turn_context()
                except Exception:
                    return failure("caller_unavailable")
                if (turn.bot_id, turn.user_id) != (caller.bot_id, caller.user_id):
                    return failure("caller_unavailable")
                trigger_id = turn.trigger_message_id
            request = FindRequest(query, bot, speaker, parse_find_time("since", since),
                                  parse_find_time("until", until), handle)
            resolver, finder = _runtime()
            access = await resolver.resolve(caller)
            return await asyncio.to_thread(finder.find, access, request, trigger_id=trigger_id)
    except InvalidTime as error:
        return invalid_time(error.field)
    except BotAccessError as error:
        return failure(error.code)
    except TimeoutError:
        return failure("timeout")
    except Exception:
        return failure("source_unavailable")
