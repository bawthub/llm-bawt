"""``messages_find``: thin trusted-caller adapter over :mod:`llm_bawt.retrieval.find`.

Deliberately outside the generic bot-access guard (``SCOPED_RETRIEVAL_TOOLS``):
that guard answers ``access_denied`` for an excluded ``bot`` filter, which
would tell the caller the bot exists. Here access is resolved per call and an
excluded or unknown bot narrows the scope to nothing, so it answers with the
same ``no_match`` as an empty in-scope bot.
"""
from __future__ import annotations

import asyncio
from functools import lru_cache
from typing import Literal

from llm_bawt.bot_access import BotAccessError, BotPolicyResolver
from llm_bawt.retrieval.find import FindRequest, MessageFinder, failure

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


def _timestamp(value: str | float | None) -> float | None:
    from .server import _parse_timestamp
    try:
        return _parse_timestamp(value)
    except ValueError:
        raise BotAccessError("invalid_request") from None


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
    recent messages. ``handle`` alone fetches the next page. Status is
    ok, no_match or unavailable(reason).
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
            request = FindRequest(query, bot, speaker, _timestamp(since), _timestamp(until), handle)
            resolver, finder = _runtime()
            access = await resolver.resolve(caller)
            return await asyncio.to_thread(finder.find, access, request, trigger_id=trigger_id)
    except BotAccessError as error:
        return failure(error.code)
    except TimeoutError:
        return failure("timeout")
    except Exception:
        return failure("source_unavailable")
