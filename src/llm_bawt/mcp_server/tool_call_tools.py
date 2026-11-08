"""Thin trusted-caller adapter for bounded historical tool-call lookup."""
from __future__ import annotations

import asyncio
from functools import lru_cache
from typing import Annotated

from pydantic import Field

from llm_bawt.bot_access import BotAccessError, BotPolicyResolver
from llm_bawt.retrieval.find import failure
from llm_bawt.retrieval.tool_calls import CallRequest, ToolCallLookup
from .find_tools import TOOL_SECONDS, InvalidTime, _caller, invalid_time, parse_find_time
from .registry import mcp


@lru_cache(maxsize=1)
def _runtime():
    from llm_bawt.utils.config import Config
    from llm_bawt.utils.db import get_shared_engine
    engine = get_shared_engine(Config())
    return BotPolicyResolver(engine), ToolCallLookup(engine)


@mcp.tool()
async def tool_calls(
    bot: str | None = None,
    turn_id: str | None = None,
    id: Annotated[int | None, Field(gt=0)] = None,
    since: str | float | None = None,
    until: str | float | None = None,
    tool_name: str | None = None,
    errors_only: bool = False,
    limit: Annotated[int, Field(ge=1, le=25)] = 10,
    before: str | None = None,
    include_result: bool = False,
    max_chars: Annotated[int, Field(ge=1000, le=8000)] = 4000,
) -> str:
    """List readable bots' calls by recent turn, or inspect one id.

    Arguments/results use best-effort secret scrubbing, not a secrecy guarantee.
    Results require id and explicit include_result. List defaults to 24h;
    continue with next plus the same filters. NULL is_error is not an error.
    Turns with no readable trigger message are omitted.
    """
    arguments = dict(bot=bot, turn_id=turn_id, id=id, since=since, until=until,
                     tool_name=tool_name, errors_only=errors_only, limit=limit,
                     before=before, include_result=include_result, max_chars=max_chars)
    try:
        async with asyncio.timeout(TOOL_SECONDS):
            caller, trigger_id = _caller("tool_calls", arguments)
            request = CallRequest(**{**arguments, "since": parse_find_time("since", since),
                                     "until": parse_find_time("until", until)})
            resolver, lookup = _runtime()
            access = await resolver.resolve(caller)
            return await asyncio.to_thread(lookup.lookup, access, request, trigger_id=trigger_id)
    except InvalidTime as error:
        return invalid_time(error.field)
    except BotAccessError as error:
        return failure(error.code)
    except (ValueError, OverflowError):
        return failure("invalid_request")
    except TimeoutError:
        return failure("timeout")
    except Exception:
        return failure("source_unavailable")
