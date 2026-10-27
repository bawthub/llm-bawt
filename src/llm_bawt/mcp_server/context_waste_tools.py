"""``context_waste_report``: the recurring context-waste digest (TASK-1585).

Resolves BotAccess per call: the caller's aggregate scope (searchable bots
plus itself) minus every private bot, the caller included, and only the
caller's own user. A ``bot`` filter only narrows that set; outside it the
answer is the same ``access_denied`` whether the bot is private, excluded
or unknown. Any policy failure is a sanitized ``unavailable``, never an
empty or partial report.
"""
from __future__ import annotations

import asyncio
from functools import lru_cache

from llm_bawt.bot_access import BotAccessError, BotPolicyResolver
from llm_bawt.retrieval.find import failure, serialize
from llm_bawt.service.context_waste import build_report, caller_scope
from llm_bawt.service.context_waste.registry_keys import declared_mcp_keys
from llm_bawt.service.context_waste.report import digest

from .bot_access_identity import trusted_caller
from .registry import mcp

TOOL_SECONDS = 20


@lru_cache(maxsize=1)
def _runtime():
    from llm_bawt.utils.config import Config
    from llm_bawt.utils.db import get_shared_engine
    config = Config()
    return get_shared_engine(config), config


def _thresholds(config) -> dict:
    from llm_bawt.service.context_waste import SETTING_KEY, resolve_thresholds
    from llm_bawt.service.dependencies import get_runtime_settings_store
    try:
        settings = get_runtime_settings_store(config).get_scope_settings("global", "*")
    except Exception:
        settings = {}
    return resolve_thresholds(settings.get(SETTING_KEY))


def _report(engine, config, access, bot, days, mcp_keys=None) -> str:
    scope = caller_scope(access, bot=bot, days=days)
    return serialize(digest(build_report(engine, scope, _thresholds(config), mcp_keys)))


@mcp.tool(name="context_waste_report")
async def context_waste_report(bot: str | None = None, days: int = 7) -> str:
    """Recurring context waste across bots you may read, private bots excluded.

    Repeated identical tool output, oversized results with the cheaper call,
    and the tool-result share of transcripts. Names, counts, sizes and hashes
    only. ``days`` is 1-90. Status is ok or unavailable(reason).
    """
    try:
        async with asyncio.timeout(TOOL_SECONDS):
            caller = trusted_caller("context_waste_report", {"bot": bot, "days": days})
            if caller is None:
                raise BotAccessError("caller_unavailable")
            engine, config = _runtime()
            access = await BotPolicyResolver(engine).resolve(caller)
            mcp_keys = await declared_mcp_keys()
            return await asyncio.to_thread(_report, engine, config, access, bot, days, mcp_keys)
    except BotAccessError as error:
        return failure(error.code)
    except TimeoutError:
        return failure("timeout")
    except Exception:
        return failure("source_unavailable")
