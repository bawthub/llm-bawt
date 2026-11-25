"""On/off switch for the long-term memory system (TASK-2653).

``memory_system_enabled`` resolves bot row > global row > default (on). For a
bot where it resolves false: automatic extraction stops, memory MCP tools are
hidden and refused, cold-start memory injection is skipped and context_recall
refuses the ``memories`` source. A bot row can keep one bot on while the
global value is off. Stored rows are kept, so setting it back to true restores
the system unchanged. Working memory and user profile attributes are separate
systems and are not affected.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

SETTING_KEY = "memory_system_enabled"
# Tools that read or write the ``memories`` table, plus extraction into it.
EXTRA_MEMORY_TOOLS = frozenset({"facts_extract", "system_run_maintenance"})
# Test seam only: the unit suite pins this so memory tests run without a DB.
_override: bool | None = None


def is_memory_tool(name: str) -> bool:
    return name.startswith("memory_") or name in EXTRA_MEMORY_TOOLS


def _stored_bool(store: Any, scope_type: str, scope_id: str) -> bool | None:
    """The stored boolean, None when the row is absent; anything else raises."""
    text = store.get_raw_value_json(scope_type, scope_id, SETTING_KEY)
    if text is None:
        return None
    value = json.loads(text)
    if not isinstance(value, bool):
        raise ValueError(f"{SETTING_KEY} must be a JSON boolean")
    return value


def memory_system_enabled(config: Any = None, bot_id: str | None = None) -> bool:
    """Resolve bot row > global row > on. Fails closed: an unavailable store,
    undecodable row or non-boolean value counts as off."""
    if _override is not None:
        return _override
    if config is None:
        from .utils.config import Config
        config = Config()
    try:
        from .service.dependencies import get_runtime_settings_store

        store = get_runtime_settings_store(config)
        slug = (bot_id or "").strip().lower()
        if slug:
            value = _stored_bool(store, "bot", slug)
            if value is not None:
                return value
        value = _stored_bool(store, "global", "*")
        return True if value is None else value
    except Exception:
        logger.warning("memory_system_enabled unreadable; treating memory as disabled", exc_info=True)
        return False


def current_turn_memory_enabled() -> bool:
    """Switch for the MCP caller's bot (signed turn capability), else global."""
    from .mcp_server.task_association import current_task_turn_context

    try:
        bot_id = current_task_turn_context().bot_id
    except Exception:
        bot_id = None
    return memory_system_enabled(bot_id=bot_id)
