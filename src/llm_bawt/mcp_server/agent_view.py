"""What agent-facing MCP message tools return, versus what storage holds (TASK-1320).

Message rows carry an ``author`` presentation envelope for the UI: name,
color, avatar URL and ``avatar_render``, a self-hosted data: URL of the avatar
image. That render is kilobytes of base64 that tells an agent nothing, and
it was being returned for every row of ``messages_search_all``,
``messages_get`` and ``messages_get_by_id``. Those tools drop it here, at the
MCP boundary, for agent callers only.

The app's own service client also reads through these tools: REST
``/v1/history`` and ``/v1/memory/message`` go MemoryClient → MCP
``messages_get``/``messages_get_by_id`` and hand the author envelope to the
UI. A verified ``service`` caller therefore gets the full envelope; agent
turns (``turn``/``approved``) and unidentified callers get the stripped one.
"""

from __future__ import annotations

from typing import Any

from llm_bawt.bot_access.context import current_bot_access

#: Author-envelope keys agents never need (display data only).
AGENT_HIDDEN_AUTHOR_KEYS = frozenset({"avatar_render"})


def for_caller(value: Any) -> Any:
    """``value`` as the current MCP caller should see it.

    The app's service client (REST/UI) keeps display data; everyone else gets
    :func:`for_agent`.
    """
    access = current_bot_access()
    if access is not None and access.caller is not None and access.caller.kind == "service":
        return value
    return for_agent(value)


def for_agent(value: Any) -> Any:
    """``value`` with every ``author`` envelope stripped of display-only keys.

    Walks dicts and lists (a single message, a list of messages, or the
    ``{message, before, after}`` context shape). Returns new containers and
    never mutates the input; other values are returned as-is.
    """
    if isinstance(value, list):
        return [for_agent(item) for item in value]
    if isinstance(value, dict):
        out: dict[Any, Any] = {}
        for key, item in value.items():
            if key == "author" and isinstance(item, dict):
                out[key] = {k: v for k, v in item.items() if k not in AGENT_HIDDEN_AUTHOR_KEYS}
            else:
                out[key] = for_agent(item)
        return out
    return value
