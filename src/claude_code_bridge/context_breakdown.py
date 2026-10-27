"""Bounded facts from the CLI's /context response; never store raw tool schemas."""

from __future__ import annotations

import math
from typing import Any


TOP_ITEMS = 10
MAX_NAME_LENGTH = 160
MESSAGE_TOTALS = (
    "toolCallTokens", "toolResultTokens", "attachmentTokens",
    "assistantMessageTokens", "userMessageTokens",
)


def _tokens(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    if not math.isfinite(value) or value < 0:
        return 0
    return int(value)


def _items(value: Any) -> list[dict]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _top(value: Any, *, key: str = "name") -> list[dict]:
    candidates = [
        {"name": item[key][:MAX_NAME_LENGTH], "tokens": _tokens(item.get("tokens"))}
        for item in _items(value)
        if isinstance(item.get(key), str) and item[key]
    ]
    return sorted(candidates, key=lambda item: (-item["tokens"], item["name"]))[:TOP_ITEMS]


def compact_context_breakdown(raw: dict, *, read_ms: float) -> dict | None:
    """Project only stable counts and bounded top names from a native SDK read.

    No paths, descriptions, grids, tool arguments, or whole-message content cross
    this boundary. A malformed/empty SDK response must not claim a breakdown.
    """
    if not isinstance(raw, dict) or not isinstance(raw.get("categories"), list):
        return None
    categories = [
        {
            "name": item["name"][:MAX_NAME_LENGTH],
            "tokens": _tokens(item.get("tokens")),
            "isDeferred": item.get("isDeferred") is True,
        }
        for item in _items(raw["categories"])
        if isinstance(item.get("name"), str) and item["name"]
    ]
    if not categories:
        return None
    skills = raw.get("skills") if isinstance(raw.get("skills"), dict) else {}
    mcp_tools = _items(raw.get("mcpTools"))
    result = {
        "read_ms": round(max(0.0, read_ms), 1),
        "cli_total_tokens": _tokens(raw.get("totalTokens")),
        "categories": categories[:32],
        "skills": {
            "total": _tokens(skills.get("totalSkills")),
            "included": _tokens(skills.get("includedSkills")),
            "tokens": _tokens(skills.get("tokens")),
            "top": _top(skills.get("skillFrontmatter")),
        },
        "mcp_tools": {
            "total": len(mcp_tools),
            "tokens": sum(_tokens(item.get("tokens")) for item in mcp_tools),
            "top": _top(mcp_tools),
        },
    }
    for field, output_key in (
        ("memoryFiles", "memory_files"),
        ("agents", "agents"),
        ("systemPromptSections", "system_prompt_sections"),
        ("systemTools", "system_tools"),
    ):
        if isinstance(raw.get(field), list):
            entries = _items(raw[field])
            result[output_key] = {
                "total": len(entries),
                "tokens": sum(_tokens(item.get("tokens")) for item in entries),
            }
    messages = raw.get("messageBreakdown")
    if isinstance(messages, dict):
        result["message_breakdown"] = {
            key: _tokens(messages.get(key)) for key in MESSAGE_TOTALS
        }
    return result
