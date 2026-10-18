"""Approximate the context loaded by a local Skill call, not its tiny ACK result."""

from __future__ import annotations

import re

from agent_bridge.skill_selection import claude_skill_names, legacy_claude_skill_roots
from agent_bridge.tool_results import count_result_tokens

_MAX_SKILL_BYTES = 1_000_000
_FRONTMATTER = re.compile(r"\A---\r?\n.*?\r?\n---\r?\n", re.DOTALL)


def skill_context_tokens(arguments: dict | None, *, skill_bundle: str | None = None) -> int | None:
    """Estimate the loaded legacy SKILL.md body, never a selected bundle's file.

    The SDK adds framing and referenced files may be loaded later. This is not
    provider billing or a measure of the exact incremental prompt size.
    """
    if skill_bundle is not None:
        return None  # Bundle selectors map to immutable plugin packages, not ambient roots.
    name = (arguments or {}).get("skill")
    if not isinstance(name, str) or name not in claude_skill_names(None):
        return None
    for configured_root in legacy_claude_skill_roots():
        root = configured_root.resolve()
        skill_file = (root / name / "SKILL.md").resolve()
        if not skill_file.is_relative_to(root) or not skill_file.is_file():
            continue
        try:
            if skill_file.stat().st_size > _MAX_SKILL_BYTES:
                return None
            text = skill_file.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        return count_result_tokens(_FRONTMATTER.sub("", text, count=1)) or None
    return None
