"""The Skill selectors one proxied request may call (TASK-1550).

A managed selection (TASK-1309) is authoritative: the bridge's catalog is used
as-is. An unmanaged (ambient) run only knows the skill directories on disk, but
the Claude CLI also advertises skills it ships itself (bundled skills) and
plugin skills. For ambient runs the callable set is therefore the bridge catalog
plus whatever the request itself advertises: an enum already on the SDK's
Skill schema and the names in the CLI's skill listing.
"""
from __future__ import annotations

import re
from typing import Any

#: Value of :data:`SKILL_SCOPE_HEADER` for an unmanaged run.
AMBIENT_SCOPE = "ambient"
SKILL_SCOPE_HEADER = "X-LLM-Bawt-Skill-Scope"

_NAME = r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}"
_NAME_RE = re.compile(rf"^{_NAME}$")
_LISTING_MARKER = "available for use with the Skill tool"
_ITEM_RE = re.compile(rf"^- ({_NAME})(?::\s|:?$)")
_MAX_NAMES = 256


def parse_skill_scope(value: str | None) -> bool:
    """True only for an explicit ambient scope; absent keeps the strict catalog."""
    return (value or "").strip().lower() == AMBIENT_SCOPE


def _texts(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from _texts(item)
    elif isinstance(value, dict):
        if value.get("type") in (None, "text") and isinstance(value.get("text"), str):
            yield value["text"]
        content = value.get("content")
        if content is not None and value.get("type") != "tool_result":
            yield from _texts(content)


def _listed_names(text: str) -> list[str]:
    names: list[str] = []
    for block in text.split(_LISTING_MARKER)[1:]:
        started = False
        for line in block.splitlines()[1:]:
            match = _ITEM_RE.match(line.strip())
            if match:
                names.append(match.group(1))
                started = True
            elif started and line.strip():
                break
    return names


def advertised_skill_names(body: dict) -> list[str]:
    """Names the request advertises: the Skill schema enum, then CLI listings."""
    names: list[str] = []
    for tool in body.get("tools") or []:
        if not isinstance(tool, dict) or tool.get("name") != "Skill":
            continue
        params = tool.get("input_schema") or tool.get("parameters")
        skill = (params or {}).get("properties", {}).get("skill") if isinstance(params, dict) else None
        enum = skill.get("enum") if isinstance(skill, dict) else None
        if isinstance(enum, list):
            names.extend(value for value in enum if isinstance(value, str))
    for source in (body.get("system"), body.get("messages")):
        for text in _texts(source):
            if _LISTING_MARKER in text:
                names.extend(_listed_names(text))
    return names


def callable_skill_names(
    body: dict, catalog: tuple[str, ...], *, ambient: bool,
    disabled: tuple[str, ...] = (),
) -> tuple[str, ...]:
    """The Skill enum for this request.

    An allow-list catalog is never widened. Ambient and deny-list runs add the
    advertised skills. Explicit disables are removed in every case.
    """
    blocked = set(disabled)
    if not ambient:
        return tuple(n for n in catalog if n not in blocked)
    merged = [*catalog, *advertised_skill_names(body)]
    return tuple(dict.fromkeys(
        n for n in merged if _NAME_RE.fullmatch(n) and n not in blocked
    ))[:_MAX_NAMES]
