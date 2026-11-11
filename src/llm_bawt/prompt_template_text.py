"""Dependency-free prompt parsing shared by runtime and fixture promotion."""

from __future__ import annotations

import re
from typing import Any

_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _scan_template(body: str):
    """Yield text and identifier tokens; preserve literal JSON and legacy escapes."""
    open_kinds: list[str] = []
    text: list[str] = []
    i, n = 0, len(body)
    while i < n:
        ch = body[i]
        if ch == "{":
            if body.startswith("{{", i):
                open_kinds.append("esc")
                text.append("{")
                i += 2
                continue
            close = body.find("}", i + 1)
            if close != -1 and _IDENT_RE.fullmatch(body, i + 1, close):
                if text:
                    yield "text", "".join(text)
                    text = []
                yield "var", body[i + 1:close]
                i = close + 1
                continue
            open_kinds.append("lit")
            text.append("{")
            i += 1
            continue
        if ch == "}":
            top = open_kinds[-1] if open_kinds else None
            if top == "lit":
                open_kinds.pop()
                text.append("}")
                i += 1
                continue
            if body.startswith("}}", i) and top in ("esc", None):
                if top == "esc":
                    open_kinds.pop()
                text.append("}")
                i += 2
                continue
        text.append(ch)
        i += 1
    if text:
        yield "text", "".join(text)


def extract_placeholders(body: str) -> list[str]:
    """Extract distinct ``{identifier}`` placeholders in encounter order."""
    return list(dict.fromkeys(value for kind, value in _scan_template(body or "") if kind == "var"))


def render_template(body: str, variables: dict[str, Any]) -> str:
    """One-pass substitution; literal braces and replacement values stay literal."""
    return "".join(
        str(variables[value]) if kind == "var" else value
        for kind, value in _scan_template(body or "")
    )
