"""Best-effort history-copy hygiene, not an authorization or secret guarantee."""
from __future__ import annotations

import json
import re
from typing import Any

MASK = "[redacted]"
_SECRET_KEY = re.compile(
    r"token|secret|password|passwd|api[_-]?key|authorization|cookie|credential", re.I
)
_TOKENS = re.compile(
    r"(?i)\bBearer\s+[^\s'\";,]+"
    r"|\b(?:sk-[a-z0-9_-]{8,}|ghp_[a-z0-9_]+|github_pat_[a-z0-9_]+|xox[baprs]-[a-z0-9_-]+)"
    r"|\beyJ[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+"
)
_AUTH_HEADER = re.compile(
    r"(?i)(\bauthorization[ \t]*[:=][ \t]*)"
    r"(?:[a-z][\w-]*[ \t]+)?[^\s'\";,]+"
)
_SECRET_FLAG = re.compile(
    r"""(?ix)
    (?<![\w-])(--[\w-]{0,80}(?:token|secret|password|passwd|api[_-]?key|credential)[\w-]{0,80})
    ([ \t]+)("(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|[^\s;,&]+)
    """
)
_URL_USERINFO = re.compile(
    r"""(?i)(\b[a-z][a-z0-9+.-]{1,30}://)[^\s/@'"<>]+@"""
)
_ASSIGNMENT = re.compile(
    r"""(?ix)
    (?<![\w.-])(?P<key>["']?[\w.-]{0,128}(?:token|secret|password|passwd|api[_-]?key|authorization|cookie|credential)[\w.-]{0,128}["']?)
    (?P<sep>\s*(?:=|:)\s*)
    (?P<value>\[redacted\]|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|[^\s,;&}\]]+)
    """
)


def scrub_text(value: str) -> str:
    value = _AUTH_HEADER.sub(lambda m: m[1] + MASK, value)
    value = _SECRET_FLAG.sub(lambda m: m[1] + m[2] + MASK, value)
    value = _URL_USERINFO.sub(lambda m: m[1] + MASK + "@", value)
    value = _ASSIGNMENT.sub(lambda m: m["key"] + m["sep"] + MASK, value)
    return _TOKENS.sub(MASK, value)


def scrub(value: Any, *, field_chars: int, schema_json: str | None = None,
          _truncated: list[bool] | None = None) -> Any:
    """Scrub before clipping. Schema masking supplements the heuristic patterns."""
    if schema_json is not None:
        from llm_bawt.ops.service import display_args
        value = display_args(value, schema_json)
    remaining = 100
    truncated = _truncated if _truncated is not None else [False]

    def bounded(value, size):
        truncated[0] |= len(value) > size
        return clip(value, size)

    def visit(item, depth=0):
        nonlocal remaining
        remaining -= 1
        if remaining < 0 or depth > 8:
            truncated[0] = True
            return "[truncated]"
        if isinstance(item, dict):
            result = {}
            for key, val in list(item.items())[:30]:
                key = str(key)
                result[bounded(scrub_text(key), 80)] = (
                    MASK if _SECRET_KEY.search(key) else visit(val, depth + 1)
                )
            if len(item) > 30:
                truncated[0] = True
                result["…"] = "[truncated]"
            return result
        if isinstance(item, list):
            truncated[0] |= len(item) > 30
            result = [visit(v, depth + 1) for v in item[:30]]
            return result + (["[truncated]"] if len(item) > 30 else [])
        if isinstance(item, str):
            return bounded(scrub_text(item), field_chars)
        if item is None or isinstance(item, (bool, int, float)):
            return item
        return bounded(scrub_text(str(item)), field_chars)

    return visit(value)


def clip(value: str, chars: int) -> str:
    return value if len(value) <= chars else value[:max(0, chars - 1)] + "…"


def arguments_view(raw: str | None, *, field_chars: int, max_chars: int,
                   schema_json: str | None = None) -> dict:
    """Keep JSON structure where possible, but never return the unsanitized fallback."""
    try:
        value = json.loads(raw) if raw is not None else None
    except (ValueError, RecursionError):
        value = raw
    truncated = [False]
    cleaned = scrub(value, field_chars=field_chars, schema_json=schema_json, _truncated=truncated)
    rendered = json.dumps(cleaned, ensure_ascii=False, separators=(",", ":"))
    if len(rendered) > max_chars:
        return {"text": clip(rendered, max_chars), "truncated": True}
    return {"value": cleaned, "truncated": truncated[0]}
