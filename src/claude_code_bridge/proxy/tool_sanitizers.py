"""Tool-specific argument sanitizer pipeline for proxy stream translators.

Reasoning models (GPT-5.x family) leak chain-of-thought tokens into tool call
argument values.  The JSON envelope is structurally valid, but string field
contents are corrupted -- or the entire argument string has trailing garbage
after a complete JSON object.

This module provides:

1. A composable registry of per-tool sanitizers applied to parsed argument dicts.
2. A JS-code trailing-token stripper for code-execution tools
   (browser_run_code_unsafe, browser_evaluate).
3. A JSON trailing-text recovery function for the json.loads failure path.

Integration: both stream.py and stream_cc.py call through this shared pipeline.
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JS scanning helpers
# ---------------------------------------------------------------------------
# These handle the JS subset needed to track brace depth correctly:
# strings (single/double), template literals (with nested ${} expressions),
# and comments (single-line // and multi-line /* */).
#
# Regex literals are NOT tracked -- a regex containing unbalanced braces
# (e.g. /}/) would throw off the depth counter.  This is acceptable:
# unbalanced-brace regex is vanishingly rare in Playwright automation code,
# and a false negative (not stripping) is always safe.
# ---------------------------------------------------------------------------


def _skip_js_string(code: str, start: int) -> int:
    """Skip past a single- or double-quoted string.

    ``start`` must point at the opening quote.
    Returns index one past the closing quote, or ``len(code)`` if unterminated.
    """
    quote = code[start]
    i = start + 1
    n = len(code)
    while i < n:
        if code[i] == "\\":
            i += 2
            continue
        if code[i] == quote:
            return i + 1
        i += 1
    return n


def _skip_js_template_expr(code: str, start: int) -> int:
    """Skip a ``${...}`` expression inside a template literal.

    ``start`` is one past the opening ``{`` of ``${``.
    Handles nested braces, strings, comments, and nested template literals.
    Returns index one past the closing ``}``.
    """
    i = start
    n = len(code)
    depth = 1
    while i < n and depth > 0:
        c = code[i]
        # Comments
        if c == "/" and i + 1 < n:
            if code[i + 1] == "/":
                nl = code.find("\n", i + 2)
                i = nl + 1 if nl != -1 else n
                continue
            if code[i + 1] == "*":
                end = code.find("*/", i + 2)
                i = end + 2 if end != -1 else n
                continue
        # Strings
        if c in ("'", '"'):
            i = _skip_js_string(code, i)
            continue
        # Nested template
        if c == "`":
            i = _skip_js_template(code, i)
            continue
        # Braces
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return n


def _skip_js_template(code: str, start: int) -> int:
    """Skip past a template literal (backtick string).

    ``start`` must point at the opening backtick.
    Handles ``${...}`` interpolations with nested expressions.
    Returns index one past the closing backtick, or ``len(code)``.
    """
    i = start + 1
    n = len(code)
    while i < n:
        c = code[i]
        if c == "\\":
            i += 2
            continue
        if c == "`":
            return i + 1
        if c == "$" and i + 1 < n and code[i + 1] == "{":
            i = _skip_js_template_expr(code, i + 2)
            continue
        i += 1
    return n


def _scan_js_balanced_end(code: str) -> int | None:
    """Find where the first outermost brace pair closes.

    Scans ``code`` respecting JS strings, template literals, and comments.
    Returns the index one past the closing ``}``, or ``None`` if no balanced
    point is found (no braces, unterminated, or unmatched).
    """
    i = 0
    n = len(code)
    depth = 0

    while i < n:
        c = code[i]

        # Comments
        if c == "/" and i + 1 < n:
            if code[i + 1] == "/":
                nl = code.find("\n", i + 2)
                i = nl + 1 if nl != -1 else n
                continue
            if code[i + 1] == "*":
                end = code.find("*/", i + 2)
                i = end + 2 if end != -1 else n
                continue

        # String literals
        if c in ("'", '"'):
            i = _skip_js_string(code, i)
            continue

        # Template literals
        if c == "`":
            i = _skip_js_template(code, i)
            continue

        # Braces
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i + 1
            if depth < 0:
                return None  # Unmatched } -- bail

        i += 1

    return None


# ---------------------------------------------------------------------------
# Built-in sanitizers
# ---------------------------------------------------------------------------

# Tool name suffix -> parameter names that carry JS code.
_JS_CODE_PARAMS: dict[str, list[str]] = {
    "browser_run_code_unsafe": ["code"],
    "browser_evaluate": ["expression"],
}


def strip_js_trailing_tokens(
    args: dict[str, Any], tool_name: str
) -> dict[str, Any]:
    """Strip trailing non-JS tokens from code parameters.

    When a reasoning model leaks chain-of-thought after a function expression's
    closing brace, this sanitizer finds the balanced end and truncates.  Only
    modifies the value when there is substantive non-JS content after the first
    balanced brace pair.
    """
    # Which params to clean based on tool name suffix.
    params_to_clean: list[str] = []
    for suffix, params in _JS_CODE_PARAMS.items():
        if tool_name == suffix or tool_name.endswith(f"__{suffix}"):
            params_to_clean = params
            break

    if not params_to_clean:
        return args

    modified = False
    result = dict(args)

    for param in params_to_clean:
        value = result.get(param)
        if not isinstance(value, str):
            continue

        end_pos = _scan_js_balanced_end(value)
        if end_pos is None:
            continue  # No braces to balance -- leave as-is

        trailing = value[end_pos:]
        trailing_stripped = trailing.strip()

        # Nothing after the closing brace (or just whitespace/semicolons).
        if not trailing_stripped or trailing_stripped == ";":
            continue

        # If the first non-whitespace char after } is a JS continuation
        # operator (method chaining, bracket access, call, or closing a
        # wrapper paren/bracket), this is likely valid code.  Leave it alone.
        if trailing_stripped[0] in ".([])":
            continue

        cleaned = value[:end_pos]
        logger.warning(
            "Sanitizer stripped trailing tokens from %s.%s: "
            "kept ...%r, removed %r (%d chars)",
            tool_name,
            param,
            value[max(0, end_pos - 30) : end_pos],
            trailing[:60] + ("..." if len(trailing) > 60 else ""),
            len(trailing),
        )
        result[param] = cleaned
        modified = True

    return result


# ---------------------------------------------------------------------------
# JSON trailing-text recovery
# ---------------------------------------------------------------------------


def recover_trailing_json(raw: str) -> str | None:
    """Extract valid JSON from a string with trailing leaked reasoning.

    When the model appends chain-of-thought text after a complete JSON object or
    array, ``json.loads`` fails on the whole string.  This function finds the end
    of the first complete JSON value and returns just that prefix, or ``None`` if
    recovery is not possible.

    Only attempts recovery when ``raw`` starts with ``{`` or ``[``.
    """
    stripped = raw.strip()
    if not stripped or stripped[0] not in ("{", "["):
        return None

    depth = 0
    i = 0
    n = len(stripped)

    while i < n:
        c = stripped[i]
        # JSON strings are always double-quoted.
        if c == '"':
            i += 1
            while i < n:
                if stripped[i] == "\\":
                    i += 2
                    continue
                if stripped[i] == '"':
                    i += 1
                    break
                i += 1
            continue
        if c in ("{", "["):
            depth += 1
        elif c in ("}", "]"):
            depth -= 1
            if depth == 0:
                candidate = stripped[: i + 1]
                try:
                    json.loads(candidate)
                except json.JSONDecodeError:
                    i += 1
                    continue  # Structural mismatch -- keep scanning
                remainder = stripped[i + 1 :].strip()
                if remainder:
                    logger.warning(
                        "Recovered JSON with %d chars of trailing text removed",
                        len(remainder),
                    )
                return candidate
        i += 1

    return None


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ToolSanitizer = Callable[[dict[str, Any], str], dict[str, Any]]

# List of (matcher, sanitizer) pairs.  Applied in registration order.
_REGISTRY: list[tuple[Callable[[str], bool], ToolSanitizer]] = []


def register(
    matcher: str | Callable[[str], bool],
    sanitizer: ToolSanitizer,
) -> None:
    """Register a sanitizer for tools whose names match ``matcher``.

    ``matcher`` can be:

    - A string: matched as a suffix against the tool name (handles
      MCP-prefixed names like ``mcp__playwright__browser_run_code_unsafe``).
    - A callable: receives the tool name, returns ``True`` to apply.

    Multiple sanitizers can match one tool -- they compose in registration order.
    """
    if isinstance(matcher, str):
        suffix = matcher

        def _suffix_match(name: str, _s: str = suffix) -> bool:
            return name == _s or name.endswith(f"__{_s}")

        _REGISTRY.append((_suffix_match, sanitizer))
    else:
        _REGISTRY.append((matcher, sanitizer))


def sanitize_tool_arguments(
    parsed: dict[str, Any],
    tool_name: str,
    *,
    allowed_skill_names: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Run all matching sanitizers on a parsed tool argument dict.

    Returns the (possibly modified) argument dict.  Sanitizers compose in
    registration order; each receives the output of the previous one.
    """
    result = parsed
    for matcher, sanitizer in _REGISTRY:
        try:
            if matcher(tool_name):
                result = sanitizer(result, tool_name)
        except Exception:
            logger.exception(
                "Tool sanitizer %s failed on %s -- skipping",
                getattr(sanitizer, "__name__", repr(sanitizer)),
                tool_name,
            )
    if tool_name == "Skill" and allowed_skill_names:
        result = normalize_skill_arguments(result, allowed_skill_names)
    return result


_SKILL_ARGS_SUFFIX_RE = re.compile(r"^(?P<name>.+?);args=(?P<args>.*?);?$", re.DOTALL)


def normalize_skill_arguments(
    args: dict[str, Any], allowed_skill_names: tuple[str, ...]
) -> dict[str, Any]:
    """Repair only malformed Skill selectors that resolve to an exact catalog entry.

    This intentionally does not fuzzy-match or resurrect removed skills. It
    handles the observed proxy-model corruptions: Unicode format controls,
    trailing punctuation, and an ``;args=...`` fragment appended to ``skill``.
    """
    raw = args.get("skill")
    if not isinstance(raw, str):
        return args
    allowed = set(allowed_skill_names)
    if raw in allowed:
        return args

    candidate = "".join(ch for ch in raw if unicodedata.category(ch) != "Cf").strip()
    recovered_args: str | None = None
    suffix = _SKILL_ARGS_SUFFIX_RE.fullmatch(candidate)
    if suffix:
        candidate = suffix.group("name").strip()
        encoded_args = suffix.group("args").strip()
        if encoded_args:
            try:
                decoded = json.loads(encoded_args)
            except json.JSONDecodeError:
                return args
            if not isinstance(decoded, str):
                return args
            recovered_args = decoded
    candidate = candidate.rstrip(";:").strip()
    if candidate not in allowed:
        return args

    result = dict(args)
    result["skill"] = candidate
    if recovered_args and not result.get("args"):
        result["args"] = recovered_args
    logger.warning(
        "Normalized malformed Skill selector to %r (removed_or_moved=%d chars)",
        candidate,
        max(len(raw) - len(candidate), 0),
    )
    return result


# ---------------------------------------------------------------------------
# Built-in registrations (run at import time)
# ---------------------------------------------------------------------------


def _matches_js_code_tool(name: str) -> bool:
    """True if ``name`` matches a known JS code execution tool."""
    return any(
        name == suffix or name.endswith(f"__{suffix}")
        for suffix in _JS_CODE_PARAMS
    )


register(_matches_js_code_tool, strip_js_trailing_tokens)
