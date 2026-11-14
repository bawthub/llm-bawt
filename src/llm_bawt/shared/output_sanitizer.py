"""Utilities for stripping model/tool protocol leakage from user-facing text."""

from __future__ import annotations

import json
import re


_ROUTING_FRAGMENT_RE = re.compile(
    r"(?:^|\s)to=(?:functions|multi_tool_use)\.[^\n\r]*",
    re.IGNORECASE,
)

_TOOL_USES_START_RE = re.compile(r'\{\s*"tool_uses"\s*:', re.IGNORECASE)

# Fallback for malformed (undecodable) payloads: previous best-effort match.
_TOOL_USES_JSON_RE = re.compile(
    r'\{\s*"tool_uses"\s*:\s*\[.*?\]\s*\}',
    re.DOTALL | re.IGNORECASE,
)

_JSON_DECODER = json.JSONDecoder()


def _strip_tool_uses_json(text: str) -> str:
    """Remove complete ``{"tool_uses": ...}`` objects, nested content included."""
    out: list[str] = []
    pos = 0
    while True:
        match = _TOOL_USES_START_RE.search(text, pos)
        if not match:
            out.append(text[pos:])
            return "".join(out)
        start = match.start()
        out.append(text[pos:start])
        try:
            _, end = _JSON_DECODER.raw_decode(text, start)
        except ValueError:
            fallback = _TOOL_USES_JSON_RE.match(text, start)
            if fallback:
                end = fallback.end()
            else:
                out.append(text[start:match.end()])
                end = match.end()
        pos = end

_TOOL_CALL_XML_RE = re.compile(
    r"<tool_call>.*?</tool_call>",
    re.DOTALL,
)



def strip_tool_protocol_leakage(text: str) -> str:
    """Remove leaked tool-routing protocol fragments from model text.

    This strips artifacts such as:
    - ``to=functions.some_tool commentary ...``
    - ``to=multi_tool_use.parallel ...``
    - Raw JSON payloads like ``{"tool_uses": [...]}``
    - ``<tool_call>...</tool_call>`` XML blocks
    """
    if not text:
        return ""

    cleaned = _TOOL_CALL_XML_RE.sub("", text)
    cleaned = _strip_tool_uses_json(cleaned)
    cleaned = _ROUTING_FRAGMENT_RE.sub(" ", cleaned)

    # Remove control characters often left behind after malformed protocol output.
    cleaned = re.sub(r"[\x00-\x08\x0e-\x1f\x7f-\x9f]", "", cleaned)

    # Trim trailing spaces per line and squeeze blank runs; leading indentation
    # and interior spacing are content (code, nested lists, YAML, tables).
    cleaned = re.sub(r"[ \t]+$", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    # Strip leftover leading whitespace, but keep the first line's indentation
    # when the reply itself started indented (e.g. an indented code block).
    cleaned = cleaned.rstrip()
    body = cleaned.lstrip()
    indent = re.match(r"[ \t]*", text.lstrip("\r\n")).group(0)
    if indent and cleaned.lstrip("\r\n").startswith(indent + body[:1]):
        return indent + body
    return body
