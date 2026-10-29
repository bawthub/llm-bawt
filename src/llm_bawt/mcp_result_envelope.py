"""Identify the app's MCP-result continuation envelope in stored messages.

The provider requires this internal result as a user-role message. Legacy rows
have user authorship too, so maintenance must recognize the *whole protocol
frame*, not merely words a user could quote inside an ordinary message.
"""
from __future__ import annotations

import json
from typing import Any

MCP_RESULT_ENVELOPE_PREFIX = "[LLM_BAWT_MCP_TOOL_RESULT]"
MCP_RESULT_ENVELOPE_VERSION = 2
_MCP_RESULT_END = "[/LLM_BAWT_MCP_TOOL_RESULT_V2]"

# Exact historical tails are immutable: stored v1 rows must still classify as
# tool results after operators edit the v2 guidance in the prompt registry.
LEGACY_MCP_RESULT_GUIDANCE = (
    "The MCP tool has already been executed or refused server-side. Do not "
    "retry or re-issue the tool. Treat the result above as authoritative and "
    "continue from it."
)
LEGACY_RESET_GUIDANCE = (
    "The user reset the conversation with /new while this result was pending. "
    "Continue in this fresh session using its message history; do not restore "
    "the previous SDK context or re-execute the completed operation."
)


def format_mcp_result_envelope(payload: dict[str, Any], guidance: str) -> str:
    """Frame data and editable guidance without making guidance part of identity."""
    if not guidance.strip():
        raise ValueError("MCP result guidance must not be blank")
    canonical = json.dumps({**payload, "frame_version": MCP_RESULT_ENVELOPE_VERSION},
                           ensure_ascii=False, sort_keys=True, default=str)
    return f"{MCP_RESULT_ENVELOPE_PREFIX}\n{canonical}\n{guidance}\n{_MCP_RESULT_END}"


def append_reset_guidance(content: str, guidance: str) -> str:
    """Place the /new note inside the versioned frame, before its end marker."""
    if (not guidance.strip() or not is_mcp_result_envelope(content)
            or not content.endswith("\n" + _MCP_RESULT_END)):
        raise ValueError("Expected a versioned MCP result envelope and reset guidance")
    return content.removesuffix("\n" + _MCP_RESULT_END) + f"\n{guidance}\n{_MCP_RESULT_END}"


def is_mcp_result_envelope(content: Any) -> bool:
    """Match a complete v2 frame or the exact stored v1 writer output."""
    prefix = MCP_RESULT_ENVELOPE_PREFIX + "\n"
    if not isinstance(content, str) or not content.startswith(prefix):
        return False
    try:
        payload, end = json.JSONDecoder().raw_decode(content[len(prefix):])
    except ValueError:
        return False
    if not isinstance(payload, dict) or payload.get("kind") != "mcp_tool_result":
        return False
    if not all(isinstance(payload.get(key), str) and payload[key] for key in (
        "approval_request_id", "tool_name"
    )):
        return False
    tool_use_id = payload.get("original_tool_use_id")
    if ("original_tool_use_id" not in payload
            or tool_use_id is not None and not isinstance(tool_use_id, str)
            or not isinstance(payload.get("is_error"), bool)
            or "result" not in payload):
        return False
    tail = content[len(prefix) + end:]
    if payload.get("frame_version") == MCP_RESULT_ENVELOPE_VERSION:
        canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        return (content[len(prefix):len(prefix) + end] == canonical
                and tail.startswith("\n") and tail.endswith("\n" + _MCP_RESULT_END)
                and bool(tail[1:-len("\n" + _MCP_RESULT_END)].strip()))
    return ("frame_version" not in payload
            and tail in ("\n" + LEGACY_MCP_RESULT_GUIDANCE,
                         "\n" + LEGACY_MCP_RESULT_GUIDANCE + "\n" + LEGACY_RESET_GUIDANCE))
