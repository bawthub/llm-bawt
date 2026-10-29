"""Identify the app's MCP-result continuation envelope in stored messages.

The provider requires this internal result as a user-role message. Legacy rows
have user authorship too, so maintenance must recognize the *whole protocol
frame*, not merely words a user could quote inside an ordinary message.
"""
from __future__ import annotations

import json
from typing import Any

MCP_RESULT_ENVELOPE_PREFIX = "[LLM_BAWT_MCP_TOOL_RESULT]"
_MCP_RESULT_SUFFIX = (
    "\nThe MCP tool has already been executed or refused server-side. Do not "
    "retry or re-issue the tool. Treat the result above as authoritative and "
    "continue from it."
)
_RESET_SUFFIX = (
    "\nThe user reset the conversation with /new while this result was pending. "
    "Continue in this fresh session using its message history; do not restore "
    "the previous SDK context or re-execute the completed operation."
)


def is_mcp_result_envelope(content: Any) -> bool:
    """Match a complete app-generated frame, including preexisting stored rows."""
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
    return content[len(prefix) + end:] in (_MCP_RESULT_SUFFIX, _MCP_RESULT_SUFFIX + _RESET_SUFFIX)
