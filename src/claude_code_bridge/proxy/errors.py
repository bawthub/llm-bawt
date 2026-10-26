"""Anthropic-shaped error envelopes.

The Claude Agent SDK's HTTP client matches against Anthropic's error
shape — ``{type: "error", error: {type, message}}`` — when deciding
whether a turn failed. Wrap every proxy/upstream failure so retry,
auth-recovery, and surface-to-user behaviors stay correct.
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import HTTPException


# Maps OpenAI-ish error codes onto Anthropic error type strings. Anthropic's
# canonical set is small; everything we don't recognize falls under
# ``api_error`` (5xx) or ``invalid_request_error`` (4xx).
_TYPE_MAP = {
    "invalid_request_error": "invalid_request_error",
    "invalid_api_key": "authentication_error",
    "authentication_error": "authentication_error",
    "permission_error": "permission_error",
    "rate_limit_error": "rate_limit_error",
    "rate_limit_exceeded": "rate_limit_error",
    "not_found_error": "not_found_error",
    "overloaded_error": "overloaded_error",
    "api_error": "api_error",
}


def anthropic_error(
    status_code: int,
    message: str,
    error_type: str | None = None,
) -> HTTPException:
    """Return a FastAPI HTTPException carrying an Anthropic-shaped body.

    Use this from routes for non-stream failures. For mid-stream failures,
    emit an ``error`` SSE frame instead (see ``stream.responses_to_anthropic_sse``).
    """
    if error_type:
        atype = _TYPE_MAP.get(error_type, error_type)
    else:
        atype = (
            "invalid_request_error"
            if 400 <= status_code < 500
            else "api_error"
        )
    detail: dict[str, Any] = {
        "type": "error",
        "error": {"type": atype, "message": message},
    }
    return HTTPException(status_code=status_code, detail=detail)


def final_error_frame(error_type: str, message: str) -> bytes:
    """One Anthropic SSE error shape for adapter and route failures."""
    payload = {"type": "error", "error": {"type": error_type, "message": message}}
    return f"event: error\ndata: {json.dumps(payload, separators=(',', ':'))}\n\n".encode()


def auth_failed_error(message: str | None = None) -> HTTPException:
    msg = message or (
        "ChatGPT OAuth bundle is invalid or expired. Re-run `codex login` "
        "on the bridge host."
    )
    return anthropic_error(401, msg, error_type="authentication_error")
