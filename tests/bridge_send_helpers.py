"""Test helper: parse raw send fields, then run production preflight."""

from __future__ import annotations

from typing import Any

from agent_bridge.publisher import COMMANDS_STREAM
from claude_code_bridge.send_preflight import SendPreflight, prepare_parsed_send
from claude_code_bridge.send_request import SendRequest


async def prepare_send(
    bridge: Any, fields: dict, msg_id: str, async_redis: Any,
) -> SendPreflight | None:
    """Mirror send_admission's parse/validate step before preflight."""
    request = SendRequest.from_fields(fields)
    if not request.request_id or not request.message:
        await async_redis.xack(COMMANDS_STREAM, "claude-code-bridge", msg_id)
        return None
    return await prepare_parsed_send(bridge, request, msg_id, async_redis)
