"""Trusted agent-turn adapter to the shared message routing service."""
from __future__ import annotations

import httpx

from .server import mcp
from .task_association import current_task_turn_capability, current_task_turn_context
from ..task_turn_context import TASK_TURN_CONTEXT_HEADER, TaskTurnContextError

_APP_BASE_URL = "http://app:8642"


@mcp.tool(name="bots_route_message")
async def bots_route_message(
    message: str,
    idempotency_key: str,
    candidate_bot_ids: list[str] | None = None,
    model_endpoint_id: int | None = None,
    session_policy: str = "continue",
    task_id: str | None = None,
    project_id: str | None = None,
) -> dict:
    """Queue one separate routed bot turn; never steer or broaden opt-in.

    The caller's signed turn capability, not model-supplied bot/user names,
    supplies its owner and author. Reuse the same key to inspect a retry.
    """
    try:
        context = current_task_turn_context()
    except TaskTurnContextError as exc:
        return {"state": "not_routed", "error": str(exc), "dispatched": False}
    if not message.strip() or not idempotency_key.strip():
        return {"state": "not_routed", "error": "message and idempotency_key are required", "dispatched": False}
    body = {
        "user_id": context.user_id,
        "message": message,
        "idempotency_key": idempotency_key,
        "candidate_bot_ids": candidate_bot_ids,
        "model_endpoint_id": model_endpoint_id,
        "session_policy": session_policy,
        "scope": "personal",
        "task_id": task_id,
        "project_id": project_id,
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{_APP_BASE_URL}/v1/message-routing/route", json=body,
            headers={TASK_TURN_CONTEXT_HEADER: current_task_turn_capability()},
            timeout=20.0,
        )
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail")
        except ValueError:
            detail = response.text
        return {"state": "not_routed", "error": detail, "status_code": response.status_code, "dispatched": False}
    receipt = response.json()
    return {**receipt, "dispatched": bool(receipt.get("delivery_id"))}
