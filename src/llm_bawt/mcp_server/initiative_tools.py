"""Compact MCP access to restart-safe BawtHub initiative state.

Bots tagged ``private`` or ``inactive`` (TASK-1329) are outside initiative
scope; an unreadable tag snapshot refuses every write.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Literal

import httpx

from ..bawthub_auth import service_auth
from ..bot_tags import BotTags, TagSnapshot
from .registry import mcp
from ..task_turn_context import require_conversation_authority
from .task_association import current_task_turn_context

_INTERNAL_URL = os.getenv(
    "BAWTHUB_TASK_ASSOCIATION_INTERNAL_URL", "http://frontend-prod:3002"
).rstrip("/")


def _run(tags: TagSnapshot) -> dict[str, Any]:
    context = require_conversation_authority(current_task_turn_context())
    excluded = tags.excluded_reason(context.bot_id)
    if excluded:
        raise ValueError(f"{excluded} is outside initiative scope")
    return {
        "sessionId": context.session_id,
        "turnId": context.turn_id,
        "triggerMessageId": context.trigger_message_id,
        "botId": context.bot_id,
    }


def _error(error: httpx.HTTPStatusError) -> dict[str, Any]:
    try:
        payload = error.response.json()
    except Exception:
        payload = None
    message = payload.get("error") if isinstance(payload, dict) else str(error)
    result: dict[str, Any] = {"error": str(message), "status": error.response.status_code}
    if isinstance(payload, dict):
        # Held/deferred refusals carry why and until when (TASK-894).
        for key in ("code", "heldBy", "holdReason", "deferReason", "deferUntil", "deferRef"):
            if key in payload:
                result[key] = payload[key]
    return result


@mcp.tool(name="initiative_state")
async def initiative_state(
    task_id: str,
    action: Literal["get", "acquire", "checkpoint", "hold", "resume"],
    snapshot: dict[str, Any] | None = None,
    expected_generation: int | None = None,
    run_id: str | None = None,
    lease_token: str | None = None,
    lease_seconds: int = 900,
    release_lease: bool = False,
    event: dict[str, Any] | None = None,
    defer: dict[str, Any] | None = None,
    resolved_ref: str | None = None,
    reason: str | None = None,
) -> dict:
    """Read, lease, CAS-checkpoint, hold or resume recurring task state using trusted turn identity.

    acquire: omit snapshot to resume the stored plan; pass resolved_ref when the
    wake reports the approval the initiative is deferred on. A 423 means held
    (a person or a cap paused it: stop), a 429 means deferred (stop until
    defer_until or the approval wake). checkpoint: defer={"reason": "quota" |
    "failure", "until": ISO} or {"reason": "approval", "ref": id} parks the
    initiative and releases the lease instead of retrying. hold needs reason.
    """
    path = f"/internal/tasks/{task_id}/initiative"
    run = None if action == "get" else _run(await asyncio.to_thread(BotTags().snapshot))
    try:
        async with httpx.AsyncClient(base_url=_INTERNAL_URL, timeout=10.0,
                                     auth=service_auth()) as client:
            if action == "get":
                state = await client.get(path)
                state.raise_for_status()
                events = await client.get(f"{path}/events", params={"limit": 50})
                events.raise_for_status()
                return {**state.json(), **events.json()}
            if action in ("hold", "resume"):
                response = await client.post(f"{path}/hold", json={
                    "held": action == "hold", "actor": run["botId"], "reason": reason})
                response.raise_for_status()
                return response.json()
            body: dict[str, Any] = {
                "snapshot": snapshot,
                "run": run,
                "leaseSeconds": lease_seconds,
            }
            if expected_generation is not None:
                body["expectedGeneration"] = expected_generation
            if action == "acquire":
                if resolved_ref:
                    body["resolvedRef"] = resolved_ref
                response = await client.post(f"{path}/acquire", json=body)
            else:
                body.update({
                    "runId": run_id,
                    "leaseToken": lease_token,
                    "event": event,
                    "releaseLease": release_lease,
                })
                if defer is not None:
                    body["defer"] = defer
                response = await client.post(f"{path}/checkpoint", json=body)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as error:
        return _error(error)
