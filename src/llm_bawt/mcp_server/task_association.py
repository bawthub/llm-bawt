"""Trusted BawtHub task association operations for MCP task tools."""

from __future__ import annotations

import asyncio
import logging
import os
from contextvars import ContextVar, Token
from typing import Any

import httpx

from agent_bridge.mcp_call_context import MCP_REQUEST_CONTEXT_HEADER

from ..bawthub_auth import service_auth
from ..task_turn_context import (
    SENDER_TURN_CONTEXT_HEADER,
    TASK_TURN_CONTEXT_HEADER,
    TaskTurnContext,
    is_synthetic_turn_id,
    open_task_turn_context,
)

logger = logging.getLogger(__name__)

_HEADER_BYTES = TASK_TURN_CONTEXT_HEADER.lower().encode("ascii")
_REQUEST_CONTEXT_HEADER_BYTES = MCP_REQUEST_CONTEXT_HEADER.lower().encode("ascii")
_INTERNAL_URL = os.getenv(
    "BAWTHUB_TASK_ASSOCIATION_INTERNAL_URL",
    "http://frontend-prod:3002",
).rstrip("/")
_TIMEOUT = 10.0
_current_capability: ContextVar[str | None] = ContextVar(
    "task_turn_context_capability",
    default=None,
)
_current_mcp_request_context: ContextVar[str | None] = ContextVar(
    "mcp_request_context",
    default=None,
)
# TASK-1362: whether the request carried the capability header at all, even a
# blank or undecodable one that binds as no capability.
_capability_header_present: ContextVar[bool] = ContextVar(
    "task_turn_context_header_present",
    default=False,
)


def set_current_task_turn_capability(value: str | None) -> Token:
    """Bind one HTTP request's opaque capability to its async task context."""
    return _current_capability.set(str(value or "").strip() or None)


def reset_current_task_turn_capability(token: Token) -> None:
    _current_capability.reset(token)


def current_task_turn_capability() -> str | None:
    """Return the opaque capability bound to this HTTP request, if any."""
    return _current_capability.get()


def with_sender_turn(headers: dict[str, str] | None = None) -> dict[str, str] | None:
    """``headers`` plus this request's capability under the sender header.

    The delivery routes use it only to record which turn sent a delivery
    (TASK-1249); without a capability the headers are returned unchanged.
    """
    capability = current_task_turn_capability()
    if not capability:
        return headers
    return {**(headers or {}), SENDER_TURN_CONTEXT_HEADER: capability}


def task_turn_capability_header_present() -> bool:
    """True when this request sent the capability header, valid or not."""
    return _capability_header_present.get() or bool(current_task_turn_capability())


def current_mcp_request_context() -> str | None:
    """Return the request-local signed Codex envelope, if supplied."""
    return _current_mcp_request_context.get()


def set_current_mcp_request_context(value: str | None) -> Token:
    """Bind one HTTP request's signed bridge envelope for tests/adapters."""
    return _current_mcp_request_context.set(str(value or "").strip() or None)


def reset_current_mcp_request_context(token: Token) -> None:
    _current_mcp_request_context.reset(token)


def current_task_turn_context() -> TaskTurnContext:
    return open_task_turn_context(current_task_turn_capability())


def task_turn_context_header_name() -> str:
    return TASK_TURN_CONTEXT_HEADER


def _header_from_asgi_scope(scope: dict[str, Any], header: bytes) -> str | None:
    for raw_name, raw_value in scope.get("headers") or []:
        if raw_name.lower() == header:
            try:
                return raw_value.decode("ascii").strip() or None
            except UnicodeDecodeError:
                return None
    return None


def capability_header_in_asgi_scope(scope: dict[str, Any]) -> bool:
    """True when the capability header is present, whatever its value."""
    return any(raw_name.lower() == _HEADER_BYTES for raw_name, _ in scope.get("headers") or [])


def capability_from_asgi_scope(scope: dict[str, Any]) -> str | None:
    """Extract the single opaque capability header from an ASGI HTTP scope."""
    return _header_from_asgi_scope(scope, _HEADER_BYTES)


def request_context_from_asgi_scope(scope: dict[str, Any]) -> str | None:
    """Extract the signed bridge-request envelope from an ASGI HTTP scope."""
    return _header_from_asgi_scope(scope, _REQUEST_CONTEXT_HEADER_BYTES)


class TaskTurnCapabilityMiddleware:
    """Bind the MCP HTTP header to a ContextVar for one ASGI request only."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        binding = set_current_task_turn_capability(capability_from_asgi_scope(scope))
        present_binding = _capability_header_present.set(capability_header_in_asgi_scope(scope))
        request_binding = set_current_mcp_request_context(
            request_context_from_asgi_scope(scope)
        )
        try:
            await self.app(scope, receive, send)
        finally:
            reset_current_mcp_request_context(request_binding)
            _capability_header_present.reset(present_binding)
            reset_current_task_turn_capability(binding)


async def associate_current_task(task_ref: str) -> dict[str, Any]:
    """Associate the trusted current session+turn to ``task_ref`` as AGENT."""
    context = current_task_turn_context()

    # Synthetic inter-bot-delivery turn ids (``turn-delivery-<hex>``) are
    # signed into the capability for audit but are NOT valid ``AgentTaskTurn.turnId``
    # values on the BawtHub side. Per the TASK-706 design, delivery-turn
    # attribution rides on ``triggerMessageId`` alone; the resolve batch and
    # PUT bodies both filter delivery ids so BawtHub's strict validator can
    # stay strict (``turn-<32 hex>`` only). Null the turnId here instead of
    # letting the PUT 400 the whole association.
    turn_id_for_put = None if is_synthetic_turn_id(context.turn_id) else context.turn_id
    body = {
        "sessionId": context.session_id,
        "botId": context.bot_id,
        "userId": context.user_id,
        "source": "AGENT",
        "turn": {
            "triggerMessageId": context.trigger_message_id,
            "turnId": turn_id_for_put,
            "source": "AGENT",
        },
    }
    # TASK-1472: the BawtHub service bearer authenticates llm-bawt as a service
    # caller (BawtHub auth v1 requires it on the internal listener). Trust in
    # WHICH turn is associated still comes from the signed Fernet capability
    # above, which is a correctness mechanism, not auth.
    async with httpx.AsyncClient(
        base_url=_INTERNAL_URL,
        timeout=_TIMEOUT,
        auth=service_auth(),
    ) as client:
        response = await client.put(
            f"/internal/tasks/{task_ref}/associations",
            json=body,
        )
        response.raise_for_status()
        payload = response.json()
    association = payload.get("association") if isinstance(payload, dict) else None
    result = {
        "ok": True,
        "taskRef": task_ref,
        "source": "AGENT",
        "association": association,
    }
    # This is deliberately after the association commits. Memory failures
    # cannot roll back or masquerade as a failed task/turn association.
    canonical = association.get("taskId") if isinstance(association, dict) else None
    if isinstance(canonical, str) and canonical:
        try:
            from llm_bawt.working_memory.service import WorkingMemoryService
            from llm_bawt.working_memory.contracts import WorkingMemoryError
            from llm_bawt.utils.config import Config
            from llm_bawt.utils.db import get_shared_engine

            service = WorkingMemoryService(get_shared_engine(Config()))
            link = await asyncio.to_thread(service.link_associated, context, canonical)
            result["workingMemoryLink"] = {
                "state": link.get("state", "linked"), "taskId": canonical,
                "itemId": link.get("item", {}).get("id"),
            }
        except WorkingMemoryError as error:
            # An off/private bot has no working-memory surface at all.
            if error.code not in ("protected_bot", "not_enabled"):
                result["workingMemoryLink"] = error.as_dict()
                logger.warning("task association succeeded but working-memory link failed: %s", error.code)
        except Exception:
            result["workingMemoryLink"] = {"error": "unavailable", "message": "task link not verified"}
            logger.warning("task association succeeded but working-memory link failed", exc_info=True)
    return result
