"""Trusted MCP identity adapters; supplied bad credentials never become anonymous."""
from __future__ import annotations

from contextvars import ContextVar

from llm_bawt.bot_access import BotAccessError, TrustedCaller
from llm_bawt.bot_access.service import SERVICE_HEADER, open_service_capability
from llm_bawt.task_turn_context import (
    TASK_TURN_CONTEXT_HEADER, TASK_TURN_CONTEXT_TTL_SECONDS,
    open_task_turn_context,
)
from .approval_interceptor import current_approved_caller_context
from .task_association import current_task_turn_capability

_service_capability: ContextVar[str | None] = ContextVar("mcp_service_capability", default=None)
_raw_turn_capability: ContextVar[str | None] = ContextVar("mcp_raw_turn_capability", default=None)


def trusted_caller(name: str, arguments: dict) -> TrustedCaller | None:
    identities: list[TrustedCaller] = []
    try:
        approved = current_approved_caller_context()
        if approved is not None:
            identities.append(TrustedCaller(approved.bot_id, approved.user_id, "approved"))
        # The transport copy preserves malformed/blank/duplicate headers that
        # the older association adapter historically discarded.
        raw = _raw_turn_capability.get()
        if raw is None:
            raw = current_task_turn_capability()
        if raw is not None:
            turn = open_task_turn_context(raw)
            identities.append(TrustedCaller(turn.bot_id, turn.user_id, "turn",
                                             turn.issued_at + TASK_TURN_CONTEXT_TTL_SECONDS))
        service = _service_capability.get()
        if service is not None:
            identities.append(open_service_capability(service, name, arguments))
        for identity in identities:
            identity.validate()
        if len({(identity.bot_id, identity.user_id) for identity in identities}) > 1:
            raise BotAccessError("caller_unavailable")
        return identities[0] if identities else None
    except Exception:
        raise BotAccessError("caller_unavailable") from None


def strict_header(scope: dict, header: str) -> str | None:
    values = [value for key, value in scope.get("headers", [])
              if key.lower() == header.lower().encode("ascii")]
    if not values:
        return None
    if len(values) != 1:
        return "invalid-duplicate-header"
    try:
        return values[0].decode("ascii")  # blank is supplied, not absent
    except UnicodeDecodeError:
        return "invalid-nonascii-header"


class BotAccessMiddleware:
    """Separate purpose-bound service envelope; no approval/capability bypass."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        service = _service_capability.set(strict_header(scope, SERVICE_HEADER))
        turn = _raw_turn_capability.set(strict_header(scope, TASK_TURN_CONTEXT_HEADER))
        try:
            await self.app(scope, receive, send)
        finally:
            _raw_turn_capability.reset(turn)
            _service_capability.reset(service)
