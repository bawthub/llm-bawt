"""Resolve approval result delivery without changing the approved invocation."""
from __future__ import annotations

import json
from ..approval_policies import KIND_MCP
from ..memory.session_reset_lineage import SessionResetLineage


def approval_origin_session(row) -> str | None:
    if not row.caller_context_json:
        return None
    context = json.loads(row.caller_context_json)
    if not isinstance(context, dict):
        raise ValueError("Stored caller context is invalid")
    return context.get("session_id") or None


def approval_delivery_session(service, row) -> str | None:
    origin = approval_origin_session(row)
    if not origin:
        return None
    client = service.get_memory_client(row.bot_id, row.user_id)
    if client is None:
        raise ValueError("Approval continuation session lookup unavailable")
    target = SessionResetLineage.resolve(
        client.get_session, origin, bot_id=row.bot_id, user_id=row.user_id,
    )
    # A native harness grant authorizes a specific original SDK session/request.
    # Unlike an already-executed MCP result, it cannot move to a fresh session.
    if target != origin and row.request_kind != KIND_MCP:
        raise ValueError("Session was reset; native tool approval needs a fresh request")
    return target
