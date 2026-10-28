"""Delegated ops approval (TASK-1609): ``/v1/ops-approvals``.

A bot holding the ``approve_ops`` grant tag resolves another bot's pending
``ops_run`` approval card, the same card a person resolves in chat.

* ``GET /v1/ops-approvals``: the cards the caller may resolve.
* ``POST /v1/ops-approvals/{request_id}/resolve``: ``{decision, reason}``.

The only caller is a bot's signed turn capability
(``X-LLM-Bawt-Task-Turn-Context``, forwarded by the ``ops_approvals_*`` MCP
tools). Bot, user and turn come from it, never from the body. Each request
re-checks the grant live, so a revoke takes effect on the next call. A card is
resolvable when it is a pending first-party ``ops_run`` for the same user, its
requester isn't private, and the caller didn't request it. A card the caller
can't see (another user's, a private requester's, or none) is ``not_found``
either way.

The decision goes through ``resolve_mcp_card``, the path the chat card takes,
with ``resolved_via=bot_grant`` written in the same compare-and-set. A caller
that loses the race gets the winner's decision. The listing shows the
operation and its arguments with ``x-sensitive`` values masked. It never
returns the stored prompt, subject or raw arguments, which embed them.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from ...approval_models import (
    BOT_RESOLVER_PREFIX,
    KIND_MCP,
    OPS_RUN_TOOL,
    REQ_PENDING,
    RESOLVED_VIA_BOT_GRANT,
)
from ...bot_tags import BehaviorTag, grants
from ...bot_tags.facade import BotTags
from ...ops.service import display_args
from ...task_turn_context import (
    TASK_TURN_CONTEXT_HEADER,
    TaskTurnContext,
    TaskTurnContextError,
    open_task_turn_context,
)
from .approval_policies import _store, resolve_mcp_card
from .bot_tags import bot_tag_store
from .service_auth import NO_STORE, read_body

router = APIRouter(tags=["Approval Policies"])

REASON_MAX = 1000


class OpsApprovalDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    decision: Literal["approve", "deny"]
    reason: str = Field(min_length=1, max_length=REASON_MAX)


class OpsApprovalCard(BaseModel):
    request_id: str
    requested_by: str
    created_at: datetime | None
    severity: str
    operation: str
    title: str | None
    risk_level: str | None
    args: Any


class OpsApprovalList(BaseModel):
    data: list[OpsApprovalCard]


class OpsApprovalResult(BaseModel):
    request_id: str
    status: str
    resolved_by: str | None
    resolved_by_bot: str | None
    reason: str
    awaiting_admission: bool = False
    execution_state: str | None = None


def _refuse(http_status: int, code: str, message: str, /, **extra) -> HTTPException:
    """A refusal; ``extra`` (e.g. the winner's ``status``) rides in the detail."""
    return HTTPException(http_status, {"code": code, "message": message, **extra}, headers=NO_STORE)


def _caller(request: Request) -> TaskTurnContext:
    token = request.headers.get(TASK_TURN_CONTEXT_HEADER)
    if not token:
        raise _refuse(403, "turn_capability_required", "call this from a bot's signed turn")
    try:
        return open_task_turn_context(token)
    except TaskTurnContextError as exc:
        raise _refuse(403, "turn_capability_invalid", str(exc)) from exc


def _holder_and_tags(bot: str):
    """Whether ``bot`` holds ``approve_ops`` now, and the tag snapshot to judge cards by."""
    with bot_tag_store().engine.connect() as conn:
        return grants.holds_grant(conn, bot, BehaviorTag.APPROVE_OPS), BotTags().snapshot(conn)


def _visible(row, caller: TaskTurnContext, tags) -> bool:
    """The caller's user's card, from a non-private requester; fails closed."""
    return (row is not None and row.user_id == caller.user_id and tags.available
            and not tags.has(row.bot_id, BehaviorTag.PRIVATE))


def _is_ops_card(row) -> bool:
    return row.request_kind == KIND_MCP and row.tool_name == OPS_RUN_TOOL


def _card(row) -> OpsApprovalCard | None:
    """The listing view of one card, or None when its snapshot can't be shown safely."""
    try:
        snapshot = json.loads(row.operations_snapshot_json or "null")
        operation = snapshot["operation"]
        args = display_args(snapshot["resolved_args"], json.dumps(snapshot["schema"]))
        slug = operation["slug"]
    except (TypeError, KeyError, ValueError, AttributeError):
        return None
    if not isinstance(slug, str) or not slug:
        return None
    return OpsApprovalCard(
        request_id=row.id, requested_by=row.bot_id, created_at=row.created_at,
        severity=row.severity or "medium", operation=slug, title=operation.get("title"),
        risk_level=operation.get("risk_level"), args=args)


def _result(row, result: dict | None = None) -> OpsApprovalResult:
    result = result or {}
    return OpsApprovalResult(
        request_id=row.id, status=row.status, resolved_by=row.resolved_by,
        resolved_by_bot=row.resolved_by_bot, reason=row.resolution_message or "",
        awaiting_admission=bool(result.get("awaiting_admission")),
        execution_state=row.execution_state)


def _require_holder(caller: TaskTurnContext):
    holds, tags = _holder_and_tags(caller.bot_id)
    if not holds:
        raise _refuse(403, "grant_required", "this bot doesn't hold the approve_ops grant")
    return tags


@router.get("/v1/ops-approvals", response_model=OpsApprovalList)
async def list_ops_approvals(request: Request):
    """Pending ops cards the calling grant holder may resolve, oldest first."""
    caller = _caller(request)
    tags = await asyncio.to_thread(_require_holder, caller)
    rows = await asyncio.to_thread(_store().list_pending_ops_runs, user_id=caller.user_id)
    cards = [_card(row) for row in rows
             if _visible(row, caller, tags) and row.bot_id != caller.bot_id]
    return OpsApprovalList(data=[card for card in cards if card is not None])


@router.post("/v1/ops-approvals/{request_id}/resolve", response_model=OpsApprovalResult)
async def resolve_ops_approval(request_id: str, request: Request):
    """Approve or deny another bot's pending ops card as the calling grant holder."""
    caller = _caller(request)
    body = await read_body(request, OpsApprovalDecision)
    tags = await asyncio.to_thread(_require_holder, caller)
    store = _store()
    row = await asyncio.to_thread(store.get_request, request_id)
    if not _visible(row, caller, tags):
        raise _refuse(404, "not_found", "no such approval request")
    if not _is_ops_card(row):
        raise _refuse(409, "not_ops_card", "only ops_run approvals can be resolved here")
    if row.bot_id == caller.bot_id:
        raise _refuse(403, "self_approval_refused", "a bot can't approve its own request")
    if row.status != REQ_PENDING:
        raise _refuse(409, "already_resolved", "the request is already resolved",
                      **_result(row).model_dump())
    resolver = f"{BOT_RESOLVER_PREFIX}{caller.bot_id}"
    result, final = await resolve_mcp_card(
        store, row, outcome=body.decision, message=body.reason, resolved_by=resolver,
        resolved_via=RESOLVED_VIA_BOT_GRANT, resolved_turn_id=caller.turn_id)
    won = (final.resolved_via == RESOLVED_VIA_BOT_GRANT and final.resolved_by == resolver
           and final.resolved_turn_id == caller.turn_id)
    if not won:
        raise _refuse(409, "already_resolved", "another resolver decided first",
                      **_result(final).model_dump())
    return _result(final, result)
