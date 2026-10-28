"""Grant tag tools (TASK-1609).

* ``bots_grant_set``: a grant manager grants or revokes a grant tag.
* ``ops_approvals_list`` / ``ops_approval_resolve``: a bot holding
  ``approve_ops`` lists and decides other bots' pending ops approvals.

Thin forwarders of the signed turn capability: the app's
``PUT /v1/bot-grants/{tag}/bots/{slug}`` (``bot_tags.grants``) and
``/v1/ops-approvals`` (``service/routes/ops_approvals.py``) open it and decide
everything. The caller is never an argument, so a bot can only act as itself.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any
from urllib.parse import quote

from ..task_turn_context import TASK_TURN_CONTEXT_HEADER
from .registry import mcp
from .task_association import current_task_turn_capability

logger = logging.getLogger(__name__)

_APP_BASE_URL = os.getenv("LLM_BAWT_APP_BASE_URL", "http://localhost:8642").rstrip("/")
_TIMEOUT_S = 30.0


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def _error(error: str, message: str, **extra: Any) -> str:
    return _dump({"error": error, "message": message, **extra})


async def _request(method: str, path: str, body: dict[str, Any] | None, capability: str):
    import httpx

    async with httpx.AsyncClient() as client:
        return await client.request(method, f"{_APP_BASE_URL}{path}", json=body, timeout=_TIMEOUT_S,
                                    headers={TASK_TURN_CONTEXT_HEADER: capability})


async def _forward(method: str, path: str, body: dict[str, Any] | None, *, unknown: str) -> str:
    """Call the app as this turn; a refusal comes back as ``{error, message}``.

    ``unknown`` is the message when the call's outcome can't be known.
    """
    capability = current_task_turn_capability()
    if not capability:
        return _error("unauthorized", "a trusted turn capability is required")
    try:
        response = await _request(method, path, body, capability)
    except Exception:
        logger.exception("signed-turn call failed: %s %s", method, path)
        return _error("unavailable", unknown)
    try:
        payload = response.json()
    except ValueError:
        payload = {"detail": response.text}
    if response.status_code >= 400:
        detail = payload.get("detail") if isinstance(payload, dict) else payload
        if isinstance(detail, dict) and detail.get("code"):
            extra = {k: v for k, v in detail.items() if k not in ("code", "message")}
            return _error(str(detail["code"]), str(detail.get("message") or ""), **extra)
        return _error("rejected", str(detail), status_code=response.status_code)
    return _dump(payload)


def _all_text(*values: object) -> bool:
    return all(isinstance(v, str) and v.strip() for v in values)


@mcp.tool(name="bots_grant_set")
async def bots_grant_set(bot_id: str, tag: str, granted: bool, reason: str) -> str:
    """Grant or revoke a grant tag (such as approve_ops) on another bot.

    Only bots holding the configured grant-manager role may; never your own
    grants, never a private bot. Idempotent; a change is audited with the reason.

    Args:
        bot_id: The bot to grant or revoke.
        tag: The grant tag.
        granted: true grants, false revokes.
        reason: Why; stored on the audit row.
    """
    if not _all_text(bot_id, tag, reason):
        return _error("invalid_request", "bot_id, tag and reason are required")
    path = f"/v1/bot-grants/{quote(tag.strip(), safe='')}/bots/{quote(bot_id.strip().lower(), safe='')}"
    return await _forward("PUT", path, {"granted": bool(granted), "reason": reason},
                          unknown="the write's outcome is unknown; check GET /v1/bot-grants")


@mcp.tool(name="ops_approvals_list")
async def ops_approvals_list() -> str:
    """List other bots' pending ops approvals you may decide (approve_ops holders).

    Each card shows the requester, the operation and its arguments, sensitive
    values masked. Your own requests are not listed.
    """
    return await _forward("GET", "/v1/ops-approvals", None,
                          unknown="the approvals list is unavailable; try again")


@mcp.tool(name="ops_approval_resolve")
async def ops_approval_resolve(request_id: str, decision: str, reason: str) -> str:
    """Approve or deny another bot's pending ops approval (approve_ops holders).

    Recorded as your decision with the reason, shown on the requester's card.
    Never your own request. If someone decided first, returns already_resolved
    with their decision.

    Args:
        request_id: The approval request id from ops_approvals_list.
        decision: "approve" or "deny".
        reason: Why; shown with the decision.
    """
    if not _all_text(request_id, decision, reason):
        return _error("invalid_request", "request_id, decision and reason are required")
    return await _forward(
        "POST", f"/v1/ops-approvals/{quote(request_id.strip(), safe='')}/resolve",
        {"decision": decision.strip().lower(), "reason": reason},
        unknown="the decision's outcome is unknown; check ops_approvals_list")
