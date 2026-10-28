"""MCP tools for a task's append-only review history (TASK-1147, TASK-1615).

Split from task_tools so the review contract can grow without growing that
module past the monolith gate. Imported by server.py to register the tools.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Literal, NotRequired, TypedDict

import httpx

from llm_bawt.bot_access import BotAccessError

from .review_provenance import resolve_review_provenance, signed_review_turn
from .server import mcp
from .task_api import (
    api_get as _api_get,
    api_post as _api_post,
    headers as _headers,
    http_error as _http_error,
)

logger = logging.getLogger(__name__)


@mcp.tool(name="tasks_reviews_list")
async def list_task_reviews(
    task_id: str,
    limit: int = 20,
    cursor: str | None = None,
) -> dict:
    """Read a task's append-only review history, newest first.

    Args:
        task_id: Task ID (cuid) or shortId (e.g. "TASK-42").
        limit: Page size (default 20, capped at 50).
        cursor: Opaque next-page cursor from the previous response.

    Returns:
        The task-scoped review page and continuation cursor as returned by
        BawtHub, or an error dict. Reviews are separate from task response.
    """
    params = {"limit": str(max(1, min(limit, 50)))}
    if cursor is not None:
        params["cursor"] = cursor
    try:
        return await _api_get(f"/tasks/{task_id}/reviews", params=params)
    except httpx.HTTPStatusError as error:
        return _http_error(error)


async def _review_task_refs(task_id: str) -> list[str]:
    """The reviewed task's cuid and shortId, so either form matches a delivery."""
    try:
        task = await _api_get(f"/tasks/{task_id}")
    except Exception:
        logger.warning("review provenance: task %s unavailable", task_id, exc_info=True)
        return []
    if not isinstance(task, dict):
        return []
    return [ref for ref in (task.get("id"), task.get("shortId")) if isinstance(ref, str) and ref]


class ReviewFindingInput(TypedDict):
    # One structured review finding (TASK-1615). A comment, not a docstring:
    # a docstring would ship in every catalog read as a schema description.
    severity: Literal["BLOCKER", "SHOULD_FIX", "NIT", "QUESTION", "PRE_EXISTING"]
    title: str
    detail: str
    file: NotRequired[str]
    line: NotRequired[int]
    failure_scenario: NotRequired[str]


def _finding_payload(finding: ReviewFindingInput) -> dict:
    """BawtHub's wire shape. Validation stays in BawtHub, the one source of truth."""
    out = dict(finding)
    if "failure_scenario" in out:
        out["failureScenario"] = out.pop("failure_scenario")
    return out


@mcp.tool(name="tasks_reviews_add")
async def add_task_review(
    task_id: str,
    stage: Literal["DESIGN", "IMPLEMENTATION"],
    verdict: Literal["APPROVE", "APPROVE_WITH_CHANGES", "REQUEST_CHANGES", "COMMENT"],
    body: str,
    bot_id: str,
    target_revision: str | None = None,
    idempotency_key: str | None = None,
    findings: list[ReviewFindingInput] | None = None,
) -> dict:
    """Append a review decision or comment without overwriting the executor response.

    Args:
        task_id: Task ID (cuid) or shortId (e.g. "TASK-42").
        stage: DESIGN or IMPLEMENTATION; design reviews can happen before REVIEW.
        verdict: APPROVE, APPROVE_WITH_CHANGES, REQUEST_CHANGES, or COMMENT.
        body: Non-empty review text; for an approval without notes use a short rationale.
        bot_id: Your own bot ID, sent as actor context, not as a review author field.
        target_revision: Optional description digest or commit/revision reference.
        idempotency_key: Stable key to safely retry the same submission.
        findings: Optional, in order (max 40). BLOCKER needs REQUEST_CHANGES
            and failure_scenario; line needs file.

    Returns:
        The review receipt as returned by BawtHub, or an error dict. This does
        not change the task's status, description, response, or assignee.
    """
    if not bot_id.strip():
        return {"error": "bot_id is required for review attribution"}
    payload = {"stage": stage, "verdict": verdict, "body": body}
    if target_revision is not None:
        payload["targetRevision"] = target_revision
    if idempotency_key is not None:
        payload["idempotencyKey"] = idempotency_key
    signed_arguments = {
        "task_id": task_id, "stage": stage, "verdict": verdict, "body": body,
        "bot_id": bot_id, "target_revision": target_revision,
        "idempotency_key": idempotency_key,
    }
    if findings is not None:
        payload["findings"] = [_finding_payload(finding) for finding in findings]
        # Named only when sent, so existing callers' invocation hash is unchanged.
        signed_arguments["findings"] = findings
    # TASK-1249: models come from the caller's signed turn, never from
    # arguments or the body. A bad or foreign credential is refused.
    try:
        turn = signed_review_turn(bot_id, signed_arguments)
    except BotAccessError as error:
        return {"error": error.code}
    task_refs = await _review_task_refs(task_id) if turn is not None else []
    payload.update(await asyncio.to_thread(resolve_review_provenance, turn, task_refs))
    try:
        return await _api_post(
            f"/tasks/{task_id}/reviews", json=payload, headers=_headers(bot_id)
        )
    except httpx.HTTPStatusError as error:
        return _http_error(error)

