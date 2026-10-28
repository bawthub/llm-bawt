"""MCP tools for a task's append-only review history (TASK-1147, TASK-1615).

Split from task_tools so the review contract can grow without growing that
module past the monolith gate. Imported by server.py to register the tools.

A review from a signed turn goes to BawtHub's bearer-protected internal
listener, which stores it as INTERNAL_SIGNED (TASK-1615 slice 2). Only such a
review can be rechecked, and rechecks and finding resolutions are written
there alone. An unsigned review keeps the public route.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Literal, NotRequired, TypedDict
from urllib.parse import quote

import httpx

from llm_bawt.bot_access import BotAccessError

from ..bawthub_auth import service_auth
from .review_provenance import SIGNED_TURN, resolve_review_provenance, signed_review_turn
from .server import mcp
from .task_api import (
    TIMEOUT,
    api_get as _api_get,
    api_post as _api_post,
    headers as _headers,
    http_error as _http_error,
    json_body as _json_body,
)

logger = logging.getLogger(__name__)

# The same internal listener as task association and initiative state.
_INTERNAL_URL = os.getenv(
    "BAWTHUB_TASK_ASSOCIATION_INTERNAL_URL", "http://frontend-prod:3002"
).rstrip("/")


class InternalReviewRouteMissing(Exception):
    """BawtHub answered a plain-text 404: its build predates the internal review routes."""


async def _internal_post(path: str, json: dict) -> Any:
    async with httpx.AsyncClient(
        base_url=_INTERNAL_URL, timeout=TIMEOUT, auth=service_auth()
    ) as client:
        response = await client.post(path, json=json)
    # A missing route is the framework's text 404. The routes' own 404s
    # (task_not_found, review_not_found) are JSON and are real answers.
    content_type = response.headers.get("content-type", "")
    if response.status_code == 404 and "application/json" not in content_type:
        raise InternalReviewRouteMissing(path)
    return _json_body(response)


def _internal_error(error: httpx.HTTPStatusError) -> dict:
    """The HTTP error with BawtHub's stable ``code`` and ``details`` kept."""
    out = _http_error(error)
    try:
        body = error.response.json()
    except Exception:
        return out
    if isinstance(body, dict):
        for key in ("code", "details"):
            if body.get(key) is not None:
                out[key] = body[key]
    return out


def _internal_path(task_id: str, *parts: str) -> str:
    segments = [quote(part, safe="") for part in (task_id, "reviews", *parts)]
    return "/internal/tasks/" + "/".join(segments)


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
    file: NotRequired[str | None]
    line: NotRequired[int | None]
    failure_scenario: NotRequired[str | None]


Resolution = Literal["FIXED", "DEFERRED", "WITHDRAWN"]


class FindingResolutionInput(TypedDict):
    # How a recheck settles one r1 finding: FIXED with the fix revision,
    # DEFERRED to an existing task (never a Blocker), or WITHDRAWN.
    finding_id: str
    resolution: Resolution
    revision: NotRequired[str | None]
    task: NotRequired[str | None]
    note: NotRequired[str | None]


_FINDING_OPTIONAL = frozenset({"file", "line", "failure_scenario"})
_RESOLUTION_OPTIONAL = frozenset({"revision", "task", "note"})


def _present(entry: dict, optional: frozenset[str]) -> dict:
    # Strict-schema clients send every optional key, as null or "". Either
    # means absent; BawtHub rejects e.g. a task on a FIXED resolution.
    # Required keys pass through untouched, so BawtHub validates them.
    return {
        key: value for key, value in entry.items()
        if key not in optional or (value is not None and value != "")
    }


def _finding_payload(finding: ReviewFindingInput) -> dict:
    """BawtHub's wire shape. Validation stays in BawtHub, the one source of truth."""
    out = _present(dict(finding), _FINDING_OPTIONAL)
    if "failure_scenario" in out:
        out["failureScenario"] = out.pop("failure_scenario")
    return out


def _resolution_payload(resolution: FindingResolutionInput) -> dict:
    out = _present(dict(resolution), _RESOLUTION_OPTIONAL)
    out["findingId"] = out.pop("finding_id")
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
    rechecks_review_id: str | None = None,
    resolutions: list[FindingResolutionInput] | None = None,
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
        rechecks_review_id: Your r1's id; records this as its one r2 (signed turn, idempotency_key).
        resolutions: r2 only, settling r1 findings; Blockers FIXED or WITHDRAWN before APPROVE.

    Returns:
        The review receipt as returned by BawtHub, or an error dict. This does
        not change the task's status, description, response, or assignee.
    """
    if not bot_id.strip():
        return {"error": "bot_id is required for review attribution"}
    recheck = rechecks_review_id is not None
    if resolutions is not None and not recheck:
        return {"error": "resolutions need rechecks_review_id", "code": "invalid_review"}
    if recheck and idempotency_key is None:
        return {"error": "a recheck needs idempotency_key", "code": "invalid_review"}
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
    # New arguments are named only when sent, so existing callers'
    # invocation hash is unchanged.
    if findings is not None:
        payload["findings"] = [_finding_payload(finding) for finding in findings]
        signed_arguments["findings"] = findings
    if recheck:
        signed_arguments["rechecks_review_id"] = rechecks_review_id
    if resolutions is not None:
        signed_arguments["resolutions"] = resolutions
    # TASK-1249: models come from the caller's signed turn, never from
    # arguments or the body. A bad or foreign credential is refused.
    try:
        turn = signed_review_turn(bot_id, signed_arguments)
    except BotAccessError as error:
        return {"error": error.code}
    if recheck and turn is None:
        # Checked before any request: an unsigned r2 could clear Blockers.
        return {"error": "recheck_requires_signed_turn", "code": "recheck_requires_signed_turn"}
    task_refs = await _review_task_refs(task_id) if turn is not None else []
    payload.update(await asyncio.to_thread(resolve_review_provenance, turn, task_refs))
    if turn is None:
        try:
            return await _api_post(
                f"/tasks/{task_id}/reviews", json=payload, headers=_headers(bot_id)
            )
        except httpx.HTTPStatusError as error:
            return _http_error(error)
    internal = {**payload, "actorBotId": bot_id}
    if recheck:
        path = _internal_path(task_id, rechecks_review_id, "recheck")
        internal["resolutions"] = [_resolution_payload(item) for item in resolutions or []]
    else:
        path = _internal_path(task_id)
    try:
        return await _internal_post(path, internal)
    except InternalReviewRouteMissing:
        if recheck:
            return {"error": "This BawtHub build cannot record rechecks yet",
                    "code": "recheck_unavailable"}
        # A BawtHub build from before the internal route still records a
        # signed review on its public route, as it did before slice 2.
        logger.info("internal review route missing; using the public route for %s", task_id)
        try:
            return await _api_post(
                f"/tasks/{task_id}/reviews", json=payload, headers=_headers(bot_id)
            )
        except httpx.HTTPStatusError as error:
            return _http_error(error)
    except httpx.HTTPStatusError as error:
        return _internal_error(error)


@mcp.tool(name="tasks_reviews_resolve_finding")
async def resolve_review_finding(
    task_id: str,
    review_id: str,
    finding_id: str,
    resolution: Resolution,
    bot_id: str,
    revision: str | None = None,
    task: str | None = None,
    note: str | None = None,
) -> dict:
    """Settle one non-Blocker review finding outside a recheck. Needs a signed turn.

    Args:
        task_id: Task ID or shortId.
        review_id: Review holding the finding.
        finding_id: Finding to settle.
        resolution: FIXED needs revision, DEFERRED needs task; only the reviewer withdraws.
        bot_id: Your own bot ID.
        revision: Fix commit SHA.
        task: Existing follow-up task.
        note: Optional reason.

    Returns:
        The updated finding as returned by BawtHub, or an error dict.
    """
    if not bot_id.strip():
        return {"error": "bot_id is required for review attribution"}
    signed_arguments = {
        "task_id": task_id, "review_id": review_id, "finding_id": finding_id,
        "resolution": resolution, "bot_id": bot_id, "revision": revision,
        "task": task, "note": note,
    }
    try:
        turn = signed_review_turn(bot_id, signed_arguments, tool="tasks_reviews_resolve_finding")
    except BotAccessError as error:
        return {"error": error.code}
    if turn is None:
        return {"error": "resolve_requires_signed_turn", "code": "resolve_requires_signed_turn"}
    body: dict[str, Any] = {"actorBotId": bot_id, "resolution": resolution, "provenance": SIGNED_TURN}
    body.update(_present({"revision": revision, "task": task, "note": note}, _RESOLUTION_OPTIONAL))
    path = _internal_path(task_id, review_id, "findings", finding_id, "resolve")
    try:
        return await _internal_post(path, body)
    except InternalReviewRouteMissing:
        return {"error": "This BawtHub build cannot resolve findings yet",
                "code": "resolve_unavailable"}
    except httpx.HTTPStatusError as error:
        return _internal_error(error)
