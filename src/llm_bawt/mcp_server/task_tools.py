"""MCP tools for BawtHub task CRUD and dependency operations.

Imported by server.py to register the task tool group.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Annotated, Any, Literal

import httpx
from pydantic import Field

from llm_bawt.bot_access import BotAccessError

from .review_provenance import resolve_review_provenance, signed_review_turn
from .server import mcp
from .task_association import associate_current_task
from .task_board_state import (
    _ARCHIVE,
    _FOCUS,
    FocusAction,
    FocusPosition,
    _focus_patch,
    _patch_error,
)
from .task_images import (
    UNSUPPORTED as IMAGE_EDIT_UNSUPPORTED,
    ImageRefError,
    image_edit,
    linked_turn_images,
    newly_added,
    resolve_image_refs,
)
from .task_step_types import (
    BLOCKED_DEFINITION,
    CHECK_LINKS_DOC,
    TASK_STATUSES,
    CheckLinkInput,
    TaskStepInput,
    validate_step_inputs,
)
from .task_api import (
    API_PREFIX as _API_PREFIX,
    NotJsonResponse,
    api_delete as _api_delete,
    api_get as _api_get,
    api_patch as _api_patch,
    api_post as _api_post,
    compact_project_list_payload as _compact_project_list_payload,
    compact_task as _compact_task,
    compact_task_list_payload as _compact_task_list_payload,
    get_client as _get_client,
    headers as _headers,
    http_error as _http_error,
    json_body as _json_body,
)

logger = logging.getLogger(__name__)

async def list_projects() -> list[dict] | dict:
    """Compatibility wrapper; the registered tool lives in project_tools."""
    try:
        return _compact_project_list_payload(await _api_get("/projects"))
    except httpx.HTTPStatusError as error:
        return _http_error(error)


def _task_status_docs(function):
    """Render the shared enum before FastMCP captures the function docstring."""
    function.__doc__ = (
        (function.__doc__ or "")
        .replace("{task_statuses}", ", ".join(TASK_STATUSES))
        .replace("{blocked_definition}", BLOCKED_DEFINITION)
        .replace("{check_links_doc}", CHECK_LINKS_DOC)
    )
    return function


@mcp.tool(name="tasks_list")
@_task_status_docs
async def list_tasks(
    status: str | None = None,
    project_id: str | None = None,
    q: str | None = None,
    limit: int = 20,
    focused: bool | None = None,
    archived: bool | None = None,
) -> dict:
    """List agent tasks with optional filters.

    Use this to find tasks by status, project, or keyword search.
    Returns tasks sorted by most recently updated, except focused=true, which
    returns the user's focus list in rank order (focusRank 1 first).
    Archived tasks are included unless archived=False.

    Args:
        status: One task status: {task_statuses}. No comma lists.
                Omitted or empty filter excludes DELETED.
        project_id: Filter to tasks in a specific project (cuid).
                    Use "none" for unassigned tasks.
        q: Search query - matches title, description, and shortId.
        limit: Maximum tasks to return (default 20, max 50).
        focused: True for focused tasks only, in focus order; False for
                 unfocused tasks only (the backlog). Combines with the other
                 filters. Change focus with tasks_update(focus=...).
        archived: True for archived (finished, off the board) tasks only;
                  False to leave them out. Change it with
                  tasks_update(archived=...).

    Returns:
        Dict with "tasks" list and "total" count.
        Each task is a compact row with title/status metadata and counts;
        focused tasks also carry focusRank and focusedAt, archived tasks
        archivedAt.
        Use tasks_get or tasks_get_context for full descriptions and steps.
    """
    logger.debug(
        "MCP tool invoked: tools/list_tasks status=%s project=%s q=%s focused=%s archived=%s",
        status, project_id, q, focused, archived,
    )
    params: dict[str, str] = {
        "limit": str(min(limit, 50)),
        "sort": "focus" if focused else "updated",
    }
    if status:
        params["status"] = status
    if project_id:
        params["projectId"] = project_id
    if q:
        params["q"] = q
    filters = {
        param: (state, value)
        for param, state, value in (("focused", _FOCUS, focused), ("archived", _ARCHIVE, archived))
        if value is not None
    }
    for param, (_state, value) in filters.items():
        params[param] = "true" if value else "false"
    try:
        payload = await _api_get("/tasks", params=params)
    except httpx.HTTPStatusError as e:
        return _http_error(e)
    for param, (state, value) in filters.items():
        if not state.filter_applied(payload, value):
            # A BawtHub without the state ignores the filter and returns every
            # task; never present that as the filtered list.
            return {"error": f"{state.unsupported}; the {param}={str(value).lower()} list was not read."}
    return _compact_task_list_payload(payload)


@mcp.tool(name="tasks_board")
async def tasks_board(
    view: str | None = None,
    updated: str | None = None,
    bot: str | None = None,
    q: str | None = None,
    project: str | None = None,
    priority: str | None = None,
    limit: int = 20,
) -> dict:
    """The task board as data: the lanes BawtHub's /tasks/board lays out.

    TASK-1460: a thin forwarder to BawtHub's GET /api/tasks/board, which lays
    the board out with the page's own lane rules, so the same filters give
    the same lanes, cards and counts as the page with every column shown.
    Hiding a column is a browser preference the endpoint never sees, so a
    column the page hides by default (the status view's Cancelled) is
    included. The args are the board URL's params; unknown view/updated
    values fall back to the board's defaults, and the reply echoes the ones
    applied.

    Args:
        view: "kanban" (default) or "status".
        updated: The Updated time filter: 1h, 6h, 24h, 48h, 7d (default),
                 30d or all. Applies to windowed lanes only.
        bot: Assigned bot slug.
        q: Search over TASK-ID, title, project name and bot.
        project: Project id (cuid), or "none" for unassigned tasks.
        priority: One priority, e.g. HIGH.
        limit: Rows per lane (default 20, max 50); counts stay whole.

    Returns:
        Dict with view, updated, q, filters, now and lanes: every column in
        order, each followed by the drawers it shows (`in` names the
        column). Each lane has id, label, windowed, shown, older (left out by
        the time filter) and compact rows (shortId, title, status, priority,
        agentBotId, focusRank, archivedAt, updatedAt, blockedReason).
        Use tasks_get for a task's description and steps.
    """
    params = {
        key: value
        for key, value in (
            ("view", view), ("updated", updated), ("bot", bot),
            ("q", q), ("project", project), ("priority", priority),
        )
        if value
    }
    params["limit"] = str(max(1, min(limit, 50)))
    try:
        return await _api_get("/board", params=params)
    except httpx.HTTPStatusError as e:
        return _http_error(e)
    except NotJsonResponse as e:
        return {"error": str(e)}


@mcp.tool(name="tasks_search_semantic")
async def search_tasks_semantic(
    query: str,
    limit: int = 20,
) -> dict:
    """Find tasks by meaning, not just keywords (semantic / vector search).

    Ranks tasks by how close their content is to your query using embeddings,
    so "auth is broken" surfaces a task titled "login returns 500" even with no
    shared words. Prefer this over ``tasks_list(q=...)`` when you're looking for
    related work by concept; use ``tasks_list`` when you need exact status/project
    filters or a literal substring match.

    Searches across BOTH the task's title+description and its completed-work
    response, returning whichever is the closer match. If the embedding service
    is unavailable it transparently degrades to a keyword scan (``mode`` tells
    you which ran).

    Args:
        query: Natural-language description of what you're looking for.
        limit: Maximum tasks to return (default 20, max 100).

    Returns:
        Dict with:
          - ``mode``: "semantic" (vector search) or "keyword" (fallback).
          - ``results``: ranked list of compact task rows, each with id,
            shortId, title, description, status, priority, projectId, and
            ``score`` (cosine similarity in [0,1], higher = closer; 0 for
            keyword-fallback rows).
        Use tasks_get for full description, response, and steps.
    """
    logger.debug("MCP tool invoked: tools/search_tasks_semantic q=%s limit=%s", query, limit)
    if not query or not query.strip():
        return {"error": "query is required"}
    body = {"query": query, "limit": min(limit, 100)}
    try:
        return await _api_post("/tasks/search/semantic", json=body)
    except httpx.HTTPStatusError as e:
        return _http_error(e)


@mcp.tool(name="tasks_get")
async def get_task(
    task_id: str,
) -> dict:
    """Get full details of a single task by ID or shortId.

    Args:
        task_id: Task ID (cuid) or shortId (e.g. "TASK-42").

    Returns:
        Full task object with title, description, status, priority,
        response, checkLinks, steps (ordered), project info, and dependencies.
        Returns error dict if not found.
    """
    logger.debug("MCP tool invoked: tools/get_task id=%s", task_id)
    try:
        return await _api_get(f"/tasks/{task_id}")
    except httpx.HTTPStatusError as e:
        return _http_error(e)


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


@mcp.tool(name="tasks_reviews_add")
async def add_task_review(
    task_id: str,
    stage: Literal["DESIGN", "IMPLEMENTATION"],
    verdict: Literal["APPROVE", "APPROVE_WITH_CHANGES", "REQUEST_CHANGES", "COMMENT"],
    body: str,
    bot_id: str,
    target_revision: str | None = None,
    idempotency_key: str | None = None,
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
    # TASK-1249: models come from the caller's signed turn, never from
    # arguments or the body. A bad or foreign credential is refused.
    try:
        turn = signed_review_turn(bot_id, {
            "task_id": task_id, "stage": stage, "verdict": verdict, "body": body,
            "bot_id": bot_id, "target_revision": target_revision,
            "idempotency_key": idempotency_key,
        })
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


@mcp.tool(name="tasks_associate_current")
async def associate_task_to_current_turn(task_id: str) -> dict:
    """Associate the trusted current chat session and exact turn to a task.

    Call this when you actually begin or resume work on an existing task during
    ordinary chat. Supply only the task ID (cuid) or shortId; the server obtains the
    session, turn, trigger-message, bot, and user identifiers from a signed
    current-turn capability. Never infer association merely because TASK-N text
    appears in a prompt, quote, or negative reference.

    This capability is currently available on Claude SDK/proxy turns. Calls from
    a harness without trusted request-local MCP headers fail closed.
    """
    logger.debug("MCP tool invoked: tasks_associate_current id=%s", task_id)
    try:
        return await associate_current_task(task_id)
    except httpx.HTTPStatusError as error:
        return _http_error(error)
    except ValueError as error:
        return {"error": str(error)}


async def _task_is_unowned(task_id: str) -> bool:
    """True only when the task is confirmed to have no owner (fail closed)."""
    try:
        current = await _api_get(f"/tasks/{task_id}")
    except (httpx.HTTPError, ValueError) as error:
        logger.warning("tasks_update: owner lookup for %s failed: %s", task_id, error)
        return False
    return isinstance(current, dict) and not current.get("agentBotId")


@mcp.tool(name="tasks_update")
@_task_status_docs
async def update_task(
    task_id: str,
    status: str | None = None,
    response: str | None = None,
    model_id: str | None = None,
    title: str | None = None,
    description: str | None = None,
    outcome: str | None = None,
    priority: str | None = None,
    planned: bool | None = None,
    project_id: str | None = None,
    agent_bot_id: str | None = None,
    bot_id: str | None = None,
    associate_current_turn: bool = False,
    blocked_reason: str | None = None,
    check_links: list[CheckLinkInput] | None = None,
    focus: Annotated[
        FocusAction | None,
        Field(description="User's focus list: on (default bottom; a focused task stays put) or off."),
    ] = None,
    focus_position: Annotated[
        FocusPosition | None,
        Field(description="Place in the focus list; implies focus on."),
    ] = None,
    focus_anchor: Annotated[
        str | None,
        Field(description="Task cuid/TASK-N for before/after; must be focused."),
    ] = None,
    archived: Annotated[
        bool | None,
        Field(description="True takes a finished task off the board; false restores it."),
    ] = None,
    add_image_urls: list[str] | None = None,
    remove_image_urls: list[str] | None = None,
    attach_turn_images: bool = True,
) -> dict:
    """Update a task's fields. Only provided fields are changed.

    Common patterns:
    - Start work: status="IN_PROGRESS", model_id="claude-opus-4-6"
    - Visually confirmable UI/UX: status="REVIEW", check_links=[...],
      outcome="Plain-language change"
    - Other finished work: record review (Tier 0: self-check), attributable
      pushed commit and test/live check in response; Snark or Rook closes
      status="COMPLETED" after verifying those receipts.
    - Report failure: status="FAILED", response="What went wrong"

    Images: attach every relevant image from the conversation (screenshots,
    mockups, error captures). Images reach every bot that works the task;
    don't make them re-capture with Playwright. Linking the current turn
    attaches that message's own images automatically.

    Moving a task to REVIEW requires it to have an owner. You don't need to set
    one by hand: when status="REVIEW", the task has no owner, and you pass
    bot_id (your bot), it becomes the owner. bot_id never reassigns an owned
    task; pass agent_bot_id only to deliberately hand it to a different bot.

    IMPORTANT: REVIEW is only for UI/UX the user can visually inspect, and
    needs check_links. For backend, infra, tooling
    and skills, Snark or Rook verifies the review (Tier 0: self-check), pushed
    attributable commit and test/live receipt before closing COMPLETED,
    including their own reviewed work. Leave incomplete work open with its
    blocker recorded. No agent may move a task out of BUG status.

    BUG status is human-locked: you may SET a task to BUG (to flag a defect for
    the user), but you CANNOT move a task OUT of BUG — the server rejects such
    transitions from agents. Only a human can clear BUG status.

    {blocked_definition} Example: status="BLOCKED", blocked_reason="Approve
    the shared-DB migration, then ship." While BLOCKED, blocked_reason alone
    updates the reason.

    Focus is the user's ordered list of work they chose to get done. When
    they say "focus X", "put Y first" or "drop Z", set it here in one call:
    any bot or user may, and BawtHub records who did. Examples: focus="on" (append at the
    bottom), focus_position="top", focus_position="after",
    focus_anchor="TASK-42", focus="off". A task that is or becomes
    COMPLETED/FAILED/CANCELLED/DELETED can't be focused (409); the server also
    clears focus when a task finishes. A 409 includes "focused", the current
    order, so you can re-anchor. Read the list with tasks_list(focused=True).

    Archive takes Done tasks off the board when the user says "archive the
    done tasks": archived=True works only when the task is or becomes
    COMPLETED/FAILED/CANCELLED (otherwise 400; 409 if its status changed
    meanwhile, so re-read before retrying); reopening or deleting a task
    unarchives it. Archived tasks stay in tasks_list unless archived=False.

    Args:
        task_id: Task ID (cuid) or shortId (e.g. "TASK-42").
        status: New task status: {task_statuses}. You may set BUG,
                but cannot transition a task out of BUG (human-only).
        response: Summary text - your final answer / work output.
        model_id: Model identifier (e.g. "claude-opus-4-6").
        title: Updated task title.
        description: Updated description / spec.
        outcome: Plain-language behavior change for a non-engineer (1-3
                 sentences, no file names/checklists). Draft when planning;
                 overwrite with what actually shipped at REVIEW/COMPLETED.
        priority: URGENT, HIGH, MEDIUM, LOW, or NONE.
        planned: True after writing spec + steps.
        project_id: Move task to a different project (cuid).
        agent_bot_id: Assign task to a specific bot. Defaults to bot_id when
                      moving an UNOWNED task to REVIEW; pass explicitly only to
                      hand the task to a different bot.
        bot_id: YOUR bot ID (the actor) for activity attribution. Never pass
                another bot's id. Fills the owner only for an unowned task
                moving to REVIEW.
        associate_current_turn: Also link this trusted current chat turn to the
                task. Use when claiming/starting ordinary-chat work. The server
                supplies all correlation IDs; unsupported harnesses fail closed.
        blocked_reason: What Nick must do or decide. Required with
                        status="BLOCKED"; rejected unless the task is or becomes
                        BLOCKED. Cleared automatically when the task leaves BLOCKED.
        check_links: {check_links_doc}
        focus: "on" or "off". "on" without a position puts an unfocused task
               at the bottom and leaves a focused task where it is.
        focus_position: "top", "bottom", "before" or "after"; implies "on".
        focus_anchor: Task (cuid or TASK-N) to place before/after; it must
                      be focused itself.
        archived: True archives a finished task; False unarchives it.
                  Combines with status in the same update.
        add_image_urls: Images to attach: chat upload ids (ma_...) or upload
                  URLs. Unknown ids are refused. Already attached = no-op.
        remove_image_urls: Images to detach, same forms. Missing = no-op.
                  Only the change is sent, so images someone else attaches
                  meanwhile are kept.
        attach_turn_images: With associate_current_turn, also attach the
                  images on the message that started this turn (default True;
                  listed in turnImagesAttached). False opts out.

    Returns:
        Compact updated task summary, or error dict. Use tasks_get for full detail.
    """
    logger.debug("MCP tool invoked: tools/update_task id=%s status=%s", task_id, status)
    body: dict[str, Any] = {}
    if status is not None:
        body["status"] = status
    if response is not None:
        body["response"] = response
    if model_id is not None:
        body["modelId"] = model_id
    if title is not None:
        body["title"] = title
    if description is not None:
        body["description"] = description
    if outcome is not None:
        body["outcome"] = outcome
    if priority is not None:
        body["priority"] = priority
    if planned is not None:
        body["planned"] = planned
    if project_id is not None:
        body["projectId"] = project_id
    if agent_bot_id is not None:
        body["agentBotId"] = agent_bot_id
    if blocked_reason is not None:
        body["blockedReason"] = blocked_reason
    if check_links is not None:
        body["checkLinks"] = check_links
    focus_body, focus_error = _focus_patch(focus, focus_position, focus_anchor)
    if focus_error:
        return {"error": focus_error}
    if focus_body is not None:
        body["focus"] = focus_body
    if archived is not None:
        body["archived"] = archived

    # REVIEW requires the task to have an owner (server guard
    # reviewTransitionMissingBot). When the task has NO owner, the submitting
    # bot (bot_id) fills it so the caller needn't know the internal rule.
    # TASK-1015: bot_id is the ACTOR, never an ownership claim — it must not
    # reassign a task that already has an owner (TASK-1013 became Al's when
    # Caid passed bot_id="al"). Reassignment requires explicit agent_bot_id.
    if status == "REVIEW" and agent_bot_id is None and bot_id:
        if await _task_is_unowned(task_id):
            body["agentBotId"] = bot_id

    turn_urls, turn_error = await linked_turn_images(associate_current_turn, attach_turn_images)
    edit = None
    if add_image_urls or remove_image_urls or turn_urls:
        try:
            edit = await image_edit(lambda ref: _api_get(f"/tasks/{ref}"), task_id,
                                    add_image_urls, remove_image_urls, turn_urls)
        except ImageRefError as error:
            return {"error": str(error)}
        except httpx.HTTPStatusError as error:
            return _http_error(error)
        if edit is not None:
            body.update(edit.body())

    if not body:
        if associate_current_turn:
            return _turn_images(await associate_task_to_current_turn(task_id), [], turn_error)
        return {"error": "No fields to update"}

    try:
        updated = await _api_patch(
            f"/tasks/{task_id}",
            json=body,
            headers=_headers(bot_id),
        )
        if isinstance(updated, dict):
            unconfirmed = [
                f"{state.unsupported}; {state.name} was not changed" for state in (_FOCUS, _ARCHIVE)
                if state.patch_key in body and state.row_key not in updated
            ]
            skipped = {state.patch_key for state in (_FOCUS, _ARCHIVE) if state.row_key not in updated}
            images_landed = edit is None or edit.landed(updated)
            if not images_landed:
                unconfirmed.append(IMAGE_EDIT_UNSUPPORTED)
                skipped |= {"imageUrlsAdd", "imageUrlsRemove"}
            turn_attached = [url for url in turn_urls if url in edit.fresh] if edit and images_landed else []
            if unconfirmed:
                # An older BawtHub ignores the unknown field but applies the
                # rest of the PATCH. Say exactly that instead of reporting OK.
                applied = sorted(key for key in body if key not in skipped)
                result = {
                    "error": "; ".join(unconfirmed)
                    + (", the other fields were applied." if applied else "."),
                    "updated": applied,
                    "task": _compact_task(updated),
                }
            else:
                result = {
                    "ok": True,
                    "updated": sorted(body.keys()),
                    "task": _compact_task(updated),
                    "detail": "Use tasks_get for full task description, response, steps, and dependencies.",
                }
            if associate_current_turn:
                try:
                    task_ref = str(updated.get("shortId") or task_id)
                    result["currentTurnAssociation"] = await associate_current_task(task_ref)
                except (httpx.HTTPStatusError, ValueError) as error:
                    result["currentTurnAssociation"] = (
                        _http_error(error)
                        if isinstance(error, httpx.HTTPStatusError)
                        else {"error": str(error)}
                    )
            return _turn_images(result, turn_attached, turn_error)
        return updated
    except httpx.HTTPStatusError as e:
        return _patch_error(e)


def _turn_images(result: Any, attached: list[str], error: str | None) -> Any:
    """Name the linked turn's images this call attached, so a bot can drop an irrelevant one."""
    if isinstance(result, dict):
        if attached:
            result["turnImagesAttached"] = attached
        if error:
            result["turnImagesError"] = error
    return result


@mcp.tool(name="tasks_delete")
async def delete_task(
    task_id: str,
    bot_id: str | None = None,
) -> dict:
    """Delete an agent task.

    WARNING: This permanently removes the task and all of its steps.
    Prefer marking tasks as CANCELLED via tasks_update unless you're
    cleaning up scratch / duplicate / clearly-bogus entries.

    Args:
        task_id: Task ID (cuid) or shortId (e.g. "TASK-42").
        bot_id: Your bot ID for activity attribution.

    Returns:
        Confirmation dict, or error dict.
    """
    logger.debug("MCP tool invoked: tools/delete_task id=%s", task_id)
    try:
        return await _api_delete(
            f"/tasks/{task_id}",
            headers=_headers(bot_id),
        )
    except httpx.HTTPStatusError as e:
        return _http_error(e)


async def _dependency_task_id(task_ref: str) -> str:
    """Resolve only short refs; the dependency API requires an actual task ID."""
    if not task_ref.startswith("TASK-"):
        return task_ref
    task = await _api_get(f"/tasks/{task_ref}")
    task_id = task.get("id") if isinstance(task, dict) else None
    if not isinstance(task_id, str) or not task_id:
        raise ValueError(f"Task lookup for {task_ref} returned no task ID")
    return task_id


@mcp.tool(name="tasks_add_dependency")
async def add_task_dependency(
    task_id: str,
    depends_on_id: str,
    bot_id: str | None = None,
) -> dict:
    """Add a dependency: declare that ``task_id`` must wait for ``depends_on_id``.

    Cycles are rejected by the server. A task cannot depend on itself.

    Args:
        task_id: The dependent task (cuid or TASK-N). This is the task
                 that should NOT start until the dependency is done.
        depends_on_id: The prerequisite task (cuid or TASK-N).
        bot_id: Your bot ID for activity attribution.

    Returns:
        Updated dependent-task object with refreshed ``dependsOn`` list,
        or error dict on cycle/missing-target.
    """
    logger.debug(
        "MCP tool invoked: tools/add_task_dependency task=%s deps_on=%s",
        task_id,
        depends_on_id,
    )
    try:
        resolved_task_id = await _dependency_task_id(task_id)
        resolved_dep_id = await _dependency_task_id(depends_on_id)
        return await _api_post(
            f"/tasks/{resolved_task_id}/dependencies",
            json={"depId": resolved_dep_id},
            headers=_headers(bot_id),
        )
    except httpx.HTTPStatusError as e:
        return _http_error(e)
    except ValueError as e:
        return {"error": str(e)}


@mcp.tool(name="tasks_remove_dependency")
async def remove_task_dependency(
    task_id: str,
    depends_on_id: str,
    bot_id: str | None = None,
) -> dict:
    """Remove a dependency from a task.

    Args:
        task_id: The dependent task (cuid or TASK-N).
        depends_on_id: The prerequisite task to disconnect (cuid or TASK-N).
        bot_id: Your bot ID for activity attribution.

    Returns:
        Updated dependent-task object with refreshed ``dependsOn`` list.
    """
    logger.debug(
        "MCP tool invoked: tools/remove_task_dependency task=%s deps_on=%s",
        task_id,
        depends_on_id,
    )
    try:
        resolved_task_id = await _dependency_task_id(task_id)
        resolved_dep_id = await _dependency_task_id(depends_on_id)
        client = _get_client()
        url = f"{_API_PREFIX}/tasks/{resolved_task_id}/dependencies"
        # Some httpx versions disallow DELETE bodies via the helper, so build
        # the request explicitly and pass depId as both body and query string.
        resp = await client.request(
            "DELETE",
            url,
            params={"depId": resolved_dep_id},
            json={"depId": resolved_dep_id},
            headers=_headers(bot_id),
        )
        return _json_body(resp)
    except httpx.HTTPStatusError as e:
        return _http_error(e)
    except ValueError as e:
        return {"error": str(e)}


@mcp.tool(name="tasks_promote")
async def promote_task(
    task_id: str,
    bot_id: str | None = None,
) -> dict:
    """Promote a task into its own project.

    Creates a new project named after the task title (truncated to 100
    chars) using the task description as the project context, then
    re-parents the task into the new project. Useful when a task grows
    into a long-running effort with subtasks.

    Args:
        task_id: Task ID (cuid) or shortId.
        bot_id: Your bot ID for activity attribution.

    Returns:
        The newly created project object, or error dict.
    """
    logger.debug("MCP tool invoked: tools/promote_task id=%s", task_id)
    try:
        return await _api_post(
            f"/tasks/{task_id}/promote",
            json={},
            headers=_headers(bot_id),
        )
    except httpx.HTTPStatusError as e:
        return _http_error(e)


@mcp.tool(name="tasks_regenerate")
async def regenerate_task(
    task_id: str,
    bot_id: str | None = None,
) -> dict:
    """Server-side LLM regeneration of a task's title and steps.

    The server calls Anthropic with the task description (and any attached
    images) to generate a fresh title and a 3-8 step plan, then replaces
    the existing steps. RARELY USEFUL FOR AGENTS — you are already an LLM
    and can write better steps yourself via tasks_update + steps_add. This
    tool exists for parity with the human UI's "regenerate" button.

    Args:
        task_id: Task ID (cuid) or shortId.
        bot_id: Your bot ID for activity attribution.

    Returns:
        Updated task object with regenerated title + steps, or error dict.
    """
    logger.debug("MCP tool invoked: tools/regenerate_task id=%s", task_id)
    try:
        return await _api_post(
            f"/tasks/{task_id}/regenerate",
            json={},
            headers=_headers(bot_id),
        )
    except httpx.HTTPStatusError as e:
        return _http_error(e)


@mcp.tool(name="tasks_create")
@_task_status_docs
async def create_task(
    title: str,
    description: str | None = None,
    outcome: str | None = None,
    project_id: str | None = None,
    priority: str = "MEDIUM",
    status: str = "QUEUED",
    steps: list[TaskStepInput] | None = None,
    bot_id: str | None = None,
    associate_current_turn: bool = False,
    blocked_reason: str | None = None,
    image_urls: list[str] | None = None,
    attach_turn_images: bool = True,
) -> dict:
    """Create a new agent task.

    Use this to break work into sub-tasks or queue follow-up work.

    Images: attach every relevant image from the conversation (screenshots,
    mockups, error captures). Images reach every bot that works the task;
    don't make them re-capture with Playwright. Linking the current turn
    attaches that message's own images automatically.

    {blocked_definition}

    Args:
        title: Task title (required).
        description: Detailed description or spec.
        outcome: Intended plain-language behavior change for a non-engineer
                 (1-3 sentences). Optional; can be filled in later.
        project_id: Assign to a project (cuid). Omit for unassigned.
        priority: URGENT, HIGH, MEDIUM, LOW, or NONE (default MEDIUM).
        status: Initial status (default QUEUED): {task_statuses}.
        steps: Optional initial steps. Each dict needs "title" (str)
               and optional "type": PLAN, READ_FILE, EDIT_FILE, CREATE_FILE,
               DELETE_FILE, RUN_COMMAND, SEARCH, ASK_USER, or REVIEW.
        bot_id: Your bot ID for activity attribution.
        associate_current_turn: Link the newly created task to this trusted
                current chat session and exact turn immediately after creation.
                The server supplies all correlation IDs.
        blocked_reason: What Nick must do or decide; required when
                        status="BLOCKED" and rejected otherwise.
        image_urls: Images to attach: chat upload ids (ma_...) or upload URLs.
                    Unknown ids are refused.
        attach_turn_images: With associate_current_turn, also attach the
                    images on the message that started this turn (default
                    True; listed in turnImagesAttached). False opts out.

    Returns:
        Created task with generated shortId and, when requested, an association result.
    """
    logger.debug("MCP tool invoked: tools/create_task title=%s", title)
    body: dict[str, Any] = {
        "title": title,
        "priority": priority,
        "status": status,
    }
    if description is not None:
        body["description"] = description
    if outcome is not None:
        body["outcome"] = outcome
    if project_id is not None:
        body["projectId"] = project_id
    if blocked_reason is not None:
        body["blockedReason"] = blocked_reason
    if steps is not None:
        validation_error = validate_step_inputs(steps)
        if validation_error:
            return {"error": validation_error}
        body["steps"] = steps
    try:
        explicit = await resolve_image_refs(image_urls or [])
    except ImageRefError as error:
        return {"error": str(error)}
    turn_urls, turn_error = await linked_turn_images(associate_current_turn, attach_turn_images)
    turn_attached = newly_added(turn_urls, explicit)
    if explicit or turn_attached:
        body["imageUrls"] = [*explicit, *turn_attached]

    try:
        created = await _api_post("/tasks", json=body, headers=_headers(bot_id))
        if not associate_current_turn or not isinstance(created, dict):
            return created
        task_ref = str(created.get("shortId") or created.get("id") or "").strip()
        if not task_ref:
            return {
                "task": created,
                "currentTurnAssociation": {
                    "error": "Task was created but its response had no task identifier",
                },
            }
        try:
            association = await associate_current_task(task_ref)
        except (httpx.HTTPStatusError, ValueError) as error:
            association = (
                _http_error(error)
                if isinstance(error, httpx.HTTPStatusError)
                else {"error": str(error)}
            )
        return _turn_images({"task": created, "currentTurnAssociation": association}, turn_attached, turn_error)
    except httpx.HTTPStatusError as e:
        return _http_error(e)
