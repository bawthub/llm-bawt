"""MCP tools for BawtHub task CRUD and dependency operations.

Imported by server.py to register the task tool group.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from .server import mcp
from .task_association import associate_current_task
from .task_step_types import TaskStepInput, validate_step_inputs
from .task_api import (
    API_PREFIX as _API_PREFIX,
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
)

logger = logging.getLogger(__name__)

async def list_projects() -> list[dict]:
    """Compatibility wrapper; the registered tool lives in project_tools."""
    try:
        return _compact_project_list_payload(await _api_get("/projects"))
    except httpx.HTTPStatusError as error:
        return _http_error(error)


@mcp.tool(name="tasks_list")
async def list_tasks(
    status: str | None = None,
    project_id: str | None = None,
    q: str | None = None,
    limit: int = 20,
) -> dict:
    """List agent tasks with optional filters.

    Use this to find tasks by status, project, or keyword search.
    Returns tasks sorted by most recently updated.

    Args:
        status: Filter by status. One of: QUEUED, PLANNING, REFINED,
                IN_PROGRESS, REVIEW, COMPLETED, FAILED, CANCELLED.
        project_id: Filter to tasks in a specific project (UUID).
                    Use "none" for unassigned tasks.
        q: Search query - matches title, description, and shortId.
        limit: Maximum tasks to return (default 20, max 50).

    Returns:
        Dict with "tasks" list and "total" count.
        Each task is a compact row with title/status metadata and counts.
        Use tasks_get or tasks_get_context for full descriptions and steps.
    """
    logger.debug("MCP tool invoked: tools/list_tasks status=%s project=%s q=%s", status, project_id, q)
    params: dict[str, str] = {"limit": str(min(limit, 50)), "sort": "updated"}
    if status:
        params["status"] = status
    if project_id:
        params["projectId"] = project_id
    if q:
        params["q"] = q
    try:
        return _compact_task_list_payload(await _api_get("/tasks", params=params))
    except httpx.HTTPStatusError as e:
        return _http_error(e)


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
        task_id: Task UUID or shortId (e.g. "TASK-42").

    Returns:
        Full task object with title, description, status, priority,
        response, steps (ordered), project info, and dependencies.
        Returns error dict if not found.
    """
    logger.debug("MCP tool invoked: tools/get_task id=%s", task_id)
    try:
        return await _api_get(f"/tasks/{task_id}")
    except httpx.HTTPStatusError as e:
        return _http_error(e)


@mcp.tool(name="tasks_associate_current")
async def associate_task_to_current_turn(task_id: str) -> dict:
    """Associate the trusted current chat session and exact turn to a task.

    Call this when you actually begin or resume work on an existing task during
    ordinary chat. Supply only the task UUID or shortId; the server obtains the
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


@mcp.tool(name="tasks_update")
async def update_task(
    task_id: str,
    status: str | None = None,
    response: str | None = None,
    model_id: str | None = None,
    title: str | None = None,
    description: str | None = None,
    priority: str | None = None,
    planned: bool | None = None,
    project_id: str | None = None,
    agent_bot_id: str | None = None,
    bot_id: str | None = None,
    associate_current_turn: bool = False,
) -> dict:
    """Update a task's fields. Only provided fields are changed.

    Common patterns:
    - Start work: status="IN_PROGRESS", model_id="claude-opus-4-6"
    - Finish work: status="REVIEW", response="Summary of what was done"
    - Report failure: status="FAILED", response="What went wrong"

    Moving a task to REVIEW requires it to have an owner. You don't need to set
    one by hand: when status="REVIEW" and you pass bot_id (your bot), it's used
    as the owner automatically. Pass agent_bot_id only to assign a different bot.

    IMPORTANT: Submit your own finished work to REVIEW by default. If Nick
    directly asks you to review and close tasks, you may set verified, complete
    REVIEW tasks to COMPLETED on his behalf; leave broken or incomplete ones
    in REVIEW and explain why. This does not permit leaving BUG status.

    BUG status is human-locked: you may SET a task to BUG (to flag a defect for
    the user), but you CANNOT move a task OUT of BUG — the server rejects such
    transitions from agents. Only a human can clear BUG status.

    Args:
        task_id: Task UUID or shortId (e.g. "TASK-42").
        status: New status (QUEUED, PLANNING, REFINED, IN_PROGRESS,
                REVIEW, COMPLETED, FAILED, BUG, CANCELLED). You may set BUG
                but cannot transition a task out of BUG (human-only).
        response: Summary text - your final answer / work output.
        model_id: Model identifier (e.g. "claude-opus-4-6").
        title: Updated task title.
        description: Updated description / spec.
        priority: URGENT, HIGH, MEDIUM, LOW, or NONE.
        planned: True after writing spec + steps.
        project_id: Move task to a different project (UUID).
        agent_bot_id: Assign task to a specific bot. Defaults to bot_id when
                      moving to REVIEW (see note above); pass explicitly only to
                      hand the task to a different bot.
        bot_id: Your bot ID for activity attribution. Also becomes the task
                owner when moving to REVIEW without an explicit agent_bot_id.
        associate_current_turn: Also link this trusted current chat turn to the
                task. Use when claiming/starting ordinary-chat work. The server
                supplies all correlation IDs; unsupported harnesses fail closed.

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
    if priority is not None:
        body["priority"] = priority
    if planned is not None:
        body["planned"] = planned
    if project_id is not None:
        body["projectId"] = project_id
    if agent_bot_id is not None:
        body["agentBotId"] = agent_bot_id

    # REVIEW requires the task to have an owner (server guard
    # reviewTransitionMissingBot). The bot submitting its finished work IS that
    # owner, and we already have its id in bot_id, so default agentBotId to it
    # when the caller didn't set one explicitly. Without this, moving a task to
    # REVIEW would 400 unless the caller happened to know the internal rule and
    # pass agent_bot_id by hand. Only fills on REVIEW and only when unset —
    # an explicit agent_bot_id always wins.
    if status == "REVIEW" and agent_bot_id is None and bot_id:
        body["agentBotId"] = bot_id

    if not body:
        if associate_current_turn:
            return await associate_task_to_current_turn(task_id)
        return {"error": "No fields to update"}

    try:
        updated = await _api_patch(
            f"/tasks/{task_id}",
            json=body,
            headers=_headers(bot_id),
        )
        if isinstance(updated, dict):
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
            return result
        return updated
    except httpx.HTTPStatusError as e:
        return _http_error(e)


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
        task_id: Task UUID or shortId (e.g. "TASK-42").
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


@mcp.tool(name="tasks_add_dependency")
async def add_task_dependency(
    task_id: str,
    depends_on_id: str,
    bot_id: str | None = None,
) -> dict:
    """Add a dependency: declare that ``task_id`` must wait for ``depends_on_id``.

    Cycles are rejected by the server. A task cannot depend on itself.

    Args:
        task_id: The dependent task (UUID or shortId). This is the task
                 that should NOT start until the dependency is done.
        depends_on_id: The prerequisite task (UUID or shortId).
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
        return await _api_post(
            f"/tasks/{task_id}/dependencies",
            json={"depId": depends_on_id},
            headers=_headers(bot_id),
        )
    except httpx.HTTPStatusError as e:
        return _http_error(e)


@mcp.tool(name="tasks_remove_dependency")
async def remove_task_dependency(
    task_id: str,
    depends_on_id: str,
    bot_id: str | None = None,
) -> dict:
    """Remove a dependency from a task.

    Args:
        task_id: The dependent task (UUID or shortId).
        depends_on_id: The prerequisite task to disconnect (UUID or shortId).
        bot_id: Your bot ID for activity attribution.

    Returns:
        Updated dependent-task object with refreshed ``dependsOn`` list.
    """
    logger.debug(
        "MCP tool invoked: tools/remove_task_dependency task=%s deps_on=%s",
        task_id,
        depends_on_id,
    )
    client = _get_client()
    url = f"{_API_PREFIX}/tasks/{task_id}/dependencies"
    try:
        # Some httpx versions disallow DELETE bodies via the helper, so build
        # the request explicitly and pass depId as both body and query string.
        resp = await client.request(
            "DELETE",
            url,
            params={"depId": depends_on_id},
            json={"depId": depends_on_id},
            headers=_headers(bot_id),
        )
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as e:
        return _http_error(e)


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
        task_id: Task UUID or shortId.
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
        task_id: Task UUID or shortId.
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
async def create_task(
    title: str,
    description: str | None = None,
    project_id: str | None = None,
    priority: str = "MEDIUM",
    status: str = "QUEUED",
    steps: list[TaskStepInput] | None = None,
    bot_id: str | None = None,
    associate_current_turn: bool = False,
) -> dict:
    """Create a new agent task.

    Use this to break work into sub-tasks or queue follow-up work.

    Args:
        title: Task title (required).
        description: Detailed description or spec.
        project_id: Assign to a project (UUID). Omit for unassigned.
        priority: URGENT, HIGH, MEDIUM, LOW, or NONE (default MEDIUM).
        status: Initial status (default QUEUED).
        steps: Optional initial steps. Each dict needs "title" (str)
               and optional "type": PLAN, READ_FILE, EDIT_FILE, CREATE_FILE,
               DELETE_FILE, RUN_COMMAND, SEARCH, ASK_USER, or REVIEW.
        bot_id: Your bot ID for activity attribution.
        associate_current_turn: Link the newly created task to this trusted
                current chat session and exact turn immediately after creation.
                The server supplies all correlation IDs.

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
    if project_id is not None:
        body["projectId"] = project_id
    if steps is not None:
        validation_error = validate_step_inputs(steps)
        if validation_error:
            return {"error": validation_error}
        body["steps"] = steps

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
        return {"task": created, "currentTurnAssociation": association}
    except httpx.HTTPStatusError as e:
        return _http_error(e)
