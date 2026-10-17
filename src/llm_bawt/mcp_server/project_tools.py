"""MCP tools for projects, activity, and formatted task context.

Imported by server.py to register the project/context tool group.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from .server import mcp
from .task_api import (
    API_PREFIX as _API_PREFIX,
    api_delete as _api_delete,
    api_get as _api_get,
    api_patch as _api_patch,
    api_post as _api_post,
    compact_project_list_payload as _compact_project_list_payload,
    get_client as _get_client,
    headers as _headers,
    http_error as _http_error,
)

logger = logging.getLogger(__name__)

@mcp.tool(name="projects_list")
async def list_projects() -> list[dict]:
    """List all agent projects with task counts.

    Returns projects sorted by most recently updated.
    Use get_project() for full details including tasks and context.

    Returns:
        Compact project rows with id, name, agentBotId, taskCount, and metadata.
        Use projects_get for full context prompt and tasks.
    """
    logger.debug("MCP tool invoked: tools/list_projects")
    try:
        return _compact_project_list_payload(await _api_get("/projects"))
    except httpx.HTTPStatusError as e:
        return _http_error(e)


@mcp.tool(name="projects_get")
async def get_project(
    project_id: str,
) -> dict:
    """Get a project's details including context prompt and all tasks.

    The contextPrompt contains project-specific instructions and
    conventions. Read this before working on tasks in the project.

    Args:
        project_id: Project ID (cuid).

    Returns:
        Project with name, description, contextPrompt, tasks
        (with steps and dependencies), and configuration.
    """
    logger.debug("MCP tool invoked: tools/get_project id=%s", project_id)
    try:
        return await _api_get(f"/projects/{project_id}")
    except httpx.HTTPStatusError as e:
        return _http_error(e)


@mcp.tool(name="projects_create")
async def create_project(
    name: str,
    description: str | None = None,
    color: str = "#3b82f6",
    icon: str = "layers",
    context_prompt: str | None = None,
    agent_bot_id: str | None = None,
    bot_id: str | None = None,
) -> dict:
    """Create a new agent project.

    Projects group related tasks and carry a context prompt that agents
    read before working on tasks in the project.

    Args:
        name: Project name (required).
        description: Short description of the project's purpose.
        color: Hex color for the project badge (default "#3b82f6").
        icon: Lucide icon name for the project (default "layers").
        context_prompt: Instructions and conventions for agents working
                        on tasks in this project. Agents load this
                        automatically via get_task_context.
        agent_bot_id: Default bot assigned to new tasks in this project.
        bot_id: Your bot ID for activity attribution.

    Returns:
        Created project object with id, name, description, color,
        icon, contextPrompt, and agentBotId.
    """
    logger.debug("MCP tool invoked: tools/create_project name=%s", name)
    body: dict[str, Any] = {
        "name": name,
        "color": color,
        "icon": icon,
    }
    if description is not None:
        body["description"] = description
    if context_prompt is not None:
        body["contextPrompt"] = context_prompt
    if agent_bot_id is not None:
        body["agentBotId"] = agent_bot_id

    try:
        return await _api_post("/projects", json=body, headers=_headers(bot_id))
    except httpx.HTTPStatusError as e:
        return _http_error(e)


@mcp.tool(name="projects_update")
async def update_project(
    project_id: str,
    name: str | None = None,
    description: str | None = None,
    color: str | None = None,
    icon: str | None = None,
    context_prompt: str | None = None,
    agent_bot_id: str | None = None,
    bot_id: str | None = None,
) -> dict:
    """Update an existing project. Only provided fields are changed.

    Args:
        project_id: Project ID (cuid) (required).
        name: Updated project name.
        description: Updated description.
        color: Updated hex color (e.g. "#ef4444").
        icon: Updated Lucide icon name (e.g. "folder", "code").
        context_prompt: Updated instructions for agents working on
                        tasks in this project.
        agent_bot_id: Updated default bot for new tasks.
        bot_id: Your bot ID for activity attribution.

    Returns:
        Updated project object, or error dict.
    """
    logger.debug("MCP tool invoked: tools/update_project id=%s", project_id)
    body: dict[str, Any] = {}
    if name is not None:
        body["name"] = name
    if description is not None:
        body["description"] = description
    if color is not None:
        body["color"] = color
    if icon is not None:
        body["icon"] = icon
    if context_prompt is not None:
        body["contextPrompt"] = context_prompt
    if agent_bot_id is not None:
        body["agentBotId"] = agent_bot_id

    if not body:
        return {"error": "No fields to update"}

    try:
        return await _api_patch(
            f"/projects/{project_id}",
            json=body,
            headers=_headers(bot_id),
        )
    except httpx.HTTPStatusError as e:
        return _http_error(e)


@mcp.tool(name="projects_get_context")
async def get_project_context(
    project_id: str,
) -> str:
    """Get a project's context as a plain-text markdown briefing.

    Returns the project name and contextPrompt formatted as readable
    markdown. Use this for a lightweight read of project conventions
    when you don't need the full task list (which projects_get returns).

    Args:
        project_id: Project ID (cuid).

    Returns:
        Markdown text with the project name as an H1 heading and the
        contextPrompt under a "## Context" section. Returns an error
        string if the project is not found.
    """
    logger.debug("MCP tool invoked: tools/get_project_context id=%s", project_id)
    client = _get_client()
    url = f"{_API_PREFIX}/projects/{project_id}/context"
    try:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.text
    except httpx.HTTPStatusError as e:
        return f"Error fetching project context: HTTP {e.response.status_code}"


@mcp.tool(name="projects_delete")
async def delete_project(
    project_id: str,
    bot_id: str | None = None,
) -> dict:
    """Delete an agent project.

    WARNING: This permanently deletes the project. Tasks in the project
    are NOT deleted — they become unassigned.

    Args:
        project_id: Project ID (cuid) (required).
        bot_id: Your bot ID for activity attribution.

    Returns:
        Confirmation dict {"ok": true}, or error dict.
    """
    logger.debug("MCP tool invoked: tools/delete_project id=%s", project_id)
    try:
        return await _api_delete(
            f"/projects/{project_id}",
            headers=_headers(bot_id),
        )
    except httpx.HTTPStatusError as e:
        return _http_error(e)


# ---------------------------------------------------------------------------
# Activity Tool
# ---------------------------------------------------------------------------


@mcp.tool(name="activity_get")
async def get_activity(
    task_id: str | None = None,
    project_id: str | None = None,
    limit: int = 20,
) -> dict:
    """Get recent activity log entries.

    Shows what has happened on tasks/projects — status changes,
    dispatches, assignments, step updates, etc.

    Args:
        task_id: Filter to a specific task ID (cuid; not shortId).
        project_id: Filter to a specific project ID (cuid).
        limit: Maximum entries to return (default 20, max 100).

    Returns:
        Dict with "activities" list and "total" count.
        Each entry has type, actorType, actorId, meta, createdAt,
        and related task/project summaries.
    """
    logger.debug("MCP tool invoked: tools/get_activity task=%s project=%s", task_id, project_id)
    params: dict[str, str] = {"limit": str(min(limit, 100))}
    if task_id:
        params["taskId"] = task_id
    if project_id:
        params["projectId"] = project_id
    try:
        return await _api_get("/activity", params=params)
    except httpx.HTTPStatusError as e:
        return _http_error(e)


# ---------------------------------------------------------------------------
# Context Tool
# ---------------------------------------------------------------------------


@mcp.tool(name="tasks_get_context")
async def get_task_context(
    task_id: str,
) -> str:
    """Get a formatted briefing document for a task.

    Combines task details, step checklist, dependencies, and the
    parent project's context prompt into a single readable document.
    Load this before starting work on a task.

    Args:
        task_id: Task ID (cuid) or shortId (e.g. "TASK-42").

    Returns:
        Formatted markdown text with everything you need to work
        on the task. Returns error string if not found.
    """
    logger.debug("MCP tool invoked: tools/get_task_context id=%s", task_id)
    try:
        task = await _api_get(f"/tasks/{task_id}")
    except httpx.HTTPStatusError as e:
        return f"Error fetching task: {e}"

    if "error" in task:
        return f"Task not found: {task_id}"

    # Build formatted context document
    lines: list[str] = []

    # Task header
    lines.append(f"# {task.get('shortId', '')} — {task.get('title', 'Untitled')}")
    lines.append("")
    lines.append(f"**Status:** {task.get('status', '?')}  ")
    lines.append(f"**Priority:** {task.get('priority', '?')}  ")
    if task.get("agentBotId"):
        lines.append(f"**Assigned to:** {task['agentBotId']}  ")
    if task.get("modelId"):
        lines.append(f"**Model:** {task['modelId']}  ")
    lines.append(f"**Created:** {task.get('createdAt', '?')}  ")
    lines.append(f"**Updated:** {task.get('updatedAt', '?')}  ")
    lines.append("")

    # Description
    if task.get("description"):
        lines.append("## Description")
        lines.append("")
        lines.append(task["description"])
        lines.append("")

    # Dependencies
    deps = task.get("dependsOn", [])
    if deps:
        lines.append("## Dependencies")
        lines.append("")
        for dep in deps:
            status_icon = "✅" if dep.get("status") == "COMPLETED" else "⏳"
            lines.append(f"- {status_icon} {dep.get('shortId', '?')} — {dep.get('title', '?')} ({dep.get('status', '?')})")
        lines.append("")

    # Steps
    steps = task.get("steps", [])
    if steps:
        lines.append("## Steps")
        lines.append("")
        for step in steps:
            status_map = {
                "PENDING": "[ ]",
                "RUNNING": "[~]",
                "COMPLETED": "[x]",
                "FAILED": "[!]",
                "SKIPPED": "[-]",
            }
            checkbox = status_map.get(step.get("status", ""), "[ ]")
            step_type = step.get("type", "")
            type_label = f" ({step_type})" if step_type else ""
            lines.append(f"- {checkbox} {step.get('title', '?')}{type_label}")
            if step.get("output"):
                # Indent output under step
                for out_line in step["output"].split("\n"):
                    lines.append(f"      {out_line}")
        lines.append("")

    # Existing response
    if task.get("response"):
        lines.append("## Previous Response")
        lines.append("")
        lines.append(task["response"])
        lines.append("")

    # Project context
    project = task.get("project")
    if project:
        lines.append(f"## Project: {project.get('name', '?')}")
        lines.append("")
        # Fetch full project for context prompt
        try:
            full_project = await _api_get(f"/projects/{project['id']}")
            if full_project.get("contextPrompt"):
                lines.append("### Project Context")
                lines.append("")
                lines.append(full_project["contextPrompt"])
                lines.append("")
            if full_project.get("description"):
                lines.append("### Project Description")
                lines.append("")
                lines.append(full_project["description"])
                lines.append("")
        except httpx.HTTPStatusError:
            lines.append("_(Could not load project context)_")
            lines.append("")

    return "\n".join(lines)
