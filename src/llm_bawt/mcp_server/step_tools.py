"""MCP tools for task step CRUD.

Imported by server.py to register the step tool group.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from .server import mcp
from .task_step_types import (
    StepStatus,
    TaskStepInput,
    step_status_error,
    validate_step_inputs,
    validate_step_updates,
)
from .task_api import (
    api_delete as _api_delete,
    api_patch as _api_patch,
    api_post as _api_post,
    api_put as _api_put,
    headers as _headers,
    http_error as _http_error,
)

logger = logging.getLogger(__name__)

@mcp.tool(name="steps_update")
async def update_step(
    task_id: str,
    step_id: str | None = None,
    status: StepStatus | None = None,
    output: str | None = None,
    updates: list[dict] | None = None,
    bot_id: str | None = None,
) -> Any:
    """Update one step, or many steps in a single call.

    Call this as you work through each step:
    1. Set status="RUNNING" when you start the step.
    2. Set status="COMPLETED", output="what you did" when done.
    3. Set status="FAILED", output="error details" if it fails.
    4. Set status="SKIPPED", output="reason" to skip.

    BATCH: to update several steps at once (e.g. mark five steps COMPLETED),
    pass ``updates`` instead of the scalar args — one MCP call, one DB
    transaction, instead of N separate steps_update calls. Each entry needs
    ``step_id`` plus any of ``status``/``output``/``title``/``type``/
    ``file_path``. Steps not belonging to ``task_id`` are ignored.

    Valid statuses are exactly PENDING, RUNNING, COMPLETED, FAILED, SKIPPED
    (not IN_PROGRESS, DONE or BLOCKED); anything else is rejected before
    BawtHub is called.

    Args:
        task_id: Parent task ID (cuid) or shortId (e.g. "TASK-42").
        step_id: Step ID (cuid) (single-update mode).
        status: PENDING, RUNNING, COMPLETED, FAILED, or SKIPPED (single mode).
        output: Summary of what was done or error details (single mode).
        updates: List of dicts for batch mode, each {"step_id": ..., "status"?:
                 ..., "output"?: ..., ...}. Overrides the scalar args.
        bot_id: Your bot ID for activity attribution.

    Returns:
        Single mode: the updated step object. Batch mode: the task's steps in
        order. Error dict on failure.
    """
    if updates:
        logger.debug("MCP tool invoked: tools/update_step task=%s batch=%d", task_id, len(updates))
        validation_error = validate_step_updates(updates)
        if validation_error:
            return {"error": validation_error}
        norm: list[dict[str, Any]] = []
        for u in updates:
            entry: dict[str, Any] = {"stepId": u.get("step_id") or u.get("stepId")}
            for key in ("status", "output", "title", "type"):
                if u.get(key) is not None:
                    entry[key] = u[key]
            fp = u.get("file_path", u.get("filePath"))
            if fp is not None:
                entry["filePath"] = fp
            norm.append(entry)
        try:
            return await _api_patch(
                f"/tasks/{task_id}/steps",
                json={"updates": norm},
                headers=_headers(bot_id),
            )
        except httpx.HTTPStatusError as e:
            return _http_error(e)

    if not step_id:
        return {"error": "Provide step_id (single) or updates (batch)"}

    logger.debug("MCP tool invoked: tools/update_step task=%s step=%s status=%s", task_id, step_id, status)
    status_error = step_status_error(status, "status")
    if status_error:
        return {"error": status_error}
    body: dict[str, Any] = {}
    if status is not None:
        body["status"] = status
    if output is not None:
        body["output"] = output

    if not body:
        return {"error": "No fields to update"}

    try:
        return await _api_patch(
            f"/tasks/{task_id}/steps/{step_id}",
            json=body,
            headers=_headers(bot_id),
        )
    except httpx.HTTPStatusError as e:
        return _http_error(e)


@mcp.tool(name="steps_delete")
async def delete_step(
    task_id: str,
    step_id: str | None = None,
    step_ids: list[str] | None = None,
    bot_id: str | None = None,
) -> dict:
    """Delete one step, or many steps in a single call.

    WARNING: This permanently removes the step(s). Prefer setting status to
    SKIPPED via steps_update unless the step is genuinely a mistake — the
    audit trail is more useful than a silently-deleted row.

    BATCH: to remove several steps at once, pass ``step_ids`` instead of
    ``step_id`` — one MCP call and one DB transaction instead of N separate
    steps_delete calls. Ids not belonging to ``task_id`` are ignored. (To
    replace a task's entire checklist, prefer ``steps_set``.)

    Args:
        task_id: Parent task ID (cuid) or shortId.
        step_id: Step ID (cuid) to delete (single mode).
        step_ids: List of step IDs (cuids) to delete (batch mode). Overrides step_id.
        bot_id: Your bot ID for activity attribution.

    Returns:
        Confirmation dict, or error dict.
    """
    if step_ids:
        logger.debug("MCP tool invoked: tools/delete_step task=%s batch=%d", task_id, len(step_ids))
        try:
            return await _api_delete(
                f"/tasks/{task_id}/steps",
                json={"stepIds": step_ids},
                headers=_headers(bot_id),
            )
        except httpx.HTTPStatusError as e:
            return _http_error(e)

    if not step_id:
        return {"error": "Provide step_id (single) or step_ids (batch)"}

    logger.debug(
        "MCP tool invoked: tools/delete_step task=%s step=%s",
        task_id,
        step_id,
    )
    try:
        return await _api_delete(
            f"/tasks/{task_id}/steps/{step_id}",
            headers=_headers(bot_id),
        )
    except httpx.HTTPStatusError as e:
        return _http_error(e)


@mcp.tool(name="steps_add")
async def add_steps(
    task_id: str,
    steps: list[TaskStepInput],
    bot_id: str | None = None,
) -> list[dict] | dict:
    """Add new steps to a task.

    Steps are appended after existing steps. Use when planning work
    or when you discover additional steps mid-execution.

    Args:
        task_id: Task ID (cuid) or shortId (e.g. "TASK-42").
        steps: List of step dicts. Each needs "title" (str).
               Optional: "type" (default "PLAN") and "status"
               (default "PENDING").
               Types: PLAN, READ_FILE, EDIT_FILE, CREATE_FILE,
               DELETE_FILE, RUN_COMMAND, SEARCH, ASK_USER, REVIEW.
               Statuses: PENDING, RUNNING, COMPLETED, FAILED, SKIPPED.
        bot_id: Your bot ID for activity attribution.

    Returns:
        List of created step objects, or an error dict. The return type
        admits the dict so FastMCP's output validation cannot replace the
        real error with a "valid list" message (TASK-1313).
    """
    logger.debug("MCP tool invoked: tools/add_steps task=%s count=%d", task_id, len(steps))
    validation_error = validate_step_inputs(steps)
    if validation_error:
        return {"error": validation_error}
    try:
        return await _api_post(
            f"/tasks/{task_id}/steps",
            json=steps,
            headers=_headers(bot_id),
        )
    except httpx.HTTPStatusError as e:
        return _http_error(e)


@mcp.tool(name="steps_set")
async def set_steps(
    task_id: str,
    steps: list[TaskStepInput],
    bot_id: str | None = None,
) -> list[dict] | dict:
    """Replace a task's ENTIRE step list in one atomic call.

    Deletes all existing steps and recreates them from ``steps``, reindexed
    from 0. This is the right tool for rewriting a checklist: instead of N
    steps_delete + M steps_add calls (a wall of tool invocations), set the whole
    plan in a single transaction. The task title/description are untouched.

    Args:
        task_id: Task ID (cuid) or shortId (e.g. "TASK-42").
        steps: The full ordered list of step dicts. Each needs "title" (str);
               optional "type" (default "PLAN"), "status" (default "PENDING"),
               "output", "filePath". Pass [] to clear all steps.
               Types: PLAN, READ_FILE, EDIT_FILE, CREATE_FILE, DELETE_FILE,
               RUN_COMMAND, SEARCH, ASK_USER, REVIEW.
               Statuses: PENDING, RUNNING, COMPLETED, FAILED, SKIPPED.
        bot_id: Your bot ID for activity attribution.

    Returns:
        The task's steps in order after the replace, or an error dict. A
        rejected replace leaves the existing steps untouched.
    """
    logger.debug("MCP tool invoked: tools/set_steps task=%s count=%d", task_id, len(steps))
    validation_error = validate_step_inputs(steps)
    if validation_error:
        return {"error": validation_error}
    try:
        return await _api_put(
            f"/tasks/{task_id}/steps",
            json=steps,
            headers=_headers(bot_id),
        )
    except httpx.HTTPStatusError as e:
        return _http_error(e)
