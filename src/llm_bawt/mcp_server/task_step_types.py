"""Canonical task-step payload types shared by BawtHub MCP tools."""

from __future__ import annotations

from typing import Literal, NotRequired, Required, TypedDict

StepType = Literal[
    "PLAN",
    "READ_FILE",
    "EDIT_FILE",
    "CREATE_FILE",
    "DELETE_FILE",
    "RUN_COMMAND",
    "SEARCH",
    "ASK_USER",
    "REVIEW",
]

STEP_TYPES: tuple[StepType, ...] = (
    "PLAN",
    "READ_FILE",
    "EDIT_FILE",
    "CREATE_FILE",
    "DELETE_FILE",
    "RUN_COMMAND",
    "SEARCH",
    "ASK_USER",
    "REVIEW",
)
_STEP_TYPE_SET = frozenset(STEP_TYPES)

# Mirrors Prisma TaskStatus; task tools accept str so HTTP can normalize case.
TASK_STATUSES = (
    "QUEUED", "PLANNING", "REFINED", "IN_PROGRESS", "REVIEW",
    "COMPLETED", "FAILED", "BUG", "CANCELLED", "DELETED", "BLOCKED",
)

# TASK-1368: Nick's definition of BLOCKED, the only gate before COMPLETED.
# Shared by the tasks_create/tasks_update descriptions and docstrings.
BLOCKED_DEFINITION = (
    "BLOCKED is the only gate before COMPLETED and makes the task Nick's action "
    "item, so it should be rare. Block only for: (1) a design/product decision "
    "only Nick can make; (2) a real-world dependency bots can't satisfy "
    "(restart, deploy/ship, credentials, hardware, an external party); (3) a "
    "change that would go against Nick's stated intent, beyond what reasoning "
    "can settle. Never for mandate technicalities, small process deviations, "
    "reviewer nits, bot-vs-bot disagreements (Gavel mediates) or work waiting "
    "its turn; misusing BLOCKED is a process failure. Entering BLOCKED requires "
    "blocked_reason: what Nick must do or decide. Leaving BLOCKED clears it."
)

# Mirrors BawtHub's Prisma StepStatus enum (TASK-1313). Anything else used to
# reach BawtHub and come back as an opaque HTTP 500.
StepStatus = Literal["PENDING", "RUNNING", "COMPLETED", "FAILED", "SKIPPED"]

STEP_STATUSES: tuple[StepStatus, ...] = (
    "PENDING",
    "RUNNING",
    "COMPLETED",
    "FAILED",
    "SKIPPED",
)
_STEP_STATUS_SET = frozenset(STEP_STATUSES)


class TaskStepInput(TypedDict, total=False):
    """JSON shape accepted by task creation and step-list mutation tools."""

    title: Required[str]
    type: NotRequired[StepType]
    status: NotRequired[StepStatus]
    output: NotRequired[str]
    file_path: NotRequired[str]
    filePath: NotRequired[str]


def step_type_error(value: object, field: str) -> str | None:
    """Error naming the valid step types, or None when absent or valid."""
    if value is None or (isinstance(value, str) and value in _STEP_TYPE_SET):
        return None
    return f"{field} must be one of: " + ", ".join(STEP_TYPES)


def step_status_error(value: object, field: str) -> str | None:
    """Error naming the valid step statuses, or None when absent or valid."""
    if value is None or (isinstance(value, str) and value in _STEP_STATUS_SET):
        return None
    return f"{field} must be one of: " + ", ".join(STEP_STATUSES)


def validate_step_inputs(steps: list[TaskStepInput]) -> str | None:
    """Return a precise validation error without sending bad data downstream."""
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            return f"steps[{index}] must be an object"
        title = step.get("title")
        if not isinstance(title, str) or not title.strip():
            return f"steps[{index}].title must be a non-empty string"
        error = step_type_error(
            step.get("type"), f"steps[{index}].type"
        ) or step_status_error(step.get("status"), f"steps[{index}].status")
        if error:
            return error
    return None


def validate_step_updates(updates: list[dict]) -> str | None:
    """Validate steps_update batch entries the same way as step inputs."""
    for index, update in enumerate(updates):
        if not isinstance(update, dict):
            return f"updates[{index}] must be an object"
        if not (update.get("step_id") or update.get("stepId")):
            return f"updates[{index}] needs a step_id"
        error = step_type_error(
            update.get("type"), f"updates[{index}].type"
        ) or step_status_error(update.get("status"), f"updates[{index}].status")
        if error:
            return error
    return None
