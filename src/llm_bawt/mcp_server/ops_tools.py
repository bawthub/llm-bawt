"""Generic ops MCP tools (TASK-639).

Three tools, deliberately narrow so the agent surface stays stable as the
catalog grows:

* :func:`ops_list_operations` — discover what operations exist right now.
* :func:`ops_run` — invoke one by slug with validated args. Never accepts a
  raw command string; the executable command comes from the DB row.
* :func:`ops_job_status` — poll a running/terminal job.

These tools do NOT own their own approval gating — the approval-aware
FastMCP interceptor (Slice E) evaluates a policy against every call before
the function runs. Until the interceptor is wired the tools remain
approval-agnostic: the operator can gate the same three tools via a
classic bridge-hook policy on ``ops_run``/``ops_job_status`` names.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

from .registry import mcp

logger = logging.getLogger(__name__)


_CONFIG_SINGLETON = None


def _get_ops_service():
    """Lazy accessor — resolve via a module-cached ``Config`` (the same
    singleton pattern as :mod:`profile_tools` and :mod:`storage`).

    ``get_service()`` is not usable here: the MCP server runs as its own
    process, so the FastAPI service singleton is never set in it and the
    accessor raises ``Service not initialized``.

    The ``Config`` must be cached, not constructed per call. The store cache
    in ``service.dependencies`` is keyed on ``id(config)``, so a fresh
    ``Config()`` each call misses the cache and rebuilds every ops store —
    re-running schema bootstrap DDL on every invocation, and (before the
    lock was made re-entrant) deadlocking the MCP event loop outright.
    Caching also avoids ``id()`` reuse aliasing a store to a dead config.
    """
    global _CONFIG_SINGLETON
    if _CONFIG_SINGLETON is None:
        from ..utils.config import Config

        _CONFIG_SINGLETON = Config()
    from ..service.dependencies import get_ops_service

    return get_ops_service(_CONFIG_SINGLETON)


# ---------------------------------------------------------------------------
# ops_list_operations
# ---------------------------------------------------------------------------

@mcp.tool(name="ops_list_operations")
async def ops_list_operations(include_disabled: bool = False) -> dict:
    """List every operation the agent can invoke via ``ops_run``.

    Args:
        include_disabled: If True, include disabled rows (operator view).

    Returns:
        Dict with keys:
          - ``operations``: list of ``{slug, title, description, risk_level,
            category, target_host, timeout_seconds, args_schema,
            args_defaults}``
          - ``total``: count in the list
    """
    ops = await asyncio.to_thread(_get_ops_service)
    rows = await asyncio.to_thread(ops.list_operations_for_agent, include_disabled=include_disabled)
    return {"operations": rows, "total": len(rows)}


# ---------------------------------------------------------------------------
# ops_run
# ---------------------------------------------------------------------------

def _caller_provenance() -> dict[str, str]:
    """Who is running this job, for the ``ops_jobs.caller_*`` audit columns.

    Two arrival paths:

    * approved server-side execution — the resolve route replays the persisted
      approval row's identity through a contextvar;
    * a directly allowed call — the agent's signed per-turn capability is still
      on the contextvar stack.

    Never sourced from tool arguments: caller identity must not be forgeable
    through the public MCP schema. Returns ``{}`` when neither is available
    (manual/generic MCP clients), leaving the columns null rather than guessing.
    """
    from .approval_interceptor import current_approved_caller_context

    approved = current_approved_caller_context()
    if approved is not None:
        return {
            "caller_bot_id": approved.bot_id or None,
            "caller_user_id": approved.user_id or None,
            "caller_turn_id": approved.turn_id or None,
            "caller_session_key": approved.session_key or None,
            "caller_backend": approved.backend or None,
            "approval_request_id": approved.approval_request_id or None,
        }

    from ..task_turn_context import open_task_turn_context
    from .task_association import current_task_turn_capability

    try:
        # Raises (never returns None) when no capability is present.
        turn = open_task_turn_context(current_task_turn_capability())
    except Exception:  # noqa: BLE001 — provenance is best-effort, never fatal
        return {}
    return {
        "caller_bot_id": turn.bot_id or None,
        "caller_user_id": turn.user_id or None,
        "caller_turn_id": turn.turn_id or None,
    }


@mcp.tool(name="ops_run")
async def ops_run(
    operation: str,
    args: dict[str, Any] | None = None,
    idempotency_key: str | None = None,
) -> dict:
    """Run a catalogued operation.

    The agent supplies a slug + validated args — the executable command
    itself lives in the DB and is authored by the operator. Unknown args
    are rejected before any script runs.

    Args:
        operation: Operation slug (e.g. ``"llm-bawt.restart-app"``). Get
            valid slugs from ``ops_list_operations``.
        args: Arguments matching the operation's declared JSON Schema.
        idempotency_key: Optional stable key. A retry with the same key
            returns the pre-existing job without re-dispatching.

    Returns:
        ``{job_id, operation, state, submitted_at, host_unit_name}`` on
        success. On failure raises with a machine-readable ``code`` in the
        error message: ``operation_not_found`` | ``operation_disabled`` |
        ``args_invalid`` | ``executor_unavailable`` | ``dispatch_failed``.
    """
    from ..ops.service import OpsDispatchError

    if not operation or not isinstance(operation, str):
        raise ValueError("operation slug is required")
    ops = await asyncio.to_thread(_get_ops_service)
    from .approval_interceptor import current_approved_caller_context

    approved = current_approved_caller_context()
    # ApprovedCallerContext is minted by the trusted replay path, NEVER from
    # public tool input. Contextvars propagate across asyncio.to_thread.
    snapshot = approved.operations_snapshot if approved else None
    if approved is not None and snapshot is None:
        raise RuntimeError("[snapshot_invalid] approved operation has no persisted invocation snapshot; request fresh approval")
    # The caller's key wins, including through approval replay, so identical
    # approved calls collapse onto one job (TASK-959). Without one, an approved
    # call keys on its durable approval id: a reclaimed execution lease after a
    # crash then returns the already-created job instead of dispatching again.
    idem = (idempotency_key or "").strip() or (
        approved.approval_request_id if approved and approved.approval_request_id else uuid.uuid4().hex
    )
    provenance = _caller_provenance()
    try:
        result = await asyncio.to_thread(
            ops.dispatch_job,
            operation_slug=operation,
            args={} if args is None else args,
            idempotency_key=idem,
            approved_snapshot=snapshot,
            **provenance,
        )
    except OpsDispatchError as exc:
        # Surface a structured error the agent can parse. MCP tools
        # signal errors by raising; the caller sees a tool_result with
        # is_error=True and the message text.
        raise RuntimeError(f"[{exc.code}] {exc}") from exc
    return {
        "id": result["id"],
        "job_id": result["id"],
        "operation": result["operation"],
        "state": result["state"],
        "submitted_at": result["submitted_at"],
        "host_unit_name": result.get("host_unit_name"),
        "idempotency_key": idem,
    }


# ---------------------------------------------------------------------------
# ops_job_status
# ---------------------------------------------------------------------------

@mcp.tool(name="ops_job_status")
async def ops_job_status(
    job_id: str,
    output_tail_bytes: int | None = None,
) -> dict:
    """Poll a job. Reconciles the host status file into the DB on the way
    if the job is still active.

    Args:
        job_id: Job id from a prior ``ops_run`` call.
        output_tail_bytes: When non-null, include the last N bytes of the
            job's combined stdout/stderr in the response. Default: omit.

    Returns:
        Full job metadata dict including ``state``, ``exit_code``,
        ``terminal`` flag, ``started_at`` / ``finished_at``, and
        ``output_tail`` when requested.
    """
    if not job_id:
        raise ValueError("job_id is required")
    ops = await asyncio.to_thread(_get_ops_service)
    result = await asyncio.to_thread(
        ops.get_job_status,
        job_id,
        output_tail_bytes=output_tail_bytes,
        reconcile_if_active=True,
    )
    if result is None:
        raise LookupError(f"no job with id {job_id!r}")
    return result


__all__ = ["ops_list_operations", "ops_run", "ops_job_status"]
