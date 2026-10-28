"""Generic ops MCP tools (TASK-639).

Three tools, deliberately narrow so the agent surface stays stable as the
catalog grows:

* :func:`ops_list_operations` — discover what operations exist right now.
* :func:`ops_run` — invoke one by slug with validated args. Never accepts a
  raw command string; the executable command comes from the DB row.
* :func:`ops_job_status` — poll a running/terminal job.
* :func:`ops_quiesce_ack` — a held turn answers a quiesce steer (TASK-1233).

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

from ..ops.approved_admission import dispatch_error_text, ops_run_receipt
from ..ops.service import OpsDispatchError
from ..task_turn_context import open_task_turn_context
from .approval_interceptor import current_approved_caller_context
from .registry import mcp
from .task_association import current_task_turn_capability, current_task_turn_context

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
    """Who is running a directly allowed job, for the ``ops_jobs.caller_*`` columns.

    Read from the agent's signed per-turn capability on the contextvar stack,
    never from tool arguments: caller identity must not be forgeable through
    the public MCP schema. Returns ``{}`` when none is present (manual/generic
    MCP clients), leaving the columns null rather than guessing. Approved calls
    never reach this tool; ops admission records their provenance from the
    persisted approval row (TASK-1220).
    """
    try:
        # Raises (never returns None) when no capability is present.
        turn = open_task_turn_context(current_task_turn_capability())
    except Exception:  # noqa: BLE001 — provenance is best-effort, never fatal
        return {}
    return {
        "caller_bot_id": turn.bot_id or None,
        "caller_user_id": turn.user_id or None,
        "caller_turn_id": turn.turn_id or None,
        # Not a column: the session a direct quiesce job's receipt returns to.
        "caller_session_id": turn.session_id or None,
    }


def _caller_instruction(ops, result: dict, bot_id: str | None) -> dict | None:
    """End-your-turn instruction while a quiesce job holds the caller's own
    bot (TASK-1233), else ``None``. Never fails the call: an ops_run whose job
    already exists must not read as an error an agent would retry."""
    from ..ops.models import JOB_QUIESCING
    from ..ops.quiesce import caller_instruction

    if result.get("state") != JOB_QUIESCING:
        return None
    try:
        return caller_instruction(ops.store.engine, result["id"], bot_id)
    except Exception:  # noqa: BLE001 - advisory; the job itself is unaffected
        logger.warning("quiesce caller instruction unavailable for %s", result.get("id"), exc_info=True)
        return None


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
        Job metadata (`job_id`, optional `release_id`, state). Errors carry
        a machine-readable code; inspect it rather than retrying blindly.
    """
    if not operation or not isinstance(operation, str):
        raise ValueError("operation slug is required")
    if current_approved_caller_context() is not None:
        # An approved ops_run is admitted by the ops reconciler from the
        # persisted approval row, never replayed through this tool (TASK-1220).
        raise RuntimeError("[admission_owned] approved ops_run is admitted by the ops reconciler, not executed here")
    ops = await asyncio.to_thread(_get_ops_service)
    idem = (idempotency_key or "").strip() or uuid.uuid4().hex
    provenance = _caller_provenance()
    try:
        result = await asyncio.to_thread(
            ops.dispatch_job,
            operation_slug=operation,
            args={} if args is None else args,
            idempotency_key=idem,
            **provenance,
        )
    except OpsDispatchError as exc:
        # Surface a structured error the agent can parse. MCP tools
        # signal errors by raising; the caller sees a tool_result with
        # is_error=True and the message text.
        raise RuntimeError(dispatch_error_text(exc)) from exc
    instruction = await asyncio.to_thread(_caller_instruction, ops, result, provenance.get("caller_bot_id"))
    return ops_run_receipt(result, idem, caller_instruction=instruction)


# ---------------------------------------------------------------------------
# ops_job_status
# ---------------------------------------------------------------------------

@mcp.tool(name="ops_job_status")
async def ops_job_status(
    job_id: str,
    output_tail_bytes: int | None = None,
) -> dict:
    """Read a job's current status. A read only: the ops reconciler records
    progress, so ``observed_at`` vs ``last_reconcile_at`` shows how fresh it is.

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
        reconcile_if_active=False,
    )
    if result is None:
        raise LookupError(f"no job with id {job_id!r}")
    result["caller_instruction"] = await asyncio.to_thread(
        _caller_instruction, ops, result, _caller_provenance().get("caller_bot_id"))
    return result


# ---------------------------------------------------------------------------
# ops_quiesce_ack
# ---------------------------------------------------------------------------

def _record_quiesce_ack(bot_id: str, turn_id: str, status: str, note: str | None) -> int | None:
    from ..ops.quiesce import record_ack

    return record_ack(_get_ops_service().store.engine, bot_id=bot_id, turn_id=turn_id,
                      status=status, note=note)


@mcp.tool(name="ops_quiesce_ack")
async def ops_quiesce_ack(status: str, note: str | None = None) -> dict:
    """Answer a quiesce request for your current turn.

    An operation is waiting for your turn to finish before it runs. Say
    whether you are wrapping up (``"stopping"``) or cannot stop safely
    (``"cannot_stop"``, with a short ``note`` saying why).

    Args:
        status: ``"stopping"`` or ``"cannot_stop"``.
        note: Optional short explanation shown to the operator.

    Returns:
        ``{"recorded": true, "gate_id": ...}``.
    """
    from ..ops.quiesce import ACK_STATUSES

    if status not in ACK_STATUSES:
        raise ValueError(f"status must be one of {list(ACK_STATUSES)}")
    # Identity comes only from the signed per-turn capability, never from args.
    turn = await asyncio.to_thread(current_task_turn_context)
    gate_id = await asyncio.to_thread(_record_quiesce_ack, turn.bot_id, turn.turn_id, status, note)
    if gate_id is None:
        raise LookupError("this turn is not held by an active quiesce")
    return {"recorded": True, "gate_id": gate_id}


__all__ = ["ops_list_operations", "ops_run", "ops_job_status", "ops_quiesce_ack"]
