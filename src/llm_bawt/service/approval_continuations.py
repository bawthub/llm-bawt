"""Durable approval continuations and stranded-execution recovery.

MCP results and server-owned harness decisions use the same fenced outbox.
The legacy spawn helper remains import-compatible but resolve no longer uses
fire-and-forget dispatch. Generic uncertain side effects are never replayed.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import timedelta
from uuid import NAMESPACE_URL, uuid5
from ..approval_policies import KIND_MCP, REQ_APPROVED, REQ_DENIED, REQ_CANCELLED, REQ_RESPONDED, CONT_DISPATCHING
from ..approval_models import _as_aware_utc, _utcnow
from typing import Any

from .dependencies import get_service
from .approval_continuation_session import approval_delivery_session, approval_origin_session
from .inter_bot_claims import turn_admission_claim
from .turn_admission import RESERVED, TargetBusy, turn_request_extensions


MCP_RESULT_ENVELOPE_PREFIX = "[LLM_BAWT_MCP_TOOL_RESULT]"


def build_mcp_result_prompt(payload: dict[str, Any]) -> str:
    """Protocol-safe fallback carrying the actual persisted MCP result.

    The original tool already returned a non-blocking pending result, so some
    Claude CLI versions reject a second native ToolResultBlock for that tool-use
    id. This deterministic user-message envelope delivers the real normalized
    result without asking the model to execute anything again.
    """
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return (
        f"{MCP_RESULT_ENVELOPE_PREFIX}\n{canonical}\n"
        "The MCP tool has already been executed or refused server-side. Do not "
        "retry or re-issue the tool. Treat the result above as authoritative and "
        "continue from it."
    )


def continuation_payload_from_row(row, *, result_override: Any = None) -> dict[str, Any]:
    if result_override is not None:
        result = result_override
    else:
        try:
            result = json.loads(row.result_json or "null")
        except json.JSONDecodeError:
            result = row.result_json
    terminal_error = (
        isinstance(result_override, dict)
        and result_override.get("state")
        in {"failed", "error", "timed_out", "lost", "cancelled"}
    )
    return {
        "kind": "mcp_tool_result",
        "approval_request_id": row.id,
        "original_tool_use_id": row.tool_use_id,
        "tool_name": row.tool_name,
        "result": result,
        "is_error": bool(row.result_is_error) or terminal_error,
    }


def _continuation_session(row):
    return approval_origin_session(row)


def _terminal_ops_result(service, row) -> dict[str, Any] | None:
    """Return terminal ops receipt, or None while the restart is still active."""
    if row.request_kind != KIND_MCP or row.tool_name != "ops_run":
        return {}
    try:
        current = json.loads(row.result_json or "null")
    except json.JSONDecodeError:
        return {}
    if not isinstance(current, dict) or not current.get("job_id"):
        return {}
    ops = getattr(service, "_ops_service", None)
    if ops is None:
        from .dependencies import get_ops_service
        ops = get_ops_service(service.config)
    result = ops.get_job_status(
        str(current["job_id"]), output_tail_bytes=4096, reconcile_if_active=True,
    )
    if not result or not result.get("terminal"):
        return None
    return result


def _persist_terminal_ops_tool_result(service, row, result: dict[str, Any]) -> None:
    """Make the original tool card agree with the terminal ops receipt."""
    turn_store = getattr(service, "_turn_log_store", None)
    engine = getattr(turn_store, "engine", None)
    if engine is None or not row.tool_use_id:
        return
    try:
        from .tool_call_store import ToolCallStore

        ToolCallStore(engine).resolve_approval_result(
            tool_use_id=row.tool_use_id,
            approval_request_id=row.id,
            approval_status=row.status,
            result=result,
            is_error=result.get("state")
            in {"failed", "error", "timed_out", "lost", "cancelled"},
        )
    except Exception:
        log.exception("Could not persist terminal ops tool result id=%s", row.id)


def _continuation_identity(row):
    if not row.continuation_id:
        raise ValueError("Approval continuation has no durable identity")
    # Message IDs are UUID-sized (VARCHAR(36)); continuation/outbox IDs are
    # not. Keep turn/request identities unchanged: the exact harness grant is
    # bound to that request ID, and existing turn receipts must remain visible.
    def message_id(role):
        return str(uuid5(NAMESPACE_URL, f"llm-bawt:approval:{row.continuation_id}:{role}"))

    return {"inter_bot_turn_id": "turn-" + row.continuation_id,
            "inter_bot_bridge_request_id": "req_delivery_approval_" + row.continuation_id,
            "user_message_id": message_id("user"),
            "assistant_message_id": message_id("assistant")}


def validate_approval_continuation_claim(service, request, claim, *, store=None):
    """Reuse existing deterministic transport fields, gated by a Python-only claim."""
    store = store if store is not None else service._tool_approval_policy_store
    row = store.get_request(claim[0])
    if (row is None or row.continuation_state != CONT_DISPATCHING
            or row.continuation_claim_token != claim[1]
            or row.bot_id != request.bot_id or row.user_id != request.user
            or row.turn_id != request.parent_turn_id
            or approval_delivery_session(service, row) != request.session_id
            or request.inter_bot_delivery_id is not None
            or any(getattr(request, key) != value for key, value in _continuation_identity(row).items())
            or row.continuation_next_attempt_at is None
            or _as_aware_utc(row.continuation_next_attempt_at) < _utcnow() - timedelta(seconds=120)):
        raise ValueError("Invalid or stale approval continuation claim")


async def _send_harness_grant(service, store, row, *, claim_token):
    if row.status != REQ_APPROVED or row.grant_state == "sent":
        return
    from .routes.approval_policies import _subscriber
    subscriber = _subscriber()
    if subscriber is None:
        store.set_grant_state(row.id, "failed", "Approval bridge unavailable")
        raise RuntimeError("Approval bridge unavailable; grant not sent")
    if not store.claim_client_grant(row.id, continuation_claim_token=claim_token):
        raise RuntimeError("Approval grant dispatch uncertain; manual reconciliation required")
    try:
        await subscriber.send_approval_grant(
            session_key=row.session_key or "main", grant_key=row.grant_key,
            backend=row.backend, request_id=row.id)
    except Exception as exc:
        store.set_grant_state(row.id, "uncertain", str(exc))
        raise
    store.set_grant_state(row.id, "sent")
    await asyncio.sleep(0.75)


class ContinuationLeaseLost(RuntimeError):
    """The outbox claim was lost mid-dispatch; another worker may own it."""


# Waiting behind a running turn is normal; the poll interval backs off with
# the wait so a long turn (or an unresolved zombie row) never hot-loops.
ADMISSION_WAIT_WARN_S = 60.0


def _admission_wait_delay(row) -> float:
    since = getattr(row, "execution_finished_at", None) or getattr(row, "resolved_at", None)
    waited = (_utcnow() - _as_aware_utc(since)).total_seconds() if since else 0.0
    if waited < 300:
        return 2.0
    if waited < 3600:
        return 10.0
    return ADMISSION_WAIT_WARN_S


def _release_unstarted(turn_store, turn_id: str, claim) -> None:
    """Drop this claim's reservation if its turn never started.

    Fenced in the DB transaction by the reservation's owner token AND by the
    live outbox claim: a superseded worker cannot delete the reclaimer's
    re-armed reservation of the same deterministic id.
    """
    release = getattr(turn_store, "release_reservation", None)
    if release is None:
        return
    try:
        release(turn_id, owner=claim.owner, fence=claim.fence)
    except Exception:
        log.exception("Could not release approval continuation reservation %s", turn_id)


async def dispatch_mcp_result_continuation(
    service, store, row, *, result_override: Any = None,
) -> None:
    """Deliver MCP results or harness decisions with a fenced, renewable claim."""
    from .schemas import ChatCompletionRequest, ChatMessage, McpToolResultContinuation

    token = row.continuation_claim_token
    identity = _continuation_identity(row)
    payload = (
        continuation_payload_from_row(row, result_override=result_override)
        if row.request_kind == KIND_MCP else None
    )
    prompt = (build_mcp_result_prompt(payload) if payload else
              build_respond_prompt(row.resolution_message, row.subject, row.tool_name)
              if row.status == REQ_RESPONDED else
              build_continuation_prompt(row.status == REQ_APPROVED, row.subject, row.tool_name,
                                        tool_arguments_json=row.tool_arguments_json))
    request = ChatCompletionRequest(
        messages=[ChatMessage(role="user", content=prompt)], bot_id=row.bot_id, user=row.user_id,
        stream=True, parent_turn_id=row.turn_id, session_id=_continuation_session(row),
        continuation_payload=McpToolResultContinuation(**payload) if payload else None, **identity,
    )
    request._internal_approval_claim = (row.id, token)

    async def heartbeat():
        while True:
            await asyncio.sleep(20)
            if not await asyncio.to_thread(store.renew_continuation_claim, row.id, token):
                raise ContinuationLeaseLost("Approval continuation lease lost")

    turn_store = getattr(service, "_turn_log_store", None)
    # One trusted claim fences reservation, consumption (save_turn) and release.
    admission_claim = turn_admission_claim(service, request, store=store)

    async def drive():
        # A committed successful deterministic turn needs only its outbox ack.
        prior = turn_store.get_turn(identity["inter_bot_turn_id"]) if turn_store is not None else None
        if prior is not None and prior.ended_at is not None:
            if prior.status in ("ok", "completed") and not prior.error_text:
                return
            raise RuntimeError("Previous continuation turn failed; manual reconciliation required")
        # A bare reservation wrote no messages; only a started turn pins identity.
        started = prior is not None and prior.status != RESERVED
        target_session = await asyncio.to_thread(approval_delivery_session, service, row)
        if target_session != request.session_id:
            if started:
                # The deterministic turn has already written messages. Do not
                # reuse their IDs in a different session or replay uncertain work.
                raise RuntimeError("Cannot redirect an already-started approval continuation after reset")
            request.session_id = target_session
            request.messages[0].content += (
                "\nThe user reset the conversation with /new while this result was pending. "
                "Continue in this fresh session using its message history; do not restore "
                "the previous SDK context or re-execute the completed operation."
            )
        # Fail fast on a superseded claim before ANY side effect; the DB fences
        # on reserve/grant/release close the remaining check-then-act window.
        await asyncio.to_thread(
            validate_approval_continuation_claim, service, request,
            request._internal_approval_claim, store=store,
        )
        if not started:
            # TASK-1082: claim the idle target through the shared admission
            # authority BEFORE any external side effect (the one-shot harness
            # grant). Busy -> TargetBusy -> deferred, and no grant is sent.
            # While reserved, ordinary starts and other producers lose.
            # Fail closed: no admission authority means no unfenced dispatch.
            reserve = getattr(turn_store, "reserve_idle_turn", None)
            if reserve is None or not await asyncio.to_thread(
                reserve,
                turn_id=identity["inter_bot_turn_id"],
                bot_id=row.bot_id,
                user_id=row.user_id,
                user_prompt=request.messages[0].content,
                request_json={
                    "messages": [{"role": "user", "content": request.messages[0].content}],
                    "session_id": request.session_id,
                    **turn_request_extensions(request),
                },
                trigger_message_id=identity["user_message_id"],
                owner=admission_claim.owner,
                fence=admission_claim.fence,
            ):
                raise RuntimeError("Turn admission unavailable; refusing unfenced continuation")
        if row.request_kind != KIND_MCP:
            await _send_harness_grant(service, store, row, claim_token=token)
        async for chunk in service.chat_completion_stream(request):
            # Some adapters yield an error envelope rather than raising.
            if isinstance(chunk, str):
                for line in chunk.splitlines():
                    if line.startswith("data: ") and line[6:] != "[DONE]":
                        try:
                            event = json.loads(line[6:])
                        except json.JSONDecodeError:
                            continue
                        if isinstance(event, dict) and event.get("error"):
                            raise RuntimeError(str(event["error"]))
        if turn_store is not None:
            turn = turn_store.get_turn(identity["inter_bot_turn_id"])
            if turn is None or turn.ended_at is None:
                raise RuntimeError("Approval continuation did not finalize")
            if turn.status not in ("ok", "completed") or turn.error_text:
                raise RuntimeError(turn.error_text or f"Continuation ended with {turn.status}")

    renewal = asyncio.create_task(heartbeat())
    driver = asyncio.create_task(drive())
    try:
        done, _ = await asyncio.wait((renewal, driver), return_when=asyncio.FIRST_COMPLETED)
        if renewal in done:
            await renewal
        await driver
        store.mark_continuation_delivered(row.id, claim_token=token)
    except asyncio.CancelledError:
        raise
    except TargetBusy as busy:
        # Not an attempt: the target is executing (or reserved/ambiguous).
        # Nothing was started; save_turn already dropped its own reservation.
        _release_unstarted(turn_store, identity["inter_bot_turn_id"], admission_claim)
        delay = _admission_wait_delay(row)
        if delay >= ADMISSION_WAIT_WARN_S:
            log.warning("Approval continuation id=%s still waiting for idle target: %s", row.id, busy)
        store.defer_continuation_claim(
            row.id, claim_token=token, reason=f"waiting for idle target: {busy}",
            delay_seconds=delay,
        )
    except Exception as error:
        if not isinstance(error, ContinuationLeaseLost):
            # The heartbeat already proved this worker lost authority.
            _release_unstarted(turn_store, identity["inter_bot_turn_id"], admission_claim)
        store.mark_continuation_failed(row.id, error=str(error), claim_token=token)
        raise
    finally:
        for task in (renewal, driver):
            task.cancel()
        await asyncio.gather(renewal, driver, return_exceptions=True)


async def recover_approval_requests(store):
    """Recover committed approvals, never blindly replay generic side effects."""
    from .routes.approval_policies import _resolve_mcp_request
    for row in await asyncio.to_thread(store.find_recoverable_mcp_requests):
        try:
            await _resolve_mcp_request(store, row,
                outcome={REQ_APPROVED: "approve", REQ_DENIED: "deny", REQ_CANCELLED: "cancel", REQ_RESPONDED: "respond"}.get(row.status, "deny"),
                message=row.resolution_message, resolved_by=row.resolved_by)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("Approval execution recovery failed id=%s", row.id)
    for row in await asyncio.to_thread(store.find_stranded_mcp_results):
        await asyncio.to_thread(store.enqueue_continuation, row.id)
    # Repair the approval-commit / enqueue gap for server-owned harness decisions.
    for row in await asyncio.to_thread(store.find_stranded_harness_continuations):
        await asyncio.to_thread(store.prepare_harness_continuation, row.id)


async def dispatch_due_continuations_once(service, store, *, limit: int = 20) -> None:
    """Run one recover/dispatch pass for the durable continuation outbox."""
    await recover_approval_requests(store)
    due = await asyncio.to_thread(store.find_pending_continuations, limit=limit)
    for pending in due:
        # An approved restart starts before its continuation. Claiming the
        # continuation while the job is active creates an in-flight agent turn
        # that the restart can kill. Wait for the durable terminal job receipt,
        # then start a fresh continuation from the recovered app/bridge process
        # with that final receipt as its result.
        terminal_ops_result = await asyncio.to_thread(
            _terminal_ops_result, service, pending,
        )
        if terminal_ops_result is None:
            continue
        if terminal_ops_result:
            await asyncio.to_thread(
                _persist_terminal_ops_tool_result,
                service,
                pending,
                terminal_ops_result,
            )
        claimed = await asyncio.to_thread(
            store.claim_continuation, pending.id, lease_seconds=120,
        )
        if claimed is None:
            continue
        try:
            await dispatch_mcp_result_continuation(
                service,
                store,
                claimed,
                result_override=terminal_ops_result or None,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("Approval continuation failed id=%s", claimed.id)


async def run_mcp_continuation_outbox(service, store, *, idle_seconds: float = 2.0) -> None:
    """Lifespan recovery and outbox loop; DB outages do not kill the worker."""
    while True:
        try:
            await dispatch_due_continuations_once(service, store)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("Approval recovery/outbox pass failed; retrying")
        await asyncio.sleep(idle_seconds)


log = logging.getLogger(__name__)


def spawn_continuation(
    *,
    bot_id: str,
    user_id: str,
    prompt: str,
    parent_turn_id: str | None,
    grant_settle_s: float = 0.0,
) -> None:
    """Dispatch the post-resolution continuation turn SERVER-SIDE.

    This is the fix for the "approving did nothing" gap: the continuation that
    re-issues the approved call (or tells the agent it was denied) used to
    depend on the *client* making a second /v1/chat/completions call. Surfaces
    that only resolve — the admin page, the API, a script — never messaged the
    bot, so the approved command silently never ran and the grant expired.

    We drive the real streaming pipeline detached (fire-and-forget) so all the
    normal side effects happen — turn log, persistence, and the unified Redis
    SSE events the chat UI renders. The HTTP resolve response returns
    immediately; the turn streams into the chat on its own.

    ``grant_settle_s`` gives the bridge a beat to store the one-shot approval
    grant (sent on a different Redis stream than chat.send) before the re-issued
    tool call arrives, so it isn't re-gated by a race.
    """
    try:
        from .schemas import ChatCompletionRequest, ChatMessage

        service = get_service()
        req = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content=prompt)],
            bot_id=bot_id,
            user=user_id,
            stream=True,
            parent_turn_id=parent_turn_id,
        )

        async def _drive() -> None:
            try:
                if grant_settle_s:
                    await asyncio.sleep(grant_settle_s)
                async for _ in service.chat_completion_stream(req):
                    pass
            except Exception:  # noqa: BLE001
                log.exception(
                    "server-side continuation dispatch failed (bot=%s parent_turn=%s)",
                    bot_id, parent_turn_id,
                )

        asyncio.create_task(_drive())
    except Exception:  # noqa: BLE001
        # Never let a dispatch failure break the resolve response itself.
        log.exception(
            "failed to spawn continuation (bot=%s parent_turn=%s)",
            bot_id, parent_turn_id,
        )


def build_continuation_prompt(
    approved: bool, subject: str, tool_name: str, *, tool_arguments_json: str | None = None,
) -> str:
    # Approved retries must reproduce the complete invocation. Truncating the
    # command or omitting timeout/cwd options makes the exact grant unusable.
    shown = subject if approved or len(subject) <= 400 else subject[:397] + "…"
    if approved and tool_arguments_json:
        arguments = json.loads(tool_arguments_json)
        if not isinstance(arguments, dict):
            raise ValueError("Stored approved tool arguments must be an object")
        shown = json.dumps(arguments, ensure_ascii=False, sort_keys=True)
    if approved:
        return (
            f"[The user APPROVED the {tool_name} action you requested. Re-issue "
            f"exactly this call now and continue:\n\n{shown}]"
        )
    return (
        f"[The user DENIED the {tool_name} action you requested:\n\n{shown}\n\n"
        f"Do not attempt it again. Continue without it, or explain what you need.]"
    )


def build_respond_prompt(message: str, subject: str, tool_name: str) -> str:
    """Continuation for a 'respond' resolution: the tool is NOT run; the user's
    own guidance steers the agent. Empty message falls back to a neutral note so
    the turn still closes cleanly."""
    shown = subject if len(subject) <= 400 else subject[:397] + "…"
    msg = (message or "").strip()
    if msg:
        return (
            f"[The user did NOT run the {tool_name} action you requested:\n\n{shown}\n\n"
            f"They responded with this guidance — follow it and do not re-issue the "
            f"original call unless it tells you to:\n\n{msg}]"
        )
    return (
        f"[The user reviewed the {tool_name} action you requested and chose not to "
        f"run it, with no further instruction. Continue without it.]"
    )


__all__ = [
    "MCP_RESULT_ENVELOPE_PREFIX",
    "spawn_continuation",
    "build_continuation_prompt",
    "build_respond_prompt",
    "build_mcp_result_prompt",
    "continuation_payload_from_row",
    "dispatch_mcp_result_continuation",
    "run_mcp_continuation_outbox",
]
