"""Approval-gated tool policy admin + resolve routes (TASK-293).

Source of truth for the feature. CRUD over the policy rules, the compiled
bundle bridges fetch, the audit list of gated requests, and the resolve
endpoint a user hits to approve/deny a pending request (which grants the bridge
a one-shot allow and returns a continuation prompt the client dispatches).
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.exc import SQLAlchemyError
from agent_bridge.approval import evaluate
from ...approval_validation import candidate_policy
from ..approval_execution import ApprovalExecutionLeaseLost
from ..tool_call_store import ToolCallStore
from ...approval_policies import ApprovalStoreUnavailable, CONT_DELIVERED, CONT_PENDING, CONT_DISPATCHING

from ..approval_continuations import (
    build_continuation_prompt,
    build_respond_prompt,
)
from ..dependencies import (
    get_service,
    get_tool_approval_policy_store,
    get_turn_log_store,
)
from ...approval_policies import (
    EXEC_FAILED,
    EXEC_SKIPPED,
    EXEC_SUCCEEDED,
    KIND_MCP,
    KIND_ORCHESTRATION,
    REQ_APPROVED,
    REQ_CANCELLED,
    REQ_DENIED,
    REQ_RESPONDED,
)

log = logging.getLogger(__name__)
router = APIRouter()


def _store():
    service = get_service()
    store = get_tool_approval_policy_store(service.config)
    if store.engine is None:
        raise HTTPException(status_code=503, detail="Approval policy database unavailable")
    return store


# ---------------------------------------------------------------------------
# Request/response schemas
# ---------------------------------------------------------------------------

class PolicyUpsert(BaseModel):
    enabled: bool | None = None
    backend_scope: str | None = None
    tool_name: str | None = None
    matcher_type: str | None = None
    pattern: str | None = None
    field: str | None = None
    action: str | None = None
    severity: str | None = None
    category: str | None = None
    approval_prompt: str | None = None
    order: int | None = None
    order_index: int | None = None

    def writable(self) -> dict:
        return {k: v for k, v in self.model_dump().items() if v is not None}


class ResolveRequest(BaseModel):
    decision: str = Field(..., description="'approve', 'deny', 'cancel', or 'respond'")
    bot_id: str = Field("", description="Bot slug (for tab fanout)")
    user_id: str = Field("", description="User id (for tab fanout)")
    resolved_by: str | None = None
    # 'respond' only: the user's own guidance sent to the agent instead of the
    # canned refusal. Optional — empty falls back to a neutral "not run" note.
    message: str = Field("", description="Custom continuation text for 'respond'")
    # When True (default), the SERVER dispatches the continuation turn so the bot
    # is reliably told "approved, run exactly this" regardless of which surface
    # resolved the request (admin page, API, script — not just the chat card).
    # A client that dispatches its own continuation must pass False to avoid a
    # double turn.
    dispatch_continuation: bool = Field(
        True, description="Server owns continuation by default; chat clients that dispatch must explicitly pass false",
    )


# ---------------------------------------------------------------------------
# Policy CRUD
# ---------------------------------------------------------------------------

@router.get("/v1/tool-approval-policies", tags=["Approval Policies"])
def list_policies():
    store = _store()
    rows = store.list_all()
    return {"policies": [r.to_api() for r in rows], "total": len(rows)}


@router.get("/v1/tool-approval-policies/bundle", tags=["Approval Policies"])
def get_bundle(etag: str | None = None):
    """Compiled bundle a bridge fetches. If ``etag`` matches, returns
    ``{unchanged: true}`` so the bridge can skip re-parsing."""
    store = _store()
    try:
        bundle = store.compile_bundle()
    except (ApprovalStoreUnavailable, SQLAlchemyError) as exc:
        raise HTTPException(status_code=503, detail="Approval policy database unavailable") from exc
    if etag and etag == bundle.etag:
        return {"unchanged": True, "etag": bundle.etag, "version": bundle.version}
    return bundle.to_dict()


@router.post("/v1/tool-approval-policies", tags=["Approval Policies"], status_code=201)
def create_policy(body: PolicyUpsert):
    store = _store()
    try:
        row = store.create(body.writable())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return row.to_api()


@router.post("/v1/tool-approval-policies/seed-defaults", tags=["Approval Policies"])
def seed_defaults():
    store = _store()
    seeded = store.seed_defaults()
    return {"seeded": seeded, "total": len(store.list_all())}


class PolicyPreview(BaseModel):
    backend: str = Field(min_length=1, max_length=64)
    tool_name: str = Field(min_length=1, max_length=128)
    tool_input: dict[str, Any]
    policies: list[dict[str, Any]] | None = Field(default=None, max_length=1000)


@router.post("/v1/tool-approval-policies/preview", tags=["Approval Policies"])
def preview_policy(body: PolicyPreview):
    try:
        policies = (_store().compile_bundle().policies if body.policies is None else
                    [candidate_policy(item, index) for index, item in enumerate(body.policies)])
        decision = evaluate(policies, body.backend, body.tool_name, body.tool_input)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (ApprovalStoreUnavailable, SQLAlchemyError) as exc:
        raise HTTPException(status_code=503, detail="Approval policy database unavailable") from exc
    return {"action": decision.action.value, "severity": decision.severity.value,
            "policy_id": decision.policy.id if decision.policy else None,
            "subject": decision.subject,
            "reason": "First matching policy" if decision.policy else f"No matching policy; default {decision.action.value}"}


@router.get("/v1/tool-approval-policies/status", tags=["Approval Policies"])
def policy_status():
    store = get_tool_approval_policy_store(get_service().config)
    try:
        bundle = store.compile_bundle()
        available, etag = True, bundle.etag
    except (ApprovalStoreUnavailable, SQLAlchemyError):
        available, etag = False, None
    return {
        "database_available": available, "bundle_etag": etag, "default_action": "allow",
        "coverage": [
            {"backend": "mcp", "enforcement": "server", "scope": "first-party bawthub tools", "configured": True},
            {"backend": "claude-code", "enforcement": "bridge_hooks", "scope": "native tools and external MCP", "configured": None},
            {"backend": "codex", "enforcement": "unsupported", "configured": None},
            {"backend": "openclaw", "enforcement": "unsupported", "configured": None},
            {"backend": "direct", "enforcement": "unsupported", "configured": None},
        ],
        "notes": ["Bridge runtime configuration and reload acknowledgements are not reported by this API.",
                  "Decision audit covers first-party MCP evaluations and Claude bridge gate events; Codex/OpenClaw native tools and direct clients are uncovered.",
                  "Claude audit uses best-effort Redis delivery and app-side database commits; transport outages, app loss or exhausted persistence retries can leave gaps. Audit failure never changes a bridge gate decision.",
                  "Audit subjects are fully redacted; invocation hashes correlate exact calls without storing their inputs.",
                  "Published reload means Redis accepted invalidation, not that any bridge installed this bundle.",
                  "Policy edits do not rewrite already-approved MCP snapshots; Claude retries re-evaluate current policies and hard deny still wins.",
                  "Approved operations use immutable snapshots; disabled/deleted operations block new invocations only."],
    }


@router.get("/v1/tool-approval-policies/{policy_id}/revisions", tags=["Approval Policies"])
def policy_revisions(policy_id: str, limit: int = 50, offset: int = 0):
    limit, offset = min(max(limit, 1), 200), max(0, offset)
    try:
        rows, total = _store().page_revisions(policy_id, limit=limit, offset=offset)
    except (ApprovalStoreUnavailable, SQLAlchemyError) as exc:
        raise HTTPException(status_code=503, detail="Approval policy database unavailable") from exc
    return {"revisions": [row.to_api() for row in rows], "total": total,
            "limit": limit, "offset": offset}


@router.get("/v1/tool-approval-decisions", tags=["Approval Policies"])
def list_decisions(limit: int = 50, offset: int = 0):
    limit, offset = min(max(limit, 1), 200), max(0, offset)
    rows, total = _store().list_decisions(limit=limit, offset=offset)
    return {"decisions": [row.to_api() for row in rows], "total": total, "limit": limit, "offset": offset}


@router.get("/v1/tool-approval-policies/{policy_id}", tags=["Approval Policies"])
def get_policy(policy_id: str):
    store = _store()
    row = store.get(policy_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Policy '{policy_id}' not found")
    return row.to_api()


@router.patch("/v1/tool-approval-policies/{policy_id}", tags=["Approval Policies"])
def update_policy(policy_id: str, body: PolicyUpsert):
    store = _store()
    try:
        row = store.update(policy_id, body.writable())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if row is None:
        raise HTTPException(status_code=404, detail=f"Policy '{policy_id}' not found")
    return row.to_api()


@router.delete("/v1/tool-approval-policies/{policy_id}", tags=["Approval Policies"])
def delete_policy(policy_id: str):
    store = _store()
    if not store.delete(policy_id):
        raise HTTPException(status_code=404, detail=f"Policy '{policy_id}' not found")
    return {"success": True, "id": policy_id}


@router.post("/v1/admin/reload-tool-approval-policies", tags=["Admin"])
async def reload_policies():
    """Publish cache invalidation; delivery is not a bridge installation ACK.

    Claude also refreshes on its cache TTL. Other bridges have no gate here.
    """
    store = _store()
    try:
        bundle = store.compile_bundle()
    except (ApprovalStoreUnavailable, SQLAlchemyError) as exc:
        raise HTTPException(status_code=503, detail="Approval policy database unavailable") from exc
    published, subscribers = False, None
    try:
        sub = _subscriber()
        if sub is not None:
            subscribers = await sub.publish_approval_reload()
            # None from an older implementation is not proof of publication.
            published = isinstance(subscribers, int) and not isinstance(subscribers, bool)
    except Exception:  # noqa: BLE001
        log.warning("Approval reload publish failed")
    return {"status": "published" if published else "publish_failed",
            "published": published, "subscribers": subscribers,
            "bridge_installed": "unknown", "etag": bundle.etag,
            "policies": len(bundle.policies)}


# ---------------------------------------------------------------------------
# Request audit + resolve
# ---------------------------------------------------------------------------


def _decode_stored_args(row) -> dict[str, Any]:
    try:
        value = json.loads(row.tool_arguments_json or "{}")
    except (json.JSONDecodeError, TypeError) as error:
        raise RuntimeError("stored MCP tool arguments are malformed") from error
    if not isinstance(value, dict):
        raise RuntimeError("stored MCP tool arguments are not an object")
    return value


def _normalize_mcp_result(result: Any) -> Any:
    """Convert FastMCP content blocks/results into a durable JSON value."""
    if isinstance(result, dict):
        return result
    if isinstance(result, (str, int, float, bool)) or result is None:
        return result
    if isinstance(result, (list, tuple)):
        normalized: list[Any] = []
        for block in result:
            text_value = getattr(block, "text", None)
            if isinstance(text_value, str):
                try:
                    normalized.append(json.loads(text_value))
                except json.JSONDecodeError:
                    normalized.append(text_value)
                continue
            model_dump = getattr(block, "model_dump", None)
            normalized.append(model_dump() if callable(model_dump) else str(block))
        return normalized[0] if len(normalized) == 1 else normalized
    model_dump = getattr(result, "model_dump", None)
    return model_dump() if callable(model_dump) else str(result)


def _mcp_result_is_error(result: Any) -> bool:
    if getattr(result, "isError", False) or getattr(result, "is_error", False):
        return True
    if isinstance(result, dict):
        if (result.get("isError") or result.get("is_error") or result.get("error")
                or result.get("success") is False or result.get("ok") is False
                or result.get("status") in ("failed", "error", "denied")
                or result.get("state") in ("failed", "error", "timed_out", "lost", "cancelled")):
            return True
        return any(_mcp_result_is_error(result[key]) for key in ("content", "structuredContent") if key in result)
    if isinstance(result, (list, tuple)):
        return any(_mcp_result_is_error(item) for item in result)
    text = getattr(result, "text", None)
    if isinstance(text, str):
        try:
            return _mcp_result_is_error(json.loads(text))
        except json.JSONDecodeError:
            pass
    return False


def _stored_mcp_resolution(row) -> dict[str, Any]:
    result: Any = None
    if row.result_json:
        try:
            result = json.loads(row.result_json)
        except json.JSONDecodeError:
            result = row.result_json
    return {
        "ok": row.execution_state in (EXEC_SUCCEEDED, EXEC_SKIPPED) and not bool(row.result_is_error),
        "detail": "already_resolved",
        "status": row.status,
        "request_id": row.id,
        "tool_use_id": row.tool_use_id,
        "bot_id": row.bot_id,
        "parent_turn_id": row.turn_id,
        "already_resolved": True,
        "request_kind": KIND_MCP,
        "execution_state": row.execution_state,
        "result": result,
        "result_is_error": bool(row.result_is_error),
        "continuation_prompt": None,
        "server_dispatched": row.continuation_state in (CONT_PENDING, CONT_DISPATCHING, CONT_DELIVERED),
        "continuation_status": row.continuation_state,
    }


async def _resolve_mcp_request(store, row, *, outcome: str, message: str, resolved_by: str | None):
    """Resolve and claim one MCP invocation; replay only idempotent operations."""
    new_status = {
        "approve": REQ_APPROVED,
        "deny": REQ_DENIED,
        "cancel": REQ_CANCELLED,
        "respond": REQ_RESPONDED,
    }[outcome]
    updated = store.resolve_request(row.id, status=new_status, resolved_by=resolved_by, message=message, continuation_owner="server")
    if updated is None:
        raise HTTPException(status_code=404, detail="Request disappeared during resolve")

    # Concurrent opposing resolvers must obey the persisted winner, not their body.
    row = updated
    outcome = {REQ_APPROVED: "approve", REQ_DENIED: "deny", REQ_CANCELLED: "cancel", REQ_RESPONDED: "respond"}.get(row.status, "deny")
    message = row.resolution_message
    if row.execution_state in (EXEC_SUCCEEDED, EXEC_FAILED, EXEC_SKIPPED):
        return _stored_mcp_resolution(row)
    if outcome != "approve":
        if outcome == "cancel":
            payload = {"status": "cancelled", "message": "The MCP call was cancelled."}
        elif outcome == "respond":
            payload = {
                "status": "responded",
                "message": message or "The user chose not to run the MCP call.",
            }
        else:
            payload = {"status": "denied", "message": "The MCP call was denied."}
        completed = store.complete_mcp_execution(
            row.id,
            result_json=json.dumps(payload, ensure_ascii=False),
            is_error=outcome != "cancel",
            skipped=True,
        )
    else:
        claimed = store.claim_mcp_execution(row.id)
        if claimed is None:
            current = store.get_request(row.id)
            return _stored_mcp_resolution(current or updated)
        try:
            from ...mcp_server.approval_interceptor import ApprovedCallerContext
            from ...mcp_server.registry import mcp

            stored_args = _decode_stored_args(claimed)
            if claimed.tool_name == "ops_run" and not claimed.operations_snapshot_json:
                raise ValueError("Legacy operation approval has no immutable snapshot; create a new approval")
            from ..approval_execution import run_with_execution_lease

            result = await run_with_execution_lease(store, claimed, mcp.call_approved_tool(
                claimed.tool_name,
                stored_args,
                expected_invocation_hash=claimed.invocation_hash or "",
                # Provenance the tool records in its own ledger. Read from the
                # persisted row, never from agent-supplied input.
                caller_context=ApprovedCallerContext(
                    bot_id=claimed.bot_id or "",
                    user_id=claimed.user_id or "",
                    turn_id=claimed.turn_id or "",
                    session_key=claimed.session_key or "",
                    backend=claimed.backend or "",
                    approval_request_id=claimed.id,
                    operations_snapshot=json.loads(claimed.operations_snapshot_json) if claimed.operations_snapshot_json else None,
                ),
            ))
            payload = _normalize_mcp_result(result)
            is_error = _mcp_result_is_error(result) or _mcp_result_is_error(payload)
            execution_error = "Tool returned an error result" if is_error else None
        except ApprovalExecutionLeaseLost:
            raise
        except Exception as error:  # noqa: BLE001
            log.exception("Approved MCP execution failed id=%s", row.id)
            payload = {"status": "failed", "error": str(error)}
            is_error, execution_error = True, str(error)
        # Persistence failures are not tool failures; leave the claim for safe recovery.
        completed = store.complete_mcp_execution(
            row.id, result_json=json.dumps(payload, ensure_ascii=False, default=str),
            is_error=is_error, error=execution_error, claim_token=claimed.execution_claim_token,
        )
    if completed is None:
        raise HTTPException(status_code=500, detail="Could not persist MCP execution result")
    response = _stored_mcp_resolution(completed)
    response["detail"] = completed.status
    response["already_resolved"] = False
    return response


@router.get("/v1/tool-approval-requests", tags=["Approval Policies"])
def list_requests(status: str | None = None, bot_id: str | None = None, limit: int = 50, offset: int = 0):
    store = _store()
    limit, offset = min(max(limit, 1), 200), max(offset, 0)
    rows = store.list_requests(status=status, bot_id=bot_id, limit=limit, offset=offset)
    return {"requests": [r.to_api() for r in rows],
            "total": store.count_requests(status=status, bot_id=bot_id), "limit": limit, "offset": offset}


@router.get("/v1/tool-approval-requests/{request_id}/result", tags=["Approval Policies"])
def request_result(request_id: str):
    row = _store().get_request(request_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Approval request not found")
    return _stored_mcp_resolution(row)


@router.post("/v1/chat/approvals/{request_id}/resolve", tags=["Approval Policies"])
async def resolve_approval(request_id: str, body: ResolveRequest):
    """Approve, deny, cancel, or respond to a pending gated tool call.

    On approve: record it, grant the bridge a one-shot allow keyed by the
    request's grant_key, and return a continuation prompt the client dispatches
    so the model re-issues the now-allowed call. On deny: record it and return a
    prompt telling the model it was refused. On cancel: record it and return a
    null continuation — the request is silently dropped without warning the
    agent (no grant, no token-costing acknowledgement). On respond: the tool is
    NOT run (no grant), but the user's own ``message`` becomes the continuation
    instead of the canned refusal — for correcting false-positive gates with
    bespoke guidance. Idempotent on already-resolved.
    """
    store = _store()
    row = store.get_request(request_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"No approval request id={request_id}")

    decision = (body.decision or "").strip().lower()
    if decision in ("approve", "approved", "allow"):
        outcome = "approve"
    elif decision in ("deny", "denied", "reject"):
        outcome = "deny"
    elif decision in ("cancel", "cancelled", "canceled", "abort", "dismiss"):
        outcome = "cancel"
    elif decision in ("respond", "reply", "guide", "message"):
        outcome = "respond"
    else:
        raise HTTPException(
            status_code=400,
            detail="decision must be 'approve', 'deny', 'cancel', or 'respond'",
        )
    approved = outcome == "approve"

    bot_id = row.bot_id
    user_id = row.user_id
    message = (body.message or "").strip()

    if row.request_kind == KIND_ORCHESTRATION:
        # A durable server-side release coordinator owns the follow-up. Never
        # grant an agent tool, send a continuation, or run the deploy here.
        already_resolved = row.status != "pending"
        new_status = {"approve": REQ_APPROVED, "deny": REQ_DENIED,
                      "cancel": REQ_CANCELLED, "respond": REQ_RESPONDED}[outcome]
        row = store.resolve_request(
            request_id, status=new_status, resolved_by=body.resolved_by,
            message=message, continuation_owner="none",
        )
        if row is None:
            raise HTTPException(status_code=404, detail="Request disappeared during resolve")
        await _fanout_resolved(_subscriber(), bot_id, user_id, request_id, row.turn_id, row.status)
        return {
            "ok": True, "detail": "already_resolved" if already_resolved else row.status,
            "status": row.status, "request_id": request_id, "bot_id": bot_id,
            "continuation_prompt": None, "parent_turn_id": row.turn_id,
            "already_resolved": already_resolved,
            "cancelled": row.status == REQ_CANCELLED,
            "server_dispatched": False, "continuation_owner": "none",
        }

    if row.request_kind == KIND_MCP:
        # MCP approvals are server-owned: execute the exact stored invocation
        # once, persist its actual result, and never ask the model to re-issue.
        result = await _resolve_mcp_request(
            store,
            row,
            outcome=outcome,
            message=message,
            resolved_by=body.resolved_by,
        )
        # The SDK call already ended with an ``approval_required`` placeholder.
        # Replace that exact card's persisted result with the server-owned
        # execution outcome. Approval IDs and SDK tool-use IDs are distinct on
        # this path; never assume request_id is the tool identity.
        try:
            turn_store = get_turn_log_store()
            ToolCallStore(turn_store.engine).resolve_approval_result(
                tool_use_id=row.tool_use_id or "",
                approval_request_id=request_id,
                approval_status=result["status"],
                result=result.get("result"),
                is_error=bool(result.get("result_is_error")),
            )
        except Exception:
            log.debug("Could not reconcile MCP approval tool card %s", request_id, exc_info=True)
        await _fanout_resolved(
            _subscriber(), bot_id, user_id, request_id, row.turn_id, result["status"],
            tool_use_id=row.tool_use_id,
            result=result.get("result"),
            is_error=bool(result.get("result_is_error")),
        )
        return result

    already_resolved = row.status != "pending"
    new_status = {"approve": REQ_APPROVED, "deny": REQ_DENIED,
                  "cancel": REQ_CANCELLED, "respond": REQ_RESPONDED}[outcome]
    row = store.resolve_request(
        request_id, status=new_status, resolved_by=body.resolved_by, message=message,
        continuation_owner="server" if body.dispatch_continuation else "client",
    )
    if row is None:
        raise HTTPException(status_code=404, detail="Request disappeared during resolve")
    approved = row.status == REQ_APPROVED
    prompt = (None if row.status == REQ_CANCELLED else
              build_respond_prompt(row.resolution_message, row.subject, row.tool_name)
              if row.status == REQ_RESPONDED else
              build_continuation_prompt(approved, row.subject, row.tool_name))
    try:
        get_turn_log_store().set_approval_status(
            tool_use_id=request_id, approval_request_id=request_id, approval_status=row.status)
    except Exception:
        log.debug("Could not stamp approval tool card %s", request_id, exc_info=True)

    if prompt and row.continuation_owner == "server":
        row = store.prepare_harness_continuation(request_id)
    elif approved and row.continuation_owner == "client" and row.grant_state != "sent":
        subscriber = _subscriber()
        if subscriber is None:
            store.set_grant_state(request_id, "failed", "Approval bridge unavailable")
            raise HTTPException(status_code=503, detail="Approval recorded, but bridge grant unavailable; retry resolution")
        if store.claim_client_grant(request_id):
            try:
                await subscriber.send_approval_grant(
                    session_key=row.session_key or "main", grant_key=row.grant_key,
                    backend=row.backend, request_id=request_id)
            except Exception as exc:
                # Redis may have accepted a send whose response was lost. Never blindly re-grant.
                store.set_grant_state(request_id, "uncertain", str(exc))
                raise HTTPException(status_code=503, detail="Approval grant delivery uncertain; manual reconciliation required") from exc
            store.set_grant_state(request_id, "sent")
        else:
            row = store.get_request(request_id)
            if row.grant_state != "sent":
                raise HTTPException(status_code=503, detail="Approval grant is dispatching or uncertain; do not dispatch continuation")
    await _fanout_resolved(_subscriber(), bot_id, user_id, request_id, row.turn_id, row.status)
    server_owned = row.continuation_owner == "server" and prompt is not None
    return {
        "ok": True, "detail": "already_resolved" if already_resolved else row.status,
        "status": row.status, "request_id": request_id, "bot_id": bot_id,
        "continuation_prompt": None if server_owned else prompt, "parent_turn_id": row.turn_id,
        "already_resolved": already_resolved, "cancelled": row.status == REQ_CANCELLED,
        "server_dispatched": server_owned and row.continuation_state in (CONT_PENDING, CONT_DISPATCHING, CONT_DELIVERED),
        "continuation_status": row.continuation_state, "continuation_owner": row.continuation_owner,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _subscriber():
    try:
        from ...agent_backends.agent_bridge import get_agent_subscriber

        return get_agent_subscriber()
    except Exception:  # noqa: BLE001
        return None


async def _fanout_resolved(
    subscriber,
    bot_id,
    user_id,
    request_id,
    turn_id,
    status,
    *,
    tool_use_id=None,
    result=None,
    is_error=None,
) -> None:
    """Fan out resolution and, for MCP, its terminal original-call result."""
    if subscriber is None:
        return
    try:
        await subscriber._redis.xadd(
            f"events:{bot_id}:{user_id}",
            {"payload": json.dumps({
                "_type": "approval_resolved",
                "bot_id": bot_id,
                "user_id": user_id,
                "request_id": request_id,
                "turn_id": turn_id,
                "status": status,
                "tool_use_id": tool_use_id,
                "result": result,
                "is_error": is_error,
                "ts": time.time(),
            }, ensure_ascii=False, default=str)},
            maxlen=5000,
            approximate=True,
        )
    except Exception:  # noqa: BLE001
        log.debug("failed to publish approval_resolved for %s", request_id, exc_info=True)
