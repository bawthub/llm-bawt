"""Durable inter-bot delivery enqueue and inspection API."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from fastapi import APIRouter, HTTPException, Query, Request

from ...bot_types import durable_agent_backend
from ...bots import BotManager
from ...inter_bot_delivery_query import IN_FLIGHT_STATUSES, parse_statuses
from ...inter_bot_delivery_retry import DeliveryRetryConflict, retry_failed_delivery
from ...inter_bot_send import (  # noqa: F401  (re-exported for importers and tests)
    PRIVILEGED_FIELDS,
    BatchDeliveryCreate,
    DeliveryCreate,
    DeliverySender,
    _format_delivery_message,
    _gate_disposition,
    _target_context_window,
)
from ...request_identity import AdmissionUnavailable, IdentityConflict, InvalidIdentity
from ..dependencies import get_service
from ..delivery_reply import DeliveryReplyMismatch, resolve_delivery_reply
from ..inter_bot_reply_view import ReplyViewMismatch, inter_bot_reply_view

router = APIRouter()


def _dispatcher():
    service = get_service()
    dispatcher = service._inter_bot_dispatcher
    if dispatcher is None or dispatcher.store.engine is None:
        raise HTTPException(status_code=503, detail="Inter-bot delivery service unavailable")
    return dispatcher


def _sender() -> DeliverySender:
    return DeliverySender(lambda: get_service(), dispatcher=_dispatcher)


def _respond(status_code: int, result: dict):
    if status_code != 202:
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=status_code, content=result)
    return result


@router.post("/v1/inter-bot-deliveries", tags=["Inter-Bot Deliveries"], status_code=202)
async def create_delivery(body: DeliveryCreate, request: Request = None):  # type: ignore[assignment]
    """Persist a callback and dispatch it through the target's next idle turn.

    Service callers (BawtHub task dispatch) may also pin the trigger UUID, the
    target thread and memory retrieval. Those privileged fields are refused
    with 403 ``privileged_field`` on a request carrying a bot turn capability.
    That stops tool-level injection, not a direct curl to this unauthenticated
    route (TASK-1420).
    """
    return _respond(*await _sender().admit(body, request))


@router.post("/v1/inter-bot-deliveries/batch", tags=["Inter-Bot Deliveries"], status_code=202)
async def create_delivery_batch(body: BatchDeliveryCreate, request: Request = None):  # type: ignore[assignment]
    """Fan one message out to a target selector (TASK-1431).

    ``targets`` is ``{"bots": [...]}`` (or a bare list), ``{"all_agents": true}``,
    ``{"backends": [...]}`` or ``{"tags": [...]}``. Private and inactive bots
    never come out of a selector. Returns ``{batch_id, mode, dispatched,
    deliveries, refused}``; a bot sender needs an ``idempotency_key`` and may
    start at most 3 batches per 10 minutes.
    """
    return _respond(*await _sender().submit(body, request))


@router.get("/v1/inter-bot-deliveries/{delivery_id}", tags=["Inter-Bot Deliveries"])
def get_delivery(delivery_id: str):
    record = _dispatcher().store.get(delivery_id)
    if record is None or "prompt_schedule" in (record.metadata or {}):
        raise HTTPException(status_code=404, detail="Delivery not found")
    return record.to_api()


@router.get("/v1/inter-bot-deliveries/{delivery_id}/reply", tags=["Inter-Bot Deliveries"])
def get_delivery_reply(
    delivery_id: str,
    bot_id: str = Query(..., min_length=1),
    user_id: str = Query(..., min_length=1),
    user_message_id: str = Query(..., min_length=1),
):
    """Read the exact queued turn's final text, never an unverified ACK."""
    dispatcher = _dispatcher()
    try:
        reply = resolve_delivery_reply(
            dispatcher.store, get_service()._turn_log_store, delivery_id,
            bot_id=bot_id, user_id=user_id, user_message_id=user_message_id,
        )
    except DeliveryReplyMismatch as exc:
        raise HTTPException(status_code=404, detail="Delivery not found") from exc
    if reply is None:
        raise HTTPException(status_code=404, detail="Delivery not found")
    return reply


@router.get("/v1/inter-bot-deliveries/{delivery_id}/answer", tags=["Inter-Bot Deliveries"])
def get_inter_bot_answer(
    delivery_id: str,
    sender_bot_id: str = Query(..., min_length=1),
    user_id: str = Query(..., min_length=1),
):
    """Show a sender the exact, persisted reply to its own dedicated turn."""
    dispatcher = _dispatcher()
    record = dispatcher.store.get(delivery_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Delivery not found")
    payload = dispatcher.store.payload(delivery_id) or {}
    service = get_service()
    if payload.get("user") != user_id or record.sender_bot_id != sender_bot_id:
        raise HTTPException(status_code=404, detail="Delivery not found")
    answer_record = record
    if record.status == "FAILED":
        # A manual Retry (TASK-1111) or a restart retry (TASK-1231B) is a
        # distinct, idempotent delivery with no send tool call in the origin
        # thread. Keep the original card as its single anchor and follow only
        # the verified retry child.
        from ...inter_bot_delivery_retry import verified_retry_child

        child = verified_retry_child(dispatcher.store, record)
        if child is not None and (dispatcher.store.payload(child.id) or {}).get("user") == user_id:
            answer_record = child
    try:
        view = inter_bot_reply_view(
            answer_record, sender_bot_id=sender_bot_id, user_id=user_id,
            turns=service._turn_log_store,
            memory_client=service.get_memory_client(answer_record.target_bot_id, user_id),
        )
        if answer_record is not record:
            view["retry_delivery_id"] = answer_record.id
            view["delivery_id"] = record.id  # preserve the origin tool-call identity
        return view
    except ReplyViewMismatch as exc:
        raise HTTPException(status_code=404, detail="Delivery not found") from exc


@router.get("/v1/inter-bot-deliveries", tags=["Inter-Bot Deliveries"])
def list_deliveries(
    sender_bot_id: str | None = None,
    target_bot_id: str | None = None,
    bot_id: str | None = Query(None, description="Sender OR target"),
    status: str | None = Query(None, description="One status or comma-separated list"),
    active_only: bool = Query(False, description="Only QUEUED, STEERING, DISPATCHING"),
    idempotency_key: str | None = Query(None, max_length=256),
    task_id: str | None = Query(None, max_length=128),
    since: datetime | None = Query(None, description="Only rows created at/after this ISO time"),
    include_chat: bool = Query(True, description="Include users' chat queue rows and self-addressed rows"),
    view: Literal["full", "compact"] = "full",
    limit: int = Query(50, ge=1, le=200),
):
    if active_only and status:
        raise HTTPException(status_code=422, detail="Use either active_only or status, not both")
    try:
        statuses = IN_FLIGHT_STATUSES if active_only else parse_statuses(status)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    rows = _dispatcher().store.list(
        sender_bot_id=sender_bot_id,
        target_bot_id=target_bot_id,
        bot_id=bot_id,
        status=statuses,
        idempotency_key=idempotency_key,
        task_id=task_id,
        since=since,
        limit=limit,
        exclude_prompt_origin=True,
        exclude_user_chat=not include_chat,
    )
    project = (lambda row: row.to_compact()) if view == "compact" else (lambda row: row.to_api())
    return {"deliveries": [project(row) for row in rows], "total": len(rows)}


@router.post("/v1/inter-bot-deliveries/{delivery_id}/retry", tags=["Inter-Bot Deliveries"], status_code=202)
async def retry_delivery(delivery_id: str):
    """Explicitly resend one failed callback through the durable dispatcher."""
    dispatcher = _dispatcher()
    original = dispatcher.store.get(delivery_id)
    if original is None or any(key in original.metadata for key in (
        "prompt_schedule", "ordinary_chat", "maintenance_user_queue",
    )) or original.message_kind == "CONTEXT_MAINTENANCE":
        raise HTTPException(status_code=404, detail="Delivery not found")
    target = BotManager(get_service().config).get_bot(original.target_bot_id)
    if target is None or not durable_agent_backend(target):
        raise HTTPException(status_code=409, detail="Target bot cannot accept durable delivery")
    try:
        retried = retry_failed_delivery(dispatcher.store, delivery_id)
    except DeliveryRetryConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (IdentityConflict, InvalidIdentity, AdmissionUnavailable) as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    if retried is None:
        raise HTTPException(status_code=404, detail="Delivery not found")
    record, duplicate = retried
    if not duplicate:
        await dispatcher._emit(record)
    dispatcher.wake(record.target_bot_id)
    return {**record.to_api(duplicate=duplicate), "retry_of_delivery_id": delivery_id}


@router.post("/v1/inter-bot-deliveries/{delivery_id}/cancel", tags=["Inter-Bot Deliveries"])
async def cancel_delivery(delivery_id: str):
    dispatcher = _dispatcher()
    before = dispatcher.store.get(delivery_id)
    # Prompt receipts belong exclusively to the owner-scoped scheduling API.
    # MCP inter-bot tools call this route too; never pass them to store.cancel.
    if before is None or "prompt_schedule" in (before.metadata or {}):
        raise HTTPException(status_code=404, detail="Delivery not found")
    record = dispatcher.store.cancel(delivery_id)
    if record is None or record.status != "CANCELLED":
        raise HTTPException(status_code=409, detail=f"Delivery in state {before.status} cannot be cancelled")
    await dispatcher._emit(record)
    return record.to_api()
