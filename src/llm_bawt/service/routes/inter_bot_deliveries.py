"""Durable inter-bot delivery enqueue and inspection API."""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from ...agent_context import (
    AgentContextStore,
    normalize_session_policy,
    preventive_session_policy,
    validate_session_policy,
)
from ...bot_types import durable_agent_backend
from ...bots import BotManager
from ...inter_bot_delivery import submission_result
from ...inter_bot_delivery_user import resolve_delivery_user
from ...inter_bot_delivery_retry import DeliveryRetryConflict, retry_failed_delivery
from ...message_authorship import AuthorReference, normalize_author
from ...request_identity import AdmissionUnavailable, IdentityConflict, InvalidIdentity
from ...model_catalog import bot_model_ref
from ..dependencies import get_service
from ..delivery_reply import DeliveryReplyMismatch, resolve_delivery_reply
from ..inter_bot_reply_view import ReplyViewMismatch, inter_bot_reply_view

router = APIRouter()


class DeliveryCreate(BaseModel):
    sender_bot_id: str = "unknown"
    author_entity_type: Literal["user", "bot"] | None = None
    author_entity_id: str | None = Field(default=None, max_length=128)
    target_bot_id: str
    message: str
    max_tokens: int | None = None
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    timeout_seconds: float = Field(default=1800.0, ge=1.0, le=86400.0)
    idempotency_key: str | None = Field(default=None, max_length=256)
    project_id: str | None = Field(default=None, max_length=128)
    task_id: str | None = Field(default=None, max_length=128)
    message_kind: Literal["READY", "BLOCKED", "PROGRESS"] | str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    reply_mode: Literal["async", "waited"] = "async"
    max_attempts: int = Field(default=5, ge=1, le=20)
    prefer_steer: bool = Field(
        default=True,
        description="Steer an active Claude turn; false waits for one safe idle turn.",
    )
    session_policy: Literal[
        "continue", "reset_retain_history", "reset_without_history"
    ] | None = Field(
        default=None,
        description="Pre-delivery SDK/durable-thread policy; default continue.",
    )
    reset_session_before_delivery: bool | None = Field(
        default=None,
        description="Compatibility field normalized into session_policy.",
    )
    retain_history: bool | None = Field(
        default=None,
        description="Compatibility reset-history choice; requires reset_session_before_delivery.",
    )
    reset_reason: str | None = Field(default=None, max_length=1000)


def _dispatcher():
    service = get_service()
    dispatcher = service._inter_bot_dispatcher
    if dispatcher is None or dispatcher.store.engine is None:
        raise HTTPException(status_code=503, detail="Inter-bot delivery service unavailable")
    return dispatcher


def _format_delivery_message(
    message: str, author: AuthorReference, reply_mode: str = "async",
) -> str:
    if author.entity_type == "bot" and author.entity_id != "unknown":
        formatted = f"Message from bot '{author.entity_id}': {message}"
        if reply_mode == "waited":
            formatted += (
                "\n\n[Reply channel: awaited delivery. Your final response on this turn "
                "is the answer to the sending bot's tool call (or its durable receipt "
                "if the wait times out). Give the answer in your final response. "
                "Do not send a separate message back merely to repeat that answer; "
                "an intentional, distinct follow-up is still allowed by setting "
                "separate_message_reason on bots_send_message.]"
            )
        return formatted
    return message


def _target_context_window(config: Any, target_bot: Any) -> int | None:
    """Resolve context health from the bot's pinned catalog endpoint."""
    model_ref = bot_model_ref(config, target_bot)
    return config.get_model_context_window(model_ref) if model_ref else None


@router.post("/v1/inter-bot-deliveries", tags=["Inter-Bot Deliveries"], status_code=202)
async def create_delivery(body: DeliveryCreate):
    """Persist a callback and dispatch it through the target's next idle turn."""
    target = body.target_bot_id.strip().lower()
    sender = body.sender_bot_id.strip().lower() or "unknown"
    try:
        explicit_author = normalize_author(
            body.author_entity_type,
            body.author_entity_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    author = explicit_author or AuthorReference.bot(sender)
    if body.reply_mode == "waited" and (body.prefer_steer or author.entity_type != "bot" or sender == "unknown"):
        raise HTTPException(status_code=422, detail="Waited replies require a bot-authored when-idle delivery")
    if not target:
        raise HTTPException(status_code=422, detail="target_bot_id is required")
    if not body.message.strip():
        raise HTTPException(status_code=422, detail="message is required")
    target_bot = BotManager(get_service().config).get_bot(target)
    target_exists = target_bot is not None
    durable_capable = bool(target_bot and durable_agent_backend(target_bot))
    try:
        delivery_user = resolve_delivery_user(
            get_service().config,
            author.entity_id if author.entity_type == "user" else None,
        )
        policy = normalize_session_policy(
            body.session_policy,
            reset_session_before_delivery=body.reset_session_before_delivery,
            retain_history=body.retain_history,
        )
        backend = (target_bot.agent_backend or target_bot.harness) if target_bot else None
        automatic_reason = None
        policy_was_explicit = any(
            value is not None
            for value in (
                body.session_policy,
                body.reset_session_before_delivery,
                body.retain_history,
            )
        )
        if target_bot and policy.value == "continue" and not policy_was_explicit:
            from ...runtime_settings import RuntimeSettingsResolver
            from ...setting_definitions import setting_default

            resolver = RuntimeSettingsResolver(
                get_service().config, bot=target_bot
            )
            warning_percent = max(1, min(99, int(resolver.resolve(
                "agent_context_warning_percent",
                fallback=setting_default("agent_context_warning_percent", 75),
            ))))
            critical_percent = max(
                warning_percent + 1,
                min(100, int(resolver.resolve(
                    "agent_context_critical_percent",
                    fallback=setting_default("agent_context_critical_percent", 90),
                ))),
            )
            ceiling = _target_context_window(get_service().config, target_bot)
            health = AgentContextStore(get_service().config).health(
                bot_id=target,
                user_id=delivery_user,
                backend=backend,
                configured_ceiling=ceiling,
                warning_ratio=warning_percent / 100,
                critical_ratio=critical_percent / 100,
            )
            policy, automatic_reason = preventive_session_policy(
                requested=policy,
                health_state=health["state"],
                configured_critical_policy=str(resolver.resolve(
                    "agent_context_critical_policy",
                    fallback=setting_default(
                        "agent_context_critical_policy", "reset_retain_history"
                    ),
                )),
                backend=backend,
            )
        if target_bot:
            validate_session_policy(backend, policy)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    # Maintenance is a harness command, not a conversational bot callback.
    # Prefixing it with sender attribution prevents the CLI recognizing /compact.
    # Keep author metadata on the durable row; never unwrap ordinary messages.
    compact_maintenance = (
        body.message_kind == "CONTEXT_MAINTENANCE"
        and body.metadata.get("context_action") == "compact"
    )
    if compact_maintenance and (
        backend != "claude-code"
        or body.prefer_steer
        or body.reply_mode == "waited"
        or body.message.split(maxsplit=1)[0] != "/compact"
    ):
        raise HTTPException(status_code=422, detail="Compaction maintenance requires an idle Claude /compact command")
    formatted = body.message if compact_maintenance else _format_delivery_message(
        body.message, author, body.reply_mode,
    )
    payload = {
        "messages": [{"role": "user", "content": formatted}],
        "reply_mode": body.reply_mode,
        "prefer_steer": body.prefer_steer,
        "bot_id": target,
        "user": delivery_user,
        "max_tokens": body.max_tokens,
        "temperature": body.temperature,
        "extract_memory": False,
        "augment_memory": True,
        "stream": False,
        "inter_bot_timeout_seconds": body.timeout_seconds,
    }
    dispatcher = _dispatcher()
    try:
        record, duplicate = dispatcher.store.enqueue(
            sender_bot_id=sender, target_bot_id=target, message=body.message,
            payload=payload, author=author, idempotency_key=body.idempotency_key,
            project_id=body.project_id, task_id=body.task_id,
            message_kind=body.message_kind, metadata=body.metadata,
            max_attempts=body.max_attempts, session_policy=policy,
            reset_reason=body.reset_reason or automatic_reason, producer="bot",
        )
    except (IdentityConflict, InvalidIdentity, AdmissionUnavailable) as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    if not duplicate and not target_exists:
        record = dispatcher.store.mark_failed(record.id, f"Unknown target bot '{target}'") or record
    elif not duplicate and not durable_capable:
        record = dispatcher.store.mark_failed(
            record.id,
            f"Target bot '{target}' does not support durable when-idle delivery",
        ) or record
    if not duplicate:
        await dispatcher._emit(record)
    if durable_capable:
        dispatcher.wake(target)
    current = dispatcher.store.get(record.id) or record
    stored_payload = dispatcher.store.payload(current.id) or {}
    status_code, result = submission_result(
        current,
        duplicate=duplicate,
        target_exists=target_exists,
        requested_delivery=(
            "steer_or_idle" if stored_payload.get("prefer_steer", True) else "when_idle"
        ),
    )
    if status_code != 202:
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=status_code, content=result)
    return result


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
        # Manual Retry (TASK-1111) creates a distinct, idempotent delivery
        # without another send tool call in the origin thread. Keep the original
        # card as its single anchor and follow only the verified retry child.
        retries = dispatcher.store.list(
            sender_bot_id=sender_bot_id, target_bot_id=record.target_bot_id,
            idempotency_key=f"manual-retry:{record.id}", limit=1,
        )
        if retries and retries[0].metadata.get("manual_retry_of") == record.id:
            child = retries[0]
            if (dispatcher.store.payload(child.id) or {}).get("user") == user_id:
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
    status: str | None = None,
    idempotency_key: str | None = Query(None, max_length=256),
    limit: int = Query(50, ge=1, le=200),
):
    rows = _dispatcher().store.list(
        sender_bot_id=sender_bot_id,
        target_bot_id=target_bot_id,
        status=status,
        idempotency_key=idempotency_key,
        limit=limit,
        exclude_prompt_origin=True,
    )
    return {"deliveries": [row.to_api() for row in rows], "total": len(rows)}


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
