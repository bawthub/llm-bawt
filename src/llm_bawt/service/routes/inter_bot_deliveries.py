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
from ...bots import BotManager
from ...inter_bot_delivery import submission_result
from ...message_authorship import AuthorReference, normalize_author
from ...model_catalog import bot_model_ref
from ..dependencies import get_service

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


def _format_delivery_message(message: str, author: AuthorReference) -> str:
    if author.entity_type == "bot" and author.entity_id != "unknown":
        return f"Message from bot '{author.entity_id}': {message}"
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
    if not target:
        raise HTTPException(status_code=422, detail="target_bot_id is required")
    if not body.message.strip():
        raise HTTPException(status_code=422, detail="message is required")
    target_bot = BotManager(get_service().config).get_bot(target)
    target_exists = target_bot is not None
    durable_capable = bool(
        target_bot
        and (
            target_bot.agent_backend in {"claude-code", "codex", "openclaw"}
            or target_bot.harness in {"claude-code", "codex", "openclaw"}
        )
    )
    try:
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
                user_id=getattr(get_service().config, "DEFAULT_USER", "nick"),
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

    formatted = _format_delivery_message(body.message, author)
    payload = {
        "messages": [{"role": "user", "content": formatted}],
        "prefer_steer": body.prefer_steer,
        "bot_id": target,
        "user": (
            author.entity_id
            if author.entity_type == "user"
            else getattr(get_service().config, "DEFAULT_USER", "nick")
        ),
        "max_tokens": body.max_tokens,
        "temperature": body.temperature,
        "extract_memory": False,
        "augment_memory": True,
        "stream": False,
        "inter_bot_timeout_seconds": body.timeout_seconds,
    }
    dispatcher = _dispatcher()
    record, duplicate = dispatcher.store.enqueue(
        sender_bot_id=sender,
        target_bot_id=target,
        message=body.message,
        payload=payload,
        author=author,
        idempotency_key=body.idempotency_key,
        project_id=body.project_id,
        task_id=body.task_id,
        message_kind=body.message_kind,
        metadata=body.metadata,
        max_attempts=body.max_attempts,
        session_policy=policy,
        reset_reason=body.reset_reason or automatic_reason,
    )
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
