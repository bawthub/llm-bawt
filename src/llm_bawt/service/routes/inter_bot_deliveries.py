"""Durable inter-bot delivery enqueue and inspection API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query, Request
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
from ...inter_bot_delivery_query import IN_FLIGHT_STATUSES, parse_statuses
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
    user_message_id: str | None = Field(
        default=None, max_length=64,
        description="Caller-minted trigger UUID; also the default idempotency key.",
    )
    # Privileged (see PRIVILEGED_FIELDS): they change what the target's turn sees.
    session_id: str | None = Field(
        default=None, max_length=36,
        description="Deliver into this (target bot, user) thread instead of the active one.",
    )
    augment_memory: bool | None = Field(
        default=None, description="Memory retrieval for the turn; default true.",
    )
    raw_message: bool | None = Field(
        default=None, description="Store the message verbatim, without sender attribution.",
    )


# Fields a bot tool must never set: each changes another bot's turn. The real
# fence is the MCP tool schema (bots_send_message builds a fixed body without
# them). The route additionally refuses them on any request carrying a bot turn
# capability. Neither stops a direct HTTP call: /v1/* has no service auth, the
# same as /v1/chat/completions (TASK-1420 adds it).
PRIVILEGED_FIELDS = ("session_id", "augment_memory", "raw_message")


def _privileged_fields(body: DeliveryCreate) -> list[str]:
    return [name for name in PRIVILEGED_FIELDS if name in body.model_fields_set]


def _refuse_bot_privileged(request: Request | None, fields: list[str]) -> None:
    from ...task_turn_context import TASK_TURN_CONTEXT_HEADER

    # Presence, not validity: a bot turn's request is refused even if its
    # capability has expired or is forged.
    if fields and request is not None and request.headers.get(TASK_TURN_CONTEXT_HEADER):
        raise HTTPException(status_code=403, detail={
            "error": "privileged_field", "fields": fields,
            "detail": "These fields are for service callers, not bot tools.",
        })


def _require_owned_session(session_id: str, bot_id: str, user_id: str) -> None:
    """Never deliver into another bot's or user's thread, or a deleted one.

    Archived threads are allowed: task dispatch mints them born-archived.
    """
    from sqlalchemy import text as sa_text

    engine = _dispatcher().store.engine
    with engine.connect() as conn:
        owned = conn.execute(sa_text(
            "SELECT 1 FROM sessions WHERE id=:id AND bot_id=:bot AND user_id=:user"
            " AND status <> 'deleted'"
        ), {"id": session_id, "bot": bot_id, "user": user_id}).first()
    if owned is None:
        raise HTTPException(status_code=404, detail={
            "error": "session_not_found",
            "detail": f"No session {session_id!r} for bot {bot_id!r} and this user",
        })


def _dispatcher():
    service = get_service()
    dispatcher = service._inter_bot_dispatcher
    if dispatcher is None or dispatcher.store.engine is None:
        raise HTTPException(status_code=503, detail="Inter-bot delivery service unavailable")
    return dispatcher


def _format_delivery_message(message: str, author: AuthorReference) -> str:
    """The stored delivery text: sender attribution plus the message, nothing else.

    The reply-channel note is instruction text, so it is never stored here; it
    is added to the outbound copy only (see ``inter_bot_reply_channel``).
    """
    if author.entity_type == "bot" and author.entity_id != "unknown":
        return f"Message from bot '{author.entity_id}': {message}"
    return message


def _is_real_bot(bots: BotManager, bot_id: str | None) -> bool:
    slug = (bot_id or "").strip().lower()
    return slug not in ("", "unknown") and bots.get_bot(slug) is not None


def _target_context_window(config: Any, target_bot: Any) -> int | None:
    """Resolve context health from the bot's pinned catalog endpoint."""
    model_ref = bot_model_ref(config, target_bot)
    return config.get_model_context_window(model_ref) if model_ref else None


def _gate_disposition(request: Request | None, body: "DeliveryCreate", sender: str, target: str) -> str:
    """``hold`` only for a trusted self-model continuation; every send is ``deny``.

    Trust is the signed turn capability (minted server-side, forwarded only by
    the self_model tool), never a body field: bots_send_message cannot set it.
    A continuation is the signed bot's own follow-up turn (sender == target).
    """
    from ...task_turn_context import (
        TASK_TURN_CONTEXT_HEADER, TaskTurnContextError, open_task_turn_context,
    )
    from ..system_gate_admission import ON_GATE_DENY, ON_GATE_HOLD

    capability = request.headers.get(TASK_TURN_CONTEXT_HEADER) if request is not None else None
    if not capability or (body.message_kind or "").upper() != "MODEL_SWITCH_CONTINUE":
        return ON_GATE_DENY
    try:
        caller = open_task_turn_context(capability)
    except TaskTurnContextError:
        return ON_GATE_DENY
    own = caller.bot_id.strip().lower() == sender == target
    return ON_GATE_HOLD if own else ON_GATE_DENY


@router.post("/v1/inter-bot-deliveries", tags=["Inter-Bot Deliveries"], status_code=202)
async def create_delivery(body: DeliveryCreate, request: Request = None):  # type: ignore[assignment]
    """Persist a callback and dispatch it through the target's next idle turn.

    Service callers (BawtHub task dispatch) may also pin the trigger UUID, the
    target thread and memory retrieval. Those privileged fields are refused
    with 403 ``privileged_field`` on a request carrying a bot turn capability.
    That stops tool-level injection, not a direct curl to this unauthenticated
    route (TASK-1420).
    """
    target = body.target_bot_id.strip().lower()
    sender = body.sender_bot_id.strip().lower() or "unknown"
    privileged = _privileged_fields(body)
    _refuse_bot_privileged(request, privileged)
    on_gate = _gate_disposition(request, body, sender, target)
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
    # Only NewSessionLifecycle stamps this key. Receipts, recovery and scheduled
    # reset continuity trust its origin, so a caller must never supply one.
    if "new_session_intent" in body.metadata:
        raise HTTPException(status_code=422, detail="new_session_intent is server-owned delivery metadata")
    bots = BotManager(get_service().config)
    if author.entity_type == "bot" and not _is_real_bot(bots, author.entity_id):
        # A bot-authored row renders as that bot; "unknown" is not one (TASK-1241).
        raise HTTPException(
            status_code=422,
            detail=f"Bot-authored deliveries need a real sender bot, got {author.entity_id!r}",
        )
    target_bot = bots.get_bot(target)
    target_exists = target_bot is not None
    durable_capable = bool(target_bot and durable_agent_backend(target_bot))
    # A service dispatch: fail typed before admission rather than leaving a
    # FAILED row that also burns the caller's pinned UUID.
    service_dispatch = bool(privileged) or body.user_message_id is not None
    if service_dispatch and not target_exists:
        raise HTTPException(status_code=404, detail={
            "error": "unknown_target", "detail": f"Unknown target bot {target!r}",
        })
    if service_dispatch and not durable_capable:
        # Service dispatch needs a durable agent turn; say so instead of
        # leaving a FAILED row the caller would have to discover.
        raise HTTPException(status_code=422, detail={
            "error": "target_not_durable",
            "detail": f"Target bot {target!r} does not support durable agent delivery",
        })
    # A reset admits through NewSessionLifecycle, which mints its own identity
    # and rotates the active thread, so neither pinned field survives it.
    reset_requested = (body.session_policy or "continue") != "continue" or bool(
        body.reset_session_before_delivery
    )
    for pinned in ("session_id", "user_message_id"):
        if getattr(body, pinned) is not None and reset_requested:
            raise HTTPException(status_code=422, detail=f"{pinned} cannot be combined with a session reset")
    # Steering joins whatever turn is running, in the active thread, whose
    # capability signs that turn's trigger, not the caller's pinned UUID.
    for pinned in ("session_id", "user_message_id"):
        if getattr(body, pinned) is not None and body.prefer_steer:
            raise HTTPException(status_code=422, detail=f"{pinned} requires prefer_steer=false")
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
        # A pinned thread is not the active one, so active-thread health must
        # not reset anything on its behalf; and an automatic reset would drop
        # a caller's pinned trigger UUID.
        if (target_bot and policy.value == "continue" and not policy_was_explicit
                and body.session_id is None and body.user_message_id is None):
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
    verbatim = compact_maintenance or bool(body.raw_message)
    formatted = body.message if verbatim else _format_delivery_message(body.message, author)
    payload = {
        "messages": [{"role": "user", "content": formatted}],
        "reply_mode": body.reply_mode,
        "prefer_steer": body.prefer_steer,
        "bot_id": target,
        "user": delivery_user,
        "max_tokens": body.max_tokens,
        "temperature": body.temperature,
        "extract_memory": False,
        "augment_memory": True if body.augment_memory is None else body.augment_memory,
        "stream": False,
        "inter_bot_timeout_seconds": body.timeout_seconds,
    }
    if body.session_id is not None:
        _require_owned_session(body.session_id, target, delivery_user)
        payload["session_id"] = body.session_id
    dispatcher = _dispatcher()
    try:
        if policy.value != "continue":
            from ..session_lifecycle.intent import Admission, Continuity, NewSessionIntent, Origin
            from ..session_lifecycle.lifecycle import NewSessionLifecycle

            intent = NewSessionIntent(
                bot_id=target, user_id=delivery_user, author=author,
                continuity=(Continuity.RETAIN if policy.value == "reset_retain_history"
                            else Continuity.CLEAN),
                admission=Admission.WAIT, origin=Origin.BOT_DELIVERY,
                idempotency_key=body.idempotency_key or f"bot-reset:{uuid4()}",
                follow_up=body.message, reason=body.reset_reason or automatic_reason,
            )
            record, duplicate = NewSessionLifecycle(get_service()).admit_bot_delivery(
                intent, payload=payload, metadata=body.metadata,
                sender_bot_id=sender, project_id=body.project_id, task_id=body.task_id,
                message_kind=body.message_kind, max_attempts=body.max_attempts,
                reset_reason=body.reset_reason or automatic_reason, producer="bot",
                on_gate=on_gate,
            )
        else:
            record, duplicate = dispatcher.store.enqueue(
                sender_bot_id=sender, target_bot_id=target, message=body.message,
                payload=payload, author=author,
                idempotency_key=body.idempotency_key or body.user_message_id,
                user_message_id=body.user_message_id,
                project_id=body.project_id, task_id=body.task_id,
                message_kind=body.message_kind, metadata=body.metadata,
                max_attempts=body.max_attempts, session_policy=policy,
                reset_reason=body.reset_reason or automatic_reason, producer="bot",
                on_gate=on_gate,
            )
    except (IdentityConflict, InvalidIdentity, AdmissionUnavailable) as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
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
