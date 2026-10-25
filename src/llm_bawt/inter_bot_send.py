"""One send path for durable inter-bot deliveries (TASK-1431 design r3 §1).

``DeliverySender`` admits every producer's sends. ``admit`` is the existing
single-target route, byte-compatible; ``submit`` is a fan-out batch: one
normal delivery per resolved target, each in its own target-locked
transaction, in sorted target order. ``_admit_one`` is ``create_delivery``'s
per-target body, extracted unchanged (its HTTP errors raise
:class:`DeliveryRefused`, an ``HTTPException`` with the same status and
detail), plus the ``steer_only`` mode.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal
from uuid import uuid4

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .agent_context import (
    AgentContextStore,
    normalize_session_policy,
    preventive_session_policy,
    validate_session_policy,
)
from .bot_types import durable_agent_backend
from .bots import BotManager
from .inter_bot_delivery import CANCELLED, submission_result
from .inter_bot_delivery_user import resolve_delivery_user
from .inter_bot_steer_only import (
    NOT_NEEDED_PREFIX, STEER_ONLY, STEER_ONLY_KEY, STEER_OR_IDLE, WHEN_IDLE,
)
from .message_authorship import AuthorReference, normalize_author
from .model_catalog import bot_model_ref
from .request_identity import AdmissionUnavailable, IdentityConflict, InvalidIdentity

SendModeName = Literal["steer_or_idle", "when_idle", "steer_only"]

# Fields a bot tool must never set: each changes another bot's turn. The real
# fence is the MCP tool schema (bots_send_message builds a fixed body without
# them). The route additionally refuses them on any request carrying a bot turn
# capability. Neither stops a direct HTTP call: /v1/* has no service auth, the
# same as /v1/chat/completions (TASK-1420 adds it).
PRIVILEGED_FIELDS = ("session_id", "augment_memory", "raw_message")

# Metadata only the server writes. Receipts, recovery and scheduled reset
# continuity trust ``new_session_intent``; the fan-out rate limit counts
# ``batch_id``; ``batch`` records a batch's resolved targets; ``control`` is the
# control lane's pin. A caller supplying any of them is refused on every route.
SERVER_OWNED_METADATA_KEYS = ("new_session_intent", "batch_id", "batch", "control")


class DeliveryRefused(HTTPException):
    """A send refused before any row was written.

    An HTTPException so the routes report it unchanged; typed so a batch can
    record it per target and carry on.
    """


class _DeliveryFields(BaseModel):
    """Fields every send carries, whatever its targets."""

    sender_bot_id: str = "unknown"
    author_entity_type: Literal["user", "bot"] | None = None
    author_entity_id: str | None = Field(default=None, max_length=128)
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

    @property
    def reset_requested(self) -> bool:
        return (self.session_policy or "continue") != "continue" or bool(
            self.reset_session_before_delivery
        )


class DeliveryCreate(_DeliveryFields):
    target_bot_id: str
    reply_mode: Literal["async", "waited"] = "async"
    prefer_steer: bool = Field(
        default=True,
        description="Steer an active Claude turn; false waits for one safe idle turn.",
    )
    steer_only: bool = Field(
        default=False,
        description="Steer an executing Claude turn or do nothing; never starts a turn.",
    )
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

    @model_validator(mode="after")
    def _steer_only_is_a_steer(self) -> DeliveryCreate:
        if self.steer_only and not self.prefer_steer:
            raise ValueError("steer_only requires prefer_steer=true")
        if self.steer_only and self.reset_requested:
            raise ValueError("steer_only never resets a session; use session_policy=continue")
        return self


class BatchDeliveryCreate(_DeliveryFields):
    """``POST /v1/inter-bot-deliveries/batch``. Unknown fields are refused.

    There is deliberately no ``lane``, ``target_run`` or ``control`` field
    (design r3 §2.3): until service auth exists, no HTTP body may carry them.
    """

    model_config = ConfigDict(extra="forbid")

    targets: dict[str, Any] | list[str]
    exclude: list[str] = Field(default_factory=list)
    include_self: bool = False
    mode: SendModeName = STEER_OR_IDLE
    reply_mode: Literal["async", "waited"] = "async"

    def for_target(self, target: str, metadata: dict[str, Any]) -> DeliveryCreate:
        common = self.model_dump(
            exclude={"targets", "exclude", "include_self", "mode", "metadata"},
            exclude_unset=True,
        )
        return DeliveryCreate(
            **common, target_bot_id=target, metadata=metadata,
            prefer_steer=self.mode != WHEN_IDLE, steer_only=self.mode == STEER_ONLY,
        )


@dataclass(frozen=True)
class Principal:
    """Who is sending: ``bot`` and ``user`` arrive over HTTP, ``system`` in-process."""

    kind: Literal["bot", "user", "system"]
    entity_id: str

    @classmethod
    def of_author(cls, author: AuthorReference) -> Principal:
        return cls("user" if author.entity_type == "user" else "bot", author.entity_id or "")


def refuse_server_owned_metadata(metadata: dict[str, Any]) -> None:
    for key in SERVER_OWNED_METADATA_KEYS:
        if key in metadata:
            raise DeliveryRefused(status_code=422, detail=f"{key} is server-owned delivery metadata")


def _privileged_fields(body: DeliveryCreate) -> list[str]:
    return [name for name in PRIVILEGED_FIELDS if name in body.model_fields_set]


def _refuse_bot_privileged(request: Any, fields: list[str]) -> None:
    from .task_turn_context import TASK_TURN_CONTEXT_HEADER

    # Presence, not validity: a bot turn's request is refused even if its
    # capability has expired or is forged.
    if fields and request is not None and request.headers.get(TASK_TURN_CONTEXT_HEADER):
        raise DeliveryRefused(status_code=403, detail={
            "error": "privileged_field", "fields": fields,
            "detail": "These fields are for service callers, not bot tools.",
        })


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


def _gate_disposition(request: Any, body: DeliveryCreate, sender: str, target: str) -> str:
    """``hold`` only for a trusted self-model continuation; every send is ``deny``.

    Trust is the signed turn capability (minted server-side, forwarded only by
    the self_model tool), never a body field: bots_send_message cannot set it.
    A continuation is the signed bot's own follow-up turn (sender == target).
    """
    from .service.system_gate_admission import ON_GATE_DENY, ON_GATE_HOLD
    from .task_turn_context import (
        TASK_TURN_CONTEXT_HEADER, TaskTurnContextError, open_task_turn_context,
    )

    capability = request.headers.get(TASK_TURN_CONTEXT_HEADER) if request is not None else None
    if not capability or (body.message_kind or "").upper() != "MODEL_SWITCH_CONTINUE":
        return ON_GATE_DENY
    try:
        caller = open_task_turn_context(capability)
    except TaskTurnContextError:
        return ON_GATE_DENY
    own = caller.bot_id.strip().lower() == sender == target
    return ON_GATE_HOLD if own else ON_GATE_DENY


def _steer_only_receipt(
    current: Any, status_code: int, result: dict[str, Any],
) -> tuple[int, dict[str, Any]]:
    """A steer_only receipt: settling as not needed is its success, not a failure."""
    result = {**result, "delivery": STEER_ONLY}
    if current.status == CANCELLED and str(current.last_error or "").startswith(NOT_NEEDED_PREFIX):
        result.update(success=True, note=(
            "The target had no executing turn to steer, so nothing was delivered "
            f"({current.last_error}). steer_only never starts a turn."))
        return 202, result
    if result.get("success"):
        result["note"] = (
            "Delivery is durable and will steer the target's executing Claude turn; "
            "if none is running it settles as not needed and never starts a turn.")
    return status_code, result


class DeliverySender:
    """Admits durable sends for every producer."""

    def __init__(self, service: Callable[[], Any], *, dispatcher: Callable[[], Any]):
        # Both are looked up on use, so request validation never needs a live service.
        self._service = service
        self._dispatcher = dispatcher

    @property
    def service(self) -> Any:
        return self._service()

    async def admit(self, body: DeliveryCreate, request: Any = None) -> tuple[int, dict[str, Any]]:
        """The single-target route: one call to ``_admit_one``."""
        refuse_server_owned_metadata(body.metadata)
        return await self._admit_one(body, request)

    async def submit(self, body: BatchDeliveryCreate, request: Any = None) -> tuple[int, dict[str, Any]]:
        """A fan-out batch (design r3 §1.2, §1.3)."""
        from .inter_bot_fanout_batch import FanoutBatch

        refuse_server_owned_metadata(body.metadata)
        return await FanoutBatch(self, body, request).run()

    def _require_owned_session(self, session_id: str, bot_id: str, user_id: str) -> None:
        """Never deliver into another bot's or user's thread, or a deleted one.

        Archived threads are allowed: task dispatch mints them born-archived.
        """
        from sqlalchemy import text as sa_text

        engine = self._dispatcher().store.engine
        with engine.connect() as conn:
            owned = conn.execute(sa_text(
                "SELECT 1 FROM sessions WHERE id=:id AND bot_id=:bot AND user_id=:user"
                " AND status <> 'deleted'"
            ), {"id": session_id, "bot": bot_id, "user": user_id}).first()
        if owned is None:
            raise DeliveryRefused(status_code=404, detail={
                "error": "session_not_found",
                "detail": f"No session {session_id!r} for bot {bot_id!r} and this user",
            })

    async def _admit_one(self, body: DeliveryCreate, request: Any = None) -> tuple[int, dict[str, Any]]:
        """Persist one callback and dispatch it through the target's next idle turn.

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
            raise DeliveryRefused(status_code=422, detail=str(exc)) from exc
        author = explicit_author or AuthorReference.bot(sender)
        if body.reply_mode == "waited" and (body.prefer_steer or author.entity_type != "bot" or sender == "unknown"):
            raise DeliveryRefused(status_code=422, detail="Waited replies require a bot-authored when-idle delivery")
        if not target:
            raise DeliveryRefused(status_code=422, detail="target_bot_id is required")
        if not body.message.strip():
            raise DeliveryRefused(status_code=422, detail="message is required")
        bots = BotManager(self.service.config)
        if author.entity_type == "bot" and not _is_real_bot(bots, author.entity_id):
            # A bot-authored row renders as that bot; "unknown" is not one (TASK-1241).
            raise DeliveryRefused(
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
            raise DeliveryRefused(status_code=404, detail={
                "error": "unknown_target", "detail": f"Unknown target bot {target!r}",
            })
        if service_dispatch and not durable_capable:
            # Service dispatch needs a durable agent turn; say so instead of
            # leaving a FAILED row the caller would have to discover.
            raise DeliveryRefused(status_code=422, detail={
                "error": "target_not_durable",
                "detail": f"Target bot {target!r} does not support durable agent delivery",
            })
        # A reset admits through NewSessionLifecycle, which mints its own identity
        # and rotates the active thread, so neither pinned field survives it.
        reset_requested = body.reset_requested
        for pinned in ("session_id", "user_message_id"):
            if getattr(body, pinned) is not None and reset_requested:
                raise DeliveryRefused(status_code=422, detail=f"{pinned} cannot be combined with a session reset")
        # Steering joins whatever turn is running, in the active thread, whose
        # capability signs that turn's trigger, not the caller's pinned UUID.
        for pinned in ("session_id", "user_message_id"):
            if getattr(body, pinned) is not None and body.prefer_steer:
                raise DeliveryRefused(status_code=422, detail=f"{pinned} requires prefer_steer=false")
        try:
            delivery_user = resolve_delivery_user(
                self.service.config,
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
            # a caller's pinned trigger UUID. A steer_only send never starts a
            # turn, so it never takes the reset turn either.
            if (target_bot and policy.value == "continue" and not policy_was_explicit
                    and body.session_id is None and body.user_message_id is None
                    and not body.steer_only):
                from .runtime_settings import RuntimeSettingsResolver
                from .setting_definitions import setting_default

                resolver = RuntimeSettingsResolver(
                    self.service.config, bot=target_bot
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
                ceiling = _target_context_window(self.service.config, target_bot)
                health = AgentContextStore(self.service.config).health(
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
            raise DeliveryRefused(status_code=422, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise DeliveryRefused(status_code=503, detail=str(exc)) from exc

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
            raise DeliveryRefused(status_code=422, detail="Compaction maintenance requires an idle Claude /compact command")
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
        if body.steer_only:
            payload[STEER_ONLY_KEY] = True
        if body.session_id is not None:
            self._require_owned_session(body.session_id, target, delivery_user)
            payload["session_id"] = body.session_id
        dispatcher = self._dispatcher()
        try:
            if policy.value != "continue":
                from .service.session_lifecycle.intent import Admission, Continuity, NewSessionIntent, Origin
                from .service.session_lifecycle.lifecycle import NewSessionLifecycle

                intent = NewSessionIntent(
                    bot_id=target, user_id=delivery_user, author=author,
                    continuity=(Continuity.RETAIN if policy.value == "reset_retain_history"
                                else Continuity.CLEAN),
                    admission=Admission.WAIT, origin=Origin.BOT_DELIVERY,
                    idempotency_key=body.idempotency_key or f"bot-reset:{uuid4()}",
                    follow_up=body.message, reason=body.reset_reason or automatic_reason,
                )
                record, duplicate = NewSessionLifecycle(self.service).admit_bot_delivery(
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
            raise DeliveryRefused(status_code=exc.status_code, detail=str(exc)) from exc
        except ValueError as exc:
            raise DeliveryRefused(status_code=409, detail=str(exc)) from exc
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
                STEER_OR_IDLE if stored_payload.get("prefer_steer", True) else WHEN_IDLE
            ),
        )
        if stored_payload.get(STEER_ONLY_KEY):
            status_code, result = _steer_only_receipt(current, status_code, result)
        return status_code, result
