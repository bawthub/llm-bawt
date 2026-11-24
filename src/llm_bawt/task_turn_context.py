"""Trusted current-turn capability for agent task association.

The app mints one short-lived Fernet token from identifiers it already owns.
Bridges forward the opaque token as an HTTP MCP header; the central MCP server
verifies and opens it. Raw session/turn/message identifiers never become model
arguments, and no process-local context is assumed across Redis/SDK/MCP hops.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from uuid import UUID

from cryptography.fernet import InvalidToken

from .service.providers.crypto import _get_fernet

TASK_TURN_CONTEXT_HEADER = "X-LLM-Bawt-Task-Turn-Context"
# TASK-1249: the same signed capability, forwarded by bots_send_message only to
# record which turn sent a delivery. A separate header so it never grants what
# TASK_TURN_CONTEXT_HEADER does on the delivery route (self-model gate hold).
SENDER_TURN_CONTEXT_HEADER = "X-LLM-Bawt-Sender-Turn-Context"
TASK_TURN_CONTEXT_TTL_SECONDS = 6 * 60 * 60
TASK_TURN_CONTEXT_BACKENDS = frozenset({"claude-code", "codex"})
# Three canonical shapes are minted server-side and must round-trip through the
# capability: ordinary chat turns (``turn-<32 hex>``), synthetic approval
# continuations (``turn-approval-cont-<32 hex>``), and synthetic inter-bot
# delivery turns (``turn-delivery-<32 hex>``, produced by
# ``inter_bot_delivery.stable_ids`` for durable dispatch/callbacks). Rejecting
# the delivery shape silently killed ``tasks_update(associate_current_turn=true)``
# on every delivery-dispatched claude-code turn (TASK-783) because the mint
# raised, the exception was swallowed to a warning, no capability header was
# sent, and the MCP tool then failed closed with "No trusted current-turn
# context is available for this MCP request". The delivery shape is stored in
# the signed capability but NEVER written as an ``AgentTaskTurn.turnId`` on the
# BawtHub side — attribution for delivery turns rides on ``triggerMessageId``
# alone per the TASK-706 design (see ``task_association.associate_current_task``
# for the PUT-body filter and ``scripts/commit-recon`` for the resolve-batch
# filter).
_TURN_ID_RE = re.compile(r"^turn-[0-9a-f]{32}$")
_DELIVERY_TURN_ID_RE = re.compile(r"^turn-delivery-[0-9a-f]{32}$")
_APPROVAL_CONTINUATION_TURN_ID_RE = re.compile(
    r"^turn-approval-cont-[0-9a-f]{32}$"
)
_ACTOR_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9@._:-]{0,63}$")
_SENTINEL_IDENTITIES = {"unknown", "none", "null", "undefined"}


def is_delivery_turn_id(turn_id: str | None) -> bool:
    """True if ``turn_id`` is a synthetic inter-bot-delivery id."""
    return bool(_DELIVERY_TURN_ID_RE.fullmatch(str(turn_id or "")))


def is_approval_continuation_turn_id(turn_id: str | None) -> bool:
    """True if ``turn_id`` is a server-owned approval continuation id."""
    return bool(
        _APPROVAL_CONTINUATION_TURN_ID_RE.fullmatch(str(turn_id or ""))
    )


def is_synthetic_turn_id(turn_id: str | None) -> bool:
    """True for trusted turn identities not valid as canonical task turn IDs."""
    return is_delivery_turn_id(turn_id) or is_approval_continuation_turn_id(turn_id)


class TaskTurnContextError(ValueError):
    """The current-turn capability is absent, malformed, invalid, or expired."""


# (auth account ID, canonical origin, original web admission time)
InspectionClaim = tuple[str, str, int]


def is_inspection_turn_id(turn_id: str) -> bool:
    """Eligible ID shapes; the signed inspection claim carries the authority."""
    return bool(_TURN_ID_RE.fullmatch(turn_id) or _DELIVERY_TURN_ID_RE.fullmatch(turn_id))


@dataclass(frozen=True)
class TaskTurnContext:
    session_id: str
    turn_id: str
    trigger_message_id: str
    bot_id: str
    user_id: str
    issued_at: int
    # Present only on ordinary turns admitted by the signed-in web frontend.
    inspection: InspectionClaim | None = None
    # TASK-2599: proves only which bot/user is calling. The session id is a
    # fresh placeholder, so consumers that trust it as a conversation (task
    # association, session-ownership shortcuts, initiative/continuation scope)
    # must refuse an identity-only context.
    identity_only: bool = False


def require_conversation_authority(context: TaskTurnContext) -> TaskTurnContext:
    """The context, or an error when it proves identity but owns no conversation."""
    if getattr(context, "identity_only", False):
        raise TaskTurnContextError(
            "This turn's identity owns no conversation; association and "
            "session-scoped actions are unavailable"
        )
    return context


def backend_supports_task_turn_context(backend: str | None) -> bool:
    """Whether a first-party bridge propagates the trusted turn capability."""
    return str(backend or "").strip().lower() in TASK_TURN_CONTEXT_BACKENDS


def _canonical_uuid(value: str, field: str) -> str:
    try:
        parsed = UUID(str(value))
    except (ValueError, TypeError, AttributeError) as error:
        raise TaskTurnContextError(f"{field} must be a canonical UUID") from error
    canonical = str(parsed)
    if str(value) != canonical:
        raise TaskTurnContextError(f"{field} must be a canonical lowercase UUID")
    return canonical


def _actor_id(value: str, field: str) -> str:
    normalized = str(value or "").strip()
    if (
        not _ACTOR_ID_RE.fullmatch(normalized)
        or normalized.lower() in _SENTINEL_IDENTITIES
    ):
        raise TaskTurnContextError(f"{field} has an invalid format")
    return normalized


def _validate_context(context: TaskTurnContext) -> TaskTurnContext:
    turn_id = str(context.turn_id or "").strip()
    if not (
        _TURN_ID_RE.fullmatch(turn_id)
        or _DELIVERY_TURN_ID_RE.fullmatch(turn_id)
        or _APPROVAL_CONTINUATION_TURN_ID_RE.fullmatch(turn_id)
    ):
        raise TaskTurnContextError(
            "turn_id must match a canonical ordinary, delivery, or approval "
            "continuation turn identifier"
        )
    inspection = context.inspection
    if inspection is not None:
        if (
            not is_inspection_turn_id(turn_id)
            or not isinstance(inspection, (tuple, list)) or len(inspection) != 3
            or not all(isinstance(part, str) and 0 < len(part) <= 512 for part in inspection[:2])
            or type(inspection[2]) is not int or not 0 < inspection[2] <= context.issued_at
        ):
            raise TaskTurnContextError("Invalid inspection provenance")
        inspection = (inspection[0], inspection[1], inspection[2])
    if type(context.identity_only) is not bool:
        raise TaskTurnContextError("identity_only must be a boolean")
    return TaskTurnContext(
        session_id=_canonical_uuid(context.session_id, "session_id"),
        turn_id=turn_id,
        trigger_message_id=_canonical_uuid(
            context.trigger_message_id,
            "trigger_message_id",
        ),
        bot_id=_actor_id(context.bot_id, "bot_id"),
        user_id=_actor_id(context.user_id, "user_id"),
        issued_at=int(context.issued_at),
        inspection=inspection,
        identity_only=context.identity_only,
    )


def mint_task_turn_context(
    *,
    session_id: str,
    turn_id: str,
    trigger_message_id: str,
    bot_id: str,
    user_id: str,
    issued_at: int | None = None,
    inspection: InspectionClaim | None = None,
    identity_only: bool = False,
) -> str:
    """Mint an opaque capability for exactly one server-owned chat turn."""
    context = _validate_context(TaskTurnContext(
        session_id=session_id,
        turn_id=turn_id,
        trigger_message_id=trigger_message_id,
        bot_id=bot_id,
        user_id=user_id,
        issued_at=int(time.time()) if issued_at is None else int(issued_at),
        inspection=inspection,
        identity_only=identity_only,
    ))
    fields = asdict(context)
    if inspection is None:
        fields.pop("inspection")  # keep the existing token shape when dormant
    if not identity_only:
        fields.pop("identity_only")
    payload = json.dumps(fields, separators=(",", ":"), sort_keys=True)
    return _get_fernet().encrypt(payload.encode("utf-8")).decode("ascii")


def mint_for_agent_turn(
    *,
    backend: str | None,
    session_id: str | None,
    turn_id: str,
    trigger_message_id: str,
    bot_id: str,
    user_id: str,
    inspection: InspectionClaim | None = None,
) -> str | None:
    """Best-effort capability for one agent turn, shared by stream/non-stream.

    Association is optional metadata: a signing/config failure must never block
    the chat turn. The MCP tool later fails closed with a clear no-context error.
    """
    if not (backend_supports_task_turn_context(backend) and session_id):
        return None
    try:
        return mint_task_turn_context(
            session_id=session_id,
            turn_id=turn_id,
            trigger_message_id=trigger_message_id,
            bot_id=bot_id,
            user_id=user_id,
            inspection=inspection,
        )
    except Exception as error:  # noqa: BLE001 - see docstring
        import logging

        logging.getLogger(__name__).warning(
            "Could not mint trusted task-turn context for %s: %s", turn_id, error,
        )
        return None


def open_task_turn_context(
    token: str | None,
    *,
    ttl_seconds: int = TASK_TURN_CONTEXT_TTL_SECONDS,
) -> TaskTurnContext:
    """Verify and decode a current-turn capability, failing closed."""
    normalized = str(token or "").strip()
    if not normalized:
        raise TaskTurnContextError(
            "No trusted current-turn context is available for this MCP request"
        )
    try:
        raw = _get_fernet().decrypt(normalized.encode("ascii"), ttl=ttl_seconds)
    except (InvalidToken, UnicodeEncodeError) as error:
        raise TaskTurnContextError("Current-turn context is invalid or expired") from error
    try:
        body = json.loads(raw)
        context = TaskTurnContext(**body)
    except (json.JSONDecodeError, TypeError, UnicodeDecodeError) as error:
        raise TaskTurnContextError("Current-turn context payload is malformed") from error
    return _validate_context(context)
