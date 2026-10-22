"""Who is sending an inter-bot message (TASK-1241).

The sender is an identity claim, so it comes from server-signed context
whenever one exists, never from a model-supplied argument. ``sender_bot_id``
is only a fallback for callers that have no signed turn (external MCP clients,
backends that do not forward the capability). Those callers must name
themselves explicitly, and the delivery route still has to resolve that name
to a real bot.
"""

from __future__ import annotations

from ..task_turn_context import TaskTurnContextError

# Placeholders that mean "nobody". They never count as an explicit sender.
_UNATTRIBUTED = frozenset({"", "unknown"})


class SenderIdentityError(ValueError):
    """The send has no trustworthy sender, or it claims someone else's."""


def _normalize(value: str | None) -> str:
    return str(value or "").strip().lower()


def _signed_caller_bot_id() -> str | None:
    """The calling bot from server-owned context, or None when there is none.

    An approved call replays later in the app process with no turn-scoped
    capability, so its original caller travels in the approval context.
    """
    from .approval_interceptor import current_approved_caller_context
    from .task_association import current_task_turn_context

    approved = current_approved_caller_context()
    if approved is not None and _normalize(approved.bot_id):
        return _normalize(approved.bot_id)
    try:
        return _normalize(current_task_turn_context().bot_id) or None
    except TaskTurnContextError:
        return None


def resolve_sender_bot_id(claimed: str | None) -> str:
    """Return the sender to stamp on a delivery, or raise SenderIdentityError."""
    explicit = _normalize(claimed)
    signed = _signed_caller_bot_id()
    if signed:
        if explicit not in _UNATTRIBUTED and explicit != signed:
            raise SenderIdentityError(
                f"sender_bot_id {explicit!r} does not match the signed caller "
                f"{signed!r}; omit sender_bot_id and it is filled in for you"
            )
        return signed
    if explicit in _UNATTRIBUTED:
        raise SenderIdentityError(
            "No signed caller context is available, so sender_bot_id is "
            "required and must be your bot id; the message was not sent"
        )
    return explicit
