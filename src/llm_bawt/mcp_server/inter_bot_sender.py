"""Who is sending an inter-bot message (TASK-1241).

The sender is an identity claim, so it comes from server-signed context
whenever one exists, never from a model-supplied argument. ``sender_bot_id``
is only a fallback for callers that have no signed turn (external MCP clients,
backends that do not forward the capability). Those callers must name
themselves explicitly, and the delivery route still has to resolve that name
to a real bot.

TASK-1362: an agent outside the family (no signed turn, no bot of its own to
name) is represented by a bot tagged ``external`` AND ``inactive``. When the
caller opts in (``bots_send_message`` only) and there is no capability header
at all, exactly one such bot becomes the default sender. A header that is
present but does not open never reaches that default: the caller had a turn
and lost it, so guessing an identity for it would be wrong.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..task_turn_context import TaskTurnContextError

# Placeholders that mean "nobody". They never count as an explicit sender.
_UNATTRIBUTED = frozenset({"", "unknown"})

_REQUIRED_MESSAGE = (
    "No signed caller context is available, so sender_bot_id is "
    "required and must be your bot id; the message was not sent"
)


class SenderIdentityError(ValueError):
    """The send has no trustworthy sender, or it claims someone else's."""


@dataclass(frozen=True)
class _SignedCaller:
    """What server-owned context says about the caller."""

    bot_id: str | None
    # A capability header arrived but did not open (invalid, expired, malformed).
    capability_rejected: bool = False


def _normalize(value: str | None) -> str:
    return str(value or "").strip().lower()


def _signed_caller() -> _SignedCaller:
    """The calling bot from server-owned context, if any.

    An approved call replays later in the app process with no turn-scoped
    capability, so its original caller travels in the approval context.
    """
    from .approval_interceptor import current_approved_caller_context
    from .task_association import current_task_turn_capability, current_task_turn_context

    approved = current_approved_caller_context()
    if approved is not None and _normalize(approved.bot_id):
        return _SignedCaller(_normalize(approved.bot_id))
    try:
        return _SignedCaller(_normalize(current_task_turn_context().bot_id) or None)
    except TaskTurnContextError:
        presented = bool(str(current_task_turn_capability() or "").strip())
        return _SignedCaller(None, capability_rejected=presented)


def _external_default_from(tags) -> str:
    """The one bot tagged external AND inactive in ``tags``, or SenderIdentityError.

    Chosen by tag from one validated snapshot, never by slug.
    """
    from ..bot_tags import BehaviorTag

    candidates = sorted(
        bot for bot in tags.known_bots
        if tags.has(bot, BehaviorTag.EXTERNAL) and tags.has(bot, BehaviorTag.INACTIVE)
    )
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise SenderIdentityError(
            "No signed caller context is available and no outside-agent bot is "
            "set up, so the message was not sent. Pass sender_bot_id with your "
            "own bot id, or create a bot for outside agents and tag it both "
            "'external' and 'inactive'."
        )
    raise SenderIdentityError(
        "No signed caller context is available and several outside-agent bots "
        f"are tagged external and inactive ({', '.join(candidates)}); pass "
        "sender_bot_id with the one you are. The message was not sent."
    )


def _resolve(claimed: str | None) -> tuple[str | None, bool]:
    """``(sender, may_default)``: the sender, or None with whether the
    outside-agent default may stand in. Raises on a spoofed sender."""
    explicit = _normalize(claimed)
    caller = _signed_caller()
    if caller.bot_id:
        if explicit not in _UNATTRIBUTED and explicit != caller.bot_id:
            raise SenderIdentityError(
                f"sender_bot_id {explicit!r} does not match the signed caller "
                f"{caller.bot_id!r}; omit sender_bot_id and it is filled in for you"
            )
        return caller.bot_id, False
    if explicit not in _UNATTRIBUTED:
        return explicit, False
    # A header that arrived but did not open means the caller had a turn and
    # lost it; never guess an identity for it.
    return None, not caller.capability_rejected


def resolve_sender_bot_id(claimed: str | None) -> str:
    """Return the sender to stamp on a delivery, or raise SenderIdentityError."""
    sender, _ = _resolve(claimed)
    if sender is None:
        raise SenderIdentityError(_REQUIRED_MESSAGE)
    return sender


async def resolve_message_sender(claimed: str | None, policy: Callable[[], Any]) -> str:
    """``resolve_sender_bot_id`` plus the outside-agent default (TASK-1362).

    Only ``bots_send_message`` uses this. ``policy`` returns a
    ``BotPolicyResolver``; it is called only when the default is needed. An
    unreadable policy fails closed with the ordinary "name yourself" error.
    """
    from llm_bawt.bot_access import BotAccessError

    sender, may_default = _resolve(claimed)
    if sender is not None:
        return sender
    if not may_default:
        raise SenderIdentityError(_REQUIRED_MESSAGE)
    try:
        access = await policy().resolve()
    except BotAccessError:
        raise SenderIdentityError(_REQUIRED_MESSAGE) from None
    return _external_default_from(access.tags)
