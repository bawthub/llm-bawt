"""Reply-channel notes added to the outbound copy of bot-authored deliveries.

The recipient of a bot message needs to know whether its final response
reaches the sender. A waited delivery returns that response to the sender's
tool call; an async delivery does not, so an answer must be sent back with
``bots_send_message``. Each note is a prompt-registry key resolved per
delivery (target-bot override, then global, then the seed below), so it is
visible and editable in BawtHub's Prompt Registry.

The ``DEFAULT_*`` constants are seed data only. This module has no side
effects, so the registry can import it.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

REPLY_CHANNEL_WAITED_KEY = "inter_bot.reply_channel.waited"
REPLY_CHANNEL_ASYNC_KEY = "inter_bot.reply_channel.async"
REPLY_CHANNEL_WAITED_NO_RETURN_KEY = "inter_bot.reply_channel.waited_no_return"

DEFAULT_REPLY_CHANNEL_WAITED = (
    "[Reply channel: awaited delivery. Your final response on this turn "
    "is the answer to the sending bot's tool call (or its durable receipt "
    "if the wait times out). Give the answer in your final response. "
    "Do not send a separate message back merely to repeat that answer; "
    "an intentional, distinct follow-up is still allowed by setting "
    "separate_message_reason on bots_send_message.]"
)

DEFAULT_REPLY_CHANNEL_ASYNC = (
    "[Reply channel: async delivery. The sending bot is not waiting, and your "
    "final response on this turn is NOT returned to it; it stays in your own "
    "chat. If this message asks for a reply, verdict, decision, or result, send "
    "it back yourself with bots_send_message(target_bot_id=\"{sender_bot_id}\") "
    "before ending your turn (include task_id when the work is task-scoped). If "
    "no response is needed, do not send one.]"
)

# TASK-1362: an inactive sender (the outside-agent bot) cannot receive a bot
# message, so its waited note drops the separate follow-up, and an async
# delivery from it gets no note at all.
DEFAULT_REPLY_CHANNEL_WAITED_NO_RETURN = (
    "[Reply channel: awaited delivery. Your final response on this turn "
    "is the answer to the sending agent's tool call (or its durable receipt "
    "if the wait times out). Give the answer in your final response. The "
    "sender cannot receive bot messages, so do not send one back.]"
)

_KEY_BY_MODE = {
    "waited": (REPLY_CHANNEL_WAITED_KEY, DEFAULT_REPLY_CHANNEL_WAITED),
    "async": (REPLY_CHANNEL_ASYNC_KEY, DEFAULT_REPLY_CHANNEL_ASYNC),
    "waited_no_return": (REPLY_CHANNEL_WAITED_NO_RETURN_KEY, DEFAULT_REPLY_CHANNEL_WAITED_NO_RETURN),
}
# Delivery reply modes; ``waited_no_return`` is chosen from the sender's tags.
_DELIVERY_MODES = frozenset({"waited", "async"})


def _sender_is_inactive(sender_bot_id: str) -> bool:
    """True only when a readable tag snapshot says the sender is inactive.

    An unreadable snapshot keeps today's note: a spare reply instruction is
    harmless, a missing one loses an answer.
    """
    from .bot_tags import BehaviorTag, BotTags

    snapshot = BotTags().snapshot()
    return snapshot.available and snapshot.has(sender_bot_id, BehaviorTag.INACTIVE)


def resolve_reply_channel_note(reply_mode: str, *, sender_bot_id: str, target_bot_id: str | None) -> str:
    """Return the live reply-channel note for ``reply_mode``.

    A blank body or a registry error falls back to the seed, so a store hiccup
    never silently drops the reply contract the recipient depends on.
    """
    key, seed = _KEY_BY_MODE[reply_mode]
    body = seed
    try:
        from llm_bawt.prompt_registry import get_prompt_resolver

        resolved = get_prompt_resolver().resolve(
            key, scope_type="bot" if target_bot_id else "global", scope_id=target_bot_id,
        )
        if resolved and (resolved.body or "").strip():
            body = resolved.body
    except Exception as error:  # registry/DB failure must not break delivery
        logger.warning("reply-channel prompt resolve failed for %s, using seed: %s", key, error)
    # Substitute by replacement, never .format(): an edited body may contain
    # literal braces (JSON examples) that must survive untouched.
    return body.replace("{sender_bot_id}", sender_bot_id)


def delivery_reply_channel_note(record: Any) -> str | None:
    """The note for one durable delivery, or None when none applies.

    The note is instruction text for the recipient, not part of the message.
    Callers append it to the OUTBOUND copy only (agent ``context_suffix`` or the
    steer wire text); the stored payload, history row and timeline stay clean.
    Only bot-authored conversational deliveries get one: a user or reserved
    (``system``) author has no reply channel, and context maintenance is a bare harness command
    (``/compact``) that any trailing text would break. An inactive sender
    cannot be messaged back, so its async delivery gets no note either.
    """
    if record is None or record.message_kind == "CONTEXT_MAINTENANCE":
        return None
    from .bot_identities import RESERVED_JOB_BOT_IDS
    from .inter_bot_delivery_retry import is_system_retry, retry_reply_to

    author = record.author
    if is_system_retry(record):
        # A system-authored retry (TASK-1231B) answers to the original bot
        # sender; a retry of a user's request has no bot to reply to.
        sender = retry_reply_to(record)
    elif (author.entity_type == "bot" and author.entity_id != "unknown"
          and author.entity_id not in RESERVED_JOB_BOT_IDS):
        # Reserved identities (system notices such as ops-blocked) cannot
        # receive a reply.
        sender = author.entity_id
    else:
        sender = None
    if sender is None:
        return None
    mode = record.reply_mode if record.reply_mode in _DELIVERY_MODES else "async"
    if _sender_is_inactive(sender):
        if mode == "async":
            return None
        mode = "waited_no_return"
    return resolve_reply_channel_note(
        mode, sender_bot_id=sender, target_bot_id=record.target_bot_id,
    )


def request_reply_channel_note(service: Any, request: Any) -> str | None:
    """Resolve the note for a claim-validated delivery turn request.

    ``inter_bot_delivery_id`` is server-generated and validated against the
    dispatcher claim before a turn is admitted, so the record it names is the
    trusted source of sender and reply mode; nothing caller-supplied is read.
    """
    delivery_id = getattr(request, "inter_bot_delivery_id", None)
    if not delivery_id:
        return None
    store = getattr(getattr(service, "_inter_bot_dispatcher", None), "store", None)
    if store is None:
        return None
    try:
        return delivery_reply_channel_note(store.get(delivery_id))
    except Exception as error:  # the note is advisory; it must never fail the turn
        logger.warning("reply-channel note for delivery %s unavailable: %s", delivery_id, error)
        return None


def append_request_reply_channel_note(service: Any, request: Any, suffix: str | None) -> str | None:
    """Join the delivery's note onto an outbound-only agent ``context_suffix``."""
    note = request_reply_channel_note(service, request)
    return "\n\n".join(part for part in (suffix, note) if part) or None
