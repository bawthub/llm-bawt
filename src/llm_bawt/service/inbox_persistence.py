"""Repair accepted inbox history and ack it, with no SDK input path."""
from __future__ import annotations

import asyncio

from ..inter_bot_inbox_settlement import InboxSettlement
from .steer_persistence import SteerDestination, persist_accepted_input


async def repair_inbox_history(service, store, *, delivery_id: str, claim_token: str,
                               subscriber, persist_anchor):
    settlement = InboxSettlement(store)
    repair = await asyncio.to_thread(settlement.history_repair, delivery_id, claim_token)
    if repair is None:
        return None
    record, binding = repair.record, repair.evidence.binding
    persisted = await persist_accepted_input(
        service,
        destination=SteerDestination(
            binding.bot_id, binding.user_id, binding.turn_id, binding.trigger_message_id, binding.session_id,
        ),
        message_id=record.user_message_id, message=repair.message, author=record.author,
        subscriber=subscriber, persist_anchor=persist_anchor,
        content_offset=repair.content_offset, resolve_offset=False,
    )
    if not persisted:
        return None
    # Async persistence may have outlived this lease, received a proven rejection
    # or raced takeover. Only the current token + original proof can now ack.
    return await asyncio.to_thread(settlement.acknowledge, delivery_id, claim_token, repair.evidence)
