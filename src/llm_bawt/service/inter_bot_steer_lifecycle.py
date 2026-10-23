"""Settle a bot delivery from its steer's bridge lifecycle (TASK-1300).

The claude-code bridge publishes ``steer_lifecycle`` for every held steer
command (``agent_bridge.steer_contract``). For a ``bot``-origin command the
event names its delivery through ``steer_rpc_id``
(``steer_delivery_<hex>[_g<k>]``, :class:`SteerGeneration`) and the run it was
steered into. This module applies the event to that delivery row through the
row-locked :class:`SteerLifecycleFence`:

* ``started`` / ``completed``: generation ``k`` ran; it can never be reopened.
* a rejection with a not-run proof: the row re-queues at ``k+1`` and the
  dispatcher is woken to deliver it (``steer_rejected``: it waits for the
  target turn to end, then takes the idle fallback).
* ``uncertain``: the row is fenced where it is and never re-sent; its
  delivery event is re-published with ``steer_uncertain`` so the sender's
  receipt says the message may not have run.

Anything else (``queued``, a rejection without proof, which only follows a
start, or a user/system steer) changes nothing.
"""
from __future__ import annotations

import logging
from typing import Any, Mapping

from agent_bridge.steer_contract import (
    LIFECYCLE_REJECTED,
    LIFECYCLE_STARTED,
    LIFECYCLE_UNCERTAIN,
    NOT_RUN_PROOFS,
)

from ..inter_bot_delivery_steer_fence import IGNORED, REOPENED, SteerLifecycleFence
from .inter_bot_dispatcher_steer import SteerGeneration

logger = logging.getLogger(__name__)

STARTED = "started"
UNCERTAIN = "uncertain"


def apply_bot_steer_lifecycle(
    dispatcher: Any, payload: Mapping[str, Any], *, target_turn_id: str | None,
) -> str:
    """Apply one lifecycle event to its delivery; returns what happened.

    Blocking (row-locked PostgreSQL writes): call it off the event loop.
    ``target_turn_id`` is the turn the run belongs to; the fence only touches
    a row steered into that turn at the event's own generation.
    """
    if payload.get("origin") != "bot":
        return IGNORED
    parsed = SteerGeneration.parse_request_id(payload.get("steer_rpc_id"))
    if parsed is None or not target_turn_id:
        logger.warning(
            "bot steer lifecycle without a delivery identity: rpc=%r turn=%r state=%r",
            payload.get("steer_rpc_id"), target_turn_id, payload.get("state"),
        )
        return IGNORED
    delivery_id, generation = parsed
    state, proof = payload.get("state"), payload.get("proof")
    fence = SteerLifecycleFence(dispatcher.store)
    if state in LIFECYCLE_STARTED:
        return STARTED if fence.note_steer_started(delivery_id, generation, target_turn_id) else IGNORED
    if state in LIFECYCLE_REJECTED and proof in NOT_RUN_PROOFS:
        outcome, record = fence.reopen_rejected_steer(delivery_id, generation, target_turn_id, str(proof))
        logger.info(
            "delivery %s steer generation %s %s (%s): %s",
            delivery_id, generation, state, proof, outcome,
        )
        if outcome == REOPENED and record is not None:
            # QUEUED again: publish that lifecycle, then wake the dispatcher.
            dispatcher.announce_admitted(record)
        return outcome
    if state == LIFECYCLE_UNCERTAIN:
        marked = fence.mark_steer_uncertain(
            delivery_id, generation, target_turn_id, str(proof or LIFECYCLE_UNCERTAIN),
        )
        if marked is None:
            return IGNORED
        # The row stays DELIVERED; this event is what tells the sender and the
        # relay receipt that the message may not have run.
        dispatcher.announce_lifecycle(marked)
        return UNCERTAIN
    return IGNORED


__all__ = ["STARTED", "UNCERTAIN", "apply_bot_steer_lifecycle"]
