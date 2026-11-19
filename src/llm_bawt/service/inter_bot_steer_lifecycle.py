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


def _apply_inbox(dispatcher, payload, target_turn_id) -> str:
    from ..inter_bot_inbox_claim import InboxRunBinding
    from ..inter_bot_inbox_settlement import InboxSettlement, InboxWriteProof

    try:
        envelope = payload["inbox_write"]
        raw = envelope["evidence"]
        evidence = InboxWriteProof(InboxRunBinding(**raw["binding"]), raw["generation"],
                                   raw["command_uuid"], raw["write_token"])
        if (type(evidence.generation) is not int or evidence.binding.turn_id != target_turn_id
                or evidence.command_uuid != payload.get("message_id")
                or evidence.binding.request_id != payload.get("run_request_id")
                or SteerGeneration.parse_request_id(payload.get("steer_rpc_id")) != (
                    envelope["delivery_id"], evidence.generation,
                )):
            return IGNORED
    except (KeyError, TypeError, ValueError):
        return IGNORED
    record = InboxSettlement(dispatcher.store).note_lifecycle(
        envelope["delivery_id"], evidence, payload.get("state"), proof=payload.get("proof"),
    )
    if record is None:
        return IGNORED
    if record.status == "QUEUED":
        dispatcher.announce_admitted(record)
        return REOPENED
    dispatcher.announce_lifecycle(record)
    return STARTED if payload.get("state") in LIFECYCLE_STARTED else str(payload.get("state"))


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
    if "inbox_write" in payload:
        return _apply_inbox(dispatcher, payload, target_turn_id)
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
        started = fence.note_steer_started(delivery_id, generation, target_turn_id)
        if not started:
            return IGNORED
        if started is not True:
            # TASK-2480: STARTED settled the accepted steer DELIVERED.
            dispatcher.announce_lifecycle(started)
        return STARTED
    if state in LIFECYCLE_REJECTED and proof in NOT_RUN_PROOFS:
        outcome, record = fence.reopen_rejected_steer(delivery_id, generation, target_turn_id, str(proof))
        logger.info(
            "delivery %s steer generation %s %s (%s): %s",
            delivery_id, generation, state, proof, outcome,
        )
        status = getattr(record, "status", None)
        if record is not None and outcome == REOPENED and status in (None, "QUEUED"):
            # QUEUED again: publish that lifecycle, then wake the dispatcher.
            dispatcher.announce_admitted(record)
        elif record is not None and status in ("FAILED", "CANCELLED"):
            # TASK-2480: attempts exhausted, or an evidence conflict.
            dispatcher.announce_lifecycle(record)
        return outcome
    if state == LIFECYCLE_UNCERTAIN:
        # TASK-2480: an accepted steer becomes a noted resend or UNKNOWN
        # (delivery/recovery); one still settling its claim keeps the marker.
        marked, child = fence.settle_uncertain_steer(
            delivery_id, generation, target_turn_id, str(proof or LIFECYCLE_UNCERTAIN),
        )
        if marked is None:
            return IGNORED
        dispatcher.announce_lifecycle(marked)
        if child is not None:
            dispatcher.announce_admitted(child)
        return UNCERTAIN
    return IGNORED


__all__ = ["STARTED", "UNCERTAIN", "apply_bot_steer_lifecycle"]
