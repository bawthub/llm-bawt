"""Internal bridge inbox operations; no app-origin SDK steering or rollout."""
from __future__ import annotations

import asyncio
from dataclasses import asdict
from typing import Literal

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, ConfigDict, Field

from ...inter_bot_batch import last_text
from ...inter_bot_inbox_claim import INBOX, InboxRunBinding, prepare_inbox_write
from ...inter_bot_inbox_settlement import InboxSettlement, InboxWriteProof
from .. import bridge_gate
from ..dependencies import get_service
from ..inbox_persistence import repair_inbox_history
from ..inter_bot_dispatcher_steer import SteerGeneration
from ..turn_admission import classify_target
from .chat import _persist_interrupt_anchor, _steer_framing
from .inter_bot_deliveries import _dispatcher


def _bridge(request: Request):
    bridge_gate.check(request, bridge_gate.BROKER)


router = APIRouter(prefix="/v1/internal/inbox", dependencies=[Depends(_bridge)])


class InboxRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RepairClaim(InboxRequest):
    bot_id: str = Field(min_length=1)
    claim_owner: str = Field(min_length=1, max_length=256)


class RunClaim(RepairClaim):
    user_id: str = Field(min_length=1)
    backend: str = Field(min_length=1)
    session_key: str = Field(min_length=1)
    request_id: str = Field(min_length=1)


class Operation(InboxRequest):
    delivery_id: str = Field(min_length=1)
    claim_token: str = Field(min_length=1)


class WriteOperation(Operation):
    binding: InboxRunBinding


class Lifecycle(InboxRequest):
    delivery_id: str = Field(min_length=1)
    evidence: InboxWriteProof
    state: Literal["queued", "started", "completed", "cancelled", "discarded", "refused", "uncertain"]
    proof: str | None = None
    content_offset: int | None = Field(default=None, ge=0, strict=True)


def _claimed(record):
    if record is None:
        return None
    marker = record.metadata[INBOX]
    return {
        "delivery_id": record.id, "claim_token": record.claim_token,
        "evidence": asdict(InboxWriteProof(
            InboxRunBinding(**marker["binding"]), marker["generation"],
            marker["command_uuid"], marker.get("write_token", record.claim_token),
        )),
    }


@router.post("/claim")
def claim(body: RunClaim):
    """The actual run owner asks; the durable authority revalidates that run."""
    store = _dispatcher().store
    target = body.bot_id.strip().lower()
    with store.engine.connect() as conn:
        binding = InboxRunBinding.read(
            conn, target=target, active=classify_target(conn, target, exclude_turn_id=None).executing,
        )
    if binding is None or (
        binding.user_id, binding.backend, binding.session_key, binding.request_id,
    ) != (body.user_id, body.backend, body.session_key, body.request_id):
        return {"claim": None}
    record = store.claim_next(
        target, claim_owner=body.claim_owner, steer_capable=True, inbox_run=binding,
    )
    result = _claimed(record)
    if result is not None:
        from ...inter_bot_reply_channel import delivery_reply_channel_note

        # These are the existing app-rendered trust/reply boundaries. The body
        # saved to history remains the original payload, without the wire note.
        message = last_text(store.payload(record.id))
        suffix = delivery_reply_channel_note(record)
        wire = f"{message}\n\n{suffix}" if suffix else message
        result.update(
            steer_rpc_id=SteerGeneration.of(record).request_id,
            **_steer_framing(wire, origin="bot", priority="next", bot_id=target, user_id=binding.user_id),
        )
    return {"claim": result}


@router.post("/prepare")
def prepare(body: WriteOperation):
    return {"authorized": prepare_inbox_write(
        _dispatcher().store, body.delivery_id, body.claim_token, body.binding,
    )}


@router.post("/repair/claim")
def claim_repair(body: RepairClaim):
    # No active-run lookup and no SDK framing: a finished run's accepted input
    # may still need history repair, never a second input permission.
    return {"claim": _claimed(InboxSettlement(_dispatcher().store).claim_repair(
        body.bot_id, claim_owner=body.claim_owner,
    ))}


@router.post("/repair")
async def repair(body: Operation):
    from ...agent_backends.agent_bridge import get_agent_subscriber

    dispatcher = _dispatcher()
    record = await repair_inbox_history(
        get_service(), dispatcher.store, delivery_id=body.delivery_id,
        claim_token=body.claim_token, subscriber=get_agent_subscriber(),
        persist_anchor=_persist_interrupt_anchor,
    )
    if record is None:
        # A lost successful repair response is retried with the same claim
        # token. Reuse only that acknowledged receipt, not fresh input permission.
        prior = await asyncio.to_thread(dispatcher.store.get, body.delivery_id)
        marker = prior.metadata.get(INBOX) if prior is not None else None
        if isinstance(marker, dict) and marker.get("phase") == "acknowledged":
            evidence = InboxWriteProof(
                InboxRunBinding(**marker["binding"]), marker["generation"],
                marker["command_uuid"], marker["write_token"],
            )
            record = await asyncio.to_thread(
                InboxSettlement(dispatcher.store).acknowledge, body.delivery_id, body.claim_token, evidence,
            )
    if record is not None:
        dispatcher.announce_lifecycle(record)
    return {"acknowledged": record is not None}


@router.post("/lifecycle")
async def lifecycle(body: Lifecycle):
    dispatcher = _dispatcher()
    record = await asyncio.to_thread(
        InboxSettlement(dispatcher.store).note_lifecycle, body.delivery_id, body.evidence,
        body.state, proof=body.proof, content_offset=body.content_offset,
    )
    if record is not None:
        if record.status == "QUEUED":
            dispatcher.announce_admitted(record)
        else:
            dispatcher.announce_lifecycle(record)
    return {"recorded": record is not None}
