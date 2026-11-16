"""Positive SDK acceptance evidence for a bridge-owned durable inbox claim.

The caller claims outside the run lock. ``prepare_write`` must durably fence
the claim as WRITING before returning True; a lost/uncertain preparation never
authorizes a write. ``owns_run`` checks SessionQueue's exact active-client
identity and Stop state, again under the input lock after that awaited RPC.

This boundary does not acknowledge a delivery or release a claim. In
particular, a successful transport write is not an acceptance receipt and a
cancelled caller leaves durable recovery to the claim's owner.
"""
from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from agent_bridge.maintenance import is_compact_command
from agent_bridge.steer_contract import (
    LIFECYCLE_REJECTED, LIFECYCLE_STARTED, PROOF_CLI_TERMINAL, PROOF_SWEEP_RECEIPT,
)

if TYPE_CHECKING:
    from .active_run import ClaudeActiveRun


@dataclass(frozen=True)
class InboxInputReceipt:
    """Evidence for one operation on one run, never an implicit durable ack."""

    outcome: Literal["accepted", "rejected", "uncertain", "not_written"]
    request_id: str
    command_uuid: str
    state: str | None = None
    proof: str | None = None
    detail: str | None = None
    acceptance: Literal["queued", "started"] | None = None


def _receipt(run: ClaudeActiveRun, command_uuid: str, *, detail: str | None = None) -> InboxInputReceipt:
    outcome: Literal["accepted", "rejected", "uncertain"]
    proof: str | None
    state = run._settled.get(command_uuid) or run._commands.get(command_uuid)
    acceptance: Literal["queued", "started"] | None = None
    if command_uuid in run._started_ever or command_uuid in run._replayed:
        outcome, proof = "accepted", None
        acceptance = "started"
    elif state in LIFECYCLE_REJECTED:
        outcome = "rejected"
        proof = PROOF_SWEEP_RECEIPT if command_uuid in run._settled else PROOF_CLI_TERMINAL
    elif state == "queued" or state in LIFECYCLE_STARTED:
        outcome, proof = "accepted", None
        acceptance = "queued" if state == "queued" else "started"
    else:
        outcome, proof = "uncertain", None
    return InboxInputReceipt(
        outcome, run.request_id, command_uuid, state=state, proof=proof, detail=detail,
        acceptance=acceptance,
    )


async def _wait_receipt(run: ClaudeActiveRun, command_uuid: str) -> InboxInputReceipt:
    if run._commands.get(command_uuid) == "sent" and command_uuid not in run._settled:
        try:
            await asyncio.wait_for(
                run._command_events[command_uuid].wait(), timeout=run._steer_timeout_s,
            )
        except asyncio.TimeoutError:
            return _receipt(run, command_uuid, detail="cli_ack_timeout")
    return _receipt(run, command_uuid)


async def accept_inbox_input(
    run: ClaudeActiveRun, *,
    command_uuid: str,
    steer_rpc_id: str,
    framed_message: str,
    interrupt_item: str,
    owns_run: Callable[[], bool],
    prepare_write: Callable[[], Awaitable[bool]],
) -> InboxInputReceipt:
    """Attempt one noninterrupting input, with no automatic replay on uncertainty.

    Identity is stable across claim-token changes and unique per generation.
    Re-delivery of a known operation only reads its receipt, including a
    rejection: only durable admission may advance the generation. In-memory
    absence is NOT recovery evidence after restart; the caller must pass the
    durable WRITING fence before any new write is possible.
    """
    if str(uuid.UUID(command_uuid)) != command_uuid or not steer_rpc_id.strip():
        raise ValueError("Inbox input requires canonical UUID and operation identity")
    if not framed_message.strip() or not interrupt_item.strip():
        raise ValueError("Inbox input requires app-rendered framing")

    def not_written(detail: str) -> InboxInputReceipt:
        return InboxInputReceipt("not_written", run.request_id, command_uuid, detail=detail)

    def identity_matches() -> bool:
        held = run._held.get(command_uuid)
        return bool(held and held.origin == "bot" and held.steer_rpc_id == steer_rpc_id
                    and held.item == interrupt_item)

    override: object | None = None

    def finish(receipt: InboxInputReceipt) -> InboxInputReceipt:
        if override is not None and receipt.outcome in {"rejected", "not_written"}:
            run._rollback_steer_over_background(override)
        return receipt

    # A duplicate must never turn an old uncertainty/rejection into a fresh write.
    if command_uuid in run._commands:
        if not identity_matches():
            return InboxInputReceipt(
                "uncertain", run.request_id, command_uuid, detail="command_identity_conflict",
            )
        return await _wait_receipt(run, command_uuid)
    if run.maintenance or is_compact_command(framed_message):
        return not_written("maintenance_active")
    if not run.lifecycle_supported:
        return not_written("next_unsupported")
    if not owns_run():
        return not_written("run_owner_changed")
    try:
        run._refuse_new_command()
    except RuntimeError as exc:
        return not_written(str(exc))

    # No run input lock is held across the durable preparation RPC.
    if await prepare_write() is not True:
        return not_written("claim_write_not_authorized")
    async with run._steer_lock:
        # Another invocation may have completed while preparation awaited.
        if command_uuid in run._commands:
            if not identity_matches():
                return InboxInputReceipt(
                    "uncertain", run.request_id, command_uuid, detail="command_identity_conflict",
                )
        else:
            if not owns_run():
                return not_written("run_owner_changed")
            try:
                run._refuse_new_command()
            except RuntimeError as exc:
                return not_written(str(exc))
            override = run._mark_steer_over_background()
            try:
                await run._write_next_locked(
                    framed_message, interrupt_item, command_uuid,
                    origin="bot", steer_rpc_id=steer_rpc_id,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # Registration precedes any possible SDK write. Once registered,
                # an exception cannot prove that stdin was untouched.
                if command_uuid not in run._commands:
                    return finish(not_written(str(exc)))
                return finish(_receipt(run, command_uuid, detail=f"sdk_write_failed:{type(exc).__name__}"))
    # A slow/missing acknowledgment must not block interrupt-now or cancel.
    return finish(await _wait_receipt(run, command_uuid))
