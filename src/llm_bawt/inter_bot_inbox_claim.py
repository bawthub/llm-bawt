"""Running-only inbox admission and one-shot permission to attempt an SDK write.

No consumer advertises this capability yet. PostgreSQL holds the permission:
a lost preparation response leaves WRITING, never permission to try again.
Acceptance, persistence and acknowledgment are separate operations.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Mapping

from sqlalchemy import text

from .inter_bot_batch import plain
from .steer_capabilities import INBOX_NEXT, NEXT, run_steer_caps

INBOX = "_inbox"


@dataclass(frozen=True)
class InboxRunBinding:
    bot_id: str
    user_id: str
    turn_id: str
    backend: str
    session_key: str
    request_id: str
    trigger_message_id: str
    session_id: str

    @classmethod
    def read(cls, conn, *, target: str, active) -> InboxRunBinding | None:
        """Resolve only an executing, bound, capable run's owned conversation."""
        if (not active or active.get("status") not in {"pending", "streaming"}
                or not active.get("agent_backend")
                or not active.get("agent_session_key") or not active.get("agent_request_id")
                or not active.get("user_id")
                or not {NEXT, INBOX_NEXT} <= run_steer_caps(
                    backend=active["agent_backend"], caps=active.get("agent_steer_caps"),
                )):
            return None
        from .service.steer_session import owned_turn_destination

        destination = owned_turn_destination(
            conn, turn_id=active["id"], bot_id=target, user_id=active["user_id"],
        )
        if destination is None:
            return None
        trigger, session = destination
        return cls(target, active["user_id"], active["id"], active["agent_backend"],
                   active["agent_session_key"], active["agent_request_id"], trigger, session)


def inbox_marker(row: Mapping[str, Any]) -> dict[str, Any] | None:
    marker = (row.get("metadata_json") or {}).get(INBOX)
    return marker if isinstance(marker, dict) else None


def eligible(row, *, user_id: str, config) -> bool:
    """Only new normal async text; never steal a previous app-side steer."""
    payload = row.get("payload_json") or {}
    return bool(
        (row.get("lane") or "normal") == "normal"
        and row.get("delivery_mode") is None
        and row.get("transport_accepted_at") is None
        and inbox_marker(row) is None
        and payload.get("prefer_steer", True)
        and plain(row, user_id=user_id, config=config)
    )


def claimed_metadata(binding: InboxRunBinding, row) -> dict[str, Any]:
    generation = max(0, int((row.get("metadata_json") or {}).get("steer_generation") or 0))
    return {
        "phase": "claimed",
        "generation": generation,
        "command_uuid": str(uuid.uuid5(uuid.NAMESPACE_URL, f"inbox:{row['id']}:{generation}")),
        "binding": asdict(binding),
    }


def prepare_inbox_write(
    store, delivery_id: str, claim_token: str, binding: InboxRunBinding,
    *, content_offset: int | None = None, freeze_offset: bool = False,
    invocation_id: str | None = None,
) -> bool:
    """True exactly once per claim, before any possible input; network awaits are outside.

    A retry after a lost response sees WRITING and returns False. The original
    write token and run binding stay durable for later lifecycle proof.
    The internal API supplies a turn-scoped offset sampled before this fence;
    freeze_offset retains even an explicitly unknown value, never a retry's
    later position. Direct store callers may retain first-receipt anchoring.
    """
    if content_offset is not None and (type(content_offset) is not int or content_offset < 0):
        raise ValueError("content_offset must be a nonnegative integer")
    if store.engine is None or not claim_token:
        return False
    from .service.turn_admission import classify_target, lock_target
    from .service.turn_gates import GateActor, gate_blocks

    with store.engine.begin() as conn:
        lock_target(conn, binding.bot_id)
        row = conn.execute(text("""
            SELECT *, lease_expires_at > now() AS lease_live
            FROM inter_bot_deliveries WHERE id=:id FOR UPDATE
        """), {"id": delivery_id}).mappings().first()
        marker = inbox_marker(row) if row else None
        if (not marker or marker.get("phase") != "claimed"
                or marker.get("binding") != asdict(binding)
                or row["target_bot_id"] != binding.bot_id
                or row["target_turn_id"] != binding.turn_id
                or row["status"] != "STEERING" or row["delivery_mode"] != "steer"
                or row["claim_token"] != claim_token or not row["lease_live"]):
            return False
        if store._explicit_stop_requested(conn, binding.turn_id):
            return False
        if store._explicit_stop_requested(conn, row["turn_id"]):
            return False
        if gate_blocks(conn, binding.bot_id, GateActor("delivery_claim", row["turn_id"])) is not None:
            return False
        from .service.approval_fanout import claim_fence
        from .maintenance_delivery import MaintenanceBarrier

        # This claim has not reached input yet: unlike an ambiguous legacy
        # steer, it still honors a maintenance barrier added after the claim.
        unaccepted = {**row, "delivery_mode": None}
        if claim_fence(conn, row) is not None or MaintenanceBarrier(conn, binding.bot_id).prepare(unaccepted) is not None:
            return False
        state = classify_target(conn, binding.bot_id, exclude_turn_id=row["turn_id"])
        if InboxRunBinding.read(conn, target=binding.bot_id, active=state.executing) != binding:
            return False
        marker = {**marker, "phase": "writing", "write_token": claim_token}
        if invocation_id is not None:
            marker["write_invocation"] = invocation_id
        if freeze_offset:
            marker["content_offset"] = content_offset
        conn.execute(text("""
            UPDATE inter_bot_deliveries SET
                metadata_json=metadata_json || CAST(:merge AS jsonb), updated_at=now()
            WHERE id=:id
        """), {"id": delivery_id, "merge": json.dumps({INBOX: marker})})
        return True
