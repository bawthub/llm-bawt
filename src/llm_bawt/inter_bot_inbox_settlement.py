"""Original-write lifecycle proof, persistence-only takeover and durable inbox ack.

These store operations are not a scheduler or input transport. Facts from the
original SDK operation remain valid across a repair-owner change; only the
current claim token can acknowledge successful history persistence.
"""
from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime

from sqlalchemy import text

from agent_bridge.steer_contract import LIFECYCLE_REJECTED, LIFECYCLE_STARTED, NOT_RUN_PROOFS

from .inter_bot_batch import last_text
from .inter_bot_delivery_ledger import DeliveryTransitionEvents
from .inter_bot_delivery_record import DeliveryRecord
from .inter_bot_delivery_steer_fence import (
    STEER_STARTED, STEER_UNCERTAIN, SteerLifecycleFence, _generation,
)
from .inter_bot_inbox_claim import INBOX, InboxRunBinding, inbox_marker
from .utils.bot_partition import sanitize_table_name


@dataclass(frozen=True)
class InboxWriteProof:
    binding: InboxRunBinding
    generation: int
    command_uuid: str
    write_token: str

    def matches(self, row) -> bool:
        marker = inbox_marker(row)
        return bool(
            marker and marker.get("phase") in {"writing", "uncertain", "accepted", "acknowledged"}
            and row["status"] in {"STEERING", "DELIVERED"} and row["delivery_mode"] == "steer"
            and row["target_bot_id"] == self.binding.bot_id and row["target_turn_id"] == self.binding.turn_id
            and marker.get("binding") == asdict(self.binding)
            and marker.get("generation") == self.generation == _generation(row["metadata_json"])
            and marker.get("command_uuid") == self.command_uuid
            and bool(self.write_token) and marker.get("write_token") == self.write_token
        )


@dataclass(frozen=True)
class InboxHistoryRepair:
    record: DeliveryRecord
    evidence: InboxWriteProof
    message: str
    content_offset: int | None


class InboxSettlement:
    def __init__(self, store):
        self.store = store

    def repair_targets(self, *, backend: str, after: str = "", limit: int = 100) -> list[str]:
        """Paged actor discovery, including accepted rows with live leases.

        This grants no claim or SDK permission. Any same-backend bridge may
        repair history; the existing per-target claim fence chooses one owner.
        """
        if self.store.engine is None:
            return []
        with self.store.engine.connect() as conn:
            return list(conn.execute(text("""
                SELECT DISTINCT target_bot_id FROM inter_bot_deliveries
                WHERE status='STEERING' AND delivery_mode='steer'
                  AND metadata_json->'_inbox'->>'phase'='accepted'
                  AND metadata_json->'_inbox'->'binding'->>'backend'=:backend
                  AND transport_accepted_at IS NOT NULL AND target_bot_id > :after
                ORDER BY target_bot_id LIMIT :limit
            """), {"backend": backend, "after": after, "limit": max(1, min(limit, 500))}).scalars())

    def release_not_written(
        self, delivery_id: str, claim_token: str, evidence: InboxWriteProof,
        invocation_id: str, *, stop_requested: bool = False,
    ) -> DeliveryRecord | None:
        """Only the positively authorized invocation proves untouched input.

        Neither preparation denial/loss nor missing process-local command state
        is proof. The input boundary supplies this receipt only after its True
        grant and a terminal return before any possible SDK write.
        """
        if self.store.engine is None or not claim_token or not invocation_id:
            return None
        with self.store.engine.begin() as conn:
            row = SteerLifecycleFence(self.store)._locked_row(conn, delivery_id)
            marker = inbox_marker(row) if row else None
            if (not marker or marker.get("phase") != "writing"
                    or row["claim_token"] != claim_token or not evidence.matches(row)
                    or marker.get("write_invocation") != invocation_id
                    or marker.get("acceptance") or row["transport_accepted_at"] is not None):
                return None
            durable_stop = self.store._explicit_stop_requested(conn, evidence.binding.turn_id)
            if stop_requested and not durable_stop:
                return None
            if durable_stop:
                from .runtime_setting_resolution import resolve_global_runtime_setting
                from .service.turn_gates import live_gate_for_owner
                from .service.user_stop_hold import HOLD_SETTING, hold_owner

                # Stop audit alone suffices only when queue holds are disabled.
                # A resumed/expired hold must not strand an offline receipt.
                ttl = int(resolve_global_runtime_setting(self.store.config, HOLD_SETTING) or 0)
                if ttl > 0 and live_gate_for_owner(
                    conn, hold_owner(evidence.binding.bot_id), fresh=True,
                ) is None:
                    raw = conn.execute(text(
                        "SELECT request_json FROM turn_logs WHERE id=:id"
                    ), {"id": evidence.binding.turn_id}).scalar_one()
                    try:
                        stamp = datetime.fromisoformat(json.loads(raw)["abort_request"]["requested_at"])
                    except (KeyError, TypeError, ValueError):
                        return None
                    if stamp.tzinfo is None or not conn.execute(text("""
                        SELECT EXISTS (
                            SELECT 1 FROM system_gates WHERE owner_id=:owner
                              AND created_at <= :stamp AND expires_at > :stamp
                              AND (released_at IS NULL OR released_at >= :stamp)
                        )
                    """), {"owner": hold_owner(evidence.binding.bot_id), "stamp": stamp}).scalar_one():
                        return None
            return SteerLifecycleFence.reopen_claimed(conn, row, "inbox_not_written")

    def note_lifecycle(
        self, delivery_id: str, evidence: InboxWriteProof, state: str, *,
        proof: str | None = None, content_offset: int | None = None,
    ) -> DeliveryRecord | None:
        """Record evidence from the original write, not transport success.

        Only a late start resolves uncertainty; an old queued receipt does not.
        Rejection must prove non-execution and cannot overrule a recorded start.
        """
        if content_offset is not None and (type(content_offset) is not int or content_offset < 0):
            raise ValueError("content_offset must be a nonnegative integer")
        if self.store.engine is None:
            return None
        with self.store.engine.begin() as conn:
            row = SteerLifecycleFence(self.store)._locked_row(conn, delivery_id)
            if row is None or not evidence.matches(row):
                return None
            marker = inbox_marker(row)
            started = marker.get("acceptance") == "started"
            if state in LIFECYCLE_REJECTED:
                if proof not in NOT_RUN_PROOFS or started:
                    return None
                reopen = (SteerLifecycleFence.reopen_delivered if row["status"] == "DELIVERED"
                          else SteerLifecycleFence.reopen_claimed)
                return reopen(conn, row, proof)
            merge = {}
            if state == "uncertain":
                if started:
                    return None
                marker = {**marker, "phase": "uncertain"}
                merge[STEER_UNCERTAIN] = {"generation": evidence.generation, "reason": proof or state}
            elif state == "queued" or state in LIFECYCLE_STARTED:
                if state == "queued" and marker["phase"] == "uncertain":
                    return None
                started = started or state in LIFECYCLE_STARTED
                marker = {**marker,
                          "phase": "acknowledged" if row["status"] == "DELIVERED" else "accepted",
                          "acceptance": "started" if started else "queued",
                          "content_offset": marker.get("content_offset", content_offset)}
                if started:
                    merge[STEER_STARTED] = {"generation": evidence.generation}
                    merge[STEER_UNCERTAIN] = None
            else:
                return None
            merge[INBOX] = marker
            updated = conn.execute(text("""
                UPDATE inter_bot_deliveries SET
                    metadata_json=metadata_json || CAST(:merge AS jsonb),
                    transport_accepted_at=CASE WHEN :accepted
                        THEN COALESCE(transport_accepted_at, now()) ELSE transport_accepted_at END,
                    updated_at=now() WHERE id=:id RETURNING *
            """), {"id": delivery_id, "merge": json.dumps(merge), "accepted": state != "uncertain"}).mappings().one()
            DeliveryTransitionEvents.stage(conn, updated, reason_code="inbox_lifecycle")
            return DeliveryRecord.from_mapping(updated)

    def claim_repair(self, target_bot_id: str, *, claim_owner: str, lease_seconds: int = 300) -> DeliveryRecord | None:
        """Take only an expired accepted operation, never a new input permission."""
        if self.store.engine is None:
            return None
        from .service.turn_admission import lock_target

        target = target_bot_id.strip().lower()
        with self.store.engine.begin() as conn:
            lock_target(conn, target)
            row = conn.execute(text("""
                SELECT * FROM inter_bot_deliveries WHERE target_bot_id=:target
                  AND status='STEERING' AND delivery_mode='steer'
                  AND metadata_json->'_inbox'->>'phase'='accepted'
                  AND transport_accepted_at IS NOT NULL AND lease_expires_at < now()
                ORDER BY ordinal LIMIT 1 FOR UPDATE
            """), {"target": target}).mappings().first()
            if row is None:
                return None
            updated = conn.execute(text("""
                UPDATE inter_bot_deliveries SET claim_token=:token, claim_owner=:owner,
                    lease_expires_at=now()+(:lease * interval '1 second'), updated_at=now()
                WHERE id=:id RETURNING *
            """), {"id": row["id"], "token": f"claim-{uuid.uuid4().hex}", "owner": claim_owner,
                   "lease": max(60, lease_seconds)}).mappings().one()
            return DeliveryRecord.from_mapping(updated)

    @staticmethod
    def _destination_owned(conn, evidence: InboxWriteProof) -> bool:
        """Check the original durable destination, including after run completion."""
        from .service.steer_session import owned_turn_destination

        binding = evidence.binding
        turn = conn.execute(text("""
            SELECT bot_id, user_id, agent_backend, agent_session_key, agent_request_id
            FROM turn_logs WHERE id=:id
        """), {"id": binding.turn_id}).mappings().first()
        return bool(turn and (
            turn["bot_id"], turn["user_id"], turn["agent_backend"], turn["agent_session_key"], turn["agent_request_id"]
        ) == (binding.bot_id, binding.user_id, binding.backend, binding.session_key, binding.request_id)
            and owned_turn_destination(
                conn, turn_id=binding.turn_id, bot_id=binding.bot_id, user_id=binding.user_id,
            ) == (binding.trigger_message_id, binding.session_id))

    def history_repair(self, delivery_id: str, claim_token: str) -> InboxHistoryRepair | None:
        """Pinned persistence context, fenced before asynchronous history work."""
        if self.store.engine is None or not claim_token:
            return None
        with self.store.engine.begin() as conn:
            row = SteerLifecycleFence(self.store)._locked_row(conn, delivery_id)
            marker = inbox_marker(row) if row else None
            if (not marker or marker.get("phase") != "accepted" or row["status"] != "STEERING"
                    or row["claim_token"] != claim_token or row["transport_accepted_at"] is None):
                return None
            evidence = InboxWriteProof(InboxRunBinding(**marker["binding"]), marker["generation"],
                                       marker["command_uuid"], marker["write_token"])
            message = last_text(row["payload_json"])
            if message is None or not evidence.matches(row) or not self._destination_owned(conn, evidence):
                return None
            return InboxHistoryRepair(DeliveryRecord.from_mapping(row), evidence, message, marker.get("content_offset"))

    def acknowledge(self, delivery_id: str, claim_token: str, evidence: InboxWriteProof) -> DeliveryRecord | None:
        """Ack only the current claim's positively accepted, correctly persisted input."""
        if self.store.engine is None or not claim_token:
            return None
        with self.store.engine.begin() as conn:
            row = SteerLifecycleFence(self.store)._locked_row(conn, delivery_id)
            marker = inbox_marker(row) if row else None
            if (marker and row["status"] == "DELIVERED" and marker.get("phase") == "acknowledged"
                    and marker.get("ack_token") == claim_token and evidence.matches(row)):
                return DeliveryRecord.from_mapping(row)
            if (row is None or row["status"] != "STEERING" or row["claim_token"] != claim_token
                    or not evidence.matches(row) or inbox_marker(row).get("phase") != "accepted"
                    or row["transport_accepted_at"] is None or not self._destination_owned(conn, evidence)):
                return None
            marker, binding = inbox_marker(row), evidence.binding
            message = conn.execute(text("""
                SELECT role, content, session_id, author_entity_type, author_entity_id,
                       interrupt_source_message_id, interrupt_content_offset
                FROM messages WHERE bot_id=:bot AND id=:id
            """), {"bot": sanitize_table_name(binding.bot_id), "id": row["user_message_id"]}).mappings().first()
            if (not message or message["role"] != "user" or message["session_id"] != binding.session_id
                    or message["content"] != last_text(row["payload_json"])
                    or message["author_entity_type"] != row["author_entity_type"]
                    or message["author_entity_id"] != row["author_entity_id"]):
                return None
            offset = marker.get("content_offset")
            if offset is not None and (
                message["interrupt_source_message_id"] != binding.trigger_message_id
                or message["interrupt_content_offset"] != offset
            ):
                return None
            updated = conn.execute(text("""
                UPDATE inter_bot_deliveries SET status='DELIVERED', delivered_at=now(),
                    claim_token=NULL, claim_owner=NULL, lease_expires_at=NULL,
                    next_retry_at=NULL, last_error=NULL,
                    metadata_json=metadata_json || CAST(:merge AS jsonb), updated_at=now()
                WHERE id=:id RETURNING *
            """), {"id": delivery_id, "merge": json.dumps({
                INBOX: {**marker, "phase": "acknowledged", "ack_token": claim_token},
            })}).mappings().one()
            DeliveryTransitionEvents.stage(conn, updated, reason_code="inbox_acknowledged")
            return DeliveryRecord.from_mapping(updated)

    def persist_history(
        self, delivery_id: str, claim_token: str, evidence: InboxWriteProof,
        write: Callable[[], bool],
    ) -> bool:
        """Serialize all history mutation with rejection, takeover and generation.

        This synchronous boundary must run entirely in the writer's worker
        thread. Releasing a lock in a cancelled coroutine while add_message or
        the anchor thread still runs would let it overwrite a successor.
        """
        if self.store.engine is None or not claim_token:
            return False
        with self.store.engine.begin() as conn:
            row = SteerLifecycleFence(self.store)._locked_row(conn, delivery_id)
            if (row is None or row["status"] != "STEERING" or row["claim_token"] != claim_token
                    or not evidence.matches(row) or inbox_marker(row).get("phase") != "accepted"
                    or row["transport_accepted_at"] is None or not self._destination_owned(conn, evidence)):
                return False
            return write()
