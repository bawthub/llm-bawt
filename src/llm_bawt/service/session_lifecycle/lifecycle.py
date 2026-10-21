"""Durable new-session submission and observation on the ordinary delivery path.

A scheduled reset is a literal System-authored /new, not a synthetic session
rotation. Candidate selection is only a prefilter: both admission and claim
recheck the same measurement under the target lock. A delivery terminal state
is not a successful reset until its provider and receipt evidence is verified.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import asdict, replace
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import text

from ...bot_types import durable_agent_backend
from ...bots import BotManager
from ...inter_bot_delivery_record import DeliveryRecord
from ..jobs.idle_context_candidates import IdleContextCandidates
from ..turn_admission import classify_target, lock_target
from .intent import Admission, Continuity, NewSessionIntent, NewSessionResult, Origin, Outcome, Stage


class NewSessionLifecycle:
    """Submit once, then observe exactly that durable delivery until bounded time."""

    def __init__(self, service: Any):
        self.service = service
        dispatcher = getattr(service, "_inter_bot_dispatcher", None)
        if dispatcher is None or dispatcher.store.engine is None:
            raise RuntimeError("new-session delivery service is unavailable")
        self.dispatcher = dispatcher
        self.store = dispatcher.store
        self.candidates = IdleContextCandidates(self.store.engine)

    @staticmethod
    def rotate_delivery_claim(
        conn, *, head: dict, payload: dict, target_bot_id: str, user_id: str,
    ) -> tuple[dict, str | None, str | None]:
        """Execute a WAIT policy's one keyed transition under the claim lock."""
        from ...agent_context import SessionPolicy
        from ...memory.session_reset_lineage import SessionResetLineage

        policy = SessionPolicy(head.get("session_policy") or SessionPolicy.CONTINUE.value)
        if policy is SessionPolicy.CONTINUE:
            return payload, None, None
        continuity = {
            SessionPolicy.RESET_RETAIN_HISTORY: Continuity.RETAIN,
            SessionPolicy.RESET_WITHOUT_HISTORY: Continuity.CLEAN,
        }[policy]
        record = DeliveryRecord.from_mapping(head)
        intent = NewSessionIntent(
            bot_id=target_bot_id, user_id=user_id, author=record.author,
            continuity=continuity, admission=Admission.WAIT,
            origin=Origin.BOT_DELIVERY, idempotency_key=head["id"],
            follow_up=record.message,
        )
        if head.get("reset_status") in {"ROTATED", "COMPLETED"} and head.get("new_session_id"):
            payload["session_id"] = head["new_session_id"]
            payload["inter_bot_session_policy"] = policy.value
            payload["inter_bot_seed_session_id"] = head.get("old_session_id")
            return payload, head.get("old_session_id"), head["new_session_id"]

        old_session = conn.execute(text("""
            SELECT id FROM sessions WHERE bot_id=:bot AND user_id=:user
              AND status='active' AND ended_at IS NULL
            ORDER BY started_at DESC LIMIT 1 FOR UPDATE
        """), {"bot": intent.bot_id, "user": intent.user_id}).scalar_one_or_none()
        # Preserve persisted delivery IDs' historical successor namespace.
        new_session = str(uuid.uuid5(uuid.NAMESPACE_URL, f"inter-bot-session:{head['id']}"))
        SessionResetLineage.rotate(
            None, connection=conn, bot_id=intent.bot_id, user_id=intent.user_id,
            new_id=new_session, idempotency_key=intent.idempotency_key,
            accept_pointer=True, context_lifecycle={
                "action": policy.value,
                "reason": head.get("reset_reason") or "inter-bot pre-delivery policy",
                "delivery_id": head["id"], "old_session_id": old_session,
            },
        )
        conn.execute(text("""
            UPDATE inter_bot_deliveries SET reset_status='ROTATED',
                old_session_id=:old_session, new_session_id=:new_session,
                reset_at=now(), updated_at=now() WHERE id=:id
        """), {"id": head["id"], "old_session": old_session,
               "new_session": new_session})
        payload["session_id"] = new_session
        payload["inter_bot_session_policy"] = policy.value
        payload["inter_bot_seed_session_id"] = old_session
        conn.execute(text("""
            UPDATE inter_bot_deliveries SET payload_json=CAST(:payload AS jsonb),
                updated_at=now() WHERE id=:id
        """), {"id": head["id"],
                "payload": json.dumps(payload, ensure_ascii=False, default=str)})
        return payload, old_session, new_session

    @staticmethod
    def scheduled_skip_reason(
        conn, intent: NewSessionIntent, *, own_delivery_id: str | None = None,
    ) -> str | None:
        """Recheck pointer, ownership, context and idle state inside target lock.

        Never query a protected bot's sessions/turns. The target lock is held by
        both ordinary enqueue and claim, so no new turn can enter between these
        checks and reservation. No producer may turn SKIP into a delayed queue.
        """
        parameters = intent.scheduler_parameters
        candidates = IdleContextCandidates(
            None, protected_bot_ids=parameters.protected_bot_ids if parameters else (),
        )
        if intent.bot_id.strip().lower() in candidates.protected_bot_ids:
            return "protected bot"
        row = candidates.read_one(conn, intent.bot_id)
        if row is None:
            return "target bot not eligible"
        if row["session_id"] != intent.expected_session_id or row["user_id"] != intent.user_id:
            return "accepted session or owner changed"
        if row["usage_turn_id"] != intent.expected_usage_turn_id:
            return "context measurement changed"
        reason = candidates.skip_reason(row, now=datetime.now(timezone.utc), parameters=parameters)
        if reason:
            return reason
        try:
            usage = json.loads(row["token_usage_json"] or "null")
            native = int(usage.get("resident_tokens") or 0)
            if not native:
                # Input tokens are only a credible resident-context fallback if
                # the provider supplied a window for this exact observation.
                # Billing-wide input totals can be larger than a context window.
                window = int(usage.get("context_window") or 0)
                if window <= 0 or int(usage.get("input_tokens") or 0) > window:
                    return "resident context measurement is not bounded by provider window"
        except (AttributeError, TypeError, ValueError, OverflowError):
            return "resident context measurement is invalid"
        state = classify_target(conn, intent.bot_id, exclude_turn_id=None)
        if not state.idle:
            return "target has active or reserved work"
        queued = conn.execute(text("""
            SELECT 1 FROM inter_bot_deliveries
            WHERE target_bot_id=:bot AND status='QUEUED' AND id != :own LIMIT 1
        """), {"bot": intent.bot_id, "own": own_delivery_id or ""}).first()
        if queued:
            return "target has queued work"
        return None

    @staticmethod
    def _decode(raw: dict) -> NewSessionIntent:
        from ...message_authorship import AuthorReference

        author = raw.get("author") or {}
        return NewSessionIntent(**{
            **raw, "author": AuthorReference(
                author.get("entity_type"), author.get("entity_id"),
                author.get("status", "canonical"),
            ),
        })

    @staticmethod
    def bind_wait_predecessor(conn, head: dict, *, bot_id: str, user_id: str) -> None:
        """Fence a typed WAIT reset to the active thread at its actual claim.

        Enqueue-time observations are informational: earlier queued /new commands
        can rotate the thread before this delivery starts. Store the claim-time
        predecessor atomically with the turn reservation under the target lock.
        Scheduler SKIP observations are never rewritten.
        """
        raw = (head.get("metadata_json") or {}).get("new_session_intent")
        if not isinstance(raw, dict) or raw.get("origin") != Origin.CHAT_COMMAND:
            return
        from ...memory.session_reset_lineage import SessionResetLineage

        # On a safe, unaccepted retry the app may have already committed its
        # keyed rotation. That successor is the transition's durable identity;
        # never rebind its predecessor to the now-active successor itself.
        successor = conn.execute(text("""
            SELECT bot_id, user_id, session_metadata FROM sessions WHERE id=:id FOR UPDATE
        """), {"id": SessionResetLineage.delivery_successor_id(head["id"])}).mappings().one_or_none()
        if successor is not None:
            lineage = SessionResetLineage.metadata(successor["session_metadata"])
            claimed = (head.get("metadata_json") or {})
            predecessors = lineage.get("reset_predecessor_ids")
            if (successor["bot_id"] != bot_id or successor["user_id"] != user_id
                    or lineage.get("rotation_delivery_id") != head["id"]
                    or "new_session_claim_predecessor_id" not in claimed
                    or predecessors != ([claimed["new_session_claim_predecessor_id"]]
                                        if claimed["new_session_claim_predecessor_id"] else [])):
                raise ValueError("Typed /new retry has inconsistent rotation provenance")
            return
        predecessors = conn.execute(text("""
            SELECT id FROM sessions WHERE bot_id=:bot AND user_id=:user
              AND status='active' AND ended_at IS NULL FOR UPDATE
        """), {"bot": bot_id, "user": user_id}).scalars().all()
        if len(predecessors) > 1:
            raise ValueError("Typed /new has multiple active predecessors")
        conn.execute(text("""
            UPDATE inter_bot_deliveries SET metadata_json=jsonb_set(
                metadata_json, '{new_session_claim_predecessor_id}',
                CAST(:predecessor AS jsonb)), updated_at=now()
            WHERE id=:id AND status='QUEUED'
        """), {"id": head["id"], "predecessor": json.dumps(predecessors[0] if predecessors else None)})

    @classmethod
    def skip_stale_claim(cls, conn, head: dict) -> bool:
        """Terminate a stale scheduled command before any reservation or rotation."""
        raw = (head.get("metadata_json") or {}).get("new_session_intent")
        if not isinstance(raw, dict):
            return False
        intent = cls._decode(raw)
        if intent.origin not in {Origin.SCHEDULER, Origin.CONTEXT_API, Origin.SESSION_API}:
            return False
        # Once the bridge accepted this exact deterministic run, a retry must
        # reattach even if its own rotation/turn invalidated the original gauge.
        # Never turn accepted work into a new stale-work admission decision.
        if head.get("transport_accepted_at") is not None:
            return False
        if intent.origin is Origin.SCHEDULER:
            reason = cls.scheduled_skip_reason(
                conn, intent, own_delivery_id=head["id"],
            )
        else:
            from ...bot_session_state import BotSessionState

            pointer = BotSessionState.resolve(conn, bot_id=intent.bot_id)
            if (pointer is None or pointer["id"] != intent.expected_session_id
                    or pointer["user_id"] != intent.user_id
                    or pointer["status"] != "active" or pointer["ended_at"] is not None):
                reason = "accepted session or owner changed"
            elif not classify_target(conn, intent.bot_id,
                                     exclude_turn_id=head["turn_id"]).idle:
                reason = "target has active or reserved work"
            else:
                reason = None
        if reason is None:
            return False
        conn.execute(text("""
            UPDATE inter_bot_deliveries
            SET status='FAILED', last_error=:reason, next_retry_at=NULL,
                metadata_json=jsonb_set(metadata_json, '{new_session_skip_reason}',
                                        to_jsonb(CAST(:reason AS text))),
                updated_at=now()
            WHERE id=:id AND status='QUEUED'
        """), {"id": head["id"], "reason": reason})
        return True

    def _admit(self, intent: NewSessionIntent) -> tuple[DeliveryRecord | None, str | None]:
        if intent.origin is Origin.BOT_DELIVERY:
            record, _ = self.admit_bot_delivery(intent)
            return record, None
        if intent.origin in {Origin.CONTEXT_API, Origin.SESSION_API}:
            return self._admit_api_reset(intent)
        if intent.origin is not Origin.SCHEDULER:
            raise NotImplementedError("new-session origin has no admitted adapter")
        if intent.admission is not Admission.SKIP:
            raise ValueError("scheduled reset must skip rather than wait")
        if intent.bot_id.strip().lower() in self.candidates.protected_bot_ids:
            return None, "protected bot"
        bot = BotManager(self.service.config).get_bot(intent.bot_id)
        if bot is None or durable_agent_backend(bot) != "claude-code":
            return None, "target does not support scheduled fresh provider sessions"
        engine = self.store.engine
        with engine.begin() as conn:
            lock_target(conn, intent.bot_id)
            # A keyed retry attaches before re-evaluating an observation that
            # was intentionally invalidated by its own accepted /new.
            existing = conn.execute(text("""
                SELECT * FROM inter_bot_deliveries
                WHERE author_entity_type=:type AND author_entity_id=:author
                  AND sender_bot_id=:sender AND target_bot_id=:bot
                  AND idempotency_key=:key
            """), {"type": intent.author.entity_type, "author": intent.author.entity_id,
                    "sender": intent.author.entity_id, "bot": intent.bot_id,
                    "key": intent.idempotency_key}).mappings().first()
            if existing:
                record = DeliveryRecord.from_mapping(existing)
                stored = record.metadata.get("new_session_intent")
                if not isinstance(stored, dict):
                    raise ValueError("existing delivery lacks verifiable new-session intent")
                stored = dict(stored)
                submitted = self._encode(intent)
                # A sweep may have new settings, but the same measured context
                # must attach to the original operation, never replay a reset.
                # Claim-time checks keep the ORIGINAL persisted policy snapshot.
                for key in ("run_id", "scheduler_parameters"):
                    stored.pop(key, None)
                    submitted.pop(key, None)
                if stored != submitted:
                    raise ValueError("idempotency key belongs to another reset intent")
                return record, None
            reason = self.scheduled_skip_reason(conn, intent)
            if reason:
                return None, reason
            payload = {
                "bot_id": intent.bot_id, "user": intent.user_id,
                "messages": [{"role": "user", "content": "/new"}],
                "prefer_steer": False, "stream": False,
                "extract_memory": False, "augment_memory": True,
            }
            record, _ = self.store.enqueue(
                sender_bot_id=intent.author.entity_id,
                target_bot_id=intent.bot_id, author=intent.author,
                message="/new", payload=payload, connection=conn,
                idempotency_key=intent.idempotency_key,
                metadata={"new_session_intent": self._encode(intent)},
                producer="bot",
            )
            return record, None

    def admit_bot_delivery(
        self, intent: NewSessionIntent, *, payload: dict | None = None,
        metadata: dict | None = None, sender_bot_id: str | None = None,
        project_id: str | None = None, task_id: str | None = None,
        message_kind: str | None = None, max_attempts: int = 5,
        reset_reason: str | None = None, producer: str = "bot",
        connection=None,
    ) -> tuple[DeliveryRecord, bool]:
        """Public bot-send and internal callers share one policy admission.

        A caller-owned connection joins the route receipt and delivery intent
        without committing or closing the caller's transaction.
        """
        from ...agent_context import SessionPolicy, validate_session_policy

        if (intent.origin is not Origin.BOT_DELIVERY or intent.admission is not Admission.WAIT
                or intent.follow_up is None
                or intent.continuity not in {Continuity.RETAIN, Continuity.CLEAN}):
            raise ValueError("delivery reset requires a WAIT follow-up and explicit continuity")
        bot = BotManager(self.service.config).get_bot(intent.bot_id)
        if bot is None:
            raise ValueError("target bot is missing")
        policy = (SessionPolicy.RESET_RETAIN_HISTORY if intent.continuity is Continuity.RETAIN
                  else SessionPolicy.RESET_WITHOUT_HISTORY)
        validate_session_policy(durable_agent_backend(bot), policy)
        if payload is None:
            payload = {
                "bot_id": intent.bot_id, "user": intent.user_id,
                "messages": [{"role": "user", "content": intent.follow_up}],
                "prefer_steer": False, "stream": False,
            }
        else:
            if payload.get("bot_id") != intent.bot_id or payload.get("user") != intent.user_id:
                raise ValueError("reset delivery payload target differs from intent")
        record, duplicate = self.store.enqueue(
            sender_bot_id=sender_bot_id or intent.author.entity_id,
            target_bot_id=intent.bot_id, author=intent.author,
            message=intent.follow_up, idempotency_key=intent.idempotency_key,
            session_policy=policy,
            metadata={**(metadata or {}), "new_session_intent": self._encode(intent)},
            payload=payload, project_id=project_id, task_id=task_id,
            message_kind=message_kind, max_attempts=max_attempts,
            reset_reason=reset_reason or intent.reason, producer=producer,
            connection=connection,
        )
        if record.metadata.get("new_session_intent") != self._encode(intent):
            raise ValueError("idempotency key belongs to another reset intent")
        return record, duplicate

    def _admit_api_reset(self, intent: NewSessionIntent) -> tuple[DeliveryRecord | None, str | None]:
        """Admit a standalone reset only against this bot's idle accepted pointer."""
        from ...agent_context import SessionPolicy, validate_session_policy
        from ...bot_session_state import BotSessionState

        if (intent.admission is not Admission.SKIP or intent.follow_up is not None
                or intent.continuity not in {Continuity.RETAIN, Continuity.CLEAN}):
            raise ValueError("standalone API reset requires SKIP, explicit continuity, and no follow-up")
        bot = BotManager(self.service.config).get_bot(intent.bot_id)
        if bot is None:
            raise ValueError("target bot is missing")
        backend = durable_agent_backend(bot)
        policy = (SessionPolicy.RESET_RETAIN_HISTORY
                  if intent.continuity is Continuity.RETAIN
                  else SessionPolicy.RESET_WITHOUT_HISTORY)
        validate_session_policy(backend, policy)
        with self.store.engine.begin() as conn:
            lock_target(conn, intent.bot_id)
            existing = conn.execute(text("""
                SELECT * FROM inter_bot_deliveries
                WHERE author_entity_type=:type AND author_entity_id=:author
                  AND sender_bot_id=:sender AND target_bot_id=:bot
                  AND idempotency_key=:key
            """), {"type": intent.author.entity_type, "author": intent.author.entity_id,
                    "sender": intent.author.entity_id, "bot": intent.bot_id,
                    "key": intent.idempotency_key}).mappings().first()
            if existing:
                record = DeliveryRecord.from_mapping(existing)
                stamped = record.metadata.get("new_session_intent")
                if not isinstance(stamped, dict):
                    raise ValueError("existing delivery lacks verifiable reset intent")
                expected = stamped.get("expected_session_id")
                if (not isinstance(expected, str) or not expected
                        or stamped != self._encode(replace(intent, expected_session_id=expected))
                        or record.session_policy != policy.value
                        or record.message != "/new"
                        or not (existing["payload_json"] or {}).get("inter_bot_reset_only")):
                    raise ValueError("idempotency key belongs to another reset intent")
                return record, None
            pointer = BotSessionState.resolve(conn, bot_id=intent.bot_id)
            if (pointer is None or pointer["user_id"] != intent.user_id
                    or pointer["status"] != "active" or pointer["ended_at"] is not None):
                return None, "accepted session or owner changed"
            if intent.expected_session_id and intent.expected_session_id != pointer["id"]:
                return None, "accepted session changed"
            if not classify_target(conn, intent.bot_id, exclude_turn_id=None).idle:
                return None, "target has active or reserved work"
            queued = conn.execute(text("""
                SELECT 1 FROM inter_bot_deliveries WHERE target_bot_id=:bot
                  AND status IN ('QUEUED', 'DISPATCHING', 'STEERING') LIMIT 1
            """), {"bot": intent.bot_id}).first()
            if queued:
                return None, "target has queued work"
            stamped = replace(intent, expected_session_id=pointer["id"])
            record, _ = self.store.enqueue(
                sender_bot_id=intent.author.entity_id, target_bot_id=intent.bot_id,
                author=intent.author, message="/new", session_policy=policy,
                reset_reason=intent.reason,
                idempotency_key=intent.idempotency_key,
                metadata={"new_session_intent": self._encode(stamped)},
                payload={
                    "bot_id": intent.bot_id, "user": intent.user_id,
                    "messages": [{"role": "user", "content": "/new"}],
                    "inter_bot_reset_only": True,
                    "prefer_steer": False, "stream": False,
                    "extract_memory": False, "augment_memory": True,
                }, connection=conn,
            )
            return record, None

    @staticmethod
    def _encode(intent: NewSessionIntent) -> dict:
        value = asdict(intent)
        value["author"] = intent.author.to_dict()
        if intent.scheduler_parameters is not None:
            value["scheduler_parameters"] = intent.scheduler_parameters.model_dump(mode="json")
        else:
            # Preserve the byte shape of pre-parameter intents during recovery.
            value.pop("scheduler_parameters", None)
        return value

    def _observe(self, intent: NewSessionIntent, delivery_id: str) -> NewSessionResult | None:
        record = self.store.get(delivery_id)
        if record is None:
            return NewSessionResult(outcome=Outcome.AMBIGUOUS, stage=Stage.ADMISSION,
                                    delivery_id=delivery_id, detail="delivery disappeared")
        fields = {"delivery_id": record.id, "turn_id": record.turn_id,
                  "old_session_id": (record.old_session_id if intent.origin is Origin.BOT_DELIVERY
                                     else intent.expected_session_id)}
        if record.status == "FAILED" and record.metadata.get("new_session_skip_reason"):
            return NewSessionResult(outcome=Outcome.SKIPPED, stage=Stage.ADMISSION,
                                    detail=record.metadata["new_session_skip_reason"], **fields)
        if record.status == "FAILED" and record.metadata.get("new_session_ambiguous"):
            return NewSessionResult(outcome=Outcome.AMBIGUOUS, stage=Stage.PROVIDER_BIND,
                                    detail=record.last_error or "accepted reset outcome unknown", **fields)
        if record.status in {"FAILED", "CANCELLED"}:
            return NewSessionResult(outcome=Outcome.FAILED, stage=Stage.PROVIDER_BIND,
                                    detail=record.last_error or record.status, **fields)
        if record.status != "DELIVERED":
            return None
        if intent.origin in {Origin.BOT_DELIVERY, Origin.CONTEXT_API, Origin.SESSION_API}:
            from .policy_receipt import BotPolicyReceipt

            completed = BotPolicyReceipt.ensure(self.store.engine, record)
        else:
            from .receipt import NewSessionReceipt

            completed = NewSessionReceipt.ensure(self.store.engine, record, intent)
        if completed is not None:
            return completed
        return NewSessionResult(outcome=Outcome.AMBIGUOUS, stage=Stage.RECEIPT,
                                detail="delivery finished without provider binding and persisted receipt",
                                **fields)

    async def submit(
        self, intent: NewSessionIntent, *, timeout_seconds: float = 300,
    ) -> NewSessionResult:
        record, reason = await asyncio.to_thread(self._admit, intent)
        if reason:
            return NewSessionResult(outcome=Outcome.SKIPPED, stage=Stage.ADMISSION,
                                    detail=reason)
        assert record is not None
        self.dispatcher.wake(intent.bot_id)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + max(0.0, timeout_seconds)
        while True:
            result = await asyncio.to_thread(self._observe, intent, record.id)
            if result is not None:
                return result
            remaining = deadline - loop.time()
            if remaining <= 0:
                return NewSessionResult(
                    outcome=Outcome.AMBIGUOUS, stage=Stage.ADMISSION,
                    delivery_id=record.id, turn_id=record.turn_id,
                    detail="reset still in flight; retry the same key to observe it",
                )
            await asyncio.sleep(min(0.5, remaining))
