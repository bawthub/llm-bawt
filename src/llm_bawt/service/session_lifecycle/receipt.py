"""Persist bridge reset evidence, then commit one application-authored receipt.

A model's text or a DELIVERED transport flag alone is never reset evidence.
The bridge's terminal reset acknowledgement is bound to its deterministic
request, delivery, and newly rotated thread before a receipt can be recorded.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from uuid import NAMESPACE_URL, uuid5

from sqlalchemy import text

from agent_bridge.continuity import RESET_CONFIRMATION, format_reset_receipt

from ..chat_message_text import last_user_text
from ..system_prompt_pin import RESET_SKILLS_KEY
from .intent import NewSessionIntent, NewSessionResult, Origin, Outcome, Stage


def reset_confirmation(session_metadata) -> str:
    """App-authored receipt text citing only bridge-acknowledged continuity.

    Bridge ack prose is never authority. The successor's continuity manifest
    is app-captured and becomes ``restored`` only through the digest-bound
    bridge receipt, so it is the one record this text may describe. No seed,
    an empty selection, or an unacknowledged/failed restore keeps the plain
    confirmation (plus any recorded injected skills).
    """
    metadata = session_metadata if isinstance(session_metadata, dict) else {}
    continuity = metadata.get("continuity")
    restored = isinstance(continuity, dict) and continuity.get("status") == "restored"
    # TASK-1750: skills the reset turn's own render carried, recorded on this
    # session by the pin step; absent (unknown backend, failed resolve) says nothing.
    return format_reset_receipt(continuity if restored else None,
                                metadata.get(RESET_SKILLS_KEY))


def receipt_content(receipt: dict) -> str:
    """Exact text a committed receipt wrote; legacy receipts predate the field."""
    content = receipt.get("content", RESET_CONFIRMATION)
    return content if isinstance(content, str) else ""


class NewSessionReceipt:
    @classmethod
    def reconcile_typed(cls, engine) -> list:
        """Repair a crash between DELIVERED and receipt, without replaying /new."""
        from ...inter_bot_delivery_record import DeliveryRecord
        from .recovery import NewSessionRecovery

        with engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT * FROM inter_bot_deliveries WHERE status='DELIVERED'
                  AND metadata_json->'new_session_intent'->>'origin'='chat_command'
                  AND metadata_json->'new_session_receipt' IS NULL
                ORDER BY updated_at LIMIT 50
            """)).mappings().all()
        settled = []
        for row in rows:
            record = DeliveryRecord.from_mapping(row)
            try:
                intent = cls.typed_intent(record)
            except (TypeError, ValueError):
                intent = None
            result = cls.ensure(engine, record, intent) if intent is not None else None
            if result is None:
                current = NewSessionRecovery.fail_missing_typed_receipt(
                    engine, record.id, "typed reset finished without provider binding and persisted receipt",
                )
            else:
                with engine.connect() as conn:
                    latest = conn.execute(text("""
                        SELECT * FROM inter_bot_deliveries WHERE id=:id
                    """), {"id": record.id}).mappings().one()
                current = DeliveryRecord.from_mapping(latest)
            if current is not None:
                settled.append(current)
        return settled

    @staticmethod
    def typed_intent(record) -> NewSessionIntent | None:
        """Reconstruct only a server-stamped literal command, not model prose."""
        raw = record.metadata.get("new_session_intent")
        if (not isinstance(raw, dict) or raw.get("origin") != Origin.CHAT_COMMAND
                or record.session_policy != "continue"):
            return None
        from .execution import NewSessionExecution
        from .lifecycle import NewSessionLifecycle

        follow_up = NewSessionExecution.typed_follow_up(record.message)
        if record.message.strip() != "/new" and follow_up is None:
            return None
        intent = NewSessionLifecycle._decode(raw)
        if (intent.bot_id != record.target_bot_id or intent.author != record.author
                or intent.follow_up != follow_up):
            raise ValueError("typed reset provenance does not match its delivery")
        return intent

    @staticmethod
    def record_provider_ack(engine, *, request_id: str, bot_id: str, event) -> bool:
        """Store an exact bridge ack before continuing the canonical stream."""
        ack = event.raw.get("new_session_ack") if isinstance(event.raw, dict) else None
        if (not isinstance(ack, dict) or event.run_id != request_id
                or event.provider not in {"claude-code", "codex"}):
            return False
        # A bridge-side seed failure must never become a successful reset.
        if not (ack.get("clean_start") or ack.get("provider_session_id")):
            return False
        if not isinstance(ack.get("thread_session_id"), str) or not ack["thread_session_id"]:
            return False
        digest = ack.get("seed_digest")
        count = ack.get("seeded_count")
        if digest is not None and (
            not isinstance(digest, str) or not digest
            or not isinstance(count, int) or isinstance(count, bool) or count < 0
        ):
            return False
        with engine.begin() as conn:
            result = conn.execute(text("""
                UPDATE inter_bot_deliveries
                SET metadata_json=jsonb_set(metadata_json, '{new_session_provider_ack}',
                                           CAST(:ack AS jsonb)), updated_at=now()
                WHERE status='DISPATCHING' AND delivery_mode='turn'
                  AND inter_bot_deliveries.payload_json->>'inter_bot_bridge_request_id'=:request_id
                  AND target_bot_id=:bot AND user_message_id=:message_id
                  AND metadata_json->'new_session_provider_ack' IS NULL
                  AND (
                    -- The ack names the exact seed it applied; it must be the
                    -- one the app bound to this delivery (TASK-1579). A seeded
                    -- start can never be acknowledged as a clean one.
                    (:seed_digest <> ''
                     AND metadata_json->'new_session_seed'->>'payload_digest'=:seed_digest
                     AND metadata_json->'new_session_seed'->>'count'=CAST(:seeded_count AS text)
                     AND (:seeded_count=0
                          OR (:clean_start=false AND :provider_session_id <> '')))
                    -- Pre-receipt bridges. Claude: a resumable session, or a
                    -- clean start where the app bound no seed. Codex never
                    -- seeded, so only where the app bound no seed.
                    OR (:seed_digest='' AND (
                        (:provider='claude-code'
                         AND ((:clean_start=false AND :provider_session_id <> '')
                              OR (:clean_start=true
                                  AND COALESCE(metadata_json->'new_session_seed'->>'count', '0')='0')))
                        OR (:provider='codex'
                            AND COALESCE(metadata_json->'new_session_seed'->>'count', '0')='0')))
                  )
                  AND (
                    (metadata_json->'new_session_intent'->>'origin'
                       IN ('scheduler', 'chat_command')
                       AND (:provider='claude-code'
                            OR (:provider='codex'
                                AND metadata_json->'new_session_intent'->>'origin'='chat_command'
                                AND ((:clean_start=true AND :provider_session_id='')
                                     OR (:provider_session_id <> ''
                                         AND (payload_json->>'inter_bot_typed_follow_up'='true'
                                              OR :seeded_count > 0))))))
                    OR (session_policy IN ('reset_retain_history', 'reset_without_history')
                        AND reset_status='ROTATED' AND :provider IN ('claude-code', 'codex')
                        AND payload_json->>'session_id'=:thread_id
                        AND (
                          :provider_session_id <> ''
                          OR (payload_json->>'inter_bot_reset_only'='true'
                              AND metadata_json->'new_session_intent'->>'origin'
                                  IN ('context_api', 'session_api')
                              AND (session_policy='reset_without_history'
                                   -- Retaining an empty history is a clean start,
                                   -- provable only by a digest-bound receipt.
                                   OR (session_policy='reset_retain_history'
                                       AND :seed_digest <> '' AND :seeded_count=0))
                              AND :clean_start=true AND :provider_session_id='')
                        ))
                  )
            """), {"ack": json.dumps({**ack, "provider": event.provider}),
                    "request_id": request_id, "provider": event.provider,
                    "clean_start": ack.get("clean_start") is True,
                    "thread_id": ack["thread_session_id"],
                    "provider_session_id": str(ack.get("provider_session_id") or ""),
                    "seed_digest": digest or "",
                    "seeded_count": count if digest else -1,
                    "bot": bot_id,
                    "message_id": event.trigger_message_id or ""})
            return bool(result.rowcount)

    @staticmethod
    def ensure(engine, record, intent) -> NewSessionResult | None:
        """Return completed only after provider evidence and receipt commit.

        This transaction can be retried after an app crash. It preserves the
        canonical assistant bubble ID and overwrites any bridge-generated ack
        with deterministic app text; UI history and turn log agree on refresh.
        """
        chat = record.metadata.get("new_session_execution") == "chat"
        ack = record.metadata.get("new_session_provider_ack")
        if (chat and ack is not None) or (not chat and not isinstance(ack, dict)):
            return None
        from .outbox import NewSessionOutbox

        NewSessionOutbox(engine)
        receipt_id = str(uuid5(NAMESPACE_URL, f"new-session-receipt:{record.id}"))
        with engine.begin() as conn:
            row = conn.execute(text("""
                SELECT d.status, d.user_message_id, d.turn_id, d.target_bot_id,
                       d.payload_json, d.metadata_json, t.status AS turn_status,
                       t.ended_at, t.error_text, t.assistant_message_id,
                       t.trigger_message_id, t.user_id
                FROM inter_bot_deliveries d JOIN turn_logs t ON t.id=d.turn_id
                WHERE d.id=:id FOR UPDATE OF d
            """), {"id": record.id}).mappings().one_or_none()
            if row is None or row["status"] != "DELIVERED" or row["ended_at"] is None:
                return None
            if (row["turn_status"] not in {"ok", "completed"} or row["error_text"]
                    or row["user_id"] != intent.user_id
                    or row["trigger_message_id"] != row["user_message_id"]
                    or row["target_bot_id"] != intent.bot_id):
                return None
            metadata = row["metadata_json"] or {}
            if ((metadata.get("new_session_execution") == "chat") != chat
                    or metadata.get("new_session_provider_ack") != ack):
                return None
            from ...memory.postgresql import _sanitize_table_name
            storage_bot = _sanitize_table_name(intent.bot_id)
            user_message = conn.execute(text("""
                SELECT session_id, content, author_entity_type, author_entity_id
                FROM messages WHERE bot_id=:storage_bot AND id=:mid
            """), {"storage_bot": storage_bot, "mid": record.user_message_id}).mappings().one_or_none()
            if (user_message is None or not isinstance(user_message["content"], str)
                    or user_message["content"].strip() != record.message
                    or user_message["author_entity_type"] != intent.author.entity_type
                    or user_message["author_entity_id"] != intent.author.entity_id):
                return None
            new_id = user_message["session_id"]
            if (not new_id or new_id == intent.expected_session_id
                    or (not chat and new_id != ack.get("thread_session_id"))):
                return None
            session = conn.execute(text("""
                SELECT bot_id, user_id, session_metadata FROM sessions WHERE id=:session
            """), {"session": new_id}).mappings().one_or_none()
            if session is None or session["bot_id"] != intent.bot_id or session["user_id"] != intent.user_id:
                return None
            verified_predecessor = intent.expected_session_id
            if intent.origin is Origin.CHAT_COMMAND:
                from ...memory.session_reset_lineage import SessionResetLineage

                if new_id != SessionResetLineage.delivery_successor_id(record.id):
                    return None
                lineage = SessionResetLineage.metadata(session["session_metadata"])
                predecessors = lineage.get("reset_predecessor_ids")
                claim = (row["metadata_json"] or {})
                # WAIT commands may sit behind another /new: the predecessor
                # observed at enqueue is not the one this delivery rotates.
                # Require the claim-time binding and the rotation's durable
                # successor link to agree; SKIP scheduler fences stay unchanged.
                if "new_session_claim_predecessor_id" not in claim:
                    return None
                claimed_predecessor = claim["new_session_claim_predecessor_id"]
                if (claimed_predecessor is not None
                        and (not isinstance(claimed_predecessor, str) or not claimed_predecessor)):
                    return None
                if (lineage.get("rotation_delivery_id") != record.id
                        or predecessors != ([claimed_predecessor] if claimed_predecessor else [])):
                    return None
                verified_predecessor = claimed_predecessor
                for predecessor in predecessors:
                    prior = conn.execute(text("""
                        SELECT bot_id, user_id, session_metadata FROM sessions WHERE id=:id
                    """), {"id": predecessor}).mappings().one_or_none()
                    if (prior is None or prior["bot_id"] != intent.bot_id
                            or prior["user_id"] != intent.user_id
                            or SessionResetLineage.metadata(prior["session_metadata"])
                            .get("reset_successor_id") != new_id):
                        return None
            provider_key = None
            if chat:
                # A stateless chat model has no provider session or reset ack.
                # Its proof is the keyed lineage and exact completed transcript.
                if (session["session_metadata"] or {}).get("agent_session_keys"):
                    return None
            else:
                provider_key = ack.get("provider_session_id")
                provider = ack.get("provider")
                if provider not in {"claude-code", "codex"}:
                    return None
                bound_key = (session["session_metadata"] or {}).get(
                    "agent_session_keys", {}).get(provider.replace("-", "_"))
                if provider_key:
                    if (provider == "codex" and intent.follow_up is None) or bound_key != provider_key:
                        return None
                elif ack.get("clean_start") is not True or bound_key:
                    # A clean reset defers provider minting until the next turn;
                    # a stale key on the successor disproves the fresh binding.
                    return None
            if intent.origin not in {Origin.SCHEDULER, Origin.CHAT_COMMAND} or not row["assistant_message_id"]:
                return None
            # Stateless chat streams the plain text before this commit; agents
            # get the seed they actually restored, read under this same lock.
            confirmation = RESET_CONFIRMATION if chat else reset_confirmation(session["session_metadata"])
            if intent.follow_up is not None:
                return NewSessionReceipt._confirm_follow_up(
                    conn, record, row, intent, storage_bot, provider_key, new_id,
                    verified_predecessor, receipt_id, user_message["content"], confirmation,
                )
            prior_receipt = (row["metadata_json"] or {}).get("new_session_receipt")
            if prior_receipt is not None:
                if (not isinstance(prior_receipt, dict)
                        or prior_receipt.get("id") != receipt_id
                        or prior_receipt.get("message_id") != row["assistant_message_id"]
                        or prior_receipt.get("new_session_id") != new_id
                        or prior_receipt.get("provider_session_id") != provider_key):
                    return None
                # Replay completes only against the exact committed bubble,
                # as the follow-up and policy receipts already require.
                committed = conn.execute(text("""
                    SELECT content FROM messages WHERE bot_id=:storage_bot AND id=:assistant
                      AND role='assistant' AND session_id=:session
                """), {"storage_bot": storage_bot, "assistant": row["assistant_message_id"],
                        "session": new_id}).scalar_one_or_none()
                if committed != receipt_content(prior_receipt):
                    return None
                return NewSessionResult(
                    outcome=Outcome.COMPLETED, stage=Stage.RECEIPT,
                    delivery_id=record.id, turn_id=record.turn_id,
                    old_session_id=verified_predecessor,
                    new_session_id=new_id, receipt_message_id=row["assistant_message_id"],
                )
            # Chat never calls a provider for bare /new. Require that the
            # normal turn finalizer persisted the app-authored confirmation;
            # unlike an agent, there is no bridge prose to overwrite.
            if chat:
                reply = conn.execute(text("""
                    SELECT id FROM messages WHERE bot_id=:storage_bot AND id=:assistant
                      AND role='assistant' AND session_id=:session
                      AND content=:content AND author_entity_type='bot'
                      AND author_entity_id=:bot
                """), {"storage_bot": storage_bot, "assistant": row["assistant_message_id"],
                        "session": new_id, "content": RESET_CONFIRMATION,
                        "bot": intent.bot_id}).scalar_one_or_none()
                if (not reply or conn.execute(text("""
                    SELECT response_text FROM turn_logs WHERE id=:turn
                """), {"turn": row["turn_id"]}).scalar_one() != RESET_CONFIRMATION):
                    return None
            else:
                # The bridge acknowledgement is real, but its prose is not
                # completion authority; atomically replace it with app text.
                reply = conn.execute(text("""
                    UPDATE messages SET content=:content, author_entity_type='bot',
                        author_entity_id=:bot, processed=true
                    WHERE bot_id=:storage_bot AND id=:assistant AND role='assistant'
                      AND session_id=:session RETURNING id
                """), {"content": confirmation, "bot": intent.bot_id,
                        "storage_bot": storage_bot, "assistant": row["assistant_message_id"],
                        "session": new_id}).scalar_one_or_none()
                if not reply:
                    return None
                conn.execute(text("""
                    UPDATE turn_logs SET response_text=:content WHERE id=:turn
                """), {"content": confirmation, "turn": row["turn_id"]})
            conn.execute(text("""
                UPDATE inter_bot_deliveries
                SET metadata_json=jsonb_set(metadata_json, '{new_session_receipt}',
                                           CAST(:receipt AS jsonb)), updated_at=now()
                WHERE id=:id
            """), {"id": record.id, "receipt": json.dumps({
                "id": receipt_id, "message_id": reply, "new_session_id": new_id,
                "provider_session_id": provider_key, "content": confirmation,
                "confirmed_at": datetime.now(timezone.utc).isoformat(),
            })})
            NewSessionOutbox.enqueue(
                conn, delivery_id=record.id, bot_id=intent.bot_id,
                user_id=intent.user_id, new_session_id=new_id, message_id=reply,
            )
        return NewSessionResult(
            outcome=Outcome.COMPLETED, stage=Stage.RECEIPT,
            delivery_id=record.id, turn_id=record.turn_id,
            old_session_id=verified_predecessor,
            new_session_id=new_id, receipt_message_id=reply,
        )

    @staticmethod
    def _confirm_follow_up(
        conn, record, row, intent, storage_bot, provider_key, new_id, predecessor, receipt_id,
        visible_command, confirmation,
    ) -> NewSessionResult | None:
        """Keep the model's task answer and add one separately-authored reset receipt."""
        from .execution import NewSessionExecution
        from .outbox import NewSessionOutbox

        payload = row["payload_json"] or {}
        command = last_user_text(payload.get("messages"))
        if (payload.get("inter_bot_typed_follow_up") is not True
                or not isinstance(command, str) or command != visible_command
                or command.strip() != record.message
                or NewSessionExecution.typed_follow_up(command) != intent.follow_up
                or (provider_key is None and row["metadata_json"].get("new_session_execution") != "chat")
                or (provider_key is not None
                    and (not isinstance(provider_key, str) or not provider_key))):
            return None
        answer = conn.execute(text("""
            SELECT session_id, content, author_entity_type, author_entity_id FROM messages
            WHERE bot_id=:bot AND id=:id AND role='assistant'
        """), {"bot": storage_bot, "id": row["assistant_message_id"]}).mappings().one_or_none()
        if (answer is None or answer["session_id"] != new_id
                or (provider_key is None and (
                    answer["author_entity_type"] != "bot"
                    or answer["author_entity_id"] != intent.bot_id))
                or answer["content"] != conn.execute(text("""
                    SELECT response_text FROM turn_logs WHERE id=:turn
                """), {"turn": record.turn_id}).scalar_one()):
            return None
        message_id = str(uuid5(NAMESPACE_URL, f"new-session-typed-follow-up:{record.id}"))
        previous = (row["metadata_json"] or {}).get("new_session_receipt")
        if previous is not None:
            if (not isinstance(previous, dict) or previous.get("id") != receipt_id
                    or previous.get("message_id") != message_id
                    or previous.get("new_session_id") != new_id
                    or previous.get("provider_session_id") != provider_key):
                return None
            committed = conn.execute(text("""
                SELECT content FROM messages WHERE bot_id=:bot AND id=:id
                  AND role='assistant' AND session_id=:session
            """), {"bot": storage_bot, "id": message_id,
                    "session": new_id}).scalar_one_or_none()
            if committed != receipt_content(previous):
                return None
        else:
            timestamp = conn.execute(text("""
                SELECT timestamp FROM messages WHERE bot_id=:bot AND id=:id
            """), {"bot": storage_bot, "id": record.user_message_id}).scalar_one()
            conn.execute(text("""
                INSERT INTO messages (bot_id, id, role, content, timestamp,
                                      session_id, author_entity_type,
                                      author_entity_id, processed)
                VALUES (:bot, :id, 'assistant', :content, :timestamp,
                        :session, 'bot', :author, true)
            """), {"bot": storage_bot, "id": message_id,
                    "content": confirmation,
                    "timestamp": float(timestamp) - 0.0001,
                    "session": new_id, "author": intent.bot_id})
            conn.execute(text("""
                UPDATE inter_bot_deliveries
                SET metadata_json=jsonb_set(metadata_json, '{new_session_receipt}',
                                           CAST(:receipt AS jsonb)), updated_at=now()
                WHERE id=:id
            """), {"id": record.id, "receipt": json.dumps({
                "id": receipt_id, "message_id": message_id,
                "new_session_id": new_id,
                "provider_session_id": provider_key,
                "content": confirmation,
                "confirmed_at": datetime.now(timezone.utc).isoformat(),
            })})
            NewSessionOutbox.enqueue(
                conn, delivery_id=record.id, bot_id=intent.bot_id,
                user_id=intent.user_id, new_session_id=new_id, message_id=message_id,
            )
        return NewSessionResult(
            outcome=Outcome.COMPLETED, stage=Stage.RECEIPT,
            delivery_id=record.id, turn_id=record.turn_id,
            old_session_id=predecessor, new_session_id=new_id,
            receipt_message_id=message_id,
        )
