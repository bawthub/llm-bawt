"""Independent queued-reset receipt checks through ordinary admission."""
from __future__ import annotations

import asyncio
import json
import uuid
from types import SimpleNamespace

import pytest
from sqlalchemy import text

from llm_bawt.inter_bot_delivery import InterBotDeliveryStore
from llm_bawt.memory.session_reset_lineage import SessionResetLineage
from llm_bawt.service.ordinary_turn_queue import OrdinaryTurnQueue
from llm_bawt.service.schemas import ChatCompletionRequest, ChatMessage
from llm_bawt.service.session_lifecycle.receipt import NewSessionReceipt
from tests.test_new_session_typed_pg import _admit, _dispatcher, _finish_bridge

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]


@pytest.mark.parametrize("prompt", [" /new", "/new ", "/new\n"])
def test_standalone_command_whitespace_keeps_reset_receipt_contract(rig, prompt):
    deliveries = InterBotDeliveryStore(rig.config)
    queue = OrdinaryTurnQueue(SimpleNamespace(engine=rig.engine), deliveries)
    request = ChatCompletionRequest(
        bot_id=rig.bot, user=rig.user, model="fixture", stream=True,
        messages=[ChatMessage(role="user", content=prompt)],
        user_message_id=str(uuid.uuid4()), assistant_message_id=str(uuid.uuid4()),
    )
    decision = queue.decide(request=request, bot_id=rig.bot, user_id=rig.user,
                            user_prompt=prompt)
    record = deliveries.get(decision.delivery_id)
    # The actual bridge recognizes these as standalone /new, not follow-up text.
    assert record.metadata.get("new_session_intent", {}).get("origin") == "chat_command"
    assert NewSessionReceipt.typed_intent(record) is not None


def test_two_queued_new_commands_receipt_their_actual_predecessors(rig):
    deliveries, first = _admit(rig)
    _, second = _admit(rig)
    dispatcher = _dispatcher(deliveries)
    successors = []
    predecessor = rig.session_id
    for record in (first, second):
        claim, _payload, successor, event = _finish_bridge(rig, deliveries, record)
        successors.append(successor)
        assert NewSessionReceipt.record_provider_ack(
            rig.engine, request_id=event.run_id, bot_id=rig.bot, event=event,
        )
        delivered = deliveries.mark_delivered(
            claim.id, claim.claim_token, response_model="fixture", response_chars=12,
        )
        asyncio.run(dispatcher._finish_delivered(delivered))
        current = deliveries.get(record.id)
        assert current.status == "DELIVERED", f"reset {len(successors)}: {current.last_error}"
        assert current.metadata["new_session_receipt"]["new_session_id"] == successor
        result = NewSessionReceipt.ensure(
            rig.engine, current, NewSessionReceipt.typed_intent(current),
        )
        assert result.old_session_id == predecessor
        predecessor = successor
    assert successors[0] != successors[1]
    assert deliveries.claim_next(rig.bot, claim_owner="replay", steer_capable=False) is None


def test_unaccepted_retry_preserves_first_rotation_predecessor(rig):
    deliveries, record = _admit(rig)
    first_claim = deliveries.claim_next(rig.bot, claim_owner="first", steer_capable=False)
    successor = SessionResetLineage.delivery_successor_id(record.id)
    # App rotation committed, but the provider never accepted the command.
    with rig.engine.begin() as conn:
        SessionResetLineage.rotate(
            None, connection=conn, bot_id=rig.bot, user_id=rig.user,
            new_id=successor, idempotency_key=record.id, accept_pointer=True,
            working_memory_mode="pinned",
        )
    assert deliveries.requeue(record.id, first_claim.claim_token,
                              "bridge not ready", delay_seconds=0) is not None
    claim, _payload, retried_successor, event = _finish_bridge(rig, deliveries, record)
    assert retried_successor == successor
    assert NewSessionReceipt.record_provider_ack(
        rig.engine, request_id=event.run_id, bot_id=rig.bot, event=event,
    )
    delivered = deliveries.mark_delivered(
        claim.id, claim.claim_token, response_model="fixture", response_chars=12,
    )
    asyncio.run(_dispatcher(deliveries)._finish_delivered(delivered))
    current = deliveries.get(record.id)
    assert current.status == "DELIVERED", current.last_error
    result = NewSessionReceipt.ensure(
        rig.engine, current, NewSessionReceipt.typed_intent(current),
    )
    assert result.old_session_id == rig.session_id
    assert result.new_session_id == successor


@pytest.mark.parametrize("damage", [
    "wrong_delivery", "missing_predecessors", "extra_predecessor", "wrong_backlink",
])
def test_typed_receipt_rejects_corrupted_rotation_lineage(rig, damage):
    deliveries, record = _admit(rig)
    claim, _payload, successor, event = _finish_bridge(rig, deliveries, record)
    assert NewSessionReceipt.record_provider_ack(
        rig.engine, request_id=event.run_id, bot_id=rig.bot, event=event,
    )
    delivered = deliveries.mark_delivered(
        claim.id, claim.claim_token, response_model="fixture", response_chars=12,
    )
    target = rig.session_id if damage == "wrong_backlink" else successor
    with rig.engine.begin() as conn:
        metadata = conn.execute(text(
            "SELECT session_metadata FROM sessions WHERE id=:id"
        ), {"id": target}).scalar_one()
        if damage == "wrong_delivery":
            metadata["rotation_delivery_id"] = "another-delivery"
        elif damage == "missing_predecessors":
            metadata.pop("reset_predecessor_ids")
        elif damage == "extra_predecessor":
            metadata["reset_predecessor_ids"].append("unrelated-predecessor")
        else:
            metadata["reset_successor_id"] = "another-successor"
        conn.execute(text("UPDATE sessions SET session_metadata=:meta WHERE id=:id"),
                     {"id": target, "meta": json.dumps(metadata)})
    assert NewSessionReceipt.ensure(
        rig.engine, delivered, NewSessionReceipt.typed_intent(delivered),
    ) is None
    assert "new_session_receipt" not in deliveries.get(record.id).metadata
