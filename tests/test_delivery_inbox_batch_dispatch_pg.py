"""Reviewer probe for actual restart-child dispatch; run from tests/."""
import asyncio
import json
from types import SimpleNamespace

import pytest
from sqlalchemy import text

from llm_bawt.inter_bot_delivery_retry import fail_claim_with_restart_retry
from llm_bawt.service.inter_bot_claims import turn_admission_claim, validate_inter_bot_claim
from llm_bawt.service.inter_bot_dispatcher import InterBotDeliveryDispatcher
from llm_bawt.service.schemas import ChatCompletionRequest
from llm_bawt.service.turn_admission import turn_start_extensions, write_reservation
from llm_bawt.service.turn_lifecycle import TurnLifecycleMixin
from tests.test_delivery_inbox_batch_pg import _claim, _opt_in, _send
from tests.test_delivery_inbox_batch_review_pg import _shutdown
from tests.test_turn_gates_pg import gated  # noqa: F401

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]


@pytest.mark.parametrize("recovery", [False, True])
def test_restart_child_keeps_interruption_note_at_actual_dispatch(gated, monkeypatch, recovery):  # noqa: F811
    monkeypatch.setattr("llm_bawt.inter_bot_delivery_retry.resolve_restart_retry_text", lambda _: "Resume")
    bot = gated.bot()
    _opt_in(gated, bot)
    deliveries, _ = _send(gated, bot, "head-A")
    _, follower = _send(gated, bot, "follower-B")
    head = _claim(deliveries, bot)
    _shutdown(deliveries, head)
    if recovery:
        with deliveries.engine.begin() as conn:
            conn.execute(text("UPDATE inter_bot_deliveries SET lease_expires_at=now()-interval '1 second' WHERE id=:id"),
                         {"id": head.id})
        rows = deliveries.recover_expired()
        child = next(row for row in rows if row.metadata.get("restart_retry_of") == head.id)
    else:
        _, child = fail_claim_with_restart_retry(deliveries, head, head.claim_token, "shutdown")
    assert child is not None
    stored = deliveries.payload(child.id)["messages"][-1]["content"]
    assert stored.startswith("Resume\n\n") and "follower-B" in stored
    assert deliveries.get(follower.id).metadata["batch_carried_by"] == child.id
    claimed = _claim(deliveries, bot)
    assert claimed.id == child.id
    captured = []

    class Captured(BaseException):
        pass

    async def capture(request):
        captured.append(request.messages[-1].content)
        raise Captured()
        yield

    dispatcher = InterBotDeliveryDispatcher.__new__(InterBotDeliveryDispatcher)
    dispatcher.store = deliveries
    dispatcher.service = SimpleNamespace(chat_completion_stream=capture)
    with pytest.raises(Captured):
        asyncio.run(dispatcher._dispatch(claimed))
    assert captured[0].startswith("Resume\n\n"), captured[0]


@pytest.mark.parametrize("batch", [False, True], ids=["single", "batch"])
def test_dispatch_preserves_batch_audit_ids_through_completion(gated, batch):  # noqa: F811
    """Use real start, enrichment and terminal writers, without a provider."""
    bot = gated.bot()
    _opt_in(gated, bot)
    deliveries, head = _send(
        gated, bot, "head-A",
        inter_bot_delivery_ids=["forged-head", "forged-follower"],
        _internal_inter_bot_delivery_ids=["forged-head", "forged-follower"],
    )
    members = [_send(gated, bot, message)[1] for message in ("B", "C")] if batch else []
    claimed = _claim(deliveries, bot)
    stored_payload = deliveries.payload(head.id)
    captured = []

    class AuditService(TurnLifecycleMixin):
        pass

    class Captured(BaseException):
        pass

    service = AuditService()
    service._turn_log_store = gated.store
    service._default_bot = bot
    service.config = SimpleNamespace(DEBUG_TURN_LOG=False)

    async def capture(request):
        validate_inter_bot_claim(service, request)
        claim = turn_admission_claim(service, request)
        TurnLifecycleMixin._persist_turn_log(
            service, turn_id=head.turn_id, request_id="batch-audit-probe",
            path="/v1/chat/completions", stream=True, model=None,
            bot_id=bot, user_id=request.user or "nick", status="streaming",
            latency_ms=None, user_prompt=request.messages[-1].content,
            prepared_messages=request.messages, response_text="",
            trigger_message_id=request.user_message_id,
            request_extensions=turn_start_extensions(
                request, is_agent_backend=True, admission_claim=claim,
            ),
            admission_claim=claim,
        )
        captured.append(request)
        raise Captured()
        yield

    dispatcher = InterBotDeliveryDispatcher.__new__(InterBotDeliveryDispatcher)
    dispatcher.store = deliveries
    dispatcher.service = service
    dispatcher._loop = None
    dispatcher._wake_event = asyncio.Event()
    service.chat_completion_stream = capture
    service._inter_bot_dispatcher = dispatcher
    with pytest.raises(Captured):
        asyncio.run(dispatcher._dispatch(claimed))

    row = gated.store.get_turn(head.turn_id)
    audit = json.loads(row.request_json)
    assert row.status == "streaming"
    assert audit["inter_bot_delivery_id"] == head.id
    if batch:
        assert audit["inter_bot_delivery_ids"] == [head.id, *(m.id for m in members)]
        assert captured[0].messages[-1].content == claimed.metadata["batch_prompt"]
    else:
        assert "inter_bot_delivery_ids" not in audit

    enriched = [
        {"role": "system", "content": "prepared context"},
        {"role": "user", "content": captured[0].messages[-1].content},
    ]
    service._update_turn_log(turn_id=head.turn_id, prepared_messages=enriched)
    enriched_audit = json.loads(gated.store.get_turn(head.turn_id).request_json)
    assert enriched_audit["messages"] == enriched
    assert enriched_audit["inter_bot_delivery_id"] == head.id
    assert enriched_audit.get("inter_bot_delivery_ids") == audit.get("inter_bot_delivery_ids")

    # Even a later payload with forged correlation cannot overwrite membership
    # already stamped by the trusted dispatcher for this attempt.
    gated.store.update_turn(
        turn_id=head.turn_id,
        request_payload={
            "messages": enriched, "inter_bot_delivery_id": "forged-head",
            **({"inter_bot_delivery_ids": ["forged-head"]} if batch else {}),
        },
    )
    overwritten = json.loads(gated.store.get_turn(head.turn_id).request_json)
    assert overwritten["inter_bot_delivery_id"] == head.id
    assert overwritten.get("inter_bot_delivery_ids") == audit.get("inter_bot_delivery_ids")

    finalized = []
    service._finalize_turn(
        llm_bawt=SimpleNamespace(finalize_response=lambda *args, **kwargs: finalized.append(args)),
        turn_id=head.turn_id, response_text="done", tool_context="", tool_call_details=[],
        prepared_messages=enriched, user_prompt=captured[0].messages[-1].content,
        model="test", bot_id=bot, user_id="nick", elapsed_ms=1,
        stream=False, end_reason="stop",
    )
    row = gated.store.get_turn(head.turn_id)
    completed_audit = json.loads(row.request_json)
    assert finalized and row.status == "ok" and row.ended_at is not None
    assert row.response_text == "done" and row.end_reason == "stop"
    assert completed_audit["messages"] == enriched
    assert completed_audit["inter_bot_delivery_id"] == head.id
    assert completed_audit.get("inter_bot_delivery_ids") == audit.get("inter_bot_delivery_ids")
    assert deliveries.payload(head.id) == stored_payload
    assert "inter_bot_claim_token" not in completed_audit


@pytest.mark.parametrize("replacement", [None, ["head", "new-member"]], ids=["single", "rebatched"])
def test_delivery_audit_membership_is_attempt_scoped(gated, replacement):  # noqa: F811
    bot = gated.bot()
    turn_id = f"audit-rearm-{gated.tag}"
    with gated.store.engine.begin() as conn:
        assert write_reservation(
            conn, turn_id=turn_id, bot_id=bot, user_id="nick", user_prompt="old batch",
            request_json={"inter_bot_delivery_id": "old-head", "inter_bot_delivery_ids": ["old-head", "old-member"]},
            trigger_message_id=None,
        )
    gated.store.update_turn(turn_id=turn_id, status="error", end_reason="error")
    payload = {"messages": [], "inter_bot_delivery_id": "head"}
    if replacement:
        payload["inter_bot_delivery_ids"] = replacement
    with gated.store.engine.begin() as conn:
        assert write_reservation(
            conn, turn_id=turn_id, bot_id=bot, user_id="nick", user_prompt="retry",
            request_json=payload, trigger_message_id=None,
        )
    audit = json.loads(gated.store.get_turn(turn_id).request_json)
    assert audit["inter_bot_delivery_id"] == "head"
    assert audit.get("inter_bot_delivery_ids") == replacement


def test_public_payload_cannot_supply_batch_audit_ids():
    request = ChatCompletionRequest.model_validate({
        "messages": [{"role": "user", "content": "ordinary"}],
        "inter_bot_delivery_ids": ["forged-head", "forged-follower"],
        "_internal_inter_bot_delivery_ids": ["forged-head", "forged-follower"],
    })
    assert "inter_bot_delivery_ids" not in turn_start_extensions(
        request, is_agent_backend=True, admission_claim=None,
    )
