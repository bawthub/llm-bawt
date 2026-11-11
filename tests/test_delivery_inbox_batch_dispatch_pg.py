"""Reviewer probe for actual restart-child dispatch; run from tests/."""
import asyncio
from types import SimpleNamespace

import pytest
from sqlalchemy import text

from llm_bawt.inter_bot_delivery_retry import fail_claim_with_restart_retry
from llm_bawt.service.inter_bot_dispatcher import InterBotDeliveryDispatcher
from tests.test_delivery_inbox_batch_pg import _claim, _opt_in, _send
from tests.test_delivery_inbox_batch_review_pg import _shutdown
from tests.test_turn_gates_pg import gated  # noqa: F401

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]


@pytest.mark.parametrize("recovery", [False, True])
def test_restart_child_keeps_interruption_note_at_actual_dispatch(gated, monkeypatch, recovery):
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
