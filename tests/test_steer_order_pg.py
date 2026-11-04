"""PostgreSQL tests for TASK-1412: a user's steer never passes their own queued send.

Repro: a /new typed while a turn ran queued behind it; plain text typed later
steered into that turn ahead of it, then the /new reset the session that had
just answered. Composer steers (``queued_send_ahead``) and the dispatcher's
steer jump (``load_steer_jumper``) both now wait behind that row.
"""
from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest
from sqlalchemy import text as sa_text

from llm_bawt.inter_bot_delivery import InterBotDeliveryStore
from llm_bawt.service.ordinary_turn_queue import OrdinaryTurnQueue
from llm_bawt.service.schemas import ChatCompletionRequest, ChatMessage
from llm_bawt.service.steer_order import queued_send_ahead
from tests.test_turn_gates_pg import _open, gated  # noqa: F401  (gated is a fixture)

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]


def _send(gated, bot, content, *, user="nick", session_id=None, voice=False):
    """Admit one composer send exactly as /v1/chat/completions does."""
    if bot not in gated.senders:
        gated.senders.append(bot)  # ordinary rows are sent "from" the target
    deliveries = InterBotDeliveryStore(gated.store.config)
    request = ChatCompletionRequest(
        bot_id=bot, user=user, model="fixture", stream=True, session_id=session_id,
        voice_input=voice, messages=[ChatMessage(role="user", content=content)],
        user_message_id=str(uuid.uuid4()), assistant_message_id=str(uuid.uuid4()),
    )
    decision = OrdinaryTurnQueue(SimpleNamespace(engine=gated.store.engine), deliveries).decide(
        request=request, bot_id=bot, user_id=user, user_prompt=content,
    )
    return deliveries, decision.delivery_id


def _ahead(gated, bot, *, user="nick", session_id="s-live"):
    return queued_send_ahead(gated.store.engine, bot_id=bot, user_id=user, session_id=session_id)


def test_a_queued_new_fences_the_users_later_composer_steer(gated):
    bot = gated.bot()
    _open(gated.store, bot)
    deliveries, reset = _send(gated, bot, "/new")

    assert _ahead(gated, bot) == reset
    assert _ahead(gated, bot, user="someone-else") is None  # only the author's own sends

    with deliveries.engine.begin() as conn:
        conn.execute(sa_text("UPDATE inter_bot_deliveries SET status='CANCELLED' WHERE id=:id"), {"id": reset})
    assert _ahead(gated, bot) is None  # a cancelled /new lifts the fence


def test_only_sends_for_this_conversation_fence(gated):
    bot = gated.bot()
    _open(gated.store, bot)
    _, other_thread = _send(gated, bot, "for an archived thread", session_id="s-archived")
    assert _ahead(gated, bot, session_id="s-live") is None
    assert _ahead(gated, bot, session_id="s-archived") == other_thread

    _, plain = _send(gated, bot, "queued with no explicit thread")
    assert _ahead(gated, bot, session_id="s-live") == plain


def test_a_later_steer_row_never_jumps_the_users_queued_new(gated):
    """TASK-1758/1803 let a steer row pass its author's earlier plain wait
    rows, and an ordinary /new looked like one. A /new at the head is held by
    the maintenance barrier; this is the /new queued behind another wait row."""
    from tests.test_delivery_claim_gate_pg import _enqueue

    bot = gated.bot()
    deliveries, other = _enqueue(gated, bot, "another sender's after-the-turn row")
    with deliveries.engine.begin() as conn:
        conn.execute(sa_text(
            "UPDATE inter_bot_deliveries SET payload_json=payload_json || '{\"prefer_steer\": false}'::jsonb"
            " WHERE id=:id"), {"id": other.id})
    _, reset = _send(gated, bot, "/new")
    _, later = _send(gated, bot, "typed after the /new")
    with deliveries.engine.begin() as conn:  # a ready steer row (as a voice send is)
        conn.execute(sa_text(
            "UPDATE inter_bot_deliveries SET payload_json=payload_json || '{\"prefer_steer\": true}'::jsonb"
            " WHERE id=:id"), {"id": later})
    _open(gated.store, bot)

    assert deliveries.claim_next(bot, claim_owner="worker", steer_capable=True) is None
    with deliveries.engine.connect() as conn:
        rows = dict(conn.execute(sa_text(
            "SELECT id, status FROM inter_bot_deliveries WHERE id IN (:a, :b)"
        ), {"a": reset, "b": later}).all())
    assert rows == {reset: "QUEUED", later: "QUEUED"}
