"""PostgreSQL tests for TASK-1692: the "in the room" freeze.

An operator freeze with ``admit_user`` still lets the user's own chat start a
turn (a Send now stamp scoped to the room gates), queues bot work instead of
refusing it, releases that queue on thaw, and tells the agent per turn. A plain
freeze is unchanged, and another live gate still holds the user's chat.
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest

from llm_bawt.inter_bot_delivery import InterBotDeliveryStore
from llm_bawt.service.operator_freeze import freeze, thaw
from llm_bawt.service.ordinary_turn_queue import OrdinaryTurnQueue
from llm_bawt.service.schemas import ChatCompletionRequest, ChatMessage
from llm_bawt.service.turn_admission import SystemGateRefused
from llm_bawt.service.turn_gates import GateScope, GateScopeError, GateSpec, room_gate_ids
from tests.test_delivery_claim_gate_pg import _claim, _enqueue, _row
from tests.test_turn_gates_pg import _activate, _spec, gated  # noqa: F401  (gated is a fixture)

pytestmark = pytest.mark.usefixtures("private_app_pg")


def _freeze(gated, bot, *, admit_user):
    actor = f"room-{uuid.uuid4().hex[:10]}"
    gated.owners.append(f"freeze:{actor}")
    return actor, freeze(gated.store.engine, actor=actor, reason="in the room",
                         scope=GateScope.for_bots(bot), ttl_seconds=900, admit_user=admit_user)


def _user_says(gated, deliveries, bot, text="nick, iterating live"):
    gated.senders.append(bot)
    queue = OrdinaryTurnQueue(SimpleNamespace(engine=gated.store.engine), deliveries)
    request = ChatCompletionRequest(
        bot_id=bot, user="nick", model="fixture", stream=True,
        messages=[ChatMessage(role="user", content=text)],
        user_message_id=str(uuid.uuid4()), assistant_message_id=str(uuid.uuid4()),
    )
    return queue.decide(request=request, bot_id=bot, user_id="nick", user_prompt=text)


@pytest.mark.integration
def test_admit_user_is_only_for_an_operator_deny_gate():
    with pytest.raises(GateScopeError):
        GateSpec(owner_kind="job", owner_id="j", mode="deny", scope=GateScope.every_agent(),
                 ttl_seconds=60, admit_user=True)
    with pytest.raises(GateScopeError):
        GateSpec(owner_kind="operator", owner_id="o", mode="hold", scope=GateScope.every_agent(),
                 ttl_seconds=60, admit_user=True)


@pytest.mark.integration
def test_plain_freeze_still_refuses_the_users_chat_and_bot_sends(gated):
    bot = gated.bot()
    _actor, frozen = _freeze(gated, bot, admit_user=False)
    assert frozen.admit_user is False and frozen.to_api()["admit_user"] is False
    with gated.store.engine.connect() as conn:
        assert room_gate_ids(conn, bot) == frozenset()
    deliveries = InterBotDeliveryStore(gated.store.config)
    with pytest.raises(SystemGateRefused):
        _user_says(gated, deliveries, bot)
    with pytest.raises(SystemGateRefused):
        _enqueue(gated, bot, "bot to bot")


@pytest.mark.integration
def test_room_freeze_admits_the_users_chat_and_queues_bot_work_until_thaw(gated):
    bot = gated.bot()
    actor, frozen = _freeze(gated, bot, admit_user=True)
    assert frozen.to_api()["admit_user"] is True

    deliveries, bot_row = _enqueue(gated, bot, "bot to bot")  # held, not refused
    assert _row(deliveries, bot_row.id)["status"] == "QUEUED"
    assert _claim(deliveries, bot) is None  # nothing but the user's chat may run

    decision = _user_says(gated, deliveries, bot)
    user_row = deliveries.get(decision.delivery_id)
    assert user_row.metadata["send_now"] == {"user_id": "nick", "gate_ids": [frozen.gate_id]}

    claimed = _claim(deliveries, bot)
    assert claimed is not None and claimed.id == decision.delivery_id
    assert _row(deliveries, bot_row.id)["status"] == "QUEUED"  # the bot's message still waits

    # Thaw releases the queue.
    assert thaw(gated.store.engine, frozen.gate_id, actor=actor).released
    with deliveries.engine.begin() as conn:
        from sqlalchemy import text as sa_text
        conn.execute(sa_text("UPDATE inter_bot_deliveries SET status='DELIVERED' WHERE id=:id"),
                     {"id": claimed.id})
        conn.execute(sa_text("UPDATE turn_logs SET status='ok', ended_at=now() WHERE id=:id"),
                     {"id": claimed.turn_id})
    released = _claim(deliveries, bot)
    assert released is not None and released.id == bot_row.id


@pytest.mark.integration
def test_another_gate_still_holds_the_users_chat_in_a_room_freeze(gated):
    bot = gated.bot()
    _actor, frozen = _freeze(gated, bot, admit_user=True)
    owner = gated.owner()
    quiesce = _activate(gated.store, _spec(owner, GateScope.for_bots(bot))).gate
    deliveries = InterBotDeliveryStore(gated.store.config)

    decision = _user_says(gated, deliveries, bot)

    stamp = deliveries.get(decision.delivery_id).metadata["send_now"]
    assert stamp["gate_ids"] == [frozen.gate_id] and quiesce.id not in stamp["gate_ids"]
    assert _claim(deliveries, bot) is None  # the ops quiesce still holds it


@pytest.mark.integration
def test_bot_authored_ordinary_start_is_never_stamped(gated):
    bot = gated.bot()
    _freeze(gated, bot, admit_user=True)
    deliveries = InterBotDeliveryStore(gated.store.config)
    sender = gated.bot()
    gated.senders.append(sender)
    queue = OrdinaryTurnQueue(SimpleNamespace(engine=gated.store.engine), deliveries)
    request = ChatCompletionRequest(
        bot_id=bot, user="nick", model="fixture", stream=True,
        messages=[ChatMessage(role="user", content="from a bot")],
        user_message_id=str(uuid.uuid4()), assistant_message_id=str(uuid.uuid4()),
    )
    request._internal_inter_bot_sender_id = sender
    decision = queue.decide(request=request, bot_id=bot, user_id="nick", user_prompt="from a bot")
    assert "send_now" not in deliveries.get(decision.delivery_id).metadata
    assert _claim(deliveries, bot) is None


@pytest.mark.integration
def test_notice_is_prefixed_only_while_the_room_freeze_is_live(gated):
    from llm_bawt.room_freeze_notice import with_room_freeze_notice

    bot = gated.bot()
    engine = gated.store.engine
    assert with_room_freeze_notice("hi", engine=engine, bot_id=bot) == "hi"
    actor, frozen = _freeze(gated, bot, admit_user=True)
    noticed = with_room_freeze_notice("hi", engine=engine, bot_id=bot)
    assert noticed.endswith("\n\nhi") and "in the room" in noticed and "Do not commit" in noticed
    assert with_room_freeze_notice("/new", engine=engine, bot_id=bot) == "/new"
    thaw(engine, frozen.gate_id, actor=actor)
    assert with_room_freeze_notice("hi", engine=engine, bot_id=bot) == "hi"
