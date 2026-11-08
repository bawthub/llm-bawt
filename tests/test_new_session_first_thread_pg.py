"""TASK-2151: the first /new of a brand-new owner names the thread it rotates.

With no active thread at claim, the claim used to record a null predecessor,
then the agent turn's binding created the owner's first thread and the rotation
archived it, so the receipt check saw ``[thread] != []`` and reported a reset
that had succeeded as failed.
"""
from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest
from sqlalchemy import text

from llm_bawt.inter_bot_delivery import InterBotDeliveryStore
from llm_bawt.memory.session_reset_lineage import SessionResetLineage
from llm_bawt.service.ordinary_turn_queue import OrdinaryTurnQueue
from llm_bawt.service.schemas import ChatCompletionRequest, ChatMessage

pytest_plugins = ["tests.session_lifecycle_pg_fixtures"]
pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]


def _active(engine, bot, user):
    with engine.connect() as conn:
        return conn.execute(text("""
            SELECT id FROM sessions WHERE bot_id=:bot AND user_id=:user
              AND status='active' AND ended_at IS NULL
        """), {"bot": bot, "user": user}).scalars().all()


def test_first_new_claim_binds_the_thread_its_rotation_archives(rig):
    with rig.engine.begin() as conn:
        conn.execute(text("""
            UPDATE sessions SET status='archived', ended_at=now(), archived_at=now()
            WHERE bot_id=:bot AND user_id=:user
        """), {"bot": rig.bot, "user": rig.user})
    assert _active(rig.engine, rig.bot, rig.user) == []

    deliveries = InterBotDeliveryStore(rig.config)
    queue = OrdinaryTurnQueue(SimpleNamespace(engine=rig.engine), deliveries)
    request = ChatCompletionRequest(
        bot_id=rig.bot, user=rig.user, model="fixture", stream=True,
        messages=[ChatMessage(role="user", content="/new")],
        user_message_id=str(uuid.uuid4()), assistant_message_id=str(uuid.uuid4()),
    )
    decision = queue.decide(
        request=request, bot_id=rig.bot, user_id=rig.user,
        user_prompt="/new", chat_reset=True,
    )
    claim = deliveries.claim_next(rig.bot, claim_owner="first-new", steer_capable=False)
    assert claim and claim.id == decision.delivery_id

    claimed = claim.metadata["new_session_claim_predecessor_id"]
    assert claimed and claimed != rig.session_id
    assert _active(rig.engine, rig.bot, rig.user) == [claimed]

    # The turn's binding now finds that thread instead of minting another, and
    # the keyed rotation archives exactly the claimed predecessor.
    successor = SessionResetLineage.delivery_successor_id(claim.id)
    SessionResetLineage.rotate(
        rig.engine, bot_id=rig.bot, user_id=rig.user, new_id=successor,
        idempotency_key=claim.id, accept_pointer=True, working_memory_mode="pinned",
    )
    with rig.engine.connect() as conn:
        lineage = SessionResetLineage.metadata(conn.execute(text(
            "SELECT session_metadata FROM sessions WHERE id=:id"
        ), {"id": successor}).scalar_one())
    assert lineage["reset_predecessor_ids"] == [claimed]
    assert lineage["rotation_delivery_id"] == claim.id
