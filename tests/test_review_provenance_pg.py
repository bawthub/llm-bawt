"""TASK-1249 on real PostgreSQL: the review provenance reads see only this turn's task deliveries."""
from __future__ import annotations

import json
import uuid

import pytest
from sqlalchemy import text

from llm_bawt.inter_bot_delivery import InterBotDeliveryStore
from llm_bawt.inter_bot_delivery_steer_fence import STEER_REJECTED_PENDING, STEER_STARTED
from llm_bawt.mcp_server import review_provenance as provenance
from llm_bawt.utils.config import Config
from llm_bawt.utils.db import get_shared_engine

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]

REVIEWER = "fixture-reviewer"
AUTHOR = "fixture-author"


@pytest.fixture
def engine(private_app_pg):
    InterBotDeliveryStore(private_app_pg)  # app-owned delivery DDL
    engine = get_shared_engine(Config())
    assert engine is not None
    return engine


def _delivery(engine, *, sender=AUTHOR, target=REVIEWER, target_turn, task_id, sender_turn,
              status="DELIVERED", mode="turn", accepted=True, metadata=None):
    key = uuid.uuid4().hex
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO inter_bot_deliveries
                (id, sender_bot_id, author_entity_id, target_bot_id, message, user_message_id,
                 turn_id, task_id, status, target_turn_id, sender_turn_id, delivery_mode,
                 transport_accepted_at, metadata_json)
            VALUES (:id, :sender, :sender, :target, 'review please', :msg, :turn, :task,
                    :status, :target_turn, :sender_turn, :mode,
                    CASE WHEN :accepted THEN now() END, CAST(:metadata AS jsonb))
        """), {"id": f"dlv-{key}", "sender": sender, "target": target, "msg": f"msg-{key}",
               "turn": f"turn-{key}", "task": task_id, "target_turn": target_turn,
               "sender_turn": sender_turn, "status": status, "mode": mode,
               "accepted": accepted, "metadata": json.dumps(metadata or {})})


def _started(generation=0):
    return {"steer_generation": generation, STEER_STARTED: {"generation": generation}}


def _turn(engine, bot_id, model):
    turn_id = f"turn-{uuid.uuid4().hex}"
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO turn_logs (id, created_at, path, stream, bot_id, user_id, status,
                                   user_prompt, response_text, model)
            VALUES (:id, now(), '/test', true, :bot, 'fixture-user', 'ok', '', '', :model)
        """), {"id": turn_id, "bot": bot_id, "model": model})
    return turn_id


def test_requesting_deliveries_match_this_turn_task_and_sender_only(engine):
    review_turn = f"turn-{uuid.uuid4().hex}"
    _delivery(engine, target_turn=review_turn, task_id="TASK-7", sender_turn="turn-s1")
    _delivery(engine, target_turn=review_turn, task_id="cfixturetask0001", sender_turn="turn-s1")
    # Same turn, another task; another turn, same task; the reviewer's own send.
    _delivery(engine, target_turn=review_turn, task_id="TASK-8", sender_turn="turn-s2")
    _delivery(engine, target_turn="turn-elsewhere", task_id="TASK-7", sender_turn="turn-s3")
    _delivery(engine, sender=REVIEWER, target_turn=review_turn, task_id="TASK-7", sender_turn="turn-s4")
    # Another bot's turn that happens to share the id is not this reviewer's.
    _delivery(engine, target="fixture-other", target_turn=review_turn, task_id="TASK-7",
              sender_turn="turn-s5")

    rows = provenance.requesting_deliveries(review_turn, REVIEWER, ["cfixturetask0001", "task-7"])
    assert rows == [(AUTHOR, "turn-s1")]
    assert provenance.requesting_deliveries(review_turn, REVIEWER, []) == []


@pytest.mark.parametrize(("label", "row"), [
    # Claimed for this turn but never transported: target_turn_id is written at claim.
    ("claimed steer", {"status": "STEERING", "mode": "steer", "accepted": False}),
    ("accepted steer, not started", {"status": "STEERING", "mode": "steer"}),
    ("steer with a pending rejection", {"status": "STEERING", "mode": "steer", "metadata": {
        "steer_generation": 0, STEER_REJECTED_PENDING: {"generation": 0, "proof": "refused"}}}),
    ("rejected and re-queued", {"status": "QUEUED", "mode": "steer_rejected", "accepted": False,
                                "metadata": {"steer_generation": 1}}),
    # Generation 0 started earlier; the row was reopened and generation 1 is not started.
    ("stale start marker", {"status": "STEERING", "mode": "steer",
                            "metadata": {"steer_generation": 1, STEER_STARTED: {"generation": 0}}}),
    ("idle claim not yet accepted", {"status": "DISPATCHING", "mode": "turn", "accepted": False}),
    ("failed idle turn", {"status": "FAILED", "mode": "turn"}),
])
def test_a_delivery_this_turn_did_not_provably_receive_is_no_requester(engine, label, row):
    review_turn = f"turn-{uuid.uuid4().hex}"
    _delivery(engine, target_turn=review_turn, task_id="TASK-7", sender_turn="turn-s1", **row)
    assert provenance.requesting_deliveries(review_turn, REVIEWER, ["task-7"]) == [], label


def test_an_unreceived_claim_neither_invents_nor_crowds_out_a_requester(engine):
    review_turn = f"turn-{uuid.uuid4().hex}"
    _delivery(engine, target_turn=review_turn, task_id="TASK-7", sender_turn="turn-s1",
              status="STEERING", mode="steer", metadata=_started())
    # Another bot's same-task steer, claimed for this turn but not started.
    _delivery(engine, sender="fixture-third", target_turn=review_turn, task_id="TASK-7",
              sender_turn="turn-s2", status="STEERING", mode="steer")
    assert provenance.requesting_deliveries(review_turn, REVIEWER, ["task-7"]) == [
        (AUTHOR, "turn-s1")]


def test_turn_model_is_read_from_the_turns_own_bot(engine):
    turn_id = _turn(engine, AUTHOR, "model-a@path-a")
    assert provenance.turn_model(turn_id, AUTHOR) == "model-a@path-a"
    assert provenance.turn_model(turn_id, REVIEWER) is None
    assert provenance.turn_model("turn-missing", AUTHOR) is None
