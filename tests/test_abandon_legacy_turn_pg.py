"""TASK-2541: closing an abandoned legacy turn never overrides a prior outcome."""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import text

from llm_bawt.service.turn_logs import TurnLogStore
from tests.pg_isolation import require_test_db_config

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]


@pytest.fixture
def store():
    store = TurnLogStore(require_test_db_config())
    ids: list[str] = []
    yield store, ids
    with store.engine.begin() as conn:
        for turn_id in ids:
            conn.execute(text("DELETE FROM turn_logs WHERE id = :id"), {"id": turn_id})


def _open(store, ids, status):
    turn_id = f"turn-{uuid.uuid4().hex}"
    ids.append(turn_id)
    store.save_turn(
        turn_id=turn_id, request_id=None, path="/v1/chat/completions", stream=True,
        model="fixture-model", bot_id="fixture-bot", user_id="fixture-user",
        status=status, latency_ms=None, user_prompt="hi", request_payload=None,
        response_text="", tool_calls=None,
    )
    return turn_id


@pytest.mark.parametrize("status,expected", [
    ("streaming", "cancelled"), ("pending", "cancelled"), ("cancelling", "aborted"),
])
def test_open_legacy_turn_closes(store, status, expected):
    s, ids = store
    turn_id = _open(s, ids, status)
    assert s.abandon_open_legacy_turn(turn_id) is True
    row = s.get_turn(turn_id)
    assert (row.status, row.end_reason) == (expected, "aborted")
    assert row.ended_at is not None


def test_settled_turn_is_left_alone(store):
    s, ids = store
    turn_id = _open(s, ids, "streaming")
    s.update_turn(turn_id=turn_id, status="aborted", end_reason="user_stop", latency_ms=5)
    before = s.get_turn(turn_id)
    assert s.abandon_open_legacy_turn(turn_id) is False
    after = s.get_turn(turn_id)
    assert (after.status, after.end_reason, after.ended_at) == (
        before.status, before.end_reason, before.ended_at)
