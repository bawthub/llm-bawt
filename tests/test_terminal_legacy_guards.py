"""Unfenced legacy stores cannot mutate protocol rows, including open rows."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

import pytest
from sqlalchemy import create_engine
from sqlmodel import Session

from llm_bawt.service.turn_logs import TurnLog, TurnLogStore
from llm_bawt.service.turn_terminal_protocol import FollowerFenceLost
from llm_bawt.service.turn_termination import settle_open_turn


PROTOCOL_STATES = ("binding", "open", "pending", "projected", "published", "publication_expired")


@pytest.fixture
def store(monkeypatch):
    engine = create_engine("sqlite://")
    TurnLog.__table__.create(engine)
    result = TurnLogStore.__new__(TurnLogStore)
    result.engine = engine
    result.ttl_hours = None
    effects = Mock()
    monkeypatch.setattr(
        "llm_bawt.service.turn_terminal_effects.record_terminal_effects", effects,
    )
    monkeypatch.setattr(
        "llm_bawt.service.turn_termination.reconcile_finished_tool", Mock(),
    )
    monkeypatch.setattr(
        "llm_bawt.service.usage.balance.balance_update_for_turn", lambda *args: None,
    )
    result.effects = effects
    yield result
    engine.dispose()


def add(store, *, turn_id="protocol", state="open", **extra):
    values = dict(
        id=turn_id, bot_id="test-terminal", user_id="test-user",
        terminal_protocol_state=state, status="streaming",
        response_text="original partial", reasoning="original reasoning",
        assistant_message_id="assistant", history_session_id="session",
        request_json='{"audit":"preserve"}', token_usage_json='{"output_tokens":4}',
    )
    values.update(extra)
    with Session(store.engine) as session:
        session.add(TurnLog(**values))
        session.commit()


def state_of(store, turn_id="protocol"):
    return store.get_turn(turn_id).model_dump()


@pytest.mark.parametrize("state", PROTOCOL_STATES)
def test_generic_update_cannot_bypass_lease_or_snapshot(store, state):
    now = datetime.now(timezone.utc)
    add(
        store, state=state,
        ended_at=None if state == "open" else now,
        terminal_snapshot_json=None if state == "open" else '{"winner":"original"}',
    )
    before = state_of(store)
    store.update_turn(
        turn_id="protocol", status="error", end_reason="error",
        response_text="stale replacement", assistant_message_id="wrong-assistant",
        request_payload={"messages": []}, token_usage={"output_tokens":999},
        latency_ms=999, error_text="wrong failure", animation="wrong",
        question_id="wrong-question", parent_turn_id="wrong-parent",
        agent_request_id="wrong-run", agent_session_key="wrong-key",
        tts_scrubbed=True, abort_cause="late-cancellation",
    )
    assert state_of(store) == before
    store.effects.assert_not_called()


@pytest.mark.parametrize("reasoning", [None, "stale reasoning"])
def test_both_partial_sql_branches_exclude_protocol_but_keep_legacy(store, reasoning):
    add(store)
    add(store, turn_id="legacy", state=None)
    before = state_of(store)
    for turn_id in ("protocol", "legacy"):
        store.update_partial_response(
            turn_id=turn_id, response_text="replacement", reasoning=reasoning,
        )
    assert state_of(store) == before
    legacy = store.get_turn("legacy")
    assert legacy.response_text == "replacement"
    assert legacy.reasoning == ("original reasoning" if reasoning is None else reasoning)


def test_timeout_repair_is_legacy_only(store):
    ended = datetime.now(timezone.utc)
    for turn_id, state in (("protocol", "pending"), ("legacy", None)):
        add(
            store, turn_id=turn_id, state=state, status="timeout",
            end_reason="timeout", ended_at=ended,
            terminal_snapshot_json='{"winner":"timeout"}' if state else None,
        )
    before = state_of(store)
    for turn_id in ("protocol", "legacy"):
        store.update_turn(
            turn_id=turn_id, status="ok", end_reason="stop",
            response_text="late success", latency_ms=60000,
        )
    assert state_of(store) == before
    legacy = store.get_turn("legacy")
    assert legacy.status == "ok" and legacy.end_reason == "stop"
    assert legacy.response_text == "late success" and legacy.ended_at != ended
    store.effects.assert_called_once()


@pytest.mark.parametrize("state", PROTOCOL_STATES)
def test_shared_legacy_settle_refuses_protocol_before_any_mutation(store, state):
    add(store, state=state)
    before = state_of(store)
    with Session(store.engine) as session:
        row = session.get(TurnLog, "protocol")
        with pytest.raises(FollowerFenceLost, match="immutable commit"):
            settle_open_turn(
                session, row, audit_key="user_stop", audit={"late": True},
                now=datetime.now(timezone.utc),
            )
        session.commit()
    assert state_of(store) == before
    store.effects.assert_not_called()


def test_confirmed_start_reap_ignores_even_expired_protocol_owner(store):
    now = datetime.now(timezone.utc)
    add(
        store, follower_owner="crashed-owner",
        follower_lease_until=now - timedelta(minutes=1),
        created_at=now - timedelta(minutes=2),
    )
    add(store, turn_id="legacy", state=None, created_at=now - timedelta(minutes=2))
    add(store, turn_id="current", state=None, created_at=now)
    before = state_of(store)
    reaped = store.reap_other_open_turns(bot_id="test-terminal", current_turn_id="current")
    assert [row["id"] for row in reaped] == ["legacy"]
    assert state_of(store) == before
    assert store.get_turn("legacy").status == "timeout"
    assert store.get_turn("current").ended_at is None
    store.effects.assert_called_once()


@pytest.mark.parametrize("state,history_done,removed", [
    (None, False, True),
    ("binding", False, False),
    ("open", False, False),
    ("pending", False, False),
    ("projected", True, False),
    ("published", False, False),
    ("published", True, True),
    ("publication_expired", False, False),
    ("publication_expired", True, True),
])
def test_short_telemetry_ttl_preserves_unfinished_protocol_work(store, state, history_done, removed):
    now = datetime.now(timezone.utc)
    add(
        store, state=state, created_at=now - timedelta(hours=2),
        terminal_history_projected_at=now if history_done else None,
    )
    store.ttl_hours = 1
    store._cleanup_expired_if_due(force=True)
    assert (store.get_turn("protocol") is None) is removed
