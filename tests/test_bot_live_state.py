"""TASK-1400: hermetic contracts for the batched live bot state reader.

The PostgreSQL acceptance test (test_bot_live_state_pg.py) reads every source
for real; these pin the wiring: turn state is whatever the admission
classifier says, every source is called once for all bots, and a failing
source is reported unavailable instead of defaulting to idle.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine, event, text as sa_text
from sqlalchemy.pool import StaticPool

from llm_bawt import approval_request_store, bot_session_state, inter_bot_delivery_query
from llm_bawt.bot_live_state import BotIdentity, BotLiveStateReader
from llm_bawt.service import turn_admission, turn_gates
from llm_bawt.service.turn_admission import TargetState, classify_target, classify_targets
from llm_bawt.service.turn_gates import GateBlock

BOTS = ["fixture-busy", "fixture-idle"]


@pytest.fixture
def engine():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False},
                           poolclass=StaticPool)
    with engine.begin() as conn:
        conn.execute(sa_text("""
            CREATE TABLE turn_logs (
                id TEXT PRIMARY KEY, bot_id TEXT, status TEXT, agent_session_key TEXT,
                agent_request_id TEXT, user_id TEXT, created_at TEXT, ended_at TEXT,
                end_reason TEXT)
        """))
    return engine


def _turn(engine, turn_id, bot, *, status="streaming", created="2026-10-24T10:00:00",
          ended=None, end_reason=None):
    with engine.begin() as conn:
        conn.execute(sa_text("""
            INSERT INTO turn_logs (id, bot_id, status, user_id, created_at, ended_at, end_reason)
            VALUES (:id, :bot, :status, 'fixture-user', :created, :ended, :reason)
        """), {"id": turn_id, "bot": bot, "status": status, "created": created,
               "ended": ended, "reason": end_reason})


@pytest.fixture
def sources(monkeypatch):
    """Fake every non-turn source; record each call's bot set."""
    calls: dict[str, list] = {}

    def fake(name, result):
        def call(conn, *args, **kwargs):
            calls.setdefault(name, []).append(sorted(kwargs.get("bot_ids") or args[0]))
            return result
        return call

    monkeypatch.setattr(inter_bot_delivery_query, "in_flight_by_target",
                        fake("queue", {"fixture-busy": (2, None)}))
    monkeypatch.setattr(approval_request_store, "pending_counts_by_bot",
                        fake("approvals", {"fixture-busy": 1}))
    monkeypatch.setattr(turn_gates, "covering_gates", fake("gate", {}))
    monkeypatch.setattr(bot_session_state.BotSessionState, "resolve_many",
                        staticmethod(fake("session", {"fixture-busy": None, "fixture-idle": None})))
    return calls


def _read(engine):
    return BotLiveStateReader(engine).read(BotIdentity(b, "agent", "claude-code") for b in BOTS)


def test_classify_targets_matches_classify_target_per_bot(engine):
    _turn(engine, "t-busy", "fixture-busy")
    _turn(engine, "t-res", "fixture-res", status="reserved")
    _turn(engine, "t-a1", "fixture-amb", created="2026-10-24T10:00:00")
    _turn(engine, "t-a2", "fixture-amb", created="2026-10-24T11:00:00")
    _turn(engine, "t-done", "fixture-idle", status="ok", ended="2026-10-24T09:00:00")
    bots = ["fixture-busy", "fixture-res", "fixture-amb", "fixture-idle"]

    with engine.connect() as conn:
        batched = classify_targets(conn, bots)
        single = {bot: classify_target(conn, bot, exclude_turn_id=None) for bot in bots}

    assert batched == single
    assert [batched[b].kind for b in bots] == ["executing", "reserved", "ambiguous", "idle"]
    assert batched["fixture-amb"].blocking_id() == "t-a2"  # newest first, as before


def test_classify_targets_is_one_statement_and_honours_exclusion(engine):
    _turn(engine, "t-own", "fixture-busy")
    seen: list[str] = []
    event.listen(engine, "before_cursor_execute", lambda *a: seen.append(a[2]))

    with engine.connect() as conn:
        states = classify_targets(conn, BOTS * 3, exclude_turn_id="t-own")

    assert len(seen) == 1
    assert set(states) == set(BOTS) and all(state.idle for state in states.values())


def test_turn_state_comes_from_the_admission_classifier(engine, sources, monkeypatch):
    seen: list[list[str]] = []

    def classify(conn, bot_ids, *, exclude_turn_id=None):
        seen.append(sorted(bot_ids))
        return {
            "fixture-busy": TargetState(({"id": "turn-1", "status": "streaming",
                                          "created_at": "2026-10-24T10:00:00"},)),
            "fixture-idle": TargetState(()),
        }

    monkeypatch.setattr(turn_admission, "classify_targets", classify)

    states = _read(engine)

    assert seen == [sorted(BOTS)]
    assert states["fixture-busy"]["turn"] == {
        "kind": "executing", "turn_id": "turn-1", "started_at": "2026-10-24T10:00:00"}
    assert states["fixture-idle"]["turn"] == {"kind": "idle", "turn_id": None, "started_at": None}
    assert states["fixture-busy"]["queued_deliveries"] == {"count": 2, "oldest_at": None}
    assert states["fixture-idle"]["queued_deliveries"] == {"count": 0, "oldest_at": None}
    assert states["fixture-busy"]["pending_approvals"] == 1
    assert states["fixture-idle"]["pending_approvals"] == 0


def test_every_source_is_called_once_for_all_bots(engine, sources):
    _read(engine)

    assert {name: len(calls) for name, calls in sources.items()} == {
        "queue": 1, "approvals": 1, "gate": 1, "session": 1}
    assert all(calls[0] == sorted(BOTS) for calls in sources.values())


def test_last_turn_portable_query_picks_the_newest_ended_row(engine, sources):
    _turn(engine, "t-1", "fixture-idle", status="error", ended="2026-10-24T08:00:00",
          end_reason="error")
    _turn(engine, "t-2", "fixture-idle", status="ok", ended="2026-10-24T09:00:00",
          end_reason="complete")
    _turn(engine, "t-open", "fixture-idle")

    states = _read(engine)

    assert states["fixture-idle"]["last_turn"] == {
        "ended_at": "2026-10-24T09:00:00", "status": "ok", "end_reason": "complete"}
    assert states["fixture-busy"]["last_turn"] is None


def test_a_failed_turn_source_is_unavailable_never_idle(engine, sources, monkeypatch):
    def broken(conn, bot_ids, *, exclude_turn_id=None):
        raise RuntimeError("turn_logs unreadable")

    monkeypatch.setattr(turn_admission, "classify_targets", broken)

    states = _read(engine)

    for state in states.values():
        assert state["turn"] is None and state["unavailable"] == ["turn"]
    assert states["fixture-busy"]["pending_approvals"] == 1  # other sources still answer


def test_a_failed_gate_source_reports_gated_unknown(engine, sources, monkeypatch):
    def broken(conn, profiles):
        raise RuntimeError("system_gates unreadable")

    monkeypatch.setattr(turn_gates, "covering_gates", broken)

    state = _read(engine)["fixture-idle"]

    assert state["gated"] is None and state["gate"] is None and state["unavailable"] == ["gate"]


def test_a_covering_gate_is_reported(engine, sources, monkeypatch):
    block = GateBlock(id=7, mode="hold", owner_kind="job", owner_id="job-1",
                      reason="restart", expires_at="2026-10-24T12:00:00")
    monkeypatch.setattr(turn_gates, "covering_gates", lambda conn, profiles: {"fixture-busy": block})

    states = _read(engine)

    assert states["fixture-busy"]["gated"] is True
    assert states["fixture-busy"]["gate"] == {"id": 7, "mode": "hold", "owner_kind": "job",
                                              "reason": "restart",
                                              "expires_at": "2026-10-24T12:00:00"}
    assert states["fixture-idle"]["gated"] is False


def test_no_database_reports_everything_unavailable():
    state = BotLiveStateReader(None).read([BotIdentity("fixture-idle", "agent", None)])["fixture-idle"]

    assert state["turn"] is None and state["gated"] is None
    assert state["unavailable"] == sorted(BotLiveStateReader.SOURCES)
