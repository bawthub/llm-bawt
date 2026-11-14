"""Fenced recovery composition without inference, live UI or shared services."""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel

from agent_bridge.events import AgentEvent, AgentEventKind
from agent_bridge.run_contract import RunTerminal, TeardownReport
from agent_bridge.terminal_replay import ReplayDisposition, RetainedRun
from llm_bawt.approval_models import ToolApprovalRequest
from llm_bawt.service.chat_pending_questions import PendingQuestion
from llm_bawt.service.tool_call_store import ToolCallRecord
from llm_bawt.service.turn_log_schema import TurnLog
from llm_bawt.service.turn_terminal_evidence import load_recovery_evidence, recovery_candidate
from llm_bawt.service.turn_terminal_observe import RunObservation
from llm_bawt.service.turn_terminal_protocol import FollowerFenceLost, TurnTerminalProtocol
from llm_bawt.service import turn_terminal_recovery as mod
from tests.test_turn_terminal_protocol import insert_turn, read


@pytest.fixture
def harness(monkeypatch):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(engine, tables=[TurnLog.__table__, ToolCallRecord.__table__,
                                               PendingQuestion.__table__, ToolApprovalRequest.__table__])
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE sessions (id TEXT PRIMARY KEY, bot_id TEXT, user_id TEXT, status TEXT)"))
    effects = Mock()
    monkeypatch.setattr("llm_bawt.service.turn_terminal_effects.record_terminal_effects", effects)
    monkeypatch.setattr("llm_bawt.service.usage.balance.balance_update_for_turn", lambda *a, **kw: None)
    monkeypatch.setattr(mod, "reconcile_finished_tool_safely", Mock())
    store = SimpleNamespace(engine=engine, _encode_token_usage=lambda usage, model: json.dumps(usage),
                            changed_files_summary=Mock(return_value=[]))
    protocol = TurnTerminalProtocol(store)
    turn = insert_turn(engine, response_text="durable partial", reasoning="durable reasoning",
                       agent_request_id="request", agent_session_key="fixture:user", agent_backend="claude-code")
    terminal = RunTerminal("request", "claude-code", "boot", TeardownReport(exited=True))
    done = AgentEvent("aggregate", "fixture:user", "run", AgentEventKind.ASSISTANT_DONE,
                      "system", text="full answer", token_usage={"output_tokens": 5})
    replay = RetainedRun(ReplayDisposition.COMPLETE, "retained_terminal_run", (done,), terminal)
    observer = AsyncMock(return_value=RunObservation(replay, None))
    monkeypatch.setattr(mod, "observe_bound_run", observer)
    redis = SimpleNamespace(hgetall=AsyncMock(return_value={}))
    subscriber = SimpleNamespace(_redis=redis, _pub_redis=object())
    projection = AsyncMock()
    monkeypatch.setattr(mod, "project_committed_terminal", projection)
    abort = AsyncMock(return_value=None)
    status = AsyncMock(return_value=(None, None))
    monkeypatch.setattr(mod, "BridgeRunProbe", lambda factory, **kwargs: SimpleNamespace(abort=abort, status=status))
    monkeypatch.setattr(mod.TurnExecution, "record_exit", Mock(return_value=True))
    yield SimpleNamespace(engine=engine, protocol=protocol, turn=turn, observer=observer, redis=redis,
                          subscriber=subscriber, projection=projection, abort=abort, terminal=terminal,
                          effects=effects, done=done, status=status)
    engine.dispose()


def run(h):
    return asyncio.run(mod.recover_terminal_turn(h.protocol, h.turn, h.subscriber))


def test_complete_recovery_commits_before_projection_without_rerunning_effects(harness):
    h = harness
    async def project(*args, **kwargs):
        row = read(h.engine, h.turn)
        assert row.terminal_protocol_state == "pending"
        assert json.loads(row.terminal_snapshot_json)["response_text"] == "full answer"
    h.projection.side_effect = project
    result = run(h)
    assert result.won and result.snapshot["completion"]["status"] == "completed"
    assert result.snapshot["completion"]["token_usage"] == {"output_tokens": 5}
    h.abort.assert_not_awaited()
    h.effects.assert_called_once()
    h.projection.assert_awaited_once()


def test_missing_stream_requires_bounded_abort_and_preserves_durable_partial(harness):
    h = harness
    h.observer.return_value = RunObservation(RetainedRun(ReplayDisposition.MISSING, "run_stream_missing"),
                                             "run_stream_missing")
    result = run(h)
    h.abort.assert_awaited_once_with(backend="claude-code", session_key="fixture:user", request_id="request")
    assert result.snapshot["response_text"] == "durable partial"
    assert result.snapshot["reasoning"] == "durable reasoning"
    assert result.snapshot["completion"]["status"] == "error"
    payload = json.loads(read(h.engine, h.turn).request_json)
    assert payload["recovery"]["execution_exit_confirmed"] is False
    assert "run_stream_missing" in result.snapshot["completion"]["error_text"]


def test_retained_exit_proof_avoids_abort_but_never_promotes_missing_text(harness):
    h = harness
    h.observer.return_value = RunObservation(None, "run_stream_missing")
    h.redis.hgetall.return_value = h.terminal.to_fields()
    result = run(h)
    h.abort.assert_not_awaited()
    assert result.snapshot["completion"]["status"] == "error"
    assert result.snapshot["response_text"] == "durable partial"


def test_foreign_terminal_record_is_not_exit_evidence(harness):
    h = harness
    h.observer.return_value = RunObservation(None, "run_stream_missing")
    h.redis.hgetall.return_value = RunTerminal("foreign", "claude-code", "boot", TeardownReport(exited=True)).to_fields()
    run(h)
    h.abort.assert_awaited_once()


def test_complete_retention_without_aggregate_falls_back_not_delta_success(harness):
    h = harness
    h.observer.return_value = RunObservation(RetainedRun(ReplayDisposition.COMPLETE, "complete", (), h.terminal), None)
    result = run(h)
    assert result.snapshot["response_text"] == "durable partial"
    assert result.snapshot["completion"]["status"] == "error"
    assert "aggregate_unavailable" in result.snapshot["completion"]["error_text"]


def test_stop_during_observation_keeps_winner_and_denies_recovery_commit(harness):
    h = harness
    async def stop(*args, **kwargs):
        h.protocol.interrupt(h.turn, status="aborted", end_reason="aborted")
        return RunObservation(None, "run_stream_missing")
    h.observer.side_effect = stop
    with pytest.raises(FollowerFenceLost):
        run(h)
    assert read(h.engine, h.turn).status == "aborted"
    h.effects.assert_called_once()
    h.projection.assert_not_awaited()  # next projection pass owns that winner


def test_committed_retry_only_projects_and_never_observes(harness):
    h = harness
    h.protocol.interrupt(h.turn, status="error", end_reason="error")
    run(h)
    h.observer.assert_not_awaited()
    h.abort.assert_not_awaited()
    h.projection.assert_awaited_once()
    h.effects.assert_called_once()


def test_healthy_owner_is_not_stolen(harness):
    h = harness
    lease = h.protocol.claim(h.turn)
    run(h)
    assert read(h.engine, h.turn).follower_owner == lease.token
    h.observer.assert_not_awaited()
    h.effects.assert_not_called()


def test_cancelled_aggregate_persists_restart_cause(harness):
    h = harness
    h.done.raw = {"end_reason": "aborted", "abort_cause": "host_shutdown"}
    result = run(h)
    assert result.snapshot["completion"]["status"] == "cancelled"
    assert json.loads(read(h.engine, h.turn).request_json)["abort_cause"]["cause"] == "host_shutdown"


def test_unbound_loss_has_no_history_content_or_bridge_observation(harness):
    h = harness
    h.turn = insert_turn(h.engine, unbound=True)
    result = run(h)
    assert result.snapshot["history_disposition"] == "unbound"
    assert result.snapshot["response_text"] == ""
    h.observer.assert_not_awaited()
    h.abort.assert_not_awaited()


@pytest.mark.parametrize("state", ["running", "queued", "preflight"])
def test_pending_but_confirmed_live_run_is_left_for_next_pass(harness, state):
    h = harness
    h.observer.return_value = RunObservation(RetainedRun(ReplayDisposition.PENDING, "pending"),
                                             "recovery_run_not_terminal")
    h.status.return_value = (SimpleNamespace(request_id="request", backend="claude-code",
        boot_id="boot", state=SimpleNamespace(value=state)), None)
    assert run(h) is None
    row = read(h.engine, h.turn)
    assert row.ended_at is None and row.terminal_snapshot_json is None
    h.abort.assert_not_awaited()
    h.projection.assert_not_awaited()
    h.effects.assert_not_called()


def test_pending_without_positive_liveness_uses_bounded_fallback(harness):
    h = harness
    h.observer.return_value = RunObservation(RetainedRun(ReplayDisposition.PENDING, "pending"),
                                             "recovery_run_not_terminal")
    result = run(h)
    h.status.assert_awaited_once()
    h.abort.assert_awaited_once()
    assert result.snapshot["completion"]["status"] == "error"


def test_durable_tool_result_is_recovered_into_snapshot(harness):
    h = harness
    row = read(h.engine, h.turn)
    with Session(h.engine) as session:
        session.add(ToolCallRecord(turn_id=h.turn, bot_id=row.bot_id, user_id=row.user_id,
                                   call_id="call", tool_name="Read", result_text="durable result", result_complete=True))
        session.commit()
    result = run(h)
    assert result.snapshot["tool_context"] == "[Read]\ndurable result"


def test_missing_interaction_receipt_does_not_replay_it(harness):
    h = harness
    interaction = AgentEvent("ask", "fixture:user", "run", AgentEventKind.AWAIT_TOOL_RESULT,
                             "system", tool_use_id="missing")
    h.observer.return_value = RunObservation(RetainedRun(ReplayDisposition.COMPLETE, "complete",
        (interaction, h.done), h.terminal), None)
    result = run(h)
    assert result.snapshot["completion"]["status"] == "error"
    assert "interaction_receipt_missing" in result.snapshot["completion"]["error_text"]


def test_evidence_loader_rejects_expired_owner(harness):
    h = harness
    lease = h.protocol.claim(h.turn)
    from tests.test_turn_terminal_protocol import expire
    expire(h.engine, h.turn)
    with pytest.raises(FollowerFenceLost):
        load_recovery_evidence(h.protocol, lease)


def test_candidate_without_replay_never_discards_persisted_partial(harness):
    h = harness
    lease = h.protocol.claim(h.turn)
    evidence = load_recovery_evidence(h.protocol, lease)
    candidate = recovery_candidate(evidence, None, incomplete_reason="transport_unavailable")
    assert candidate["response_text"] == "durable partial"
    assert candidate["status"] == "error" and candidate["end_reason"] == "error"
