"""DB-free tests for the TASK-1233 gate primitive, refusal type and HTTP edges."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine

from llm_bawt.request_identity import AdmissionUnavailable as IdentityAdmissionUnavailable
from llm_bawt.service.schemas import ChatCompletionRequest, ChatMessage
from llm_bawt.service.turn_admission import AdmissionClaim, SystemGateRefused, TargetBusy
from llm_bawt.service.turn_gates import (
    GateActor,
    GateBlock,
    GateScope,
    GateScopeError,
    GateSpec,
    execution_exit_drains,
    gate_blocks,
    gate_is_live,
    has_execution_evidence,
    initial_control_state,
)
from tests.test_nonstream_turn_execution import _service


def _block(mode="hold", reason="restart-bridge claude-code"):
    return GateBlock(id=7, mode=mode, owner_kind="job", owner_id="job-1", reason=reason,
                     expires_at=datetime(2026, 10, 22, 13, 0, tzinfo=timezone.utc))


def test_gates_are_postgresql_only():
    engine = create_engine("sqlite://")
    with engine.connect() as conn:
        assert gate_blocks(conn, "caid", GateActor("save_turn", "t")) is None
        assert gate_is_live(conn, 1) is False


def test_scope_covers_agent_bots_only():
    for scope in (GateScope.every_agent(), GateScope.on_backends("claude-code"), GateScope.for_bots("mira")):
        # A chat bot (Mira's profile: bot_type chat, no backend) is never covered.
        assert not scope.covers("mira", bot_type="chat", agent_backend=None)
    assert GateScope.every_agent().covers("caid", bot_type="agent", agent_backend="claude-code")
    assert GateScope.on_backends("Codex").covers("x", bot_type=None, agent_backend="codex")
    assert not GateScope.on_backends("codex").covers("caid", bot_type="agent", agent_backend="claude-code")
    assert GateScope.for_bots("Caid").covers("caid", bot_type="agent", agent_backend="claude-code")


@pytest.mark.parametrize("scope", [GateScope.every_agent(), GateScope.on_backends("a", "b"),
                                   GateScope.for_bots("x")])
def test_scope_round_trips_through_json(scope):
    assert GateScope.from_json(scope.to_json()) == scope


def test_scope_and_spec_reject_ambiguity():
    with pytest.raises(GateScopeError):
        GateScope()
    with pytest.raises(GateScopeError):
        GateScope(all_agents=True, bots=frozenset({"x"}))
    base = dict(owner_kind="job", owner_id="j", mode="hold", scope=GateScope.every_agent(), ttl_seconds=60)
    GateSpec(**base)
    for bad in ({"owner_kind": "bot"}, {"owner_id": " "}, {"mode": "pause"}, {"ttl_seconds": 0},
                {"allow_actors": frozenset({"control"})}):
        with pytest.raises(GateScopeError):
            GateSpec(**{**base, **bad})


def test_actor_kinds_are_a_closed_set_with_no_control():
    with pytest.raises(ValueError):
        GateActor("control")
    assert not hasattr(GateActor("steer", "t"), "control")


def test_execution_evidence_is_the_bound_request_id_only():
    assert has_execution_evidence({"agent_request_id": "req-1"})
    assert not has_execution_evidence({"agent_request_id": None, "agent_session_key": "s"})


_EXIT = {"backend": "claude-code", "request_id": "req-1", "boot_id": "b", "source": "a",
         "exited": True, "at": "2026-10-22T15:00:00+00:00"}


@pytest.mark.parametrize(("record", "request_id", "backend", "expected"), [
    (_EXIT, "req-1", "claude-code", (True, "execution_exit:a")),
    ({**_EXIT, "source": "c"}, "req-1", None, (True, "execution_exit:c")),  # member without a backend
    (_EXIT, "req-1", "Claude-Code", (True, "execution_exit:a")),
    (_EXIT, None, "claude-code", (False, "no_request_id")),
    ({**_EXIT, "request_id": None}, None, None, (False, "no_request_id")),  # null never matches null
    (None, "req-1", "claude-code", (False, "no_execution_exit")),
    ("exited", "req-1", "claude-code", (False, "no_execution_exit")),
    (_EXIT, "req-2", "claude-code", (False, "request_mismatch")),
    (_EXIT, "req-1", "codex", (False, "backend_mismatch")),
    ({**_EXIT, "source": "d"}, "req-1", "claude-code", (False, "unknown_exit_source")),
    ({**_EXIT, "exited": False}, "req-1", "claude-code", (False, "exit_unconfirmed")),
    ({**_EXIT, "exited": None}, "req-1", "claude-code", (False, "exit_unconfirmed")),
    ({**_EXIT, "exited": "true"}, "req-1", "claude-code", (False, "exit_unconfirmed")),
])
def test_execution_exit_drains_only_on_a_confirmed_exit_of_this_request(record, request_id, backend, expected):
    assert execution_exit_drains(record, request_id=request_id, backend=backend) == expected


def _member_row(*, turn_present=None, request_id="req-1", member_exit=None, turn_exit=None, disposition="running"):
    return {"id": 3, "turn_id": "t", "disposition": disposition, "backend": "claude-code",
            "agent_request_id": request_id, "member_exit": member_exit, "turn_present": turn_present,
            "turn_request_id": None,
            "request_json": json.dumps({"execution_exit": turn_exit}) if turn_exit else None}


@pytest.mark.parametrize(("row", "expected"), [
    (_member_row(request_id=None), (True, "never_bound")),  # test_Z (c): nothing was ever sent
    (_member_row(), (False, "turn_missing:no_execution_exit")),
    (_member_row(member_exit=_EXIT), (True, "execution_exit:a")),  # test_Z (b): one rule, member home
    (_member_row(member_exit={**_EXIT, "request_id": "req-2"}), (False, "turn_missing:request_mismatch")),
    (_member_row(member_exit={**_EXIT, "exited": False}), (False, "turn_missing:exit_unconfirmed")),
    (_member_row(request_id=None, disposition="revoked_unstarted"), (True, "revoked_unstarted")),
    # While the turn row exists it is the only evidence home.
    (_member_row(turn_present="t", member_exit=_EXIT), (False, "no_execution_exit")),
    (_member_row(turn_present="t", turn_exit=_EXIT), (True, "execution_exit:a")),
])
def test_a_missing_turn_drains_on_member_row_evidence_or_never_bound(row, expected):
    from llm_bawt.service.turn_gates import _drain_status

    status = _drain_status(row)
    assert (status.drained, status.reason) == expected


def test_execution_exit_survives_late_turn_writes_and_drops_on_rearm():
    """update_turn rewrites request_json through request_payload_with_audit: a
    stamped exit must survive that, and a re-arm (new attempt) must drop it."""
    from llm_bawt.service.turn_termination import request_payload_with_audit

    previous = json.dumps({"messages": [], "execution_exit": _EXIT})
    assert json.loads(request_payload_with_audit(previous, {"messages": ["late"]}))["execution_exit"] == _EXIT
    assert "execution_exit" not in json.loads(request_payload_with_audit(previous, {"messages": []}, rearm=True))


def test_only_a_job_gate_steers_and_never_a_revoked_member():
    assert initial_control_state("job", "running") == "pending"
    assert initial_control_state("job", "accepted") == "pending"
    assert initial_control_state("job", "revoked_unstarted") == "not_needed"
    assert initial_control_state("operator", "running") == "not_needed"


@pytest.mark.parametrize(("mode", "code", "status"), [("hold", "system_held", 409),
                                                      ("deny", "system_frozen", 503)])
def test_refusal_is_both_a_busy_target_and_an_http_admission_refusal(mode, code, status):
    refused = SystemGateRefused(_block(mode))
    assert isinstance(refused, TargetBusy)  # durable producers defer, not an attempt
    assert isinstance(refused, IdentityAdmissionUnavailable)  # HTTP edges map status_code
    assert (refused.code, refused.status_code) == (code, status)
    assert str(refused).startswith(f"{code}: restart-bridge claude-code (until 2026-10-22T13:00")


@pytest.mark.parametrize(("mode", "status"), [("hold", 409), ("deny", 503)])
def test_nonstream_route_returns_the_gate_status(monkeypatch, mode, status):
    from llm_bawt.service.routes import chat

    service = Mock(chat_completion=AsyncMock(side_effect=SystemGateRefused(_block(mode))))
    monkeypatch.setattr(chat, "get_service", lambda: service)
    app = FastAPI()
    app.include_router(chat.router)
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json={
            "model": "m", "stream": False, "messages": [{"role": "user", "content": "hi"}]})
    assert response.status_code == status
    assert response.json()["detail"].startswith(SystemGateRefused(_block(mode)).code)


def _stream(monkeypatch, *, claim):
    service = _service(Mock(side_effect=AssertionError("must not execute")))
    service._persist_turn_log = Mock(side_effect=SystemGateRefused(_block()))
    monkeypatch.setattr("llm_bawt.service.dependencies.get_service", lambda: service)
    monkeypatch.setattr("llm_bawt.service.routes.history.maybe_build_session_seed", Mock(return_value=[]))
    monkeypatch.setattr("llm_bawt.service.chat_streaming.prepare_stream_request_attachments",
                        AsyncMock(return_value=("hi", [], [], None)))
    monkeypatch.setattr("llm_bawt.service.chat_streaming.admit_ordinary_agent_turn", lambda *a, **k: None)
    monkeypatch.setattr("llm_bawt.service.chat_streaming.turn_admission_claim", lambda *_: claim)
    request = ChatCompletionRequest(
        bot_id="caid", user="nick", model="test-model", stream=True,
        messages=[ChatMessage(role="user", content="hi")],
        user_message_id="00000000-0000-4000-8000-000000000001",
    )

    async def collect():
        return [event async for event in service.chat_completion_stream(request)]

    return service, collect


def test_direct_stream_start_is_refused_in_band(monkeypatch):
    service, collect = _stream(monkeypatch, claim=None)
    events = asyncio.run(collect())
    assert any("request_failed: system_held" in event for event in events)
    assert events[-1] == "data: [DONE]\n\n"
    service._get_llm_bawt.return_value.execute_llm_query.assert_not_called()


def test_durable_stream_start_reraises_for_the_producer_to_defer(monkeypatch):
    _, collect = _stream(monkeypatch, claim=AdmissionClaim(owner="claim", fence=lambda _: True))
    with pytest.raises(SystemGateRefused):
        asyncio.run(collect())
