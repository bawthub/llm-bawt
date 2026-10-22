"""TASK-1082 slice 1: one backend admission authority for agent turns.

Hermetic tests cover the approval outbox (MCP + harness approve/deny/respond)
against the shared admission contract. ``integration`` tests exercise the real
per-bot lock, classification and reservation fence on PostgreSQL.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from sqlalchemy import text as sa_text
from sqlalchemy.pool import StaticPool
from sqlmodel import create_engine

from agent_bridge.mcp_call_context import canonical_invocation_hash
from llm_bawt.approval_policies import (
    CONT_DELIVERED,
    CONT_PENDING,
    REQ_APPROVED,
    REQ_DENIED,
    REQ_RESPONDED,
    ToolApprovalPolicyStore,
)
from llm_bawt.service import approval_continuations as continuations
from llm_bawt.service.approval_continuations import (
    _continuation_identity,
    dispatch_mcp_result_continuation,
)
from llm_bawt.service.schemas import ChatCompletionRequest, ChatMessage
from tests.test_mcp_approval_continuations import FakeTurnStore as SharedFakeTurnStore
from llm_bawt.service.turn_admission import (
    AdmissionClaim,
    AdmissionFenceLost,
    TargetBusy,
    TargetState,
    TurnAdmission,
    turn_request_extensions,
)


# ---------------------------------------------------------------------------
# Hermetic fixtures
# ---------------------------------------------------------------------------


def _store():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    return ToolApprovalPolicyStore(None, engine=engine)


def _mcp_claim(store, request_id="req-mcp-1"):
    args = {"operation": "demo", "args": {}}
    context = {
        "session_id": "session-1", "turn_id": "turn-1",
        "trigger_message_id": "message-1", "bot_id": "snark", "user_id": "nick",
        "issued_at": 1, "agent_request_id": "agent-request-1",
        "session_key": "snark:nick", "backend": "claude-code", "tool_use_id": "toolu-1",
    }
    row = store.record_mcp_request(
        request_id=request_id, tool_use_id="toolu-1", mcp_server="bawthub",
        bot_id="snark", user_id="nick", turn_id="turn-1",
        trigger_message_id="message-1", session_key="snark:nick",
        backend="claude-code", tool_name="generic", tool_arguments=args,
        subject="subject", grant_key="g", policy_id=None, severity="high",
        prompt="Approve?", invocation_hash=canonical_invocation_hash("generic", args),
        continuation_capable=True, caller_context_json=json.dumps(context),
    )
    store.resolve_request(row.id, status=REQ_APPROVED)
    claimed = store.claim_mcp_execution(row.id)
    store.complete_mcp_execution(
        row.id, result_json='{"answer": 42}', is_error=False,
        claim_token=claimed.execution_claim_token,
    )
    store.enqueue_continuation(row.id)
    return store.claim_continuation(row.id)


def _harness_claim(store, status, request_id="h-1"):
    row = store.record_request(
        request_id=request_id, bot_id="snark", user_id="nick", turn_id="turn-1",
        backend="claude-code", tool_name="Bash", tool_arguments={"command": "x"},
        subject="x", grant_key="g", policy_id=None, severity="high",
        prompt="Approve?", session_id="session-1",
    )
    store.resolve_request(
        row.id, status=status, message="do y instead", continuation_owner="server",
    )
    store.prepare_harness_continuation(row.id)
    return store.claim_continuation(row.id)


class FakeTurnStore(SharedFakeTurnStore):
    """Admission double: busy until ``busy`` is cleared (fenced like the DB)."""

    def __init__(self, busy="target 'snark' is executing (turn-parent)"):
        super().__init__(busy=busy)


class FakeService:
    def __init__(self, store, turn_store, *, save_busy=False):
        self._tool_approval_policy_store = store
        self._turn_log_store = turn_store
        if getattr(turn_store, "engine", None) is None:
            turn_store.engine = store.engine
        self.config = None
        self.requests = []
        self.save_busy = save_busy

    def get_memory_client(self, bot_id, user_id):
        return SimpleNamespace(get_session=lambda session_id: {
            "id": session_id, "bot_id": bot_id, "user_id": user_id,
            "status": "active", "session_metadata": {},
        })

    async def chat_completion_stream(self, request):
        continuations.validate_approval_continuation_claim(
            self, request, request._internal_approval_claim,
        )
        if self.save_busy:
            # The authoritative save_turn decision lost the race.
            raise TargetBusy("target 'snark' is executing (turn-late)")
        self.requests.append(request)
        self._turn_log_store.turn = SimpleNamespace(
            ended_at=1, status="ok", error_text=None,
        )
        yield "data: [DONE]\n\n"


class Grants:
    def __init__(self):
        self.sent = []

    async def send_approval_grant(self, **kwargs):
        self.sent.append(kwargs)


@pytest.fixture
def grants(monkeypatch):
    from llm_bawt.service.routes import approval_policies as routes

    value = Grants()
    monkeypatch.setattr(routes, "_subscriber", lambda: value)
    monkeypatch.setattr(continuations.asyncio, "sleep", _no_sleep(asyncio.sleep))
    return value


def _no_sleep(real_sleep):
    async def fake(seconds):
        # Skip the grant settle delay; keep heartbeat sleeps effectively parked.
        if seconds >= 20:
            await real_sleep(3600)
        await real_sleep(0)
    return fake


def _run(service, store, row):
    asyncio.run(dispatch_mcp_result_continuation(service, store, row))


# ---------------------------------------------------------------------------
# Admission derivation (shared by stream and nonstream)
# ---------------------------------------------------------------------------


def _request(**kwargs):
    return ChatCompletionRequest(
        messages=[ChatMessage(role="user", content="hi")], bot_id="snark", **kwargs,
    )


def test_final_turn_update_keeps_ordinary_retry_digest_and_abort_audit():
    from llm_bawt.service.turn_termination import request_payload_with_audit

    merged = json.loads(request_payload_with_audit(
        json.dumps({"ordinary_request_digest": "abc", "abort_request": {"source": "chat_stop"}}),
        {"messages": [{"role": "user", "content": "sent"}]},
    ))
    assert merged == {
        "messages": [{"role": "user", "content": "sent"}],
        "ordinary_request_digest": "abc",
        "abort_request": {"source": "chat_stop"},
    }


def test_public_requests_cannot_select_idle_only_admission():
    request = _request()
    assert TurnAdmission.for_request(request) is TurnAdmission.START
    assert "admission" not in turn_request_extensions(request)
    # Public JSON cannot set the Python-only approval authority.
    spoofed = ChatCompletionRequest.model_validate({
        "messages": [{"role": "user", "content": "hi"}],
        "_internal_approval_claim": ["req", "token"],
    })
    assert TurnAdmission.for_request(spoofed) is TurnAdmission.START


def test_approval_and_delivery_requests_are_idle_only():
    approval = _request()
    approval._internal_approval_claim = ("req", "token")
    delivery = _request(inter_bot_delivery_id="delivery-1")
    for request in (approval, delivery):
        assert TurnAdmission.for_request(request) is TurnAdmission.IDLE_ONLY
        assert turn_request_extensions(request)["admission"] == "idle_only"
        assert TurnAdmission.from_payload(turn_request_extensions(request)) is TurnAdmission.IDLE_ONLY


def test_target_state_classification_never_guesses():
    running = {"id": "a", "status": "streaming"}
    other = {"id": "b", "status": "pending"}
    reserved = {"id": "r", "status": "reserved"}
    assert TargetState(()).kind == "idle"
    assert TargetState((running,)).executing == running
    assert TargetState((running, other)).kind == "ambiguous"
    assert TargetState((running, other)).executing is None
    assert TargetState((running, reserved)).kind == "reserved"
    assert TargetState((reserved,)).executing is None
    assert TargetState((running,)).admits(TurnAdmission.START)
    assert not TargetState((running,)).admits(TurnAdmission.IDLE_ONLY)
    assert not TargetState((reserved,)).admits(TurnAdmission.START)


@pytest.mark.parametrize("error", [TargetBusy, AdmissionFenceLost])
@pytest.mark.parametrize("stream", [True, False])
def test_persist_turn_log_propagates_admission_outcomes(stream, error):
    from llm_bawt.service.turn_lifecycle import TurnLifecycleMixin

    class Store:
        def save_turn(self, **kwargs):
            raise error("not admitted")

    host = object.__new__(TurnLifecycleMixin)
    host._turn_log_store = Store()
    with pytest.raises(error):
        host._persist_turn_log(
            turn_id="t", request_id="r", path="/v1/chat/completions", stream=stream,
            model="m", bot_id="snark", user_id="nick",
            status="streaming" if stream else "pending", latency_ms=None,
            user_prompt="hi", prepared_messages=[], response_text="",
            request_extensions={"admission": "idle_only"},
        )


@pytest.mark.parametrize("stream", [True, False], ids=["stream", "nonstream"])
@pytest.mark.parametrize("with_claim", [False, True], ids=["ordinary", "claimed"])
def test_persist_turn_log_fails_closed_on_store_error(stream, with_claim):
    from llm_bawt.service.turn_admission import AdmissionUnavailable
    from llm_bawt.service.turn_lifecycle import TurnLifecycleMixin

    class BrokenStore:
        def save_turn(self, **kwargs):
            raise ConnectionError("admission database disconnected")

    host = object.__new__(TurnLifecycleMixin)
    host._turn_log_store = BrokenStore()
    claim = AdmissionClaim(owner="test-owner", fence=lambda conn: True) if with_claim else None
    with pytest.raises(AdmissionUnavailable, match="turn admission failed") as exc:
        host._persist_turn_log(
            turn_id="t", request_id="r", path="/v1/chat/completions", stream=stream,
            model="m", bot_id="snark", user_id="nick",
            status="streaming" if stream else "pending", latency_ms=None,
            user_prompt="hi", prepared_messages=[], response_text="",
            admission_claim=claim,
        )
    assert isinstance(exc.value.__cause__, ConnectionError)


@pytest.mark.parametrize("stream", [True, False], ids=["stream", "nonstream"])
@pytest.mark.parametrize("configured", [False, True], ids=["database-less", "configured"])
@pytest.mark.parametrize("admission", ["start", "idle_only", "claimed"])
def test_missing_engine_only_allows_database_less_ordinary_start(stream, configured, admission):
    from llm_bawt.service.turn_admission import AdmissionUnavailable
    from llm_bawt.service.turn_logs import TurnLogStore

    store = object.__new__(TurnLogStore)
    store.engine = None
    store.database_configured = configured
    claim = AdmissionClaim(owner="test-owner", fence=lambda conn: True) if admission == "claimed" else None
    payload = {"messages": [], "admission": "idle_only" if admission == "idle_only" else "start"}
    kwargs = dict(
        turn_id="turn-test", request_id="r", path="/v1/chat/completions", stream=stream,
        model="m", bot_id="snark", user_id="nick",
        status="streaming" if stream else "pending", latency_ms=None,
        user_prompt="hi", request_payload=payload, response_text="", tool_calls=None,
        admission_claim=claim,
    )
    if not configured and admission == "start":
        assert store.save_turn(**kwargs) is None
    else:
        with pytest.raises(AdmissionUnavailable, match="database unavailable"):
            store.save_turn(**kwargs)


# ---------------------------------------------------------------------------
# Reservation consumption fence: the real save_turn on SQLite (sequential
# prearranged takeovers; PostgreSQL row locking is covered by integration tests)
# ---------------------------------------------------------------------------


@pytest.fixture
def sqlite_turns(monkeypatch):
    from llm_bawt.service.turn_logs import TurnLog, TurnLogStore

    approvals = _store()
    TurnLog.__table__.create(approvals.engine, checkfirst=True)
    turns = object.__new__(TurnLogStore)
    turns.engine = approvals.engine
    monkeypatch.setattr(turns, "_cleanup_expired_if_due", lambda: None)
    monkeypatch.setattr(turns, "_encode_token_usage", lambda *args: None)
    return approvals, turns


def _sqlite_row(turns, turn_id, *, status="reserved", owner=None, ended=False):
    from sqlmodel import Session

    from llm_bawt.service.turn_logs import TurnLog

    payload = {"admission": "idle_only"}
    if owner:
        payload["reservation_owner"] = owner
    with Session(turns.engine) as session:
        session.add(TurnLog(
            id=turn_id, bot_id="snark", user_id="nick", status=status,
            request_json=json.dumps(payload), ended_at=(1 if ended else None),
        ))
        session.commit()


def _consume(turns, turn_id, claim, *, admission="idle_only", stream=True):
    turns.save_turn(
        turn_id=turn_id, request_id="r", path="/v1/chat/completions", stream=stream,
        model="m", bot_id="snark", user_id="nick",
        status="streaming" if stream else "pending",
        latency_ms=None, user_prompt="result",
        request_payload={"messages": [], "admission": admission},
        response_text="", tool_calls=None, admission_claim=claim,
    )


def _claimed_request(approvals, row):
    from llm_bawt.service.inter_bot_claims import turn_admission_claim

    request = _request()
    request._internal_approval_claim = (row.id, row.continuation_claim_token)
    return turn_admission_claim(FakeService(approvals, FakeTurnStore(busy=None)), request)


def test_owner_with_live_claim_consumes_its_reservation(sqlite_turns):
    approvals, turns = sqlite_turns
    row = _mcp_claim(approvals)
    claim = _claimed_request(approvals, row)
    _sqlite_row(turns, "turn-c", owner=claim.owner)
    _consume(turns, "turn-c", claim)
    turn = turns.get_turn("turn-c")
    assert turn.status == "streaming" and turn.ended_at is None


def test_consuming_a_reservation_keeps_its_audits(sqlite_turns):
    """TASK-1232 r2-M1: save_turn's reserved UPDATE is the second re-arm write."""
    approvals, turns = sqlite_turns
    row = _mcp_claim(approvals)
    claim = _claimed_request(approvals, row)
    _sqlite_row(turns, "turn-c", owner=claim.owner)
    audits = {"recovery": {"cause": "orphaned_by_restart"}, "abort_request": {"source": "chat_stop"},
              "attempts": [{"n": 1, "status": "error", "error": "429"}]}  # TASK-1234 Fix 2
    with turns.engine.begin() as conn:
        conn.execute(sa_text("UPDATE turn_logs SET request_json=:j WHERE id='turn-c'"), {"j": json.dumps({
            "admission": "idle_only", "reservation_owner": claim.owner, **audits,
            "abort_cause": {"cause": "host_shutdown"},
        })})
    _consume(turns, "turn-c", claim)
    payload = json.loads(turns.get_turn("turn-c").request_json)
    assert payload["messages"] == [] and {key: payload[key] for key in audits} == audits
    assert "abort_cause" not in payload


@pytest.mark.parametrize("successor_reserved", [False, True])
def test_superseded_claim_cannot_consume_any_reservation(sqlite_turns, successor_reserved):
    approvals, turns = sqlite_turns
    old = _mcp_claim(approvals)
    stale = _claimed_request(approvals, old)
    new = approvals.claim_continuation(old.id, lease_seconds=0)
    # Either the old worker's own reservation is still there (lease reclaimed
    # but not yet re-reserved) or the successor has re-armed the same id.
    owner = new.continuation_claim_token if successor_reserved else stale.owner
    _sqlite_row(turns, "turn-c", owner=owner)
    with pytest.raises(AdmissionFenceLost):
        _consume(turns, "turn-c", stale)
    turn = turns.get_turn("turn-c")
    assert turn.status == "reserved"
    assert json.loads(turn.request_json)["reservation_owner"] == owner


def test_superseded_claim_cannot_insert_a_fresh_turn(sqlite_turns):
    approvals, turns = sqlite_turns
    old = _mcp_claim(approvals)
    stale = _claimed_request(approvals, old)
    approvals.claim_continuation(old.id, lease_seconds=0)
    with pytest.raises(AdmissionFenceLost):
        _consume(turns, "turn-c", stale)
    assert turns.get_turn("turn-c") is None


def test_superseded_claim_on_busy_target_leaves_successor_reservation(sqlite_turns):
    approvals, turns = sqlite_turns
    old = _mcp_claim(approvals)
    stale = _claimed_request(approvals, old)
    new = approvals.claim_continuation(old.id, lease_seconds=0)
    _sqlite_row(turns, "turn-c", owner=new.continuation_claim_token)
    _sqlite_row(turns, "turn-running", status="streaming")
    with pytest.raises(AdmissionFenceLost):
        _consume(turns, "turn-c", stale)
    assert turns.get_turn("turn-c").status == "reserved"


def test_live_claim_on_busy_target_releases_only_its_own_reservation(sqlite_turns):
    approvals, turns = sqlite_turns
    claim = _claimed_request(approvals, _mcp_claim(approvals))
    _sqlite_row(turns, "turn-c", owner=claim.owner)
    _sqlite_row(turns, "turn-running", status="streaming")
    with pytest.raises(TargetBusy):
        _consume(turns, "turn-c", claim)
    assert turns.get_turn("turn-c") is None
    assert turns.get_turn("turn-running").status == "streaming"


def test_owned_reservation_needs_its_claim_and_unowned_rejects_one(sqlite_turns):
    approvals, turns = sqlite_turns
    claim = _claimed_request(approvals, _mcp_claim(approvals))
    # Public/ordinary ingress (no claim) cannot consume a fenced reservation.
    _sqlite_row(turns, "turn-owned", owner=claim.owner)
    with pytest.raises(AdmissionFenceLost):
        _consume(turns, "turn-owned", None, admission="start")
    assert turns.get_turn("turn-owned").status == "reserved"
    with turns.engine.begin() as conn:  # clear it so the target is otherwise idle
        conn.execute(sa_text("DELETE FROM turn_logs WHERE id='turn-owned'"))
    # A durable-delivery reservation carries no owner: consumed without a
    # claim (unchanged), never by an approval claim.
    _sqlite_row(turns, "turn-delivery")
    with pytest.raises(AdmissionFenceLost):
        _consume(turns, "turn-delivery", claim)
    assert turns.get_turn("turn-delivery").status == "reserved"
    _consume(turns, "turn-delivery", None)
    assert turns.get_turn("turn-delivery").status == "streaming"


def _sqlite_delivery_claim(turns):
    from llm_bawt.service.inter_bot_claims import turn_admission_claim

    delivery_id = "delivery-test"
    turn_id = "turn-delivery-test"
    payload = {"inter_bot_delivery_id": delivery_id, "admission": "idle_only"}
    with turns.engine.begin() as conn:
        conn.execute(sa_text("""
            CREATE TABLE inter_bot_deliveries (
                id TEXT, status TEXT, claim_token TEXT, target_bot_id TEXT,
                turn_id TEXT, user_message_id TEXT, payload_json TEXT
            )
        """))
        conn.execute(sa_text("""
            INSERT INTO inter_bot_deliveries
                (id, status, claim_token, target_bot_id, turn_id, user_message_id, payload_json)
            VALUES (:id, 'DISPATCHING', 'old-token', 'snark', :turn, 'msg-1', :payload)
        """), {
            "id": delivery_id, "turn": turn_id,
            "payload": json.dumps({"inter_bot_bridge_request_id": "req-1"}),
        })
    _sqlite_row(turns, turn_id)  # claim_next's unowned reservation
    with turns.engine.begin() as conn:
        conn.execute(sa_text("UPDATE turn_logs SET request_json=:payload WHERE id=:id"),
                     {"payload": json.dumps(payload), "id": turn_id})

    request = SimpleNamespace(
        inter_bot_delivery_id=delivery_id, inter_bot_claim_token="old-token",
        bot_id="snark", inter_bot_turn_id=turn_id, user_message_id="msg-1",
        inter_bot_bridge_request_id="req-1",
    )
    service = SimpleNamespace(
        _inter_bot_dispatcher=SimpleNamespace(store=SimpleNamespace(engine=turns.engine)),
    )
    return request, service, turn_admission_claim(service, request)


@pytest.mark.parametrize("stream", [True, False], ids=["stream", "nonstream"])
def test_valid_delivery_claim_cannot_consume_approval_owned_reservation(sqlite_turns, stream):
    _, turns = sqlite_turns
    request, _, claim = _sqlite_delivery_claim(turns)
    turn_id = request.inter_bot_turn_id
    owned_payload = {
        "admission": "idle_only", "inter_bot_delivery_id": request.inter_bot_delivery_id,
        "reservation_owner": "approval-owner",
    }
    with turns.engine.begin() as conn:
        conn.execute(sa_text("UPDATE turn_logs SET request_json=:payload WHERE id=:id"),
                     {"payload": json.dumps(owned_payload), "id": turn_id})
        assert claim.fence(conn)  # valid delivery authority; only ownership must reject it

    with pytest.raises(AdmissionFenceLost, match="another claim"):
        _consume(turns, turn_id, claim, stream=stream)
    turn = turns.get_turn(turn_id)
    assert turn.status == "reserved" and turn.ended_at is None
    assert json.loads(turn.request_json) == owned_payload


@pytest.mark.parametrize("stream", [True, False], ids=["stream", "nonstream"])
def test_delivery_claim_is_fenced_at_consumption(sqlite_turns, stream):
    from llm_bawt.service.inter_bot_claims import turn_admission_claim

    _, turns = sqlite_turns
    request, service, old_claim = _sqlite_delivery_claim(turns)
    delivery_id = request.inter_bot_delivery_id
    turn_id = request.inter_bot_turn_id
    payload = {"inter_bot_delivery_id": delivery_id, "admission": "idle_only"}
    # The claim looked valid at entry; reclaim happened before save_turn.
    with turns.engine.begin() as conn:
        conn.execute(sa_text("UPDATE inter_bot_deliveries SET claim_token='new-token' WHERE id=:id"),
                     {"id": delivery_id})
    with pytest.raises(AdmissionFenceLost):
        _consume(turns, turn_id, old_claim, stream=stream)
    with pytest.raises(AdmissionFenceLost):
        _consume(turns, turn_id, None, stream=stream)  # ordinary caller cannot steal it
    assert turns.get_turn(turn_id).status == "reserved"

    request.inter_bot_claim_token = "new-token"
    new_claim = turn_admission_claim(service, request)
    # Entry can validate a steer, but only an exact DISPATCHING claim may
    # consume this fallback reservation. Every identity component is fenced.
    with turns.engine.begin() as conn:
        from llm_bawt.inter_bot_claim_validation import validate_delivery_claim

        conn.execute(sa_text("UPDATE inter_bot_deliveries SET status='STEERING' WHERE id=:id"),
                     {"id": delivery_id})
        assert validate_delivery_claim(
            conn, delivery_id=delivery_id, claim_token="new-token",
            target_bot_id="snark", turn_id=turn_id,
            user_message_id="msg-1", bridge_request_id="req-1",
        )  # entry accepts a steer
    with pytest.raises(AdmissionFenceLost):
        _consume(turns, turn_id, new_claim, stream=stream)
    with turns.engine.begin() as conn:
        conn.execute(sa_text("UPDATE inter_bot_deliveries SET status='DISPATCHING' WHERE id=:id"),
                     {"id": delivery_id})
    for field in ("inter_bot_delivery_id", "bot_id", "inter_bot_turn_id",
                  "user_message_id", "inter_bot_bridge_request_id"):
        original = getattr(request, field)
        setattr(request, field, "wrong-id")
        with pytest.raises(AdmissionFenceLost, match="no longer held"):
            _consume(turns, turn_id, turn_admission_claim(service, request), stream=stream)
        setattr(request, field, original)
    assert turns.get_turn(turn_id).status == "reserved"
    with turns.engine.begin() as conn:
        conn.execute(sa_text("UPDATE turn_logs SET ended_at=CURRENT_TIMESTAMP WHERE id=:id"),
                     {"id": turn_id})
    with pytest.raises(AdmissionFenceLost, match="no longer open"):
        _consume(turns, turn_id, new_claim, stream=stream)
    with turns.engine.begin() as conn:
        conn.execute(sa_text("DELETE FROM turn_logs WHERE id=:id"), {"id": turn_id})
    with pytest.raises(AdmissionFenceLost, match="disappeared"):
        _consume(turns, turn_id, new_claim, stream=stream)
    _sqlite_row(turns, turn_id)
    with turns.engine.begin() as conn:
        conn.execute(sa_text("UPDATE turn_logs SET request_json=:payload WHERE id=:id"),
                     {"payload": json.dumps({**payload, "inter_bot_delivery_id": "other-delivery"}),
                      "id": turn_id})
    with pytest.raises(AdmissionFenceLost, match="another claim"):
        _consume(turns, turn_id, new_claim, stream=stream)
    with turns.engine.begin() as conn:
        conn.execute(sa_text("UPDATE turn_logs SET request_json=:payload WHERE id=:id"),
                     {"payload": json.dumps(payload), "id": turn_id})

    _consume(turns, turn_id, new_claim, stream=stream)
    assert turns.get_turn(turn_id).status == ("streaming" if stream else "pending")


def test_reserved_turns_are_not_executing_steer_targets(sqlite_turns):
    _, turns = sqlite_turns
    _sqlite_row(turns, "turn-reserved", owner="approval-owner")
    rows, count = turns.list_turns(bot_id="snark", user_id="nick", active_only=True)
    assert rows == [] and count == 0
    assert turns.active_turn_for_bot("snark") is None
    # Direct inspection still exposes the reservation for debugging.
    rows, count = turns.list_turns(bot_id="snark", status="reserved")
    assert count == 1 and rows[0].id == "turn-reserved"

    _sqlite_row(turns, "turn-running", status="streaming")
    rows, count = turns.list_turns(bot_id="snark", active_only=True)
    assert count == 1 and [row.id for row in rows] == ["turn-running"]
    assert turns.active_turn_for_bot("snark").id == "turn-running"


def test_in_turn_readout_and_admission_share_one_unbounded_predicate(sqlite_turns, monkeypatch):
    """TASK-1231C: an orphan older than the old 1800s window still blocks
    admission, so the in-turn readout must report it instead of aging it out."""
    from datetime import datetime, timedelta, timezone

    from llm_bawt.service.routes import turn_logs as turn_log_routes

    _, turns = sqlite_turns
    _sqlite_row(turns, "turn-orphan", status="streaming")
    with turns.engine.begin() as conn:
        conn.execute(sa_text("UPDATE turn_logs SET created_at=:old WHERE id='turn-orphan'"),
                     {"old": datetime.now(timezone.utc) - timedelta(hours=2)})
    assert not turns.peek_target_state("snark").idle
    assert turns.active_turn_for_bot("snark").id == "turn-orphan"
    monkeypatch.setattr(turn_log_routes, "get_turn_log_store", lambda: turns)
    readout = turn_log_routes.bot_in_turn("snark")
    assert readout["in_turn"] is True and readout["turn_id"] == "turn-orphan"


def test_admission_claim_fails_closed_without_a_fence_capable_store():
    from llm_bawt.service.inter_bot_claims import turn_admission_claim

    request = _request()
    assert turn_admission_claim(SimpleNamespace(), request) is None
    request._internal_approval_claim = ("req", "token")
    with pytest.raises(RuntimeError, match="refusing unfenced"):
        turn_admission_claim(SimpleNamespace(_tool_approval_policy_store=object()), request)
    delivery = SimpleNamespace(inter_bot_delivery_id="delivery-1")
    with pytest.raises(RuntimeError, match="refusing unfenced"):
        turn_admission_claim(SimpleNamespace(), delivery)


# ---------------------------------------------------------------------------
# Approval outbox through the shared admission
# ---------------------------------------------------------------------------


def test_busy_target_defers_mcp_continuation_without_starting_a_turn(grants):
    store = _store()
    row = _mcp_claim(store)
    attempts = row.continuation_attempts
    turn_store = FakeTurnStore()
    service = FakeService(store, turn_store)

    _run(service, store, row)

    assert service.requests == []
    assert turn_store.reservations == []
    saved = store.get_request(row.id)
    assert saved.continuation_state == CONT_PENDING
    assert saved.continuation_claim_token is None
    assert saved.continuation_attempts == attempts - 1  # waiting is not an attempt
    assert "waiting for idle target" in saved.continuation_last_error
    assert "turn-parent" in saved.continuation_last_error


@pytest.mark.parametrize("status", [REQ_APPROVED, REQ_DENIED, REQ_RESPONDED])
def test_busy_target_sends_no_harness_grant(grants, status):
    store = _store()
    row = _harness_claim(store, status)
    service = FakeService(store, FakeTurnStore())

    _run(service, store, row)

    assert grants.sent == []
    assert service.requests == []
    saved = store.get_request(row.id)
    assert saved.continuation_state == CONT_PENDING
    assert saved.grant_state == "pending"


@pytest.mark.parametrize("status", [REQ_APPROVED, REQ_DENIED, REQ_RESPONDED])
def test_idle_target_reserves_before_grant_then_dispatches_once(grants, status):
    store = _store()
    row = _harness_claim(store, status)
    turn_store = FakeTurnStore(busy=None)
    order = []
    original_reserve = turn_store.reserve_idle_turn

    def reserve(**kwargs):
        order.append("reserve")
        return original_reserve(**kwargs)

    async def grant(**kwargs):
        order.append("grant")
        grants.sent.append(kwargs)

    turn_store.reserve_idle_turn = reserve
    grants.send_approval_grant = grant
    service = FakeService(store, turn_store)

    _run(service, store, row)

    identity = _continuation_identity(row)
    assert len(service.requests) == 1
    assert turn_store.reservations[0]["turn_id"] == identity["inter_bot_turn_id"]
    assert turn_store.reservations[0]["trigger_message_id"] == identity["user_message_id"]
    if status == REQ_APPROVED:
        assert order == ["reserve", "grant"]
        assert grants.sent[0]["request_id"] == row.id
    else:
        assert order == ["reserve"] and grants.sent == []
    assert store.get_request(row.id).continuation_state == CONT_DELIVERED


def test_losing_authoritative_save_defers_and_releases_own_reservation(grants):
    store = _store()
    row = _mcp_claim(store)
    turn_store = FakeTurnStore(busy=None)
    service = FakeService(store, turn_store, save_busy=True)

    _run(service, store, row)

    identity = _continuation_identity(row)
    assert turn_store.released == [identity["inter_bot_turn_id"]]
    saved = store.get_request(row.id)
    assert saved.continuation_state == CONT_PENDING
    assert "turn-late" in saved.continuation_last_error


def test_deferred_continuation_runs_exactly_once_after_target_goes_idle(grants):
    from datetime import datetime, timedelta, timezone

    from sqlmodel import Session

    from llm_bawt.approval_policies import ToolApprovalRequest

    store = _store()
    row = _harness_claim(store, REQ_APPROVED)
    turn_store = FakeTurnStore()
    service = FakeService(store, turn_store)

    _run(service, store, row)  # busy -> deferred, no grant
    assert grants.sent == [] and service.requests == []

    with Session(store.engine) as session:
        saved = session.get(ToolApprovalRequest, row.id)
        saved.continuation_next_attempt_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        session.add(saved)
        session.commit()
    turn_store.busy = None
    retried = store.claim_continuation(row.id)
    _run(service, store, retried)

    assert len(grants.sent) == 1
    assert len(service.requests) == 1
    request = service.requests[0]
    identity = _continuation_identity(row)
    assert request.inter_bot_turn_id == identity["inter_bot_turn_id"]
    assert request.inter_bot_bridge_request_id == identity["inter_bot_bridge_request_id"]
    assert store.get_request(row.id).continuation_state == CONT_DELIVERED

    # A duplicate terminal pass is an ack-only no-op: no second turn or grant.
    assert store.claim_continuation(row.id) is None


def test_defer_is_fenced_by_claim_token():
    store = _store()
    row = _mcp_claim(store)
    assert not store.defer_continuation_claim(
        row.id, claim_token="stale", reason="busy", delay_seconds=1,
    )
    assert store.get_request(row.id).continuation_claim_token == row.continuation_claim_token


def test_bare_reservation_may_still_follow_new(grants):
    """A reserved-only prior row wrote no messages; /new redirect stays legal."""
    store = _store()
    row = _mcp_claim(store)
    turn_store = FakeTurnStore(busy=None)
    turn_store.turn = SimpleNamespace(ended_at=None, status="reserved", error_text=None)
    service = FakeService(store, turn_store)
    service.get_memory_client = lambda bot_id, user_id: SimpleNamespace(get_session=lambda sid: {
        "session-1": {"id": "session-1", "bot_id": bot_id, "user_id": user_id,
                      "status": "archived",
                      "session_metadata": {"reset_successor_id": "session-2"}},
        "session-2": {"id": "session-2", "bot_id": bot_id, "user_id": user_id,
                      "status": "active",
                      "session_metadata": {"reset_predecessor_ids": ["session-1"]}},
    }[sid])

    _run(service, store, row)

    assert service.requests[0].session_id == "session-2"
    assert turn_store.reservations[0]["request_json"]["session_id"] == "session-2"


class PeekingTurnStore(FakeTurnStore):
    """Turn store exposing the advisory unlocked peek, like ``TurnLogStore``."""

    def __init__(self, peek_rows, busy=None):
        super().__init__(busy=busy)
        self.peek_rows = peek_rows
        self.peeks = []

    def peek_target_state(self, bot_id, *, exclude_turn_id=None):
        from llm_bawt.service.turn_admission import TargetState

        self.peeks.append((bot_id, exclude_turn_id))
        return TargetState(tuple(self.peek_rows))


def _counting_sessions(service):
    reads = []

    def get_memory_client(bot_id, user_id):
        def get_session(session_id):
            reads.append(session_id)
            return {"id": session_id, "bot_id": bot_id, "user_id": user_id,
                    "status": "active", "session_metadata": {}}
        return SimpleNamespace(get_session=get_session)

    service.get_memory_client = get_memory_client
    return reads


@pytest.mark.parametrize("make_row", [_mcp_claim, lambda s: _harness_claim(s, REQ_APPROVED)])
def test_busy_peek_defers_before_any_session_lineage_read(grants, make_row):
    """TASK-1236: a waiting continuation must not hammer sessions_get every poll."""
    store = _store()
    row = make_row(store)
    turn_store = PeekingTurnStore([{"id": "turn-parent", "status": "streaming"}])
    service = FakeService(store, turn_store)
    reads = _counting_sessions(service)

    _run(service, store, row)

    assert reads == []
    assert turn_store.peeks == [("snark", _continuation_identity(row)["inter_bot_turn_id"])]
    assert turn_store.reservations == [] and service.requests == [] and grants.sent == []
    saved = store.get_request(row.id)
    assert saved.continuation_state == CONT_PENDING
    assert saved.continuation_claim_token is None
    assert "waiting for idle target" in saved.continuation_last_error
    assert "executing (turn-parent)" in saved.continuation_last_error


def test_idle_peek_still_defers_to_locked_reservation_authority(grants):
    """The peek is advisory: a target that turns busy before the lock still defers."""
    store = _store()
    row = _mcp_claim(store)
    turn_store = PeekingTurnStore([], busy="target 'snark' is executing (turn-raced)")
    service = FakeService(store, turn_store)
    reads = _counting_sessions(service)

    _run(service, store, row)

    assert reads  # idle-looking target proceeds to resolve its delivery session
    assert turn_store.reservations == [] and service.requests == []
    saved = store.get_request(row.id)
    assert saved.continuation_state == CONT_PENDING
    assert "turn-raced" in saved.continuation_last_error


def test_turn_log_store_peek_reports_open_rows_excluding_own_turn(sqlite_turns):
    _approvals, turns = sqlite_turns
    assert turns.peek_target_state("snark", exclude_turn_id="turn-c").idle
    _sqlite_row(turns, "turn-c", owner="me")
    assert turns.peek_target_state("snark", exclude_turn_id="turn-c").idle
    _sqlite_row(turns, "turn-running", status="cancelling")
    state = turns.peek_target_state("snark", exclude_turn_id="turn-c")
    assert not state.idle and state.describe() == "executing (turn-running)"


def test_long_wait_backs_off_instead_of_hot_looping():
    from datetime import datetime, timedelta, timezone

    now = datetime.now(timezone.utc)
    fresh = SimpleNamespace(execution_finished_at=now, resolved_at=now)
    long = SimpleNamespace(execution_finished_at=now - timedelta(hours=2), resolved_at=None)
    assert continuations._admission_wait_delay(fresh) == 2.0
    assert continuations._admission_wait_delay(long) == continuations.ADMISSION_WAIT_WARN_S
