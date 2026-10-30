"""Independent TASK-1082 admission review; SQLite/fakes only, no live services."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from tests import test_turn_admission as base
from llm_bawt.approval_policies import REQ_APPROVED
from llm_bawt.service import approval_continuations as continuations


@pytest.fixture
def grants(monkeypatch):
    from llm_bawt.service.routes import approval_policies as routes

    result = base.Grants()
    monkeypatch.setattr(routes, "_subscriber", lambda: result)
    monkeypatch.setattr(continuations.asyncio, "sleep", base._no_sleep(asyncio.sleep))
    return result


def test_unavailable_admission_must_not_send_grant(grants):
    store = base._store()
    row = base._harness_claim(store, REQ_APPROVED)
    turns = base.FakeTurnStore(busy=None)
    # Real TurnLogStore.reserve_idle_turn returns False with no DB engine.
    turns.reserve_idle_turn = lambda **kwargs: False
    service = base.FakeService(store, turns)

    try:
        base._run(service, store, row)
    except RuntimeError:
        pass  # A visible fail-closed error is acceptable; a grant is not.

    assert grants.sent == [], "Harness grant escaped without an admission reservation"
    assert service.requests == []


def test_stale_claim_must_not_send_grant_or_touch_new_owner_reservation(grants):
    store = base._store()
    old = base._harness_claim(store, REQ_APPROVED)
    new = store.claim_continuation(old.id, lease_seconds=0)
    assert new.continuation_claim_token != old.continuation_claim_token
    turns = base.FakeTurnStore(busy=None)
    turns.turn = SimpleNamespace(ended_at=None, status="reserved", error_text=None)
    service = base.FakeService(store, turns)

    with pytest.raises(ValueError, match="Invalid or stale approval continuation claim"):
        base._run(service, store, old)

    assert grants.sent == [], "Stale owner sent a grant before validating its claim"
    assert turns.reservations == [], "Stale owner rewrote the successor's reservation"
    assert turns.released == [], "Stale owner dropped the successor's reservation"
    assert store.get_request(old.id).continuation_claim_token == new.continuation_claim_token


def test_error_after_claim_takeover_must_not_release_successor_reservation(grants):
    store = base._store()
    old = base._mcp_claim(store)
    turns = base.FakeTurnStore(busy=None)
    successors = []

    class TakeoverService(base.FakeService):
        async def chat_completion_stream(self, request):
            # A worker pause expires its lease; another owner acquires the same
            # deterministic continuation and reservation before this worker errors.
            successors.append(store.claim_continuation(old.id, lease_seconds=0))
            raise RuntimeError("transport failed after lease takeover")
            yield  # async generator, same interface as the real stream

    with pytest.raises(RuntimeError, match="transport failed after lease takeover"):
        base._run(TakeoverService(store, turns), store, old)

    assert successors[0].continuation_claim_token != old.continuation_claim_token
    assert store.get_request(old.id).continuation_claim_token == successors[0].continuation_claim_token
    assert turns.released == [], "An unfenced cleanup deleted the new owner's reservation"


def test_detected_lease_loss_preserves_reservation_and_cancels_driver(monkeypatch):
    async def scenario():
        store = base._store()
        row = base._mcp_claim(store)
        turns = base.FakeTurnStore(busy=None)
        driver_started = asyncio.Event()
        driver_cancelled = asyncio.Event()
        never = asyncio.Event()
        real_sleep = asyncio.sleep

        async def controlled_sleep(seconds):
            if seconds == 20:
                await driver_started.wait()
            else:
                await real_sleep(0)

        monkeypatch.setattr(continuations.asyncio, "sleep", controlled_sleep)
        monkeypatch.setattr(store, "renew_continuation_claim", lambda *args: False)

        class WaitingService(base.FakeService):
            async def chat_completion_stream(self, request):
                driver_started.set()
                try:
                    await never.wait()
                    yield "data: [DONE]\n\n"
                finally:
                    driver_cancelled.set()

        service = WaitingService(store, turns)
        with pytest.raises(continuations.ContinuationLeaseLost):
            await asyncio.wait_for(
                continuations.dispatch_mcp_result_continuation(service, store, row),
                timeout=2,
            )
        assert turns.reservations
        assert turns.released == []
        assert driver_cancelled.is_set()

    asyncio.run(scenario())


def test_stale_stream_cannot_consume_successor_reservation(monkeypatch):
    """Real save_turn must fence reservation consumption, not only creation.

    SQLite executes the real single-threaded transition with FOR UPDATE removed
    (SQLite lacks that syntax). This is a prearranged takeover, not evidence of
    PostgreSQL locking or concurrent transactions.
    """
    import json

    from sqlalchemy import event
    from sqlmodel import Session

    from llm_bawt.service.schemas import ChatCompletionRequest, ChatMessage, McpToolResultContinuation
    from llm_bawt.service.turn_admission import AdmissionFenceLost, turn_request_extensions
    from llm_bawt.service.turn_logs import TurnLog, TurnLogStore

    approvals = base._store()
    old = base._mcp_claim(approvals)
    identity = continuations._continuation_identity(old)
    # An MCP claim always carries its payload, as dispatch builds it (TASK-1807).
    request = ChatCompletionRequest(
        messages=[ChatMessage(role="user", content="result")],
        bot_id=old.bot_id, user=old.user_id, parent_turn_id=old.turn_id,
        session_id="session-1", **identity,
        continuation_payload=McpToolResultContinuation(**continuations.continuation_payload_from_row(old)),
    )
    request._internal_approval_claim = (old.id, old.continuation_claim_token)
    service = base.FakeService(approvals, base.FakeTurnStore(busy=None))
    # The real stream validates here, then does preparation before save_turn.
    continuations.validate_approval_continuation_claim(service, request, request._internal_approval_claim)
    new = approvals.claim_continuation(old.id, lease_seconds=0)
    assert new.continuation_claim_token != old.continuation_claim_token

    engine = approvals.engine
    TurnLog.__table__.create(engine, checkfirst=True)
    with Session(engine) as session:
        session.add(TurnLog(
            id=identity["inter_bot_turn_id"], bot_id=old.bot_id, user_id=old.user_id,
            status="reserved", request_json=json.dumps({
                "reservation_owner": new.continuation_claim_token,
                "admission": "idle_only",
            }),
        ))
        session.commit()

    @event.listens_for(engine, "before_cursor_execute", retval=True)
    def sqlite_without_row_lock(conn, cursor, statement, parameters, context, executemany):
        return statement.replace(" FOR UPDATE", ""), parameters

    turns = object.__new__(TurnLogStore)
    turns.engine = engine
    monkeypatch.setattr(turns, "_cleanup_expired_if_due", lambda: None)
    monkeypatch.setattr(turns, "_encode_token_usage", lambda *args: None)
    with pytest.raises(AdmissionFenceLost):
        turns.save_turn(
            turn_id=identity["inter_bot_turn_id"], request_id="stale-worker",
            path="/v1/chat/completions", stream=True, model="test-model",
            bot_id=old.bot_id, user_id=old.user_id, status="streaming",
            latency_ms=None, user_prompt="result",
            request_payload=turn_request_extensions(request), response_text="", tool_calls=None,
        )
    saved = turns.get_turn(identity["inter_bot_turn_id"])
    assert saved.status == "reserved"
    assert json.loads(saved.request_json)["reservation_owner"] == new.continuation_claim_token
