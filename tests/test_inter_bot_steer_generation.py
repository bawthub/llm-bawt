"""Steer generations for durable deliveries (TASK-1251).

The bridge dedupes a steer RPC on its request id and replays the first cached
result. These tests drive ``InterBotDeliveryDispatcher._steer`` against a fake
bridge with that cache and a run that registers late, and a fake store that
applies ``requeue`` the way the PostgreSQL store does (the PG half is in
test_inter_bot_steer_generation_pg.py).
"""
from __future__ import annotations

import asyncio
import copy
import uuid
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from llm_bawt.service.inter_bot_dispatcher import InterBotDeliveryDispatcher
from llm_bawt.service.inter_bot_dispatcher_steer import (
    NOT_READY_BOUND,
    SteerGeneration,
)

DELIVERY_ID = "delivery-abc123"
BASE_ID = "steer_delivery_abc123"
TARGET = "turn-target"


# --- SteerGeneration -------------------------------------------------------


def _record(metadata=None, delivery_id=DELIVERY_ID):
    return SimpleNamespace(id=delivery_id, metadata=metadata or {})


def test_generation_zero_keeps_the_legacy_request_id():
    assert SteerGeneration.of(_record()).request_id == BASE_ID


def test_later_generations_suffix_the_request_id():
    steer = SteerGeneration.of(_record({"steer_generation": 3}))
    assert steer.request_id == f"{BASE_ID}_g3"


@pytest.mark.parametrize("metadata", [
    {"steer_generation": "junk"}, {"steer_generation": -2}, {"steer_generation": None},
    {"steer_not_ready": "junk"}, {"steer_not_ready": {"count": "x"}}, {},
])
def test_malformed_metadata_reads_as_generation_zero(metadata):
    steer = SteerGeneration.of(_record(metadata))
    assert steer.request_id == BASE_ID
    assert steer.not_ready_count_for(TARGET) == 0


def test_advance_counts_not_ready_per_target_and_restarts_on_a_new_target():
    steer = SteerGeneration.of(_record({
        "steer_generation": 4,
        "steer_not_ready": {"target_turn_id": TARGET, "count": 4},
    }))
    assert steer.not_ready_count_for(TARGET) == 4
    assert steer.advance(not_ready_target=TARGET) == {
        "steer_generation": 5,
        "steer_not_ready": {"target_turn_id": TARGET, "count": 5},
    }
    assert steer.not_ready_count_for("turn-other") == 0
    assert steer.advance(not_ready_target="turn-other") == {
        "steer_generation": 5,
        "steer_not_ready": {"target_turn_id": "turn-other", "count": 1},
    }
    # A definitive rejection advances the id only; the count is left alone.
    assert steer.advance() == {"steer_generation": 5}


def test_backoff_doubles_from_one_second_and_caps_at_fifteen():
    assert [SteerGeneration.backoff_seconds(n) for n in range(1, 8)] == [1, 2, 4, 8, 15, 15, 15]


# --- classifier ------------------------------------------------------------

classify = InterBotDeliveryDispatcher._steer_failure_action


@pytest.mark.parametrize("error", ["no_active_run", "active_run_mismatch"])
def test_bridge_no_run_against_an_open_target_waits_under_the_next_generation(error):
    assert classify(error, accepted=False, target_open=True) == "wait_ready_next_generation"
    assert classify(error, accepted=False, target_open=True,
                    not_ready_count=NOT_READY_BOUND - 1) == "wait_ready_next_generation"


@pytest.mark.parametrize("error", ["no_active_run", "active_run_mismatch"])
def test_bridge_no_run_falls_back_once_the_target_closed_or_the_bound_is_spent(error):
    assert classify(error, accepted=False, target_open=False) == "fallback"
    assert classify(error, accepted=False, target_open=True,
                    not_ready_count=NOT_READY_BOUND) == "fallback"


@pytest.mark.parametrize("error", [
    "maintenance_active", "maintenance_command_requires_turn",
    "Turn is no longer active", "Turn not found",
    "steer_unframed",  # TASK-1300: refused before the CLI, so it never ran
    # TASK-1344: the target already produced its final result. It will never
    # take a steer again; wait for it to end, then use an idle turn.
    "run_finishing",
])
def test_other_definitive_rejections_fall_back_even_with_an_open_target(error):
    assert classify(error, accepted=False, target_open=True) == "fallback"


def test_app_side_not_ready_waits_without_a_new_generation():
    assert classify("Active bridge run is not ready", accepted=False, target_open=True) == "wait_ready"


@pytest.mark.parametrize("error", [
    "no_active_run", "active_run_mismatch", "maintenance_active", "Turn not found",
    "Active bridge run is not ready", "chat.steer timed out after 10s",
])
def test_an_accepted_steer_only_ever_replays_the_same_rpc(error):
    assert classify(error, accepted=True, target_open=True) == "retry_same_steer"


# --- dispatcher against a caching bridge -----------------------------------


class _Bridge:
    """The bridge's steer dedupe: the first result per request id is cached
    and replayed. The target's run registers after ``registers_after`` fresh
    (uncached) steer commands."""

    def __init__(self, *, registers_after: int | None = 0, persisted=None):
        self.registers_after = registers_after
        self.cache: dict[str, str] = {}
        self.fresh = 0
        self.calls: list[tuple[str, str]] = []
        self.persisted = list(persisted or [])
        self.forced: list[str] = []

    async def steer_active_turn(self, service, request, *, steer_request_id, author, wire_suffix):
        from llm_bawt.service.routes.chat import ChatSteerResponse

        self.calls.append((request.turn_id, steer_request_id))
        if self.forced:
            raise HTTPException(status_code=409, detail=self.forced.pop(0))
        if steer_request_id not in self.cache:
            registered = self.registers_after is not None and self.fresh >= self.registers_after
            self.fresh += 1
            self.cache[steer_request_id] = "ok" if registered else "no_active_run"
        if self.cache[steer_request_id] != "ok":
            raise HTTPException(status_code=409, detail=self.cache[steer_request_id])
        persisted = self.persisted.pop(0) if self.persisted else True
        return ChatSteerResponse(ok=True, detail="ok", turn_id=request.turn_id,
                                 message_id=request.message_id, persisted=persisted)

    @property
    def request_ids(self) -> list[str]:
        return [request_id for _turn, request_id in self.calls]


class _Store:
    """One delivery row. ``requeue`` mirrors the PG UPDATE; ``claim`` mirrors
    the claim's ``existing_steer`` branch (same target, attempt + 1)."""

    def __init__(self):
        self.row = SimpleNamespace(
            id=DELIVERY_ID, status="QUEUED", claim_token=None,
            target_turn_id=TARGET, target_bot_id="byte", turn_id="turn-own",
            user_message_id="msg-1", message="hello", author=None,
            metadata={"source": "test"}, attempt_count=0, max_attempts=5,
            transport_accepted_at=None, delivery_mode="steer", last_error=None,
        )
        self.delays: list[float] = []
        self.modes: list[str] = []

    def claim(self, *, target_turn_id: str | None = None):
        row = self.row
        assert row.status == "QUEUED"
        if target_turn_id is not None:  # the claim chose a newer run
            row.target_turn_id, row.delivery_mode = target_turn_id, "steer"
        assert row.delivery_mode == "steer", "the claim would not steer this row"
        row.status, row.claim_token = "STEERING", uuid.uuid4().hex
        row.attempt_count += 1
        return copy.deepcopy(row)

    def payload(self, _id):
        return {"messages": [{"content": "hello"}], "user": "nick"}

    def get(self, _id):
        return copy.deepcopy(self.row)

    def renew_lease(self, *_a):
        return True

    def mark_transport_accepted(self, _id, token):
        assert token == self.row.claim_token
        self.row.transport_accepted_at = "accepted"
        return True

    def mark_delivered(self, _id, token, **_k):
        assert token == self.row.claim_token
        self.row.status = "DELIVERED"
        return self.row

    def fail_claim(self, _id, token, error):
        self.row.status, self.row.last_error = "FAILED", error
        return self.row

    def requeue(self, _id, token, error, *, delay_seconds=0, clear_delivery_mode=False,
                reject_steer_target=False, refund_attempt=False, metadata_merge=None):
        row = self.row
        assert token == row.claim_token and row.status == "STEERING"
        row.status, row.claim_token, row.last_error = "QUEUED", None, error
        if reject_steer_target:
            row.delivery_mode = "steer_rejected"
        if refund_attempt:
            row.attempt_count = max(0, row.attempt_count - 1)
        if metadata_merge:
            row.metadata = {**row.metadata, **copy.deepcopy(dict(metadata_merge))}
        self.delays.append(delay_seconds)
        self.modes.append(row.delivery_mode)
        return row


def _dispatcher(monkeypatch, bridge: _Bridge, *, turns: dict | None = None):
    monkeypatch.setattr("llm_bawt.service.routes.chat.steer_active_turn", bridge.steer_active_turn)
    monkeypatch.setattr("llm_bawt.inter_bot_reply_channel.delivery_reply_channel_note", lambda _r: None)
    turns = turns if turns is not None else {TARGET: SimpleNamespace(ended_at=None, status="streaming")}
    dispatcher = InterBotDeliveryDispatcher.__new__(InterBotDeliveryDispatcher)
    dispatcher.service = SimpleNamespace(
        config=SimpleNamespace(DEFAULT_USER="nick"),
        _turn_log_store=SimpleNamespace(get_turn=lambda turn_id: turns.get(turn_id)),
    )
    dispatcher.store = _Store()
    dispatcher.emitted = []

    async def _emit(record):
        dispatcher.emitted.append(record.status)

    dispatcher._emit = _emit
    return dispatcher


def _steer_once(dispatcher, **claim):
    asyncio.run(dispatcher._steer(dispatcher.store.claim(**claim)))
    return dispatcher.store.row


def _steer_until_settled(dispatcher, limit: int = 60):
    row = dispatcher.store.row
    for _ in range(limit):
        if row.status != "QUEUED" or row.delivery_mode != "steer":
            return row
        row = _steer_once(dispatcher)
    raise AssertionError("delivery never settled")  # pragma: no cover - diagnostic


def test_a_steer_racing_target_registration_is_delivered_into_that_target(monkeypatch):
    """D1 + D2: the run registers after three fresh commands. Each retry of the
    first id would only re-read its cached rejection, and a fallback would park
    the delivery behind the very turn it was meant for."""
    bridge = _Bridge(registers_after=3)
    dispatcher = _dispatcher(monkeypatch, bridge)

    row = _steer_until_settled(dispatcher)

    assert row.status == "DELIVERED"
    assert "steer_rejected" not in dispatcher.store.modes
    assert bridge.calls == [(TARGET, BASE_ID), (TARGET, f"{BASE_ID}_g1"),
                            (TARGET, f"{BASE_ID}_g2"), (TARGET, f"{BASE_ID}_g3")]
    assert dispatcher.store.delays == [1, 2, 4]
    assert row.attempt_count == 1  # only the delivering claim is an attempt
    assert row.metadata["source"] == "test"
    assert row.metadata["steer_not_ready"] == {"target_turn_id": TARGET, "count": 3}


def test_not_ready_waits_are_recorded_on_the_row(monkeypatch):
    bridge = _Bridge(registers_after=None)
    dispatcher = _dispatcher(monkeypatch, bridge)
    row = _steer_once(dispatcher)
    assert row.last_error == "steer target not ready: no_active_run (generation 0)"
    assert row.metadata["steer_generation"] == 1


@pytest.mark.parametrize("error", ["no_active_run", "maintenance_active", "Turn not found"])
def test_a_definitive_rejection_advances_so_the_next_target_is_never_refused_by_the_cache(
    monkeypatch, error,
):
    """A: the rejection of generation k is cached for seven days. When the
    claim later steers a newer run, it must use k+1."""
    bridge = _Bridge(registers_after=0)
    bridge.forced = [error]  # the bridge (or app) refuses generation 0 outright
    dispatcher = _dispatcher(monkeypatch, bridge, turns={})  # the target already ended
    if error == "no_active_run":
        bridge.forced = []
        bridge.cache[BASE_ID] = "no_active_run"  # cached from the old run

    row = _steer_once(dispatcher)
    assert row.delivery_mode == "steer_rejected"
    assert row.metadata["steer_generation"] == 1
    assert row.attempt_count == 0

    row = _steer_once(dispatcher, target_turn_id="turn-newer")
    assert row.status == "DELIVERED"
    assert bridge.calls[-1] == ("turn-newer", f"{BASE_ID}_g1")


def test_a_target_that_never_registers_falls_back_after_the_bound(monkeypatch):
    """B: bounded. After NOT_READY_BOUND not-ready generations the delivery
    waits for the turn to end, as before TASK-1251."""
    bridge = _Bridge(registers_after=None)
    dispatcher = _dispatcher(monkeypatch, bridge)

    row = _steer_until_settled(dispatcher)

    assert row.status == "QUEUED" and row.delivery_mode == "steer_rejected"
    assert len(bridge.calls) == NOT_READY_BOUND + 1
    assert len(set(bridge.request_ids)) == NOT_READY_BOUND + 1  # never a cached re-read
    assert dispatcher.store.delays[:NOT_READY_BOUND] == [
        SteerGeneration.backoff_seconds(n) for n in range(1, NOT_READY_BOUND + 1)
    ]
    assert max(dispatcher.store.delays) == 15
    assert row.attempt_count == 0
    assert row.metadata["steer_generation"] == NOT_READY_BOUND + 1


def test_an_accepted_but_unpersisted_steer_replays_the_accepted_id(monkeypatch):
    """C9: once the bridge accepted generation k, nothing advances it: a fresh
    id could deliver the message twice."""
    bridge = _Bridge(registers_after=1, persisted=[False, True])
    dispatcher = _dispatcher(monkeypatch, bridge)

    _steer_once(dispatcher)  # g0: no run yet
    row = _steer_once(dispatcher)  # g1: accepted, persistence failed
    assert row.status == "QUEUED" and row.transport_accepted_at == "accepted"
    bridge.forced = ["no_active_run"]  # e.g. the run ended since
    row = _steer_once(dispatcher)
    assert row.status == "QUEUED" and row.delivery_mode == "steer"
    row = _steer_once(dispatcher)

    assert row.status == "DELIVERED"
    assert bridge.request_ids == [BASE_ID] + [f"{BASE_ID}_g1"] * 3
    assert row.metadata["steer_generation"] == 1


def test_a_timeout_replays_the_same_generation(monkeypatch):
    bridge = _Bridge(registers_after=0)
    bridge.forced = ["chat.steer timed out after 10s"]
    dispatcher = _dispatcher(monkeypatch, bridge)
    _steer_once(dispatcher)
    row = _steer_once(dispatcher)
    assert row.status == "DELIVERED"
    assert bridge.request_ids == [BASE_ID, BASE_ID]
    assert "steer_generation" not in row.metadata


# --- chat.py: a caller-chosen id is sent once -------------------------------


def _chat_steer(monkeypatch, steer_request_id):
    from llm_bawt import bots
    from llm_bawt.agent_backends import agent_bridge
    from llm_bawt.service.routes.chat import ChatSteerRequest, steer_active_turn

    sends: list[str | None] = []
    sleeps: list[float] = []

    async def send_steer(**kwargs):
        sends.append(kwargs.get("request_id"))
        return {"ok": False, "error": "no_active_run"}

    async def sleep(seconds):
        sleeps.append(seconds)

    # Restored by monkeypatch; nothing else in this test awaits a sleep.
    monkeypatch.setattr("llm_bawt.service.routes.chat.asyncio.sleep", sleep)
    monkeypatch.setattr(agent_bridge, "get_agent_subscriber",
                        lambda: SimpleNamespace(send_steer=send_steer))
    monkeypatch.setattr(bots, "BotManager", lambda _config: SimpleNamespace(
        get_bot=lambda _bot_id: SimpleNamespace(agent_backend="claude-code")))
    turn = SimpleNamespace(
        id="turn-live", bot_id="byte", user_id="nick", status="streaming", ended_at=None,
        agent_session_key="byte:nick", agent_request_id="req-live", user_prompt="work",
    )
    service = SimpleNamespace(config=None, _turn_log_store=SimpleNamespace(get_turn=lambda _id: turn))
    with pytest.raises(HTTPException) as caught:
        asyncio.run(steer_active_turn(
            service,
            ChatSteerRequest(turn_id=turn.id, message="hi", message_id="m-1",
                             bot_id="byte", user_id="nick"),
            steer_request_id=steer_request_id,
        ))
    assert caught.value.detail == "no_active_run"
    return sends, sleeps


def test_a_caller_chosen_steer_id_is_sent_once(monkeypatch):
    """Re-sending a deduped id re-reads the cached result; the caller retries."""
    assert _chat_steer(monkeypatch, f"{BASE_ID}_g2") == ([f"{BASE_ID}_g2"], [])


def test_a_composer_steer_still_retries_while_the_run_registers(monkeypatch):
    sends, sleeps = _chat_steer(monkeypatch, None)
    assert len(sends) == 8 and sleeps == [0.25] * 7


def test_the_not_ready_bound_restarts_when_the_claim_picks_a_new_target(monkeypatch):
    """B: the count is per target turn. A newer run gets its own bound."""
    bridge = _Bridge(registers_after=None)
    turns = {TARGET: SimpleNamespace(ended_at=None, status="streaming"),
             "turn-newer": SimpleNamespace(ended_at=None, status="reserved")}
    dispatcher = _dispatcher(monkeypatch, bridge, turns=turns)
    for _ in range(NOT_READY_BOUND - 1):
        row = _steer_once(dispatcher)
    assert row.metadata["steer_not_ready"] == {"target_turn_id": TARGET, "count": NOT_READY_BOUND - 1}

    row = _steer_once(dispatcher, target_turn_id="turn-newer")
    assert row.delivery_mode == "steer"
    assert row.metadata["steer_not_ready"] == {"target_turn_id": "turn-newer", "count": 1}
    assert dispatcher.store.delays[-1] == 1
