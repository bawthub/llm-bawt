"""bots_send_message mode resolution + cross-mode idempotency (TASK-1015).

Replays the 2026-10-04 Al -> Caid incident: a waited send interrupted mid-wait,
then a same-key async retry. Before TASK-1015 the waited path bypassed the
durable table entirely and the retry started a second target turn.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

import llm_bawt.mcp_server.server as server  # registers tools; import first
import llm_bawt.mcp_server.inter_bot_tools as tools  # noqa: E402
from llm_bawt.mcp_server.inter_bot_send_mode import (
    STEER_OR_IDLE,
    WHEN_IDLE,
    SendModeError,
    resolve_send_mode,
)


def _mode(**overrides: Any):
    kwargs = dict(
        delivery=None, queue_if_busy=False, fire_and_forget=None,
        wait_for_reply=False, task_id=None, idempotency_key=None,
    )
    kwargs.update(overrides)
    return resolve_send_mode(**kwargs)


# ---- resolver -------------------------------------------------------------

def test_default_is_async_steer_or_idle() -> None:
    mode = _mode()
    assert (mode.delivery, mode.wait) == (STEER_OR_IDLE, False)


def test_queue_if_busy_and_aliases_mean_when_idle() -> None:
    assert _mode(queue_if_busy=True).delivery == WHEN_IDLE
    for alias in ("when_idle", "when-idle", "queued", "queue_if_busy"):
        assert _mode(delivery=alias).delivery == WHEN_IDLE
    assert _mode(delivery="immediate").delivery == STEER_OR_IDLE


def test_incident_combo_is_durable_waited_when_idle() -> None:
    mode = _mode(
        delivery="when_idle", queue_if_busy=True, fire_and_forget=False,
        task_id="TASK-1013", idempotency_key="TASK-1013:START",
    )
    assert (mode.delivery, mode.wait) == (WHEN_IDLE, True)


@pytest.mark.parametrize("overrides", [
    {"wait_for_reply": True, "fire_and_forget": True},
    {"queue_if_busy": True, "delivery": "steer_or_idle"},
    {"queue_if_busy": True, "delivery": "immediate"},
    {"wait_for_reply": True, "delivery": "steer_or_idle"},
    {"fire_and_forget": False, "delivery": "immediate"},
    {"wait_for_reply": True, "task_id": "TASK-1"},
    {"delivery": "sideways"},
])
def test_conflicting_combinations_are_rejected(overrides: dict) -> None:
    with pytest.raises(SendModeError):
        _mode(**overrides)


def test_task_handoff_wait_allowed_with_key() -> None:
    assert _mode(wait_for_reply=True, task_id="TASK-1", idempotency_key="TASK-1:READY").wait


# ---- tool-level behavior --------------------------------------------------

class FakeDurable:
    """Stand-in for inter_bot_deliveries' (sender, target, key) uniqueness."""

    def __init__(self) -> None:
        self.rows: dict[tuple, dict] = {}
        self.enqueue_calls: list[dict] = []

    async def enqueue(self, **kwargs: Any) -> dict:
        self.enqueue_calls.append(kwargs)
        key = (kwargs["sender_bot_id"], kwargs["target_bot_id"], kwargs["idempotency_key"])
        if key in self.rows and kwargs["idempotency_key"]:
            return {**self.rows[key], "duplicate": True}
        n = len(self.rows) + 1
        row = {"delivery_id": f"delivery-{n}", "turn_id": f"turn-delivery-{n}",
               "status": "QUEUED", "duplicate": False}
        self.rows[key] = row
        return row

    async def find(self, sender: str, target: str, key: str) -> dict | None:
        return self.rows.get((sender, target, key))


def _install(monkeypatch: pytest.MonkeyPatch, durable: FakeDurable, *, busy: dict | None = None):
    async def check(_target: str):
        return busy

    monkeypatch.setattr(server, "_enqueue_durable", durable.enqueue)
    monkeypatch.setattr(server, "_find_delivery_by_key", durable.find)
    monkeypatch.setattr(server, "_check_bot_in_turn", check)
    monkeypatch.setattr(server, "_bot_send_wait_ceiling_seconds", lambda: 300.0)


def test_interrupted_wait_then_same_key_async_retry_reuses_one_delivery(monkeypatch) -> None:
    durable = FakeDurable()
    _install(monkeypatch, durable)

    async def interrupted(_receipt: dict, _wait: float) -> dict:
        raise asyncio.CancelledError  # caller steered mid-wait

    monkeypatch.setattr(server, "_await_delivery", interrupted)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(server.send_message_to_bot(
            target_bot_id="caid", message="execute", sender_bot_id="al",
            delivery="when_idle", queue_if_busy=True, fire_and_forget=False,
            idempotency_key="TASK-1013:START", task_id="TASK-1013",
        ))
    # The receipt was persisted BEFORE the wait began.
    assert len(durable.rows) == 1

    retry = asyncio.run(server.send_message_to_bot(
        target_bot_id="caid", message="execute (retry)", sender_bot_id="al",
        delivery="when_idle", fire_and_forget=True,
        idempotency_key="TASK-1013:START", task_id="TASK-1013",
    ))
    assert retry["duplicate"] is True
    assert retry["delivery_id"] == "delivery-1"
    assert retry["dispatched"] is True and retry["mode"] == "async"
    assert len(durable.rows) == 1


def test_waited_retry_reattaches_even_while_target_busy(monkeypatch) -> None:
    durable = FakeDurable()
    durable.rows[("al", "caid", "K")] = {"delivery_id": "delivery-9", "turn_id": "turn-delivery-9",
                                         "status": "DISPATCHING"}
    _install(monkeypatch, durable, busy={"turn_id": "turn-delivery-9", "status": "streaming"})
    seen: list[dict] = []

    async def wait(receipt: dict, _wait: float) -> dict:
        seen.append(receipt)
        return {**receipt, "status": "DELIVERED", "success": True, "content": "report"}

    monkeypatch.setattr(server, "_await_delivery", wait)
    result = asyncio.run(server.send_message_to_bot(
        target_bot_id="caid", message="x", sender_bot_id="al",
        wait_for_reply=True, idempotency_key="K",
    ))
    assert result["success"] is True and result["content"] == "report"
    assert seen[0]["delivery_id"] == "delivery-9" and seen[0]["duplicate"] is True
    assert durable.enqueue_calls == []


def test_waited_send_to_busy_target_has_no_side_effect(monkeypatch) -> None:
    durable = FakeDurable()
    _install(monkeypatch, durable, busy={"turn_id": "turn-x", "status": "streaming"})
    result = asyncio.run(server.send_message_to_bot(
        target_bot_id="caid", message="x", sender_bot_id="al", wait_for_reply=True,
    ))
    assert result["in_turn"] is True and result["dispatched"] is False
    assert durable.enqueue_calls == []


def test_conflict_is_rejected_before_any_side_effect(monkeypatch) -> None:
    durable = FakeDurable()
    _install(monkeypatch, durable)
    result = asyncio.run(server.send_message_to_bot(
        target_bot_id="caid", message="x", sender_bot_id="al",
        wait_for_reply=True, delivery="steer_or_idle",
    ))
    assert result["success"] is False and result["dispatched"] is False
    assert durable.enqueue_calls == []


def test_waited_enqueue_is_when_idle_with_target_budget(monkeypatch) -> None:
    durable = FakeDurable()
    _install(monkeypatch, durable)

    async def wait(receipt: dict, wait_seconds: float) -> dict:
        return {**receipt, "success": True, "wait": wait_seconds}

    monkeypatch.setattr(server, "_await_delivery", wait)
    result = asyncio.run(server.send_message_to_bot(
        target_bot_id="caid", message="x", sender_bot_id="al",
        wait_for_reply=True, timeout_seconds=60,
    ))
    call = durable.enqueue_calls[0]
    assert call["prefer_steer"] is False
    assert call["timeout_seconds"] == 1800.0  # target turn is not killed by caller budget
    assert result["wait"] == 60 and result["mode"] == "waited"


# ---- bounded wait ----------------------------------------------------------

def test_await_delivery_reads_reply_from_exact_turn(monkeypatch) -> None:
    responses = {
        "/v1/inter-bot-deliveries/delivery-1": {"status": "DELIVERED", "turn_id": "turn-delivery-1"},
        "/v1/turn-logs/turn-delivery-1": {"response": "the reply", "status": "ok", "model": "m"},
    }

    async def get_json(path: str):
        return responses.get(path)

    monkeypatch.setattr(server, "_get_json", get_json)
    result = asyncio.run(tools._await_delivery(
        {"delivery_id": "delivery-1", "status": "QUEUED"}, 5.0,
    ))
    assert result["success"] is True and result["content"] == "the reply"
    assert result["turn_status"] == "ok"


def test_await_delivery_timeout_is_in_flight_not_failure_to_dispatch(monkeypatch) -> None:
    async def get_json(_path: str):
        return {"status": "DISPATCHING"}

    monkeypatch.setattr(server, "_get_json", get_json)
    monkeypatch.setattr(tools, "_now", iter([0.0, 0.0, 10.0]).__next__)
    result = asyncio.run(tools._await_delivery(
        {"delivery_id": "delivery-1", "status": "QUEUED"}, 1.0,
    ))
    assert result["error"] == "timeout" and result["in_flight"] is True
    assert result["dispatched"] is True
    assert "SAME idempotency_key" in result["warning"]


def test_await_delivery_surfaces_failed_terminal(monkeypatch) -> None:
    async def get_json(_path: str):
        return {"status": "FAILED", "last_error": "dead-lettered"}

    monkeypatch.setattr(server, "_get_json", get_json)
    result = asyncio.run(tools._await_delivery(
        {"delivery_id": "delivery-1", "status": "QUEUED"}, 5.0,
    ))
    assert result["success"] is False and result["error"] == "dead-lettered"
