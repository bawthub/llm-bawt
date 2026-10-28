"""Unit tests for the queued chat wait reason's wire shape (TASK-1676)."""

from __future__ import annotations

from datetime import datetime, timezone

from llm_bawt.service.delivery_wait_reason import reason_for
from llm_bawt.service.ordinary_turn_queue import OrdinaryTurnDecision
from llm_bawt.service.turn_gates import GateBlock

NOW = datetime.now(timezone.utc)


def _block(mode):
    return GateBlock(id=1, mode=mode, owner_kind="job", owner_id="j", reason=None, expires_at=NOW)


def test_reason_for_names_the_gate_kind():
    assert reason_for(None) == {"kind": "turn_running", "gate": None}
    assert reason_for(_block("hold")) == {"kind": "system_hold", "gate": "quiesce"}
    assert reason_for(_block("deny")) == {"kind": "system_hold", "gate": "freeze"}


def test_queued_receipt_carries_the_wait_reason():
    reason = {"kind": "system_hold", "gate": "quiesce"}
    decision = OrdinaryTurnDecision("queued", "t", "u", "a", "d", "QUEUED", wait_reason=reason)
    assert decision.receipt()["wait_reason"] == reason
    assert "wait_reason" not in OrdinaryTurnDecision("queued", "t", "u", "a", "d", "QUEUED").receipt()


def test_listing_survives_a_gate_read_failure():
    from types import SimpleNamespace

    from llm_bawt.service.delivery_wait_reason import annotate_rows

    class Broken:
        def connect(self):
            raise RuntimeError("db down")

    record = SimpleNamespace(status="QUEUED", metadata={"ordinary_chat": {}}, target_bot_id="b")
    assert annotate_rows(Broken(), [record], [{"delivery_id": "d"}]) == [{"delivery_id": "d"}]


def test_loop_listens_before_reading_gates_and_republishes_on_connect(monkeypatch):
    """Codex r1 B1: a change between the read and the LISTEN must not be lost."""
    import asyncio
    from types import SimpleNamespace

    from llm_bawt.service import gate_change_republisher as rep

    calls = []

    class Waiter:
        def __init__(self, engine, *, channel):
            self.connected = False

        def ensure_listening(self):
            calls.append("listen")
            fresh, self.connected = not self.connected, True
            return fresh

        def wait(self, stop, timeout):
            calls.append("wait")
            raise asyncio.CancelledError

        def close(self):
            calls.append("close")

    monkeypatch.setattr(rep, "next_expiry_wait", lambda engine: calls.append("read") or None)
    monkeypatch.setattr(rep, "republish_waiting", lambda d: calls.append("republish"))
    engine = SimpleNamespace(dialect=SimpleNamespace(name="postgresql"))
    service = SimpleNamespace(_inter_bot_dispatcher=SimpleNamespace(store=SimpleNamespace(engine=engine)))
    try:
        asyncio.run(rep.run_gate_change_republisher(service, waiter_factory=Waiter))
    except asyncio.CancelledError:
        pass
    assert calls == ["listen", "read", "republish", "wait", "close"]


def test_a_dropped_listener_reconnects_and_republishes_without_waiting_out_the_lease(monkeypatch):
    """Codex r2 B1: a listener failure mid-wait reconnects promptly and republishes."""
    import asyncio
    from types import SimpleNamespace

    from llm_bawt.service import gate_change_republisher as rep

    calls = []

    class Waiter:
        def __init__(self, engine, *, channel):
            self.connected, self.waits = False, 0

        def ensure_listening(self):
            calls.append("listen")
            fresh, self.connected = not self.connected, True
            return fresh

        def wait(self, stop, timeout):
            self.waits += 1
            calls.append(f"wait{self.waits}")
            if self.waits == 1:
                raise OSError("connection reset")
            raise asyncio.CancelledError

        def close(self):
            calls.append("close")
            self.connected = False

    monkeypatch.setattr(rep, "RECONNECT_SECONDS", 0)
    monkeypatch.setattr(rep, "next_expiry_wait", lambda engine: calls.append("read") or 3600.0)
    monkeypatch.setattr(rep, "republish_waiting", lambda d: calls.append("republish"))
    engine = SimpleNamespace(dialect=SimpleNamespace(name="postgresql"))
    service = SimpleNamespace(_inter_bot_dispatcher=SimpleNamespace(store=SimpleNamespace(engine=engine)))
    try:
        asyncio.run(rep.run_gate_change_republisher(service, waiter_factory=Waiter))
    except asyncio.CancelledError:
        pass
    assert calls == ["listen", "read", "republish", "wait1", "close",
                     "listen", "read", "republish", "wait2", "close"]


def test_waiter_wait_raises_on_a_broken_connection():
    """The override does not swallow a broken socket into a long sleep."""
    import threading

    import pytest

    from llm_bawt.service.gate_change_republisher import GateChangeWaiter

    class Broken:
        notifies = []

        def poll(self):
            raise OSError("gone")

    waiter = GateChangeWaiter(engine=None)
    waiter._conn = Broken()
    with pytest.raises(OSError):
        waiter.wait(threading.Event(), 3600)
