"""TASK-1357: an approval result reaches a busy bot by steer, not only at idle."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

from llm_bawt.approval_policies import CONT_DELIVERED, CONT_PENDING
from llm_bawt.service import approval_continuation_steer as steer
from llm_bawt.service import approval_continuations as conts
from llm_bawt.service.approval_continuations import dispatch_mcp_result_continuation

from tests.test_mcp_approval_continuations import FakeService, FakeTurnStore, _ready_row, _store


class FakeDeliveryStore:
    def __init__(self, *, fail=False, row=None):
        self.enqueued = []
        self.fail = fail
        self.row = row
        self.engine = self

    # enqueue / announce
    def enqueue(self, **kwargs):
        if self.fail:
            raise RuntimeError("db down")
        self.enqueued.append(kwargs)
        return SimpleNamespace(id="d-1", target_bot_id=kwargs["target_bot_id"]), len(self.enqueued) > 1

    # engine.connect() -> conn.execute(...).mappings().first()
    def connect(self):
        store = self

        class Conn:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def execute(self, *_args, **_kwargs):
                return SimpleNamespace(mappings=lambda: SimpleNamespace(first=lambda: store.row))

        return Conn()


class FakeDispatcher:
    def __init__(self, store):
        self.store = store
        self.announced = []
        self.woken = []

    def announce_admitted(self, record):
        self.announced.append(record.id)

    def wake(self, bot_id):
        self.woken.append(bot_id)


def _service(store, *, busy=None):
    service = FakeService()
    service._turn_log_store = FakeTurnStore(busy=busy)
    service._inter_bot_dispatcher = FakeDispatcher(store)
    return service


def _started_row(generation=1, started=True):
    metadata = {"steer_generation": generation}
    if started:
        metadata["steer_started"] = {"generation": generation}
    return {"delivery_mode": "steer", "metadata_json": metadata}


def test_busy_target_gets_one_steer_only_delivery_and_no_turn():
    store = _store()
    row = _ready_row(store)
    deliveries = FakeDeliveryStore()
    service = _service(deliveries, busy="executing turn-x")

    asyncio.run(dispatch_mcp_result_continuation(service, store, row))

    assert service.requests == []
    assert service._turn_log_store.reservations == []
    [sent] = deliveries.enqueued
    assert sent["idempotency_key"] == f"approval-cont-steer:{row.id}"
    assert sent["target_bot_id"] == row.bot_id
    assert sent["payload"]["steer_only"] is True and sent["payload"]["prefer_steer"] is True
    assert sent["payload"]["session_id"] is None
    assert '"job_id": "job-1"' in sent["message"]
    assert service._inter_bot_dispatcher.announced == ["d-1"]
    persisted = store.get_request(row.id)
    assert persisted.continuation_state == CONT_PENDING
    assert "waiting for idle target" in persisted.continuation_last_error


def test_started_steer_marks_delivered_without_a_second_turn():
    store = _store()
    row = _ready_row(store)
    service = _service(FakeDeliveryStore(row=_started_row()))

    asyncio.run(dispatch_mcp_result_continuation(service, store, row))

    assert service.requests == []
    assert service._turn_log_store.reservations == []
    assert store.get_request(row.id).continuation_state == CONT_DELIVERED


def test_unstarted_or_skipped_steer_falls_back_to_the_idle_turn():
    for found in (None, _started_row(started=False),
                  {"delivery_mode": "skipped", "metadata_json": {}}):
        store = _store()
        row = _ready_row(store)
        service = _service(FakeDeliveryStore(row=found))

        asyncio.run(dispatch_mcp_result_continuation(service, store, row))

        assert len(service.requests) == 1
        assert store.get_request(row.id).continuation_state == CONT_DELIVERED


def test_stale_generation_start_does_not_count():
    row = SimpleNamespace(id="r", bot_id="b", request_kind="mcp")
    stale = {"delivery_mode": "steer",
             "metadata_json": {"steer_generation": 2, "steer_started": {"generation": 1}}}
    service = SimpleNamespace(_inter_bot_dispatcher=FakeDispatcher(FakeDeliveryStore(row=stale)))
    assert steer.steer_landed(service, row) is False


def test_steer_enqueue_failure_keeps_the_idle_path():
    store = _store()
    row = _ready_row(store)
    service = _service(FakeDeliveryStore(fail=True), busy="executing turn-x")

    asyncio.run(dispatch_mcp_result_continuation(service, store, row))

    persisted = store.get_request(row.id)
    assert persisted.continuation_state == CONT_PENDING
    assert "waiting for idle target" in persisted.continuation_last_error


def test_harness_and_missing_dispatcher_are_never_steered():
    harness = SimpleNamespace(id="r", bot_id="b", request_kind="harness", user_id="u")
    deliveries = FakeDeliveryStore(row=_started_row())
    service = SimpleNamespace(_inter_bot_dispatcher=FakeDispatcher(deliveries))
    steer.request_steer(service, harness, "x")
    assert deliveries.enqueued == []
    assert steer.steer_landed(service, harness) is False
    mcp = SimpleNamespace(id="r", bot_id="b", request_kind="mcp", user_id="u")
    assert steer.steer_landed(SimpleNamespace(), mcp) is False
    steer.request_steer(SimpleNamespace(), mcp, "x")  # no dispatcher: silent no-op


def test_retry_reuses_the_same_steer_key():
    store = _store()
    row = _ready_row(store)
    deliveries = FakeDeliveryStore()
    service = _service(deliveries, busy="executing turn-x")
    asyncio.run(dispatch_mcp_result_continuation(service, store, row))
    conts.request_steer(service, row, "again")
    keys = {d["idempotency_key"] for d in deliveries.enqueued}
    assert keys == {f"approval-cont-steer:{row.id}"}
    assert service._inter_bot_dispatcher.woken == [row.bot_id]
