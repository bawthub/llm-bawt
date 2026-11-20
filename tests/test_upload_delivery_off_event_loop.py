"""TASK-2552: the serving loop stays runnable while an upload's storage write or a
delivery's admission transaction is held."""

import asyncio
import base64
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import HTTPException

from llm_bawt import inter_bot_send as send
from llm_bawt.media.object_store import BlobBackendUnavailable
from llm_bawt.service.routes import inter_bot_deliveries as route
from llm_bawt.service.routes import uploads


async def _while_blocked(start, *, complete=lambda: None):
    loop = asyncio.get_running_loop()
    entered = asyncio.Event()
    release = threading.Event()
    ran = []

    def blocking(*args, **kwargs):
        ran.append(threading.get_ident())
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(5), "blocking work kept the serving loop from running"
        return complete(*args, **kwargs)

    work = asyncio.create_task(start(blocking))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        assert len(ran) == 1
        assert ran[0] != threading.get_ident()
        assert not work.done(), "the loop did not run while the worker was held"
    finally:
        release.set()
    return await work


PNG = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\0" * 32).decode()


class _JsonRequest:
    headers = {"content-type": "application/json"}

    async def json(self):
        return {"data_url": f"data:image/png;base64,{PNG}", "filename": "fixture.png"}


def _upload(monkeypatch, blocking):
    store = SimpleNamespace(db=object(), upload=blocking)
    monkeypatch.setattr(uploads, "_store", lambda: store)
    monkeypatch.setattr(uploads, "asset_to_upload_response_dict", lambda asset: {"id": asset})
    return uploads.upload_asset(_JsonRequest(), source="chat_upload", x_entity_id="fixture-user", file=None)


def test_upload_storage_write_keeps_loop_runnable(monkeypatch):
    def stored(**kwargs):
        assert kwargs["owner_user_id"] == "fixture-user" and kwargs["raw_bytes"]
        return "fixture-asset"

    response = asyncio.run(_while_blocked(lambda b: _upload(monkeypatch, b), complete=stored))
    assert response.status_code == 200


@pytest.mark.parametrize(("error", "status"), [
    (ValueError("bad source"), 400),
    (BlobBackendUnavailable("down"), 503),
    (RuntimeError("boom"), 500),
])
def test_upload_errors_keep_their_status_off_loop(monkeypatch, error, status):
    def fail(**_):
        raise error

    with pytest.raises(HTTPException) as exc:
        asyncio.run(_upload(monkeypatch, fail))
    assert exc.value.status_code == status


@pytest.fixture
def delivery(monkeypatch):
    record = SimpleNamespace(id="fixture-delivery")
    store = Mock(engine=True)
    store.get.return_value = record
    store.payload.return_value = {}
    dispatcher = SimpleNamespace(store=store, _emit=AsyncMock(), wake=Mock())
    service = SimpleNamespace(config=SimpleNamespace(DEFAULT_USER="fixture-user"),
                              _inter_bot_dispatcher=dispatcher)
    bot = SimpleNamespace(agent_backend="claude-code", harness="claude-code")
    bots = {"fixture-target": bot, "fixture-sender": bot}
    monkeypatch.setattr(route, "get_service", lambda: service)
    monkeypatch.setattr(send, "BotManager", lambda config: SimpleNamespace(get_bot=bots.get))
    monkeypatch.setattr(send, "submission_result", lambda *a, **kw: (202, {"success": True}))
    return SimpleNamespace(record=record, store=store, dispatcher=dispatcher)


def _body(**kwargs):
    return route.DeliveryCreate(**{
        "sender_bot_id": "fixture-sender", "target_bot_id": "fixture-target",
        "message": "fixture message", "prefer_steer": False,
        "author_entity_type": "user", "author_entity_id": "fixture-user",
        "session_policy": "continue", **kwargs})


def test_delivery_admission_transaction_keeps_loop_runnable(delivery):
    async def start(blocking):
        delivery.store.enqueue.side_effect = blocking
        return await route.create_delivery(_body(idempotency_key="fixture-key"), SimpleNamespace(headers={}))

    asyncio.run(_while_blocked(start, complete=lambda *a, **kw: (delivery.record, False)))
    assert delivery.store.enqueue.call_args.kwargs["idempotency_key"] == "fixture-key"
    delivery.dispatcher._emit.assert_awaited_once_with(delivery.record)


def test_duplicate_admission_is_not_emitted_again(delivery):
    delivery.store.enqueue.return_value = (delivery.record, True)
    asyncio.run(route.create_delivery(_body(idempotency_key="fixture-key"), SimpleNamespace(headers={})))
    delivery.dispatcher._emit.assert_not_awaited()


def test_admission_value_error_still_maps_to_409(delivery):
    delivery.store.enqueue.side_effect = ValueError("conflicting key")
    with pytest.raises(HTTPException) as exc:
        asyncio.run(route.create_delivery(_body(idempotency_key="fixture-key"), SimpleNamespace(headers={})))
    assert exc.value.status_code == 409



def test_unknown_target_refusal_is_settled_inside_admission(delivery):
    """A refusal rides the admission transaction, not a later awaited write
    that a cancelled request could skip (TASK-2552)."""
    delivery.store.enqueue.return_value = (delivery.record, False)
    asyncio.run(route.create_delivery(_body(target_bot_id="fixture-missing", idempotency_key="fixture-key"),
                                      SimpleNamespace(headers={})))
    assert "fixture-missing" in delivery.store.enqueue.call_args.kwargs["refusal"]
    delivery.store.mark_failed.assert_not_called()
