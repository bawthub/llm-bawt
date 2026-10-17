"""HTTP retry publishes a lifecycle event only for a new delivery."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from llm_bawt.inter_bot_delivery_retry import DeliveryRetryConflict
from llm_bawt.service.routes import inter_bot_deliveries as routes


@pytest.mark.asyncio
async def test_retry_emits_new_row_once_and_wakes_target(monkeypatch):
    original = SimpleNamespace(metadata={}, message_kind="READY", target_bot_id="snark")
    new = SimpleNamespace(target_bot_id="snark", to_api=lambda **kw: {
        "delivery_id": "delivery-new", "duplicate": kw.get("duplicate", False),
    })
    emitted: list[str] = []
    woken: list[str] = []
    dispatcher = SimpleNamespace(store=SimpleNamespace(get=lambda _id: original),
                                 _emit=lambda row: _emit(row), wake=woken.append)

    async def _emit(row):
        emitted.append(row.to_api()["delivery_id"])

    monkeypatch.setattr(routes, "_dispatcher", lambda: dispatcher)
    monkeypatch.setattr(routes, "get_service", lambda: SimpleNamespace(config=object()))
    monkeypatch.setattr(routes, "BotManager", lambda _config: SimpleNamespace(get_bot=lambda _id: object()))
    monkeypatch.setattr(routes, "durable_agent_backend", lambda _bot: True)
    for duplicate in (False, True):
        monkeypatch.setattr(routes, "retry_failed_delivery", lambda _store, _id: (new, duplicate))
        result = await routes.retry_delivery("delivery-old")
        assert result == {"delivery_id": "delivery-new", "duplicate": duplicate,
                          "retry_of_delivery_id": "delivery-old"}
    assert emitted == ["delivery-new"]
    assert woken == ["snark", "snark"]


@pytest.mark.asyncio
async def test_retry_rejects_hidden_or_unsafe_rows_without_emitting(monkeypatch):
    rows = [None,
            SimpleNamespace(metadata={"ordinary_chat": {}}, message_kind="READY", target_bot_id="snark"),
            SimpleNamespace(metadata={}, message_kind="READY", target_bot_id="snark")]
    dispatcher = SimpleNamespace(store=SimpleNamespace(get=lambda _id: rows.pop(0)))
    monkeypatch.setattr(routes, "_dispatcher", lambda: dispatcher)
    monkeypatch.setattr(routes, "get_service", lambda: SimpleNamespace(config=object()))
    monkeypatch.setattr(routes, "BotManager", lambda _config: SimpleNamespace(get_bot=lambda _id: object()))
    monkeypatch.setattr(routes, "durable_agent_backend", lambda _bot: True)
    monkeypatch.setattr(routes, "retry_failed_delivery",
                        lambda _store, _id: (_ for _ in ()).throw(DeliveryRetryConflict("Only failed deliveries can be retried")))
    for expected in (404, 404, 409):
        with pytest.raises(HTTPException) as exc:
            await routes.retry_delivery("delivery-old")
        assert exc.value.status_code == expected
