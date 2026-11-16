"""Internal HTTP inbox contract against private PostgreSQL, never a live SDK."""
import asyncio
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from llm_bawt.inter_bot_inbox_claim import INBOX
from llm_bawt.inter_bot_inbox_settlement import InboxSettlement
from llm_bawt.service.routes import bridge_inbox as api
from tests import test_delivery_inbox_settlement_pg as settlement
from tests import test_delivery_inbox_claim_pg as claims

rig, running = claims.rig, claims.running
pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]
PREFIX = "/v1/internal/inbox"


@pytest.fixture
def http(rig, running, monkeypatch):
    store, binding = running
    dispatcher = SimpleNamespace(
        store=store, announce_admitted=Mock(), announce_lifecycle=Mock(),
    )
    memory, service, subscriber = settlement.history(rig)
    gate = Mock()
    monkeypatch.setattr(api, "_dispatcher", lambda: dispatcher)
    monkeypatch.setattr(api, "get_service", lambda: service)
    monkeypatch.setattr(api.bridge_gate, "check", gate)
    monkeypatch.setattr(
        "llm_bawt.agent_backends.agent_bridge.get_agent_subscriber", lambda: subscriber,
    )
    monkeypatch.setattr(
        "llm_bawt.inter_bot_reply_channel.delivery_reply_channel_note", lambda _: "REPLY[channel]",
    )
    framing = Mock(return_value={"framed_message": "NEXT[callback]", "interrupt_item": "ITEM[callback]"})
    monkeypatch.setattr(api, "_steer_framing", framing)
    app = FastAPI()
    app.include_router(api.router)
    body = dict(
        bot_id=binding.bot_id, user_id=binding.user_id, backend=binding.backend,
        session_key=binding.session_key, request_id=binding.request_id, claim_owner="bridge-fixture",
    )
    with TestClient(app) as client:
        yield SimpleNamespace(client=client, store=store, binding=binding, dispatcher=dispatcher,
                              gate=gate, framing=framing, body=body, memory=memory)


def post(http, path, body):
    response = http.client.post(PREFIX + path, json=body)
    assert response.status_code == 200, response.text
    return response.json()


def test_claim_prepare_positive_receipt_then_persistence_only_ack(rig, http):
    row = claims.send(rig, http.store)
    claimed = post(http, "/claim", http.body)["claim"]
    assert claimed["delivery_id"] == row.id
    assert http.dispatcher.announce_lifecycle.call_args.args[0].status == "STEERING"
    assert claimed["evidence"]["binding"] == asdict(http.binding)
    assert claimed["claim_token"] == claimed["evidence"]["write_token"]
    assert claimed["framed_message"] == "NEXT[callback]"
    http.framing.assert_called_once_with(
        "callback\n\nREPLY[channel]", origin="bot", priority="next",
        bot_id=rig.bot, user_id=rig.user,
    )
    operation = dict(delivery_id=row.id, claim_token=claimed["claim_token"])
    prepare = operation | {"binding": claimed["evidence"]["binding"]}
    assert post(http, "/prepare", prepare) == {"authorized": True}
    assert post(http, "/prepare", prepare) == {"authorized": False}
    assert post(http, "/repair", operation) == {"acknowledged": False}
    receipt = dict(delivery_id=row.id, evidence=claimed["evidence"], state="queued", content_offset=8)
    assert post(http, "/lifecycle", receipt) == {"recorded": True}
    assert http.store.get(row.id).status == "STEERING"
    assert post(http, "/repair", operation) == {"acknowledged": True}
    assert http.store.get(row.id).status == "DELIVERED"
    assert http.dispatcher.announce_lifecycle.call_args.args[0].status == "DELIVERED"
    assert INBOX not in http.store.get(row.id).to_api()["metadata"]
    assert post(http, "/repair", operation) == {"acknowledged": True}
    assert post(http, "/repair", operation | {"claim_token": "stale"}) == {"acknowledged": False}
    assert http.gate.call_count == 8


@pytest.mark.parametrize("field", ["bot_id", "user_id", "backend", "session_key", "request_id"])
def test_foreign_run_identity_never_claims_or_reserves_idle_work(rig, http, field):
    row = claims.send(rig, http.store)
    assert post(http, "/claim", http.body | {field: "foreign"}) == {"claim": None}
    assert http.store.get(row.id).status == "QUEUED"
    assert http.store.get(row.id).attempt_count == 0
    http.framing.assert_not_called()


@pytest.mark.parametrize("path,body", [
    ("/claim", "claim"), ("/prepare", "prepare"), ("/repair/claim", "repair_claim"),
    ("/repair", "operation"), ("/lifecycle", "receipt"),
])
def test_every_internal_operation_uses_bridge_gate(rig, http, path, body):
    row = claims.send(rig, http.store)
    claimed = post(http, "/claim", http.body)["claim"]
    operation = dict(delivery_id=row.id, claim_token=claimed["claim_token"])
    bodies = {
        "claim": http.body,
        "prepare": operation | {"binding": claimed["evidence"]["binding"]},
        "repair_claim": dict(bot_id=rig.bot, claim_owner="replacement"),
        "operation": operation,
        "receipt": dict(delivery_id=row.id, evidence=claimed["evidence"], state="queued"),
    }
    http.gate.side_effect = HTTPException(401, "fixture denied")
    assert http.client.post(PREFIX + path, json=bodies[body]).status_code == 401
    assert http.store.get(row.id).metadata[INBOX]["phase"] == "claimed"


def test_repair_takeover_cannot_prepare_or_ack_with_previous_owner(rig, http):
    row = claims.send(rig, http.store)
    claimed = post(http, "/claim", http.body)["claim"]
    old = dict(delivery_id=row.id, claim_token=claimed["claim_token"])
    assert post(http, "/prepare", old | {"binding": asdict(http.binding)})["authorized"]
    assert post(http, "/lifecycle", dict(
        delivery_id=row.id, evidence=claimed["evidence"], state="started",
    ))["recorded"]
    claims.expire(http.store, row)
    replacement = post(http, "/repair/claim", dict(bot_id=rig.bot, claim_owner="replacement"))["claim"]
    assert replacement["claim_token"] != claimed["claim_token"]
    assert replacement["evidence"] == claimed["evidence"]
    assert "framed_message" not in replacement
    new = dict(delivery_id=row.id, claim_token=replacement["claim_token"])
    assert not post(http, "/prepare", new | {"binding": asdict(http.binding)})["authorized"]
    assert not post(http, "/repair", old)["acknowledged"]
    assert post(http, "/repair", new)["acknowledged"]


def test_proven_rejection_reopens_via_existing_dispatcher_wake(rig, http):
    row = claims.send(rig, http.store)
    claimed = post(http, "/claim", http.body)["claim"]
    post(http, "/prepare", dict(
        delivery_id=row.id, claim_token=claimed["claim_token"], binding=asdict(http.binding),
    ))
    receipt = dict(delivery_id=row.id, evidence=claimed["evidence"], state="refused", proof="cli_terminal")
    assert post(http, "/lifecycle", receipt) == {"recorded": True}
    http.dispatcher.announce_admitted.assert_called_once()
    assert http.store.get(row.id).status == "QUEUED"


def test_transport_success_is_not_an_accepted_lifecycle_state(rig, http):
    row = claims.send(rig, http.store)
    claimed = post(http, "/claim", http.body)["claim"]
    response = http.client.post(PREFIX + "/lifecycle", json=dict(
        delivery_id=row.id, evidence=claimed["evidence"], state="sent",
    ))
    assert response.status_code == 422
    assert http.store.get(row.id).metadata[INBOX]["phase"] == "claimed"
    assert InboxSettlement(http.store).claim_repair(rig.bot, claim_owner="replacement") is None


def test_actual_consumer_accepts_once_then_replacement_repairs_through_http(rig, http):
    from agent_bridge.session_queue import SessionQueue
    from claude_code_bridge.inbox_consumer import BotInbox, InboxConsumer, InboxRun
    from tests.test_claude_inbox_input import InputClient

    row = claims.send(rig, http.store)
    client = InputClient()
    client.run.request_id = http.binding.request_id
    queue = SessionQueue()
    queue.set_active_client(http.binding.session_key, client.run)
    context = InboxRun(rig.bot, rig.user, http.binding.backend, http.binding.session_key,
                       client.run, rig.message_id, rig.session_id)

    async def rpc(path, body):
        return await asyncio.to_thread(post, http, path, body)

    async def scenario():
        consumer = InboxConsumer(rpc, queue, owner="original")
        consumer.bots[rig.bot] = BotInbox(runs=[context])
        client.run.on_command_lifecycle = lambda payload: consumer.observe(context, payload)
        http.memory.fail = True
        assert await consumer.drain(rig.bot)
        assert len(client.frames) == 1 and http.store.get(row.id).status == "STEERING"
        # Simulate process loss: all bridge-local operation state disappears.
        claims.expire(http.store, row)
        client.run.mark_completed()
        queue.pop_active_client(http.binding.session_key)
        replacement = InboxConsumer(rpc, queue, owner="replacement")
        replacement.bots[rig.bot] = BotInbox()
        assert not await replacement.drain(rig.bot)
        assert http.store.get(row.id).status == "DELIVERED"
        assert len(client.frames) == 1
        assert http.memory.calls[0] == http.memory.calls[1]

    asyncio.run(scenario())
