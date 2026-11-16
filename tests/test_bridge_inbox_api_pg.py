"""Internal HTTP inbox contract against private PostgreSQL, never a live SDK."""
import asyncio
import json
from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from sqlalchemy import text

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
    subscriber.turn_text_offset = AsyncMock(return_value=None)
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
                              gate=gate, framing=framing, body=body, memory=memory, subscriber=subscriber)


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
    ("/release", "release"),
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
        "release": operation | dict(evidence=claimed["evidence"],
                                    invocation_id="e08a77dc-101c-4ac5-b724-a37571b17fb2"),
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
    http.subscriber.turn_text_offset.return_value = 13
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
        http.subscriber.turn_text_offset.assert_awaited_once_with(rig.bot, rig.user, rig.turn_id)
        assert http.store.get(row.id).metadata[INBOX]["content_offset"] == 13
        # Simulate process loss: all bridge-local operation state disappears.
        http.subscriber.turn_text_offset.return_value = 999
        claims.expire(http.store, row)
        client.run.mark_completed()
        queue.pop_active_client(http.binding.session_key)
        replacement = InboxConsumer(rpc, queue, owner="replacement")
        replacement.bots[rig.bot] = BotInbox()
        assert not await replacement.drain(rig.bot)
        assert http.store.get(row.id).status == "DELIVERED"
        assert len(client.frames) == 1
        assert http.memory.calls[0] == http.memory.calls[1]
        http.subscriber.turn_text_offset.assert_awaited_once()
        with rig.engine.connect() as conn:
            anchor = conn.execute(text("""
                SELECT interrupt_source_message_id, interrupt_content_offset
                FROM messages WHERE bot_id=:bot AND id=:id
            """), {"bot": rig.bot, "id": row.user_message_id}).one()
        assert tuple(anchor) == (rig.message_id, 13)

    asyncio.run(scenario())


@pytest.mark.parametrize("offset", [None, 0, 23])
def test_prepare_freezes_offset_including_unknown_across_later_receipts(rig, http, offset):
    row = claims.send(rig, http.store)
    claimed = post(http, "/claim", http.body)["claim"]
    operation = dict(delivery_id=row.id, claim_token=claimed["claim_token"], binding=asdict(http.binding))
    http.subscriber.turn_text_offset.return_value = offset
    assert post(http, "/prepare", operation)["authorized"]
    http.subscriber.turn_text_offset.return_value = 999
    assert not post(http, "/prepare", operation)["authorized"]
    assert post(http, "/lifecycle", dict(
        delivery_id=row.id, evidence=claimed["evidence"], state="queued", content_offset=999,
    ))["recorded"]
    assert http.store.get(row.id).metadata[INBOX]["content_offset"] == offset
    repair = InboxSettlement(http.store).history_repair(row.id, claimed["claim_token"])
    assert repair.content_offset == offset


def test_prepare_revalidates_run_after_awaiting_offset(rig, http):
    row = claims.send(rig, http.store)
    claimed = post(http, "/claim", http.body)["claim"]

    async def sample(bot, user, turn):
        assert (bot, user, turn) == (rig.bot, rig.user, rig.turn_id)
        with rig.engine.begin() as conn:
            conn.execute(text("""
                UPDATE turn_logs SET agent_request_id='successor' WHERE id=:id
            """), {"id": rig.turn_id})
        return 13

    http.subscriber.turn_text_offset.side_effect = sample
    assert not post(http, "/prepare", dict(
        delivery_id=row.id, claim_token=claimed["claim_token"], binding=asdict(http.binding),
    ))["authorized"]
    marker = http.store.get(row.id).metadata[INBOX]
    assert marker["phase"] == "claimed" and "content_offset" not in marker


def test_unavailable_position_freezes_unknown_instead_of_later_receipt(rig, http):
    row = claims.send(rig, http.store)
    claimed = post(http, "/claim", http.body)["claim"]
    http.subscriber.turn_text_offset.side_effect = RuntimeError("fixture subscriber unavailable")
    assert post(http, "/prepare", dict(
        delivery_id=row.id, claim_token=claimed["claim_token"], binding=asdict(http.binding),
    ))["authorized"]
    assert post(http, "/lifecycle", dict(
        delivery_id=row.id, evidence=claimed["evidence"], state="started", content_offset=999,
    ))["recorded"]
    marker = http.store.get(row.id).metadata[INBOX]
    assert "content_offset" in marker and marker["content_offset"] is None


def release_operation(rig, http):
    row = claims.send(rig, http.store)
    claimed = post(http, "/claim", http.body)["claim"]
    operation = dict(
        delivery_id=row.id, claim_token=claimed["claim_token"], evidence=claimed["evidence"],
        invocation_id="e08a77dc-101c-4ac5-b724-a37571b17fb2",
    )
    assert post(http, "/prepare", dict(
        delivery_id=row.id, claim_token=claimed["claim_token"], binding=asdict(http.binding),
        invocation_id=operation["invocation_id"],
    ))["authorized"]
    return row, operation


def test_only_authorized_invocation_can_release_once_and_advance_generation(rig, http):
    row, operation = release_operation(rig, http)
    assert post(http, "/release", operation) == {"released": True, "retry": False}
    reopened = http.store.get(row.id)
    assert reopened.status == "QUEUED" and reopened.metadata["steer_generation"] == 1
    assert reopened.metadata[INBOX] is None
    assert not reopened.claim_token and reopened.transport_accepted_at is None
    assert post(http, "/release", operation) == {"released": False, "retry": False}
    assert http.store.get(row.id).metadata["steer_generation"] == 1
    http.dispatcher.announce_admitted.assert_called_once()
    assert not post(http, "/lifecycle", dict(
        delivery_id=row.id, evidence=operation["evidence"], state="started",
    ))["recorded"]


@pytest.mark.parametrize("foreign", [
    "invocation_id", "claim_token", "write_token", "generation", "command_uuid", "binding",
])
def test_release_rejects_stale_invocation_claim_and_full_original_proof(rig, http, foreign):
    row, operation = release_operation(rig, http)
    wrong = deepcopy(operation)
    if foreign in {"invocation_id", "claim_token"}:
        wrong[foreign] = "23a5af98-013f-4ae6-94c6-54c2f69e3713"
    elif foreign == "binding":
        wrong["evidence"]["binding"]["session_id"] = "foreign"
    elif foreign == "generation":
        wrong["evidence"][foreign] += 1
    else:
        wrong["evidence"][foreign] = "foreign"
    assert not post(http, "/release", wrong)["released"]
    assert http.store.get(row.id).metadata[INBOX]["phase"] == "writing"
    assert post(http, "/release", operation)["released"]


@pytest.mark.parametrize("state", ["queued", "started", "uncertain"])
def test_lifecycle_settlement_wins_over_late_not_written_release(rig, http, state):
    row, operation = release_operation(rig, http)
    assert post(http, "/lifecycle", dict(
        delivery_id=row.id, evidence=operation["evidence"], state=state,
    ))["recorded"]
    assert post(http, "/release", operation) == {"released": False, "retry": False}
    assert http.store.get(row.id).status == "STEERING"


@pytest.mark.parametrize("ttl", [0, 600])
def test_local_stop_release_waits_for_durable_audit_and_enabled_hold(rig, http, monkeypatch, ttl):
    from llm_bawt import runtime_setting_resolution
    from llm_bawt.service.user_stop_hold import open_or_extend

    row, operation = release_operation(rig, http)
    operation["stop_requested"] = True
    monkeypatch.setattr(runtime_setting_resolution, "resolve_global_runtime_setting", lambda *a, **kw: ttl)
    assert post(http, "/release", operation) == {"released": False, "retry": True}
    with rig.engine.begin() as conn:
        conn.execute(text("UPDATE turn_logs SET request_json=:request WHERE id=:id"), {
            "id": rig.turn_id, "request": json.dumps({"abort_request": {"source": "chat_stop"}}),
        })
    if ttl:
        assert post(http, "/release", operation) == {"released": False, "retry": True}
        with rig.engine.begin() as conn:
            open_or_extend(conn, rig.bot, ttl)
    assert post(http, "/release", operation) == {"released": True, "retry": False}
    assert http.store.get(row.id).status == "QUEUED"


@pytest.mark.parametrize("hold_state", ["resumed", "expired", "unrelated", "wrong_bot", "late"])
def test_offline_stop_proof_can_release_after_its_hold_ended(rig, http, monkeypatch, hold_state):
    from llm_bawt import runtime_setting_resolution
    from llm_bawt.service.user_stop_hold import hold_owner, open_or_extend

    row, operation = release_operation(rig, http)
    operation["stop_requested"] = True
    monkeypatch.setattr(runtime_setting_resolution, "resolve_global_runtime_setting", lambda *a, **kw: 600)
    with rig.engine.begin() as conn:
        open_or_extend(conn, rig.bot, 600)
        stamp = conn.execute(text("SELECT now()-interval '5 minutes'")).scalar_one()
        conn.execute(text("""
            UPDATE system_gates SET
                owner_id=CASE WHEN :wrong_bot THEN :other_owner ELSE owner_id END,
                created_at=CASE WHEN :late THEN now()-interval '4 minutes'
                                ELSE now()-interval '10 minutes' END,
                expires_at=CASE WHEN :resumed THEN now()+interval '1 hour'
                                ELSE now()-interval '1 minute' END,
                released_at=CASE WHEN :unrelated THEN now()-interval '6 minutes'
                                 WHEN :resumed THEN now()-interval '1 minute' ELSE NULL END
            WHERE owner_id=:owner
        """), {"owner": hold_owner(rig.bot), "resumed": hold_state == "resumed",
               "unrelated": hold_state == "unrelated", "wrong_bot": hold_state == "wrong_bot",
               "other_owner": hold_owner("unrelated-fixture"), "late": hold_state == "late"})
        conn.execute(text("UPDATE turn_logs SET request_json=:request WHERE id=:id"), {
            "id": rig.turn_id, "request": json.dumps({
                "abort_request": {"source": "chat_stop", "requested_at": stamp.isoformat()},
            }),
        })
    released = post(http, "/release", operation)["released"]
    assert released == (hold_state in {"resumed", "expired"})
    assert http.store.get(row.id).status == ("QUEUED" if released else "STEERING")


def test_actual_consumer_authorized_handoff_releases_without_sdk_input(rig, http):
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
    paths = []

    async def rpc(path, body):
        paths.append(path)
        response = await asyncio.to_thread(post, http, path, body)
        if path == "/prepare":
            assert response["authorized"]
            queue.set_active_client(http.binding.session_key, object())
        return response

    async def scenario():
        consumer = InboxConsumer(rpc, queue, owner="fixture")
        consumer.bots[rig.bot] = BotInbox(runs=[context])
        assert await consumer.drain(rig.bot)
        assert not client.frames
        assert paths.index("/prepare") < paths.index("/release")
        assert "/lifecycle" not in paths and not consumer.bots[rig.bot].releases
        assert http.store.get(row.id).status == "QUEUED"
        assert http.store.get(row.id).metadata["steer_generation"] == 1

    asyncio.run(scenario())
