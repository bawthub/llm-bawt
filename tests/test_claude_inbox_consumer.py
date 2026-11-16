"""Deterministic active-client inbox consumer, with no provider or bot traffic."""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agent_bridge.session_queue import SessionQueue
from claude_code_bridge.inbox_consumer import BotInbox, InboxConsumer, InboxRun
from claude_code_bridge.send_request import SendRequest
from tests.test_claude_inbox_input import InputClient
from tests.test_claude_active_run_next_steer import BOT


class API:
    def __init__(self, operation):
        self.operation, self.calls = operation, []
        self.before_claim = self.before_prepare = None
        self.history_ok = True
        self.accepted = False
        self.lifecycle_error = None
        self.recovery = None

    async def __call__(self, path, body):
        self.calls.append((path, deepcopy(body)))
        if path == "/claim":
            if self.before_claim:
                self.before_claim()
            operation, self.operation = self.operation, None
            return {"claim": operation}
        if path == "/prepare":
            if self.before_prepare:
                self.before_prepare()
            return {"authorized": True}
        if path == "/lifecycle":
            if self.lifecycle_error:
                raise self.lifecycle_error
            self.accepted = body["state"] in {"queued", "started", "completed"}
            return {"recorded": True}
        if path == "/repair":
            return {"acknowledged": self.accepted and self.history_ok}
        if path == "/repair/claim":
            recovery, self.recovery = self.recovery, None
            return {"claim": recovery}
        raise AssertionError(path)


def rig(states=("queued",)):
    client = InputClient(states=states)
    queue = SessionQueue()
    queue.set_active_client("target:operator", client.run)
    context = InboxRun("target", "operator", "claude-code", "target:operator",
                       client.run, "trigger", "thread")
    binding = context.claim_body("unused")
    binding.pop("claim_owner")
    binding.update(turn_id="target-turn", trigger_message_id="trigger", session_id="thread")
    operation = dict(
        delivery_id="delivery-" + BOT.replace("-", ""), claim_token="claim-token",
        evidence=dict(binding=binding, generation=0, command_uuid=BOT, write_token="claim-token"),
        steer_rpc_id="steer_delivery_" + BOT.replace("-", ""),
        framed_message="NEXT[callback]", interrupt_item="ITEM[callback]",
    )
    api = API(operation)
    consumer = InboxConsumer(api, queue, owner="fixture")
    inbox = consumer.bots["target"] = BotInbox(runs=[context])
    events = []
    client.run.on_command_lifecycle = lambda payload: events.append(consumer.observe(context, payload))
    return SimpleNamespace(client=client, queue=queue, context=context, operation=operation,
                           api=api, consumer=consumer, inbox=inbox, events=events)


@pytest.mark.anyio
async def test_running_consumer_accepts_once_then_repairs_history_without_reinput():
    r = rig()
    r.api.history_ok = False
    r.api.before_prepare = lambda: pytest.fail("input lock held") if r.client.run._steer_lock.locked() else None
    assert await r.consumer.drain("target")
    assert len(r.client.frames) == 1 and r.inbox.repairs
    assert r.events[0]["inbox_write"]["evidence"] == r.operation["evidence"]
    r.api.history_ok = True
    assert not await r.consumer.drain("target")
    assert not r.inbox.repairs and len(r.client.frames) == 1
    r.client.interrupt.assert_not_awaited()
    paths = [path for path, _ in r.api.calls]
    assert paths.index("/prepare") < paths.index("/lifecycle") < paths.index("/repair")


@pytest.mark.anyio
@pytest.mark.parametrize("when", ["claim", "prepare"])
async def test_owner_handoff_never_inputs_into_a_stale_active_client(when):
    r = rig()
    setattr(r.api, f"before_{when}", lambda: r.queue.set_active_client("target:operator", object()))
    assert await r.consumer.drain("target")
    assert r.client.frames == []
    assert not any(path == "/lifecycle" for path, _ in r.api.calls)


@pytest.mark.anyio
@pytest.mark.parametrize("blocked", ["stopped", "finishing", "ambiguous", "missing"])
async def test_only_one_actual_eligible_run_can_request_a_claim(blocked):
    r = rig()
    if blocked == "stopped":
        r.queue.tombstone(r.context.session_key, r.client.run.request_id)
    elif blocked == "finishing":
        r.client.run._finishing = True
    elif blocked == "missing":
        r.queue.pop_active_client(r.context.session_key)
    else:
        r.inbox.runs.append(InboxRun("target", "operator", "claude-code", "other:operator",
                                    r.client.run, "trigger", "thread"))
        r.queue.set_active_client("other:operator", r.client.run)
    assert not await r.consumer.drain("target")
    assert not any(path == "/claim" for path, _ in r.api.calls)
    assert r.client.frames == []


@pytest.mark.anyio
async def test_no_input_replay_after_lost_durable_prepare_response():
    r = rig()

    def lost_response():
        raise TimeoutError("WRITING response lost")

    r.api.before_prepare = lost_response
    with pytest.raises(TimeoutError, match="WRITING"):
        await r.consumer.drain("target")
    assert not await r.consumer.drain("target")
    assert r.client.frames == []
    assert sum(path == "/prepare" for path, _ in r.api.calls) == 1


@pytest.mark.anyio
async def test_failed_lifecycle_transport_retries_proof_not_sdk_input():
    r = rig()
    r.api.lifecycle_error = TimeoutError("app offline")
    with pytest.raises(TimeoutError, match="offline"):
        await r.consumer.drain("target")
    assert len(r.client.frames) == 1 and r.inbox.receipts
    r.api.lifecycle_error = None
    assert not await r.consumer.drain("target")
    assert not r.inbox.receipts and not r.inbox.repairs
    assert len(r.client.frames) == 1


@pytest.mark.anyio
async def test_late_start_and_history_repair_survive_run_deregistration():
    r = rig()
    r.api.history_ok = False
    await r.consumer.drain("target")
    r.client.run.mark_completed()
    r.queue.pop_active_client(r.context.session_key)
    r.consumer.detach(r.context)
    r.client.run.note_lifecycle(BOT, "started")
    r.api.history_ok = True
    assert not await r.consumer.drain("target")
    assert any(body.get("state") == "started" for path, body in r.api.calls if path == "/lifecycle")
    assert len(r.client.frames) == 1 and not r.inbox.repairs


@pytest.mark.anyio
async def test_replacement_owner_repairs_existing_acceptance_without_an_active_run():
    r = rig()
    r.inbox.runs.clear()
    r.api.accepted = True
    r.api.recovery = r.operation
    assert not await r.consumer.drain("target")
    assert not r.inbox.repairs
    assert not any(path in {"/claim", "/prepare"} for path, _ in r.api.calls)
    assert r.client.frames == []


@pytest.mark.anyio
async def test_actor_attachment_is_single_flight_and_deregistration_wakes():
    r = rig()
    r.inbox.runs.clear()
    r.consumer.drain = AsyncMock(return_value=False)
    r.consumer.attach(r.context)
    task = r.inbox.task
    r.consumer.attach(r.context)
    assert r.inbox.task is task and len(r.inbox.runs) == 1
    r.inbox.wake.clear()
    r.consumer.detach(r.context)
    assert r.inbox.wake.is_set()
    await r.consumer.stop()
    assert task.done()


def test_legacy_send_does_not_enable_inbox_and_only_explicit_one_does():
    fields = dict(request_id="r", session_key="b:u", message="hi", model="fixture")
    assert not SendRequest.from_fields(fields).inbox_next
    assert not SendRequest.from_fields(fields | {"inbox_next": "true"}).inbox_next
    assert SendRequest.from_fields(fields | {"inbox_next": "1"}).inbox_next
    explicit = SendRequest.from_fields(fields | {"session_key": "opaque-key", "user_id": "operator"})
    assert explicit.user_id == "operator"
