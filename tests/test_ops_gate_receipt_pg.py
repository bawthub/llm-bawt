"""Dormant gate callback on real private-PG ops/delivery ledgers; no executor.

Jobs and destinations here are synthetic trusted-internal fixtures. This does
not enable gate admission, prove signed authorization or publish gate receipts.
"""
from __future__ import annotations

import json
import uuid
from types import SimpleNamespace

import pytest
from sqlalchemy import text

from llm_bawt.inter_bot_delivery import InterBotDeliveryStore
from llm_bawt.ops.gate_plan import GatePlan
from llm_bawt.ops.gate_receipt import DEFAULT_GATE_RECEIPT, gate_callback
from llm_bawt.ops.models import JOB_TERMINAL_STATES
from llm_bawt.ops.receipt_callback import QuiesceReceiptSender
from llm_bawt.ops.service import OpsService
from llm_bawt.ops.store import OpsStore
from llm_bawt.runtime_settings import BotProfileStore
from tests.test_ops_service import FakeExecutor, _op_data

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]


@pytest.fixture
def gate_rig(private_app_pg, monkeypatch):
    store = OpsStore(private_app_pg)
    BotProfileStore(private_app_pg)
    deliveries = InterBotDeliveryStore(private_app_pg)
    enqueue_calls = []
    original_enqueue = deliveries.enqueue

    def record_enqueue(**kwargs):
        enqueue_calls.append(kwargs)
        return original_enqueue(**kwargs)

    monkeypatch.setattr(deliveries, "enqueue", record_enqueue)
    bot = "test-gate-" + uuid.uuid4().hex[:12]
    with store.engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO bot_profiles (slug, name, description, system_prompt, requires_memory,
                voice_optimized, uses_tools, uses_search, uses_home_assistant, bot_type,
                agent_backend, created_at, updated_at)
            VALUES (:bot, :bot, '', '', false, false, false, false, false, 'agent',
                'claude-code', now(), now())
        """), {"bot": bot})
    # Disabled placeholder operation, never an executable gate catalog action.
    operation = store.create_operation(_op_data(slug="test.gate-" + uuid.uuid4().hex, enabled=False))
    executor = FakeExecutor()
    service = OpsService(store, executor=executor)
    plan = GatePlan("llm-bawt", "a" * 40)
    callback = gate_callback(plan, bot_id=bot, user_id="fixture-user", session_id="ended-session")
    monkeypatch.setattr("llm_bawt.ops.gate_receipt.gate_receipt_prompt", lambda _bot: DEFAULT_GATE_RECEIPT)

    def job(*, change=None, args=None, approved=False):
        payload = {**callback, **(change or {})}
        return store.create_job(operation=operation, args_json=json.dumps(args or plan.to_dict()),
            display_args_json="{}", idempotency_key=uuid.uuid4().hex,
            caller_bot_id=bot, caller_user_id="fixture-user", caller_turn_id="ended-turn",
            approval_request_id="fixture-approval" if approved else None,
            receipt_callback_json=json.dumps(payload))

    def sender(on_enqueued=None):
        # Reconstruct the service/store/sender: no originating turn or in-memory task.
        return QuiesceReceiptSender(OpsService(OpsStore(private_app_pg), executor=executor),
                                    deliveries, on_enqueued=on_enqueued)

    def rows():
        with store.engine.connect() as conn:
            return conn.execute(text("SELECT * FROM inter_bot_deliveries WHERE target_bot_id=:b"),
                                {"b": bot}).mappings().all()

    yield SimpleNamespace(store=store, service=service, executor=executor, plan=plan, callback=callback,
                          bot=bot, job=job, sender=sender, rows=rows, deliveries=deliveries,
                          enqueue_calls=enqueue_calls)
    with store.engine.begin() as conn:
        conn.execute(text("DELETE FROM inter_bot_deliveries WHERE target_bot_id=:b"), {"b": bot})
        conn.execute(text("DELETE FROM ops_jobs WHERE operation_slug=:s"), {"s": operation.slug})
        conn.execute(text("DELETE FROM bot_profiles WHERE slug=:b"), {"b": bot})


@pytest.mark.parametrize("state", sorted(JOB_TERMINAL_STATES))
def test_terminal_gate_wakes_a_new_turn_on_existing_durable_outbox(gate_rig, state):
    rig = gate_rig
    job = rig.job()
    assert rig.sender().deliver_due() == []  # Queued is not completion.
    rig.store.mark_terminal(job.id, state=state, exit_code=0 if state == "succeeded" else None)
    announced = []
    assert rig.sender(lambda row, duplicate: announced.append((row.id, duplicate))).deliver_due() == [job.id]
    [delivery] = rig.rows()
    payload = delivery["payload_json"]
    assert (delivery["sender_bot_id"], delivery["message_kind"]) == ("system", "OPS_RECEIPT")
    # The caller's ended session is never pinned; delivery uses the active one.
    assert (payload["bot_id"], payload["user"], payload["session_id"]) == (
        rig.bot, "fixture-user", None)
    assert payload["prefer_steer"] is False and payload["reply_mode"] == "async"
    assert rig.enqueue_calls[-1]["on_gate"] == "hold"
    assert announced == [(delivery["id"], False)]
    prefix, raw = delivery["message"].split("\n", 1)
    assert prefix == DEFAULT_GATE_RECEIPT
    receipt = json.loads(raw)
    assert receipt["id"] == job.id and receipt["state"] == state
    assert receipt["gate_plan"] == rig.plan.to_dict()
    assert "SHIP" in prefix and "not" in prefix  # wake != authority
    assert rig.sender().deliver_due() == []
    assert rig.executor.dispatch_calls == []  # No gate runner or generic command.
    assert rig.store.get_job(job.id).receipt_callback_delivery_id == delivery["id"]


def test_restart_after_enqueue_before_stamp_reuses_delivery(gate_rig, monkeypatch):
    rig = gate_rig
    job = rig.job()
    rig.store.mark_terminal(job.id, state="lost", error_text="worker disappeared; outcome unknown")
    enqueue = rig.deliveries.enqueue

    def crashed_after_commit(**kwargs):
        enqueue(**kwargs)
        raise RuntimeError("process died before job stamp")

    monkeypatch.setattr(rig.deliveries, "enqueue", crashed_after_commit)
    with pytest.raises(RuntimeError, match="before job stamp"):
        rig.sender().deliver(job.id)
    assert rig.store.get_job(job.id).receipt_callback_delivery_id is None
    [original] = rig.rows()
    monkeypatch.setattr(rig.deliveries, "enqueue", enqueue)
    announced = []
    assert rig.sender(lambda row, duplicate: announced.append(duplicate)).deliver_due() == [job.id]
    [same] = rig.rows()
    assert same["id"] == original["id"] and announced == [True]
    # A late success cannot replace LOST and callbacks do not launch a retry.
    rig.store.mark_terminal(job.id, state="succeeded", exit_code=0)
    assert rig.store.get_job(job.id).state == "lost"
    assert rig.sender().deliver_due() == [] and rig.executor.dispatch_calls == []


@pytest.mark.parametrize("change", [{"bot_id": "different-recipient"}, {"user_id": "different-owner"},
    {"session_id": ""}, {"kind": "generic_command"}, {"extra": True},
    {"plan": GatePlan("llm-bawt", "b" * 40).to_dict()}])
def test_bad_stored_callback_never_enqueues(gate_rig, change):
    rig = gate_rig
    job = rig.job(change=change)
    rig.store.mark_terminal(job.id, state="failed")
    with pytest.raises(ValueError):
        rig.sender().deliver(job.id)
    assert rig.rows() == [] and rig.store.get_job(job.id).receipt_callback_delivery_id is None


def test_approved_gate_job_does_not_also_send_direct_callback(gate_rig):
    rig = gate_rig
    job = rig.job(approved=True)
    rig.store.mark_terminal(job.id, state="succeeded", exit_code=0)
    with pytest.raises(ValueError, match="approval continuation"):
        rig.sender().deliver(job.id)
    assert rig.rows() == []


def test_green_run_real_docker_frames_reach_pass_callback(gate_rig):
    # TASK-1865: real Docker non-TTY attach bytes (captured from the gate runner
    # image via DockerAPI.attach_stream; stdout+stderr frames, >64KiB) once failed
    # the terminal write with a NUL error and left a green run LOST.
    from pathlib import Path

    from llm_bawt.ops.gate_driver import _tail

    raw = (Path(__file__).parent / "fixtures" / "docker_attach_frames_gate.bin").read_bytes()
    assert b"\0" in raw and len(raw) > 65536
    rig = gate_rig
    job = rig.job()
    rig.store.mark_running(job.id)
    rig.store.mark_terminal(job.id, state="succeeded", exit_code=0, output_tail=_tail(raw))
    stored = rig.store.get_job(job.id)
    assert stored.state == "succeeded" and stored.exit_code == 0
    # Real Docker may deliver stderr frames after stdout; content, not order.
    assert "\0" not in stored.output_tail and "11563 passed, 2 warnings in 210.00s" in stored.output_tail
    assert "warn 1500" in stored.output_tail
    assert rig.sender().deliver_due() == [job.id]
    [delivery] = rig.rows()
    receipt = json.loads(delivery["message"].split("\n", 1)[1])
    assert receipt["state"] == "succeeded" and receipt["exit_code"] == 0
    assert receipt["gate_plan"] == rig.plan.to_dict()
    assert "11563 passed" in receipt["output_tail"] and "\0" not in receipt["output_tail"]
