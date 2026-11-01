"""Create-only gate admission: signed provenance, immutable replay, no launch."""
from __future__ import annotations

import json
import uuid
from types import SimpleNamespace

import pytest
from cryptography.fernet import Fernet
from sqlalchemy import text

from llm_bawt import task_turn_context as codec
from llm_bawt.ops.gate_admission import GATE_REQUEST_SCHEMA, create_gate_job
from llm_bawt.ops.gate_claim import GATE_CLAIM_SCHEMA
from llm_bawt.ops.gate_history import plan_key
from llm_bawt.ops.gate_plan import GatePlan
from llm_bawt.ops.models import JOB_TERMINAL_STATES
from llm_bawt.ops.service import OpsService
from llm_bawt.ops.store import IdempotencyConflict
from tests.test_ops_service import FakeExecutor, _op_data, _store


@pytest.fixture
def admission(monkeypatch):
    fernet = Fernet(Fernet.generate_key())
    monkeypatch.setattr(codec, "_get_fernet", lambda: fernet)
    values = dict(session_id=str(uuid.uuid4()), turn_id="turn-" + uuid.uuid4().hex,
        trigger_message_id=str(uuid.uuid4()), bot_id="fixture-gate", user_id="fixture-user")
    store = _store()
    operation = store.create_operation(_op_data(execution_tag="fixture_gate"))
    plan = GatePlan("llm-bawt", "a" * 40)
    defaults = dict(operation_slug=operation.slug, plan=plan.to_dict(),
        turn_token=codec.mint_task_turn_context(**values), idempotency_key="fixture-key")

    def create(**changes):
        return create_gate_job(store, **{**defaults, **changes})

    yield SimpleNamespace(store=store, operation=operation, plan=plan, values=values,
        defaults=defaults, create=create)
    store.engine.dispose()


def test_creates_one_queued_request_with_only_signed_provenance_and_no_authority(admission, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("admission must not claim execution")

    monkeypatch.setattr(admission.store, "claim_job", forbidden)
    monkeypatch.setattr(admission.store, "claim_gate_job", forbidden)
    job = admission.create()
    assert job.state == "queued" and job.dispatched_at is None
    assert job.worker_slot_at is None and job.host_unit_name is None
    assert job.receipt_callback_json is None and job.approval_request_id is None
    assert job.caller_actor == "bot:fixture-gate"
    assert (job.caller_bot_id, job.caller_user_id, job.caller_turn_id) == (
        admission.values["bot_id"], admission.values["user_id"], admission.values["turn_id"])
    request = json.loads(job.request_payload_json)
    assert request == {"schema": GATE_REQUEST_SCHEMA, "operation": admission.operation.slug,
        "gate_plan": admission.plan.to_dict(), "caller": admission.values}
    snapshot = json.loads(job.invocation_snapshot_json)
    assert snapshot["gate_claim_schema"] == GATE_CLAIM_SCHEMA
    assert snapshot["caller"]["session_id"] == admission.values["session_id"]
    assert "spec" not in snapshot
    assert job.gate_plan_key == plan_key(admission.plan)
    assert json.loads(job.args_json) == json.loads(job.display_args_json) == admission.plan.to_dict()
    assert admission.defaults["turn_token"] not in json.dumps(job.model_dump(), default=str)
    assert "issued_at" not in job.request_payload_json
    executor = FakeExecutor()
    service = OpsService(admission.store, executor=executor)
    assert service.get_job_status(job.id)["state"] == "queued"
    assert executor.dispatch_calls == executor.reconcile_calls == []


@pytest.mark.parametrize("state", ["queued", *sorted(JOB_TERMINAL_STATES)])
def test_replay_is_read_only_even_after_catalog_change_and_completion(admission, monkeypatch, state):
    job = admission.create()
    if state != "queued":
        admission.store.mark_terminal(job.id, state=state)
    admission.store.update_operation(admission.operation.id, {"enabled": False, "execution_tag": None})
    before = admission.store.get_job(job.id).model_dump()

    def forbidden(*args, **kwargs):
        pytest.fail("a replay must not rebuild, insert, claim or attach anything")

    monkeypatch.setattr(admission.store, "get_operation_by_slug", forbidden)
    monkeypatch.setattr(admission.store, "create_job", forbidden)
    monkeypatch.setattr(admission.store, "claim_gate_job", forbidden)
    # A fresh opaque token for the same signed turn is the same request.
    token = codec.mint_task_turn_context(**admission.values, issued_at=1)
    assert admission.create(turn_token=token).model_dump() == before


@pytest.mark.parametrize("field,value", [("bot_id", "fixture-peer"), ("user_id", "fixture-peer"),
    ("turn_id", "turn-" + "b" * 32), ("session_id", "00000000-0000-0000-0000-000000000001"),
    ("trigger_message_id", "00000000-0000-0000-0000-000000000002")])
def test_key_is_bound_to_every_signed_caller_field(admission, field, value):
    job = admission.create()
    before = job.model_dump()
    token = codec.mint_task_turn_context(**{**admission.values, field: value})
    with pytest.raises(IdempotencyConflict):
        admission.create(turn_token=token)
    assert admission.store.get_job(job.id).model_dump() == before
    assert admission.store.count_jobs() == 1


@pytest.mark.parametrize("change", [{"operation_slug": "fixture.other"},
    {"plan": GatePlan("llm-bawt", "b" * 40).to_dict()}])
def test_key_cannot_change_operation_or_workload(admission, change):
    job = admission.create()
    with pytest.raises(IdempotencyConflict):
        admission.create(**change)
    assert admission.store.get_job(job.id).model_dump() == job.model_dump()
    assert admission.store.count_jobs() == 1


@pytest.mark.parametrize("token", ["", None, "fixture-gate", "forged"])
def test_invalid_capability_refuses_before_any_insert_or_key_read(admission, monkeypatch, token):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid identity must not reach the ledger")

    monkeypatch.setattr(admission.store, "get_job_by_key", forbidden)
    with pytest.raises(codec.TaskTurnContextError):
        admission.create(turn_token=token)
    assert admission.store.count_jobs() == 0


def test_expired_capability_cannot_create_or_read_a_request(admission):
    job = admission.create()
    raw = json.dumps({**admission.values, "issued_at": 1}).encode()
    expired = codec._get_fernet().encrypt_at_time(raw, current_time=1).decode()
    with pytest.raises(codec.TaskTurnContextError):
        admission.create(turn_token=expired)
    assert admission.store.get_job(job.id).model_dump() == job.model_dump()


@pytest.mark.parametrize("change", [{"repository": "bawthub"}, {"source_sha": "HEAD"},
    {"target": "unit"}, {"attempt": 2}, {"required_tag": "fixture_gate"},
    {"caller_bot_id": "fixture-peer"}, {"callback": {"bot_id": "fixture-peer"}}])
def test_nonclosed_workloads_refuse_without_creation(admission, change):
    with pytest.raises(ValueError):
        admission.create(plan={**admission.plan.to_dict(), **change})
    assert admission.store.count_jobs() == 0


@pytest.mark.parametrize("key", [None, True, 1, "", "   ", "x" * 129])
def test_idempotency_key_is_required_and_bounded(admission, key):
    with pytest.raises(ValueError, match="idempotency_key"):
        admission.create(idempotency_key=key)
    assert admission.store.count_jobs() == 0


@pytest.mark.parametrize("changes", [{"enabled": False}, {"execution_tag": None},
    {"quiesce_enabled": True}])
def test_unavailable_configuration_refuses_new_admission(admission, changes):
    admission.store.update_operation(admission.operation.id, changes)
    with pytest.raises(ValueError, match="gate operation"):
        admission.create()
    assert admission.store.count_jobs() == 0


def test_unknown_operation_refuses_new_admission(admission):
    with pytest.raises(ValueError, match="not configured"):
        admission.create(operation_slug="fixture.absent")
    assert admission.store.count_jobs() == 0


def test_caller_columns_drifting_from_request_cannot_replay(admission):
    job = admission.create()
    with admission.store.engine.begin() as conn:
        conn.execute(text("UPDATE ops_jobs SET caller_bot_id='fixture-peer' WHERE id=:id"), {"id": job.id})
    with pytest.raises(IdempotencyConflict, match="caller"):
        admission.create()
