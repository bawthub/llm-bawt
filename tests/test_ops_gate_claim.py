"""DB-owned gate grant schema and inert snapshot; no production admission."""
from __future__ import annotations

import hashlib
import uuid

import pytest
from cryptography.fernet import Fernet
from sqlalchemy import inspect, text

from llm_bawt import task_turn_context as codec
from llm_bawt.ops.gate_claim import gate_claim_snapshot
from llm_bawt.ops.gate_plan import GatePlan
from llm_bawt.ops.service import OpsService
from llm_bawt.ops.validation import canonical_json
from llm_bawt.service.routes.ops import OperationUpsert
from tests.test_ops_service import FakeExecutor, _op_data, _store


@pytest.fixture
def token(monkeypatch):
    fernet = Fernet(Fernet.generate_key())
    monkeypatch.setattr(codec, "_get_fernet", lambda: fernet)
    return codec.mint_task_turn_context(session_id=str(uuid.uuid4()), turn_id="turn-" + uuid.uuid4().hex,
        trigger_message_id=str(uuid.uuid4()), bot_id="fixture-gate", user_id="fixture-user")


def test_additive_migration_leaves_legacy_rows_unconfigured_and_is_idempotent():
    store = _store()
    old = store.create_operation(_op_data())
    before = old.to_api()
    with store.engine.begin() as conn:
        conn.execute(text("ALTER TABLE ops_operations DROP COLUMN execution_tag"))
        store._migrate_add_columns(conn)
        column = next(c for c in inspect(conn).get_columns("ops_operations") if c["name"] == "execution_tag")
        assert column["nullable"] and column["default"] is None
        store._migrate_add_columns(conn)
    assert store.get_operation(old.id).to_api() == before
    store.update_operation(old.id, {"execution_tag": "fixture_gate"})
    with store.engine.begin() as conn:
        store._migrate_add_columns(conn)
    assert store.get_operation(old.id).execution_tag == "fixture_gate"


def test_execution_tag_roundtrips_through_api_revisions_and_snapshot_hash(token):
    store = _store()
    op = store.create_operation(_op_data(execution_tag="fixture_gate"))
    assert OperationUpsert(execution_tag="fixture_gate").writable() == {"execution_tag": "fixture_gate"}
    assert OperationUpsert(execution_tag=None).writable() == {"execution_tag": None}
    service = OpsService(store, executor=FakeExecutor())
    normal = service.prepare_invocation(op.slug, {"target": "a"})
    assert normal["operation"]["execution_tag"] == "fixture_gate"
    plan = GatePlan("llm-bawt", "a" * 40)
    first = gate_claim_snapshot(op, plan, turn_token=token)
    body = {k: v for k, v in first.items() if k != "snapshot_hash"}
    assert first["snapshot_hash"] == hashlib.sha256(canonical_json(body).encode()).hexdigest()
    updated = store.update_operation(op.id, {"execution_tag": "fixture_other"})
    second = gate_claim_snapshot(updated, plan, turn_token=token)
    assert second["snapshot_hash"] != first["snapshot_hash"]
    assert updated.version == op.version + 1 and updated.script_hash == op.script_hash
    revisions, total = store.list_revisions(op.slug)
    assert total == 2 and '"execution_tag":"fixture_other"' in revisions[0].snapshot_json
    assert service.prepare_invocation(op.slug, {"target": "a"})["snapshot_hash"] != normal["snapshot_hash"]


@pytest.mark.parametrize("bad", ["", " ", "APPROVE", "a-b", "a" * 65, True, 12, ["fixture_gate"]])
def test_catalog_rejects_noncanonical_execution_tags(bad):
    with pytest.raises(ValueError, match="execution_tag"):
        _store().create_operation(_op_data(execution_tag=bad))


def test_legacy_nongate_dispatch_is_unchanged_even_with_unconfigured_gate_grant():
    store = _store()
    op = store.create_operation(_op_data())
    executor = FakeExecutor()
    service = OpsService(store, executor=executor)
    snapshot = service.prepare_invocation(op.slug, {"target": "a"})
    # A pre-column approved snapshot remains usable by the ordinary path.
    snapshot["operation"].pop("execution_tag")
    snapshot.pop("snapshot_hash")
    snapshot["snapshot_hash"] = hashlib.sha256(canonical_json(snapshot).encode()).hexdigest()
    result = service.dispatch_job(operation_slug=op.slug, args={"target": "a"}, approved_snapshot=snapshot)
    assert result["state"] == "accepted" and len(executor.dispatch_calls) == 1


def test_gate_claim_requires_postgresql_and_has_no_required_tag_override(token):
    store = _store()
    assert store.claim_gate_job("missing", turn_token=token) is False
    with pytest.raises(TypeError):
        store.claim_gate_job("missing", turn_token=token, required_tag="fixture_gate")
    with pytest.raises(ValueError):
        gate_claim_snapshot(store.create_operation(_op_data()), GatePlan("llm-bawt", "a" * 40), turn_token="forged")
