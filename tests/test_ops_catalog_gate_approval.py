"""Catalog approval fences on real signed gate admission; scratch SQLite only."""
from __future__ import annotations

import json

import pytest

from agent_bridge.mcp_call_context import canonical_invocation_hash
from llm_bawt.ops import gate_driver, gate_trust, invocation_attestation
from llm_bawt.ops.approval_request import request_invocation
from llm_bawt.ops.store import OperationChanged
from tests import test_ops_gate_trust as trust_helpers
from tests.test_ops_invocation_attestation import CALLER
from tests.test_ops_service import _make

admission = trust_helpers.admission
trusted = trust_helpers.trusted


def _admit(trusted, key="catalog-gate"):
    base = trusted.base
    return gate_driver.admit_gate(
        trusted.store, operation_slug=base.operation.slug, args=base.plan.to_dict(),
        turn_token=base.defaults["turn_token"], idempotency_key=key)


def test_required_gate_cannot_create_an_unapproved_job(trusted):
    trusted.store.update_operation(trusted.base.operation.id, {"approval_required": True})
    with pytest.raises(gate_trust.GateTrustError, match="approval_required"):
        _admit(trusted)
    assert trusted.store.count_jobs() == 0


def test_catalog_allow_then_flag_flip_refuses_gate_admission(trusted, monkeypatch):
    operation = trusted.store.update_operation(
        trusted.base.operation.id, {"command_script": '{"action":"batch_gate"}'})
    service, executor = _make(trusted.store)
    monkeypatch.setattr(invocation_attestation, "control_token", lambda: "synthetic-control")
    args = {"operation": operation.slug, "args": trusted.base.plan.to_dict()}
    envelope = invocation_attestation.mint(
        args, CALLER, canonical_invocation_hash("ops_run", args))

    class NoCards:
        def get_request(self, request_id):
            return None

    assert request_invocation(service, NoCards(), envelope) == {"status": "catalog_allow"}
    trusted.store.update_operation(operation.id, {"approval_required": True})
    with pytest.raises(gate_trust.GateTrustError, match="approval_required"):
        _admit(trusted)
    assert trusted.store.count_jobs() == 0
    assert not executor.dispatch_calls


def test_flag_flip_after_snapshot_read_refuses_under_store_revision_lock(trusted, monkeypatch):
    insert = trusted.store.create_job

    def race(**kwargs):
        trusted.store.update_operation(trusted.base.operation.id, {"approval_required": True})
        return insert(**kwargs)

    monkeypatch.setattr(trusted.store, "create_job", race)
    with pytest.raises(OperationChanged):
        _admit(trusted)
    assert trusted.store.count_jobs() == 0


@pytest.mark.parametrize("terminal", [False, True])
def test_existing_key_replay_stays_read_only_after_approval_flag_change(trusted, monkeypatch, terminal):
    first = _admit(trusted)
    if terminal:
        trusted.store.mark_terminal(first["id"], state="succeeded")
    trusted.store.update_operation(trusted.base.operation.id, {"approval_required": True})
    before = trusted.store.get_job(first["id"])

    def forbidden(*args, **kwargs):
        pytest.fail("replay must not read the catalog, insert, dispatch or attach a callback")

    monkeypatch.setattr(trusted.store, "get_operation_by_slug", forbidden)
    monkeypatch.setattr(trusted.store, "create_job", forbidden)
    monkeypatch.setattr(gate_trust, "_configured_environment", forbidden)
    assert _admit(trusted)["id"] == first["id"]
    after = trusted.store.get_job(first["id"])
    assert after.model_dump() == before.model_dump()
    assert json.loads(after.invocation_snapshot_json)["operation"]["approval_required"] is False
    assert after.approval_request_id is None
    assert trusted.store.count_jobs() == 1
