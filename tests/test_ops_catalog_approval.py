"""Catalog approval metadata preserves the old catch-all independently of risk."""
from __future__ import annotations

import pytest
from pydantic import ValidationError
from sqlalchemy import text

from llm_bawt.service.routes.ops import OperationUpsert
from tests.test_ops_store import _op_data, _store


@pytest.mark.parametrize("risk", ["low", "medium", "high", "critical"])
def test_catalog_approval_defaults_required_at_every_risk(risk):
    store = _store()
    row = store.create_operation(_op_data(risk_level=risk))
    assert row.approval_required is True
    assert row.to_api()["approval_required"] is True
    revisions, _ = store.list_revisions(row.slug)
    assert revisions[0].to_api()["operation"]["approval_required"] is True


def test_explicit_catalog_free_setting_is_versioned_and_preserved():
    store = _store()
    row = store.create_operation(_op_data(approval_required=False, risk_level="high"))
    assert row.to_api()["approval_required"] is False
    updated = store.update_operation(row.id, {"approval_required": True})
    assert updated.version == row.version + 1
    assert updated.approval_required is True
    revisions, _ = store.list_revisions(row.slug)
    assert [item.to_api()["operation"]["approval_required"] for item in revisions] == [True, False]


@pytest.mark.parametrize("value", [None, "false", 0, 1])
def test_catalog_store_refuses_nonboolean_approval_setting(value):
    store = _store()
    with pytest.raises(ValueError, match="approval_required must be boolean"):
        store.create_operation(_op_data(approval_required=value))


@pytest.mark.parametrize("value", ["false", 0, 1])
def test_catalog_api_refuses_coerced_approval_setting(value):
    with pytest.raises(ValidationError):
        OperationUpsert(approval_required=value)


def test_catalog_migration_defaults_existing_rows_without_rewriting_settings():
    store = _store()
    row = store.create_operation(_op_data())
    before = row.to_api()
    with store.engine.begin() as connection:
        connection.execute(text("ALTER TABLE ops_operations DROP COLUMN approval_required"))
        store._migrate_add_columns(connection)
    after = store.get_operation(row.id).to_api()
    assert after == before
    assert after["approval_required"] is True
    # Repeated bootstrap must not undo a later explicit approval-free setting.
    store.update_operation(row.id, {"approval_required": False})
    with store.engine.begin() as connection:
        store._migrate_add_columns(connection)
    assert store.get_operation(row.id).approval_required is False


def test_direct_admission_refuses_required_catalog_even_for_low_risk():
    from llm_bawt.ops.service import OpsDispatchError
    from tests.test_ops_service import _make

    store = _store()
    store.create_operation(_op_data(risk_level="low"))
    service, executor = _make(store)
    with pytest.raises(OpsDispatchError) as error:
        service.dispatch_job(operation_slug="test.restart-thing", args={},
                             require_catalog_free=True)
    assert error.value.code == "approval_required"
    assert not store.list_jobs() and not executor.dispatch_calls


def test_free_catalog_flag_change_is_fenced_under_job_insertion(monkeypatch):
    from llm_bawt.ops.service import OpsDispatchError
    from tests.test_ops_service import _make

    store = _store()
    row = store.create_operation(_op_data(approval_required=False))
    service, executor = _make(store)
    insert = store.create_job

    def race(**kwargs):
        store.update_operation(row.id, {"approval_required": True})
        return insert(**kwargs)

    monkeypatch.setattr(store, "create_job", race)
    with pytest.raises(OpsDispatchError) as error:
        service.dispatch_job(operation_slug=row.slug, args={}, require_catalog_free=True)
    assert error.value.code == "operation_changed"
    assert not store.list_jobs() and not executor.dispatch_calls


def test_free_catalog_existing_key_replay_stays_a_read_after_setting_change():
    from tests.test_ops_service import _make

    store = _store()
    row = store.create_operation(_op_data(approval_required=False))
    service, executor = _make(store)
    first = service.dispatch_job(operation_slug=row.slug, args={}, idempotency_key="stable",
                                 require_catalog_free=True)
    store.update_operation(row.id, {"approval_required": True})
    retry = service.dispatch_job(operation_slug=row.slug, args={}, idempotency_key="stable",
                                 require_catalog_free=True)
    assert retry["id"] == first["id"] and len(executor.dispatch_calls) == 1


@pytest.mark.parametrize("required,action", [(True, "require_approval"), (False, "allow")])
def test_ops_preview_uses_catalog_not_draft_policy_and_redacts_secrets(monkeypatch, required, action):
    from llm_bawt.service.routes import approval_policies, ops
    from tests.test_ops_service import _make

    store = _store()
    row = store.create_operation(_op_data(
        approval_required=required, risk_level="critical",
        approval_prompt_prefix="A catalog-owned prompt",
        args_schema_json={"type": "object", "properties": {
            "password": {"type": "string", "x-sensitive": True}}}))
    service, _ = _make(store)
    monkeypatch.setattr(ops, "_service", lambda: service)
    result = approval_policies.preview_policy(approval_policies.PolicyPreview(
        backend="claude-code", tool_name="ops_run",
        tool_input={"operation": row.slug, "args": {"password": "synthetic-secret"}},
        policies=[{"tool_name": "ops_run", "action": "deny"}],
    ))
    assert result["action"] == action and result["severity"] == "critical"
    assert result["policy_id"] is None and result["prompt"] == row.approval_prompt_prefix
    assert "synthetic-secret" not in str(result)
    assert not store.list_jobs()


@pytest.mark.parametrize("tool", ["ops_run", "mcp__bawthub__ops_run"])
def test_ops_policies_cannot_be_authored(tool):
    from llm_bawt.approval_policy_store import ToolApprovalPolicyStore

    store = ToolApprovalPolicyStore(None, engine=_store().engine)
    with pytest.raises(ValueError, match="not policy-evaluated"):
        store.create({"tool_name": tool, "action": "allow"})


def test_existing_ops_policy_can_be_retired_without_being_reseeded():
    from sqlmodel import Session
    from llm_bawt.approval_models import ToolApprovalPolicy
    from llm_bawt.approval_policy_store import ToolApprovalPolicyStore

    store = ToolApprovalPolicyStore(None, engine=_store().engine)
    # A pre-upgrade row, not a newly authored policy.
    with Session(store.engine) as session:
        session.add(ToolApprovalPolicy(id="seed-ops-run-require-approval",
                                      tool_name="ops_run", action="require_approval"))
        session.commit()
    assert store.delete("seed-ops-run-require-approval")
    store.seed_defaults()
    assert store.get("seed-ops-run-require-approval") is None
    assert store.list_revisions("seed-ops-run-require-approval")[0].change == "delete"
