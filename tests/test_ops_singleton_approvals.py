"""Singleton replacement uses isolated approval rows, never real release cards."""
from copy import deepcopy
import json

import pytest

from llm_bawt.approval_models import (
    ApprovalDuplicatePending, CONT_PENDING, EXEC_SKIPPED, REQ_APPROVED, REQ_PENDING,
    REQ_SUPERSEDED,
)
from llm_bawt.approval_ops_supersede import singleton_target
from tests.test_approval_mcp_state import _mcp_req, _store


def snapshot():
    return {
        "operation": {"slug": "fixture.release", "version": 1},
        "spec": {"action": "release_orchestrate", "deploy_operation": "fixture.deploy"},
        "resolved_args": {"expected_sha": "old", "release_task": "TASK-fixture"},
        "execution": {"executor_kind": "release", "target_host": "fixture", "max_concurrent": 1},
    }


def record(store, number, *, snap=None, **kwargs):
    return store.record_mcp_request(**_mcp_req(
        request_id=f"singleton-{number}", tool_use_id=f"toolu_singleton-{number}",
        invocation_hash=f"{number:064x}", bot_id="fixture-owner", user_id="fixture-user",
        session_key="fixture-owner:fixture-user",
        operations_snapshot=snap if snap is not None else snapshot(), **kwargs,
    ))


def test_new_invocation_supersedes_old_pending_card_and_only_one_is_counted():
    store = _store()
    first = record(store, 1)
    new = snapshot()
    new["resolved_args"]["expected_sha"] = "new"
    new["resolved_args"]["release_task"] = "TASK-new"
    second = record(store, 2, snap=new)
    old = store.get_request(first.id)
    assert old.status == REQ_SUPERSEDED
    assert old.execution_state == EXEC_SKIPPED
    assert old.continuation_state == CONT_PENDING
    assert old.result_is_error is False
    assert json.loads(old.result_json)["superseded_by"] == second.id
    assert second.id in old.resolution_message
    assert store.count_requests(status=REQ_PENDING, user_id="fixture-user") == 1
    assert store.get_request(second.id).status == REQ_PENDING
    assert store.find_admittable_ops_requests() == []
    store.resolve_request(first.id, status=REQ_APPROVED)
    assert store.get_request(first.id).status == REQ_SUPERSEDED
    assert store.claim_mcp_execution(first.id) is None


def test_retry_of_existing_card_never_supersedes_its_successor():
    store = _store()
    record(store, 1)
    record(store, 2)
    record(store, 1)
    assert store.get_request("singleton-2").status == REQ_PENDING
    assert store.get_request("singleton-1").status == REQ_SUPERSEDED


def test_exact_duplicate_still_refuses_without_mutating_pending_card():
    store = _store()
    record(store, 1)
    with pytest.raises(ApprovalDuplicatePending):
        store.record_mcp_request(**_mcp_req(
            request_id="duplicate", invocation_hash=f"{1:064x}",
            bot_id="fixture-owner", user_id="fixture-user",
            session_key="fixture-owner:fixture-user", operations_snapshot=snapshot(),
        ))
    assert store.get_request("singleton-1").status == REQ_PENDING


def test_approved_card_is_not_replaced():
    store = _store()
    first = record(store, 1)
    store.resolve_request(first.id, status=REQ_APPROVED)
    record(store, 2)
    assert store.get_request(first.id).status == REQ_APPROVED


@pytest.mark.parametrize("change", ["not_singleton", "different_target", "different_op", "unknown_action"])
def test_nonexclusive_or_different_targets_do_not_replace(change):
    store = _store()
    first = snapshot()
    new = deepcopy(first)
    if change == "not_singleton":
        first["execution"]["max_concurrent"] = new["execution"]["max_concurrent"] = None
    elif change == "different_target":
        new["spec"]["deploy_operation"] = "fixture.other-deploy"
    elif change == "different_op":
        new["operation"]["slug"] = "fixture.other-release"
    else:
        first["spec"]["action"] = new["spec"]["action"] = "unknown"
    record(store, 1, snap=first)
    record(store, 2, snap=new)
    assert store.count_requests(status=REQ_PENDING) == 2


def test_other_user_and_non_ops_tool_are_never_replaced():
    store = _store()
    store.record_mcp_request(**_mcp_req(
        request_id="other-user", user_id="fixture-other-user", operations_snapshot=snapshot(),
    ))
    record(store, 1, tool_name="fixture_tool")
    record(store, 2)
    assert store.count_requests(status=REQ_PENDING) == 3


def test_different_requesters_same_user_target_replace_pending_card():
    store = _store()
    record(store, 1)
    store.record_mcp_request(**_mcp_req(
        request_id="other-requester", bot_id="fixture-other", user_id="fixture-user",
        operations_snapshot=snapshot(),
    ))
    assert store.get_request("singleton-1").status == REQ_SUPERSEDED


def test_restart_targets_use_resolved_argument_not_invocation_identity():
    first = snapshot()
    first["spec"] = {"action": "restart", "compose_project": "fixture", "compose_service_from_arg": "service"}
    first["resolved_args"] = {"service": "one"}
    second = deepcopy(first)
    second["resolved_args"]["service"] = "two"
    assert singleton_target(first) != singleton_target(second)
    store = _store()
    record(store, 1, snap=first)
    record(store, 2, snap=second)
    record(store, 3, snap=first)
    assert store.get_request("singleton-1").status == REQ_SUPERSEDED
    assert store.get_request("singleton-2").status == REQ_PENDING
    assert store.get_request("singleton-3").status == REQ_PENDING


def test_superseded_outbox_result_repairs_original_tool_card():
    from llm_bawt.service.approval_continuations import _admission_card_result

    store = _store()
    record(store, 1)
    record(store, 2)
    old = store.get_request("singleton-1")
    assert _admission_card_result(old)["superseded_by"] == "singleton-2"
    assert old.id in [row.id for row in store.find_pending_continuations()]
    from llm_bawt.approval_ops_supersede import superseded_by

    assert [row.id for row in superseded_by(store, "singleton-2")] == [old.id]
    assert superseded_by(store, "not-the-successor") == []


def test_failed_new_insert_rolls_back_its_supersession(monkeypatch):
    from llm_bawt import approval_ops_supersede as replacements
    from llm_bawt.approval_models import ApprovalPersistError

    store = _store()
    record(store, 1)
    real = replacements.supersede_pending

    def fail_after_supersession(*args, **kwargs):
        real(*args, **kwargs)
        raise RuntimeError("fixture failure before commit")

    monkeypatch.setattr(replacements, "supersede_pending", fail_after_supersession)
    with pytest.raises(ApprovalPersistError):
        record(store, 2)
    assert store.get_request("singleton-1").status == REQ_PENDING
    assert store.get_request("singleton-2") is None


@pytest.mark.parametrize("value", [None, [], {"execution": []}, {"execution": {"max_concurrent": True}}])
def test_malformed_or_unknown_snapshots_do_not_opt_into_replacement(value):
    assert singleton_target(value) is None


def test_withdraw_tool_forwards_only_request_id_and_reason(monkeypatch):
    import asyncio
    from llm_bawt.mcp_server import bot_grant_tools as tools

    calls = []

    async def forward(*args, **kwargs):
        calls.append((args, kwargs))
        return "fixture-result"

    monkeypatch.setattr(tools, "_forward", forward)
    assert asyncio.run(tools.ops_approval_withdraw(" request/id ", "no longer needed")) == "fixture-result"
    assert calls[0][0] == (
        "POST", "/v1/ops-approvals/request%2Fid/withdraw", {"reason": "no longer needed"})
    assert "invalid_request" in asyncio.run(tools.ops_approval_withdraw("", "reason"))
    assert len(calls) == 1
