"""TASK-1489: freeze/thaw catalog rows and user-only approval."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from llm_bawt.approval_models import KIND_MCP, OPS_RUN_TOOL
from llm_bawt.ops.models import OpsOperation
from llm_bawt.ops.seeds import SEEDS
from llm_bawt.ops.store import OpsStore
from llm_bawt.service.approver_eligibility import approver_eligible

USER = "fixture-user"


class _Tags:
    available = True
    known_bots = frozenset({"fixture-requester"})

    def has(self, bot, tag):
        return False


def _card(slug, *, via="snapshot"):
    snapshot = json.dumps({"operation": {"slug": slug}}) if via == "snapshot" else None
    return SimpleNamespace(request_kind=KIND_MCP, tool_name=OPS_RUN_TOOL, user_id=USER,
                           bot_id="fixture-requester", operations_snapshot_json=snapshot,
                           tool_arguments_json=json.dumps({"operation": slug}))


def _eligible(row):
    return approver_eligible(holds=True, tags=_Tags(), row=row, approver="fixture-approver", user_id=USER)


@pytest.mark.parametrize("via", ["snapshot", "arguments"])
def test_freeze_cards_are_user_only(via):
    assert not _eligible(_card("llm-bawt.freeze", via=via))


@pytest.mark.parametrize("slug", ["llm-bawt.thaw", "llm-bawt.restart-app"])
def test_other_cards_stay_eligible(slug):
    assert _eligible(_card(slug))


def _seed(slug):
    return next(seed for seed in SEEDS if seed["slug"] == slug)


@pytest.mark.parametrize("slug", ["llm-bawt.freeze", "llm-bawt.thaw"])
def test_freeze_seeds_validate_and_ship_disabled(slug):
    seed = _seed(slug)
    assert seed["enabled"] is False and seed["executor_kind"] == "freeze"
    OpsStore._validate_op(OpsOperation(**OpsStore._clean_op(object.__new__(OpsStore), seed)))


def test_freeze_seed_requires_approval_and_is_high_risk():
    seed = _seed("llm-bawt.freeze")
    assert seed["approval_required"] is True and seed["risk_level"] == "high"
    assert set(json.loads(seed["args_schema_json"])["properties"]) == {"reason", "ttl_seconds", "admit_user"}


def test_freeze_spec_needs_the_freeze_executor():
    seed = {**_seed("llm-bawt.thaw"), "executor_kind": "docker", "execution_mode": "isolated"}
    store = object.__new__(OpsStore)
    with pytest.raises(ValueError, match="operator_freeze"):
        OpsStore._validate_op(OpsOperation(**OpsStore._clean_op(store, seed)))


@pytest.mark.parametrize("op", ["freeze", "thaw"])
def test_freeze_rows_pass_the_gpu_guard(op):
    """A freeze touches no container, so creating its job never asks Docker."""
    from llm_bawt.ops.gpu_guard import gpu_voice_target

    spec = {"action": "operator_freeze", "op": op}
    assert gpu_voice_target(spec, {}, executor=None) is None
    assert gpu_voice_target(spec, {}, executor=None, live=True) is None
