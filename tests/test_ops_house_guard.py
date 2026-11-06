"""TASK-2100: the approval-gated manual house restart and its occupancy guard."""
from __future__ import annotations

import json

import pytest
from sqlalchemy import text
from sqlalchemy.pool import StaticPool
from sqlmodel import create_engine

from llm_bawt.approval_defaults import _OPS_DEFAULT_POLICIES
from llm_bawt.ops import JOB_FAILED, OpsDispatchError, OpsService
from llm_bawt.ops.executor import validate_spec
from llm_bawt.ops.house_guard import HouseOccupancyUnverifiable, house_target, read_occupancy
from llm_bawt.ops.seeds import SEEDS
from llm_bawt.ops.validation import validate_args
from llm_bawt.service.routes.ops import DISPATCH_ERROR_STATUS

from tests.test_ops_service import FakeExecutor, _store

SEED = next(s for s in SEEDS if s["slug"] == "llm-bawt.restart-house")
SPEC = validate_spec(SEED["command_script"])


def _args(**over):
    return validate_args(over, SEED["args_schema_json"], SEED["args_defaults_json"])


# ── target recognition ────────────────────────────────────────────────


def test_the_row_targets_only_house_services():
    assert house_target(SPEC, _args(house="house-h1")) == "h1"
    for bad in ("claude-code-bridge", "codex-bridge", "app", "house-", "house-H1", "xhouse-h1",
                "house-h1 ", "house-../x"):
        with pytest.raises(Exception):
            _args(house=bad)


def test_house_containers_and_services_are_recognized_everywhere():
    named = {"action": "restart", "container_name_from_arg": "c"}
    assert house_target(named, {"c": "llm-bawt-house-h2"}) == "h2"
    assert house_target(named, {"c": "llm-bawt-claude-code-bridge"}) is None
    assert house_target({"action": "stop", "container_name": "llm-bawt-house-h1"}, {}) == "h1"
    assert house_target({"action": "pull", "container_name": "llm-bawt-house-h1"}, {}) is None
    assert house_target({"action": "start", "compose_project": "llm-bawt",
                         "compose_service": "house-dev3"}, {}) == "dev3"


def test_seed_shape_and_approval_policy():
    assert SEED["enabled"] is False and SPEC["compose_service_from_arg"] == "house"
    assert SPEC["health_url"] == "http://127.0.0.1:8681/health"
    assert _args(house="house-h1") == {"house": "house-h1", "override_occupied": False}
    policy = next(p for p in _OPS_DEFAULT_POLICIES if p["id"] == "seed-ops-run-house-restart")
    assert policy["action"] == "require_approval" and policy["pattern"] == "operation=llm-bawt.restart-house "
    assert DISPATCH_ERROR_STATUS["house_occupied"] == 409


# ── occupancy reading ─────────────────────────────────────────────────


class FakeRedis:
    def __init__(self, *, groups=None, turns=None, fail=False):
        self.groups = groups if groups is not None else {}
        self.turns = turns or {}
        self.fail = fail

    def exists(self, key):
        if self.fail:
            raise ConnectionError("down")
        return key in self.groups

    def xinfo_groups(self, key):
        return self.groups[key]

    def scan_iter(self, pattern, count=None):
        return iter(self.turns)

    def get(self, key):
        return self.turns.get(key)


def _profiles(*rows):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE bot_profiles (slug TEXT, house TEXT)"))
        for slug, house in rows:
            conn.execute(text("INSERT INTO bot_profiles VALUES (:s, :h)"), {"s": slug, "h": house})
    return engine


STREAM = "agent:commands:house:h1"


def test_a_vacant_house_is_free():
    redis = FakeRedis(groups={STREAM: [{"name": "claude-code-bridge", "lag": 0}]},
                      turns={"agent:turn:req:x": "agent:commands:house:h2", "agent:turn:req:y": "agent:commands"})
    assert read_occupancy(_profiles(("nova", "h2"), ("byte", "")), redis, "h1") == []


def test_tenant_running_turn_and_unread_send_each_occupy():
    engine = _profiles(("probe", "h1"))
    assert read_occupancy(engine, FakeRedis(), "h1") == ["assigned to probe"]
    turns = FakeRedis(turns={"agent:turn:req:x": STREAM})
    assert read_occupancy(_profiles(), turns, "h1") == ["1 turn(s) running"]
    lag = FakeRedis(groups={STREAM: [{"name": "g", "lag": 2}]})
    assert read_occupancy(_profiles(), lag, "h1") == ["2 unread send(s) on its stream"]
    unknown = FakeRedis(groups={STREAM: [{"name": "g", "lag": None}]})
    assert read_occupancy(_profiles(), unknown, "h1") == ["1 unread send(s) on its stream"]


def test_unreadable_occupancy_fails_closed():
    with pytest.raises(HouseOccupancyUnverifiable):
        read_occupancy(_profiles(), FakeRedis(fail=True), "h1")
    broken = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    with pytest.raises(HouseOccupancyUnverifiable):
        read_occupancy(broken, FakeRedis(), "h1")  # no bot_profiles table


# ── the service: admission, approval-only, dispatch recheck ───────────


class Occupancy:
    def __init__(self):
        self.reasons: list[str] = []
        self.error = False
        self.calls = 0

    def __call__(self, house):
        self.calls += 1
        if self.error:
            raise HouseOccupancyUnverifiable("redis down")
        return list(self.reasons)


def _service():
    store = _store()
    row = {k: v for k, v in SEED.items() if k in {
        "slug", "title", "description", "executor_kind", "target_host", "working_directory",
        "command_script", "args_schema_json", "args_defaults_json", "timeout_seconds",
        "start_delay_seconds", "max_concurrent", "risk_level", "category", "approval_prompt_prefix"}}
    store.create_operation({**row, "enabled": True})
    occupancy, executor = Occupancy(), FakeExecutor()
    return OpsService(store, executor=executor, house_occupancy=occupancy), store, executor, occupancy


def _approved(service, args, key="h-1"):
    snapshot = service.prepare_invocation(SEED["slug"], args)
    return service.dispatch_job(operation_slug=SEED["slug"], args=args, idempotency_key=key,
                                approved_snapshot=snapshot, create_fence=lambda _session: None,
                                approval_request_id="appr-1")


def test_an_unassigned_house_restarts_from_an_approved_card():
    service, store, executor, occupancy = _service()
    job = _approved(service, {"house": "house-h1"})
    assert occupancy.calls == 2  # card preparation and dispatch
    assert len(executor.dispatch_calls) == 1 and job["state"] != JOB_FAILED
    assert executor.dispatch_calls[0]["env_args"] == {"house": "house-h1", "override_occupied": False}


def test_direct_dispatch_is_refused_even_when_vacant():
    service, _store_, executor, _ = _service()
    with pytest.raises(OpsDispatchError) as exc:
        service.dispatch_job(operation_slug=SEED["slug"], args={"house": "house-h1"}, idempotency_key="d")
    assert exc.value.code == "approval_required" and not executor.dispatch_calls


def test_an_occupied_house_never_reaches_an_approver():
    service, _store_, executor, occupancy = _service()
    occupancy.reasons = ["assigned to probe"]
    with pytest.raises(OpsDispatchError) as exc:
        service.prepare_invocation(SEED["slug"], {"house": "house-h1"})
    assert exc.value.code == "house_occupied" and "assigned to probe" in str(exc.value)
    occupancy.error = True
    with pytest.raises(OpsDispatchError) as exc:
        service.prepare_invocation(SEED["slug"], {"house": "house-h1"})
    assert exc.value.code == "house_occupancy_unverifiable" and not executor.dispatch_calls


def test_a_tenant_arriving_after_approval_fails_the_job_unsubmitted():
    service, store, executor, occupancy = _service()
    args = {"house": "house-h1"}
    snapshot = service.prepare_invocation(SEED["slug"], args)
    occupancy.reasons = ["1 turn(s) running"]
    job = service.dispatch_job(operation_slug=SEED["slug"], args=args, idempotency_key="late",
                               approved_snapshot=snapshot, create_fence=lambda _session: None,
                               approval_request_id="appr-1")
    row = store.get_job(job["id"])
    assert row.state == JOB_FAILED and "house_occupied" in row.error_text and not executor.dispatch_calls


def test_unreadable_occupancy_at_dispatch_keeps_the_job_queued():
    service, store, executor, occupancy = _service()
    args = {"house": "house-h1"}
    snapshot = service.prepare_invocation(SEED["slug"], args)
    occupancy.error = True
    job = service.dispatch_job(operation_slug=SEED["slug"], args=args, idempotency_key="q",
                               approved_snapshot=snapshot, create_fence=lambda _session: None,
                               approval_request_id="appr-1")
    row = store.get_job(job["id"])
    assert row.state == "queued" and "house_occupancy_unverifiable" in (row.error_text or "")
    assert not executor.dispatch_calls


def test_the_approver_override_skips_the_occupancy_check():
    service, _store_, executor, occupancy = _service()
    occupancy.reasons = ["assigned to probe"]
    _approved(service, {"house": "house-h1", "override_occupied": True})
    assert occupancy.calls == 0 and len(executor.dispatch_calls) == 1
    snapshot = json.loads(json.dumps(executor.dispatch_calls[0]["snapshot"]))
    assert snapshot["resolved_args"]["override_occupied"] is True
