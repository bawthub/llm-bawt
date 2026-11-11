"""TASK-1263: real service admission and approval bindings, no live operations."""
from __future__ import annotations

import copy
import hashlib
import json
from types import SimpleNamespace

import pytest

from llm_bawt.ops import OpsDispatchError, OpsService
from llm_bawt.ops import sequence
from llm_bawt.ops.executor import DockerExecutor, ExecutorError, validate_spec
from llm_bawt.ops.restart_health import has_health
from tests.ops_install_rows import SEEDS, seed_all
from llm_bawt.ops.sequence_approval import approval_text
from llm_bawt.ops.worker import OPS_JOB_LABEL, target_selector
from tests.test_ops_image_deploy import _rehash
from tests.test_ops_service import FakeExecutor, _op_data, _store


SEED = next(row for row in SEEDS if row["slug"] == "llm-bawt.operation-sequence")
HEALTH = {
    "health_url": "http://127.0.0.1:8000/health",
    "health_field": "status", "health_value": "ok", "health_timeout_seconds": 30,
}
WORKER = "sha256:" + "a" * 64


def child(slug="test.first", *, name="first", action="restart", spec=None, **over):
    return _op_data(**{
        "slug": slug, "command_script": json.dumps(spec or {
            "action": action, "container_name": name, **HEALTH,
        }), "args_schema_json": '{"type":"object","additionalProperties":false,"properties":{}}',
        "start_delay_seconds": 2, **over,
    })


def step(operation="test.first", args=None):
    return {"operation": operation, "args": {} if args is None else args}


class AdmissionExecutor(FakeExecutor):
    """Read-only Docker boundary; binding uses the actual Docker selector code."""

    def __init__(self):
        super().__init__()
        self.containers = {}
        self.workers = {}
        self.checked = []
        self.preflighted = []
        self.capabilities = []
        self.capability_error = None
        self.preflight_error = None
        self.self_names = set()
        self.config_bindings = {}
        self.client = SimpleNamespace(containers=SimpleNamespace(
            get=self.containers.__getitem__, list=self.list_containers,
        ))

    def add(self, name, *, project="test", service=None):
        self.containers[name] = SimpleNamespace(
            name=name, id=hashlib.sha256(name.encode()).hexdigest(),
            labels={"com.docker.compose.project": project,
                    "com.docker.compose.service": service or name},
            reload=lambda: None,
        )

    def list_containers(self, *, all, filters):
        return [c for c in self.containers.values()
                if all_labels_match(c.labels, filters["label"])]

    def execution_settings(self, spec=None, execution_mode=None):
        return {"worker_image": self.workers.get((spec or {}).get("container_name"), WORKER)}

    def check_spec_capabilities(self, spec):
        self.capabilities.append(spec["action"])
        if self.capability_error:
            raise ExecutorError(self.capability_error)

    def check_target(self, spec, args):
        self.checked.append((spec, args))
        name, filters = target_selector(spec, args)
        if name is not None and name not in self.containers:
            raise ExecutorError("target container is missing")
        if filters and not self.list_containers(all=True, filters=filters):
            raise ExecutorError("target container is missing")

    def target_compose_identity(self, spec, args):
        name, _ = target_selector(spec, args)
        labels = self.containers[name].labels
        return labels["com.docker.compose.project"], labels["com.docker.compose.service"]

    def self_target(self, snapshot):
        if snapshot["spec"].get("container_name") in self.self_names:
            return "Compose service ops/reconciler"
        return None

    def preflight(self, snapshot):
        super().preflight(snapshot)
        self.preflighted.append(snapshot["operation"]["slug"])
        if self.preflight_error:
            raise ExecutorError(self.preflight_error)

    def sequence_target(self, snapshot):
        return DockerExecutor.sequence_target(self, snapshot)

    def config_approval(self, spec):
        return copy.deepcopy(self.config_bindings[spec["container_name"]])


def all_labels_match(labels, wanted):
    return all(labels.get(key) == value for key, value in
               (item.split("=", 1) for item in wanted))


def rig(children=None, *, parent=None, argument_properties=None, release_verifier=None):
    rows = children if children is not None else [
        child(), child("test.second", name="second"),
    ]
    store, executor = _store(), AdmissionExecutor()
    for row in rows:
        store.create_operation(row)
        spec = json.loads(row["command_script"])
        if spec.get("container_name"):
            executor.add(spec["container_name"])
    allowed = [row["slug"] for row in rows]
    schema = json.loads(SEED["args_schema_json"])
    properties = schema["properties"]["steps"]["items"]["properties"]
    properties["operation"]["enum"] = allowed
    properties["args"]["properties"] = argument_properties or {}
    data = {
        **SEED, "enabled": True,
        "command_script": json.dumps({"action": sequence.ACTION, "operations": allowed}),
        "args_schema_json": json.dumps(schema), **(parent or {}),
    }
    op = store.create_operation(data)
    return SimpleNamespace(
        store=store, executor=executor, parent=op,
        service=OpsService(store, executor=executor, house_occupancy=lambda _houses: {},
                           release_verifier=release_verifier),
    )


def prepare(r, steps=None):
    return r.service.prepare_invocation(r.parent.slug, {
        "steps": steps if steps is not None else [step()],
    })


def refused(r, steps=None, *, code="sequence_invalid", match=None):
    with pytest.raises(OpsDispatchError, match=match) as caught:
        prepare(r, steps)
    assert caught.value.code == code
    assert r.executor.dispatch_calls == []
    assert r.store.list_jobs() == []


def test_seed_is_disabled_critical_bounded_and_references_only_existing_eligible_rows():
    spec = validate_spec(SEED["command_script"])
    catalog = {row["slug"]: row for row in SEEDS}
    assert spec["action"] == sequence.ACTION
    assert SEED["enabled"] is False and SEED["risk_level"] == "critical"
    assert SEED["quiesce_enabled"] is True
    assert json.loads(SEED["quiesce_scope_json"]) == {"all_agents": True}
    schema = json.loads(SEED["args_schema_json"])
    steps = schema["properties"]["steps"]
    assert steps["minItems"] == 1 and steps["maxItems"] == sequence.MAX_STEPS == 6
    assert steps["items"]["properties"]["operation"]["enum"] == spec["operations"]
    budgets = []
    for slug in spec["operations"]:
        row = catalog[slug]
        child_spec = validate_spec(row["command_script"])
        assert row["enabled"] is False
        assert child_spec["action"] in sequence.SUPPORTED
        if child_spec["action"] in ("restart", "start"):
            assert has_health(child_spec)
        assert json.loads(row["args_schema_json"])["properties"] == {}
        budgets.append(row["start_delay_seconds"] + row["timeout_seconds"]
                       + child_spec["health_timeout_seconds"])
    assert SEED["timeout_seconds"] >= sum(sorted(budgets, reverse=True)[:sequence.MAX_STEPS])
    store = _store()
    seed_all(store)
    with pytest.raises(OpsDispatchError) as caught:
        OpsService(store, executor=AdmissionExecutor()).prepare_invocation(
            SEED["slug"], {"steps": [step(spec["operations"][0])]},
        )
    assert caught.value.code == "operation_disabled"
    # Seeding again preserves an operator's revision, including disabled state.
    store.update_operation(SEED["slug"], {"title": "Operator-owned"})
    _, skipped = seed_all(store)
    assert SEED["slug"] in skipped
    assert store.get_operation_by_slug(SEED["slug"]).title == "Operator-owned"
    assert store.get_operation_by_slug(SEED["slug"]).enabled is False


@pytest.mark.parametrize("action", ["restart", "start"])
def test_prepare_binds_order_targets_and_individually_verifiable_children(action):
    r = rig([child(action=action), child("test.second", name="second")])
    requests = [step("test.second"), step()]
    snapshot = prepare(r, requests)
    assert snapshot["quiesce"]["scope"] == {"all_agents": True}
    bound = snapshot["sequence"]["steps"]
    assert [item["snapshot"]["operation"]["slug"] for item in bound] == [
        "test.second", "test.first",
    ]
    assert [item["target"] for item in bound] == [
        {"id": r.executor.containers[name].id, "name": name} for name in ("second", "first")
    ]
    assert r.executor.preflighted == ["test.second", "test.first"]
    assert sequence.ACTION in r.executor.capabilities
    for item in bound:
        s = item["snapshot"]
        assert r.service._verify_snapshot(s, s["operation"]["slug"], s["input_args"]) == s
    assert r.service._verify_snapshot(snapshot, r.parent.slug, {"steps": requests}) == snapshot
    text = approval_text(snapshot, "Approve the window?")
    assert text.startswith("Approve the window?")
    assert text.index("[1] test.second") < text.index("[2] test.first")
    assert "quiesce held for the entire sequence" in text
    assert "Earlier successful restarts are not undone" in text
    assert "NEVER automatically retried" in text
    assert "health budget 30s" in text
    assert r.executor.dispatch_calls == [] and r.store.list_jobs() == []


def test_six_distinct_targets_are_admitted():
    r = rig([child(f"test.step-{i}", name=f"target-{i}") for i in range(6)])
    snapshot = prepare(r, [step(f"test.step-{i}") for i in range(6)])
    assert len(snapshot["sequence"]["steps"]) == sequence.MAX_STEPS


@pytest.mark.parametrize("steps", [
    [], [step()] * 7, "not-an-array", [{}], [{"operation": "test.first"}],
    [{"args": {}}], [step(args=[])], [{**step(), "extra": True}],
    [step("not.allowlisted")], [step(args={"unknown": True})],
])
def test_invalid_request_shape_and_unallowlisted_operations_fail_before_binding(steps):
    r = rig()
    refused(r, steps, code="args_invalid")
    assert r.executor.preflighted == []


def test_steps_are_required_and_top_level_arguments_are_closed():
    r = rig()
    for args in ({}, {"steps": [step()], "extra": True}):
        with pytest.raises(OpsDispatchError) as caught:
            r.service.prepare_invocation(r.parent.slug, args)
        assert caught.value.code == "args_invalid"
    assert r.executor.dispatch_calls == [] and r.store.list_jobs() == []


@pytest.mark.parametrize("over,match", [
    ({"risk_level": "high"}, "critical risk"),
    ({"quiesce_enabled": False}, "whole-window all-agent"),
    ({"quiesce_scope_json": '{"bots":["test-bot"]}'}, "whole-window all-agent"),
    ({"timeout_seconds": 91}, "parent timeout"),
])
def test_parent_requires_critical_risk_global_quiesce_and_total_child_budget(over, match):
    # Each child costs 2s delay + 60s execution + 30s health = 92s.
    refused(rig(parent=over), match=match)


def test_parent_timeout_must_cover_the_sum_not_just_each_child():
    refused(rig(parent={"timeout_seconds": 183}), [step(), step("test.second")],
            match="parent timeout")


def test_whole_window_quiesce_requires_a_fenced_worker():
    r = rig()
    r.executor._fenced = False
    refused(r, code="quiesce_unavailable")


@pytest.mark.parametrize("action", ["restart", "start"])
def test_healthless_children_are_refused(action):
    refused(rig([child(spec={"action": action, "container_name": "first"})]),
            match="bounded health check")


@pytest.mark.parametrize("action", ["stop", "pull", sequence.ACTION])
def test_unsupported_and_nested_children_are_refused_before_child_admission(action):
    spec = ({"action": action, "operations": ["test.first"]} if action == sequence.ACTION
            else {"action": action, "container_name": "first"})
    r = rig([child(spec=spec)])
    refused(r, match="unsupported sequence operation kind")
    assert r.executor.checked == [] and r.executor.preflighted == []


@pytest.mark.parametrize("slug", [
    "llm-bawt.cutover-plan", "bawthub.deploy-prod-image",
    "bawthub.reconcile-prod-config", "bawthub.reconcile-dev-frontend-config",
])
def test_other_catalog_plans_deploys_and_config_kinds_cannot_be_nested(slug):
    row = next(row for row in SEEDS if row["slug"] == slug)
    r = rig([{**row, "enabled": True}])
    refused(r, [step(slug)], match="unsupported sequence operation kind")
    assert r.executor.checked == [] and r.executor.preflighted == []


@pytest.mark.parametrize("state", ["disabled", "deleted", "missing"])
def test_child_must_be_available_in_the_current_catalog(state):
    r = rig()
    if state == "missing":
        r.store.get_operation_by_slug = lambda slug: (
            r.parent if slug == r.parent.slug else None
        )
    elif state == "disabled":
        r.store.update_operation("test.first", {"enabled": False})
    else:
        r.store.soft_delete_operation("test.first")
    refused(r, match="target operation unavailable")


def test_duplicate_targets_are_rejected_even_through_different_catalog_slugs():
    r = rig([child(), child("test.alias", name="first")])
    refused(r, [step(), step("test.alias")], match="duplicate")


def test_repeating_the_same_operation_and_target_is_rejected():
    refused(rig(), [step(), step()], match="duplicate")


@pytest.mark.parametrize("protected", ["self", "gpu", "house", "redis", "ops-worker", "different-worker"])
def test_protected_child_targets_are_refused(protected):
    name = "test-house-h1" if protected == "house" else "first"
    r = rig([child(name=name)])
    if protected == "self":
        r.executor.self_names.add(name)
    elif protected == "gpu":
        r.executor.add(name, project="bawthub", service="tts")
    elif protected == "redis":
        r.executor.add(name, project="llm-bawt", service="redis")
    elif protected == "ops-worker":
        r.executor.containers[name].labels[OPS_JOB_LABEL] = "existing-job"
    elif protected == "different-worker":
        r.executor.workers[name] = "sha256:" + "b" * 64
    refused(r)


@pytest.mark.parametrize("failure,code", [
    ("target", "target_unavailable"), ("capability", "sequence_invalid"),
    ("preflight", "sequence_invalid"),
])
def test_target_capability_and_child_preflight_refusals_create_no_job(failure, code):
    r = rig()
    if failure == "target":
        r.executor.containers.pop("first")
    elif failure == "capability":
        r.executor.capability_error = "worker lacks sequence capability"
    else:
        r.executor.preflight_error = "worker lacks restart health capability"
    refused(r, code=code)


def test_child_arguments_defaults_and_sensitive_display_use_normal_prepare():
    schema = {
        "type": "object", "additionalProperties": False, "required": ["target", "token"],
        "properties": {"target": {"type": "string", "enum": ["first", "second"]},
                       "token": {"type": "string", "x-sensitive": True}},
    }
    row = child(
        spec={"action": "restart", "container_name_from_arg": "target", **HEALTH},
        args_schema_json=json.dumps(schema), args_defaults_json='{"token":"private-default"}',
    )
    r = rig([row], argument_properties={
        "target": {"type": "string"}, "token": {"type": "string"},
    })
    r.executor.add("first")
    snapshot = prepare(r, [step(args={"target": "first"})])
    bound = snapshot["sequence"]["steps"][0]
    assert bound["snapshot"] == r.service.prepare_invocation(row["slug"], {"target": "first"})
    assert bound["snapshot"]["resolved_args"] == {"target": "first", "token": "private-default"}
    assert bound["target"]["name"] == "first"
    assert bound["display_args"] == {"target": "first", "token": "***"}
    assert "private-default" not in approval_text(snapshot, None)
    assert "private-default" not in json.dumps(sequence.initial_ledger(snapshot))
    # The parent accepts any string here; only the child's own enum rejects this.
    refused(r, [step(args={"target": "unlisted"})], code="args_invalid", match="not in enum")
    refused(r, [step()], code="args_invalid", match="required property 'target'")


@pytest.mark.parametrize("kind", ["app", "tagged"])
def test_config_children_keep_their_real_approval_bindings_and_subcards(kind):
    from llm_bawt.ops.app_config_reconcile import snapshot_approval_text
    from llm_bawt.ops.tagged_config_reconcile import approval_text as tagged_text
    from tests.test_ops_app_config_reconcile import (
        APP_OP, HEAD, SECRET, AppDocker, PublishedVerifier, binding as app_binding,
    )
    from tests.test_ops_tagged_config_reconcile import (
        SPEC as TAGGED_SPEC, CrawlDocker, binding as tagged_binding,
    )

    if kind == "app":
        row, config = APP_OP, app_binding(AppDocker())
    else:
        row = child("test.tagged", spec=TAGGED_SPEC)
        config = tagged_binding(CrawlDocker())
    r = rig([row], release_verifier=PublishedVerifier(HEAD))
    spec = json.loads(row["command_script"])
    r.executor.config_bindings[spec["container_name"]] = config
    requests = [step(row["slug"])]
    snapshot = prepare(r, requests)
    nested = snapshot["sequence"]["steps"][0]["snapshot"]
    assert nested["config"] == config
    assert nested == r.service.prepare_invocation(row["slug"], {})
    assert r.service._verify_snapshot(snapshot, r.parent.slug, {"steps": requests}) == snapshot
    subcard = (snapshot_approval_text(nested, None) if kind == "app"
               else tagged_text(spec, config, None))
    assert subcard in approval_text(snapshot, None)
    assert SECRET not in approval_text(snapshot, None)
    # Rehashing both wrappers cannot hide an invalid child configuration binding.
    nested["config"]["env"] = {"LEAK": SECRET}
    _rehash(nested)
    with pytest.raises(OpsDispatchError) as caught:
        r.service._verify_snapshot(_rehash(snapshot), r.parent.slug, {"steps": requests})
    assert caught.value.code == "snapshot_invalid"
    assert r.executor.dispatch_calls == [] and r.store.list_jobs() == []


def test_app_child_still_requires_its_source_to_be_published():
    from tests.test_ops_app_config_reconcile import (
        APP_OP, AppDocker, PublishedVerifier, binding,
    )

    r = rig([APP_OP], release_verifier=PublishedVerifier("f" * 40))
    spec = json.loads(APP_OP["command_script"])
    r.executor.config_bindings[spec["container_name"]] = binding(AppDocker())
    refused(r, [step(APP_OP["slug"])], code="source_not_published")


@pytest.mark.parametrize("tamper", [
    lambda s: s["sequence"]["steps"].reverse(),
    lambda s: s["sequence"]["steps"][0]["snapshot"]["execution"].update(timeout_seconds=1),
    lambda s: s["sequence"]["steps"][0]["snapshot"]["resolved_args"].update(extra=True),
    lambda s: s["sequence"]["steps"][1].update(target=s["sequence"]["steps"][0]["target"]),
    lambda s: s["quiesce"].update(scope={"bots": ["test-bot"]}),
])
def test_approval_verification_rejects_rehashed_order_child_and_quiesce_tampering(tamper):
    r = rig()
    requests = [step(), step("test.second")]
    snapshot = prepare(r, requests)
    tamper(snapshot)
    for nested in sequence.snapshots(snapshot):
        _rehash(nested)
    with pytest.raises(OpsDispatchError) as caught:
        r.service._verify_snapshot(_rehash(snapshot), r.parent.slug, {"steps": requests})
    assert caught.value.code == "snapshot_invalid"
    assert r.executor.dispatch_calls == [] and r.store.list_jobs() == []


def test_sequence_dispatch_requires_an_approval_before_preparing_or_creating_a_job():
    r = rig()
    with pytest.raises(OpsDispatchError) as caught:
        r.service.dispatch_job(operation_slug=r.parent.slug, args={"steps": [step()]},
                               idempotency_key="unapproved-sequence")
    assert caught.value.code == "approval_required"
    assert r.executor.checked == [] and r.executor.dispatch_calls == []
    assert r.store.list_jobs() == []
