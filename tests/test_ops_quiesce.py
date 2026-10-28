"""TASK-1233 C: quiesce operation settings, snapshot binding, the C5 job
lifecycle edges that don't need PostgreSQL, and the worker's start fence.

The driver itself is PostgreSQL-only: tests/test_ops_quiesce_pg.py.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import text

from llm_bawt.ops import JOB_FAILED, JOB_QUEUED, OpsDispatchError
from llm_bawt.ops.executor import DockerExecutor, Executor, ExecutorError, worker_request
from llm_bawt.ops.models import JOB_QUIESCING
from llm_bawt.ops.quiesce_settings import quiesce_binding, resolve_scope, validate_quiesce_settings
from llm_bawt.ops.seeds import SEEDS
from llm_bawt.ops.worker import QUIESCE_FENCE_LABEL, atomic_json, fence_expired, run
from llm_bawt.service.turn_gates import GateScope
from tests.test_ops_executor import JOB, FakeAPI, FakeContainer, setup
from tests.test_ops_service import _make, _op_data, _store

BRIDGE_MAP = {"claude-code-bridge": "claude-code", "codex-bridge": "codex"}
BRIDGE_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["bridge"],
                 "properties": {"bridge": {"type": "string", "enum": sorted(BRIDGE_MAP)}}}
FROM_ARG = json.dumps({"backend_from_arg": "bridge", "map": BRIDGE_MAP})


def _op(**over):
    fields = dict(quiesce_enabled=True, quiesce_timeout_seconds=600, quiesce_scope_json=None,
                  executor_kind="docker")
    fields.update(over)
    return SimpleNamespace(**fields)


# -- settings validation -------------------------------------------------------

@pytest.mark.parametrize("scope", [
    None, '{"all_agents": true}', '{"backends": ["claude-code"]}', '{"bots": ["byte"]}', FROM_ARG,
])
def test_valid_scopes(scope):
    validate_quiesce_settings(_op(quiesce_scope_json=scope), BRIDGE_SCHEMA)


@pytest.mark.parametrize("over,match", [
    ({"quiesce_enabled": 1}, "quiesce_enabled"),
    ({"quiesce_timeout_seconds": 0}, "quiesce_timeout_seconds"),
    ({"quiesce_timeout_seconds": 1801}, "quiesce_timeout_seconds"),
    ({"quiesce_timeout_seconds": True}, "quiesce_timeout_seconds"),
    ({"executor_kind": "release"}, "docker"),
    ({"quiesce_scope_json": "nope"}, "JSON"),
    ({"quiesce_scope_json": "[]"}, "object"),
    ({"quiesce_scope_json": "{}"}, "one of"),
    ({"quiesce_scope_json": '{"all_agents": false}'}, "all_agents"),
    ({"quiesce_scope_json": '{"backends": []}'}, "backends"),
    ({"quiesce_scope_json": '{"bots": [""]}'}, "bots"),
    ({"quiesce_scope_json": '{"bots": ["a"], "backends": ["b"]}'}, "one of"),
    ({"quiesce_scope_json": json.dumps({"backend_from_arg": "nope", "map": BRIDGE_MAP})}, "enum"),
    ({"quiesce_scope_json": json.dumps({"backend_from_arg": "bridge",
                                        "map": {"codex-bridge": "codex"}})}, "every value"),
    ({"quiesce_scope_json": json.dumps({"backend_from_arg": "bridge",
                                        "map": {**BRIDGE_MAP, "codex-bridge": ""}})}, "every value"),
])
def test_invalid_settings_rejected(over, match):
    with pytest.raises(ValueError, match=match):
        validate_quiesce_settings(_op(**over), BRIDGE_SCHEMA)


def test_disabled_settings_are_still_validated():
    with pytest.raises(ValueError):
        validate_quiesce_settings(_op(quiesce_enabled=False, quiesce_scope_json="{}"), BRIDGE_SCHEMA)
    # ...but a disabled release op is fine.
    validate_quiesce_settings(_op(quiesce_enabled=False, executor_kind="release"), BRIDGE_SCHEMA)


def test_backend_from_arg_must_be_required():
    schema = {**BRIDGE_SCHEMA, "required": []}
    with pytest.raises(ValueError, match="required"):
        validate_quiesce_settings(_op(quiesce_scope_json=FROM_ARG), schema)


def test_scope_resolution():
    assert resolve_scope(None, {}) == GateScope.every_agent()
    assert resolve_scope('{"all_agents": true}', {}) == GateScope.every_agent()
    assert resolve_scope('{"backends": ["Codex"]}', {}) == GateScope.on_backends("codex")
    assert resolve_scope('{"bots": ["byte"]}', {}) == GateScope.for_bots("byte")
    assert resolve_scope(FROM_ARG, {"bridge": "codex-bridge"}) == GateScope.on_backends("codex")
    with pytest.raises(ValueError, match="no backend"):
        resolve_scope(FROM_ARG, {"bridge": "other"})


def test_restart_bridge_seed_scopes_the_named_bridge_and_starts_disabled():
    seed = next(row for row in SEEDS if row["slug"] == "llm-bawt.restart-bridge")
    assert seed["quiesce_enabled"] is False
    schema = json.loads(seed["args_schema_json"])
    validate_quiesce_settings(_op(quiesce_scope_json=seed["quiesce_scope_json"]), schema)
    for bridge, backend in BRIDGE_MAP.items():
        assert resolve_scope(seed["quiesce_scope_json"], {"bridge": bridge}) == GateScope.on_backends(backend)


def test_store_rejects_bad_settings_and_round_trips_good_ones():
    store = _store()
    with pytest.raises(ValueError, match="quiesce"):
        store.create_operation(_op_data(quiesce_enabled=True, quiesce_scope_json="{}"))
    row = store.create_operation(_op_data(quiesce_enabled=True, quiesce_timeout_seconds=90,
                                          quiesce_scope_json='{"bots": ["byte"]}'))
    api = row.to_api()
    assert (api["quiesce_enabled"], api["quiesce_timeout_seconds"], api["quiesce_scope_json"]) == (
        True, 90, '{"bots": ["byte"]}')


def test_patch_route_edits_quiesce_settings_as_a_revision(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from llm_bawt.service.routes import ops as routes
    store = _store()
    store.create_operation(_op_data(executor_kind="docker"))
    monkeypatch.setattr(routes, "_store", lambda: store)
    app = FastAPI()
    app.include_router(routes.router)
    url = "/v1/ops/operations/test.restart-thing"
    scope = '{"backends": ["claude-code"]}'
    with TestClient(app) as client:
        edited = client.patch(url, json={"quiesce_timeout_seconds": 120, "quiesce_scope_json": scope,
                                         "actor": "operator"})
        assert edited.status_code == 200, edited.text
        body = edited.json()
        assert (body["quiesce_enabled"], body["quiesce_timeout_seconds"], body["quiesce_scope_json"]) == (
            False, 120, scope)
        revisions = client.get(f"{url}/revisions").json()
        assert revisions["total"] == 2
        assert client.patch(url, json={"quiesce_timeout_seconds": 0}).status_code == 400
        assert client.patch(url, json={"quiesce_scope_json": None}).json()["quiesce_scope_json"] is None
        assert client.get(url).json()["quiesce_timeout_seconds"] == 120


# -- snapshot binding ----------------------------------------------------------

def test_ordinary_snapshot_has_no_quiesce_entry():
    store = _store()
    store.create_operation(_op_data())
    service, _ = _make(store)
    snapshot = service.prepare_invocation("test.restart-thing", {"target": "a"})
    assert "quiesce" not in snapshot


def _quiesce_service():
    store = _store()
    store.create_operation(_op_data(quiesce_enabled=True, quiesce_timeout_seconds=120,
                                    quiesce_scope_json='{"bots": ["byte"]}'))
    return _make(store)


def test_quiesce_snapshot_binds_the_resolved_scope():
    service, _ = _quiesce_service()
    snapshot = service.prepare_invocation("test.restart-thing", {"target": "a"})
    assert snapshot["quiesce"] == {"enabled": True, "timeout_seconds": 120, "scope": {"bots": ["byte"]}}
    assert quiesce_binding(snapshot["operation"], snapshot["resolved_args"]) == snapshot["quiesce"]


def _rehash(snapshot):
    from llm_bawt.ops.validation import canonical_json
    body = {k: v for k, v in snapshot.items() if k != "snapshot_hash"}
    snapshot["snapshot_hash"] = hashlib.sha256(canonical_json(body).encode()).hexdigest()
    return snapshot


@pytest.mark.parametrize("tamper", [
    lambda s: s.pop("quiesce"),
    lambda s: s["quiesce"].update(timeout_seconds=1),
    lambda s: s["quiesce"].update(scope={"all_agents": True}),
    lambda s: s["operation"].update(quiesce_enabled=False),
])
def test_tampered_quiesce_binding_fails_verification(tamper):
    service, _ = _quiesce_service()
    snapshot = service.prepare_invocation("test.restart-thing", {"target": "a"})
    tamper(snapshot)
    with pytest.raises(OpsDispatchError) as exc:
        service._verify_snapshot(_rehash(snapshot), "test.restart-thing", {"target": "a"})
    assert exc.value.code == "snapshot_invalid"


def test_quiesce_requires_postgres():
    service, executor = _quiesce_service()
    with pytest.raises(OpsDispatchError) as exc:
        service.dispatch_job(operation_slug="test.restart-thing", args={"target": "a"},
                             idempotency_key="k1")
    assert exc.value.code == "quiesce_unavailable"
    assert executor.dispatch_calls == []
    assert service.store.list_jobs() == []


# -- C5: never claimed from QUEUED ---------------------------------------------

def _create(store, snapshot, state):
    op = store.get_operation_by_slug("test.restart-thing")
    from llm_bawt.ops.validation import canonical_json
    return store.create_job(operation=op, args_json="{}", display_args_json="{}",
                            idempotency_key=f"k-{state}", state=state,
                            invocation_snapshot_json=canonical_json(snapshot))


def test_create_job_state_follows_the_snapshot():
    service, _ = _quiesce_service()
    snapshot = service.prepare_invocation("test.restart-thing", {"target": "a"})
    with pytest.raises(ValueError, match="quiesc"):
        _create(service.store, snapshot, JOB_QUEUED)
    ordinary = {k: v for k, v in snapshot.items() if k != "quiesce"}
    with pytest.raises(ValueError, match="quiesc"):
        _create(service.store, ordinary, JOB_QUIESCING)
    with pytest.raises(ValueError, match="queued or quiescing"):
        _create(service.store, snapshot, "running")
    assert _create(service.store, snapshot, JOB_QUIESCING).state == JOB_QUIESCING


def test_a_queued_quiesce_job_is_never_claimed_and_fails_visibly():
    service, executor = _quiesce_service()
    snapshot = service.prepare_invocation("test.restart-thing", {"target": "a"})
    job = _create(service.store, snapshot, JOB_QUIESCING)
    with service.store.engine.begin() as conn:  # forge the state create_job refuses
        conn.execute(text("UPDATE ops_jobs SET state='queued' WHERE id=:id"), {"id": job.id})
    assert service.store.claim_job(job.id) is False
    service.get_job_status(job.id)
    fresh = service.store.get_job(job.id)
    assert fresh.state == JOB_FAILED and "no action submitted" in fresh.error_text
    assert executor.dispatch_calls == []


def test_without_a_driver_a_quiescing_job_is_only_reported():
    service, executor = _quiesce_service()
    snapshot = service.prepare_invocation("test.restart-thing", {"target": "a"})
    job = _create(service.store, snapshot, JOB_QUIESCING)
    assert service.get_job_status(job.id)["state"] == JOB_QUIESCING
    assert service.reconcile_active_jobs() == 1
    assert service.store.get_job(job.id).state == JOB_QUIESCING
    assert executor.dispatch_calls == [] and executor.reconcile_calls == []


# -- worker request and start fence ---------------------------------------------

def test_worker_request_is_unchanged_without_a_fence():
    assert worker_request("j", {"a": 1}) == {"job_id": "j", "snapshot": {"a": 1}}
    fence = {"generation": 7, "start_not_after": "2026-01-01T00:00:00+00:00"}
    assert worker_request("j", {"a": 1}, fence)["fence"] == fence


def test_fenced_dispatch_and_reconcile_agree_on_the_request(tmp_path, monkeypatch):
    executor, client, snapshot = setup(tmp_path, monkeypatch)
    later = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    fence = {"generation": 3, "start_not_after": later}
    executor.dispatch(job_id=JOB, snapshot=snapshot, fence=fence)
    request = tmp_path / JOB / "request.json"
    assert json.loads(request.read_text())["fence"] == fence
    digest = hashlib.sha256(request.read_bytes()).hexdigest()
    FakeAPI.calls = 0
    assert run(request, digest, api_factory=FakeAPI, sleep=lambda _s: None) == 0
    assert executor.reconcile(job_id=JOB, snapshot=snapshot, fence=fence).state == "succeeded"
    # Without the fence the reconciler names a different request: never a match.
    with pytest.raises(ExecutorError, match="identity conflict"):
        executor.reconcile(job_id=JOB, snapshot=snapshot)


class _ExitedRace(FakeContainer):
    """Docker reports exit; the terminal receipt lands between the two reads."""

    def __init__(self, labels, publish):
        super().__init__(labels, status="exited")
        self._publish = publish

    def reload(self):
        self._publish()


def test_exited_race_reread_keeps_the_fence(tmp_path, monkeypatch):
    executor, client, snapshot = setup(tmp_path, monkeypatch)
    later = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    fence = {"generation": 3, "start_not_after": later}
    executor.dispatch(job_id=JOB, snapshot=snapshot, fence=fence)
    request = tmp_path / JOB / "request.json"
    digest = hashlib.sha256(request.read_bytes()).hexdigest()
    finished = {"job_id": JOB, "request_hash": digest, "state": "succeeded", "exit_code": 0,
                "accepted_at": "x", "started_at": "x", "finished_at": "x"}
    name = executor.worker_name(JOB)
    worker = client.containers.rows[name]
    client.containers.rows[name] = _ExitedRace(
        worker.labels, lambda: atomic_json(request.with_name("receipt.json"), finished))
    assert executor.reconcile(job_id=JOB, snapshot=snapshot, fence=fence).state == "succeeded"


def test_worker_refuses_a_start_past_the_fence(tmp_path, monkeypatch):
    executor, client, snapshot = setup(tmp_path, monkeypatch)
    limit = datetime(2026, 1, 1, tzinfo=timezone.utc)
    executor.dispatch(job_id=JOB, snapshot=snapshot,
                      fence={"generation": 3, "start_not_after": limit.isoformat()})
    request = tmp_path / JOB / "request.json"
    digest = hashlib.sha256(request.read_bytes()).hexdigest()
    FakeAPI.calls = 0
    assert run(request, digest, api_factory=FakeAPI, sleep=lambda _s: None,
               clock=lambda: limit + timedelta(seconds=1)) == 1
    receipt = json.loads(request.with_name("receipt.json").read_text())
    assert receipt["state"] == "failed" and receipt["side_effect_unknown"] is False
    assert receipt["started_at"] is None and "protected interval" in receipt["error"]
    assert FakeAPI.calls == 0


def test_fence_boundary():
    limit = datetime(2026, 1, 1, tzinfo=timezone.utc)
    request = {"fence": {"generation": 1, "start_not_after": limit.isoformat()}}
    assert not fence_expired(request, limit)
    assert fence_expired(request, limit + timedelta(microseconds=1))
    assert not fence_expired({}, limit + timedelta(days=1))


# -- fence capability: the immutable worker must prove it enforces the fence ----

def _labelled_docker(monkeypatch, labels):
    images = SimpleNamespace(get=lambda _: SimpleNamespace(attrs={"Config": {"Labels": labels}}))
    executor = DockerExecutor(client_factory=lambda: SimpleNamespace(images=images))
    monkeypatch.setattr(executor, "_check_prerequisites", lambda _: None)
    return executor, images


def test_docker_refuses_a_quiesce_job_for_a_worker_without_the_fence_label(monkeypatch):
    executor, images = _labelled_docker(monkeypatch, {})
    spec = {"action": "restart", "container_name": "x"}
    quiesced = {"spec": spec, "execution": {"worker_image": "old"}, "quiesce": {"enabled": True}}
    with pytest.raises(ExecutorError, match="repin LLM_BAWT_OPS_WORKER_IMAGE"):
        executor.check_quiesce_fence({"worker_image": "old"})
    with pytest.raises(ExecutorError, match="repin LLM_BAWT_OPS_WORKER_IMAGE"):
        executor.preflight(quiesced)
    executor.preflight({"spec": spec, "execution": {"worker_image": "old"}})  # unquiesced: no label needed
    images.get = lambda _: SimpleNamespace(attrs={"Config": {"Labels": {QUIESCE_FENCE_LABEL: "1"}}})
    executor.preflight(quiesced)

    def unreachable(_):
        raise RuntimeError("daemon gone")
    images.get = unreachable
    with pytest.raises(ExecutorError, match="cannot verify quiesce fence"):
        executor.check_quiesce_fence({"worker_image": "new"})


def test_fence_capability_fails_closed_and_the_worker_image_declares_it():
    class PlainExecutor(Executor):  # overrides nothing about the fence
        kind = available = dispatch = reconcile = lambda self, **_: "plain"

    with pytest.raises(ExecutorError, match="plain executor cannot enforce the quiesce start fence"):
        PlainExecutor().check_quiesce_fence({})
    dockerfile = Path(__file__).resolve().parents[1] / "docker" / "Dockerfile.ops-worker"
    assert f'LABEL {QUIESCE_FENCE_LABEL}="1"' in dockerfile.read_text()


def test_no_quiesce_approval_is_prepared_for_an_unfenced_worker():
    service, executor = _quiesce_service()
    executor._fenced = False
    with pytest.raises(OpsDispatchError) as exc:
        service.prepare_invocation("test.restart-thing", {"target": "a"})
    assert exc.value.code == "quiesce_unavailable"
    service.store.create_operation(_op_data(slug="test.plain"))
    assert "quiesce" not in service.prepare_invocation("test.plain", {"target": "a"})


# -- (e)/(f) N1: receipts are pushed by the app, which announces them -----------

class _Part:
    def __init__(self, error=None):
        self.error, self.calls = error, 0

    def __call__(self, **_):
        self.calls += 1
        if self.error:
            raise self.error
        return 0


def test_the_reconciler_tick_admits_even_when_reconcile_fails_and_sends_no_receipts():
    from llm_bawt.ops.reconciler import OpsReconciler

    reconcile, admit = _Part(RuntimeError("reconcile down")), _Part()
    tick = OpsReconciler(SimpleNamespace(reconcile_active_jobs=reconcile),
                         admission=SimpleNamespace(admit_due=admit))
    with pytest.raises(RuntimeError, match="reconcile down"):
        tick.tick()
    assert (reconcile.calls, admit.calls) == (1, 1)
    assert not hasattr(tick, "receipts")


class _Dispatcher:
    def __init__(self):
        self.store, self.calls = SimpleNamespace(engine=object()), []

    def announce_admitted(self, record):
        self.calls.append(("announce", record.id))

    def wake(self, target):
        self.calls.append(("wake", target))


def test_a_new_receipt_is_announced_and_a_duplicate_only_wakes():
    from llm_bawt.service.ops_receipt_outbox import announce

    dispatcher = _Dispatcher()
    hook = announce(dispatcher)
    hook(SimpleNamespace(id="d1", target_bot_id="byte"), False)
    hook(SimpleNamespace(id="d1", target_bot_id="byte"), True)
    assert dispatcher.calls == [("announce", "d1"), ("wake", "byte")]


def test_the_receipt_outbox_is_off_without_a_dispatcher_or_a_postgres_ops_store(monkeypatch):
    import llm_bawt.service.ops_receipt_outbox as outbox

    assert outbox.build_receipt_sender(SimpleNamespace(_inter_bot_dispatcher=None)) is None
    sqlite = SimpleNamespace(store=SimpleNamespace(engine=SimpleNamespace(dialect=SimpleNamespace(name="sqlite"))))
    monkeypatch.setattr(outbox, "get_ops_service", lambda _config: sqlite)
    service = SimpleNamespace(_inter_bot_dispatcher=_Dispatcher(), config=None)
    assert outbox.build_receipt_sender(service) is None
    postgres = SimpleNamespace(store=SimpleNamespace(engine=SimpleNamespace(dialect=SimpleNamespace(name="postgresql"))))
    monkeypatch.setattr(outbox, "get_ops_service", lambda _config: postgres)
    sender = outbox.build_receipt_sender(service)
    assert sender.ops is postgres and sender.deliveries is service._inter_bot_dispatcher.store
    assert sender.on_enqueued is not None


def test_the_end_turn_instructions_are_registry_prompts_with_seed_fallbacks(monkeypatch):
    from llm_bawt import prompt_registry
    from llm_bawt.ops import quiesce_prompts as qp
    from llm_bawt.prompt_definitions import DEFAULT_PROMPT_DEFINITIONS

    for key in (qp.END_TURN_NEW_TURN_KEY, qp.END_TURN_NO_RECEIPT_KEY):
        assert DEFAULT_PROMPT_DEFINITIONS[key].default_body() == qp.SEEDS[key]
        seen = []
        resolved = SimpleNamespace(body="[custom]")
        monkeypatch.setattr(prompt_registry, "get_prompt_resolver", lambda: SimpleNamespace(
            resolve=lambda k, **kw: seen.append((k, kw)) or resolved))
        assert qp.resolve(key, "byte") == "[custom]"
        assert seen == [(key, {"scope_type": "bot", "scope_id": "byte"})]
        resolved.body = ""
        assert qp.resolve(key, "byte") == qp.SEEDS[key]
    assert "new turn" in qp.DEFAULT_END_TURN_NEW_TURN
    assert "new turn" not in qp.DEFAULT_END_TURN_NO_RECEIPT


def test_the_receipt_prefix_is_a_registry_prompt_with_a_seed_fallback(monkeypatch):
    from llm_bawt import prompt_registry
    from llm_bawt.ops.receipt_callback import DEFAULT_QUIESCE_RECEIPT, PROMPT_KEY, receipt_prompt
    from llm_bawt.prompt_definitions import DEFAULT_PROMPT_DEFINITIONS

    assert DEFAULT_PROMPT_DEFINITIONS[PROMPT_KEY].default_body() == DEFAULT_QUIESCE_RECEIPT
    resolved = SimpleNamespace(body="[custom receipt prefix]")
    monkeypatch.setattr(prompt_registry, "get_prompt_resolver",
                        lambda: SimpleNamespace(resolve=lambda *a, **k: resolved))
    assert receipt_prompt("byte") == "[custom receipt prefix]"
    resolved.body = "  "
    assert receipt_prompt("byte") == DEFAULT_QUIESCE_RECEIPT

    def broken():
        raise RuntimeError("registry down")
    monkeypatch.setattr(prompt_registry, "get_prompt_resolver", broken)
    assert receipt_prompt("byte") == DEFAULT_QUIESCE_RECEIPT


# -- C7: the reconciler's import closure ---------------------------------------

# A fresh interpreter importing the closure costs ~1s, over the unit budget:
# it runs in the local-integration lane (no service, no DB).
@pytest.mark.integration
@pytest.mark.offline_integration
def test_quiesce_driver_import_closure_has_no_app_wiring():
    code = (
        "import sys, llm_bawt.ops.quiesce, llm_bawt.ops.health_runtime, llm_bawt.ops.quiesce_settings\n"
        "import llm_bawt.ops.receipt_callback, llm_bawt.inter_bot_delivery\n"
        "bad = [m for m in ('fastapi', 'starlette', 'llm_bawt.service.dependencies',"
        " 'llm_bawt.service.api') if m in sys.modules]\n"
        "print(','.join(bad))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == ""
