"""TASK-1476: the ordered multi-target service plan (cutover). No live Docker or network.

Each target's own reconciler is proven in its own suite; here every step is a
stub, so these tests pin the plan's ordering, credential classes, probes,
volumes and the revert-all contract.
"""
from __future__ import annotations

import copy
import fcntl
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_bawt.env_import import SPECS as ENV_IMPORT_SPECS
from llm_bawt.ops.app_config_reconcile import ACTION as APP_CONFIG_ACTION
from llm_bawt.ops.executor import CONFIG_WORKER_LABELS, DockerExecutor, validate_spec
from llm_bawt.ops.image_deploy import DeployFailed, ImageDeployer, Swap
from llm_bawt.ops.plan_approval import PlanError, approval_text, bind_plan, credential_names
from llm_bawt.ops.plan_reconcile import (
    ACCEPTED_LOSS, ACTION as PLAN_ACTION, APP_PROBE_MODULE, BRIDGE_PROBE_MODULE, COMPOSE_PROJECT_LABEL,
    COMPOSE_VOLUME_LABEL, CREATED_BY_LABEL, DESTINATIONS, ENV_KEPT, KINDS, MIGRATED, UNCONFIGURED,
    WORKER_CAPABILITY_LABEL, CredentialLoss, PlanReconciler, classify, env_names, plan_probes, probe_argv,
    read_probe, valid_plan_snapshot, validate_plan_spec,
)
from llm_bawt.ops.probe_contract import DESTINATION_PROBES, RECORD_KEYS, RECORDS_PROBE, emit
from llm_bawt.ops.quiesce_settings import validate_quiesce_settings
from llm_bawt.ops.seeds import SEEDS
from llm_bawt.ops.service import APPROVAL_ONLY_ACTIONS, _valid_config_snapshot
from llm_bawt.ops.tagged_config_reconcile import ACTION as TAGGED_CONFIG_ACTION
from llm_bawt.ops.validation import ArgValidationError, validate_args, validate_catalog
from tests.test_ops_app_config_reconcile import (
    APP_NAME, OPERATION, SPEC as APP_SPEC, AppDocker, binding as app_binding,
)
from tests.test_ops_image_deploy import JOB_ID

NAMES = credential_names()
PROJECT = "proj"
SECRET = "sk-never-in-a-receipt"
PLAN_SEED = next(s for s in SEEDS if s["slug"] == "llm-bawt.cutover-plan")


def _records(*connected):
    return {key: {"configured": key in connected, "connected": key in connected} for key in sorted(RECORD_KEYS)}


# brave is stored (migrated), zai stays in env (kept), the rest are unconfigured.
RECORDS = _records("brave")
APP_ENV = {"PATH": "/usr/bin", **{n: SECRET for n in NAMES["brave"] + NAMES["zai"]}}
DESIRED_ENV = {"PATH": "/usr/bin", **{n: SECRET for n in NAMES["zai"]}}


def _ok(probe_id, records=None):
    return emit(probe_id, True, 200 if records is None else None, "ok", records) + "\n"


class FakeAPI:
    """Container inspects and the volume endpoints a plan reads and writes."""

    def __init__(self, envs, log):
        self.envs = envs  # container id -> env dict
        self.volumes = {}
        self.calls = []
        self.log = log
        self.side_effect_started = False

    def request(self, method, path, *, body=None, **_kw):
        self.calls.append((method, path, body))
        if method != "GET":
            self.log.append(("api", f"{method} {path}"))
        parts = path.split("/")
        if method == "GET" and parts[1] == "containers":
            return {"Config": {"Env": [f"{k}={v}" for k, v in self.envs[parts[2]].items()]}}
        if method == "GET" and parts[1] == "volumes":
            if parts[2] not in self.volumes:
                raise RuntimeError("Docker HTTP 404: no such volume")
            return self.volumes[parts[2]]
        if method == "POST" and path == "/volumes/create":
            self.volumes[body["Name"]] = {"Name": body["Name"], "Labels": body["Labels"]}
            return self.volumes[body["Name"]]
        if method == "DELETE" and parts[1] == "volumes":
            del self.volumes[parts[2]]
            return None
        raise AssertionError(f"unexpected {method} {path}")


class Step:
    """One target's reconciler: the step methods the plan calls, logged."""

    _target_lock = ImageDeployer._target_lock

    def __init__(self, slug, log, lock_root, *, kind=APP_CONFIG_ACTION, drift=True, env=None, desired_env=None,
                 mounts=None, fail=None, probe=None):
        self.slug, self.log, self.lock_root, self.kind = slug, log, lock_root, kind
        self.name = f"{PROJECT}-{slug}"
        self.spec = {"compose_project": PROJECT, "container_name": self.name}
        self.snapshot = {"config": {"drift": ["env"] if drift else []}}
        self.detail = {"warnings": []}
        self.drift = drift
        self.env = env or {"PATH": "/usr/bin"}
        self._desired = {"env": desired_env or self.env, "mounts": mounts or {}}
        self.fail = fail or {}
        self.probe = probe or {}
        self.ignored_ids = frozenset()
        self.seen_ignored = []

    def _call(self, method, *args):
        self.log.append((self.slug, method))
        error = self.fail.get(method)
        if error is not None:
            raise error(self) if callable(error) else error

    def _fail(self, message, *, known):
        raise DeployFailed(message, self.detail, outcome_known=known)

    def _prepare(self):
        self._call("prepare")
        self.seen_ignored.append(self.ignored_ids)
        return {"Id": f"{self.slug}-old", "Config": {"Env": [f"{k}={v}" for k, v in self.env.items()]}}, "image"

    def _before_swap(self):
        self._call("before_swap")

    def _verify_only(self, old):
        self._call("verify_only")
        if not self.drift:
            self.detail["unchanged"] = True
        return not self.drift

    def _swap_ref(self, binding):
        return "ref"

    def _release(self, binding):
        return None

    def _swap(self, old, old_image, new_image, ref, release):
        self._call("swap")
        return Swap(old["Id"], f"{self.slug}-new", f"{self.name}-prev")

    def _commit_swap(self, swap):
        self._call("commit")

    def _revert(self, swap):
        self._call("revert")

    def _exec(self, container_id, argv):
        self.log.append((self.slug, f"probe:{argv[-1]}@{container_id}"))
        answer = self.probe.get((argv[-1], container_id), self.probe.get(argv[-1]))
        if answer is None:
            answer = _ok(argv[-1], RECORDS if argv[-1] == RECORDS_PROBE else None)
        return answer if isinstance(answer, tuple) else (0, answer)


def _known_restored(step):
    step.detail["restored"] = True
    return DeployFailed("health failed; previous container restored and healthy", step.detail, outcome_known=True)


def _unknown(step):
    return DeployFailed("RESTORE FAILED; production state unknown", step.detail, outcome_known=False)


SLUGS = ("app", "crawl", "bridge")
KIND_OF = {"app": APP_CONFIG_ACTION, "crawl": TAGGED_CONFIG_ACTION, "bridge": APP_CONFIG_ACTION}


def plan_snapshot(*, records=RECORDS, env_now=APP_ENV, env_next=DESIRED_ENV, accept=(),
                  probes=({"id": "broker", "target": "bridge"},)):
    spec = {"action": PLAN_ACTION, "targets": list(SLUGS), "records_target": "app", "probes": list(probes)}
    now, nxt = env_names(env_now, NAMES), env_names(env_next, NAMES)
    credentials = classify(records, now, nxt, sorted(accept))
    targets = [{"slug": s, "operation": {"slug": s}, "spec": {"action": KIND_OF[s]}, "config": {},
                "newly_trusted": None} for s in SLUGS]
    return {"spec": spec, "resolved_args": {"accept_loss": sorted(accept)}, "plan": {
        "targets": targets, "records": records, "credential_names": NAMES, "env_now": now, "env_next": nxt,
        "credentials": credentials, "accept_loss": sorted(accept), "probes": plan_probes(spec, credentials)}}


@pytest.fixture
def world(tmp_path):
    """Three stub targets (app is the records target) and their Docker API."""
    log = []
    locks = tmp_path / "locks"
    steps = {"app": Step("app", log, locks, env=APP_ENV, desired_env=DESIRED_ENV),
             "crawl": Step("crawl", log, locks, kind=TAGGED_CONFIG_ACTION),
             "bridge": Step("bridge", log, locks)}
    api = FakeAPI({"app-new": DESIRED_ENV, "app-old": APP_ENV}, log)
    return SimpleNamespace(log=log, steps=steps, api=api, locks=locks)


def _reconciler(world, snap=None):
    def build(cls, api, target_snapshot, job_id, **kw):
        slug = target_snapshot["operation"]["slug"]
        assert cls is KINDS[KIND_OF[slug]][0] and api is world.api and job_id == JOB_ID
        assert set(kw) == {"lock_root", "sleep", "clock"}
        return world.steps[slug]
    return PlanReconciler(world.api, snap or plan_snapshot(), JOB_ID, lock_root=world.locks, build=build)


def _failed(world, snap=None) -> DeployFailed:
    with pytest.raises(DeployFailed) as caught:
        _reconciler(world, snap).run()
    return caught.value


def _statuses(detail):
    return [t["status"] for t in detail["targets"]]


def _writes(world):
    return [(slug, m) for slug, m in world.log if m in ("swap", "revert", "commit")]


# ── success ────────────────────────────────────────────────────────────────

def test_every_target_is_checked_before_any_swap_then_swapped_in_order_and_committed_last(world):
    receipt = json.loads(_reconciler(world).run())
    assert SECRET not in json.dumps(receipt)
    calls = [m for _s, m in world.log]
    first_swap = calls.index("swap")
    # Preflight: every target prepared and checked before the first swap.
    assert [(s, m) for s, m in world.log[:first_swap] if m in ("prepare", "before_swap")][:6] == [
        ("app", "prepare"), ("app", "before_swap"), ("crawl", "prepare"), ("crawl", "before_swap"),
        ("bridge", "prepare"), ("bridge", "before_swap")]
    assert _writes(world) == [("app", "swap"), ("crawl", "swap"), ("bridge", "swap"),
                              ("app", "commit"), ("crawl", "commit"), ("bridge", "commit")]
    # The end checks run in the NEW app; the bridge probe in the new bridge.
    assert ("app", "probe:records@app-new") in world.log
    assert ("bridge", "probe:broker@bridge-new") in world.log
    assert ("app", "probe:brave_search@app-new") in world.log
    assert world.log.index(("app", "probe:records@app-new")) < calls.index("commit")
    assert receipt["phase"] == "done" and receipt["error"] is None
    assert _statuses(receipt) == ["swapped", "swapped", "swapped"]
    assert [t["container_id"] for t in receipt["targets"]] == ["app-new", "crawl-new", "bridge-new"]
    assert receipt["credentials"] == {**{d: UNCONFIGURED for d in DESTINATIONS}, "brave": MIGRATED,
                                      "zai": ENV_KEPT}
    assert receipt["probes"] == [{"id": "broker", "target": "bridge", "ok": True, "status": 200, "code": "ok"},
                                 {"id": "brave_search", "target": "app", "ok": True, "status": 200, "code": "ok"}]


def test_a_later_targets_dependency_check_skips_the_earlier_stopped_previous_containers(world):
    _reconciler(world).run()
    # Second prepare of each target is its last-read recheck before its swap.
    assert world.steps["crawl"].seen_ignored == [frozenset(), frozenset({"app-old"})]
    assert world.steps["bridge"].seen_ignored == [frozenset(), frozenset({"app-old", "crawl-old"})]


def test_a_target_without_drift_is_verified_in_place(world):
    world.steps["crawl"].drift = False
    receipt = json.loads(_reconciler(world).run())
    assert _statuses(receipt) == ["swapped", "unchanged", "swapped"]
    assert receipt["targets"][1]["unchanged"] is True and receipt["targets"][1]["container_id"] == "crawl-old"
    assert ("crawl", "swap") not in world.log and ("crawl", "commit") not in world.log


# ── failure and revert ─────────────────────────────────────────────────────

def test_a_failed_later_target_reverts_every_earlier_one_in_reverse(world):
    world.steps["bridge"].fail = {"swap": _known_restored}
    exc = _failed(world)
    assert exc.outcome_known is True
    assert _writes(world) == [("app", "swap"), ("crawl", "swap"), ("bridge", "swap"),
                              ("crawl", "revert"), ("app", "revert")]
    assert _statuses(exc.detail) == ["reverted", "reverted", "restored"]
    assert [t["container_id"] for t in exc.detail["targets"]] == ["app-old", "crawl-old", "bridge-old"]
    assert exc.detail["error"].startswith("bridge: health failed") and "every target reverted" in exc.detail["error"]


def test_an_uncertain_target_failure_still_reverts_the_others_and_is_unknown(world):
    world.steps["crawl"].fail = {"swap": _unknown}
    exc = _failed(world)
    assert exc.outcome_known is False
    assert _writes(world) == [("app", "swap"), ("crawl", "swap"), ("app", "revert")]
    assert _statuses(exc.detail) == ["reverted", "uncertain", "untouched"]
    assert "REVERT INCOMPLETE" in exc.detail["error"]


def test_a_failed_revert_continues_with_the_rest_is_unknown_and_keeps_created_volumes(world):
    world.steps["bridge"]._desired["mounts"] = {"/data": {"type": "volume", "source": f"{PROJECT}_data",
                                                         "read_only": False}}
    world.steps["bridge"].fail = {"swap": _known_restored}
    world.steps["crawl"].fail = {"revert": RuntimeError("Docker HTTP 500: cannot start")}
    exc = _failed(world)
    assert exc.outcome_known is False
    assert _writes(world)[-2:] == [("crawl", "revert"), ("app", "revert")]
    assert _statuses(exc.detail) == ["reverted", "uncertain", "restored"]
    assert any(w.startswith("crawl revert failed") for w in exc.detail["warnings"])
    assert f"{PROJECT}_data" in world.api.volumes and "volumes_removed" not in exc.detail


@pytest.mark.parametrize("where, stdout, code", [
    ("bridge", "not json\n", "probe_unverifiable"),
    ("bridge", _ok("records", RECORDS), "probe_unverifiable"),  # a line answering another probe
    ("bridge", (1, emit("broker", True, 200, "ok")), "probe_unverifiable"),  # ok line, nonzero exit (r1 B1)
    ("bridge", emit("broker", False, 502, "http_error"), "http_error"),
    ("app", emit("brave_search", False, 401, "unauthorized"), "unauthorized"),
])
def test_a_failed_or_unverifiable_probe_reverts_everything(world, where, stdout, code):
    probe_id = "broker" if where == "bridge" else "brave_search"
    world.steps[where].probe = {probe_id: stdout}
    exc = _failed(world)
    assert exc.outcome_known is True
    assert ("app", "revert") in world.log and not any(m == "commit" for _s, m in world.log)
    failed = [p for p in exc.detail["probes"] if p["id"] == probe_id]
    assert failed == [{"id": probe_id, "target": where, "ok": False, "status": failed[0]["status"], "code": code}]
    assert f"probe_failed: {probe_id}@{where}:{code}" in exc.detail["error"]


def test_records_that_change_across_the_swap_revert(world):
    world.steps["app"].probe = {(RECORDS_PROBE, "app-new"): _ok(RECORDS_PROBE, _records())}
    exc = _failed(world)
    assert exc.outcome_known is True and "records_changed" in exc.detail["error"]
    assert ("app", "revert") in world.log


def test_an_unverifiable_records_line_after_the_swap_reverts(world):
    world.steps["app"].probe = {(RECORDS_PROBE, "app-new"): "{}\n"}
    exc = _failed(world)
    assert "probe_unverifiable: records" in exc.detail["error"] and ("app", "revert") in world.log


def test_a_recreated_app_with_other_credential_env_names_reverts(world):
    world.api.envs["app-new"] = APP_ENV  # still carries the dropped brave names
    exc = _failed(world)
    assert "credential_env_mismatch" in exc.detail["error"] and ("app", "revert") in world.log


# ── refusals before any write ──────────────────────────────────────────────

def _nothing_written(world):
    assert _writes(world) == []
    assert not [c for c in world.api.calls if c[0] in ("POST", "DELETE")]


def test_a_preflight_refusal_of_any_target_changes_nothing(world):
    world.steps["crawl"].fail = {"before_swap": lambda s: DeployFailed(
        "dependency_unresolved: redis matched 0 containers; nothing changed", s.detail, outcome_known=True)}
    exc = _failed(world)
    assert exc.outcome_known is True and exc.detail["error"].startswith("crawl: dependency_unresolved")
    assert ("bridge", "prepare") not in world.log
    _nothing_written(world)


@pytest.mark.parametrize("change, code", [
    ({"env": {"PATH": "/usr/bin"}}, "credential_env_changed_since_approval"),
    ({"desired_env": APP_ENV}, "credential_env_changed_since_approval"),
    ({"probe": {RECORDS_PROBE: _ok(RECORDS_PROBE, _records("brave", "zai"))}}, "records_changed_since_approval"),
    ({"probe": {RECORDS_PROBE: emit(RECORDS_PROBE, False, None, "error")}}, "records probe failed: error"),
])
def test_credential_facts_that_moved_since_approval_refuse_first(world, change, code):
    app = world.steps["app"]
    if "env" in change:
        app.env = change["env"]
    if "desired_env" in change:
        app._desired["env"] = change["desired_env"]
    if "probe" in change:
        app.probe = change["probe"]
    exc = _failed(world)
    assert exc.outcome_known is True and code in exc.detail["error"]
    _nothing_written(world)


def test_a_busy_target_lock_refuses_before_any_read(world):
    world.locks.mkdir(mode=0o700)
    with (world.locks / f"{PROJECT}-crawl.lock").open("a") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        exc = _failed(world)
    assert exc.outcome_known is True and "crawl: another deploy/rollback holds this target" in exc.detail["error"]
    assert world.log == []
    # Released: the next run takes every lock.
    _reconciler(world).run()


# ── volumes ────────────────────────────────────────────────────────────────

def _mount(source):
    return {"/data": {"type": "volume", "source": source, "read_only": False}}


def test_a_missing_project_volume_is_created_for_compose_and_kept_on_success(world):
    world.steps["bridge"]._desired["mounts"] = _mount(f"{PROJECT}_sessions")
    receipt = json.loads(_reconciler(world).run())
    assert world.api.volumes[f"{PROJECT}_sessions"]["Labels"] == {
        COMPOSE_PROJECT_LABEL: PROJECT, COMPOSE_VOLUME_LABEL: "sessions", CREATED_BY_LABEL: JOB_ID}
    assert receipt["volumes_created"] == [f"{PROJECT}_sessions"]
    # Created after every preflight read and before the first swap.
    create = world.log.index(("api", "POST /volumes/create"))
    assert world.log.index(("bridge", "before_swap")) < create < world.log.index(("app", "swap"))


def test_a_volume_this_job_created_is_removed_after_a_full_revert(world):
    world.steps["bridge"]._desired["mounts"] = _mount(f"{PROJECT}_sessions")
    world.steps["bridge"].fail = {"swap": _known_restored}
    exc = _failed(world)
    assert exc.outcome_known is True and exc.detail["volumes_removed"] == [f"{PROJECT}_sessions"]
    assert f"{PROJECT}_sessions" not in world.api.volumes


@pytest.mark.parametrize("name, existing, code", [
    ("shared_data", None, "volume_missing"),
    (f"{PROJECT}_data", {"Labels": {COMPOSE_PROJECT_LABEL: "other"}}, "volume_foreign"),
    (f"{PROJECT}_data", {"Labels": {COMPOSE_PROJECT_LABEL: PROJECT, COMPOSE_VOLUME_LABEL: "else"}},
     "volume_foreign"),
])
def test_a_volume_the_project_does_not_own_refuses_with_nothing_changed(world, name, existing, code):
    world.steps["crawl"]._desired["mounts"] = _mount(name)
    if existing is not None:
        world.api.volumes[name] = existing
    exc = _failed(world)
    assert exc.outcome_known is True and exc.detail["error"].startswith(code)
    _nothing_written(world)


def test_an_existing_project_volume_is_used_as_is(world):
    world.steps["crawl"]._desired["mounts"] = _mount(f"{PROJECT}_data")
    world.api.volumes[f"{PROJECT}_data"] = {"Labels": {COMPOSE_PROJECT_LABEL: PROJECT,
                                                       COMPOSE_VOLUME_LABEL: "data"}}
    receipt = json.loads(_reconciler(world).run())
    assert receipt["volumes_created"] == []


# ── credential classes, probes, spec ───────────────────────────────────────

def _names(**present):
    return {d: sorted(present.get(d, [])) for d in DESTINATIONS}


def test_credential_classes():
    now = _names(brave=NAMES["brave"], zai=NAMES["zai"], radarr=NAMES["radarr"])
    nxt = _names(zai=NAMES["zai"])
    classes = classify(_records("brave"), now, nxt, ["radarr"])
    assert classes["brave"] == MIGRATED and classes["zai"] == ENV_KEPT
    assert classes["radarr"] == ACCEPTED_LOSS and classes["sonarr"] == UNCONFIGURED
    with pytest.raises(CredentialLoss) as caught:
        classify(_records("brave"), now, nxt, [])
    assert caught.value.code == "credential_without_record" and "radarr" in str(caught.value)
    with pytest.raises(ValueError, match="lose nothing: sonarr"):
        classify(_records("brave"), now, nxt, ["radarr", "sonarr"])
    # A stored record migrates even when its env names are dropped.
    assert classify(_records("brave"), now, nxt, ["radarr"])["brave"] == MIGRATED


def test_migrated_destinations_add_their_own_probe_in_the_records_target():
    spec = {"probes": [{"id": "web_fetch", "target": "app"}], "records_target": "app"}
    credentials = {**{d: UNCONFIGURED for d in DESTINATIONS}, "sonarr": MIGRATED, "brave": MIGRATED}
    assert plan_probes(spec, credentials) == [{"id": "web_fetch", "target": "app"},
                                              {"id": "brave_search", "target": "app"},
                                              {"id": "sonarr_status", "target": "app"}]


def test_probe_argv_is_fixed_by_id():
    assert probe_argv("broker") == ["python", "-m", BRIDGE_PROBE_MODULE, "broker"]
    assert probe_argv("records") == ["python", "-m", APP_PROBE_MODULE, "records"]
    assert read_probe("web_fetch", _ok("web_fetch") + "trailing noise")["ok"] is True
    with pytest.raises(ValueError):
        read_probe("web_fetch", _ok("broker"))


def test_destinations_are_env_imports():
    assert set(DESTINATIONS) == {s.destination for s in ENV_IMPORT_SPECS} == set(DESTINATION_PROBES)


GOOD_SPEC = {"action": PLAN_ACTION, "targets": ["a", "b"], "records_target": "a",
             "probes": [{"id": "broker", "target": "b"}]}


@pytest.mark.parametrize("tamper", [
    lambda s: s.update(extra=1),
    lambda s: s.update(targets=[]),
    lambda s: s.update(targets=["a", "a"]),
    lambda s: s.update(targets=[f"t{i}" for i in range(7)], records_target="t0", probes=[]),
    lambda s: s.update(targets=["A b"]),
    lambda s: s.update(records_target="c"),
    lambda s: s.update(probes=[{"id": "records", "target": "a"}]),
    lambda s: s.update(probes=[{"id": "brave_search", "target": "a"}]),
    lambda s: s.update(probes=[{"id": "broker", "target": "c"}]),
    lambda s: s.update(probes=[{"id": "broker", "target": "b"}] * 2),
    lambda s: s.update(probes=[{"id": "broker", "target": "b", "argv": ["sh"]}]),
])
def test_plan_spec_shape_is_closed(tamper):
    validate_plan_spec(copy.deepcopy(GOOD_SPEC))
    spec = copy.deepcopy(GOOD_SPEC)
    tamper(spec)
    with pytest.raises(ValueError):
        validate_plan_spec(spec)


# ── approval binding (one app target over the app's fake Docker) ───────────

def _app_api(desired=DESIRED_ENV):
    api = AppDocker()
    container = api.by_name(APP_NAME)
    container["Config"]["Env"] = [f"{k}={v}" for k, v in APP_ENV.items()]
    return api, container


class PlanExecutor:
    def __init__(self, api, container, records=RECORDS):
        self.api, self.container, self.records = api, container, records
        self.probes = []

    def config_bind(self, spec):
        return app_binding(self.api), {"env": dict(DESIRED_ENV)}, self.container

    def probe(self, container_name, argv):
        self.probes.append((container_name, argv))
        return _ok(argv[-1], self.records)


def _op(**over):
    row = {"enabled": True, "soft_deleted_at": None, "executor_kind": "docker",
           "command_script": json.dumps(APP_SPEC), "args_schema_json": "{}", "args_defaults_json": "{}", **over}
    return SimpleNamespace(**row, to_api=lambda: {**OPERATION, "slug": "app", "command_script": row["command_script"]})


ONE_TARGET = {"action": PLAN_ACTION, "targets": ["app"], "records_target": "app", "probes": []}


def _bind(accept=None, *, op=None, executor=None):
    api, container = _app_api()
    return bind_plan(ONE_TARGET, accept, executor=executor or PlanExecutor(api, container),
                     get_operation=lambda slug: op or _op(), executor_kind="docker",
                     check_source=lambda commit: None, last_classification=lambda slug: None)


def _snap(plan, accept=()):
    return {"spec": ONE_TARGET, "resolved_args": {"accept_loss": sorted(accept)}, "plan": plan}


def test_bind_refuses_a_credential_the_app_would_lose_without_a_record():
    records = _records()  # brave not stored: dropping its env names loses it
    api, container = _app_api()
    with pytest.raises(PlanError) as caught:
        _bind(executor=PlanExecutor(api, container, records))
    assert caught.value.code == "credential_without_record" and "brave" in str(caught.value)
    assert SECRET not in str(caught.value)
    plan = _bind(["brave"], executor=PlanExecutor(api, container, records))
    assert plan["credentials"]["brave"] == ACCEPTED_LOSS and plan["accept_loss"] == ["brave"]
    text = approval_text({"plan": plan}, None)
    assert f"ACCEPTED LOSS: brave stops working (drops {', '.join(sorted(NAMES['brave']))}" in text
    assert SECRET not in text


def test_a_bound_plan_is_a_valid_snapshot_and_its_card_lists_order_classes_and_probes():
    plan = _bind()
    snap = _snap(plan)
    assert valid_plan_snapshot(snap) and _valid_config_snapshot(snap)
    assert plan["credentials"]["brave"] == MIGRATED and plan["credentials"]["zai"] == ENV_KEPT
    assert plan["probes"] == [{"id": "brave_search", "target": "app"}]
    text = approval_text(snap, "Cut over")
    assert text.splitlines()[0] == "Cut over" and "Order: app" in text
    assert "Migrated to stored records (probed after): brave" in text and "Env kept: zai" in text
    assert "Probes after: brave_search@app" in text and "[1] app" in text
    assert SECRET not in json.dumps(plan) + text


@pytest.mark.parametrize("op, code", [
    (_op(enabled=False), "plan_target_unavailable"),
    (_op(soft_deleted_at="2026-10-28"), "plan_target_unavailable"),
    (_op(executor_kind="ssh"), "plan_target_invalid"),
    (_op(command_script=json.dumps({"action": "restart", "container": "x"})), "plan_target_invalid"),
    (_op(args_schema_json=json.dumps({"type": "object", "properties": {"x": {"type": "string"}},
                                      "required": ["x"], "additionalProperties": False})),
     "plan_target_invalid"),
])
def test_a_plan_target_must_be_an_enabled_argument_free_config_row(op, code):
    with pytest.raises(PlanError) as caught:
        _bind(op=op)
    assert caught.value.code == code


def test_an_unverifiable_records_line_refuses_approval():
    api, container = _app_api()
    executor = PlanExecutor(api, container)
    executor.probe = lambda name, argv: "Traceback (most recent call last)\n"
    with pytest.raises(PlanError) as caught:
        _bind(executor=executor)
    assert caught.value.code == "probe_unverifiable"


def test_a_probe_that_exits_nonzero_refuses_approval():
    # TASK-1476 r1 B1: a valid ok line from a probe that exited nonzero is unverifiable.
    api, container = _app_api()
    executor = DockerExecutor(client_factory=lambda: None)
    output = _ok(RECORDS_PROBE, RECORDS).encode()
    target = SimpleNamespace(exec_run=lambda argv, **kw: SimpleNamespace(exit_code=1, output=output))
    executor._client = SimpleNamespace(containers=SimpleNamespace(get=lambda name: target))
    executor.config_bind = PlanExecutor(api, container).config_bind
    with pytest.raises(PlanError) as caught:
        _bind(executor=executor)
    assert caught.value.code == "records_unavailable"


@pytest.mark.parametrize("tamper", [
    lambda s: s["plan"]["credentials"].update(zai=UNCONFIGURED),
    lambda s: s["plan"]["probes"].clear(),
    lambda s: s["plan"]["env_next"].update(zai=["NOT_A_CREDENTIAL_NAME"]),
    lambda s: s["plan"]["records"]["brave"].update(connected=False, configured=False),
    lambda s: s["resolved_args"].update(accept_loss=["sonarr"]),
    lambda s: s["plan"].update(extra=1),
    lambda s: s["plan"]["targets"][0].update(newly_trusted={"bogus": []}),
    lambda s: s["plan"]["targets"][0]["config"].update(image_tag="other:latest"),
    lambda s: s["plan"]["credential_names"].update(brave=["NOT_THE_LIST"]),
])
def test_a_tampered_plan_is_not_a_valid_snapshot(tamper):
    snap = _snap(_bind())
    tamper(snap)
    assert not valid_plan_snapshot(snap) and not _valid_config_snapshot(snap)


def test_a_plan_snapshot_carries_no_single_target_config_and_other_kinds_no_plan():
    snap = _snap(_bind())
    assert not _valid_config_snapshot({**snap, "config": snap["plan"]["targets"][0]["config"]})
    api, _container = _app_api()
    single = {"spec": APP_SPEC, "resolved_args": {}, "config": app_binding(api), "newly_trusted": None}
    assert _valid_config_snapshot(single)
    assert not _valid_config_snapshot({**single, "plan": snap["plan"]})
    tagged = {**single, "spec": {**APP_SPEC, "action": TAGGED_CONFIG_ACTION}, "plan": snap["plan"]}
    assert not _valid_config_snapshot(tagged)


# ── catalog and gating ─────────────────────────────────────────────────────

def test_the_cutover_seed_orders_disabled_argument_free_config_rows():
    spec = validate_spec(PLAN_SEED["command_script"])
    assert spec["action"] == PLAN_ACTION and PLAN_SEED["enabled"] is False
    assert PLAN_SEED["risk_level"] == "critical" and "KILLS ACTIVE AGENTS" in PLAN_SEED["approval_prompt_prefix"]
    by_slug = {s["slug"]: s for s in SEEDS}
    kinds = []
    for slug in spec["targets"]:
        row = by_slug[slug]
        assert row["enabled"] is False and row["executor_kind"] == PLAN_SEED["executor_kind"]
        kinds.append(validate_spec(row["command_script"])["action"])
        assert validate_args({}, row["args_schema_json"], row["args_defaults_json"]) == {}
    assert set(kinds) <= set(KINDS)
    assert validate_spec(by_slug[spec["records_target"]]["command_script"])["action"] == APP_CONFIG_ACTION


def test_the_cutover_seed_accepts_only_known_destinations_and_valid_quiesce_settings():
    schema, _defaults = validate_catalog(PLAN_SEED["args_schema_json"], PLAN_SEED["args_defaults_json"])
    validate_quiesce_settings(SimpleNamespace(**PLAN_SEED), schema)
    args = (PLAN_SEED["args_schema_json"], PLAN_SEED["args_defaults_json"])
    assert validate_args({}, *args) == {"accept_loss": []}
    assert validate_args({"accept_loss": ["brave"]}, *args) == {"accept_loss": ["brave"]}
    with pytest.raises(ArgValidationError):
        validate_args({"accept_loss": ["nope"]}, *args)


def test_a_plan_needs_every_kinds_worker_label_and_its_own_and_runs_only_from_approval():
    labels = CONFIG_WORKER_LABELS[PLAN_ACTION]
    assert WORKER_CAPABILITY_LABEL in labels
    assert set(CONFIG_WORKER_LABELS[APP_CONFIG_ACTION]) | set(CONFIG_WORKER_LABELS[TAGGED_CONFIG_ACTION]) <= set(labels)
    assert len(labels) == len(set(labels))
    assert PLAN_ACTION in APPROVAL_ONLY_ACTIONS


def test_the_worker_image_carries_the_plan_module_probe_contract_and_label():
    dockerfile = (Path(__file__).resolve().parents[1] / "docker" / "Dockerfile.ops-worker").read_text()
    assert "src/llm_bawt/ops/plan_reconcile.py" in dockerfile
    assert "src/llm_bawt/ops/probe_contract.py" in dockerfile
    assert f'LABEL {WORKER_CAPABILITY_LABEL}="1"' in dockerfile
