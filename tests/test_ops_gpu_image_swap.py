"""TASK-2299: GPU reconcile recreating onto another LOCAL image (deploy) or back
to a deploy's recorded previous image (rollback). No live Docker or network."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from llm_bawt.ops.app_config_reconcile import (
    IMAGE_SWAP_LABEL, approval_text, valid_app_binding, valid_image_swap, validate_app_spec,
)
from llm_bawt.ops.executor import ExecutorError, validate_spec
from llm_bawt.ops.gpu_reconcile import is_gpu_reconcile
from llm_bawt.ops.image_deploy import DeployFailed
from llm_bawt.ops.service import OpsService
from llm_bawt.ops.validation import ArgValidationError, validate_args
from tests.ops_install_rows import SEEDS
from llm_bawt.ops.app_config_reconcile import AppConfigReconciler, app_config_binding
from llm_bawt.ops.compose_render import ComposeRenderer
from tests.test_ops_app_config_reconcile import (  # noqa: F401 - ``source`` is an autouse fixture
    APP_ID, APP_NAME, EXECUTION, OPERATION, SPEC, TAG, AppDocker, StubChecker, _image, reconciler, source,
)

NEW_ID = "sha256:" + "7" * 64
OTHER_ID = "sha256:" + "9" * 64
DEPLOY_JOB = "d" * 32
DEPLOY_SPEC = {**SPEC, "gpu": True, "image_swap": "deploy"}
ROLLBACK_SPEC = {**SPEC, "gpu": True, "image_swap": "rollback"}


def _seed(slug):
    return next(s for s in SEEDS if s["slug"] == slug)


@pytest.fixture(autouse=True)
def no_cdi(monkeypatch):
    """The fake app carries no CDI devices; the GPU preflight and created-container
    check are unchanged by this task and proven in test_ops_gpu_reconcile.py."""
    monkeypatch.setattr(AppConfigReconciler, "_check_gpu", lambda self: None)
    monkeypatch.setattr(AppConfigReconciler, "_after_create", lambda self, new_id: None)


def binding(api, spec):
    running = api.by_name(APP_NAME)
    return app_config_binding(spec, running, api.images[running["Image"]],
                              ComposeRenderer(api, spec, tag="app1"), tag_moved=lambda _tag: False,
                              source_checker=StubChecker())[0]


def swap_snapshot(api, spec=DEPLOY_SPEC, *, target=NEW_ID, job=None):
    bound = binding(api, spec)
    arg = {"image_id": target} if spec["image_swap"] == "deploy" else {"deploy_job_id": job}
    return {"spec": spec, "resolved_args": arg, "config": bound, "execution": EXECUTION, "newly_trusted": None,
            "image_swap": {"mode": spec["image_swap"], "target_image_id": target, "from_image_id": bound["image_id"],
                           "deploy_job_id": job}}


def gpu_api(**kwargs):
    api = AppDocker(**kwargs)
    api.images[NEW_ID] = _image(NEW_ID, [TAG])
    return api


# ── catalog ────────────────────────────────────────────────────────────────

def test_deploy_and_rollback_rows_are_the_gpu_reconcile_plus_one_argument():
    base = json.loads(_seed("llm-bawt.reconcile-local-model-bridge-config")["command_script"])
    for slug, mode, arg, good, bad in (
            ("llm-bawt.deploy-local-model-bridge-image", "deploy", "image_id", NEW_ID, "llm-bawt-local:latest"),
            ("llm-bawt.rollback-local-model-bridge-image", "rollback", "deploy_job_id", DEPLOY_JOB, "x" * 32)):
        row = _seed(slug)
        spec = validate_spec(row["command_script"])
        assert spec == {**base, "image_swap": mode} and is_gpu_reconcile(spec)  # same GPU reservation/guard
        assert row["risk_level"] == "high" and row["max_concurrent"] == 1 and not row["enabled"]
        assert validate_args({arg: good}, row["args_schema_json"], row["args_defaults_json"]) == {arg: good}
        for args in ({}, {arg: bad}):
            with pytest.raises(ArgValidationError):
                validate_args(args, row["args_schema_json"], row["args_defaults_json"])


def test_image_swap_only_on_a_gpu_spec_and_only_known_modes():
    with pytest.raises(ValueError):
        validate_app_spec({**SPEC, "image_swap": "deploy"})
    with pytest.raises(ValueError):
        validate_app_spec({**SPEC, "gpu": True, "image_swap": "pull"})


# ── snapshot binding ──────────────────────────────────────────────────────

def test_swap_snapshot_validates_and_any_disagreement_refuses():
    api = gpu_api()
    good = swap_snapshot(api)
    assert valid_app_binding(good)
    bad = [
        {**good, "resolved_args": {"image_id": OTHER_ID}},             # args differ from the bound target
        {**good, "resolved_args": {}},                                  # argument missing
        {k: v for k, v in good.items() if k != "image_swap"},           # no swap block
        {**good, "image_swap": {**good["image_swap"], "from_image_id": OTHER_ID}},  # not the bound running image
        {**good, "image_swap": {**good["image_swap"], "target_image_id": APP_ID},
         "resolved_args": {"image_id": APP_ID}},                        # same image: that is the reconcile
        {**good, "image_swap": {**good["image_swap"], "mode": "rollback"}},
        {**good, "image_swap": {**good["image_swap"], "deploy_job_id": DEPLOY_JOB}},
    ]
    assert not any(valid_app_binding(s) for s in bad)
    plain = {"spec": SPEC, "resolved_args": {}, "config": good["config"], "newly_trusted": None}
    assert valid_image_swap(plain)
    assert not valid_image_swap({**plain, "resolved_args": {"image_id": NEW_ID}})
    assert not valid_image_swap({**plain, "image_swap": good["image_swap"]})


def test_card_names_both_images_and_says_the_tag_is_not_moved():
    api = gpu_api()
    snap = swap_snapshot(api)
    text = approval_text(DEPLOY_SPEC, snap["config"], None, operation=OPERATION, newly_trusted=None,
                         image_swap=snap["image_swap"])
    assert f"IMAGE DEPLOY: {APP_ID} -> {NEW_ID}" in text and "not moved" in text
    assert "Unchanged image" not in text


# ── worker ────────────────────────────────────────────────────────────────

def test_deploy_recreates_onto_the_approved_image_and_records_both_ids(tmp_path):
    api = gpu_api()
    result = json.loads(reconciler(api, tmp_path, swap_snapshot(api)).run())
    assert api.create_bodies[0]["Image"] == NEW_ID
    assert api.by_name(APP_NAME)["Image"] == NEW_ID
    assert result["previous"]["image_id"] == APP_ID and result["deployed"]["image_id"] == NEW_ID
    assert result["image_swap"]["mode"] == "deploy" and result["phase"] == "done"


def test_failed_health_after_deploy_restores_the_previous_image(tmp_path):
    api = gpu_api(health_exit=7)
    with pytest.raises(DeployFailed) as caught:
        reconciler(api, tmp_path, swap_snapshot(api)).run()
    assert caught.value.outcome_known and caught.value.detail["restored"]
    assert api.by_name(APP_NAME)["Id"] == api.old_id and api.by_name(APP_NAME)["Image"] == APP_ID


def test_missing_target_image_refuses_before_anything_is_created(tmp_path):
    api = gpu_api()
    snap = swap_snapshot(api)
    del api.images[NEW_ID]
    with pytest.raises(DeployFailed) as caught:
        reconciler(api, tmp_path, snap).run()
    assert caught.value.outcome_known and "never pulls" in str(caught.value)
    assert caught.value.detail["deployed"] is None and not api.create_bodies


def test_rollback_restores_the_recorded_image(tmp_path):
    api = gpu_api()
    running = api.by_name(APP_NAME)
    running["Image"] = running["Config"]["Image"] = NEW_ID  # as the deploy created it: by ID
    snap = swap_snapshot(api, ROLLBACK_SPEC, target=APP_ID, job=DEPLOY_JOB)
    assert snap["config"]["image_id"] == NEW_ID and valid_app_binding(snap)
    result = json.loads(reconciler(api, tmp_path, snap).run())
    assert api.create_bodies[0]["Image"] == APP_ID and result["deployed"]["image_id"] == APP_ID


# ── approval-time binding (OpsService) ─────────────────────────────────────

class _Executor:
    def __init__(self, images):
        self.images = images

    def local_image(self, image_id):
        if image_id not in self.images:
            raise ExecutorError("image_not_local: the requested image is not on this daemon")
        return self.images[image_id]


def _service(jobs=()):
    service = object.__new__(OpsService)
    by_id = {j.id: j for j in jobs}
    service.store = SimpleNamespace(get_job=by_id.get)
    return service


CONFIG = {"image_id": APP_ID, "image_tag": f"{TAG}:latest"}


def _deploy_job(*, state="succeeded", previous=APP_ID, installed=NEW_ID, spec=DEPLOY_SPEC, phase="done"):
    record = {"phase": phase, "previous": {"image_id": previous}, "deployed": {"image_id": installed}}
    return SimpleNamespace(id=DEPLOY_JOB, state=state, invocation_snapshot_json=json.dumps({"spec": spec}),
                           output_tail=json.dumps(record))


def test_deploy_binds_a_local_image_of_this_service_only():
    ex = _Executor({NEW_ID: {"Id": NEW_ID, "RepoTags": [f"{TAG}:task-1270-e48e57d4"]},
                    OTHER_ID: {"Id": OTHER_ID, "RepoTags": ["somebody-else:latest"]}})
    bound = _service()._bind_image_swap(DEPLOY_SPEC, {"image_id": NEW_ID}, CONFIG, ex)
    assert bound == {"mode": "deploy", "target_image_id": NEW_ID, "from_image_id": APP_ID, "deploy_job_id": None}
    for target, text in ((OTHER_ID, "image_not_this_service"), ("sha256:" + "8" * 64, "image_not_local"),
                         (APP_ID, "image_unchanged"), ("abc", "full sha256")):
        ex.images.setdefault(APP_ID, {"Id": APP_ID, "RepoTags": [f"{TAG}:latest"]})
        with pytest.raises((ValueError, ExecutorError), match=text):
            _service()._bind_image_swap(DEPLOY_SPEC, {"image_id": target}, CONFIG, ex)


def test_rollback_binds_the_recorded_previous_image_only_while_the_deploy_is_live():
    ex = _Executor({APP_ID: {"Id": APP_ID, "RepoTags": []}})
    live = {**CONFIG, "image_id": NEW_ID}
    bound = _service([_deploy_job()])._bind_image_swap(ROLLBACK_SPEC, {"deploy_job_id": DEPLOY_JOB}, live, ex)
    assert bound == {"mode": "rollback", "target_image_id": APP_ID, "from_image_id": NEW_ID,
                     "deploy_job_id": DEPLOY_JOB}
    for job, config, text in (
            (_deploy_job(), CONFIG, "no longer runs the image"),         # target moved on since that deploy
            (_deploy_job(state="failed"), live, "SUCCEEDED"),
            (_deploy_job(phase="health"), live, "complete deployment record"),
            (_deploy_job(spec=SPEC), live, "not an image deploy"),
            (_deploy_job(spec={**DEPLOY_SPEC, "container_name": "other"}), live, "not an image deploy")):
        with pytest.raises(ValueError, match=text):
            _service([job])._bind_image_swap(ROLLBACK_SPEC, {"deploy_job_id": DEPLOY_JOB}, config, ex)
    with pytest.raises(ValueError, match="SUCCEEDED"):
        _service()._bind_image_swap(ROLLBACK_SPEC, {"deploy_job_id": DEPLOY_JOB}, live, ex)


def test_worker_label_constant_is_baked_into_the_worker_image():
    from pathlib import Path
    dockerfile = (Path(__file__).resolve().parents[1] / "docker" / "Dockerfile.ops-worker").read_text()
    assert f'LABEL {IMAGE_SWAP_LABEL}="1"' in dockerfile
