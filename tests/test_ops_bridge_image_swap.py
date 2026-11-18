"""TASK-2469: claude-code-bridge image deploy/rollback, and the import preflight
running in the TARGET image before anything stops. No live Docker or network."""
from __future__ import annotations

import json
import struct

import pytest

from llm_bawt.ops.app_config_reconcile import AppConfigReconciler, validate_app_spec
from llm_bawt.ops.executor import validate_spec
from llm_bawt.ops.gpu_reconcile import is_gpu_reconcile
from llm_bawt.ops.image_deploy import DeployFailed
from llm_bawt.ops.quiesce_settings import quiesce_binding
from llm_bawt.ops.validation import ArgValidationError, validate_args
from tests.ops_install_rows import SEEDS
from tests.test_ops_app_config_reconcile import (  # noqa: F401 - ``source`` is an autouse fixture
    APP_ID, APP_NAME, SPEC, TAG, AppDocker, _image, reconciler, source,
)
from tests.test_ops_gpu_image_swap import DEPLOY_JOB, NEW_ID, swap_snapshot

DEPLOY_SPEC = {**SPEC, "image_swap": "deploy"}
ROLLBACK_SPEC = {**SPEC, "image_swap": "rollback"}
VERSIONS = {"schema": 1, "harness": {"claude-agent-sdk": "0.2.184"}, "claude_cli": "2.1.9", "image": None}


def _seed(slug):
    return next(s for s in SEEDS if s["slug"] == slug)


class BridgeDocker(AppDocker):
    """AppDocker that also answers GET /versions in the container."""

    def __init__(self, *, versions=None, versions_exit=0, **kwargs):
        super().__init__(**kwargs)
        self.versions_body = json.dumps(VERSIONS) if versions is None else versions
        self.versions_exit = versions_exit
        self.images[NEW_ID] = _image(NEW_ID, [TAG])

    def request(self, method, path, *, body=None, headers=None, raw=False):
        if method == "POST" and "/tag?" in path:
            return None
        if method == "POST" and path.endswith("/exec") and any("/versions" in c for c in body["Cmd"]):
            self.exec_cmds.append(body["Cmd"])
            return {"Id": "versions1"}
        if path.startswith("/exec/versions1/"):
            if path.endswith("/start"):
                payload = self.versions_body.encode()
                return struct.pack(">BxxxI", 1, len(payload)) + payload
            return {"ExitCode": self.versions_exit}
        return super().request(method, path, body=body, headers=headers, raw=raw)


@pytest.fixture
def preflight(monkeypatch):
    """Records which image the import preflight ran in, and whether the bridge
    was still untouched; ``refuse`` names an image whose import fails."""
    seen = {"images": [], "refuse": None, "creates_before": []}

    def fake(self):
        seen["images"].append(self._approved_image_id)
        seen["creates_before"].append(len(self.api.create_bodies))
        if self._approved_image_id == seen["refuse"]:
            self._fail("import preflight failed in the target image; nothing changed", known=True)

    monkeypatch.setattr(AppConfigReconciler, "_import_preflight", fake)
    return seen


# ── catalog ────────────────────────────────────────────────────────────────

def test_rows_are_the_bridge_reconcile_plus_one_argument_and_quiesce():
    base = json.loads(_seed("llm-bawt.reconcile-claude-code-bridge-config")["command_script"])
    for slug, mode, arg, good, bad in (
            ("llm-bawt.deploy-claude-code-bridge-image", "deploy", "image_id", NEW_ID, "llm-bawt-agent-bridge:latest"),
            ("llm-bawt.rollback-claude-code-bridge-image", "rollback", "deploy_job_id", DEPLOY_JOB, "x" * 32)):
        row = _seed(slug)
        spec = validate_spec(row["command_script"])
        assert spec == {**base, "image_swap": mode} and not is_gpu_reconcile(spec)
        assert "import_preflight" in spec and spec["container_name"] == "llm-bawt-claude-code-bridge"
        assert row["risk_level"] == "critical" and row["max_concurrent"] == 1 and not row["enabled"]
        assert json.loads(row["quiesce_scope_json"]) == {"backends": ["claude-code"]}
        assert row["quiesce_enabled"] is True
        binding = quiesce_binding(row, {arg: good})
        assert binding["enabled"] and binding["timeout_seconds"] == 600
        assert binding["scope"]["backends"] == ["claude-code"]
        assert "KILLS ACTIVE AGENTS" in row["approval_prompt_prefix"]
        assert validate_args({arg: good}, row["args_schema_json"], row["args_defaults_json"]) == {arg: good}
        for args in ({}, {arg: bad}):
            with pytest.raises(ArgValidationError):
                validate_args(args, row["args_schema_json"], row["args_defaults_json"])


def test_image_swap_allowed_without_gpu_but_only_known_modes():
    assert validate_app_spec(dict(DEPLOY_SPEC))["image_swap"] == "deploy"
    with pytest.raises(ValueError):
        validate_app_spec({**SPEC, "image_swap": "pull"})


# ── preflight runs in the target image, before anything stops ─────────────

def test_deploy_preflights_the_candidate_not_the_running_image(tmp_path, preflight):
    preflight["refuse"] = APP_ID  # the old image cannot import the new source
    api = BridgeDocker()
    result = json.loads(reconciler(api, tmp_path, swap_snapshot(api, DEPLOY_SPEC)).run())
    assert preflight["images"] == [NEW_ID] and preflight["creates_before"] == [0]
    assert result["phase"] == "done" and api.by_name(APP_NAME)["Image"] == NEW_ID


def test_deploy_target_that_cannot_import_refuses_before_stop(tmp_path, preflight):
    preflight["refuse"] = NEW_ID
    api = BridgeDocker()
    with pytest.raises(DeployFailed) as caught:
        reconciler(api, tmp_path, swap_snapshot(api, DEPLOY_SPEC)).run()
    assert caught.value.outcome_known and "nothing changed" in str(caught.value)
    assert not api.create_bodies and api.by_name(APP_NAME)["Image"] == APP_ID
    assert not any(m == "POST" and p.endswith("/stop") for m, p in api.calls)


def _rolled_back_api():
    api = BridgeDocker()
    running = api.by_name(APP_NAME)
    running["Image"] = running["Config"]["Image"] = NEW_ID  # as the deploy created it
    return api


def test_rollback_preflights_the_restored_image(tmp_path, preflight):
    preflight["refuse"] = NEW_ID  # the current image would fail; the target passes
    api = _rolled_back_api()
    snap = swap_snapshot(api, ROLLBACK_SPEC, target=APP_ID, job=DEPLOY_JOB)
    json.loads(reconciler(api, tmp_path, snap).run())
    assert preflight["images"] == [APP_ID] and api.by_name(APP_NAME)["Image"] == APP_ID


def test_rollback_target_that_cannot_import_refuses_before_stop(tmp_path, preflight):
    preflight["refuse"] = APP_ID
    api = _rolled_back_api()
    snap = swap_snapshot(api, ROLLBACK_SPEC, target=APP_ID, job=DEPLOY_JOB)
    with pytest.raises(DeployFailed):
        reconciler(api, tmp_path, snap).run()
    assert not api.create_bodies and api.by_name(APP_NAME)["Image"] == NEW_ID


# ── verification: no embed probe, best-effort versions ─────────────────────

def test_deploy_records_versions_and_runs_no_embedding_probe(tmp_path, preflight):
    api = BridgeDocker()
    result = json.loads(reconciler(api, tmp_path, swap_snapshot(api, DEPLOY_SPEC)).run())
    assert result["versions"] == {k: VERSIONS[k] for k in ("schema", "harness", "claude_cli")}
    assert not any("/embed" in " ".join(c) for c in api.exec_cmds)
    assert "embedding_probe" not in json.dumps(result)


@pytest.mark.parametrize("kwargs", [{"versions_exit": 22}, {"versions": "not json"},
                                    {"versions": json.dumps({"harness": "x"})}])
def test_missing_or_bad_versions_never_fail_the_deploy(tmp_path, preflight, kwargs):
    api = BridgeDocker(**kwargs)
    result = json.loads(reconciler(api, tmp_path, swap_snapshot(api, DEPLOY_SPEC)).run())
    assert result["phase"] == "done" and result["versions"] is None
    assert api.by_name(APP_NAME)["Image"] == NEW_ID
