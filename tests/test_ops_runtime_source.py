"""TASK-1410/1496: real Git approval-to-worker runtime identity regressions."""
from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from llm_bawt.ops import OpsDispatchError, OpsService
from llm_bawt.ops.execution_mode import with_card_lines
from llm_bawt.ops.executor import spec_capability_labels, validate_spec
from llm_bawt.ops.runtime_source import (
    WORKER_CAPABILITY_LABEL, bind_runtime_source, valid_runtime_binding,
    validate_runtime_source, verify_runtime_source,
)
from llm_bawt.ops.source_state import SourceStateError
from llm_bawt.ops.worker import DockerAPI, run
from tests.test_ops_source_state import GitDocker, IMAGE, commit, git, repo as source_repo_fixture
from tests.test_ops_service import FakeExecutor, _store


@pytest.fixture
def repo(tmp_path):
    return source_repo_fixture.__wrapped__(tmp_path)


def spec_for(repo):
    return {"action": "restart", "container_name": "app", "runtime_source": {
        "project_dir": str(repo), "checker_image": IMAGE,
        "paths": ["src", "server.sh"], "mounts": {"/app/src": "src", "/app/server.sh": "server.sh"},
        "data_mounts": ["/data"]}}


def container_for(repo):
    return {"Id": "a" * 64, "Name": "/app", "Image": "sha256:" + "b" * 64,
            "Mounts": [{"Type": "bind", "Source": str(repo / p), "Destination": dest}
                       for dest, p in (("/app/src", "src"), ("/app/server.sh", "server.sh"))],
            "Config": {"Env": ["PYTHONPYCACHEPREFIX=/tmp/bytecode"]}}


def prepared(repo, tmp_path):
    scratch = tmp_path / "git-scratch"
    scratch.mkdir(exist_ok=True)
    api = GitDocker(scratch)
    spec = spec_for(repo)
    container = container_for(repo)
    args = {"expected_runtime_sha": git(repo, "rev-parse", "HEAD").strip()}
    binding = bind_runtime_source(api, spec, args, container, tag="test")
    return api, {"spec": spec, "resolved_args": args, "runtime_source": binding}, container


def test_clean_binding_and_different_head_with_only_tests_changed(repo, tmp_path):
    api, snapshot, container = prepared(repo, tmp_path)
    (repo / "test_added.py").write_text("assert True\n")
    new_head = commit(repo)
    after = verify_runtime_source(api, snapshot, container, tag="test")
    assert after["observed_head"] == new_head
    assert after["expected_runtime_sha"] != new_head
    assert after["runtime_tree_hash"] == snapshot["runtime_source"]["runtime_tree_hash"]
    assert valid_runtime_binding(snapshot)


@pytest.mark.parametrize("change", ["dirty", "committed", "staged", "untracked", "ignored", "deleted",
                                   "index-flags", "symlink", "mode"])
def test_runtime_changes_after_approval_are_refused(repo, tmp_path, change):
    api, snapshot, container = prepared(repo, tmp_path)
    module = repo / "src/pkg/module.py"
    if change in ("dirty", "committed", "staged"):
        module.write_text("VALUE = 2\n")
        if change == "committed":
            commit(repo)
        elif change == "staged":
            git(repo, "add", str(module))
    elif change in ("untracked", "ignored"):
        (repo / "src/new.py").write_text("VALUE = 2\n")
        if change == "ignored":
            (repo / ".git/info/exclude").write_text("src/new.py\n")
    elif change == "deleted":
        module.unlink()
    elif change == "index-flags":
        git(repo, "update-index", "--assume-unchanged", "src/pkg/module.py")
    elif change == "symlink":
        module.unlink()
        module.symlink_to(repo / "server.sh")
    else:
        module.chmod(0o755)
    with pytest.raises(SourceStateError, match="tree_changed"):
        verify_runtime_source(api, snapshot, container, tag="test")


def test_bytecode_uses_inspected_env(repo, tmp_path):
    api, snapshot, container = prepared(repo, tmp_path)
    (repo / "src/pkg/__pycache__").mkdir()
    (repo / "src/pkg/__pycache__/module.pyc").write_bytes(b"cache")
    verify_runtime_source(api, snapshot, container, tag="test")
    container["Config"]["Env"] = []
    with pytest.raises(SourceStateError, match="tree_changed"):
        verify_runtime_source(api, snapshot, container, tag="test")


@pytest.mark.parametrize("change", ["id", "image", "bind", "unclassified", "shadowed"])
def test_changed_target_or_mounts_refuse(repo, tmp_path, change):
    api, snapshot, container = prepared(repo, tmp_path)
    if change == "id":
        container["Id"] = "c" * 64
    elif change == "image":
        container["Image"] = "sha256:" + "d" * 64
    elif change == "bind":
        container["Mounts"][0]["Source"] += "-other"
    else:
        container["Mounts"].append({"Type": "bind", "Source": "/other",
                                    "Destination": "/app/src/pkg" if change == "shadowed" else "/other"})
    with pytest.raises(SourceStateError, match="tree_changed"):
        verify_runtime_source(api, snapshot, container, tag="test")


def test_missing_tested_sha_refuses_and_unguarded_snapshots_remain_valid(repo, tmp_path):
    api, snapshot, container = prepared(repo, tmp_path)
    with pytest.raises(SourceStateError, match="expected_runtime_sha"):
        bind_runtime_source(api, snapshot["spec"], {}, container, tag="test")
    del snapshot["runtime_source"]
    assert not valid_runtime_binding(snapshot)
    del snapshot["spec"]["runtime_source"]
    assert valid_runtime_binding(snapshot)


def test_spec_validation_and_capability(repo):
    spec = spec_for(repo)
    assert validate_spec(json.dumps(spec)) == spec
    assert WORKER_CAPABILITY_LABEL in spec_capability_labels(spec)
    assert WORKER_CAPABILITY_LABEL not in spec_capability_labels({"action": "restart"})
    for key, value in [("checker_image", "docker:latest"), ("project_dir", "relative"),
                       ("paths", ["../src"]), ("data_mounts", ["/app/src/pkg"])]:
        bad = copy.deepcopy(spec)
        bad["runtime_source"][key] = value
        with pytest.raises(ValueError):
            validate_runtime_source(bad)
    spec["action"] = "stop"
    with pytest.raises(ValueError):
        validate_spec(json.dumps(spec))


class RuntimeDocker(GitDocker):
    execute = DockerAPI.execute
    resolve_target = DockerAPI.resolve_target

    def __init__(self, scratch, container, *, on_restart=lambda: None):
        super().__init__(scratch)
        self.target, self.on_restart = container, on_restart
        self.side_effect_started = False
        self.restarts = []

    def request(self, method, path, **kwargs):
        if method == "GET" and path in ("/containers/app/json", f"/containers/{'a' * 64}/json"):
            return self.target
        if method == "POST" and path.startswith(f"/containers/{'a' * 64}/restart"):
            self.restarts.append(path)
            self.on_restart()
            return None
        return super().request(method, path, **kwargs)


def worker_run(tmp_path, snapshot, api, *, delay=lambda _: None, fence=None, clock=None):
    snapshot["execution"] = {"start_delay_seconds": 5, "timeout_seconds": 60, "max_output_bytes": 8192}
    path = tmp_path / "request.json"
    request = {"job_id": "runtime-test", "snapshot": snapshot}
    if fence is not None:
        request["fence"] = fence
    raw = json.dumps(request).encode()
    path.write_bytes(raw)
    run(path, hashlib.sha256(raw).hexdigest(), api_factory=lambda _: api, sleep=delay,
        **({"clock": clock} if clock is not None else {}))
    return json.loads((tmp_path / "receipt.json").read_text())


def test_worker_rechecks_after_start_delay_and_submits_nothing_on_drift(repo, tmp_path):
    _, snapshot, container = prepared(repo, tmp_path)
    api = RuntimeDocker(tmp_path / "git-scratch", container)
    result = worker_run(tmp_path, snapshot, api, delay=lambda _: (repo / "src/pkg/module.py").write_text("bad\n"))
    assert result["state"] == "failed" and not result["side_effect_unknown"]
    assert "tree_changed" in result["error"]
    assert api.restarts == []


def test_worker_receipt_proves_revision_and_final_identity(repo, tmp_path):
    _, snapshot, container = prepared(repo, tmp_path)
    api = RuntimeDocker(tmp_path / "git-scratch", container)
    result = worker_run(tmp_path, snapshot, api)
    assert result["state"] == "succeeded"
    assert result["runtime_source"] == result["runtime_source_after"] == snapshot["runtime_source"]
    assert snapshot["runtime_source"]["runtime_tree_hash"] in result["output_tail"]
    assert len(api.restarts) == 1


def test_completion_drift_is_not_success(repo, tmp_path):
    _, snapshot, container = prepared(repo, tmp_path)
    api = RuntimeDocker(tmp_path / "git-scratch", container,
                        on_restart=lambda: (repo / "src/pkg/module.py").write_text("late drift\n"))
    result = worker_run(tmp_path, snapshot, api)
    assert result["state"] == "lost" and result["side_effect_unknown"]
    assert "tree_changed" in result["error"]
    assert len(api.restarts) == 1


def test_source_check_cannot_outlive_quiesce_start_fence(repo, tmp_path):
    _, snapshot, container = prepared(repo, tmp_path)
    api = RuntimeDocker(tmp_path / "git-scratch", container)
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    times = iter([now, now + timedelta(seconds=11)])
    result = worker_run(tmp_path, snapshot, api,
                        fence={"start_not_after": (now + timedelta(seconds=10)).isoformat()},
                        clock=lambda: next(times))
    assert result["state"] == "failed" and not result["side_effect_unknown"]
    assert "expired during runtime check" in result["error"]
    assert api.restarts == []


def test_worker_image_contains_guard():
    dockerfile = (Path(__file__).parents[1] / "docker/Dockerfile.ops-worker").read_text()
    assert "src/llm_bawt/ops/runtime_source.py" in dockerfile
    assert f'LABEL {WORKER_CAPABILITY_LABEL}="1"' in dockerfile


def test_service_freezes_runtime_binding_and_rejects_missing_argument(repo, tmp_path):
    api, snapshot, container = prepared(repo, tmp_path)

    class RuntimeExecutor(FakeExecutor):
        def runtime_source_approval(self, spec, args):
            return bind_runtime_source(api, spec, args, container, tag="test")

    store = _store()
    op = store.create_operation({
        "slug": "test.runtime", "enabled": True, "command_script": json.dumps(snapshot["spec"]),
        "args_schema_json": json.dumps({"type": "object", "additionalProperties": False,
            "properties": {"expected_runtime_sha": {"type": "string", "pattern": "^[0-9a-f]{40}$"}},
            "required": ["expected_runtime_sha"]}),
    })
    service = OpsService(store, executor=RuntimeExecutor())
    frozen = service.prepare_invocation(op.slug, snapshot["resolved_args"])
    assert frozen["runtime_source"] == snapshot["runtime_source"]
    assert service._verify_snapshot(frozen, op.slug, snapshot["resolved_args"]) == frozen
    with pytest.raises(OpsDispatchError) as caught:
        service.prepare_invocation(op.slug, {})
    assert caught.value.code == "args_invalid"
    text = with_card_lines("Restart", frozen)
    assert frozen["runtime_source"]["expected_runtime_sha"] in text
    assert frozen["runtime_source"]["runtime_tree_hash"] in text
    with pytest.raises(ValueError, match="required expected_runtime_sha"):
        store.create_operation({"slug": "test.bad-runtime", "command_script": json.dumps(snapshot["spec"])})
