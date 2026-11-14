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
from llm_bawt.ops.executor import ConfigCheckError, spec_capability_labels, validate_spec
from llm_bawt.ops.runtime_source import (
    EXTENDED_CAPABILITY_LABEL, WORKER_CAPABILITY_LABEL, bind_runtime_source, valid_runtime_binding,
    validate_runtime_args, validate_runtime_source, verify_runtime_source,
)
from llm_bawt.ops.release import GitHubReleaseVerifier, ReleaseVerificationError
from llm_bawt.ops.source_state import SourceStateError
from llm_bawt.ops.worker import DockerAPI, run
from tests.test_ops_source_state import (
    GitDocker, IMAGE, _Forge, commit, git, repo as source_repo_fixture,
)
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
    with pytest.raises(SourceStateError, match="tree_changed") as caught:
        verify_runtime_source(api, snapshot, container, tag="test")
    expected_path = "src/new.py" if change in ("untracked", "ignored") else "src/pkg/module.py"
    assert expected_path in caught.value.names


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
        if method == "GET" and path in ("/containers/app/json", "/containers/exempt/json",
                                       f"/containers/{'a' * 64}/json"):
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


# TASK-1800: same checker and dispatch fence, extended for remaining source restarts.
def extended_spec(repo, *, mixed=False):
    spec = spec_for(repo)
    guard = spec["runtime_source"]
    guard.update(source_repository="example/service", source_branch="main")
    if mixed:
        del spec["container_name"]
        spec["container_name_from_arg"] = "target"
        spec["runtime_source"] = {"arg": "target", "targets": {"app": guard, "exempt": None}}
    return spec


def extended_schema(*, mixed=False):
    schema = {"type": "object", "additionalProperties": False, "properties": {
        "expected_runtime_sha": {"type": "string", "pattern": "^[0-9a-f]{40}$"},
        "allow_dirty": {"type": "boolean"}},
        "required": ["expected_runtime_sha"]}
    if mixed:
        schema["properties"]["target"] = {"type": "string", "enum": ["app", "exempt"]}
        schema["required"] = ["target"]
    return schema


def extended_service(repo, tmp_path, *, published=True, mixed=False, container=None):
    api = RuntimeDocker(tmp_path / "git-scratch", container or container_for(repo))
    api.scratch.mkdir(exist_ok=True)
    calls = []

    class RuntimeExecutor(FakeExecutor):
        def runtime_source_approval(self, spec, args):
            try:
                return bind_runtime_source(api, spec, args, api.target, tag="test")
            except SourceStateError as exc:
                raise ConfigCheckError(exc.code, str(exc)) from exc

    class Published:
        def branch_contains(self, repository, branch, sha):
            calls.append((repository, branch, sha))
            if isinstance(published, Exception):
                raise published
            return published

    store = _store()
    op = store.create_operation({
        "slug": "test.extended", "enabled": True,
        "command_script": json.dumps(extended_spec(repo, mixed=mixed)),
        "args_schema_json": json.dumps(extended_schema(mixed=mixed)),
        "args_defaults_json": json.dumps({"allow_dirty": False}),
    })
    return OpsService(store, executor=RuntimeExecutor(), release_verifier=Published()), op, api, calls


def test_extended_clean_uses_published_tested_commit_not_mutable_head(repo, tmp_path):
    revision = git(repo, "rev-parse", "HEAD").strip()
    (repo / "test_only.py").write_text("assert True\n")
    head = commit(repo)
    service, op, api, calls = extended_service(repo, tmp_path)
    snapshot = service.prepare_invocation(op.slug, {"expected_runtime_sha": revision})
    assert calls == [("example/service", "main", revision)]
    assert snapshot["runtime_source"]["observed_head"] == head
    assert snapshot["runtime_source"]["publication"] == "published"
    assert valid_runtime_binding(snapshot)
    assert service._verify_snapshot(snapshot, op.slug, {"expected_runtime_sha": revision}) == snapshot
    (repo / "test_only.py").write_text("assert 1\n")
    commit(repo)
    result = worker_run(tmp_path, snapshot, api)
    assert result["state"] == "succeeded"
    assert result["runtime_source_after"]["expected_runtime_sha"] == revision


@pytest.mark.parametrize("allow_dirty", [False, True])
@pytest.mark.parametrize("published", [False, ReleaseVerificationError("forge unavailable")])
def test_publication_refusals_and_explicit_unpublished_override(repo, tmp_path, allow_dirty, published):
    revision = git(repo, "rev-parse", "HEAD").strip()
    service, op, _, _ = extended_service(repo, tmp_path, published=published)
    args = {"expected_runtime_sha": revision, "allow_dirty": allow_dirty}
    if allow_dirty and published is False:
        snapshot = service.prepare_invocation(op.slug, args)
        assert snapshot["runtime_source"]["publication"] == "unpublished"
        assert "BREAK-GLASS / UNTESTED" in with_card_lines("Restart", snapshot)
    else:
        with pytest.raises(OpsDispatchError) as caught:
            service.prepare_invocation(op.slug, args)
        assert caught.value.code == ("source_not_published" if published is False else "source_unverifiable")


@pytest.mark.parametrize("change", ["tracked", "deleted", "ignored", "mode"])
def test_break_glass_captures_files_and_worker_does_not_report_tested_tree(repo, tmp_path, change):
    revision = git(repo, "rev-parse", "HEAD").strip()
    path = repo / "src/pkg/module.py"
    if change == "tracked":
        path.write_text("UNTESTED = True\n")
    elif change == "deleted":
        path.unlink()
    elif change == "mode":
        path.chmod(0o755)
    else:
        path = repo / "src/ignored.py"
        path.write_text("UNTESTED = True\n")
        (repo / ".git/info/exclude").write_text("src/ignored.py\n")
    service, op, api, _ = extended_service(repo, tmp_path)
    args = {"expected_runtime_sha": revision}
    with pytest.raises(OpsDispatchError) as caught:
        service.prepare_invocation(op.slug, args)
    assert str(path.relative_to(repo)) in str(caught.value)
    snapshot = service.prepare_invocation(op.slug, {**args, "allow_dirty": True})
    runtime = snapshot["runtime_source"]
    assert runtime["dirty_paths"] == [str(path.relative_to(repo))]
    assert valid_runtime_binding(snapshot)
    card = with_card_lines("Restart", snapshot)
    assert "BREAK-GLASS / UNTESTED" in card and runtime["dirty_paths"][0] in card
    result = worker_run(tmp_path, snapshot, api)
    assert result["state"] == "succeeded"
    assert result["runtime_source_after"] == runtime
    assert "BREAK-GLASS / UNTESTED" in result["runtime_source_warning"]
    assert "UNTESTED" in result["output_tail"]


@pytest.mark.parametrize("boundary", ["delay", "after"])
def test_dirty_override_does_not_waive_later_drift(repo, tmp_path, boundary):
    revision = git(repo, "rev-parse", "HEAD").strip()
    module = repo / "src/pkg/module.py"
    module.write_text("APPROVED_DIRTY = 1\n")
    service, op, api, _ = extended_service(repo, tmp_path)
    snapshot = service.prepare_invocation(op.slug, {"expected_runtime_sha": revision, "allow_dirty": True})
    drift = lambda: module.write_text("UNAPPROVED = 2\n")
    if boundary == "after":
        api.on_restart = drift
    result = worker_run(tmp_path, snapshot, api, delay=(lambda _: drift()) if boundary == "delay" else lambda _: None)
    assert result["state"] == ("failed" if boundary == "delay" else "lost")
    assert "tree_changed" in result["error"]
    assert len(api.restarts) == (0 if boundary == "delay" else 1)


def test_mixed_explicit_exemption_needs_no_sha_but_missing_map_never_exempts(repo, tmp_path):
    service, op, api, calls = extended_service(repo, tmp_path, mixed=True)
    snapshot = service.prepare_invocation(op.slug, {"target": "exempt"})
    assert snapshot["runtime_source"] is None and calls == []
    assert valid_runtime_binding(snapshot)
    assert worker_run(tmp_path, snapshot, api)["state"] == "succeeded"
    with pytest.raises(OpsDispatchError, match="expected_runtime_sha"):
        service.prepare_invocation(op.slug, {"target": "app"})
    bad = extended_spec(repo, mixed=True)
    del bad["runtime_source"]["targets"]["exempt"]
    with pytest.raises(ValueError, match="exactly cover"):
        validate_runtime_args(bad, extended_schema(mixed=True))
    snapshot["resolved_args"]["target"] = "missing"
    assert not valid_runtime_binding(snapshot)


@pytest.mark.parametrize("bad", ["unclassified", "selector", "legacy-map", "missing-boolean"])
def test_extended_catalog_validation_is_closed(repo, bad):
    spec, schema = extended_spec(repo, mixed=True), extended_schema(mixed=True)
    if bad == "unclassified":
        del spec["runtime_source"]["targets"]["exempt"]
    elif bad == "selector":
        spec["runtime_source"]["arg"] = "other"
    elif bad == "legacy-map":
        del spec["runtime_source"]["targets"]["app"]["source_branch"]
    else:
        del schema["properties"]["allow_dirty"]
    with pytest.raises(ValueError):
        validate_runtime_source(spec)
        validate_runtime_args(spec, schema)


def test_extended_capability_and_backend_bytecode_env(repo, tmp_path):
    spec, container = extended_spec(repo), container_for(repo)
    assert EXTENDED_CAPABILITY_LABEL in spec_capability_labels(spec)
    dockerfile = (Path(__file__).parents[1] / "docker/Dockerfile.ops-worker").read_text()
    assert f'LABEL {EXTENDED_CAPABILITY_LABEL}="1"' in dockerfile
    spec["runtime_source"]["mounts"]["/app/bawthub"] = spec["runtime_source"]["mounts"].pop("/app/src")
    container["Mounts"][0]["Destination"] = "/app/bawthub"
    container["Config"]["Env"] = ["PYTHONPYCACHEPREFIX=/tmp/bawthub-pycache"]
    (repo / "src/pkg/__pycache__").mkdir()
    (repo / "src/pkg/__pycache__/module.pyc").write_bytes(b"cache")
    scratch = tmp_path / "git-scratch"
    scratch.mkdir()
    args = {"expected_runtime_sha": git(repo, "rev-parse", "HEAD").strip(), "allow_dirty": True}
    binding = bind_runtime_source(GitDocker(scratch), spec, args, container, tag="test")
    assert binding["dirty_paths"] == []
    container["Config"]["Env"] = []
    with pytest.raises(SourceStateError, match="source_bytecode_unsafe"):
        bind_runtime_source(GitDocker(scratch), spec, args, container, tag="test")


@pytest.mark.parametrize("pythonpath", ["/data", "/data/plugins", "/app", "relative", ":/app/src"])
def test_data_mount_import_path_refuses_even_with_override(repo, tmp_path, pythonpath):
    spec, container = extended_spec(repo), container_for(repo)
    spec["runtime_source"]["data_mounts"].append("/app/storage")
    container["Config"]["Env"].append(f"PYTHONPATH={pythonpath}")
    args = {"expected_runtime_sha": git(repo, "rev-parse", "HEAD").strip(), "allow_dirty": True}
    with pytest.raises(SourceStateError, match="source_env_unsafe"):
        bind_runtime_source(GitDocker(tmp_path), spec, args, container, tag="test")


def test_dirty_card_list_is_counted_not_silently_truncated(repo, tmp_path):
    revision = git(repo, "rev-parse", "HEAD").strip()
    for i in range(25):
        (repo / f"src/new{i}.py").write_text("x = 1\n")
    service, op, _, _ = extended_service(repo, tmp_path)
    snapshot = service.prepare_invocation(op.slug, {"expected_runtime_sha": revision, "allow_dirty": True})
    card = with_card_lines("Restart", snapshot)
    assert "25 total" in card and "5 additional" in card


@pytest.mark.parametrize("status, base, expected", [
    ("behind", "a" * 40, True), ("identical", "a" * 40, True),
    ("ahead", "b" * 40, False), ("diverged", "b" * 40, False),
    ("behind", "b" * 40, False), ("missing", None, False),
])
def test_forge_published_ancestry_allows_older_tested_sha(status, base, expected):
    class Forge(_Forge):
        def open(self, request, timeout):
            if "/compare/" not in request.full_url:
                self.body = {"ref": "refs/heads/main", "object": {"type": "commit", "sha": "c" * 40}}
            elif status == "missing":
                self.status = 404
            else:
                self.body = {"status": status, "merge_base_commit": {"sha": base}}
            return super().open(request, timeout)
    forge = Forge()
    assert GitHubReleaseVerifier("", opener=forge).branch_contains("example/service", "main", "a" * 40) is expected
    assert all(auth is None for _, auth in forge.seen)


@pytest.mark.parametrize("status", [401, 403, 500])
def test_forge_comparison_failure_is_not_unpublished_override(status):
    class Forge(_Forge):
        def open(self, request, timeout):
            self.body = {"ref": "refs/heads/main", "object": {"type": "commit", "sha": "c" * 40}}
            if "/compare/" in request.full_url:
                self.status = status
            return super().open(request, timeout)
    with pytest.raises(ReleaseVerificationError):
        GitHubReleaseVerifier("", opener=Forge()).branch_contains("example/service", "main", "a" * 40)
