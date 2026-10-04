"""Docker-owned execution tests: fake SDK/API only; never a real socket."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_bawt.ops.executor import DockerExecutor, ExecutorError, validate_spec
from llm_bawt.ops.seeds import SEEDS
from llm_bawt.ops.worker import DockerAPI, atomic_json, run

JOB = "a" * 32
IMAGE = "sha256:" + "b" * 64


class NotFound(Exception):
    status_code = 404


class FakeContainer:
    def __init__(self, labels=None, status="created", mounts=None):
        self.labels = labels or {}
        self.status = status
        self.starts = 0
        self.attrs = {"Mounts": mounts or []}

    def reload(self):
        pass

    def start(self):
        self.starts += 1
        self.status = "running"


class FakeContainers:
    def __init__(self, root):
        self.rows = {"fake-app": FakeContainer(mounts=[{"Type": "volume", "Name": "receipts", "Destination": str(root), "RW": True}])}
        self.creates = []

    def get(self, name):
        if name not in self.rows:
            raise NotFound(name)
        return self.rows[name]

    def list(self, **kwargs):  # Docker SDK signature: list(all=..., filters=...)
        wanted = dict(item.split("=", 1) for item in (kwargs.get("filters") or {}).get("label", []))
        return [row for row in self.rows.values()
                if all(row.labels.get(k) == v for k, v in wanted.items())]

    def create(self, image, **kwargs):
        self.creates.append((image, kwargs))
        row = FakeContainer(kwargs["labels"])
        self.rows[kwargs["name"]] = row
        return row


def setup(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_BAWT_OPS_APP_CONTAINER", "fake-app")
    client = SimpleNamespace(containers=FakeContainers(tmp_path),
        images=SimpleNamespace(get=lambda _name: object()),
        volumes=SimpleNamespace(get=lambda _name: object()), ping=lambda: True)
    executor = DockerExecutor(client_factory=lambda: client, worker_image=IMAGE,
                              receipt_volume="receipts", receipt_root=str(tmp_path))
    snapshot = {"execution": {**executor.execution_settings(), "start_delay_seconds": 4,
                              "timeout_seconds": 3, "max_output_bytes": 4096},
                "spec": {"action": "restart", "container_name": "app"}, "resolved_args": {}}
    return executor, client, snapshot


def test_dispatch_is_accepted_not_success_and_daemon_owns_delay(tmp_path, monkeypatch):
    executor, client, snapshot = setup(tmp_path, monkeypatch)
    result = executor.dispatch(job_id=JOB, snapshot=snapshot)
    assert result.terminal_state is None
    assert result.exit_code is None
    assert Path(result.status_file_path).exists() is False
    image, config = client.containers.creates[0]
    assert image == IMAGE
    assert config["network_mode"] == "none"
    assert config["restart_policy"] == {"Name": "no"}
    assert config["volumes"]["receipts"]["bind"] == "/receipts"
    assert json.loads((tmp_path / JOB / "request.json").read_text())["snapshot"]["execution"]["start_delay_seconds"] == 4
    assert executor.reconcile(job_id=JOB, snapshot=snapshot).state == "accepted"


def test_replay_uses_same_worker_and_never_restarts_exited(tmp_path, monkeypatch):
    executor, client, snapshot = setup(tmp_path, monkeypatch)
    executor.dispatch(job_id=JOB, snapshot=snapshot)
    worker = client.containers.rows[executor.worker_name(JOB)]
    worker.status = "exited"
    executor.dispatch(job_id=JOB, snapshot=snapshot)
    assert worker.starts == 1
    assert len(client.containers.creates) == 1
    assert executor.reconcile(job_id=JOB, snapshot=snapshot).state == "lost"


def test_create_response_lost_reconciles_deterministic_worker(tmp_path, monkeypatch):
    executor, client, snapshot = setup(tmp_path, monkeypatch)
    original = client.containers.create
    def create(*args, **kwargs):
        original(*args, **kwargs)
        raise TimeoutError("response lost")
    client.containers.create = create
    with pytest.raises(ExecutorError):
        executor.dispatch(job_id=JOB, snapshot=snapshot)
    # Fresh executor/process sees deterministic created identity and starts it.
    fresh = DockerExecutor(client_factory=lambda: client)
    assert fresh.reconcile(job_id=JOB, snapshot=snapshot).state == "accepted"
    assert len(client.containers.creates) == 1
    assert client.containers.rows[executor.worker_name(JOB)].starts == 1


def test_missing_worker_never_blindly_recreated(tmp_path, monkeypatch):
    executor, client, snapshot = setup(tmp_path, monkeypatch)
    assert executor.reconcile(job_id=JOB, snapshot=snapshot).state == "lost"
    assert not client.containers.creates
    with pytest.raises(ExecutorError, match="abandoned"):
        executor.dispatch(job_id=JOB, snapshot=snapshot)
    assert not client.containers.creates


def test_wrong_mount_or_mutable_image_fails_before_create(tmp_path, monkeypatch):
    executor, client, snapshot = setup(tmp_path, monkeypatch)
    snapshot["execution"]["worker_image"] = "python:latest"
    with pytest.raises(ExecutorError, match="immutable"):
        executor.dispatch(job_id=JOB, snapshot=snapshot)
    snapshot["execution"]["worker_image"] = IMAGE
    client.containers.rows["fake-app"].attrs["Mounts"] = []
    with pytest.raises(ExecutorError, match="named volume"):
        executor.dispatch(job_id=JOB, snapshot=snapshot)
    assert not client.containers.creates


def test_request_conflict_fails_before_second_action(tmp_path, monkeypatch):
    executor, client, snapshot = setup(tmp_path, monkeypatch)
    executor.dispatch(job_id=JOB, snapshot=snapshot)
    snapshot["spec"]["action"] = "stop"
    with pytest.raises(ExecutorError, match="request conflict"):
        executor.dispatch(job_id=JOB, snapshot=snapshot)
    assert len(client.containers.creates) == 1


class FakeAPI:
    calls = 0
    side_effect_started = False
    def __init__(self, timeout):
        self.timeout = timeout
    def execute(self, spec, args):
        type(self).calls += 1
        self.side_effect_started = True
        return "completed"


def test_standalone_worker_receipt_and_no_replay_after_restart(tmp_path, monkeypatch):
    executor, client, snapshot = setup(tmp_path, monkeypatch)
    executor.dispatch(job_id=JOB, snapshot=snapshot)
    request = tmp_path / JOB / "request.json"
    digest = hashlib.sha256(request.read_bytes()).hexdigest()
    FakeAPI.calls = 0
    def delay(seconds):
        assert seconds == 4
        receipt = json.loads(request.with_name("receipt.json").read_text())
        assert receipt["state"] == "accepted"
        assert FakeAPI.calls == 0
    assert run(request, digest, api_factory=FakeAPI, sleep=delay) == 0
    # App gone/recreated: only persisted receipt is needed, even worker removed.
    client.containers.rows.pop(executor.worker_name(JOB))
    assert executor.reconcile(job_id=JOB, snapshot=snapshot).state == "succeeded"
    # A complete persisted receipt can reconcile even while Docker is offline.
    def offline(_name):
        raise ConnectionError("daemon offline")
    client.containers.get = offline
    assert executor.reconcile(job_id=JOB, snapshot=snapshot).state == "succeeded"
    assert run(request, digest, api_factory=FakeAPI, sleep=delay) == 0
    assert FakeAPI.calls == 1


def test_worker_interrupted_during_action_is_lost_not_replayed(tmp_path, monkeypatch):
    executor, _, snapshot = setup(tmp_path, monkeypatch)
    executor.dispatch(job_id=JOB, snapshot=snapshot)
    request = tmp_path / JOB / "request.json"
    digest = hashlib.sha256(request.read_bytes()).hexdigest()
    atomic_json(request.with_name("receipt.json"), {"job_id": JOB, "request_hash": digest,
        "state": "running", "started_at": "2026-01-01T00:00:00Z"})
    FakeAPI.calls = 0
    assert run(request, digest, api_factory=FakeAPI, sleep=lambda _: None) == 1
    assert FakeAPI.calls == 0
    assert executor.reconcile(job_id=JOB, snapshot=snapshot).state == "lost"


def test_real_execution_deadline_not_stop_grace(tmp_path, monkeypatch):
    import time
    executor, _, snapshot = setup(tmp_path, monkeypatch)
    snapshot["execution"]["timeout_seconds"] = 0.02
    executor.dispatch(job_id=JOB, snapshot=snapshot)
    request = tmp_path / JOB / "request.json"
    digest = hashlib.sha256(request.read_bytes()).hexdigest()
    class SlowAPI(FakeAPI):
        def execute(self, spec, args):
            self.side_effect_started = True
            time.sleep(2)
    start = time.monotonic()
    assert run(request, digest, api_factory=SlowAPI, sleep=lambda _: None) == 1
    assert time.monotonic() - start < 1
    receipt = json.loads(request.with_name("receipt.json").read_text())
    assert receipt["state"] == "timed_out"
    assert receipt["side_effect_unknown"] is True


def test_uncertain_start_attempt_is_not_retried(tmp_path, monkeypatch):
    executor, client, snapshot = setup(tmp_path, monkeypatch)
    original = client.containers.create
    def create(*args, **kwargs):
        row = original(*args, **kwargs)
        def fail_start():
            row.starts += 1
            raise TimeoutError("lost start response")
        row.start = fail_start
        return row
    client.containers.create = create
    with pytest.raises(ExecutorError):
        executor.dispatch(job_id=JOB, snapshot=snapshot)
    assert executor.reconcile(job_id=JOB, snapshot=snapshot).state == "lost"
    assert client.containers.rows[executor.worker_name(JOB)].starts == 1


def test_two_created_observers_cannot_restart_fast_exited_worker(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    executor, client, snapshot = setup(tmp_path, monkeypatch)
    original = client.containers.create
    def create(*args, **kwargs):
        row = original(*args, **kwargs)
        def fast_start():
            row.starts += 1
            row.status = "exited"
        row.start = fast_start
        return row
    client.containers.create = create
    with ThreadPoolExecutor(2) as pool:
        list(pool.map(lambda _: executor.dispatch(job_id=JOB, snapshot=snapshot), range(2)))
    assert len(client.containers.creates) == 1
    assert client.containers.rows[executor.worker_name(JOB)].starts == 1


@pytest.mark.parametrize("action", ["restart", "start", "stop", "pull"])
def test_docker_api_actions_and_separate_grace(action):
    api = DockerAPI(timeout=300)
    calls = []
    def request(method, path):
        calls.append((method, path))
        if method == "GET":
            return {"Id": "target-id", "Config": {"Image": "example:tag"}}
    api.request = request
    api.execute({"action": action, "container_name": "app", "stop_grace_seconds": 7}, {})
    assert calls[-1][0] == "POST"
    if action in ("restart", "stop"):
        assert calls[-1][1].endswith("?t=7")
    assert "300" not in calls[-1][1]


def test_compose_selector_excludes_ops_worker_with_inherited_target_labels():
    api = DockerAPI(timeout=300)
    calls = []

    def request(method, path):
        calls.append((method, path))
        if method == "GET" and path.startswith("/containers/json?"):
            return [
                {
                    "Id": "worker-id",
                    "Labels": {
                        "com.docker.compose.project": "llm-bawt",
                        "com.docker.compose.service": "app",
                        "llm-bawt.ops.job": "job-id",
                    },
                },
                {
                    "Id": "app-id",
                    "Labels": {
                        "com.docker.compose.project": "llm-bawt",
                        "com.docker.compose.service": "app",
                    },
                },
            ]
        if method == "GET" and path == "/containers/app-id/json":
            return {"Id": "app-id", "Config": {"Image": "example:tag"}}
        return None

    api.request = request
    output = api.execute({
        "action": "restart",
        "compose_project": "llm-bawt",
        "compose_service": "app",
    }, {})

    assert output == "Docker restart completed for app-id"
    assert calls[-1] == ("POST", "/containers/app-id/restart?t=10")


def test_dispatched_worker_overrides_inherited_compose_identity(tmp_path, monkeypatch):
    executor, client, snapshot = setup(tmp_path, monkeypatch)
    executor.dispatch(job_id=JOB, snapshot=snapshot)

    labels = client.containers.creates[0][1]["labels"]
    assert labels["com.docker.compose.project"] == "llm-bawt-ops-worker"
    assert labels["com.docker.compose.service"] == "ops-worker"
    assert labels["llm-bawt.ops.job"] == JOB


def test_restart_app_seed_targets_stable_explicit_container_name():
    operation = next(
        row for row in SEEDS
        if row["slug"] == "llm-bawt.restart-app"
    )
    assert json.loads(operation["command_script"]) == {
        "action": "restart",
        "container_name": "llm-bawt-app",
    }


@pytest.mark.parametrize("spec", [{}, [], {"action": "nuke", "container_name": "x"},
    {"action": "restart", "compose_service": "app"},
    {"action": "restart", "container_name": "x", "container_name_from_arg": "name"},
    {"action": "restart", "container_name": "x", "unexpected": True}])
def test_invalid_spec_rejected(spec):
    with pytest.raises(ValueError):
        validate_spec(json.dumps(spec))


# --- TASK-1002: approval-time target check ---------------------------------

def test_check_target_accepts_container_on_this_daemon(tmp_path, monkeypatch):
    executor, _client, _ = setup(tmp_path, monkeypatch)
    executor.check_target({"action": "restart", "container_name_from_arg": "container"},
                          {"container": "fake-app"})


def test_check_target_rejects_container_on_another_host(tmp_path, monkeypatch):
    # The live bug: NginxProxyManager lives on Unraid, not echo's daemon.
    executor, _client, _ = setup(tmp_path, monkeypatch)
    with pytest.raises(ExecutorError, match="'NginxProxyManager' does not exist"):
        executor.check_target({"action": "restart", "container_name_from_arg": "container"},
                              {"container": "NginxProxyManager"})


def test_check_target_compose_selector_ignores_ops_workers(tmp_path, monkeypatch):
    executor, client, _ = setup(tmp_path, monkeypatch)
    spec = {"action": "restart", "compose_project": "llm-bawt", "compose_service": "app"}
    compose = {"com.docker.compose.project": "llm-bawt", "com.docker.compose.service": "app"}
    # A worker carrying inherited Compose labels alone must not satisfy the check.
    client.containers.rows["worker"] = FakeContainer({**compose, "llm-bawt.ops.job": JOB})
    with pytest.raises(ExecutorError, match="matched 0 containers"):
        executor.check_target(spec, {})
    client.containers.rows["llm-bawt-app-1"] = FakeContainer(compose)
    executor.check_target(spec, {})


def test_check_target_daemon_error_is_executor_error(tmp_path, monkeypatch):
    executor, client, _ = setup(tmp_path, monkeypatch)

    def boom(_name):
        raise RuntimeError("socket closed")

    client.containers.get = boom
    with pytest.raises(ExecutorError, match="cannot inspect target container"):
        executor.check_target({"action": "restart", "container_name": "x"}, {})


def test_container_restart_seed_lists_only_local_targets():
    seed = next(s for s in SEEDS if s["slug"] == "container.restart")
    enum = json.loads(seed["args_schema_json"])["properties"]["container"]["enum"]
    assert "NginxProxyManager" not in enum
    assert "bawthub-public-site" not in enum
