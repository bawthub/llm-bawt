"""TASK-1697: a plain restart verifies the service actually came back."""
from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_bawt.ops.executor import ExecutorError, validate_spec
from llm_bawt.ops.process_launch import WORKER_FILES
from llm_bawt.ops.restart_health import HEALTH_KEYS, WORKER_CAPABILITY_LABEL, RestartUnhealthy
from llm_bawt.ops.seeds import SEEDS
from llm_bawt.ops.worker import DockerAPI, run
from tests.test_ops_executor import JOB, setup

HEALTH = {"health_url": "http://127.0.0.1:5173/api/health", "health_field": "status",
          "health_value": "healthy", "health_timeout_seconds": 30}
SPEC = {"action": "restart", "compose_project": "bawthub", "compose_service": "frontend", **HEALTH}


def _frame(body: bytes) -> bytes:
    return bytes([1, 0, 0, 0]) + struct.pack(">I", len(body)) + body


class FakeDocker(DockerAPI):
    """Answers the health exec with a scripted sequence of (exit, body)."""

    def __init__(self, answers, *, running=True):
        self.now = 0.0
        super().__init__(timeout=300, clock=lambda: self.now, sleep=self._sleep)
        self.answers, self.running, self.calls = list(answers), running, []

    def _sleep(self, seconds):
        self.now += seconds

    def request(self, method, path, *, body=None, headers=None, raw=False):
        self.calls.append((method, path))
        if path.startswith("/containers/json?"):
            return [{"Id": "fe", "Labels": {}}]
        if method == "GET" and path == "/containers/fe/json":
            return {"Id": "fe", "Config": {"Image": "x"},
                    "State": {"Running": self.running, "Status": "running" if self.running else "exited"}}
        if path.endswith("/exec"):
            assert body["Cmd"][-1] == HEALTH["health_url"]
            return {"Id": "ex"}
        if path == "/exec/ex/start":
            self._current = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
            return _frame(self._current[1])
        if path == "/exec/ex/json":
            return {"ExitCode": self._current[0]}
        return None


def test_restart_succeeds_only_after_health_matches():
    api = FakeDocker([(7, b""), (0, b'{"status":"degraded"}'), (0, b'{"status":"healthy"}')])
    out = api.execute(SPEC, {})
    assert "Docker restart completed for fe" in out and "health status matched" in out
    assert ("POST", "/containers/fe/restart?t=10") in api.calls


def test_the_0052_outage_shape_fails_the_job_as_a_known_outcome():
    # Vite answers but the proxied Hono API is down: curl -f exits 22 forever.
    api = FakeDocker([(22, b"")])
    with pytest.raises(RestartUnhealthy, match="not healthy within 30s.*exited 22") as info:
        api.execute(SPEC, {})
    assert info.value.outcome_known is True
    assert api.side_effect_started is True
    assert api.now >= 30


def test_a_container_that_exits_fails():
    api = FakeDocker([(0, b'{"status":"healthy"}')], running=False)
    with pytest.raises(RestartUnhealthy, match="not running"):
        api.execute(SPEC, {})


def test_response_body_is_never_echoed():
    api = FakeDocker([(0, b'{"status":"SECRETVALUE"}')])
    with pytest.raises(RestartUnhealthy) as info:
        api.execute(SPEC, {})
    assert "SECRETVALUE" not in str(info.value)


def test_worker_receipt_is_failed_not_lost(tmp_path):
    request = {"job_id": "j1", "snapshot": {
        "spec": SPEC, "resolved_args": {},
        "execution": {"start_delay_seconds": 0, "timeout_seconds": 120, "max_output_bytes": 4096}}}
    path = tmp_path / "request.json"
    raw = json.dumps(request).encode()
    path.write_bytes(raw)
    code = run(path, hashlib.sha256(raw).hexdigest(), api_factory=lambda _t: FakeDocker([(22, b"")]),
               sleep=lambda _s: None)
    receipt = json.loads((tmp_path / "receipt.json").read_text())
    assert code == 1
    assert receipt["state"] == "failed" and receipt["side_effect_unknown"] is False


def test_spec_without_health_is_unchanged():
    api = FakeDocker([(22, b"")])
    plain = {k: v for k, v in SPEC.items() if k not in HEALTH_KEYS}
    assert api.execute(plain, {}) == "Docker restart completed for fe"
    assert not any(p.endswith("/exec") for _, p in api.calls)


def test_validate_accepts_the_complete_health_set():
    assert validate_spec(json.dumps(SPEC)) == SPEC


@pytest.mark.parametrize("bad", [
    {"health_url": "http://127.0.0.1:5173/x"},  # partial set
    {**HEALTH, "health_url": "http://10.0.0.1:80/health"},
    {**HEALTH, "health_url": "file:///etc/passwd"},
    {**HEALTH, "health_value": "a b"},
    {**HEALTH, "health_value": 1},
    {**HEALTH, "health_timeout_seconds": 5},
    {**HEALTH, "health_timeout_seconds": "30"},
])
def test_validate_rejects_bad_health(bad):
    spec = {"action": "restart", "container_name": "x", **bad}
    with pytest.raises(ValueError):
        validate_spec(json.dumps(spec))


def test_health_only_on_restart_or_start():
    with pytest.raises(ValueError):
        validate_spec(json.dumps({"action": "stop", "container_name": "x", **HEALTH}))


@pytest.mark.parametrize("slug", ["bawthub.restart-frontend", "llm-bawt.restart-app"])
def test_restart_seeds_carry_a_probe_that_fits_the_job_timeout(slug):
    row = next(r for r in SEEDS if r["slug"] == slug)
    spec = validate_spec(row["command_script"])
    assert HEALTH_KEYS <= set(spec)
    budget = row["start_delay_seconds"] + spec.get("stop_grace_seconds", 10) + spec["health_timeout_seconds"]
    assert budget < row["timeout_seconds"]


def test_frontend_seed_probes_the_api_not_just_vite():
    row = next(r for r in SEEDS if r["slug"] == "bawthub.restart-frontend")
    assert json.loads(row["command_script"])["health_url"].endswith("/api/health")


def test_worker_image_carries_the_module_and_the_label():
    dockerfile = (Path(__file__).resolve().parents[1] / "docker" / "Dockerfile.ops-worker").read_text()
    assert "src/llm_bawt/ops/restart_health.py" in dockerfile
    assert f'LABEL {WORKER_CAPABILITY_LABEL}="1"' in dockerfile
    assert "restart_health.py" in WORKER_FILES


@pytest.mark.parametrize("labels,ok", [({}, False), ({WORKER_CAPABILITY_LABEL: "1"}, True)])
def test_dispatch_refuses_a_health_spec_on_an_older_worker(tmp_path, monkeypatch, labels, ok):
    executor, client, snapshot = setup(tmp_path, monkeypatch)
    client.images = SimpleNamespace(get=lambda _n: SimpleNamespace(attrs={"Config": {"Labels": labels}}))
    snapshot["spec"] = {"action": "restart", "container_name": "app", **HEALTH}
    if ok:
        executor.dispatch(job_id=JOB, snapshot=snapshot)
        assert len(client.containers.creates) == 1
    else:
        with pytest.raises(ExecutorError, match="restart-health"):
            executor.dispatch(job_id=JOB, snapshot=snapshot)
        assert client.containers.creates == []
