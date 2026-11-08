"""Wake-only host-network helper; fake network and Docker, never a real packet."""
from __future__ import annotations

import hashlib
import io
import json
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_bawt.ops import wake_on_lan as wol
from llm_bawt.ops.executor import ExecutorError, validate_spec
from llm_bawt.ops.gpu_guard import gpu_voice_target
from llm_bawt.ops.models import OpsOperation
from llm_bawt.ops.process_launch import WORKER_FILES
from llm_bawt.ops.self_target import self_target
from llm_bawt.ops.store import OpsStore
from llm_bawt.ops.worker import run
from tests.test_ops_executor import IMAGE, JOB, setup

SPEC = {
    "action": "wake_on_lan", "mac_address": "02:11:22:33:44:55",
    "broadcast_address": "192.0.2.255", "host_address": "192.0.2.10",
    "tcp_ports": [22], "health_port": 11434, "health_path": "/api/version",
    "health_field": "version", "wait_seconds": 10,
}


class Clock:
    now = 0

    def __call__(self):
        return self.now

    def sleep(self, value):
        self.now += value


def test_magic_packet_and_udp_broadcast(monkeypatch):
    calls = []

    class Socket:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            calls.append("closed")

        def setsockopt(self, *args):
            calls.append(("opt", args))

        def settimeout(self, value):
            assert value == 2

        def sendto(self, data, destination):
            calls.append((data, destination))

    monkeypatch.setattr(wol.socket, "socket", lambda *a: Socket())
    wol.send_packet(SPEC)
    packet = b"\xff" * 6 + bytes.fromhex("021122334455") * 16
    assert len(packet) == 102
    assert calls.count((packet, ("192.0.2.255", 9))) == 3
    assert ("opt", (wol.socket.SOL_SOCKET, wol.socket.SO_BROADCAST, 1)) in calls
    assert calls[-1] == "closed"


def test_already_awake_sends_nothing():
    result = wol.wake(SPEC, check=lambda *a, **k: ({"22": True}, True),
                      send=lambda _: pytest.fail("must not send"))
    assert result["ready"] and result["already_awake"] and not result["packet_sent"]


def test_waits_for_both_ssh_and_http_after_one_bounded_send():
    clock, sends, checks = Clock(), [], []
    answers = iter([({"22": False}, False), ({"22": True}, False), ({"22": True}, True)])

    def check(*a, **k):
        checks.append(clock())
        return next(answers)

    result = wol.wake(SPEC, clock=clock, sleep=clock.sleep, check=check,
                      send=lambda spec: sends.append(spec) or 3)
    assert result["ready"] and result["packet_sent"] and not result["already_awake"]
    assert sends == [SPEC] and len(checks) == 3 and clock() == 2


def test_ssh_alone_does_not_count_and_deadline_is_bounded():
    clock, sends = Clock(), []
    result = wol.wake(SPEC, clock=clock, sleep=clock.sleep,
                      check=lambda *a, **k: ({"22": True}, False),
                      send=lambda spec: sends.append(spec) or 3)
    assert not result["ready"] and result["packet_sent"]
    assert result["tcp_ready"] == {"22": True} and result["http_ready"] is False
    assert result["error"] == "readiness deadline exceeded" and clock() == 10
    assert sends == [SPEC]


def test_send_failure_reports_known_failure_without_waiting():
    def denied(_):
        raise OSError("internal details")

    result = wol.wake(SPEC, check=lambda *a, **k: ({"22": False}, False), send=denied)
    assert not result["packet_sent"] and result["error"] == "magic packet send failed"
    assert "internal details" not in json.dumps(result)


@pytest.mark.parametrize("status,body,healthy", [
    (200, b'{"version":"1.2.3"}', True),
    (200, b'{"version":true}', True),
    (200, b'{"version":1}', False),
    (200, b'{"version":""}', False),
    (200, b'{"wrong":"value"}', False),
    (200, b'[]', False),
    (200, b"<html>ok</html>", False),
    (200, b"x" * 4097, False),
    (302, b'{"version":"redirect"}', False),
    (503, b'{"version":"down"}', False),
])
def test_probe_requires_tcp_and_bounded_json_health(monkeypatch, status, body, healthy):
    requests, closed = [], []
    monkeypatch.setattr(wol.socket, "create_connection", lambda *a, **k: io.BytesIO())

    class HTTP:
        def __init__(self, host, port, timeout):
            assert (host, port) == (SPEC["host_address"], SPEC["health_port"])
            assert timeout <= 2

        def request(self, method, path, headers):
            requests.append((method, path))

        def getresponse(self):
            return SimpleNamespace(status=status, read=lambda size: body[:size])

        def close(self):
            closed.append(True)

    monkeypatch.setattr(wol.http.client, "HTTPConnection", HTTP)
    assert wol.probe(SPEC, 10, clock=lambda: 0) == ({"22": True}, healthy)
    assert requests == [("GET", "/api/version")] and closed == [True]


def test_probe_connection_errors_are_unready(monkeypatch):
    def fail(*a, **k):
        raise OSError("down")

    monkeypatch.setattr(wol.socket, "create_connection", fail)
    monkeypatch.setattr(wol.http.client.HTTPConnection, "request", fail)
    assert wol.probe(SPEC, 10, clock=lambda: 0) == ({"22": False}, False)
    assert wol.probe(SPEC, 0, clock=lambda: 0) == ({"22": False}, False)


@pytest.mark.parametrize("delta", [
    {"mac_address": "garbage"}, {"mac_address": "ff:ff:ff:ff:ff:ff"},
    {"mac_address": "00:00:00:00:00:00"}, {"host_address": "host.example"},
    {"host_address": "127.0.0.1"}, {"broadcast_address": "224.0.0.1"},
    {"tcp_ports": []}, {"tcp_ports": [22, 22]}, {"tcp_ports": [True]},
    {"tcp_ports": [65536]}, {"health_port": "11434"},
    {"health_path": "/api/version\r\nbad"}, {"health_path": "http://elsewhere/x"},
    {"health_field": "a.b"}, {"wait_seconds": True}, {"wait_seconds": 301},
    {"container_name": "other"}, {"shell": "shutdown"}, {"action": "power_off"},
])
def test_rejects_invalid_or_expanded_specs(delta):
    with pytest.raises(ValueError):
        validate_spec(json.dumps({**SPEC, **delta}))


def test_accepts_exact_fixed_spec():
    assert validate_spec(json.dumps(SPEC)) == SPEC


class HelperDocker:
    side_effect_started = False

    def __init__(self, result, *, broken=False, cleanup_broken=False):
        self.result, self.broken, self.cleanup_broken = result, broken, cleanup_broken
        self.calls = []

    def request(self, method, path, *, body=None):
        self.calls.append((method, path, body))
        if path.startswith("/containers/create?"):
            return {"Id": "helper"}
        if path.endswith("/start") and self.broken:
            raise OSError("start response lost")
        if path.endswith("/wait"):
            return {"StatusCode": 0 if self.result["ready"] else 1}
        if method == "DELETE" and self.cleanup_broken:
            raise OSError("cleanup unavailable")
        return None

    def attach_stream(self, _):
        data = json.dumps(self.result).encode()
        return io.BytesIO(bytes([1, 0, 0, 0]) + struct.pack(">I", len(data)) + data)


def snapshot():
    return {"spec": SPEC, "resolved_args": {}, "execution": {
        "worker_image": IMAGE, "start_delay_seconds": 0,
        "timeout_seconds": 40, "max_output_bytes": 4096,
    }}


def test_helper_is_host_network_but_has_no_socket_mounts_credentials_or_shell():
    api = HelperDocker({"ready": True, "packet_sent": True})
    receipt = {}
    assert "verified" in wol.run_worker(api, snapshot(), JOB, receipt)
    body = api.calls[0][2]
    assert body["Image"] == IMAGE
    assert body["Entrypoint"] == ["python3", "-I", "/ops/wake_on_lan.py"]
    assert json.loads(body["Cmd"][0]) == SPEC
    assert body["HostConfig"]["NetworkMode"] == "host"
    assert body["HostConfig"]["Binds"] == [] and body["Env"] == []
    assert body["HostConfig"]["CapDrop"] == ["ALL"]
    assert body["HostConfig"]["ReadonlyRootfs"] and not body["HostConfig"]["Privileged"]
    assert receipt["wake_on_lan"] == {"ready": True, "packet_sent": True}
    assert api.calls[-1][:2] == ("DELETE", "/containers/helper?force=1&v=1")


@pytest.mark.parametrize("ready", [False, True])
def test_worker_terminal_receipt_and_replay_fence(tmp_path, ready):
    api = HelperDocker({"ready": ready, "packet_sent": True})
    request = tmp_path / "request.json"
    raw = json.dumps({"job_id": JOB, "snapshot": snapshot()}).encode()
    request.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    assert run(request, digest, api_factory=lambda _t: api, sleep=lambda _: None) == (0 if ready else 1)
    receipt = json.loads(request.with_name("receipt.json").read_text())
    assert receipt["state"] == ("succeeded" if ready else "failed")
    assert receipt.get("side_effect_unknown", False) is False
    assert receipt["wake_on_lan"]["packet_sent"] is True
    before = len(api.calls)
    run(request, digest, api_factory=lambda _t: api)
    assert len(api.calls) == before


def test_lost_start_response_is_unknown_and_helper_cleanup_runs(tmp_path):
    api = HelperDocker({"ready": True, "packet_sent": True}, broken=True)
    request = tmp_path / "request.json"
    raw = json.dumps({"job_id": JOB, "snapshot": snapshot()}).encode()
    request.write_bytes(raw)
    assert run(request, hashlib.sha256(raw).hexdigest(), api_factory=lambda _t: api) == 1
    receipt = json.loads(request.with_name("receipt.json").read_text())
    assert receipt["state"] == "lost" and receipt["side_effect_unknown"] is True
    assert api.calls[-1][0] == "DELETE"


def test_no_local_container_or_gpu_target():
    assert gpu_voice_target(SPEC, {}, object(), live=True) is None
    assert self_target(snapshot(), own=lambda: object(), inspect=lambda _: pytest.fail("no target")) is None


def test_packaged_in_container_and_process_workers():
    dockerfile = (Path(__file__).resolve().parents[1] / "docker/Dockerfile.ops-worker").read_text()
    assert "src/llm_bawt/ops/wake_on_lan.py" in dockerfile and "wake_on_lan.py" in WORKER_FILES
    assert f'LABEL {wol.WORKER_CAPABILITY_LABEL}="1"' in dockerfile


@pytest.mark.parametrize("capable", [False, True])
def test_older_workers_refused_before_dispatch(tmp_path, monkeypatch, capable):
    executor, client, snap = setup(tmp_path, monkeypatch)
    labels = {wol.WORKER_CAPABILITY_LABEL: "1"} if capable else {}
    client.images = SimpleNamespace(get=lambda _: SimpleNamespace(attrs={"Config": {"Labels": labels}}))
    snap["spec"] = SPEC
    snap["execution"]["timeout_seconds"] = 40
    if capable:
        executor.check_target(SPEC, {})
        executor.dispatch(job_id=JOB, snapshot=snap)
        assert len(client.containers.creates) == 1
        assert client.containers.creates[0][1]["network_mode"] == "none"
    else:
        with pytest.raises(ExecutorError, match="wake-on-lan"):
            executor.dispatch(job_id=JOB, snapshot=snap)
        assert not client.containers.creates


@pytest.mark.parametrize("change", [
    {"timeout_seconds": 10},
    {"args_schema_json": json.dumps({"type": "object", "properties": {"host": {"type": "string"}}})},
])
def test_catalog_refuses_unbounded_budget_or_caller_targets(change):
    op = OpsOperation(id="test", slug="test.wake", command_script=json.dumps(SPEC),
                      timeout_seconds=40, args_schema_json='{"type":"object","properties":{}}',
                      args_defaults_json="{}")
    for key, value in change.items():
        setattr(op, key, value)
    with pytest.raises(ValueError, match="wake_on_lan"):
        OpsStore._validate_op(op)


def test_alarm_retains_sent_packet_and_last_readiness_state():
    calls = []

    def check(*a, **k):
        calls.append(True)
        if len(calls) > 1:
            raise TimeoutError("helper alarm")
        return {"22": False}, False

    result = wol.wake(SPEC, check=check, send=lambda _: 3)
    assert result["packet_sent"] is True and result["ready"] is False
    assert result["error"] == "readiness deadline exceeded"


def test_main_disarms_alarm_before_printing_known_result(monkeypatch, capsys):
    events = []
    monkeypatch.setattr(wol.sys, "argv", ["wake_on_lan.py", json.dumps(SPEC)])
    monkeypatch.setattr(wol.signal, "signal", lambda *a: None)
    monkeypatch.setattr(wol.signal, "alarm", lambda seconds: events.append(seconds))
    result = {"ready": False, "packet_sent": True}
    monkeypatch.setattr(wol, "wake", lambda spec: events.append("wake") or result)
    assert wol.main() == 1
    assert events == [11, "wake", 0]
    assert json.loads(capsys.readouterr().out) == result


def test_no_packet_after_initial_probe_exhausts_deadline():
    clock = Clock()

    def check(*a, **k):
        clock.now = 11
        return {"22": False}, False

    result = wol.wake(SPEC, clock=clock, check=check, send=lambda _: pytest.fail("too late"))
    assert not result["packet_sent"] and not result["ready"]


def test_partial_udp_failure_does_not_erase_a_successful_send(monkeypatch):
    class Socket:
        attempts = 0

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def setsockopt(self, *args):
            pass

        def settimeout(self, *_):
            pass

        def sendto(self, *_):
            self.attempts += 1
            if self.attempts != 1:
                raise OSError("later send failed")

    monkeypatch.setattr(wol.socket, "socket", lambda *a: Socket())
    assert wol.send_packet(SPEC) == 1


def test_cleanup_error_preserves_original_unknown_outcome(tmp_path):
    api = HelperDocker({"ready": True, "packet_sent": True}, broken=True, cleanup_broken=True)
    request = tmp_path / "request.json"
    raw = json.dumps({"job_id": JOB, "snapshot": snapshot()}).encode()
    request.write_bytes(raw)
    assert run(request, hashlib.sha256(raw).hexdigest(), api_factory=lambda _t: api) == 1
    receipt = json.loads(request.with_name("receipt.json").read_text())
    assert receipt["state"] == "lost" and receipt["side_effect_unknown"] is True
    assert receipt["wake_on_lan_cleanup"]["container_id"] == "helper"
    assert "start response lost" in receipt["error"]


def test_cleanup_error_cannot_claim_success():
    api = HelperDocker({"ready": True, "packet_sent": True}, cleanup_broken=True)
    receipt = {}
    with pytest.raises(wol.WakeNotReady, match="cleanup failed"):
        wol.run_worker(api, snapshot(), JOB, receipt)
    assert receipt["wake_on_lan"]["ready"] is True
    assert receipt["wake_on_lan_cleanup"]["removed"] is False


def test_catalog_default_timeout_must_cover_maximum_wait():
    op = OpsOperation(id="test", slug="test.wake", command_script=json.dumps({**SPEC, "wait_seconds": 300}),
                      timeout_seconds=300, args_schema_json='{"type":"object","properties":{}}',
                      args_defaults_json="{}")
    with pytest.raises(ValueError, match="plus 30s"):
        OpsStore._validate_op(op)
    op.timeout_seconds = 330
    OpsStore._validate_op(op)
