"""Ops-owned health: bounded cached reads, no execution or Docker API proxy."""
from __future__ import annotations

import json
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from unittest.mock import Mock

import pytest

from llm_bawt.ops.health_collector import ISOLATED_CAPABILITIES, OpsHealthCollector, observe_container
from llm_bawt.ops.health_http import OpsHealthServer
from llm_bawt.ops.health_runtime import OpsHealthRuntime
from llm_bawt.ops.health_state import ExecutorModeSample, HealthSample, OpsHealthState
from llm_bawt.ops.reconciler import OpsReconciler

HANG_GUARD_S = 10.0  # thread handshakes; only reached if one hangs
STALL_S = 10.0  # a stalled collector/handler: well past close()'s 2s join deadline


def _stall(entered, release):
    entered.set()
    # timing-ok: deliberate stall; returning on timeout only lets the thread exit
    release.wait(STALL_S)


def _until_stopped(stop):
    assert stop.wait(HANG_GUARD_S), "runtime never signalled stop"


class Clock:
    def __init__(self):
        self.elapsed = 100.0
        self.wall = datetime(2026, 10, 18, tzinfo=UTC)

    def advance(self, seconds):
        self.elapsed += seconds
        self.wall += timedelta(seconds=seconds)

    def state(self):
        return OpsHealthState(clock=lambda: self.elapsed, utcnow=lambda: self.wall)


def seed(state):
    for name in ("database", "docker", "ops.executor", "reconciler.loop"):
        state.publish(name, HealthSample(state.sample_time(), "ok"))


def attrs(name="/app", status="running", health="healthy"):
    return {"Name": name, "Id": "secret-id", "State": {
        "Status": status, "StartedAt": "2026-10-18T20:00:00.123456789Z",
        **({"Health": {"Status": health, "Log": ["secret log"]}} if health else {})},
        "Config": {"Env": ["SECRET=password"]}, "Mounts": ["secret path"],
        "NetworkSettings": {"Ports": "secret port"}, "Labels": {"private": "secret"}}


def docker(rows=None, *, error=None):
    rows = rows if rows is not None else [attrs()]
    api = SimpleNamespace(
        containers=Mock(return_value=[{"Id": str(i)} for i in range(len(rows))]),
        inspect_container=Mock(side_effect=error or (lambda i: rows[int(i)])),
    )
    image = SimpleNamespace(attrs={"Config": {"Labels": {label: "1" for _, label in ISOLATED_CAPABILITIES}}})
    return SimpleNamespace(api=api, close=Mock(), images=SimpleNamespace(get=Mock(return_value=image)),
                            create=Mock(side_effect=AssertionError("no mutation")))


@pytest.fixture
def collector(monkeypatch):
    clock = Clock()
    state = clock.state()
    client = docker()
    readiness = Mock(return_value=None)
    monkeypatch.setattr("llm_bawt.ops.health_collector.DockerExecutor.readiness", readiness)
    monkeypatch.setattr("llm_bawt.ops.health_collector.DockerExecutor.preflight",
                        Mock(side_effect=AssertionError("health must not call preflight")))
    return clock, state, client, OpsHealthCollector(state, lambda: None, client_factory=lambda: client), readiness


def request(server, path, *, method="GET"):
    url = f"http://127.0.0.1:{server.address[1]}{path}"
    try:
        response = urlopen(Request(url, method=method), timeout=2)
    except HTTPError as exc:
        response = exc
    with response:
        return response.status, json.loads(response.read())


def test_inventory_is_fixed_projection_and_uses_start_time(collector):
    clock, state, client, probe, _ = collector
    started = state.sample_time().iso
    original = client.api.inspect_container.side_effect
    client.api.inspect_container.side_effect = lambda i: (clock.advance(4), original(i))[1]
    probe.collect_docker()
    code, body = state.containers()
    assert code == 200 and body["checkedAt"] == started
    assert body["containers"] == [{"name": "app", "state": "running", "health": "healthy",
                                   "startedAt": "2026-10-18T20:00:00.123456+00:00"}]
    assert "secret" not in json.dumps(body).lower()
    client.api.containers.assert_called_once_with(all=True, limit=1001)
    client.api.inspect_container.assert_called_once_with("0")
    client.create.assert_not_called()
    client.close.assert_called_once()


@pytest.mark.parametrize("changed", [
    {"Name": "/bad?query"}, {"State": {"Status": "future-new-state"}},
    {"State": {"Status": "running", "Health": {"Status": "unknown-secret"}}},
    {"State": {"Status": "running", "StartedAt": "no timestamp"}},
])
def test_malformed_inventory_fails_closed_without_partial_rows(collector, changed):
    _, state, client, probe, _ = collector
    client.api.inspect_container.side_effect = lambda _i: {**attrs(), **changed}
    probe.collect_docker()
    code, body = state.containers()
    assert code == 503 and body["containers"] == []
    assert body["reason"] == "container observation unavailable"


def test_never_started_and_no_docker_healthcheck_are_null():
    row = attrs(health=None)
    row["State"]["StartedAt"] = "0001-01-01T00:00:00Z"
    observation = observe_container(row)
    assert observation.started_at is None and observation.health is None


def test_inventory_failure_recovery_and_size_budget(collector):
    clock, state, client, probe, _ = collector
    client.api.inspect_container.side_effect = RuntimeError("secret Docker response")
    probe.collect_docker()
    assert state.containers()[0] == 503
    assert "secret" not in json.dumps(state.ready())
    client.api.inspect_container.side_effect = lambda _i: attrs()
    probe.collect_docker()
    assert state.containers()[0] == 200
    client.api.containers.return_value = [{"Id": "0"}] * 1001
    client.api.inspect_container.reset_mock()
    probe.collect_docker()
    assert state.containers()[0] == 503
    client.api.inspect_container.assert_not_called()
    client.api.containers.return_value = [{"Id": "0"}]
    client.api.inspect_container.side_effect = lambda _i: (clock.advance(16), attrs())[1]
    probe.collect_docker()
    assert state.containers()[0] == 503


def test_duplicate_names_fail_closed(collector):
    _, state, client, probe, _ = collector
    client.api.containers.return_value = [{"Id": "0"}, {"Id": "1"}]
    client.api.inspect_container.side_effect = lambda _i: attrs()
    probe.collect_docker()
    assert state.containers()[0] == 503


def test_inventory_remains_fresh_despite_failed_readiness(collector):
    _, state, _, probe, readiness = collector
    readiness.return_value = "private mounts and credentials"
    probe.collect_docker()
    assert state.containers()[0] == 200
    code, body = state.ready()
    assert code == 503
    assert "private" not in json.dumps(body)
    assert next(c for c in body["checks"] if c["name"] == "ops.executor")["status"] == "down"


def test_isolated_mode_reports_only_first_party_capabilities(collector):
    _, state, client, probe, _ = collector
    probe.collect_docker()
    modes = next(check["modes"] for check in state.ready()[1]["checks"]
                 if check["name"] == "ops.executor")
    assert modes["isolated"] == {
        "runtime": "isolated", "status": "ok", "reason": None,
        "workerImageAvailable": True,
        "capabilityLabels": {name: True for name, _ in ISOLATED_CAPABILITIES},
    }
    assert modes["process"]["status"] == "down"  # pump has not published yet
    assert client.images.get.call_count == 1
    client.images.get.return_value.attrs["Config"]["Labels"] = {"private": "secret value"}
    probe.collect_docker()
    available = next(check["modes"]["isolated"] for check in state.ready()[1]["checks"]
                     if check["name"] == "ops.executor")
    assert available["status"] == "ok" and not any(available["capabilityLabels"].values())
    assert "private" not in json.dumps(available)
    client.images.get.side_effect = RuntimeError("secret image identifier")
    probe.collect_docker()
    isolated = next(check["modes"]["isolated"] for check in state.ready()[1]["checks"]
                    if check["name"] == "ops.executor")
    assert isolated["status"] == "down" and isolated["capabilityLabels"] is None
    assert "secret" not in json.dumps(isolated)


def test_mode_diagnostics_are_non_gating_and_stale_counters_are_hidden():
    clock = Clock()
    state = clock.state()
    seed(state)
    state.publish_mode("process", ExecutorModeSample(state.sample_time(), "ok", live_children=2,
                                                     bound=2, oldest_started=clock.elapsed - 6))
    code, body = state.ready()
    assert code == 200 and body["status"] == "ok"  # isolated was never collected
    modes = next(check["modes"] for check in body["checks"] if check["name"] == "ops.executor")
    assert modes["process"]["liveChildren"] == 2 and modes["process"]["bound"] == 2
    assert modes["process"]["oldestAgeSeconds"] == 6
    assert modes["isolated"]["status"] == "down"
    clock.advance(31)
    seed(state)  # fresh gates, but old mode sample cannot look live
    code, body = state.ready()
    assert code == 200
    process = next(check["modes"]["process"] for check in body["checks"]
                   if check["name"] == "ops.executor")
    assert process["status"] == "down" and "stale" in process["reason"]
    assert process["liveChildren"] is process["bound"] is process["oldestAgeSeconds"] is None


def test_reconciler_publishes_its_own_process_launcher():
    clock = Clock()
    state = clock.state()
    seed(state)
    executor = SimpleNamespace(process_health=Mock(return_value=(2, 4, 94.0, True)))
    service = SimpleNamespace(executors=lambda: {"docker": executor}, reconcile_active_jobs=Mock(return_value=0))
    stop = SimpleNamespace(is_set=Mock(side_effect=[False, True]), wait=Mock())
    OpsReconciler(service, interval=0).run(stop, state)
    assert executor.process_health.call_count == 1
    mode = next(check["modes"]["process"] for check in state.ready()[1]["checks"]
                if check["name"] == "ops.executor")
    assert (mode["liveChildren"], mode["bound"], mode["oldestAgeSeconds"]) == (2, 4, 6)
    executor.process_health.return_value = (0, 4, None, False)
    stop.is_set = Mock(side_effect=[False, True])
    OpsReconciler(service, interval=0).run(stop, state)
    mode = next(check["modes"]["process"] for check in state.ready()[1]["checks"]
                if check["name"] == "ops.executor")
    assert mode["status"] == "down" and mode["liveChildren"] == 0
    assert state.ready()[0] == 200


def test_database_collection_is_read_only_and_failure_is_sanitized():
    state = Clock().state()
    conn = Mock()
    conn.execute.return_value.scalar_one.return_value = 1
    context = Mock(__enter__=Mock(return_value=conn), __exit__=Mock(return_value=False))
    engine = SimpleNamespace(connect=lambda: context, dialect=SimpleNamespace(name="postgresql"))
    probe = OpsHealthCollector(state, lambda: engine)
    probe.collect_database()
    assert [str(call.args[0]) for call in conn.execute.call_args_list] == [
        "SET TRANSACTION READ ONLY", "SET LOCAL statement_timeout = '2000ms'",
        "SET LOCAL lock_timeout = '2000ms'", "SELECT 1"]
    assert state.ready()[1]["checks"][0]["status"] == "ok"
    conn.execute.side_effect = RuntimeError("postgres://private-password")
    probe.collect_database()
    assert state.ready()[1]["checks"][0]["status"] == "down"
    assert "private-password" not in json.dumps(state.ready())
    conn.execute.side_effect = None
    probe.collect_database()
    assert state.ready()[1]["checks"][0]["status"] == "ok"


def test_startup_stale_future_and_shutdown_are_never_ready():
    clock = Clock()
    state = clock.state()
    assert state.ready()[0] == state.containers()[0] == 503
    seed(state)
    assert state.ready()[0] == 200
    clock.advance(31)
    assert state.ready()[0] == state.containers()[0] == 503
    assert "stale" in state.containers()[1]["reason"]
    seed(state)
    clock.wall -= timedelta(seconds=6)
    assert state.ready()[0] == 503
    assert "future" in state.containers()[1]["reason"]
    seed(state)
    state.stop()
    seed(state)  # late worker completion must not revive shutdown readiness
    assert state.ready()[0] == state.containers()[0] == 503


def test_reconciler_loop_failure_recovery_updates_actual_progress():
    clock = Clock()
    state = clock.state()
    service = SimpleNamespace(reconcile_active_jobs=Mock(side_effect=RuntimeError("private error")))
    pump = OpsReconciler(service, interval=0)
    stop = SimpleNamespace(is_set=Mock(side_effect=[False, True]), wait=Mock())
    pump.run(stop, state)
    loop = state.ready()[1]["checks"][-1]
    assert loop == {"name": "reconciler.loop", "status": "degraded", "reason": "reconciliation cycle failed"}
    clock.advance(2)
    service.reconcile_active_jobs.side_effect = None
    service.reconcile_active_jobs.return_value = 0
    stop.is_set = Mock(side_effect=[False, True])
    pump.run(stop, state)
    assert state.ready()[1]["checks"][-1]["status"] == "ok"


def test_gets_are_cache_only_and_no_generic_proxy_is_exposed(collector):
    clock, state, client, probe, _ = collector
    probe.collect_docker()
    seed_time = state.sample_time()
    for name in ("database", "reconciler.loop"):
        state.publish(name, HealthSample(seed_time, "ok"))
    server = OpsHealthServer(state, ("127.0.0.1", 0))
    server.start()
    try:
        for _ in range(5):
            assert request(server, "/health/live") == (200, {"status": "ok"})
            assert request(server, "/health/ready")[0] == 200
            assert request(server, "/v1/health/containers")[0] == 200
        assert client.api.containers.call_count == 1
        assert client.api.inspect_container.call_count == 1
        for path in ("/containers/json", "/v1/health/containers?target=app", "/health/ready?refresh=true", "/docs"):
            assert request(server, path)[0] == 404
        for method in ("POST", "PUT", "PATCH", "DELETE"):
            assert request(server, "/v1/health/containers", method=method)[0] == 405
        clock.advance(31)
        assert request(server, "/health/ready")[0] == 503
        assert request(server, "/v1/health/containers")[0] == 503
        assert request(server, "/health/live")[0] == 200
    finally:
        server.close()
    assert not server.thread.is_alive()


@pytest.mark.integration
@pytest.mark.offline_integration
def test_http_survives_blocked_reconciler_and_collectors_and_shutdown_is_bounded():
    clock = Clock()
    state = clock.state()
    seed(state)
    entered = Event()
    release = Event()

    def block(*_args):
        _stall(entered, release)

    collectors = SimpleNamespace(run=OpsHealthCollector.run, collect_storage=block, collect_docker=block)
    runtime = OpsHealthRuntime(state=state, collector=collectors,
                              reconciler_factory=lambda _: SimpleNamespace(run=block),
                              address=("127.0.0.1", 0))
    runtime.start()
    try:
        assert entered.wait(HANG_GUARD_S)
        clock.advance(31)
        before = time.monotonic()
        assert request(runtime.http, "/health/live") == (200, {"status": "ok"})
        assert request(runtime.http, "/health/ready")[0] == 503
        assert request(runtime.http, "/v1/health/containers")[0] == 503
        assert request(runtime.http, "/v1/health/services")[0] == 503
        # timing-ok: real HTTP requests answer from cached state; waiting on the stalled collectors takes 10s
        assert time.monotonic() - before < 1
        before = time.monotonic()
        runtime.close()
        # timing-ok: real thread joins: close's 2s worker deadline vs collectors stalled for 10s
        assert time.monotonic() - before < 3.5
        assert not runtime.http.thread.is_alive()
        assert len(runtime.workers) == 3
    finally:
        release.set()
        for worker in runtime.workers:
            worker.join(HANG_GUARD_S)
            assert not worker.is_alive()


def test_compose_has_internal_only_listener_and_readiness_healthcheck():
    import yaml

    compose = yaml.safe_load((Path(__file__).parents[1] / "docker-compose.ops.yml").read_text())
    service = compose["services"]["ops-reconciler"]
    # 8650 health, 8651 the app's control channel (TASK-1169 S3); neither published.
    assert service["expose"] == ["8650", "8651"]
    assert "ports" not in service
    assert "/health/ready" in service["healthcheck"]["test"][-1]
    assert "group_add" not in service


def test_frozen_cross_stack_fixture_is_real_producer_output(collector):
    frozen = json.loads((Path(__file__).parent / "fixtures/ops_health_v1.json").read_text())
    clock, _, client, probe, _ = collector
    clock.wall = datetime.fromisoformat(frozen["inventory"]["checkedAt"])
    state = clock.state()
    probe.state = state
    seed(state)
    rows = []
    for entry in frozen["inventory"]["containers"]:
        row = attrs("/" + entry["name"], entry["state"], entry["health"])
        row["State"]["StartedAt"] = entry["startedAt"] or "0001-01-01T00:00:00Z"
        rows.append(row)
    client.api.containers.return_value = [{"Id": str(i)} for i in range(len(rows))]
    client.api.inspect_container.side_effect = lambda i: rows[int(i)]
    probe.collect_docker()
    clock.advance(5)
    assert clock.wall.isoformat() == frozen["now"]
    assert state.ready() == (200, frozen["ready"])
    assert state.containers() == (200, frozen["inventory"])


@pytest.mark.parametrize("monotonic_age,wall_age,expected", [
    (30, 30, 200), (30.001, 0, 503), (0, 30.001, 503),
    (0, -5, 200), (0, -5.001, 503),
])
def test_independent_freshness_bounds(monotonic_age, wall_age, expected):
    clock = Clock()
    state = clock.state()
    seed(state)
    clock.elapsed += monotonic_age
    clock.wall += timedelta(seconds=wall_age)
    assert state.ready()[0] == state.containers()[0] == expected


def test_fresh_inventory_does_not_require_database_or_loop_readiness(collector):
    clock, state, _, probe, _ = collector
    seed(state)
    clock.advance(31)
    probe.collect_docker()
    assert state.containers()[0] == 200
    assert state.ready()[0] == 503
    assert state.ready()[1]["checkedAt"] != state.containers()[1]["checkedAt"]


def test_bootstrap_failure_recovers_without_restarting_http():
    state = Clock().state()
    failed = Event()
    retry = Event()
    recovered = Event()
    attempts = []

    def build(_interval):
        attempts.append(1)
        if len(attempts) == 1:
            failed.set()
            raise RuntimeError("private bootstrap error")
        assert retry.wait(HANG_GUARD_S), "test never allowed the retry"
        return SimpleNamespace(run=run)

    def run(stop, health):
        health.publish("reconciler.loop", HealthSample(health.sample_time(), "ok"))
        recovered.set()
        _until_stopped(stop)

    collectors = SimpleNamespace(run=lambda _collect, stop: _until_stopped(stop),
                                 collect_storage=None, collect_docker=None)
    runtime = OpsHealthRuntime(state=state, collector=collectors, interval=0.01,
                              reconciler_factory=build, address=("127.0.0.1", 0))
    runtime.start()
    try:
        assert failed.wait(HANG_GUARD_S)
        assert request(runtime.http, "/health/live")[0] == 200
        body = request(runtime.http, "/health/ready")[1]
        assert "private" not in json.dumps(body)
        retry.set()
        assert recovered.wait(HANG_GUARD_S)
        assert request(runtime.http, "/health/ready")[1]["checks"][-1]["status"] == "ok"
        assert len(attempts) == 2
    finally:
        retry.set()
        runtime.close()


def test_successful_slow_tick_is_not_fresh_at_completion():
    clock = Clock()
    state = clock.state()
    seed(state)
    started = state.sample_time().iso
    service = SimpleNamespace(reconcile_active_jobs=lambda **_: (clock.advance(31), 0)[1])
    stop = SimpleNamespace(is_set=Mock(side_effect=[False, True]), wait=Mock())
    OpsReconciler(service, interval=0).run(stop, state)
    code, body = state.ready()
    assert code == 503 and body["checkedAt"] == started
    assert "stale" in body["checks"][-1]["reason"]


def test_maximum_inventory_fits_consumers_body_limit(collector):
    _, state, client, probe, _ = collector
    client.api.containers.return_value = [{"Id": str(i)} for i in range(1000)]
    client.api.inspect_container.side_effect = lambda i: attrs("/" + i.zfill(255))
    probe.collect_docker()
    code, body = state.containers()
    assert code == 200 and len(body["containers"]) == 1000
    assert len(json.dumps(body).encode()) < 512 * 1024


@pytest.mark.integration
@pytest.mark.offline_integration
def test_http_shutdown_is_bounded_even_if_request_handler_stalls():
    server = OpsHealthServer(Clock().state(), ("127.0.0.1", 0))
    entered, release = Event(), Event()

    server.server.handle_request = lambda: _stall(entered, release)
    server.start()
    try:
        assert entered.wait(HANG_GUARD_S)
        before = time.monotonic()
        server.close()
        # timing-ok: real server thread; close joins it for at most 2s, the stalled handler holds for 10s
        assert time.monotonic() - before < 2.5
    finally:
        release.set()
        server.thread.join(HANG_GUARD_S)
        assert not server.thread.is_alive()
