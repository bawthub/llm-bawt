"""The app/ops-service control channel and the admission-only app (TASK-1169 S3).

Real listeners on loopback, a SQLite ops store and a fake executor: the app's
admission-only OpsService prepares through the ops service's control server,
creates the job and never reaches an executor; the full service dispatches it.
"""
from __future__ import annotations

import hashlib
import http.client
import json
from threading import Event, Semaphore, Thread
from types import SimpleNamespace

import httpx
import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import create_engine

from llm_bawt.ops import JOB_ACCEPTED, JOB_DISPATCHING, JOB_QUEUED, OpsDispatchError, OpsService, OpsStore
from llm_bawt.ops import executor as executor_module
from llm_bawt.ops.control_client import BUSY_RETRY_SECONDS, OpsControlClient, build_admission_ops_service
from llm_bawt.ops.control_http import OpsControlServer
from llm_bawt.ops.control_protocol import EXECUTORS_PATH, MAX_BODY_BYTES, PREPARE_PATH, PROTOCOL_FIELD
from llm_bawt.ops.gpu_guard import GpuGuardUnverifiable
from llm_bawt.ops.health_http import OpsHealthServer
from llm_bawt.ops.health_state import OpsHealthState
from llm_bawt.ops.reconciler import JobCreatedWaiter, OpsReconciler
from llm_bawt.ops.validation import canonical_json
from tests.test_ops_service import FakeExecutor, _op_data

HANG_GUARD_S = 30.0
TOKEN = "control-token"
SLUG = "test.restart-thing"
TTS_SPEC = {"action": "restart", "compose_project": "bawthub", "compose_service": "tts"}


def _store():
    store = object.__new__(OpsStore)
    store.engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    store._ensure_tables_exist()
    return store


def _hashed(body: dict) -> dict:
    return {**body, "snapshot_hash": hashlib.sha256(canonical_json(body).encode()).hexdigest()}


class Rig:
    """The ops service (full, behind its control server) and the app (admission-only)."""

    def __init__(self, **server_kwargs):
        self.store = _store()
        self.store.create_operation(_op_data())
        self.executor = FakeExecutor()
        self.full = OpsService(self.store, executor=self.executor)
        self.server = OpsControlServer(lambda: self.full, ("127.0.0.1", 0),
                                       token_source=lambda: TOKEN, **server_kwargs)
        self.server.start()
        self.url = "http://127.0.0.1:%d" % self.server.address[1]
        self.client = OpsControlClient(self.url, token_source=lambda: TOKEN)
        self.app = OpsService(self.store, admission_only=True, preparer=self.client)

    def close(self):
        self.server.close()


@pytest.fixture
def rig():
    rig = Rig()
    yield rig
    rig.close()


def _served(service, **kwargs):
    kwargs.setdefault("token_source", lambda: TOKEN)
    server = OpsControlServer(lambda: service, ("127.0.0.1", 0), **kwargs)
    server.start()
    return server, "http://127.0.0.1:%d" % server.address[1]


# ── Prepare over the control channel ────────────────────────────────────────

def test_prepare_over_the_channel_matches_the_in_process_snapshot(rig):
    remote = rig.client.prepare(SLUG, {"target": "a"})
    assert remote == rig.full.prepare_invocation(SLUG, {"target": "a"})
    assert remote["gpu_target"] is None


@pytest.mark.parametrize(("slug", "args", "code", "status"), [
    ("nope", {}, "operation_not_found", 404),
    (SLUG, {"target": "zzz"}, "args_invalid", 400),
])
def test_refusals_keep_their_code_and_status(rig, slug, args, code, status):
    raw = httpx.post(rig.url + PREPARE_PATH, json={"operation": slug, "args": args},
                     headers={"Authorization": f"Bearer {TOKEN}"})
    assert raw.status_code == status
    assert raw.json()["refusal"]["code"] == code and raw.json()[PROTOCOL_FIELD] == 1
    with pytest.raises(OpsDispatchError) as exc:
        rig.client.prepare(slug, args)
    assert exc.value.code == code


@pytest.mark.parametrize("header", [None, "Bearer wrong", f"Basic {TOKEN}"])
def test_prepare_requires_the_control_bearer(rig, header):
    headers = {"Authorization": header} if header else {}
    response = httpx.post(rig.url + PREPARE_PATH, json={"operation": SLUG, "args": {"target": "a"}},
                          headers=headers)
    assert response.status_code == 401 and response.json()["code"] == "unauthorized"
    wrong = OpsControlClient(rig.url, token_source=lambda: "wrong")
    with pytest.raises(OpsDispatchError) as exc:
        wrong.prepare(SLUG, {"target": "a"})
    assert exc.value.code == "ops_unavailable"


def test_an_unusable_key_refuses_everyone():
    def no_key():
        raise RuntimeError("master key unavailable")

    server, url = _served(SimpleNamespace(), token_source=no_key)
    try:
        response = httpx.get(url + EXECUTORS_PATH, headers={"Authorization": f"Bearer {TOKEN}"})
        assert response.status_code == 503 and response.json()["code"] == "ops_control_unconfigured"
    finally:
        server.close()


def _raw_post(url, headers, body=b""):
    host, port = url.removeprefix("http://").split(":")
    conn = http.client.HTTPConnection(host, int(port), timeout=HANG_GUARD_S)
    try:
        conn.putrequest("POST", PREPARE_PATH)
        for name, value in headers.items():
            conn.putheader(name, value)
        conn.endheaders(body)
        response = conn.getresponse()
        return response.status, json.loads(response.read())
    finally:
        conn.close()


def test_a_body_without_length_or_over_the_limit_is_refused_unread(rig):
    auth = {"Authorization": f"Bearer {TOKEN}"}
    assert _raw_post(rig.url, auth)[0] == 411
    status, body = _raw_post(rig.url, {**auth, "Content-Length": str(MAX_BODY_BYTES + 1)})
    assert (status, body["code"]) == (413, "body_too_large")
    status, body = _raw_post(rig.url, {**auth, "Content-Length": "5"}, b"nope!")
    assert (status, body["code"]) == (400, "bad_request")


def test_unknown_routes_and_methods(rig):
    auth = {"Authorization": f"Bearer {TOKEN}"}
    assert httpx.post(rig.url + "/v1/ops/other", headers=auth).status_code == 404
    assert httpx.delete(rig.url + PREPARE_PATH, headers=auth).status_code == 405


def test_executor_readiness_comes_from_the_ops_service(rig):
    assert rig.app.executor_readiness() == {"docker": None}
    rig.executor._available = False
    assert rig.app.executor_readiness() == {"docker": "executor not available"}


# ── Bounded work ────────────────────────────────────────────────────────────

class BlockingService:
    def __init__(self):
        self.release = Event()
        self.entered = Semaphore(0)
        self.returned = Event()

    def prepare_invocation(self, slug, args):
        self.entered.release()
        try:
            assert self.release.wait(HANG_GUARD_S)
            return _hashed({"operation": {"slug": slug}})
        finally:
            self.returned.set()


def test_busy_prepares_never_delay_health_and_one_more_is_turned_away():
    service = BlockingService()
    server, url = _served(service, concurrency=2, budget_seconds=HANG_GUARD_S)
    health = OpsHealthServer(OpsHealthState(), ("127.0.0.1", 0))
    health.start()
    results = []
    workers = [Thread(target=lambda: results.append(
        OpsControlClient(url, token_source=lambda: TOKEN).prepare(SLUG, {})), daemon=True) for _ in range(2)]
    try:
        for worker in workers:
            worker.start()
        # Both slots are held by blocked prepares.
        assert service.entered.acquire(timeout=HANG_GUARD_S)
        assert service.entered.acquire(timeout=HANG_GUARD_S)
        live = httpx.get("http://127.0.0.1:%d/health/live" % health.address[1], timeout=HANG_GUARD_S)
        assert live.status_code == 200
        busy = httpx.post(url + PREPARE_PATH, json={"operation": SLUG, "args": {}},
                          headers={"Authorization": f"Bearer {TOKEN}"})
        assert busy.status_code == 503 and busy.json()["code"] == "ops_busy"
        assert busy.headers["Retry-After"] == "2"
        slept = []
        retrying = OpsControlClient(url, token_source=lambda: TOKEN, sleep=slept.append)
        with pytest.raises(OpsDispatchError) as exc:
            retrying.prepare(SLUG, {})
        assert exc.value.code == "ops_unavailable" and slept == [BUSY_RETRY_SECONDS]
    finally:
        service.release.set()
        for worker in workers:
            worker.join(HANG_GUARD_S)
            assert not worker.is_alive()
        health.close()
        server.close()
    assert len(results) == 2


def test_over_budget_is_504_and_the_slot_frees_only_when_the_work_returns():
    service = BlockingService()
    server, url = _served(service, concurrency=1, budget_seconds=0.05)
    auth = {"Authorization": f"Bearer {TOKEN}"}
    try:
        timed_out = httpx.post(url + PREPARE_PATH, json={"operation": SLUG, "args": {}}, headers=auth)
        assert timed_out.status_code == 504 and timed_out.json()["code"] == "ops_timeout"
        # The abandoned prepare still holds the only slot.
        still = httpx.post(url + PREPARE_PATH, json={"operation": SLUG, "args": {}}, headers=auth)
        assert still.json()["code"] == "ops_busy"
        service.release.set()
        assert service.returned.wait(HANG_GUARD_S)
        assert server.slots.acquire(timeout=HANG_GUARD_S)
        server.slots.release()
    finally:
        service.release.set()
        server.close()


def test_an_unexpected_failure_is_500_and_unavailable():
    def boom(_slug, _args):
        raise RuntimeError("docker exploded")

    server, url = _served(SimpleNamespace(prepare_invocation=boom))
    try:
        with pytest.raises(OpsDispatchError) as exc:
            OpsControlClient(url, token_source=lambda: TOKEN).prepare(SLUG, {})
        assert exc.value.code == "ops_unavailable" and "500 ops_internal" in str(exc.value)
    finally:
        server.close()


# ── The client fails closed on anything that isn't the protocol ─────────────

def _client(handler, **kwargs):
    return OpsControlClient("http://ops", token_source=lambda: TOKEN,
                            transport=httpx.MockTransport(handler), **kwargs)


def _protocol(status, body):
    return lambda _request: httpx.Response(status, json={PROTOCOL_FIELD: 1, **body})


@pytest.mark.parametrize("handler", [
    lambda _r: httpx.Response(404, json={"detail": "Not Found"}),  # an old reconciler
    lambda _r: httpx.Response(405, text="method not allowed"),
    lambda _r: httpx.Response(200, text="<html>proxy</html>"),
    lambda _r: httpx.Response(200, json={"snapshot": _hashed({"a": 1})}),  # no protocol field
    _protocol(500, {"code": "ops_internal", "message": "x"}),
    _protocol(502, {"refusal": {"code": "args_invalid", "message": "x"}}),  # not a refusal status
    _protocol(409, {"refusal": {"code": "Not A Code", "message": "x"}}),
    _protocol(200, {"snapshot": {**_hashed({"a": 1}), "a": 2}}),  # hash does not verify
    _protocol(200, {}),
])
def test_anything_but_a_snapshot_or_coded_refusal_is_unavailable(handler):
    with pytest.raises(OpsDispatchError) as exc:
        _client(handler).prepare(SLUG, {})
    assert exc.value.code == "ops_unavailable"


def test_a_coded_refusal_is_passed_through():
    with pytest.raises(OpsDispatchError) as exc:
        _client(_protocol(409, {"refusal": {"code": "target_unavailable", "message": "gone"}})).prepare(SLUG, {})
    assert (exc.value.code, str(exc.value)) == ("target_unavailable", "gone")


def test_one_retry_only_where_nothing_ran():
    calls = []

    def flaky(request):
        calls.append(request)
        if len(calls) == 1:
            raise httpx.ConnectError("refused")
        return httpx.Response(200, json={PROTOCOL_FIELD: 1, "snapshot": _hashed({"a": 1})})

    assert _client(flaky).prepare(SLUG, {})["a"] == 1 and len(calls) == 2

    def down(_request):
        raise httpx.ConnectError("refused")

    with pytest.raises(OpsDispatchError) as exc:
        _client(down).prepare(SLUG, {})
    assert exc.value.code == "ops_unavailable"

    reads = []

    def slow(request):
        reads.append(request)
        raise httpx.ReadTimeout("slow")

    with pytest.raises(OpsDispatchError):
        _client(slow).prepare(SLUG, {})
    assert len(reads) == 1  # it may have run: never resent


def test_readiness_must_be_a_map_of_reasons():
    with pytest.raises(OpsDispatchError):
        _client(_protocol(200, {"executors": {"docker": 3}})).executor_readiness()


# ── The admission-only app ──────────────────────────────────────────────────

def test_the_app_admits_queued_and_the_ops_service_dispatches(rig):
    job = rig.app.dispatch_job(operation_slug=SLUG, args={"target": "a"}, idempotency_key="k1")
    assert job["state"] == JOB_QUEUED
    assert rig.app.get_job_status(job["id"])["state"] == JOB_QUEUED  # a read, never a dispatch
    assert rig.executor.dispatch_calls == [] and rig.executor.reconcile_calls == []
    # The ops service's pass claims and submits it (the fake accepts at once).
    assert rig.full.get_job_status(job["id"])["state"] in (JOB_DISPATCHING, JOB_ACCEPTED)
    assert len(rig.executor.dispatch_calls) == 1


def test_an_approved_snapshot_is_admitted_without_an_executor(rig):
    approved = rig.app.prepare_invocation(SLUG, {"target": "a"})
    job = rig.app.dispatch_job(operation_slug=SLUG, args={"target": "a"}, idempotency_key="k2",
                               approved_snapshot=approved)
    assert job["state"] == JOB_QUEUED and rig.executor.dispatch_calls == []


def test_an_unreachable_ops_service_creates_nothing():
    store = _store()
    store.create_operation(_op_data())
    server, url = _served(SimpleNamespace())
    server.close()  # nothing listens there now
    app = OpsService(store, admission_only=True, preparer=OpsControlClient(url, token_source=lambda: TOKEN))
    with pytest.raises(OpsDispatchError) as exc:
        app.dispatch_job(operation_slug=SLUG, args={"target": "a"}, idempotency_key="k3")
    assert exc.value.code == "ops_unavailable"
    assert store.get_job_by_key("k3") is None


def test_the_app_never_builds_a_docker_executor(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("the app built a Docker executor")

    monkeypatch.setattr(executor_module.DockerExecutor, "__init__", forbidden)
    app = build_admission_ops_service(SimpleNamespace(OPS_CONTROL_URL="http://ops:8651"), _store())
    assert app.admission_only and app.executors() == {}
    with pytest.raises(OpsDispatchError) as exc:
        app._resolve_executor("docker")
    assert exc.value.code == "ops_admission_only"
    with pytest.raises(ValueError):
        app.register_executor(FakeExecutor())


def test_admission_only_takes_a_preparer_and_no_executor():
    with pytest.raises(ValueError):
        OpsService(_store(), admission_only=True)
    with pytest.raises(ValueError):
        OpsService(_store(), admission_only=True, preparer=object(), executor=FakeExecutor())


def test_a_gpu_voice_job_without_a_handoff_fence_is_refused_without_a_docker_read(rig):
    rig.store.create_operation(_op_data(slug="test.tts-restart", command_script=json.dumps(TTS_SPEC)))
    with pytest.raises(OpsDispatchError) as exc:
        rig.app.dispatch_job(operation_slug="test.tts-restart", args={"target": "a"}, idempotency_key="k4")
    assert exc.value.code == "gpu_handoff_required"
    assert rig.store.get_job_by_key("k4") is None


# ── The creation guard's target ─────────────────────────────────────────────

def _snapshot(spec, **extra):
    return {"spec": spec, "resolved_args": {}, "execution": {"executor_kind": "docker"}, **extra}


NAMED_START = {"action": "start", "container_name": "anything"}


@pytest.mark.parametrize(("snapshot", "expected"), [
    (_snapshot(TTS_SPEC, gpu_target="tts"), "tts"),
    (_snapshot(TTS_SPEC), "tts"),  # a snapshot from before the field: the spec proves it
    (_snapshot({**TTS_SPEC, "compose_service": "web"}), None),
    (_snapshot(NAMED_START, gpu_target="stt"), "stt"),  # only the ops service's read knows
    (_snapshot(NAMED_START, gpu_target=None), None),
])
def test_creation_target(rig, snapshot, expected):
    assert rig.app._creation_gpu_target(snapshot) == expected


@pytest.mark.parametrize("snapshot", [
    _snapshot(TTS_SPEC, gpu_target=None),  # contradicts the spec
    _snapshot(TTS_SPEC, gpu_target="stt"),
    _snapshot(NAMED_START, gpu_target="web"),  # not a voice service
    _snapshot(NAMED_START),  # needs a Docker read the app can't make
])
def test_creation_target_refuses_what_it_cannot_trust(rig, snapshot):
    with pytest.raises(GpuGuardUnverifiable):
        rig.app._creation_gpu_target(snapshot)


def test_the_ops_service_reads_an_unrecorded_target_itself():
    class Labelled(FakeExecutor):
        def target_compose_identity(self, spec, args):
            return ("bawthub", "tts")

    full = OpsService(_store(), executor=Labelled())
    assert full._creation_gpu_target(_snapshot(NAMED_START)) == "tts"


# ── The reconciler wakes on job creation ────────────────────────────────────

class CountingService:
    def __init__(self):
        self.ticks = 0

    def reconcile_active_jobs(self, *, limit, offset):
        self.ticks += 1
        return 0


class ScriptedWaiter:
    def __init__(self, wakes):
        self.wakes, self.calls, self.closed = wakes, [], False

    def wait(self, stop, timeout):
        self.calls.append(timeout)
        if len(self.calls) >= self.wakes:
            stop.set()
        return True

    def close(self):
        self.closed = True


def test_each_wake_is_one_pass_and_the_waiter_closes():
    service, waiter = CountingService(), ScriptedWaiter(3)
    OpsReconciler(service, interval=7.0, waiter=waiter).run(Event())
    assert service.ticks == 3 and waiter.calls == [7.0, 7.0, 7.0] and waiter.closed


def test_a_listener_failure_falls_back_to_the_interval_and_reconnects():
    attempts = []

    class Broken:
        def raw_connection(self):
            attempts.append(1)
            raise OSError("database unreachable")

    waiter = JobCreatedWaiter(Broken())
    stop = Event()
    stop.set()  # the fallback waits on stop; set, it returns at once
    assert waiter.wait(stop, 5.0) is False
    assert waiter.wait(stop, 5.0) is False
    assert len(attempts) == 2 and waiter._conn is None
