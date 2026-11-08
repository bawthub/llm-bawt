"""TASK-1844: GPU Compose service reconcile — CDI profile, preflight, fence."""

import json
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from llm_bawt.media.gpu_handoff_store import GpuHandoffFence, GpuHandoffStore, HandoffConflict, _state
from llm_bawt.ops.compose_config import _gpu, _running_gpu, gpu_preflight, gpu_verify
from llm_bawt.ops.compose_render import ComposeError
from llm_bawt.ops.gpu_reconcile import is_gpu_reconcile, known_outcome, nested_gpu_reconcile, settle

DESIRED = {"gpu": {"runtime": "runc", "ipc": "host", "devices": ["nvidia.com/gpu=all"]},
           "security_opt": ["apparmor:unconfined"]}
INFO = {"Runtimes": {"runc": {}, "io.containerd.runc.v2": {}},
        "DiscoveredDevices": [{"Source": "cdi", "ID": "nvidia.com/gpu=all"}]}


def created(**host):
    base = {"Runtime": "runc", "IpcMode": "host", "SecurityOpt": ["apparmor:unconfined", "label=disable"],
            "DeviceRequests": [{"Driver": "cdi", "Count": 0, "DeviceIDs": ["nvidia.com/gpu=all"]}]}
    return {"HostConfig": {**base, **host}}


# ── Compose GPU keys: native CDI only ───────────────────────────────────────

def test_cdi_service_translates():
    assert _gpu({"runtime": "runc", "ipc": "host", "devices": ["nvidia.com/gpu=all"]}) == DESIRED["gpu"]
    long = {"source": "nvidia.com/gpu=all", "target": "nvidia.com/gpu=all", "permissions": "rwm"}
    assert _gpu({"devices": [long]})["devices"] == ["nvidia.com/gpu=all"]
    assert _gpu({}) == {"runtime": "runc", "ipc": "", "devices": []}


@pytest.mark.parametrize("service,key", [
    ({"runtime": "nvidia"}, "runtime"),
    ({"ipc": "container:other"}, "ipc"),
    ({"devices": ["/dev/nvidia0:/dev/nvidia0"]}, "devices"),
    ({"devices": ["nvidia.com/gpu=all", "nvidia.com/gpu=all"]}, "devices"),
    ({"devices": [{"source": "nvidia.com/gpu=all", "target": "/dev/x"}]}, "devices"),
])
def test_non_cdi_gpu_config_refused(service, key):
    with pytest.raises(ComposeError) as err:
        _gpu(service)
    assert err.value.code == "unsupported_compose_key" and err.value.names == [key]


def test_running_non_cdi_device_never_matches():
    assert _running_gpu(created(Devices=[{"PathOnHost": "/dev/nvidia0"}])["HostConfig"])["devices"][-1] \
        == "<non-cdi device>"
    assert _running_gpu({"IpcMode": "private"})["ipc"] == ""


# ── Preflight refuses before any mutation ───────────────────────────────────

def test_preflight_admits_live_shape():
    gpu_preflight(DESIRED, INFO)
    gpu_preflight({}, None)  # not a GPU target: no daemon read needed


@pytest.mark.parametrize("info,code", [
    ({**INFO, "DiscoveredDevices": []}, "gpu_cdi_device_missing"),
    ({**INFO, "DiscoveredDevices": [{"Source": "other", "ID": "nvidia.com/gpu=all"}]}, "gpu_cdi_device_missing"),
    ({**INFO, "Runtimes": {"nvidia": {}}}, "gpu_runtime_missing"),
    (None, "gpu_daemon_unreadable"),
])
def test_preflight_refuses_missing_device_or_runtime(info, code):
    with pytest.raises(ComposeError) as err:
        gpu_preflight(DESIRED, info)
    assert err.value.code == code


# ── Created-container verification ──────────────────────────────────────────

def test_verify_accepts_daemon_implicit_label_for_host_ipc():
    assert gpu_verify(DESIRED, created()) == []


@pytest.mark.parametrize("host,name", [
    ({"IpcMode": "private"}, "gpu:~ipc"),
    ({"DeviceRequests": []}, "gpu:~devices"),
    ({"Runtime": "nvidia"}, "gpu:~runtime"),
    ({"SecurityOpt": ["label=disable"]}, "security_opt:~"),
])
def test_verify_reports_dropped_gpu_fields(host, name):
    assert name in gpu_verify(DESIRED, created(**host))


def test_implicit_label_ignored_only_for_undeclared_host_ipc():
    declared = {**DESIRED, "security_opt": ["apparmor:unconfined", "label=disable"]}
    assert gpu_verify(declared, created()) == []
    assert "security_opt:~" in gpu_verify(declared, created(SecurityOpt=["apparmor:unconfined"]))
    private = {**DESIRED, "gpu": {**DESIRED["gpu"], "ipc": ""}}
    assert "security_opt:~" in gpu_verify(private, created(IpcMode=""))


# ── Handoff reservation fence ───────────────────────────────────────────────

@pytest.fixture
def store():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    result = GpuHandoffStore(engine)
    set_state(engine, owner="voice", phase="idle")
    yield result
    engine.dispose()


def set_state(engine, **values):
    with engine.begin() as conn:
        conn.execute(_state.update().where(_state.c.id == 1).values(**values))


def reserve(store, key="k1", slug="llm-bawt.reconcile-local-model-bridge-config"):
    with Session(store.engine) as session, session.begin():
        GpuHandoffFence.gpu_reconcile(key, slug)(session)


def test_recreate_first_blocks_handoff(store):
    reserve(store)
    state = store.status()
    assert (state.phase, state.reconcile_key) == ("gpu_reconciling", "k1")
    with pytest.raises(HandoffConflict):
        store.offer(user="nick", target="video", actions=("park_voice",), expected_generation=state.generation)
    assert store.admits("k1", "any") and not store.admits("k2", "any")


def test_handoff_first_blocks_recreate(store):
    store.offer(user="nick", target="video", actions=("park_voice",), expected_generation=0)
    with pytest.raises(HandoffConflict) as err:
        reserve(store)
    assert err.value.code == "gpu_not_idle" and store.status().reconcile_key is None


def test_replay_same_key_other_key_refused(store):
    reserve(store)
    generation = store.status().generation
    reserve(store)  # same idempotency key replays
    assert store.status().generation == generation
    with pytest.raises(HandoffConflict) as err:
        reserve(store, key="k2")
    assert err.value.code == "gpu_reconcile_in_progress"


def test_video_owner_refused(store):
    set_state(store.engine, owner="video")
    with pytest.raises(HandoffConflict) as err:
        reserve(store)
    assert err.value.code == "gpu_not_idle"


def test_release_is_key_checked_and_idempotent(store):
    reserve(store)
    assert not store.release_gpu_reconcile("other")
    assert store.status().phase == "gpu_reconciling"
    assert store.release_gpu_reconcile("k1")
    assert not store.release_gpu_reconcile("k1")
    state = store.status()
    assert (state.phase, state.reconcile_key, state.owner) == ("idle", None, "voice")
    store.offer(user="nick", target="video", actions=("park_voice",), expected_generation=state.generation)


# ── Known outcomes release; uncertain ones hold ─────────────────────────────

def job(state, *, dispatched=True, exit_code=None, tail=None, key="k1"):
    return SimpleNamespace(id="j1", state=state, exit_code=exit_code, output_tail=tail,
                           dispatched_at=object() if dispatched else None, idempotency_key=key)


NOTHING = json.dumps({"phase": "verify-created", "deployed": None, "restored": False})
RESTORED = json.dumps({"phase": "health", "deployed": {"container_id": "x"}, "restored": True})
UNCERTAIN = json.dumps({"phase": "swap", "deployed": {"container_id": "x"}, "restored": False})


@pytest.mark.parametrize("terminal,expected", [
    (job("succeeded", exit_code=0), "succeeded"),
    (job("succeeded", exit_code=1), None),
    (job("failed", dispatched=False), "never_dispatched"),
    (job("cancelled", dispatched=False), "never_dispatched"),
    (job("failed", tail=NOTHING), "nothing_changed"),
    (job("failed", tail=RESTORED), "restored"),
    (job("failed", tail=UNCERTAIN), None),
    (job("failed", tail="worker crashed"), None),
    (job("failed"), None),
    (job("cancelled"), None),
    (job("lost"), None),
    (job("timed_out"), None),
])
def test_known_outcome(terminal, expected):
    assert known_outcome(terminal) == expected


def test_settle_releases_known_and_holds_uncertain(store):
    reserve(store)
    assert settle(store.engine, job("lost")) is None
    state = store.status()
    assert state.phase == "gpu_reconciling" and "uncertain outcome" in state.last_error
    assert settle(store.engine, job("lost")) is None  # replay: no churn
    assert settle(store.engine, job("failed", tail=RESTORED)) == "restored"
    assert store.status().phase == "idle"
    assert settle(store.engine, job("succeeded", exit_code=0)) == "succeeded"  # already released: no-op
    assert store.status().phase == "idle"


def test_only_standalone_app_reconcile_is_gpu():
    spec = {"action": "reconcile_app_config", "gpu": True}
    assert is_gpu_reconcile(spec)
    assert not is_gpu_reconcile({**spec, "gpu": "yes"})
    assert not is_gpu_reconcile({**spec, "action": "deploy_image"})
    assert nested_gpu_reconcile({"plan": {"targets": [{"spec": spec}]}})
    assert nested_gpu_reconcile({"sequence": {"steps": [{"snapshot": {"spec": spec}}]}})
    assert not nested_gpu_reconcile({"spec": spec})
