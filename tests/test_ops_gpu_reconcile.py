"""TASK-1844: GPU Compose service reconcile — CDI profile, preflight, fence."""

import json
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from llm_bawt.media.gpu_handoff_store import GpuHandoffFence, GpuHandoffStore, HandoffConflict, _state
from llm_bawt.ops.app_config_reconcile import app_profile
from llm_bawt.ops.compose_config import _gpu, _running_gpu, gpu_preflight, gpu_verify, translate
from llm_bawt.ops.compose_render import ComposeError
from llm_bawt.ops.gpu_reconcile import is_gpu_reconcile, known_outcome, nested_gpu_reconcile, settle
from llm_bawt.ops.seeds import SEEDS
from llm_bawt.ops.source_state import SourceStateError, source_plan

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



def test_asleep_tts_admitted_and_still_asleep_after_release(store):
    """TASK-2297: an idle-slept TTS is admitted; release keeps it asleep."""
    set_state(store.engine, owner="asleep")
    reserve(store)
    assert (store.status().phase, store.status().owner) == ("gpu_reconciling", "asleep")
    assert store.release_gpu_reconcile("k1")
    state = store.status()
    assert (state.phase, state.owner, state.reconcile_key) == ("idle", "asleep", None)


def test_tts_wake_refused_while_reconciling_asleep(store):
    set_state(store.engine, owner="asleep")
    reserve(store)
    with pytest.raises(HandoffConflict) as err:
        with Session(store.engine) as session, session.begin():
            GpuHandoffFence.tts_wake("w1")(session)
    # Not tts_not_asleep: BawtHub reads that as "already awake" (Codex, TASK-2297).
    assert err.value.code == "gpu_reconcile_in_progress"
    assert (store.status().phase, store.status().owner) == ("gpu_reconciling", "asleep")


def test_tts_sleep_refused_while_reconciling_voice(store):
    reserve(store)
    with pytest.raises(HandoffConflict) as err:
        with Session(store.engine) as session, session.begin():
            GpuHandoffFence.tts_sleep("s1")(session)
    assert err.value.code == "gpu_reconcile_in_progress"
    assert (store.status().phase, store.status().owner) == ("gpu_reconciling", "voice")


@pytest.mark.parametrize("phase", ["tts_sleeping", "tts_waking"])
def test_asleep_tts_mid_power_change_refused(store, phase):
    set_state(store.engine, owner="asleep", phase=phase)
    with pytest.raises(HandoffConflict) as err:
        reserve(store)
    assert err.value.code == "gpu_not_idle"


def test_asleep_tts_with_open_video_claims_refused(store):
    from llm_bawt.media.gpu_handoff_store import _video_jobs
    set_state(store.engine, owner="asleep")
    with store.engine.begin() as conn:
        conn.execute(_video_jobs.insert().values(id="v1", generation=0))
    with pytest.raises(HandoffConflict) as err:
        reserve(store)
    assert err.value.code == "active_video_claims" and store.status().reconcile_key is None

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


# ── The seeded target passes the production source guard ────────────────────

SEED_SPEC = json.loads(next(s for s in SEEDS if s["slug"] == "llm-bawt.reconcile-local-model-bridge-config")
                       ["command_script"])
PROJECT = "/srv/compose/llm-bawt"
# The NVIDIA CUDA base image's own Config.Env, as the running target inherits it.
CUDA_IMAGE_ENV = [
    "PATH=/app/.venv/bin:/root/.local/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin",
    "NVARCH=x86_64", "NVIDIA_REQUIRE_CUDA=cuda>=12.9 brand=unknown,driver>=535,driver<536 brand=grid,driver>=535",
    "NV_CUDA_CUDART_VERSION=12.9.79-1", "CUDA_VERSION=12.9.1", "LD_LIBRARY_PATH=/usr/local/cuda/lib64",
    "NVIDIA_VISIBLE_DEVICES=all", "NVIDIA_DRIVER_CAPABILITIES=compute,utility", "NV_CUDA_LIB_VERSION=12.9.1-1",
    "NV_NVTX_VERSION=12.9.79-1", "NV_LIBNPP_VERSION=12.4.1.87-1", "NV_LIBNPP_PACKAGE=libnpp-12-9=12.4.1.87-1",
    "NV_LIBCUSPARSE_VERSION=12.5.10.65-1", "NV_LIBCUBLAS_PACKAGE_NAME=libcublas-12-9",
    "NV_LIBCUBLAS_VERSION=12.9.1.4-1", "NV_LIBCUBLAS_PACKAGE=libcublas-12-9=12.9.1.4-1",
    "NV_LIBNCCL_PACKAGE_NAME=libnccl2", "NV_LIBNCCL_PACKAGE_VERSION=2.27.3-1", "NCCL_VERSION=2.27.3-1",
    "NV_LIBNCCL_PACKAGE=libnccl2=2.27.3-1+cuda12.9", "NVIDIA_PRODUCT_NAME=CUDA", "DEBIAN_FRONTEND=noninteractive",
    "TZ=America/New_York", "PYTHONUNBUFFERED=1", "PYTHONDONTWRITEBYTECODE=1",
    "PYTHONPYCACHEPREFIX=/tmp/llm-bawt-pycache",
]


def local_model_plan(image_env=CUDA_IMAGE_ENV, **extra_env):
    volumes = [{"type": "bind", "source": f"{PROJECT}/src", "target": "/app/src"}] + [
        {"type": "bind", "source": f"/data{t}", "target": t} for t in SEED_SPEC["data_mounts"]]
    service = {"build": {"context": PROJECT, "dockerfile": "Dockerfile"}, "image": "llm-bawt-local-model-bridge",
               "container_name": SEED_SPEC["container_name"], "command": ["python", "-m", "local_model_bridge"],
               "entrypoint": None, "environment": {**{n: "v" for n in SEED_SPEC["env_names"]}, **extra_env},
               "runtime": "runc", "ipc": "host", "security_opt": ["apparmor:unconfined"],
               "devices": [{"source": "nvidia.com/gpu=all", "target": "nvidia.com/gpu=all", "permissions": "rwm"}],
               "networks": {"default": None}, "restart": "unless-stopped", "volumes": volumes}
    rendered = {"name": SEED_SPEC["compose_project"], "services": {SEED_SPEC["compose_service"]: service}}
    desired = translate(rendered, SEED_SPEC, "llm-bawt-local-model-bridge:latest", app_profile(SEED_SPEC))
    return desired, source_plan(SEED_SPEC, {"project_dir": PROJECT, "compose_files": [f"{PROJECT}/docker-compose.yml"]},
                                {}, {"Config": {"Env": image_env}}, desired)


def test_seeded_gpu_target_admits_inherited_cuda_env():
    desired, plan = local_model_plan()
    assert desired["gpu"] == DESIRED["gpu"] and plan["paths"] == ["src"] and plan["bytecode_ok"]
    assert plan["classification"]["opaque_env"] == sorted(SEED_SPEC["env_names"])


@pytest.mark.parametrize("entry,code", [
    ("LD_LIBRARY_PATH=/root/.cache/huggingface/lib:/usr/local/cuda/lib64", "source_env_unsafe"),
    ("LD_LIBRARY_PATH=lib", "source_env_unsafe"),
    ("NVIDIA_REQUIRE_CUDA=$(id)", "source_env_unsafe"),
    ("NV_LIBNCCL_PACKAGE=a b", "source_env_unsafe"),
    ("NV_SOMETHING_NEW=1", "source_env_unclassified"),
    ("LD_PRELOAD=/usr/lib/x.so", "source_env_unclassified"),
])
def test_cuda_env_stays_default_deny(entry, code):
    with pytest.raises(SourceStateError) as err:
        local_model_plan([*CUDA_IMAGE_ENV, entry])
    assert err.value.code == code


def test_hf_home_on_the_data_mount_refuses():
    # The compose file no longer passes the host's HF_HOME through.
    with pytest.raises(SourceStateError) as err:
        local_model_plan(HF_HOME="/root/.cache/huggingface")
    assert err.value.code == "source_env_unsafe"


def test_compose_does_not_pass_hf_home_to_the_gpu_service():
    import yaml
    from pathlib import Path
    compose = yaml.safe_load((Path(__file__).parents[1] / "docker-compose.yml").read_text())
    env = compose["services"]["local-model-bridge"]["environment"]
    assert not any(str(e).split("=")[0] == "HF_HOME" for e in env)


def test_only_standalone_app_reconcile_is_gpu():
    spec = {"action": "reconcile_app_config", "gpu": True}
    assert is_gpu_reconcile(spec)
    assert not is_gpu_reconcile({**spec, "gpu": "yes"})
    assert not is_gpu_reconcile({**spec, "action": "deploy_image"})
    assert nested_gpu_reconcile({"plan": {"targets": [{"spec": spec}]}})
    assert nested_gpu_reconcile({"sequence": {"steps": [{"snapshot": {"spec": spec}}]}})
    assert not nested_gpu_reconcile({"spec": spec})


def test_gpu_reconcile_refuses_quiesce_and_nesting():
    from llm_bawt.ops.service import OpsDispatchError, OpsService
    spec = {"action": "reconcile_app_config", "gpu": True}
    for snapshot, code in (({"spec": spec, "quiesce": True}, "gpu_reconcile_quiesce_unsupported"),
                           ({"spec": {"action": "x"}, "plan": {"targets": [{"spec": spec}]}}, "gpu_reconcile_nested")):
        with pytest.raises(OpsDispatchError) as err:
            OpsService._gpu_reconcile_fence(snapshot, None, "k1", "slug")
        assert err.value.code == code
    assert OpsService._gpu_reconcile_fence({"spec": spec}, None, "k1", "slug").kind == "gpu_reconcile"
