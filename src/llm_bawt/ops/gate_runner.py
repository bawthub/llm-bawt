"""Unused, fixed-workload socketless gate launcher; no dispatch or SHIP authority.

Only trusted code may supply a staged source and DB-frozen environment evidence
plus its independently trusted digest. This module checks that binding, not the
truth/authenticity of a packager's claims. No evidence producer/configuration or
production caller exists yet. The finite runner-assets manifest is NOT evidence
of an offline dependency closure and cannot satisfy this contract.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import shutil
import signal
import stat
import threading
import time
from urllib.parse import quote, urlencode

from .gate_environment import GateEnvironmentError, PG_IMAGE as _PG_IMAGE, validate_environment
from .gate_plan import GatePlan
from .gate_source import GateSource

_PORT = "55439"
_SECONDS = 1800
_CLEANUP_SECONDS = 60
_MAX_OUTPUT = 1024 * 1024
# Image defaults reach env(1)'s loader before env -i. Do not accept loader,
# shell-startup, Python, Docker, credential or arbitrary injected defaults.
_IMAGE_ENV = {"PATH", "HOME", "LANG", "LC_ALL", "PG_MAJOR", "PG_VERSION", "PG_SHA256",
    "PGDATA", "GOSU_VERSION", "PYTHON_VERSION", "PYTHON_SHA256"}
_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
_RUN_ENV = [f"PATH={_PATH}", "HOME=/work/home", "TMPDIR=/work/tmp", "XDG_CONFIG_HOME=/work/config",
    "XDG_CACHE_HOME=/work/cache", "XDG_DATA_HOME=/work/data", "LC_ALL=C.UTF-8",
    "UV_CACHE_DIR=/work/uv-cache", "UV_OFFLINE=1", "UV_PYTHON_DOWNLOADS=never",
    "TIKTOKEN_CACHE_DIR=/opt/gate/tiktoken", "OPS_GATE_SANDBOX=1",
    "OPS_GATE_PG_BINDING=/gate/pg.json", "LLM_BAWT_TEST_WORKERS=2",
    "LLM_BAWT_UNIT_TIME_BUDGET_SECONDS=3", "LLM_BAWT_REAL_PROCESS_TIME_BUDGET_SECONDS=7"]
# Trusted constant bootstrap, executed ONLY inside the unprivileged container.
# The image's attested cache is copied to private scratch because uv writes it.
_BOOTSTRAP = """set -eu
mkdir -p /work/home /work/tmp /work/config /work/cache /work/data /work/source /work/uv-cache
cp -R /gate/source/. /work/source/
cp -R /opt/gate/uv-cache/. /work/uv-cache/
cd /work/source
exec uv run --offline --isolated --frozen --extra dev --extra service --extra mcp --extra search --extra memory python -m scripts.test_suite offline
"""


class GateLaunchError(RuntimeError):
    """Fixed code only; uncertain cleanup never constitutes retirement/retry proof."""

    def __init__(self, code: str, *, cleanup_confirmed: bool = False):
        self.code = code
        self.cleanup_confirmed = cleanup_confirmed
        super().__init__(code)


@dataclass(frozen=True)
class GateRunResult:
    exit_code: int
    output: bytes  # Untrusted, bounded Docker stream. Never echoed by this module.
    source_sha: str
    environment_sha256: str
    cleanup_confirmed: bool


def _environment(raw: bytes, expected: str, plan: GatePlan, source: GateSource) -> dict:
    # Keep the launcher's refusal codes and independently staged source check.
    try:
        data = validate_environment(raw, expected, plan, tree_sha=source.tree_sha)
    except GateEnvironmentError as exc:
        raise GateLaunchError(str(exc)) from None
    if source.source_sha != plan.source_sha or not re.fullmatch(r"[0-9a-f]{40}", source.tree_sha):
        raise GateLaunchError("environment_evidence_incompatible")
    return data


@contextmanager
def _bounded(seconds):
    # Existing worker processes use this same OS deadline mechanism. Never steal
    # another owner's timer or silently replace a hard bound with socket-idle timeouts.
    if threading.current_thread() is not threading.main_thread() or signal.getitimer(signal.ITIMER_REAL)[0]:
        raise GateLaunchError("deadline_unavailable")
    previous = signal.getsignal(signal.SIGALRM)

    def expired(_signum, _frame):
        raise GateLaunchError("deadline_exceeded")

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _missing(exc: RuntimeError) -> bool:
    # DockerAPI's structured HTTP prefix; never search arbitrary daemon stderr.
    return str(exc).startswith("Docker HTTP 404:")


def _inspect(api, name):
    try:
        return api.request("GET", f"/containers/{quote(name, safe='')}/json")
    except RuntimeError as exc:
        if _missing(exc):
            return None
        raise


def _image(api, reference, identity, *, postgres=False):
    data = api.request("GET", f"/images/{quote(reference, safe='')}/json")
    if (not isinstance(data, dict) or data.get("Id") != identity
            or reference not in (data.get("RepoDigests") or [])
            or data.get("Os") != "linux" or data.get("Architecture") != "amd64"):
        raise GateLaunchError("local_image_mismatch")
    config = data.get("Config") or {}
    env, seen = config.get("Env") or [], set()
    if not isinstance(env, list):
        raise GateLaunchError("local_image_unsafe_defaults")
    for item in env:
        if not isinstance(item, str) or "=" not in item:
            raise GateLaunchError("local_image_unsafe_defaults")
        key, value = item.split("=", 1)
        if key not in _IMAGE_ENV or key in seen or len(value) > 4096 or "\x00" in value or "\n" in value:
            raise GateLaunchError("local_image_unsafe_defaults")
        seen.add(key)
    volumes = config.get("Volumes") or {}
    if not isinstance(volumes, dict) or set(volumes) - ({"/var/lib/postgresql/data"} if postgres else set()):
        raise GateLaunchError("local_image_unsafe_defaults")


def _private(path: Path):
    if (not path.is_absolute() or path.resolve(strict=True) != path or not path.is_dir()
            or path.stat().st_uid != os.geteuid() or stat.S_IMODE(path.stat().st_mode) != 0o700):
        raise GateLaunchError("private_directory_required")


def _handoff(source: GateSource):
    root = source.directory
    _private(root.parent)
    if root.resolve(strict=True) != root or not root.is_dir() or root.stat().st_uid != os.geteuid():
        raise GateLaunchError("source_handoff_invalid")
    # Only the already verified, owned/frozen stage is eligible. Parent stays
    # 0700 on the host; bind mounts expose just the source to UID 65532, read-only.
    paths = [root, *root.rglob("*")]
    for path in paths:
        info = path.lstat()
        if info.st_uid != os.geteuid() or not (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)):
            raise GateLaunchError("source_handoff_invalid")
    for path in paths:
        if path.is_dir():
            path.chmod(0o755)
        elif path.is_relative_to(root / ".git"):
            path.chmod(0o644)


def _body(image, labels, *, postgres, source, socket, binding):
    host = {"NetworkMode": "none", "ReadonlyRootfs": True, "CapDrop": ["ALL"],
        "SecurityOpt": ["no-new-privileges:true"], "LogConfig": {"Type": "none", "Config": {}},
        "RestartPolicy": {"Name": "no"}, "AutoRemove": False, "Privileged": False,
        "Memory": (512 if postgres else 4096) * 1024 * 1024,
        "PidsLimit": 256, "NanoCpus": 2_000_000_000, "ShmSize": 64 * 1024 * 1024,
        "Mounts": [{"Type": "bind", "Source": str(socket), "Target": "/var/run/postgresql",
            "ReadOnly": not postgres, "BindOptions": {"Propagation": "rprivate"}}]}
    if postgres:
        host["Tmpfs"] = {"/tmp": "rw,noexec,nosuid,nodev,size=67108864,uid=999,gid=999,mode=1777",
            "/var/lib/postgresql/data": "rw,noexec,nosuid,nodev,size=1073741824,uid=999,gid=999,mode=0700"}
        env = [f"PATH={_PATH}", "LANG=C.UTF-8", "PGDATA=/var/lib/postgresql/data",
            "POSTGRES_HOST_AUTH_METHOD=trust", "POSTGRES_USER=test_user", "POSTGRES_DB=llm_bawt_test"]
        command = ["/usr/local/bin/docker-entrypoint.sh", "postgres", "-c", "listen_addresses=",
            "-c", f"port={_PORT}", "-c", "unix_socket_directories=/var/run/postgresql"]
    else:
        host["Mounts"] += [
            {"Type": "bind", "Source": str(source), "Target": "/gate/source", "ReadOnly": True,
                "BindOptions": {"Propagation": "rprivate"}},
            {"Type": "bind", "Source": str(binding), "Target": "/gate/pg.json", "ReadOnly": True,
                "BindOptions": {"Propagation": "rprivate"}}]
        # Scratch allows exec for fixture programs; it is bounded and not a host bind.
        host["Tmpfs"] = {"/work": "rw,nosuid,nodev,size=4294967296,uid=65532,gid=65532,mode=0700",
            "/tmp": "rw,nosuid,nodev,size=268435456,uid=65532,gid=65532,mode=1777"}
        env, command = _RUN_ENV, ["/bin/sh", "-c", _BOOTSTRAP]
    return {"Image": image, "User": "999:999" if postgres else "65532:65532",
        "Entrypoint": ["/usr/bin/env", "-i", *env], "Cmd": command, "Env": [f"PATH={_PATH}"],
        "WorkingDir": "/" if postgres else "/work", "NetworkDisabled": True,
        "AttachStdout": not postgres, "AttachStderr": not postgres, "AttachStdin": False,
        "OpenStdin": False, "Tty": False, "Healthcheck": {"Test": ["NONE"]},
        "Labels": labels, "HostConfig": host}


def _owned(state, labels):
    return (isinstance(state, dict) and isinstance(state.get("Id"), str)
        and re.fullmatch(r"[0-9a-f]{64}", state["Id"]) is not None
        and all((state.get("Config", {}).get("Labels") or {}).get(k) == v for k, v in labels.items()))


def _cleanup(api, attempted):
    confirmed = True
    for name, labels in reversed(attempted):
        try:
            state = _inspect(api, name)
            if state is None:
                continue
            if not _owned(state, labels):
                confirmed = False
                continue
            identity = state["Id"]
            api.request("DELETE", f"/containers/{identity}?force=1&v=1")
            # Check both the immutable ID and deterministic name. Never interpret
            # a successful DELETE response alone as positive absence.
            if _inspect(api, identity) is not None or _inspect(api, name) is not None:
                confirmed = False
        except GateLaunchError:
            raise  # In particular, never swallow the one-shot hard deadline.
        except Exception:  # noqa: BLE001 - all uncertain Docker states refuse
            confirmed = False
    return confirmed


def _create(api, name, body, attempted):
    attempted.append((name, body["Labels"]))  # A lost create response may still have created it.
    response = api.request("POST", "/containers/create?" + urlencode({"name": name}), body=body)
    state = _inspect(api, name)
    if (not _owned(state, body["Labels"]) or not isinstance(response, dict)
            or response.get("Id") != state["Id"] or state.get("State", {}).get("Running") is not False):
        raise GateLaunchError("container_identity_uncertain")
    return state["Id"]


def _ready(api, identity):
    deadline = time.monotonic() + 40
    while time.monotonic() < deadline:
        response = api.request("POST", f"/containers/{identity}/exec", body={"User": "999:999",
            "AttachStdout": True, "AttachStderr": True, "AttachStdin": False, "Tty": False,
            "Cmd": ["/usr/bin/env", "-i", f"PATH={_PATH}", "pg_isready", "-q", "-h",
                "/var/run/postgresql", "-p", _PORT, "-U", "test_user", "-d", "llm_bawt_test"]})
        exec_id = response.get("Id") if isinstance(response, dict) else None
        if not isinstance(exec_id, str) or not re.fullmatch(r"[0-9a-f]{64}", exec_id):
            raise GateLaunchError("postgres_readiness_uncertain")
        api.request("POST", f"/exec/{exec_id}/start", body={"Detach": False, "Tty": False}, raw=True)
        state = api.request("GET", f"/exec/{exec_id}/json")
        if not isinstance(state, dict) or state.get("Running") is not False or type(state.get("ExitCode")) is not int:
            raise GateLaunchError("postgres_readiness_uncertain")
        if state["ExitCode"] == 0:
            return
        time.sleep(0.2)
    raise GateLaunchError("postgres_not_ready")


def gate_container_names(job_id: str) -> dict[str, str]:
    """Fixed identities shared with the dormant custody observer, not authority."""
    if not isinstance(job_id, str) or not re.fullmatch(r"[0-9a-f]{32}", job_id):
        raise GateLaunchError("job_identity_invalid")
    return {role: f"llm-bawt-ops-gate-{job_id}-{role}" for role in ("pg", "runner")}


def gate_container_labels(plan: GatePlan, job_id: str, owner_nonce: str, environment_sha256: str) -> dict:
    """An internal, pre-persisted nonce may replace the old ephemeral owner."""
    gate_container_names(job_id)
    if not isinstance(owner_nonce, str) or not re.fullmatch(r"[0-9a-f]{32}", owner_nonce):
        raise GateLaunchError("custody_identity_invalid")
    return {"llm-bawt.ops.job": job_id, "llm-bawt.ops.gate.owner": owner_nonce,
        "llm-bawt.ops.gate.sha": plan.source_sha,
        "llm-bawt.ops.gate.environment": environment_sha256,
        "com.docker.compose.project": "llm-bawt-ops-gate"}


def run_gate(api, *, plan: GatePlan, source: GateSource, job_id: str, scratch_parent: Path,
             environment_json: bytes, environment_sha256: str,
             owner_nonce: str | None = None) -> GateRunResult:
    """Run once, ONLY after a separate trusted claim, using existing DockerAPI.

    No caller argv/env/paths, pulls, host execution, generic DockerAPI.execute,
    ledger writes, callbacks, retries, retirement records or host publication.
    owner_nonce is an internal identity label, never a permit or a replay token.
    Future custody integration must keep its one-shot reservation alive and fence
    execution itself; accepting a copied label here does not establish that proof.
    All arguments are internal trusted inputs, never a route/tool schema. The
    trusted packager's evidence digest MUST come independently from DB-frozen
    configuration, not from the submitted evidence. No such producer or wiring
    exists yet. Unknown cleanup retains the private scratch directory, refuses
    success and requires operator/reconciler investigation; it never permits retry.
    """
    plan = GatePlan.from_dict(plan.to_dict())
    names = gate_container_names(job_id)
    common = gate_container_labels(plan, job_id,
        os.urandom(16).hex() if owner_nonce is None else owner_nonce, environment_sha256)
    evidence = _environment(environment_json, environment_sha256, plan, source)
    parent = Path(scratch_parent)
    _private(parent)
    if parent.is_relative_to(source.directory):
        raise GateLaunchError("scratch_overlaps_source")
    attempted, scratch, result, failure, cleaned = [], None, None, None, False
    try:
        with _bounded(_SECONDS):
            # Local immutable images only: no /images/create, auth, build or pull.
            _image(api, evidence["runner_image"], evidence["runner_image_id"])
            _image(api, _PG_IMAGE, evidence["postgres_image_id"], postgres=True)
            if any(_inspect(api, name) is not None for name in names.values()):
                raise GateLaunchError("container_name_occupied")
            _handoff(source)
            candidate = parent / f"ops-gate-{job_id}"
            # Assign custody only AFTER exclusive creation succeeds. Existing
            # scratch belongs to an earlier/competing invocation and is untouched.
            candidate.mkdir(mode=0o700)
            scratch = candidate
            socket = scratch / "pg"
            socket.mkdir(mode=0o777)
            socket.chmod(0o777)  # Parent remains private; sidecar UID differs from worker/runner.
            binding = scratch / "pg.json"
            binding.write_text(json.dumps({"schema": "ops.gate.pg/v1", "job_id": job_id,
                "repo": plan.repository, "sha": plan.source_sha, "mode": "unix", "jobs": {"private-pg": {
                    "image": _PG_IMAGE, "socket_dir": "/var/run/postgresql", "port": _PORT,
                    "user": "test_user", "database": "llm_bawt_test"}}}, sort_keys=True))
            binding.chmod(0o644)
            ids = {}
            for role in ("pg", "runner"):
                labels = {**common, "com.docker.compose.service": f"gate-{role}"}
                body = _body(evidence["postgres_image_id" if role == "pg" else "runner_image_id"], labels,
                    postgres=role == "pg", source=source.directory, socket=socket, binding=binding)
                ids[role] = _create(api, names[role], body, attempted)
                if role == "pg":
                    api.request("POST", f"/containers/{ids[role]}/start")
                    _ready(api, ids[role])
            stream = api.attach_stream(ids["runner"])
            try:
                api.request("POST", f"/containers/{ids['runner']}/start")
                output = stream.read(_MAX_OUTPUT + 1)
            finally:
                stream.close()
            if not isinstance(output, bytes) or len(output) > _MAX_OUTPUT:
                raise GateLaunchError("output_bound_exceeded")
            state = _inspect(api, ids["runner"])
            status = state.get("State", {}) if _owned(state, attempted[-1][1]) else {}
            if (status.get("Running") is not False or status.get("Status") != "exited"
                    or type(status.get("ExitCode")) is not int):
                raise GateLaunchError("runner_exit_uncertain")
            if status.get("OOMKilled") is not False or status.get("Error"):
                raise GateLaunchError("runner_exit_uncertain")
            result = GateRunResult(status["ExitCode"], output, plan.source_sha, environment_sha256, True)
    except BaseException as exc:
        failure = exc
    finally:
        if not attempted and scratch is None:
            cleaned = True
        else:
            try:
                with _bounded(_CLEANUP_SECONDS):
                    cleaned = _cleanup(api, attempted)
                    if scratch is not None and cleaned:
                        shutil.rmtree(scratch)
            except BaseException:  # Never hide interrupted/uncertain cleanup behind a green result.
                cleaned = False
    if not cleaned:
        raise GateLaunchError("cleanup_uncertain") from None
    if failure is not None:
        if isinstance(failure, (KeyboardInterrupt, SystemExit)):
            raise failure
        code = failure.code if isinstance(failure, GateLaunchError) else "launcher_unavailable"
        raise GateLaunchError(code, cleanup_confirmed=True) from None
    return result
