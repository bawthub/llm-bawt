"""Durable Docker worker dispatch, no app-owned side-effect threads or SSH.

The Docker daemon owns a deterministic one-shot worker. The app submits an
immutable request on a named volume; only a persisted receipt can mean success.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from .image_deploy import DEPLOY_ACTION, IMAGE_ACTIONS
from .validation import canonical_json
from .worker import OPS_JOB_LABEL, atomic_json, is_ops_worker, target_selector

SUPPORTED_ACTIONS = frozenset({"restart", "start", "stop", "pull"})


class ExecutorError(RuntimeError):
    pass


@dataclass
class DispatchResult:
    host_unit_name: str
    status_file_path: str = ""
    log_file_path: str = ""
    terminal_state: str | None = None  # Compatibility only; never trusted by service.
    exit_code: int | None = None
    output: str | None = None


@dataclass
class ReconcileResult:
    state: str | None
    exit_code: int | None = None
    output_tail: str | None = None
    error: str | None = None
    started_at: str | None = None
    finished_at: str | None = None


class Executor(ABC):
    @abstractmethod
    def kind(self) -> str: ...
    @abstractmethod
    def available(self) -> bool: ...
    @abstractmethod
    def dispatch(self, **kwargs) -> DispatchResult: ...
    @abstractmethod
    def reconcile(self, **kwargs) -> ReconcileResult: ...

    def execution_settings(self, spec: dict | None = None) -> dict:
        return {}

    def preflight(self, snapshot: dict) -> None:
        """Read-only prerequisite validation; no worker may be created here."""
        if not self.available():
            raise ExecutorError("executor not available")

    def inspect_target_image(self, spec: dict) -> str:
        """Image ID the fixed image-action target runs now (approval-time CAS)."""
        raise ExecutorError(f"{self.kind()} executor cannot inspect image deploy targets")

    def check_target(self, spec: dict, args: dict) -> None:
        """Approval-time, read-only: the spec's target exists where this executor
        acts. Raise ExecutorError so nobody approves a guaranteed failure."""


_IMAGE_SPEC_PATTERNS = {
    "container_name": r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}",
    "compose_project": r"[a-z0-9][a-z0-9_-]{0,62}",
    "compose_service": r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,62}",
    "image_repository": r"ghcr\.io/[a-z0-9][a-z0-9._-]*/[a-z0-9][a-z0-9._/-]*",
    "github_repository": r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*",
    "workflow_path": r"\.github/workflows/[A-Za-z0-9_.-]+\.ya?ml",
    "canonical_branch": r"[A-Za-z0-9][A-Za-z0-9_./-]{0,99}",
}


def validate_image_spec(spec: dict) -> dict:
    """Fixed digest deploy/rollback target (TASK-997). No selector arguments:
    every identity is pinned in the operator-reviewed spec, never in job args."""
    allowed = {"action", "stop_grace_seconds", "health_timeout_seconds", *_IMAGE_SPEC_PATTERNS}
    if set(spec) - allowed or set(_IMAGE_SPEC_PATTERNS) - set(spec):
        raise ValueError(f"image spec requires exactly {sorted(allowed)} (stop_grace_seconds optional)")
    for key, pattern in _IMAGE_SPEC_PATTERNS.items():
        if not isinstance(spec[key], str) or not re.fullmatch(pattern, spec[key]):
            raise ValueError(f"image spec {key} is invalid")
    grace = spec.get("stop_grace_seconds", 10)
    if type(grace) is not int or not 0 <= grace <= 300:
        raise ValueError("stop_grace_seconds must be an integer in 0..300")
    health = spec.get("health_timeout_seconds")
    if type(health) is not int or not 10 <= health <= 900:
        raise ValueError("health_timeout_seconds must be an integer in 10..900")
    return spec


def validate_spec(command_script: str) -> dict:
    try:
        spec = json.loads(command_script)
    except (ValueError, TypeError) as exc:
        raise ValueError("command_script must be a Docker JSON spec") from exc
    if isinstance(spec, dict) and spec.get("action") in IMAGE_ACTIONS:
        return validate_image_spec(spec)
    allowed = {"action", "container_name", "container_name_from_arg", "compose_project",
               "compose_service", "compose_service_from_arg", "stop_grace_seconds"}
    if not isinstance(spec, dict) or set(spec) - allowed:
        raise ValueError("Docker spec must be an object with supported fields only")
    if not isinstance(spec.get("action"), str) or spec["action"] not in SUPPORTED_ACTIONS:
        raise ValueError("unsupported Docker action")
    for k, v in spec.items():
        if k != "stop_grace_seconds" and (not isinstance(v, str) or not v.strip()):
            raise ValueError(f"Docker spec {k} must be a nonempty string")
    names = sum(k in spec for k in ("container_name", "container_name_from_arg"))
    services = sum(k in spec for k in ("compose_service", "compose_service_from_arg"))
    if not ((names == 1 and not services and "compose_project" not in spec) or
            (names == 0 and services == 1 and "compose_project" in spec)):
        raise ValueError("Docker spec requires exactly one name or project+service selector")
    grace = spec.get("stop_grace_seconds", 10)
    if type(grace) is not int or not 0 <= grace <= 3600:
        raise ValueError("stop_grace_seconds must be an integer in 0..3600")
    return spec


def _stored_pull_auth() -> dict | None:
    """ghcr.io pull credential from the encrypted CredentialStore (TASK-997)."""
    from ..service.providers.github_deploy import GhcrPullAdapter
    from ..utils.config import Config
    return GhcrPullAdapter(Config()).pull_auth()


class DockerExecutor(Executor):
    def __init__(self, *, client_factory=None, worker_image: str | None = None,
                 receipt_volume: str | None = None, receipt_root: str | None = None,
                 pull_auth=None):
        self._client_factory = client_factory
        self._client = None
        self.worker_image = worker_image or os.getenv("LLM_BAWT_OPS_WORKER_IMAGE", "")
        self.receipt_volume = receipt_volume or os.getenv("LLM_BAWT_OPS_RECEIPT_VOLUME", "")
        self.receipt_root = receipt_root or os.getenv("LLM_BAWT_OPS_RECEIPT_ROOT", "/var/lib/llm-bawt-ops")
        # TASK-997: release images are pulled HERE (deploy preflight) with the
        # DB-stored ghcr-pull credential; the worker never holds one.
        self._pull_auth = pull_auth or _stored_pull_auth

    def kind(self):
        return "docker"

    @property
    def client(self):
        if self._client is None:
            if self._client_factory:
                self._client = self._client_factory()
            else:
                import docker
                self._client = docker.from_env(timeout=15)
        return self._client

    def execution_settings(self, spec=None):
        return {"worker_image": self.worker_image, "receipt_volume": self.receipt_volume,
                "receipt_root": self.receipt_root}

    def inspect_target_image(self, spec):
        try:
            return self.client.containers.get(spec["container_name"]).attrs["Image"]
        except Exception as exc:
            raise ExecutorError(f"cannot inspect deploy target {spec['container_name']}: {exc}") from exc

    def check_target(self, spec, args):
        # TASK-1002: this executor only reaches its own Docker daemon. A target
        # that lives on another host (e.g. Unraid) must fail HERE, before an
        # approval card, not as a 404 after a human approved it.
        name, filters = target_selector(spec, args)
        try:
            if name is not None:
                self.client.containers.get(name)
                return
            rows = self.client.containers.list(all=True, filters=filters)
        except Exception as exc:
            if getattr(exc, "status_code", None) == 404:
                raise ExecutorError(
                    f"target container {name!r} does not exist on this executor's Docker host") from exc
            raise ExecutorError(f"cannot inspect target container: {exc}") from exc
        matches = [row for row in rows if not is_ops_worker(row.labels)]
        if len(matches) != 1:
            raise ExecutorError(
                f"compose selector {filters['label']} matched {len(matches)} containers on this "
                "executor's Docker host, expected one")

    def available(self):
        try:
            self._check_settings(self.execution_settings())
            self.client.ping()
            return True
        except Exception:
            return False

    def preflight(self, snapshot):
        try:
            self._check_settings(snapshot["execution"])
            self.client.ping()
        except ExecutorError:
            raise
        except Exception as exc:
            raise ExecutorError(f"Docker worker prerequisites unavailable: {exc}") from exc
        if snapshot["spec"].get("action") == DEPLOY_ACTION:
            self.ensure_release_image(snapshot["release"]["image_ref"])

    def ensure_release_image(self, ref):
        """Pull the approved ``repo@digest`` if absent. Idempotent and touches no
        container, so a failure leaves the job QUEUED with a visible reason. The
        worker still re-verifies RepoDigest and release labels before any swap."""
        try:
            self.client.images.get(ref)
            return
        except Exception as exc:
            if getattr(exc, "status_code", None) != 404:
                raise ExecutorError(f"cannot inspect release image: {exc}") from exc
        try:
            auth = self._pull_auth()
        except Exception as exc:  # noqa: BLE001 - DB/key trouble is a prerequisite, not a crash
            raise ExecutorError(f"registry pull credential unavailable: {type(exc).__name__}") from exc
        if not auth:
            raise ExecutorError("connect the ghcr-pull provider credential (Providers: GitHub container registry pull)")
        try:
            self.client.images.pull(ref, auth_config=auth)
        except Exception as exc:
            # Docker errors carry the registry message, never the credential.
            raise ExecutorError(f"release image pull failed: {exc}") from exc

    def _check_settings(self, settings):
        image = settings.get("worker_image", "")
        if not re.fullmatch(r"(?:sha256:|[^\s]+@sha256:)[0-9a-f]{64}", image):
            raise ExecutorError("configure LLM_BAWT_OPS_WORKER_IMAGE as immutable image ID or digest")
        volume = settings.get("receipt_volume", "")
        root = settings.get("receipt_root", "")
        if not volume or not root or not Path(root).is_absolute():
            raise ExecutorError("configure a dedicated Docker receipt volume and absolute app mount path")
        # Never let Docker implicitly create a misspelled volume. Deployment must
        # mount this SAME named volume at receipt_root in the app.
        self.client.volumes.get(volume)
        self.client.images.get(image)  # No implicit pull at dispatch time.
        if not Path(root).is_dir():
            raise ExecutorError("receipt volume is not mounted at configured app path")
        owner = os.getenv("LLM_BAWT_OPS_APP_CONTAINER") or os.getenv("HOSTNAME", "")
        if not owner:
            raise ExecutorError("cannot identify submitting container to verify receipt mount")
        mounts = self.client.containers.get(owner).attrs.get("Mounts", [])
        if not any(m.get("Type") == "volume" and m.get("Name") == volume and
                   m.get("Destination") == root and m.get("RW") for m in mounts):
            raise ExecutorError("configured receipt path is not the configured writable named volume in this container")

    @staticmethod
    def worker_name(job_id):
        if not re.fullmatch(r"[0-9a-f]{32}", job_id):
            raise ExecutorError("invalid job id")
        return f"llm-bawt-ops-{job_id}"

    def _get_worker(self, name):
        try:
            return self.client.containers.get(name)
        except Exception as exc:
            if getattr(exc, "status_code", None) == 404:
                return None
            raise

    def dispatch(self, *, job_id: str, snapshot: dict, **_kwargs):
        self.worker_name(job_id)
        self.preflight(snapshot)
        directory = Path(snapshot["execution"]["receipt_root"]) / job_id
        directory.mkdir(mode=0o700, exist_ok=True)
        with (directory / "submission.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if (directory / "abandoned.json").exists():
                raise ExecutorError("submission abandoned by recovery; never replay")
            return self._dispatch_locked(job_id=job_id, snapshot=snapshot)

    def _dispatch_locked(self, *, job_id: str, snapshot: dict):
        settings = snapshot["execution"]
        try:
            self._check_settings(settings)
            directory = Path(settings["receipt_root"]) / job_id
            directory.mkdir(mode=0o700, exist_ok=True)
            request = {"job_id": job_id, "snapshot": snapshot}
            raw = canonical_json(request).encode()
            path = directory / "request.json"
            # Create immutable request using hard-link publication (no overwrite
            # or partially written destination), then fsync its directory.
            if not path.exists():
                temp = directory / f"request.{os.getpid()}.{os.urandom(8).hex()}.tmp"
                with temp.open("xb") as stream:
                    stream.write(raw)
                    stream.flush()
                    os.fsync(stream.fileno())
                try:
                    os.link(temp, path)
                except FileExistsError:
                    pass
                finally:
                    temp.unlink()
                fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
            if path.read_bytes() != raw:
                raise ExecutorError("immutable request conflict")
            digest = hashlib.sha256(raw).hexdigest()
            name = self.worker_name(job_id)
            worker = self._get_worker(name)
            volumes = {settings["receipt_volume"]: {"bind": "/receipts", "mode": "rw"},
                       "/var/run/docker.sock": {"bind": "/var/run/docker.sock", "mode": "rw"}}
            if worker is None:
                try:
                    worker = self.client.containers.create(
                        settings["worker_image"],
                        command=[f"/receipts/{job_id}/request.json", digest],
                        name=name, detach=True, network_mode="none", read_only=True,
                        cap_drop=["ALL"], security_opt=["no-new-privileges:true"],
                        restart_policy={"Name": "no"}, mem_limit="128m", pids_limit=32,
                        # Docker merges labels inherited from the immutable base
                        # image with labels supplied here. The current worker
                        # image was derived from the Compose app image and thus
                        # carries project=llm-bawt/service=app; override those
                        # identities so the worker can never select itself as a
                        # Compose operation target.
                        labels={
                            "com.docker.compose.project": "llm-bawt-ops-worker",
                            "com.docker.compose.service": "ops-worker",
                            OPS_JOB_LABEL: job_id,
                            "llm-bawt.ops.request": digest,
                        },
                        volumes=volumes,
                    )
                except Exception as exc:
                    if getattr(exc, "status_code", None) != 409:
                        raise
                    worker = self._get_worker(name)
            self._verify_worker(worker, job_id, digest)
            self._start_created_once(worker, directory)
            return DispatchResult(name, str(directory / "receipt.json"))
        except ExecutorError:
            raise
        except Exception as exc:
            raise ExecutorError(f"Docker worker submission uncertain: {exc}") from exc

    @staticmethod
    def _start_created_once(worker, directory):
        # Container.start is not a compare-and-swap. Two observers of 'created'
        # could otherwise start, then RESTART a fast-exited worker. Serialize
        # status/start and persist the attempt BEFORE the daemon call. A lost
        # start response remains uncertain and is never retried blindly.
        with (directory / "dispatch.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            worker.reload()
            if worker.status != "created":
                return True
            marker = directory / "start-attempt.json"
            if marker.exists():
                return False
            atomic_json(marker, {"start_attempted": True})
            worker.start()
            return True

    @staticmethod
    def _verify_worker(worker, job_id, digest):
        labels = worker.labels or {}
        if labels.get("llm-bawt.ops.job") != job_id or labels.get("llm-bawt.ops.request") != digest:
            raise ExecutorError("deterministic worker identity conflict")

    def reconcile(self, *, job_id: str, snapshot: dict, output_tail_bytes=4096, **_kwargs):
        self.worker_name(job_id)
        root = Path(snapshot["execution"]["receipt_root"])
        if not root.is_dir():
            raise ExecutorError("receipt mount unavailable; cannot determine worker outcome")
        directory = root / job_id
        directory.mkdir(mode=0o700, exist_ok=True)
        with (directory / "submission.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return ReconcileResult(None)
            return self._reconcile_locked(job_id=job_id, snapshot=snapshot, output_tail_bytes=output_tail_bytes)

    def _reconcile_locked(self, *, job_id: str, snapshot: dict, output_tail_bytes=4096):
        directory = Path(snapshot["execution"]["receipt_root"]) / job_id
        digest = hashlib.sha256(canonical_json({"job_id": job_id, "snapshot": snapshot}).encode()).hexdigest()
        try:
            receipt = None
            path = directory / "receipt.json"
            if path.exists():
                if path.stat().st_size > 4 * 1024 * 1024:
                    raise ExecutorError("oversized worker receipt")
                receipt = json.loads(path.read_text())
                if receipt.get("job_id") != job_id or receipt.get("request_hash") != digest:
                    raise ExecutorError("worker receipt identity conflict")
                state = receipt.get("state")
                if state not in {"accepted", "running", "succeeded", "failed", "lost", "timed_out"}:
                    raise ExecutorError("invalid worker receipt state")
                if state in {"succeeded", "failed", "lost", "timed_out"}:
                    if not receipt.get("finished_at"):
                        raise ExecutorError("terminal worker receipt has no finish timestamp")
                    if state == "succeeded" and receipt.get("exit_code") != 0:
                        raise ExecutorError("invalid success receipt")
                    return ReconcileResult(state, receipt.get("exit_code"),
                        (receipt.get("output_tail") or "").encode()[-max(1, output_tail_bytes):].decode(errors="replace"),
                        receipt.get("error"), receipt.get("started_at"), receipt.get("finished_at"))
            worker = self._get_worker(self.worker_name(job_id))
            if worker is None:
                atomic_json(directory / "abandoned.json", {"reason": "worker missing during reconciliation"})
                return ReconcileResult("lost", error="worker missing; side effects unknown, not replayed")
            self._verify_worker(worker, job_id, digest)
            worker.reload()
            if worker.status == "created":
                # Creation succeeded but submitter died before start. Safe to
                # start this SAME never-run identity, never create a replacement.
                if not directory.is_dir() or not self._start_created_once(worker, directory):
                    return ReconcileResult("lost", error="worker start previously attempted without confirmation; not replayed")
                return ReconcileResult("accepted")
            if worker.status in ("exited", "dead", "removing"):
                # Receipt publication may have raced our first read, immediately
                # before Docker reported exit. Re-read once after that observation.
                if path.exists():
                    final = json.loads(path.read_text())
                    if final.get("job_id") != job_id or final.get("request_hash") != digest:
                        raise ExecutorError("worker receipt identity conflict")
                    if final.get("state") in {"succeeded", "failed", "lost", "timed_out"}:
                        return self._reconcile_locked(job_id=job_id, snapshot=snapshot,
                                                      output_tail_bytes=output_tail_bytes)
                return ReconcileResult("lost", error="worker exited without terminal receipt; side effects unknown")
            if receipt and receipt.get("state") == "running":
                return ReconcileResult("running", started_at=receipt.get("started_at"))
            return ReconcileResult("accepted")
        except ExecutorError:
            raise
        except Exception as exc:
            raise ExecutorError(f"worker reconciliation unavailable: {exc}") from exc
