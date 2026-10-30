"""Process mode: run ``worker.run`` as a bounded child of the ops service (TASK-1169 S4).

The trust root is the same artifact isolated mode runs: the immutable worker
image (``LLM_BAWT_OPS_WORKER_IMAGE``). Its ``/ops/`` files are taken from a Docker
image export (no container is created), bundled into a deterministic zip and
published write-once on the receipt volume under the image ID. ``code_digest``
is the zip's sha256 and is pinned in the approved snapshot, so an edited or
unreviewed ``worker.py`` in a source checkout can never reach either mode.

The child runs ``python -I -S -B -c LAUNCHER``. The launcher re-hashes the zip
from bytes, copies it into a fully sealed memfd and zipimports from there, so
the code that executes is exactly the code that was hashed.

Liveness is the worker's own ``worker.lock`` flock plus this process's child
table; the reconciler's children die with it (PID 1), so a fresh reconciler
sees only receipts, ``spawn-intent.json`` and a free lock.
"""
from __future__ import annotations

import fcntl
import hashlib
import io
import json
import os
import posixpath
import re
import resource
import signal
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .executor_types import ExecutorError
from .worker import atomic_json, utcnow
from .worker_launch import ABANDONED, Liveness, WorkerLauncher, container_worker_name, refuse_recovery

#: The files ``docker/Dockerfile.ops-worker`` copies to ``/ops/`` (pinned by a test).
WORKER_FILES = frozenset({
    "worker.py", "image_deploy.py", "routing_reconcile.py", "compose_render.py", "compose_config.py",
    "config_reconcile.py", "app_config_reconcile.py", "sealed_container.py", "source_state.py",
    "service_identity.py", "tagged_config_reconcile.py", "plan_reconcile.py", "probe_contract.py",
    "container_remove.py", "dev_config_reconcile.py", "restart_health.py", "health_probe.py",
})
BUNDLE_MAX_BYTES = 4 * 1024 * 1024
EXPORT_MAX_BYTES = 4 * 1024 * 1024 * 1024
_ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)
#: The bundle is a package so the worker takes its relative-import branch; its
#: script branch would put the resolved memfd's parent (``/``) on sys.path.
BUNDLE_PACKAGE = "llm_bawt_ops_worker"
_SELF_CHECK_TIMEOUT = 60
_LIMITS = ((resource.RLIMIT_AS, 512 * 1024 * 1024), (resource.RLIMIT_NOFILE, 256),
           (resource.RLIMIT_CORE, 0))
_TERMINAL = ("succeeded", "failed", "lost", "timed_out")

# Runs as ``python -I -S -B -c LAUNCHER <zip> <digest> (--self-check | --run <request> <sha>)``.
LAUNCHER = r'''
import fcntl, hashlib, os, sys
zip_path, want, mode = sys.argv[1], sys.argv[2], sys.argv[3]
with open(zip_path, "rb") as stream:
    data = stream.read()
if hashlib.sha256(data).hexdigest() != want:
    sys.stderr.write("worker_code_drift: bundle bytes do not match the approved digest\n")
    sys.exit(97)
# Linux ABI values; some CPython builds lack the os/fcntl wrappers, libc has them.
MFD_CLOEXEC, MFD_ALLOW_SEALING, F_ADD_SEALS = 1, 2, 1033
F_SEAL_SEAL, F_SEAL_SHRINK, F_SEAL_GROW, F_SEAL_WRITE = 1, 2, 4, 8
if hasattr(os, "memfd_create"):
    fd = os.memfd_create("llm-bawt-ops-worker", MFD_ALLOW_SEALING | MFD_CLOEXEC)
else:
    import ctypes
    libc = ctypes.CDLL(None, use_errno=True)
    libc.memfd_create.argtypes, libc.memfd_create.restype = [ctypes.c_char_p, ctypes.c_uint], ctypes.c_int
    fd = libc.memfd_create(b"llm-bawt-ops-worker", MFD_ALLOW_SEALING | MFD_CLOEXEC)
    if fd < 0:
        raise OSError(ctypes.get_errno(), "memfd_create failed")
view = memoryview(data)
while view:
    view = view[os.write(fd, view):]
fcntl.fcntl(fd, F_ADD_SEALS, F_SEAL_SEAL | F_SEAL_SHRINK | F_SEAL_GROW | F_SEAL_WRITE)
sys.path.insert(0, "/proc/self/fd/%d" % fd)
package = "llm_bawt_ops_worker"
if mode == "--self-check":
    import importlib, zipfile
    for name in sorted(zipfile.ZipFile("/proc/self/fd/%d" % fd).namelist()):
        if name.endswith(".py") and not name.endswith("/__init__.py"):
            importlib.import_module(name[:-3].replace("/", "."))
    sys.exit(0)
if mode == "--seal-check":
    try:
        os.write(fd, b"x")
    except PermissionError:
        sys.exit(0)
    sys.exit(1)
import importlib, pathlib
worker = importlib.import_module(package + ".worker")
sys.exit(worker.run(pathlib.Path(sys.argv[4]), sys.argv[5]))
'''
LAUNCHER_DIGEST = hashlib.sha256(LAUNCHER.encode()).hexdigest()


class WorkerLaunchError(ExecutorError):
    """A fixed-code refusal: ``worker_code_drift``, ``worker_runtime_incompatible``
    or ``worker_unavailable``. The message names files and images, never content."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"{code}: {message}")


def process_worker_name(job_id: str) -> str:
    container_worker_name(job_id)  # the same job-id validation
    return f"process:{job_id}"


def _boot_id() -> str:
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        return ""


def _set_limits():  # pragma: no cover - runs in the forked child
    for kind, value in _LIMITS:
        resource.setrlimit(kind, (value, value))


def _scrubbed_env() -> dict[str, str]:
    # No database, Redis or llm-bawt settings: the worker is stdlib-only and
    # reaches only the Docker socket path it already knows.
    return {"PATH": os.environ.get("PATH") or "/usr/local/bin:/usr/bin:/bin", "LANG": "C.UTF-8"}


# ---- bundle -----------------------------------------------------------------

def bundle_from_export(stream) -> bytes:
    """The deterministic zip of an image export's ``/ops/`` files.

    Layers apply in order with OCI whiteouts. Refuses a symlink, hardlink, device,
    nested path or ``..`` under ``/ops/``, a missing ``worker.py`` and an oversize
    bundle.
    """
    if isinstance(stream, (bytes, bytearray)):
        stream = io.BytesIO(stream)
    files: dict[str, bytes] = {}
    with tarfile.open(fileobj=stream, mode="r:*") as outer:
        manifest = json.load(outer.extractfile("manifest.json"))
        if not isinstance(manifest, list) or len(manifest) != 1:
            raise WorkerLaunchError("worker_unavailable", "worker image export must hold exactly one image")
        for layer in manifest[0]["Layers"]:
            with tarfile.open(fileobj=outer.extractfile(layer), mode="r:*") as inner:
                for member in inner:
                    _apply(files, member, inner)
    if "worker.py" not in files:
        raise WorkerLaunchError("worker_unavailable", "worker image has no /ops/worker.py")
    if sum(len(data) for data in files.values()) > BUNDLE_MAX_BYTES:
        raise WorkerLaunchError("worker_unavailable", "worker bundle exceeds its size cap")
    bundle = {f"{BUNDLE_PACKAGE}/__init__.py": b""}
    bundle.update({f"{BUNDLE_PACKAGE}/{name}": data for name, data in files.items()})
    return deterministic_zip(bundle)


def _apply(files, member, inner):
    raw = member.name.lstrip("/")
    name = posixpath.normpath(raw)
    # Before normalizing hides it: ops/sub/../worker.py would become ops/worker.py.
    parts = [part for part in raw.split("/") if part not in ("", ".")]
    if ".." in parts and (parts[0] == "ops" or name == "ops" or name.startswith("ops/")):
        raise WorkerLaunchError("worker_unavailable", "worker image /ops entry escapes its directory")
    if name.startswith("./"):
        name = name[2:]
    parent, base = posixpath.split(name)
    if name in (".wh.ops", ".wh..wh..opq") or (name == "ops" and not member.isdir()):
        if name == "ops":
            raise WorkerLaunchError("worker_unavailable", "worker image /ops is not a directory")
        files.clear()
        return
    if name == "ops" or not name.startswith("ops/"):
        return
    if ".." in name.split("/"):
        raise WorkerLaunchError("worker_unavailable", "worker image /ops entry escapes its directory")
    if base == ".wh..wh..opq":
        if parent == "ops":
            files.clear()
        return
    if base.startswith(".wh."):
        if parent == "ops":
            files.pop(base[4:], None)
        return
    if member.issym() or member.islnk():
        raise WorkerLaunchError("worker_unavailable", f"worker image /{name} is a link")
    if member.isdir() or parent != "ops":
        raise WorkerLaunchError("worker_unavailable", f"worker image /{name} is not a flat /ops file")
    if not member.isfile():
        raise WorkerLaunchError("worker_unavailable", f"worker image /{name} is not a regular file")
    if member.size > BUNDLE_MAX_BYTES:
        raise WorkerLaunchError("worker_unavailable", "worker bundle exceeds its size cap")
    files[base] = inner.extractfile(member).read()


def deterministic_zip(files: dict[str, bytes]) -> bytes:
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as bundle:
        for name in sorted(files):
            info = zipfile.ZipInfo(name, date_time=_ZIP_EPOCH)
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            bundle.writestr(info, files[name])
    return out.getvalue()


def publish_once(path: Path, raw: bytes) -> None:
    """Write-once publication, the same temp, link and fsync path as request.json."""
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if not path.exists():
        temp = path.parent / f".{path.name}.{os.getpid()}.{os.urandom(8).hex()}.tmp"
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
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    if path.read_bytes() != raw:
        raise WorkerLaunchError("worker_code_drift", f"published bundle {path.name} differs from its bytes")


# ---- child supervision --------------------------------------------------------

class _Capped:
    """Drains one pipe, keeping only its last ``limit`` bytes; the rest is counted."""

    def __init__(self, pipe, limit: int):
        self.buffer = bytearray()
        self.dropped = 0
        self._pipe, self._limit = pipe, max(1, limit)
        self.thread = threading.Thread(target=self._drain, daemon=True)
        self.thread.start()

    def _drain(self):
        with self._pipe:
            for chunk in iter(lambda: self._pipe.read(65536), b""):
                self.buffer += chunk
                excess = len(self.buffer) - self._limit
                if excess > 0:
                    del self.buffer[:excess]
                    self.dropped += excess

    def record(self) -> dict:
        return {"tail": self.buffer.decode(errors="replace"), "truncated": self.dropped > 0,
                "dropped_bytes": self.dropped}


@dataclass
class _Child:
    popen: Any
    attempt: int
    started_monotonic: float = 0.0
    done: threading.Event = field(default_factory=threading.Event)


class ProcessLauncher(WorkerLauncher):
    """Process mode for :class:`DockerExecutor`; no container is ever created."""

    def __init__(self, *, client: Callable[[], Any], python: str | None = None,
                 python_image_id: str | None = None, popen=subprocess.Popen,
                 kill_grace: float | None = None, boot_id: Callable[[], str] = _boot_id,
                 clock: Callable[[], float] = time.monotonic,
                 own_image: Callable[[], tuple[str, list[str]]] | None = None):
        self._client = client
        self._python = python or sys.executable
        # The interpreter's identity: the reconciler's own immutable image.
        self._python_image_id = (python_image_id if python_image_id is not None
                                 else os.getenv("LLM_BAWT_OPS_RECONCILER_IMAGE", ""))
        self._own_image = own_image or self._inspect_own_image
        self._popen = popen
        self._kill_grace = (kill_grace if kill_grace is not None
                            else float(os.getenv("LLM_BAWT_OPS_WORKER_KILL_GRACE", "10")))
        self._boot_id = boot_id
        self._clock = clock
        # Each health tick prunes completed children after their receipt and
        # output are durable; probes then use the receipt and worker.lock.
        self._children: dict[str, _Child] = {}
        self._bundles: dict[tuple[str, str], str] = {}  # (receipt_root, image id) -> digest
        self._lock = threading.Lock()

    # -- prepare / preflight --------------------------------------------------

    def _inspect_own_image(self) -> tuple[str, list[str]]:
        """Inspect the interpreter image actually running this reconciler."""
        owner = os.getenv("LLM_BAWT_OPS_APP_CONTAINER") or os.getenv("HOSTNAME", "")
        if not owner:
            raise ValueError("the reconciler's container identity is missing")
        container = self._client().containers.get(owner)
        image_id = container.attrs["Image"]
        image = self._client().images.get(image_id)
        if image.id != image_id:
            raise ValueError("the running container's image cannot be verified")
        return image_id, image.attrs.get("RepoDigests") or []

    def _verified_interpreter(self) -> str:
        """The configured immutable ref must resolve to our running image ID."""
        ref = self._python_image_id or ""
        if not re.fullmatch(r"(?:sha256:|[^\s]+@sha256:)[0-9a-f]{64}", ref):
            raise WorkerLaunchError("worker_runtime_incompatible",
                                    "configure LLM_BAWT_OPS_RECONCILER_IMAGE as immutable image ID or digest")
        try:
            image_id, digests = self._own_image()
        except Exception as exc:
            raise WorkerLaunchError("worker_runtime_incompatible",
                                    "cannot inspect the reconciler's running image") from exc
        if (not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id or "")
                or (ref != image_id and ref not in digests)):
            raise WorkerLaunchError("worker_runtime_incompatible",
                                    "the configured interpreter image does not match the running reconciler")
        return image_id

    def interpreter_pinned(self) -> bool:
        """Read-only health check of the configured reference against Docker."""
        try:
            self._verified_interpreter()
            return True
        except WorkerLaunchError:
            return False

    def execution_settings(self, settings: dict) -> dict:
        """The process-mode pins for a snapshot: bundle, launcher and interpreter."""
        self._verified_interpreter()
        image_id = self._image_id(settings["worker_image"])
        return {"code_digest": self._ensure_bundle(settings["receipt_root"], image_id),
                "launcher_digest": LAUNCHER_DIGEST, "python_image_id": self._python_image_id}

    def preflight(self, settings: dict) -> None:
        """Refuse before any slot or spawn if what was approved can no longer run."""
        if settings.get("launcher_digest") != LAUNCHER_DIGEST:
            raise WorkerLaunchError("worker_code_drift", "the process launcher changed since approval")
        if settings.get("python_image_id") != self._python_image_id:
            raise WorkerLaunchError("worker_code_drift", "the ops service image changed since approval")
        try:
            self._verified_interpreter()
        except WorkerLaunchError as exc:
            raise WorkerLaunchError("worker_code_drift", "the approved interpreter no longer matches the running image") from exc
        path = self._bundle_path(settings["receipt_root"], self._image_id(settings["worker_image"]),
                                 settings.get("code_digest", ""))
        try:
            raw = path.read_bytes()
        except OSError:
            raise WorkerLaunchError("worker_code_drift", "the approved worker bundle is missing") from None
        if hashlib.sha256(raw).hexdigest() != settings["code_digest"]:
            raise WorkerLaunchError("worker_code_drift", "the approved worker bundle changed")

    def _image_id(self, worker_image: str) -> str:
        try:
            image_id = self._client().images.get(worker_image).id
        except Exception as exc:
            raise WorkerLaunchError("worker_unavailable", "cannot inspect the configured worker image") from exc
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id or ""):
            raise WorkerLaunchError("worker_unavailable", "worker image has no immutable ID")
        return image_id

    @staticmethod
    def _bundle_path(receipt_root, image_id, digest) -> Path:
        if not re.fullmatch(r"[0-9a-f]{64}", digest or ""):
            raise WorkerLaunchError("worker_code_drift", "snapshot has no valid worker bundle digest")
        return Path(receipt_root) / "code" / image_id.split(":", 1)[1] / f"{digest}.zip"

    def _ensure_bundle(self, receipt_root: str, image_id: str) -> str:
        key = (receipt_root, image_id)
        with self._lock:
            if key in self._bundles:
                return self._bundles[key]
        raw = self._export_bundle(image_id)
        digest = hashlib.sha256(raw).hexdigest()
        path = self._bundle_path(receipt_root, image_id, digest)
        publish_once(path, raw)
        self._self_check(path, digest)
        with self._lock:
            self._bundles[key] = digest
        return digest

    def _export_bundle(self, image_id: str) -> bytes:
        try:
            chunks = self._client().images.get(image_id).save(named=False)
            with tempfile.TemporaryFile() as spool:
                size = 0
                for chunk in chunks:
                    size += len(chunk)
                    if size > EXPORT_MAX_BYTES:
                        raise WorkerLaunchError("worker_unavailable", "worker image export exceeds its size cap")
                    spool.write(chunk)
                spool.seek(0)
                return bundle_from_export(spool)
        except WorkerLaunchError:
            raise
        except Exception as exc:
            raise WorkerLaunchError("worker_unavailable", f"cannot export the worker image: {type(exc).__name__}") from exc

    def _self_check(self, path: Path, digest: str) -> None:
        argv = [self._python, "-I", "-S", "-B", "-c", LAUNCHER, str(path), digest, "--self-check"]
        try:
            done = subprocess.run(argv, env=_scrubbed_env(), stdin=subprocess.DEVNULL, capture_output=True,
                                  timeout=_SELF_CHECK_TIMEOUT, close_fds=True, check=False)
        except (OSError, subprocess.SubprocessError) as exc:
            raise WorkerLaunchError("worker_runtime_incompatible",
                                    f"worker self-check did not run: {type(exc).__name__}") from exc
        if done.returncode != 0:
            tail = done.stderr.decode(errors="replace").strip().splitlines()[-1:] or ["no output"]
            raise WorkerLaunchError("worker_runtime_incompatible", f"worker self-check failed: {tail[0][:200]}")

    # -- launch / probe -------------------------------------------------------

    def launch(self, *, job_id, settings, directory, digest):
        name = process_worker_name(job_id)
        if (directory / "spawn-intent.json").exists():
            return name  # Spawned once already; reconciliation owns it from here.
        self._spawn(job_id=job_id, settings=settings, directory=directory, digest=digest, attempt=1)
        return name

    def probe(self, *, job_id, directory, digest, settings=None, revalidate=None):
        process_worker_name(job_id)
        with self._lock:
            child = self._children.get(job_id)
        if child is not None and not child.done.is_set():
            return Liveness.ALIVE
        if _lock_held(directory / "worker.lock"):
            return Liveness.ALIVE
        receipt = _read_json(directory / "receipt.json")
        if receipt and (receipt.get("state") in _TERMINAL or receipt.get("started_at")):
            if receipt.get("state") not in _TERMINAL:
                # Execution began and the worker is gone: never replay it.
                atomic_json(directory / ABANDONED,
                            {"reason": "worker process ended after execution began; side effects unknown"})
            return Liveness.EXITED
        intent = _read_json(directory / "spawn-intent.json") or {}
        attempt = int(intent.get("attempt") or 0)
        if attempt >= 2 or settings is None or not directory.is_dir() or (directory / ABANDONED).exists():
            return Liveness.START_UNCONFIRMED  # Fenced or spent: never revalidated, never respawned.
        # A spawn was intended before, and its child may have outlived its
        # dispatcher (own session): a refusal leaves side effects unknown.
        refuse_recovery(revalidate, directory, prior_attempt=True)
        try:
            # Nothing began executing: one respawn of the SAME request. The
            # worker re-checks the start delay and the fence inside.
            self._spawn(job_id=job_id, settings=settings, directory=directory, digest=digest,
                        attempt=attempt + 1)
        except ExecutorError:
            return Liveness.START_UNCONFIRMED
        return Liveness.STARTED

    def live_children(self) -> int:
        return self.child_stats()[0]

    def child_stats(self) -> tuple[int, float | None]:
        """Prune completed children each tick; preserve receipt-based probes."""
        with self._lock:
            for job_id, child in list(self._children.items()):
                if child.done.is_set():
                    del self._children[job_id]
            starts = [child.started_monotonic for child in self._children.values()]
        return len(starts), min(starts) if starts else None

    def _spawn(self, *, job_id, settings, directory, digest, attempt):
        self.preflight(settings)
        bundle = self._bundle_path(settings["receipt_root"], self._image_id(settings["worker_image"]),
                                   settings["code_digest"])
        atomic_json(directory / "spawn-intent.json",
                    {"attempt": attempt, "intended_at": utcnow(), "boot_id": self._boot_id()})
        argv = [self._python, "-I", "-S", "-B", "-c", LAUNCHER, str(bundle), settings["code_digest"],
                "--run", str(directory / "request.json"), digest]
        try:
            popen = self._popen(argv, cwd=str(directory), env=_scrubbed_env(), stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, close_fds=True,
                                start_new_session=True, preexec_fn=_set_limits)
        except Exception as exc:
            raise WorkerLaunchError("worker_unavailable", f"worker spawn failed: {type(exc).__name__}") from exc
        atomic_json(directory / "spawn.json", {"attempt": attempt, "pid": popen.pid, "pgid": popen.pid})
        child = _Child(popen, attempt, started_monotonic=self._clock())
        with self._lock:
            self._children[job_id] = child
        deadline = (float(settings.get("start_delay_seconds") or 0) + float(settings["timeout_seconds"])
                    + self._kill_grace)
        threading.Thread(target=self._supervise, daemon=True,
                         args=(job_id, child, directory, digest, deadline,
                               int(settings["max_output_bytes"]))).start()

    def _supervise(self, job_id, child, directory, digest, deadline, limit):
        streams = {"stdout": _Capped(child.popen.stdout, limit), "stderr": _Capped(child.popen.stderr, limit)}
        killed = False
        try:
            child.popen.wait(timeout=deadline)
        except subprocess.TimeoutExpired:
            killed = True
            try:
                os.killpg(child.popen.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.popen.wait()
        for stream in streams.values():
            stream.thread.join()
        record = {"exit_code": child.popen.returncode, **{name: stream.record() for name, stream in streams.items()}}
        atomic_json(directory / f"process-output.{child.attempt}.json", record)
        if killed:
            _write_kill_receipt(directory, job_id, digest, output_tail=diagnostic_tail(record))
        child.done.set()

    def diagnostics(self, directory):
        """The latest attempt's exit code and capped output, for a job that ended
        without a terminal receipt of its own (a crash, drift exit, traceback)."""
        attempts = sorted((int(p.name.split(".")[1]), p) for p in directory.glob("process-output.*.json")
                          if p.name.split(".")[1].isdigit())
        if not attempts:
            return None
        record = _read_json(attempts[-1][1])
        return diagnostic_tail(record) if isinstance(record, dict) else None


def diagnostic_tail(record: dict) -> str:
    """One bounded text: the exit code, then each stream's retained tail."""
    parts = [f"process worker exit code {record.get('exit_code')}"]
    for name in ("stderr", "stdout"):
        stream = record.get(name) or {}
        if stream.get("tail"):
            dropped = f" ({stream.get('dropped_bytes')} earlier bytes dropped)" if stream.get("truncated") else ""
            parts.append(f"--- {name}{dropped} ---\n{stream['tail']}")
    return "\n".join(parts)


def _lock_held(path: Path) -> bool:
    try:
        fd = os.open(path, os.O_RDONLY)
    except FileNotFoundError:
        return False
    try:
        fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
    except BlockingIOError:
        return True
    finally:
        os.close(fd)  # closing releases a shared lock we took
    return False


def _read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None


def _write_kill_receipt(directory: Path, job_id: str, digest: str, *, output_tail: str | None = None) -> None:
    """The deadline killed the worker: write its terminal state unless it did."""
    path = directory / "receipt.json"
    receipt = _read_json(path) or {"job_id": job_id, "request_hash": digest, "accepted_at": None,
                                   "started_at": None}
    if receipt.get("state") in _TERMINAL:
        return
    started = bool(receipt.get("started_at"))
    receipt.update(
        state="timed_out" if started else "failed", exit_code=None, side_effect_unknown=started,
        finished_at=utcnow(), output_tail=output_tail,
        error=("worker killed at its deadline after execution began; side effects unknown" if started
               else "worker killed at its deadline before execution began; no action submitted"))
    atomic_json(path, receipt)


__all__ = ["BUNDLE_MAX_BYTES", "BUNDLE_PACKAGE", "LAUNCHER", "LAUNCHER_DIGEST", "ProcessLauncher", "WORKER_FILES",
           "WorkerLaunchError", "bundle_from_export", "deterministic_zip", "diagnostic_tail", "process_worker_name",
           "publish_once"]
