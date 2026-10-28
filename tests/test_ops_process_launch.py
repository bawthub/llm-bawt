"""TASK-1169 S4a: process mode runs the pinned worker image's code as a child.

The code comes only from the worker image export (never this checkout), is
bundled deterministically, pinned by digest and re-verified from bytes in a
sealed memfd by the launcher. Children are bounded, supervised and observed
through the same receipt and ``worker.lock`` the isolated worker uses.
"""
from __future__ import annotations

import fcntl
import hashlib
import io
import json
import re
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

import llm_bawt.ops as ops_pkg
from llm_bawt.ops import process_launch as pl
from llm_bawt.ops.executor import DockerExecutor
from llm_bawt.ops.process_launch import (BUNDLE_PACKAGE, LAUNCHER, LAUNCHER_DIGEST, WORKER_FILES,
                                         ProcessLauncher, WorkerLaunchError, bundle_from_export)
from llm_bawt.ops.self_target import SELF_REQUIRES_ISOLATED, SelfTargetError
from llm_bawt.ops.worker_launch import RecoveryRefused
from llm_bawt.ops.worker_launch import Liveness

OPS_DIR = Path(ops_pkg.__file__).parent
REPO = OPS_DIR.parents[2]
IMAGE_ID = "sha256:" + "a" * 64
PY_IMAGE = "sha256:" + "b" * 64
JOB = "c" * 32
HANG_GUARD = 60  # timing-ok: hang guard for a real child, never an assertion bound


# ---- docker-save fixtures ------------------------------------------------------

def _layer(entries):
    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode="w") as tar:
        for name, kind, payload in entries:
            info = tarfile.TarInfo(name)
            if kind == "file":
                info.size = len(payload)
                tar.addfile(info, io.BytesIO(payload))
            elif kind == "dir":
                info.type = tarfile.DIRTYPE
                tar.addfile(info)
            elif kind == "sym":
                info.type, info.linkname = tarfile.SYMTYPE, payload
                tar.addfile(info)
            elif kind == "hard":
                info.type, info.linkname = tarfile.LNKTYPE, payload
                tar.addfile(info)
            elif kind == "fifo":
                info.type = tarfile.FIFOTYPE
                tar.addfile(info)
    return out.getvalue()


def _export(*layers):
    out = io.BytesIO()
    names = [f"l{i}/layer.tar" for i in range(len(layers))]
    with tarfile.open(fileobj=out, mode="w") as tar:
        for name, data in [("manifest.json", json.dumps([{"Layers": names}]).encode()),
                           *zip(names, (_layer(entries) for entries in layers))]:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return out.getvalue()


def _files(raw):
    import zipfile
    with zipfile.ZipFile(io.BytesIO(raw)) as bundle:
        return {name: bundle.read(name) for name in bundle.namelist()}


def _ops(**files):
    return [("ops", "dir", None)] + [(f"ops/{name}", "file", data) for name, data in files.items()]


class _Image:
    def __init__(self, image_id, export):
        self.id, self._export = image_id, export

    def save(self, named=False):
        assert named is False
        data = self._export
        return (data[i:i + 65536] for i in range(0, len(data), 65536))


class _Client:
    """images.get and image export only; any container call fails the test."""

    def __init__(self, export, image_id=IMAGE_ID):
        self.export, self.image_id = export, image_id
        outer = self

        class Images:
            @staticmethod
            def get(_ref):
                return _Image(outer.image_id, outer.export)

        class Containers:
            def __getattr__(self, name):
                raise AssertionError(f"process mode called containers.{name}")

        self.images, self.containers = Images(), Containers()


# ---- bundle extraction --------------------------------------------------------

def test_the_worker_file_list_is_the_dockerfile_copy_list():
    dockerfile = (REPO / "docker" / "Dockerfile.ops-worker").read_text()
    copied = set(re.findall(r"src/llm_bawt/ops/([a-z_]+\.py)", dockerfile))
    assert copied == WORKER_FILES
    assert all((OPS_DIR / name).is_file() for name in WORKER_FILES)


def test_layers_apply_in_order_with_whiteouts():
    raw = bundle_from_export(_export(
        _ops(**{"worker.py": b"old", "gone.py": b"x", "keep.py": b"k"}),
        [("ops/.wh.gone.py", "file", b""), ("ops/worker.py", "file", b"new"), ("etc/passwd", "file", b"no")],
    ))
    assert _files(raw) == {f"{BUNDLE_PACKAGE}/__init__.py": b"", f"{BUNDLE_PACKAGE}/keep.py": b"k",
                           f"{BUNDLE_PACKAGE}/worker.py": b"new"}


@pytest.mark.parametrize("whiteout", ["ops/.wh..wh..opq", ".wh.ops"])
def test_an_opaque_or_removed_ops_dir_drops_lower_layers(whiteout):
    raw = bundle_from_export(_export(_ops(**{"worker.py": b"a", "stale.py": b"s"}),
                                     [(whiteout, "file", b""), *_ops(**{"worker.py": b"b"})]))
    assert set(_files(raw)) == {f"{BUNDLE_PACKAGE}/__init__.py", f"{BUNDLE_PACKAGE}/worker.py"}


def test_the_bundle_is_deterministic():
    one = bundle_from_export(_export(_ops(**{"worker.py": b"w", "a.py": b"a"})))
    two = bundle_from_export(_export(_ops(**{"a.py": b"a", "worker.py": b"w"})))
    assert one == two


@pytest.mark.parametrize("entry", [
    ("ops/link.py", "sym", "/etc/passwd"),
    ("ops/hard.py", "hard", "ops/worker.py"),
    ("ops/pipe", "fifo", None),
    ("ops/sub/x.py", "file", b"x"),
    ("ops/sub", "dir", None),
    ("ops/sub/../worker.py", "file", b"x"),  # normalizes to ops/worker.py
    ("x/../ops/worker.py", "file", b"x"),
    ("ops/../ops/worker.py", "file", b"x"),
])
def test_unsafe_ops_entries_are_refused(entry):
    with pytest.raises(WorkerLaunchError) as exc:
        bundle_from_export(_export([*_ops(**{"worker.py": b"w"}), entry]))
    assert exc.value.code == "worker_unavailable"


def test_a_missing_worker_or_oversize_bundle_is_refused(monkeypatch):
    with pytest.raises(WorkerLaunchError, match="no /ops/worker.py"):
        bundle_from_export(_export(_ops(**{"other.py": b"x"})))
    monkeypatch.setattr(pl, "BUNDLE_MAX_BYTES", 10)
    with pytest.raises(WorkerLaunchError, match="size cap"):
        bundle_from_export(_export(_ops(**{"worker.py": b"w" * 11})))


# ---- launcher -----------------------------------------------------------------

def _launch(zip_path, digest, *rest):
    return subprocess.run([sys.executable, "-I", "-S", "-B", "-c", LAUNCHER, str(zip_path), digest, *rest],
                          capture_output=True, timeout=HANG_GUARD, check=False)


def _bundle_file(tmp_path, files):
    raw = bundle_from_export(_export(_ops(**files)))
    path = tmp_path / "bundle.zip"
    path.write_bytes(raw)
    return path, hashlib.sha256(raw).hexdigest()


def test_the_real_worker_modules_import_from_the_sealed_bundle(tmp_path):
    files = {name: (OPS_DIR / name).read_bytes() for name in WORKER_FILES}
    path, digest = _bundle_file(tmp_path, files)
    done = _launch(path, digest, "--self-check")
    assert done.returncode == 0, done.stderr.decode()


def test_the_memfd_is_sealed_against_writes(tmp_path):
    path, digest = _bundle_file(tmp_path, {"worker.py": b""})
    assert _launch(path, digest, "--seal-check").returncode == 0


def test_the_launcher_refuses_bytes_that_are_not_the_digest(tmp_path):
    path, digest = _bundle_file(tmp_path, {"worker.py": b""})
    path.write_bytes(path.read_bytes() + b"tamper")
    done = _launch(path, digest, "--self-check")
    assert done.returncode == 97 and b"worker_code_drift" in done.stderr


# ---- ProcessLauncher ------------------------------------------------------------

FAKE_WORKER = rb'''
import fcntl, json, os, sys, time
from datetime import datetime, timezone

def _now():
    return datetime.now(timezone.utc).isoformat()

def _receipt(directory, **fields):
    path = directory / "receipt.json"
    tmp = directory / "receipt.tmp"
    tmp.write_text(json.dumps(fields))
    os.replace(tmp, path)

def run(request_path, sha):
    directory = request_path.parent
    lock = (directory / "worker.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX)
    request = json.loads(request_path.read_text())
    behavior = request["behavior"]
    base = {"job_id": request["job_id"], "request_hash": sha, "accepted_at": _now(), "started_at": None}
    (directory / "env.json").write_text(json.dumps(dict(os.environ)))
    if behavior == "hang":
        _receipt(directory, state="accepted", **base)
        time.sleep(3600)
    if behavior == "hang_started":
        _receipt(directory, state="running", **{**base, "started_at": _now()})
        time.sleep(3600)
    if behavior == "spam":
        sys.stdout.write("x" * (request["spam"] - 3) + "END")
        sys.stdout.flush()
    if behavior == "crash":
        sys.stderr.write("early noise " * 20 + "Traceback: boom")
        sys.stderr.flush()
        sys.exit(3)
    _receipt(directory, state="succeeded", exit_code=0, finished_at=_now(), **{**base, "started_at": _now()})
    return 0
'''


@pytest.fixture
def rig(tmp_path):
    client = _Client(_export(_ops(**{"worker.py": FAKE_WORKER})))
    launcher = ProcessLauncher(client=lambda: client, python=sys.executable, python_image_id=PY_IMAGE,
                               kill_grace=0, boot_id=lambda: "boot-1")
    root = tmp_path / "receipts"
    root.mkdir()
    base = {"worker_image": IMAGE_ID, "receipt_volume": "v", "receipt_root": str(root)}
    settings = {**base, **launcher.execution_settings(base), "timeout_seconds": 30,
                "start_delay_seconds": 0, "max_output_bytes": 64}
    return launcher, settings, root, client


def _job(root, behavior, **extra):
    directory = root / JOB
    directory.mkdir(exist_ok=True)
    raw = json.dumps({"job_id": JOB, "behavior": behavior, **extra}).encode()
    (directory / "request.json").write_bytes(raw)
    return directory, hashlib.sha256(raw).hexdigest()


def _wait(launcher):
    child = launcher._children[JOB]
    assert child.done.wait(HANG_GUARD)
    return child


def test_prepare_pins_the_bundle_launcher_and_interpreter(rig):
    launcher, settings, root, _ = rig
    assert settings["launcher_digest"] == LAUNCHER_DIGEST
    assert settings["python_image_id"] == PY_IMAGE
    path = root / "code" / IMAGE_ID.split(":")[1] / f"{settings['code_digest']}.zip"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == settings["code_digest"]


def test_prepare_refuses_without_an_immutable_interpreter_identity(rig):
    launcher, settings, _, client = rig
    bare = ProcessLauncher(client=lambda: client, python_image_id="app:latest")
    with pytest.raises(WorkerLaunchError) as exc:
        bare.execution_settings(settings)
    assert exc.value.code == "worker_runtime_incompatible"


def test_a_bundle_that_does_not_import_is_runtime_incompatible(tmp_path):
    client = _Client(_export(_ops(**{"worker.py": b"import nonexistent_module_xyz\n"})))
    launcher = ProcessLauncher(client=lambda: client, python_image_id=PY_IMAGE)
    with pytest.raises(WorkerLaunchError) as exc:
        launcher.execution_settings({"worker_image": IMAGE_ID, "receipt_root": str(tmp_path)})
    assert exc.value.code == "worker_runtime_incompatible"


def test_a_job_runs_the_image_code_to_its_receipt_without_any_container(rig):
    launcher, settings, root, _ = rig
    directory, digest = _job(root, "ok")
    assert launcher.launch(job_id=JOB, settings=settings, directory=directory, digest=digest) == f"process:{JOB}"
    child = _wait(launcher)
    assert child.popen.returncode == 0
    assert json.loads((directory / "receipt.json").read_text())["state"] == "succeeded"
    assert set(json.loads((directory / "env.json").read_text())) <= {"PATH", "LANG", "PWD"}
    assert json.loads((directory / "spawn-intent.json").read_text())["attempt"] == 1
    assert json.loads((directory / "spawn.json").read_text())["pid"] == child.popen.pid
    assert launcher.probe(job_id=JOB, directory=directory, digest=digest, settings=settings) is Liveness.EXITED


def test_a_second_launch_never_spawns_again(rig):
    launcher, settings, root, _ = rig
    directory, digest = _job(root, "ok")
    launcher.launch(job_id=JOB, settings=settings, directory=directory, digest=digest)
    first = _wait(launcher)
    launcher.launch(job_id=JOB, settings=settings, directory=directory, digest=digest)
    assert launcher._children[JOB] is first


def test_output_beyond_the_cap_is_drained_and_counted(rig):
    launcher, settings, root, _ = rig
    directory, digest = _job(root, "spam", spam=640)
    launcher.launch(job_id=JOB, settings=settings, directory=directory, digest=digest)
    child = _wait(launcher)
    out = json.loads((directory / f"process-output.{child.attempt}.json").read_text())
    # The LAST bytes are kept: the end of the output is what explains a failure.
    assert len(out["stdout"]["tail"]) == settings["max_output_bytes"] and out["stdout"]["tail"].endswith("END")
    assert out["stdout"]["truncated"] and out["stdout"]["dropped_bytes"] == 640 - 64
    assert json.loads((directory / "receipt.json").read_text())["state"] == "succeeded"


@pytest.mark.parametrize("behavior, state, unknown", [("hang", "failed", False),
                                                      ("hang_started", "timed_out", True)])
def test_the_deadline_kills_the_group_and_records_what_is_known(rig, behavior, state, unknown):
    launcher, settings, root, _ = rig
    directory, digest = _job(root, behavior)
    # Half a second leaves the child time to boot and still fits the unit time budget.
    launcher.launch(job_id=JOB, settings={**settings, "timeout_seconds": 0.5}, directory=directory, digest=digest)
    child = _wait(launcher)
    receipt = json.loads((directory / "receipt.json").read_text())
    assert child.popen.returncode < 0
    assert (receipt["state"], receipt["side_effect_unknown"]) == (state, unknown)
    assert receipt["output_tail"].startswith(f"process worker exit code {child.popen.returncode}")


def test_a_worker_that_dies_without_a_receipt_leaves_its_exit_code_and_stderr_tail(rig):
    launcher, settings, root, _ = rig
    directory, digest = _job(root, "crash")
    launcher.launch(job_id=JOB, settings=settings, directory=directory, digest=digest)
    _wait(launcher)
    assert not (directory / "receipt.json").exists()
    diagnostic = launcher.diagnostics(directory)
    assert diagnostic.startswith("process worker exit code 3")
    assert "earlier bytes dropped" in diagnostic and diagnostic.endswith("Traceback: boom")


def test_a_lost_process_job_carries_the_launcher_diagnostics(rig):
    launcher, settings, root, _ = rig
    executor = DockerExecutor(client_factory=lambda: None, worker_image=IMAGE_ID,
                              receipt_volume="v", receipt_root=str(root))
    executor._launchers["process"] = launcher
    snapshot = {"execution": {"execution_mode": "process", "receipt_root": str(root)},
                "operation": {"execution_mode": "process"}}
    directory = root / JOB
    directory.mkdir()
    (directory / "spawn-intent.json").write_text(json.dumps({"attempt": 2}))
    for attempt, code in ((1, 3), (2, 97)):
        (directory / f"process-output.{attempt}.json").write_text(json.dumps(
            {"exit_code": code, "stdout": {"tail": ""}, "stderr": {"tail": f"attempt {attempt}"}}))
    result = executor.reconcile(job_id=JOB, snapshot=snapshot, output_tail_bytes=4096)
    assert result.state == "lost"
    assert result.output_tail == "process worker exit code 97\n--- stderr ---\nattempt 2"


# ---- crash table (no child of ours: a fresh reconciler) ---------------------------

def _fresh(rig):
    launcher, settings, root, client = rig
    return ProcessLauncher(client=lambda: client, python=sys.executable, python_image_id=PY_IMAGE,
                           kill_grace=0, boot_id=lambda: "boot-2"), settings, root


def test_a_held_worker_lock_is_alive(rig):
    launcher, settings, root = _fresh(rig)
    directory, digest = _job(root, "ok")
    with (directory / "worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        assert launcher.probe(job_id=JOB, directory=directory, digest=digest, settings=settings) is Liveness.ALIVE


def test_started_without_a_terminal_receipt_is_never_replayed(rig):
    launcher, settings, root = _fresh(rig)
    directory, digest = _job(root, "ok")
    (directory / "receipt.json").write_text(json.dumps({"state": "running", "started_at": "t"}))
    (directory / "spawn-intent.json").write_text(json.dumps({"attempt": 1}))
    assert launcher.probe(job_id=JOB, directory=directory, digest=digest, settings=settings) is Liveness.EXITED
    assert (directory / "abandoned.json").exists() and JOB not in launcher._children


def test_an_unstarted_job_is_respawned_once(rig):
    launcher, settings, root = _fresh(rig)
    directory, digest = _job(root, "ok")
    (directory / "spawn-intent.json").write_text(json.dumps({"attempt": 1}))
    assert launcher.probe(job_id=JOB, directory=directory, digest=digest, settings=settings) is Liveness.STARTED
    _wait(launcher)
    assert json.loads((directory / "spawn-intent.json").read_text())["attempt"] == 2
    assert json.loads((directory / "receipt.json").read_text())["state"] == "succeeded"


def test_a_second_unconfirmed_spawn_is_not_retried(rig):
    launcher, settings, root = _fresh(rig)
    directory, digest = _job(root, "ok")
    (directory / "spawn-intent.json").write_text(json.dumps({"attempt": 2}))
    assert launcher.probe(job_id=JOB, directory=directory, digest=digest,
                          settings=settings) is Liveness.START_UNCONFIRMED
    assert JOB not in launcher._children


def test_a_respawn_is_revalidated_first_and_a_refusal_propagates_as_after_a_prior_attempt(rig):
    launcher, settings, root = _fresh(rig)
    directory, digest = _job(root, "ok")
    (directory / "spawn-intent.json").write_text(json.dumps({"attempt": 1}))
    checks = []

    def refuse():
        checks.append("checked")
        raise SelfTargetError(SELF_REQUIRES_ISOLATED, "now the ops service")
    with pytest.raises(RecoveryRefused) as caught:
        launcher.probe(job_id=JOB, directory=directory, digest=digest, settings=settings, revalidate=refuse)
    # The earlier spawn's child may have outlived its dispatcher: nothing is certain.
    assert caught.value.prior_attempt is True and isinstance(caught.value.cause, SelfTargetError)
    assert checks == ["checked"] and JOB not in launcher._children
    assert json.loads((directory / "spawn-intent.json").read_text())["attempt"] == 1
    # Observing a finished job never revalidates (terminal receipt reads stay intact).
    (directory / "receipt.json").write_text(json.dumps({"state": "succeeded", "started_at": "t"}))
    assert launcher.probe(job_id=JOB, directory=directory, digest=digest, settings=settings,
                          revalidate=refuse) is Liveness.EXITED
    assert checks == ["checked"]


def test_a_respawn_refused_by_the_self_check_ends_lost_and_is_never_retried(rig, monkeypatch):
    launcher, _, root = _fresh(rig)
    executor = DockerExecutor(client_factory=lambda: None, worker_image=IMAGE_ID,
                              receipt_volume="v", receipt_root=str(root))
    executor._launchers["process"] = launcher
    snapshot = {"execution": {"execution_mode": "process", "receipt_root": str(root)},
                "operation": {"execution_mode": "process"}}
    directory = root / JOB
    directory.mkdir()
    (directory / "spawn-intent.json").write_text(json.dumps({"attempt": 1}))
    checks = []

    def refuse(_snapshot):
        checks.append("checked")
        raise SelfTargetError(SELF_REQUIRES_ISOLATED, "now the ops service")
    monkeypatch.setattr(executor, "self_target", refuse)
    result = executor.reconcile(job_id=JOB, snapshot=snapshot, output_tail_bytes=4096)
    # A spawn was intended before, so it is not known that nothing ran.
    assert result.state == "lost" and result.error.startswith("recovery start refused: " + SELF_REQUIRES_ISOLATED)
    assert "side effects unknown" in result.error and JOB not in launcher._children
    assert not (directory / "receipt.json").exists() and (directory / "abandoned.json").exists()
    # Lost from now on: no second check, no start, even once the check would pass.
    monkeypatch.setattr(executor, "self_target", lambda _snapshot: checks.append("again"))
    again = executor.reconcile(job_id=JOB, snapshot=snapshot, output_tail_bytes=4096)
    assert (again.state, again.error) == ("lost", result.error) and checks == ["checked"]
    assert JOB not in launcher._children


def test_a_failed_respawn_is_unconfirmed(rig):
    launcher, settings, root = _fresh(rig)
    directory, digest = _job(root, "ok")
    (directory / "spawn-intent.json").write_text(json.dumps({"attempt": 1}))

    def refuse(*_a, **_k):
        raise OSError("no")
    launcher._popen = refuse
    assert launcher.probe(job_id=JOB, directory=directory, digest=digest,
                          settings=settings) is Liveness.START_UNCONFIRMED


# ---- drift --------------------------------------------------------------------

@pytest.mark.parametrize("damage", ["bundle_bytes", "bundle_missing", "launcher", "python", "repinned"])
def test_anything_changed_since_approval_is_worker_code_drift(rig, damage):
    launcher, settings, root, client = rig
    path = root / "code" / IMAGE_ID.split(":")[1] / f"{settings['code_digest']}.zip"
    if damage == "bundle_bytes":
        path.chmod(0o600)
        path.write_bytes(b"other")
    elif damage == "bundle_missing":
        path.unlink()
    elif damage == "launcher":
        settings = {**settings, "launcher_digest": "0" * 64}
    elif damage == "python":
        launcher._python_image_id = "sha256:" + "d" * 64
    else:
        client.image_id = "sha256:" + "e" * 64
    with pytest.raises(WorkerLaunchError) as exc:
        launcher.preflight(settings)
    assert exc.value.code == "worker_code_drift"


def test_a_published_bundle_is_write_once(tmp_path):
    path = tmp_path / "code" / "x.zip"
    pl.publish_once(path, b"one")
    pl.publish_once(path, b"one")
    with pytest.raises(WorkerLaunchError, match="differs"):
        pl.publish_once(path, b"two")


# ---- executor wiring ------------------------------------------------------------

def test_the_docker_executor_runs_process_mode_in_the_process_launcher():
    executor = DockerExecutor(client_factory=lambda: None, worker_image=IMAGE_ID,
                              receipt_volume="v", receipt_root="/nonexistent")
    snapshot = {"execution": {"execution_mode": "process"}, "operation": {"execution_mode": "process"}}
    assert isinstance(executor._launcher_for(snapshot), ProcessLauncher)
    assert executor.supported_execution_modes() == {"process", "isolated"}
