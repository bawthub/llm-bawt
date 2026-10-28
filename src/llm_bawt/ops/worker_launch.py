"""How a DockerExecutor job's worker is started and observed (TASK-1169 S4).

The executor owns everything job-level: the immutable request, the receipt and
the mapping to job states. A :class:`WorkerLauncher` owns only the worker's
runtime: starting the one-shot ``worker.run`` for a published request, and
reporting whether that worker is still alive. Every launcher runs the same
worker against the same request, receipt and lock files, so the receipt is the
only outcome either can report.

:class:`ContainerLauncher` is the ``isolated`` mode: one deterministic, locked-down
container per job from the pinned worker image, moved here unchanged from
``DockerExecutor``.
"""
from __future__ import annotations

import fcntl
import re
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from .executor_types import ExecutorError
from .worker import OPS_JOB_LABEL, atomic_json


class Liveness(Enum):
    """What a launcher can see of a job's worker, receipt aside."""

    MISSING = "missing"                      # no worker exists for this job
    START_UNCONFIRMED = "start_unconfirmed"  # a start was attempted, never confirmed
    STARTED = "started"                      # an unstarted worker was started just now
    EXITED = "exited"                        # the worker has ended
    ALIVE = "alive"                          # the worker is running


class WorkerLauncher(ABC):
    """One way to run ``worker.run`` for a published job request."""

    @abstractmethod
    def launch(self, *, job_id: str, settings: dict, directory: Path, digest: str) -> str:
        """Start (or adopt) the job's worker; return its stable handle name."""

    @abstractmethod
    def probe(self, *, job_id: str, directory: Path, digest: str, settings: dict | None = None) -> Liveness:
        """Observe the job's worker. May start a never-started worker exactly once."""

    def execution_settings(self, settings: dict) -> dict:
        """Extra snapshot pins this runtime needs at prepare; none by default."""
        return {}

    def preflight(self, settings: dict) -> None:
        """Refuse before dispatch if the approved runtime pins no longer hold."""

    def diagnostics(self, directory: Path) -> str | None:
        """Bounded evidence for a worker that ended without a terminal receipt."""
        return None


def container_worker_name(job_id: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{32}", job_id):
        raise ExecutorError("invalid job id")
    return f"llm-bawt-ops-{job_id}"


class ContainerLauncher(WorkerLauncher):
    """Isolated mode: a deterministic one-shot worker container per job."""

    def __init__(self, *, client: Callable[[], Any], get_worker: Callable[[str], Any]):
        # Both are late-bound so the executor's lazy client and its worker
        # lookup stay the single source (and stay patchable in tests).
        self._client = client
        self._get_worker = get_worker

    def launch(self, *, job_id, settings, directory, digest):
        name = container_worker_name(job_id)
        worker = self._get_worker(name)
        volumes = {settings["receipt_volume"]: {"bind": "/receipts", "mode": "rw"},
                   "/var/run/docker.sock": {"bind": "/var/run/docker.sock", "mode": "rw"}}
        if worker is None:
            try:
                worker = self._client().containers.create(
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
        return name

    def probe(self, *, job_id, directory, digest, settings=None):
        worker = self._get_worker(container_worker_name(job_id))
        if worker is None:
            return Liveness.MISSING
        self._verify_worker(worker, job_id, digest)
        worker.reload()
        if worker.status == "created":
            # Creation succeeded but submitter died before start. Safe to
            # start this SAME never-run identity, never create a replacement.
            if not directory.is_dir() or not self._start_created_once(worker, directory):
                return Liveness.START_UNCONFIRMED
            return Liveness.STARTED
        if worker.status in ("exited", "dead", "removing"):
            return Liveness.EXITED
        return Liveness.ALIVE

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
