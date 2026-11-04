"""Batch-gate execution: trusted admission, reconciler claim and one child run.

Wires the existing gate contracts into one path. ``admit_gate`` freezes a
trusted QUEUED request with a direct callback derived from the signed turn
capability. Only the ops reconciler (``GateDriver``) claims it, then spawns
one child process per job: ``run_gate`` needs a process main thread for its
SIGALRM deadline. The child reserves launch custody and start capacity, runs
the socketless sandbox once, records the terminal state and attempts positive
retirement. The existing receipt outbox delivers the callback.

First attempts only. Nothing here retries, adopts an orphan or treats a lost
or failed state as permission for another attempt. A child that disappears
leaves its job LOST; its slot stays held until containers are positively
retired. Configuration is deployment mounts (env), never request input.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

from sqlalchemy import bindparam, text

from .models import (JOB_DISPATCHING, JOB_FAILED, JOB_LOST, JOB_QUEUED, JOB_SUCCEEDED,
                     JOB_TERMINAL_STATES)

logger = logging.getLogger(__name__)

GATE_ACTION = "batch_gate"
REPOSITORY_ENV = "LLM_BAWT_OPS_GATE_REPOSITORY"
SCRATCH_ENV = "LLM_BAWT_OPS_GATE_SCRATCH"
RECEIPT_ROOT_ENV = "LLM_BAWT_OPS_RECEIPT_ROOT"
_ORPHAN_GRACE_SECONDS = 60
_CLAIM_DEADLINE_SECONDS = 600
_DOCKER_TIMEOUT = 30.0
_OUTPUT_TAIL = 65536


@dataclass(frozen=True)
class GateDriverConfig:
    repository: Path
    scratch_parent: Path
    receipt_root: Path

    @classmethod
    def from_env(cls) -> GateDriverConfig | None:
        values = [os.getenv(name, "").strip() for name in (REPOSITORY_ENV, SCRATCH_ENV, RECEIPT_ROOT_ENV)]
        if not all(values):
            return None
        return cls(*(Path(value) for value in values))


def is_gate_snapshot(raw: str | None) -> bool:
    try:
        return bool(raw) and "gate_claim_schema" in json.loads(raw)
    except (TypeError, ValueError):
        return False


def admit_gate(store, *, operation_slug: str, args: dict, turn_token: str | None,
               idempotency_key: str) -> dict:
    """Create (or replay) one trusted QUEUED gate request. Never dispatches."""
    from .gate_trust import create_trusted_gate_job

    if not turn_token:
        raise ValueError("a batch gate needs the caller's signed turn capability")
    job = create_trusted_gate_job(store, operation_slug=operation_slug, plan=args,
                                  turn_token=turn_token, idempotency_key=idempotency_key,
                                  direct_callback=True)
    return job.to_api()


def _demux(output: bytes) -> bytes:
    """Strip Docker's non-TTY attach framing (8-byte header per frame).

    Malformed framing falls back to the raw bytes; NULs are removed below either way.
    """
    body, pos = bytearray(), 0
    while pos < len(output):
        header = output[pos:pos + 8]
        if len(header) < 8 or header[0] not in (0, 1, 2) or header[1:4] != b"\0\0\0":
            return output
        size = int.from_bytes(header[4:8], "big")
        if pos + 8 + size > len(output):
            return output
        body += output[pos + 8:pos + 8 + size]
        pos += 8 + size
    return bytes(body)


def _tail(output: bytes) -> str:
    # TASK-1865: raw frame headers carry NULs, which PostgreSQL text rejects;
    # that failed the terminal write of a green run and left it LOST.
    return _demux(output)[-_OUTPUT_TAIL:].replace(b"\0", b"").decode("utf-8", errors="replace")


def run_child(store, job_id: str, config: GateDriverConfig, *, api=None) -> str:
    """The whole post-claim attempt for one job. Returns the terminal state."""
    from .gate_custody import GateCustodyError, reserve_gate_launch
    from .gate_retire import GateRetireError, retire_gate_slot
    from .gate_runner import GateLaunchError, run_gate
    from .gate_source import GateSourceError, staged_gate_source
    from .gate_start import GateStartError, reserve_gate_start
    from .gate_trust import GateTrustError, read_trusted_gate_request
    from .worker import DockerAPI

    api = api or DockerAPI(_DOCKER_TIMEOUT)
    state, exit_code, output, error = JOB_LOST, None, None, "gate_driver_uncertain"
    try:
        request = read_trusted_gate_request(store.get_job(job_id))
        with reserve_gate_launch(store, job_id, receipt_root=config.receipt_root) as custody:
            nonce = custody.owner_nonce
        with reserve_gate_start(store, job_id, receipt_root=config.receipt_root):
            pass  # Short ledger boundary only; never the workload.
        store.mark_running(job_id)
        with staged_gate_source(request.plan, repository=config.repository,
                                staging_parent=config.scratch_parent) as source:
            result = run_gate(api, plan=request.plan, source=source, job_id=job_id,
                              scratch_parent=config.scratch_parent,
                              environment_json=request.environment_json,
                              environment_sha256=request.environment_sha256, owner_nonce=nonce)
        exit_code, output = result.exit_code, _tail(result.output)
        state, error = (JOB_SUCCEEDED, None) if result.exit_code == 0 else (JOB_FAILED, "gate_tests_failed")
    except GateLaunchError as exc:
        # Uncertain exit/cleanup is not a failure verdict: the runner may live.
        uncertain = exc.code in {"cleanup_uncertain", "runner_exit_uncertain"} or not exc.cleanup_confirmed
        state, error = (JOB_LOST if uncertain else JOB_FAILED), f"gate_{exc.code}"
    except (GateCustodyError, GateStartError, GateSourceError, GateTrustError) as exc:
        state, error = JOB_FAILED, f"gate_refused: {exc}"[:200]
    except Exception:  # noqa: BLE001 - never echo unbounded internals
        logger.exception("gate %s driver failed", job_id)
    finally:
        try:
            store.mark_terminal(job_id, state=state, exit_code=exit_code, output_tail=output, error_text=error)
        except Exception:  # noqa: BLE001
            logger.exception("gate %s terminal write failed", job_id)
        try:
            retire_gate_slot(store, job_id, receipt_root=config.receipt_root, api=api)
        except GateRetireError as exc:
            logger.warning("gate %s slot stays held: %s", job_id, exc)
        except Exception:  # noqa: BLE001
            logger.exception("gate %s retirement uncertain", job_id)
    return state


class GateDriver:
    """Reconciler-only: claim queued gate rows and own their child processes."""

    def __init__(self, store, config: GateDriverConfig, *, spawn=None, api_factory=None, clock=time.monotonic):
        self.store = store
        self.config = config
        self._spawn = spawn or self._spawn_child
        self._api_factory = api_factory
        self._children: dict[str, object] = {}
        self._lock = threading.Lock()
        self._clock = clock

    def _spawn_child(self, job_id: str):
        return subprocess.Popen([sys.executable, "-m", "llm_bawt.ops.gate_driver", job_id],
                                stdin=subprocess.DEVNULL, close_fds=True)

    def _alive(self, job_id: str) -> bool:
        with self._lock:
            child = self._children.get(job_id)
            if child is None:
                return False
            if child.poll() is None:
                return True
            self._children.pop(job_id, None)
            return False

    def dispatch(self, job) -> None:
        from .gate_trusted_claim import claim_trusted_gate_job

        if job.state != JOB_QUEUED or self._alive(job.id):
            return
        if not claim_trusted_gate_job(self.store, job.id):
            # Capacity, grant, config or a prior attempt: keep QUEUED with a
            # visible reason; the existing preflight deadline fails it.
            self.store.note_queued_error(job.id, "gate_claim_refused")
            self.store.fail_stale_queued(job.id, deadline_seconds=_CLAIM_DEADLINE_SECONDS)
            return
        with self._lock:
            self._children[job.id] = self._spawn(job.id)

    def reconcile(self, job) -> None:
        """An active claimed gate row with no live child here is orphaned."""
        if job.state in JOB_TERMINAL_STATES or job.state == JOB_QUEUED or self._alive(job.id):
            return
        stamp = job.dispatched_at
        if stamp is not None and job.state == JOB_DISPATCHING:
            from datetime import datetime, timezone
            age = (datetime.now(timezone.utc) - stamp.replace(tzinfo=timezone.utc)).total_seconds()
            if age < _ORPHAN_GRACE_SECONDS:
                return
        self.store.mark_terminal(job.id, state=JOB_LOST,
            error_text="gate_driver_orphaned: no live attempt owner; not adopted or retried")
        self.retire(job.id)

    def retire(self, job_id: str) -> bool:
        from .gate_retire import GateRetireError, retire_gate_slot
        from .worker import DockerAPI

        api = self._api_factory() if self._api_factory else DockerAPI(_DOCKER_TIMEOUT)
        try:
            retire_gate_slot(self.store, job_id, receipt_root=self.config.receipt_root, api=api)
            return True
        except GateRetireError:
            return False

    def reap(self) -> int:
        """Poll every owned child so exited ones never linger as zombies."""
        with self._lock:
            done = [job_id for job_id, child in self._children.items() if child.poll() is not None]
            for job_id in done:
                self._children.pop(job_id, None)
        return len(done)

    def retire_pending(self, limit: int = 20) -> int:
        """Terminal reserved gates still holding capacity; retire once stopped."""
        self.reap()
        with self.store.engine.connect() as conn:
            ids = conn.execute(text("SELECT id FROM ops_jobs WHERE worker_slot_at IS NOT NULL "
                "AND gate_retired_at IS NULL AND host_unit_name LIKE 'gate-custody:%' "
                "AND state IN :states ORDER BY finished_at LIMIT :limit")
                .bindparams(bindparam("states", expanding=True)),
                {"states": sorted(JOB_TERMINAL_STATES), "limit": limit}).scalars().all()
        return sum(1 for job_id in ids if not self._alive(job_id) and self.retire(job_id))


def build_gate_driver(store) -> GateDriver | None:
    config = GateDriverConfig.from_env()
    return GateDriver(store, config) if config is not None else None


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    logging.basicConfig(level=logging.INFO)
    if len(argv) != 1:
        return 2
    from ..utils.config import Config
    from .store import OpsStore

    config = GateDriverConfig.from_env()
    store = OpsStore(Config())
    if config is None or store.engine is None:
        return 2
    state = run_child(store, argv[0], config)
    return 0 if state == JOB_SUCCEEDED else 1


if __name__ == "__main__":
    sys.exit(main())
