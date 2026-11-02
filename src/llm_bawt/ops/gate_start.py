"""Dormant gate capacity reservation and bounded pre-start ledger guard.

No caller, launch, running/result transition, recovery or retirement lives here.
This is NOT a complete execution permit or lifetime capacity fence. A surviving
DB session excludes concurrent terminal writes only within the yielded boundary;
DB loss cannot stop caller code. A reserved gate keeps counting in any state
until gate_retire records positive retirement; that is not a DB-loss fence.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from . import gate_custody as custody
from . import gate_trust as trust
from .models import JOB_DISPATCHING, OpsJob, OpsOperation, _iso_utc
from .store import slot_cap_reached, slot_holders
from .validation import canonical_json
from .worker_slots import _lock_worker_slots, live_worker_slots, process_spawn_held, worker_concurrency

_START = "gate-start.json"
_SCHEMA = "llm-bawt.ops.gate-start/v1"


class GateStartError(RuntimeError):
    """Fixed refusals, never underlying filesystem/DB error text."""


@dataclass(frozen=True)
class GateStartBoundary:
    """Identity valid only inside the surviving context; not a launch permit."""

    job_id: str
    owner_nonce: str
    start_sha256: str


def _bound(job, directory):
    request = custody._request(job)
    if (job.state != JOB_DISPATCHING or job.started_at is not None
            or job.finished_at is not None or job.exit_code is not None
            or not isinstance(job.host_unit_name, str)
            or not job.host_unit_name.startswith(custody._MARKER)
            or job.status_file_path != str(directory / custody._INTENT)
            or any(value is not None for value in
                   (job.process_spawn_hold_at, job.process_spawn_hold_target_json))):
        raise GateStartError("gate_start_claim_invalid")
    nonce = custody._read_intent(directory, job, request)
    return nonce


def _unreserved(job, directory):
    nonce = _bound(job, directory)
    if job.worker_slot_at is not None or job.log_file_path is not None:
        raise GateStartError("gate_start_already_reserved")
    return nonce


def _record(job, nonce, stamp):
    return canonical_json({"schema": _SCHEMA, "job_id": job.id, "owner_nonce": nonce,
        "custody_sha256": job.host_unit_name.removeprefix(custody._MARKER),
        "dispatched_at": _iso_utc(job.dispatched_at), "worker_slot_at": _iso_utc(stamp)}).encode()


def _publish(directory, raw):
    # Exclusive creation intentionally leaves any partial/uncertain evidence in
    # place. An orphan never authorizes another attempt; there is no repair path.
    fd = os.open(directory / _START, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        os.fchmod(stream.fileno(), 0o600)
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _reserve(conn, job_id, directory):
    # Same global lock order as process claims: operation -> slots -> job.
    slug = conn.execute(text("SELECT operation_slug FROM ops_jobs WHERE id=:id"),
                        {"id": job_id}).scalar_one_or_none()
    row = conn.execute(text("SELECT * FROM ops_operations WHERE slug=:slug FOR UPDATE"),
                       {"slug": slug}).mappings().first()
    if row is None:
        raise GateStartError("gate_start_operation_unavailable")
    operation = OpsOperation(**row)
    if not operation.enabled or operation.soft_deleted_at is not None or operation.quiesce_enabled:
        raise GateStartError("gate_start_operation_unavailable")
    _lock_worker_slots(conn)
    row = conn.execute(text("SELECT * FROM ops_jobs WHERE id=:id FOR UPDATE"),
                       {"id": job_id}).mappings().first()
    job = OpsJob(**row) if row is not None else None
    nonce = _unreserved(job, directory)
    if job.operation_slug != operation.slug:
        raise GateStartError("gate_start_claim_invalid")
    # Gate sandboxes conservatively share the existing budget explicitly: they
    # are NOT mislabeled Docker process jobs to bypass needs_worker_slot's policy.
    if process_spawn_held(conn) or live_worker_slots(conn) >= worker_concurrency():
        raise GateStartError("gate_start_capacity_unavailable")
    original = trust._json(job.invocation_snapshot_json, trust._MAX_SNAPSHOT)["execution"]["max_concurrent"]
    limits = [cap for cap in (original, operation.max_concurrent) if cap is not None]
    if slot_cap_reached(slot_holders(conn, operation.slug, exclude_job_id=job_id),
                        min(limits) if limits else None):
        raise GateStartError("gate_start_capacity_unavailable")
    if {path.name for path in directory.iterdir()} != {"submission.lock", custody._INTENT}:
        raise GateStartError("gate_start_journal_occupied")
    stamp = conn.execute(text("SELECT clock_timestamp()")).scalar_one()
    raw = _record(job, nonce, stamp)
    result = conn.execute(text("UPDATE ops_jobs SET worker_slot_at=:stamp,log_file_path=:path "
        "WHERE id=:id AND state=:state AND worker_slot_at IS NULL AND log_file_path IS NULL"),
        {"id": job_id, "state": JOB_DISPATCHING, "stamp": stamp, "path": str(directory / _START)})
    if result.rowcount != 1:
        raise GateStartError("gate_start_reservation_conflict")
    _publish(directory, raw)
    return raw


def _guard(conn, job_id, directory, expected):
    # Phase two never acquires operation/slot locks after this job lock.
    row = conn.execute(text("SELECT * FROM ops_jobs WHERE id=:id FOR UPDATE"),
                       {"id": job_id}).mappings().first()
    job = OpsJob(**row) if row is not None else None
    nonce = _bound(job, directory)
    if (job.worker_slot_at is None or job.log_file_path != str(directory / _START)
            or _record(job, nonce, job.worker_slot_at) != expected
            or {path.name for path in directory.iterdir()} != {"submission.lock", custody._INTENT, _START}):
        raise GateStartError("gate_start_reservation_invalid")
    fd = os.open(directory / _START, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        custody._regular(stream.fileno())
        raw = stream.read(custody._BOUND + 1)
    if raw != expected:
        raise GateStartError("gate_start_reservation_invalid")
    return GateStartBoundary(job_id, nonce, hashlib.sha256(raw).hexdigest())


@contextmanager
def reserve_gate_start(store, job_id: str, *, receipt_root: Path):
    """Commit capacity/one-shot evidence, then lock/recheck before yielding.

    Call only after reserve_gate_launch's consumer scope has ended. The same
    submission flock stays held through both transactions and consumer scope.
    Phase one locks operation -> global slots -> fresh job, reserves capacity
    and publishes/fsyncs an exclusive marker before committing. No successful
    commit means no yield. A durable marker or ledger reservation prevents all
    reissues, including after orphan, journal loss, uncertain commit or terminal.

    Phase two re-locks the fresh job: a preceding terminal write denies, a later
    one waits only while this transaction/session survives. Caller scope must
    remain a short boundary, never the gate workload. No metadata/slot cleanup
    happens on return or exception, and no lifecycle transition is fabricated.
    Historic claim authorization is prospective; no grant/config cache or new
    execution authority is introduced. Runtime support, live start integration,
    lost-session/copied-identity fencing and positive retirement are still absent.
    """
    try:
        custody._store(store, job_id)
        root = custody._private(receipt_root)
        directory = custody._private(root / job_id)
        with custody._submission(directory, create=False):
            _unreserved(store.get_job(job_id), directory)
            with store.engine.begin() as conn:
                expected = _reserve(conn, job_id, directory)
            with store.engine.begin() as conn:
                boundary = _guard(conn, job_id, directory, expected)
                yield boundary
    except custody.GateCustodyError as exc:
        code = "gate_start_busy" if str(exc) == "gate_custody_busy" else "gate_start_custody_invalid"
        raise GateStartError(code) from None
    except OSError:
        raise GateStartError("gate_start_io_uncertain") from None
    except SQLAlchemyError:
        raise GateStartError("gate_start_ledger_uncertain") from None
