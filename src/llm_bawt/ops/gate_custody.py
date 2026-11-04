"""Dormant dispatch custody on the existing ops ledger/receipt directory.

A reservation is a one-shot identity, NOT worker capacity or a launch permit.
No production caller, scheduler, process spawn, runner call, retry, terminal
transition, cleanup, retirement, callback or trusted gate publication lives here.
The receipt root is trusted private storage, never a runner mount or user input.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from .gate_receipt import check_gate_callback
from .gate_runner import GateLaunchError, gate_container_labels, gate_container_names
from .gate_trust import read_trusted_gate_request
from .models import JOB_DISPATCHING, OpsJob, _iso_utc
from .validation import canonical_json

_SCHEMA = "llm-bawt.ops.gate-custody/v1"
_INTENT = "gate-intent.json"
_MARKER = "gate-custody:"
_BOUND = 4096


class GateCustodyError(RuntimeError):
    """Fixed refusals; never include caller/evidence/daemon/DB exception text."""


@dataclass(frozen=True)
class GateLaunchCustody:
    job_id: str
    owner_nonce: str
    intent_sha256: str


@dataclass(frozen=True)
class GateContainerObservation:
    status: str  # missing / created / running / exited; not a result or retirement.
    container_id: str | None = None


@dataclass(frozen=True)
class GateCustodyObservation:
    busy: bool
    postgres: GateContainerObservation | None = None
    runner: GateContainerObservation | None = None


def _store(store, job_id):
    store._require()
    if store.engine.dialect.name != "postgresql":
        raise GateCustodyError("gate_custody_postgresql_required")
    try:
        return gate_container_names(job_id)
    except GateLaunchError:
        raise GateCustodyError("gate_custody_job_invalid") from None


def _private(path):
    path = Path(path)
    if (not path.is_absolute() or path.resolve(strict=True) != path
            or not stat.S_ISDIR(path.lstat().st_mode)
            or path.stat().st_uid != os.geteuid()
            or stat.S_IMODE(path.stat().st_mode) != 0o700):
        raise GateCustodyError("gate_custody_private_directory_required")
    return path


def _regular(fd):
    info = os.fstat(fd)
    if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid()
            or info.st_nlink != 1 or stat.S_IMODE(info.st_mode) != 0o600):
        raise GateCustodyError("gate_custody_file_invalid")


@contextmanager
def _submission(directory, *, create):
    path = directory / "submission.lock"
    flags = os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK
    if create:
        try:
            fd = os.open(path, flags | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            fd = os.open(path, flags)
        else:
            os.fchmod(fd, 0o600)
    else:
        fd = os.open(path, flags)  # Recovery never creates even a lock file.
    try:
        _regular(fd)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise GateCustodyError("gate_custody_busy") from None
        yield
    finally:
        os.close(fd)


def _request(job):
    if job is None or job.dispatched_at is None or job.retry_count != 0:
        raise GateCustodyError("gate_custody_claim_required")
    # Do not reinterpret generic/legacy, approved or quiescing jobs, or foreign callbacks.
    if (job.approval_request_id is not None or job.quiesce_gate_id is not None
            or job.quiesce_fence_json is not None):
        raise GateCustodyError("gate_custody_claim_required")
    if job.receipt_callback_json is not None:
        # The signed admission's own direct callback (TASK-1865); anything else refuses.
        try:
            check_gate_callback(job, json.loads(job.receipt_callback_json))
        except (ValueError, TypeError):
            raise GateCustodyError("gate_custody_claim_required") from None
    try:
        return read_trusted_gate_request(job)
    except (ValueError, TypeError):
        raise GateCustodyError("gate_custody_request_invalid") from None


def _reservable(job):
    request = _request(job)
    if (job.state != JOB_DISPATCHING or any(value is not None for value in
            (job.host_unit_name, job.status_file_path, job.log_file_path,
             job.started_at, job.finished_at, job.exit_code))):
        raise GateCustodyError("gate_custody_already_reserved_or_closed")
    return request


def _record(job, request, nonce):
    if not isinstance(nonce, str) or not re.fullmatch(r"[0-9a-f]{32}", nonce):
        raise GateCustodyError("gate_custody_intent_invalid")
    return canonical_json({"schema": _SCHEMA, "job_id": job.id, "owner_nonce": nonce,
        "dispatched_at": _iso_utc(job.dispatched_at),
        "invocation_sha256": hashlib.sha256(job.invocation_snapshot_json.encode()).hexdigest(),
        "request_sha256": hashlib.sha256(job.request_payload_json.encode()).hexdigest(),
        "gate_plan": request.plan.to_dict(), "environment_sha256": request.environment_sha256,
        "containers": gate_container_names(job.id)}).encode()


def _publish(directory, raw):
    # Same temp + hard-link + fsync pattern as existing ops request publication,
    # but an existing intent NEVER authorizes another reservation, even identical.
    fd, name = tempfile.mkstemp(prefix=".gate-intent-", dir=directory)
    temp = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            os.fchmod(stream.fileno(), 0o600)
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temp, directory / _INTENT)
    finally:
        temp.unlink()
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@contextmanager
def reserve_gate_launch(store, job_id: str, *, receipt_root: Path):
    """Reserve once after the trusted claim; retain submission.lock through yield.

    A fresh authenticated DISPATCHING row is locked again before metadata CAS.
    Intent is fsynced before commit/yield. A crash/rollback may leave an orphan
    intent; uncertain commit or later journal loss leaves durable row metadata.
    Either blocks a second reservation. Nothing rolls back/deletes that evidence.
    Requires claim_trusted_gate_job, not generic claim_job (which prepopulates
    incompatible host metadata). Callers still owe worker capacity, supported
    runtime, terminal/start fencing and trusted execution integration. A copied
    nonce can outlive this local lock; this unused identity is NOT that permit.
    """
    _store(store, job_id)
    try:
        _reservable(store.get_job(job_id))  # Reject before creating local state.
        root = _private(receipt_root)
        directory = root / job_id
        directory.mkdir(mode=0o700, exist_ok=True)
        _private(directory)
        with _submission(directory, create=True):
            # Existing partial intent/temp/ordinary worker evidence is never adopted.
            if {path.name for path in directory.iterdir()} != {"submission.lock"}:
                raise GateCustodyError("gate_custody_journal_occupied")
            with store.engine.begin() as conn:
                row = conn.execute(text("SELECT * FROM ops_jobs WHERE id=:id FOR UPDATE"),
                    {"id": job_id}).mappings().first()
                job = OpsJob(**row) if row is not None else None
                request = _reservable(job)
                nonce = os.urandom(16).hex()
                raw = _record(job, request, nonce)
                result = conn.execute(text("UPDATE ops_jobs SET host_unit_name=:name,status_file_path=:path "
                    "WHERE id=:id AND state=:state AND host_unit_name IS NULL AND status_file_path IS NULL"),
                    {"id": job_id, "state": JOB_DISPATCHING,
                     "name": _MARKER + hashlib.sha256(raw).hexdigest(),
                     "path": str(directory / _INTENT)})
                if result.rowcount != 1:
                    raise GateCustodyError("gate_custody_reservation_conflict")
                _publish(directory, raw)
                # Metadata only: claim already emitted the lifecycle transition.
            custody = GateLaunchCustody(job_id, nonce, hashlib.sha256(raw).hexdigest())
            yield custody  # Only after successful DB commit, with local lock held.
    except OSError:
        raise GateCustodyError("gate_custody_io_uncertain") from None
    except SQLAlchemyError:
        raise GateCustodyError("gate_custody_ledger_uncertain") from None


def _read_intent(directory, job, request):
    fd = os.open(directory / _INTENT, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        _regular(stream.fileno())
        raw = stream.read(_BOUND + 1)
    if (len(raw) > _BOUND
            or job.host_unit_name != _MARKER + hashlib.sha256(raw).hexdigest()):
        raise GateCustodyError("gate_custody_intent_invalid")
    try:
        value = json.loads(raw)
        nonce = value["owner_nonce"]
        expected = _record(job, request, nonce)
    except (ValueError, TypeError, KeyError, UnicodeError, RecursionError):
        raise GateCustodyError("gate_custody_intent_invalid") from None
    if raw != expected:
        raise GateCustodyError("gate_custody_intent_invalid")
    return nonce


def _observe(api, name, labels):
    try:
        state = api.request("GET", f"/containers/{name}/json")
    except RuntimeError as exc:
        if str(exc).startswith("Docker HTTP 404:"):
            return GateContainerObservation("missing")
        raise GateCustodyError("gate_custody_inspection_uncertain") from None
    except Exception:
        raise GateCustodyError("gate_custody_inspection_uncertain") from None
    if not isinstance(state, dict):
        raise GateCustodyError("gate_custody_identity_uncertain")
    identity, config, status = state.get("Id"), state.get("Config"), state.get("State")
    if (not isinstance(identity, str) or not re.fullmatch(r"[0-9a-f]{64}", identity)
            or state.get("Name") not in (name, "/" + name)
            or not isinstance(config, dict) or not isinstance(config.get("Labels"), dict)
            or any(config["Labels"].get(key) != value for key, value in labels.items())
            or not isinstance(status, dict)):
        raise GateCustodyError("gate_custody_identity_uncertain")
    phase = status.get("Status")
    if (not isinstance(phase, str) or phase not in {"created", "running", "exited"}
            or status.get("Running") is not (phase == "running")):
        raise GateCustodyError("gate_custody_identity_uncertain")
    return GateContainerObservation(phase, identity)


def observe_gate_launch(store, job_id: str, *, receipt_root: Path, api) -> GateCustodyObservation:
    """Nonblocking, read-only recovery observation; never a retirement or retry.

    Authenticate the ledger/request/claim-time and exact retained intent before
    GET-only inspection of both role names and nonce-bound labels. Even two 404s
    mean only observations at those instants: no replay, removal, terminal/SHIP
    result or durable positive retirement follows. A terminal ledger row does not
    make an old container safe to adopt. No directories/files/metadata are written.
    """
    names = _store(store, job_id)
    try:
        root = _private(receipt_root)
        directory = _private(root / job_id)
        with _submission(directory, create=False):
            job = store.get_job(job_id)
            request = _request(job)
            if (not isinstance(job.host_unit_name, str)
                    or not re.fullmatch(r"gate-custody:[0-9a-f]{64}", job.host_unit_name)
                    or job.status_file_path != str(directory / _INTENT)):
                raise GateCustodyError("gate_custody_reservation_missing")
            nonce = _read_intent(directory, job, request)
            labels = gate_container_labels(request.plan, job_id, nonce, request.environment_sha256)
            observed = {role: _observe(api, name,
                {**labels, "com.docker.compose.service": f"gate-{role}"}) for role, name in names.items()}
            return GateCustodyObservation(False, observed["pg"], observed["runner"])
    except GateCustodyError as exc:
        if str(exc) == "gate_custody_busy":
            return GateCustodyObservation(True)
        raise
    except OSError:
        raise GateCustodyError("gate_custody_io_uncertain") from None
    except SQLAlchemyError:
        raise GateCustodyError("gate_custody_ledger_uncertain") from None
