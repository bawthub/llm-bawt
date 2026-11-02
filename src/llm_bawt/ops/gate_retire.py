"""Dormant positive retirement of a slot-reserved gate's shared capacity.

A gate that reserved a worker slot keeps counting against the shared and
operation budgets in ANY ledger state until this stamps ``gate_retired_at``.
A terminal row (lost/timed_out included) never proves its containers stopped.

Retirement needs, under the same submission flock that prevents reissue:
a terminal row with a durable start reservation, and a GET-only enumeration of
every container carrying the exact job+nonce labels (labels are immutable, so a
rename cannot hide one) where each match, re-inspected by immutable ID, is
exited with explicit RestartPolicy "no" -- or there are no matches at all.
Anything else refuses and the slot stays held. No removal, retry, result,
callback, launch or lifecycle transition happens here; no production caller.

Limits retained: this cannot fence a late creator that outlived DB loss with a
copied nonce, nor a manual restart of a retained exited container.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
from urllib.parse import quote, urlencode

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from . import gate_custody as custody
from . import gate_start as start
from .gate_runner import gate_container_labels
from .models import JOB_TERMINAL_STATES, OpsJob

_ID = re.compile(r"[0-9a-f]{64}")
_MAX_MATCHES = 8


class GateRetireError(RuntimeError):
    """Fixed refusals; never daemon/DB/filesystem exception text."""


def _get(api, path):
    try:
        return api.request("GET", path)
    except Exception:
        raise GateRetireError("gate_retire_inspection_uncertain") from None


def _stopped(api, labels):
    filters = {"label": [f"{key}={value}" for key, value in sorted(labels.items())]}
    rows = _get(api, "/containers/json?" + urlencode({"all": "1", "filters": json.dumps(filters)}))
    if not isinstance(rows, list) or len(rows) > _MAX_MATCHES:
        raise GateRetireError("gate_retire_inspection_uncertain")
    for row in rows:
        identity = row.get("Id") if isinstance(row, dict) else None
        if not isinstance(identity, str) or not _ID.fullmatch(identity):
            raise GateRetireError("gate_retire_identity_uncertain")
        state = _get(api, f"/containers/{quote(identity, safe='')}/json")
        config = state.get("Config") if isinstance(state, dict) else None
        host = state.get("HostConfig") if isinstance(state, dict) else None
        status = state.get("State") if isinstance(state, dict) else None
        found = config.get("Labels") if isinstance(config, dict) else None
        policy = host.get("RestartPolicy") if isinstance(host, dict) else None
        if (not isinstance(state, dict) or state.get("Id") != identity or not isinstance(found, dict)
                or any(found.get(key) != value for key, value in labels.items())):
            raise GateRetireError("gate_retire_identity_uncertain")
        if (not isinstance(status, dict) or status.get("Status") != "exited"
                or status.get("Running") is not False or status.get("Restarting") is not False
                or not isinstance(policy, dict) or policy.get("Name") != "no"):
            raise GateRetireError("gate_retire_not_stopped")


def _eligible(job, directory):
    if (job is None or job.state not in JOB_TERMINAL_STATES or job.worker_slot_at is None
            or job.gate_retired_at is not None or job.log_file_path != str(directory / start._START)
            or not isinstance(job.host_unit_name, str)
            or not job.host_unit_name.startswith(custody._MARKER)
            or job.status_file_path != str(directory / custody._INTENT)):
        raise GateRetireError("gate_retire_ineligible")


def retire_gate_slot(store, job_id: str, *, receipt_root: Path, api) -> None:
    """Stamp positive retirement once, or raise a fixed refusal leaving the slot held."""
    try:
        custody._store(store, job_id)
        directory = custody._private(custody._private(receipt_root) / job_id)
        with custody._submission(directory, create=False):
            job = store.get_job(job_id)
            _eligible(job, directory)
            request = custody._request(job)
            nonce = custody._read_intent(directory, job, request)
            _stopped(api, gate_container_labels(request.plan, job_id, nonce, request.environment_sha256))
            with store.engine.begin() as conn:
                row = conn.execute(text("SELECT * FROM ops_jobs WHERE id=:id FOR UPDATE"),
                                   {"id": job_id}).mappings().first()
                fresh = OpsJob(**row) if row is not None else None
                _eligible(fresh, directory)
                if (fresh.worker_slot_at != job.worker_slot_at
                        or fresh.host_unit_name != job.host_unit_name):
                    raise GateRetireError("gate_retire_ineligible")
                result = conn.execute(text(
                    "UPDATE ops_jobs SET gate_retired_at=clock_timestamp() "
                    "WHERE id=:id AND gate_retired_at IS NULL AND worker_slot_at IS NOT NULL"),
                    {"id": job_id})
                if result.rowcount != 1:
                    raise GateRetireError("gate_retire_conflict")
    except custody.GateCustodyError as exc:
        code = "gate_retire_busy" if str(exc) == "gate_custody_busy" else "gate_retire_custody_invalid"
        raise GateRetireError(code) from None
    except OSError:
        raise GateRetireError("gate_retire_io_uncertain") from None
    except SQLAlchemyError:
        raise GateRetireError("gate_retire_ledger_uncertain") from None
