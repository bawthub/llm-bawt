"""Positive gate retirement on disposable PG/files only; fake GET-only Docker."""
from __future__ import annotations

import json
from urllib.parse import parse_qs, urlsplit

import pytest

from llm_bawt.ops import gate_retire as retire
from llm_bawt.ops.gate_runner import gate_container_labels
from llm_bawt.ops.gate_trust import read_trusted_gate_request
from llm_bawt.ops.store import slot_holders
from llm_bawt.ops.worker_slots import live_worker_slots
from tests import test_ops_gate_start_pg as starts

rig = starts.rig
configured = starts.configured
claimed = starts.claimed
ready = starts.ready
pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]

ID = "a" * 64


def labels(c):
    raw = json.loads((starts.directory(c) / "gate-intent.json").read_bytes())
    request = read_trusted_gate_request(c.rig.store.get_job(c.job.id))
    return gate_container_labels(request.plan, c.job.id, raw["owner_nonce"], request.environment_sha256)


class Docker:
    """GET-only: label-filtered list plus inspect by immutable ID."""

    def __init__(self, inspected=None, listed=None):
        self.inspected, self.listed, self.calls = inspected or {}, listed, []

    def request(self, method, path):
        assert method == "GET"
        self.calls.append(path)
        if path.startswith("/containers/json?"):
            query = parse_qs(urlsplit(path).query)
            self.filters = json.loads(query["filters"][0])
            assert query["all"] == ["1"]
            return self.listed if self.listed is not None else [{"Id": k} for k in self.inspected]
        key = path.split("/")[2]
        if key not in self.inspected:
            raise RuntimeError("Docker HTTP 404: fixture missing")
        return self.inspected[key]


def container(c, *, status="exited", running=False, restarting=False, policy="no", label_override=None):
    found = {**labels(c), **(label_override or {})}
    return {"Id": ID, "Config": {"Labels": found}, "State": {"Status": status, "Running": running,
        "Restarting": restarting}, "HostConfig": {"RestartPolicy": {"Name": policy}}}


@pytest.fixture
def terminal(ready):
    with starts.reserve(ready):
        pass
    ready.rig.store.mark_terminal(ready.job.id, state="lost")
    return ready


def run(c, api):
    return retire.retire_gate_slot(c.rig.store, c.job.id, receipt_root=c.root, api=api)


def held(c):
    with c.rig.engine.begin() as conn:
        live = live_worker_slots(conn)
        holders = len(slot_holders(conn, c.job.operation_slug))
    return live, holders


def test_terminal_gate_keeps_global_and_operation_slot_until_retired(terminal):
    c = terminal
    assert held(c) == (1, 1)  # lost row alone proves nothing; counted exactly once.
    peer = starts.process_job(c)
    assert not c.rig.other.claim_job(peer.id)
    api = Docker({ID: container(c)})
    run(c, api)
    assert c.rig.store.get_job(c.job.id).gate_retired_at is not None
    assert held(c) == (0, 0)
    assert all(path.startswith("/containers/json?") or path == f"/containers/{ID}/json" for path in api.calls)
    assert sorted(api.filters["label"]) == sorted(f"{k}={v}" for k, v in labels(c).items())
    with pytest.raises(retire.GateRetireError, match="ineligible"):
        run(c, api)  # One-shot stamp; no rewrite.


def test_no_matching_containers_is_positive_absence(terminal):
    run(terminal, Docker(listed=[]))
    assert held(terminal) == (0, 0)


def test_active_gate_counts_once_and_cannot_retire(ready):
    c = ready
    with starts.reserve(c):
        assert held(c) == (1, 1)
        with pytest.raises(retire.GateRetireError):
            run(c, Docker(listed=[]))
    with pytest.raises(retire.GateRetireError, match="ineligible"):
        run(c, Docker(listed=[]))  # dispatching, not terminal.
    assert held(c) == (1, 1)


@pytest.mark.parametrize("change", ["running", "created", "restarting", "always", "unless-stopped",
                                    "no-policy", "label", "id", "inspect-404", "list-error", "list-shape"])
def test_unproven_stop_retains_slot(terminal, change):
    c = terminal
    if change == "list-error":
        api = Docker()
        api.request = lambda method, path: (_ for _ in ()).throw(RuntimeError("daemon secret"))
    elif change == "list-shape":
        api = Docker(listed={"Id": ID})
    elif change == "inspect-404":
        api = Docker(listed=[{"Id": ID}])
    else:
        body = {"running": lambda: container(c, status="running", running=True),
            "created": lambda: container(c, status="created"),
            "restarting": lambda: container(c, restarting=True),
            "always": lambda: container(c, policy="always"),
            "unless-stopped": lambda: container(c, policy="unless-stopped"),
            "no-policy": lambda: {**container(c), "HostConfig": {}},
            "label": lambda: container(c, label_override={"llm-bawt.ops.gate.owner": "0" * 32}),
            "id": lambda: {**container(c), "Id": "b" * 64}}[change]()
        api = Docker({ID: body})
    with pytest.raises(retire.GateRetireError) as error:
        run(c, api)
    assert "secret" not in str(error.value)
    assert c.rig.store.get_job(c.job.id).gate_retired_at is None
    assert held(c) == (1, 1)


def test_terminal_non_gate_rows_stay_excluded(ready):
    c = ready
    before = held(c)  # The unreserved dispatching gate itself is an ordinary holder.
    peer = starts.process_job(c)
    c.rig.sql("UPDATE ops_jobs SET worker_slot_at=now(),state='lost' WHERE id=:id", id=peer.id)
    assert held(c) == before == (0, 1)


def test_unreserved_terminal_gate_is_ineligible_and_not_counted(ready):
    c = ready
    c.rig.store.mark_terminal(c.job.id, state="cancelled")
    assert held(c) == (0, 0)
    with pytest.raises(retire.GateRetireError, match="ineligible"):
        run(c, Docker(listed=[]))


def test_busy_submission_lock_refuses(terminal):
    c = terminal
    with (starts.helpers.custody._submission(starts.directory(c), create=False),
          pytest.raises(retire.GateRetireError, match="^gate_retire_busy$")):
        run(c, Docker(listed=[]))
    assert held(c) == (1, 1)
