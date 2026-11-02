"""Custody observations are bounded GET-only data, never a launch or result."""
from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from llm_bawt.ops import gate_custody as custody
from llm_bawt.ops.gate_plan import GatePlan
from llm_bawt.ops.gate_runner import gate_container_labels, gate_container_names

JOB = "1" * 32
NONCE = "2" * 32
PLAN = GatePlan("llm-bawt", "3" * 40)
NAME = gate_container_names(JOB)["runner"]
LABELS = {**gate_container_labels(PLAN, JOB, NONCE, "4" * 64),
    "com.docker.compose.service": "gate-runner"}


class InspectOnly:
    def __init__(self, response=None, error=None):
        self.response, self.error, self.calls = response, error, []

    def request(self, method, path):
        assert method == "GET" and path == f"/containers/{NAME}/json"
        self.calls.append((method, path))
        if self.error is not None:
            raise self.error
        return copy.deepcopy(self.response)


def state(phase="running"):
    return {"Id": "5" * 64, "Name": "/" + NAME, "Config": {"Labels": LABELS.copy()},
        "State": {"Status": phase, "Running": phase == "running", "ExitCode": 0}}


@pytest.mark.parametrize("phase", ["created", "running", "exited"])
def test_exact_owned_state_is_only_an_observation(phase):
    api = InspectOnly(state(phase))
    result = custody._observe(api, NAME, LABELS)
    assert result == custody.GateContainerObservation(phase, "5" * 64)
    assert api.calls == [("GET", f"/containers/{NAME}/json")]
    assert not hasattr(result, "cleanup_confirmed") and not hasattr(result, "exit_code")


@pytest.mark.parametrize("error,missing", [
    (RuntimeError("Docker HTTP 404: missing"), True),
    (RuntimeError("Docker HTTP 500: daemon text mentions HTTP 404"), False),
    (OSError("fixture sensitive transport detail"), False)])
def test_only_structured_404_is_missing_and_no_error_content_escapes(error, missing):
    api = InspectOnly(error=error)
    if missing:
        assert custody._observe(api, NAME, LABELS) == custody.GateContainerObservation("missing")
    else:
        with pytest.raises(custody.GateCustodyError, match="^gate_custody_inspection_uncertain$"):
            custody._observe(api, NAME, LABELS)
    assert len(api.calls) == 1


@pytest.mark.parametrize("mutation", ["id", "name", "config", "labels", "owner", "job", "sha",
    "environment", "role", "state", "phase", "phase-type", "running", "bool"])
def test_uncertain_or_foreign_identity_is_never_adopted(mutation):
    data = state()
    if mutation == "id":
        data["Id"] = "fixture-short"
    elif mutation == "name":
        data["Name"] = NAME + "-other"
    elif mutation == "config":
        data["Config"] = []
    elif mutation == "labels":
        data["Config"]["Labels"] = None
    elif mutation in {"owner", "job", "sha", "environment", "role"}:
        key = {"owner": "llm-bawt.ops.gate.owner", "job": "llm-bawt.ops.job",
            "sha": "llm-bawt.ops.gate.sha", "environment": "llm-bawt.ops.gate.environment",
            "role": "com.docker.compose.service"}[mutation]
        data["Config"]["Labels"][key] = "another-invocation"
    elif mutation == "state":
        data["State"] = None
    elif mutation == "phase":
        data["State"]["Status"] = "restarting"
    elif mutation == "phase-type":
        data["State"]["Status"] = []
    elif mutation == "running":
        data["State"]["Running"] = False
    else:
        data["State"]["Running"] = 1
    with pytest.raises(custody.GateCustodyError, match="gate_custody_identity_uncertain"):
        custody._observe(InspectOnly(data), NAME, LABELS)


def test_nonpostgresql_refuses_before_files_or_transactions(tmp_path):
    store = SimpleNamespace(_require=lambda: None,
        engine=SimpleNamespace(dialect=SimpleNamespace(name="sqlite")))
    root = tmp_path / "never-created"
    with pytest.raises(custody.GateCustodyError, match="gate_custody_postgresql_required"):
        with custody.reserve_gate_launch(store, JOB, receipt_root=root):
            pytest.fail("non-PG reservation yielded")
    with pytest.raises(custody.GateCustodyError, match="gate_custody_postgresql_required"):
        custody.observe_gate_launch(store, JOB, receipt_root=root, api=InspectOnly())
    assert not root.exists()
