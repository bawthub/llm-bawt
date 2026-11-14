"""TASK-2231: regressions for the llm-bawt ops audit fixes."""
from __future__ import annotations

import errno
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from llm_bawt.ops import catalog_drift, control_client, control_http, process_launch as pl, sequence
from llm_bawt.ops.control_protocol import CLIENT_TIMEOUT_SECONDS, SERVER_BUDGET_SECONDS
from llm_bawt.ops.dev_config_reconcile import _MEMORY_TEXT, memory_text
from llm_bawt.ops.restart_health import RestartUnhealthy, wait_healthy


# -- dev config: a lone mem_limit -------------------------------------------

def test_a_lone_memory_cap_formats_to_text_its_own_validator_accepts():
    text = memory_text({"Memory": 536870912, "MemorySwap": 0})
    assert text == "536870912 bytes, swap Docker default"
    assert _MEMORY_TEXT.fullmatch(text)
    for other in ({"Memory": 0, "MemorySwap": 0}, {"Memory": 5, "MemorySwap": -1}, {"Memory": 5, "MemorySwap": 9}):
        assert _MEMORY_TEXT.fullmatch(memory_text(other))
    assert not _MEMORY_TEXT.fullmatch("5 bytes, swap Docker")
    assert not _MEMORY_TEXT.fullmatch("5 bytes, swap docker default")


# -- control client vs server budget ----------------------------------------

def test_the_control_client_waits_longer_than_the_server_lets_work_run():
    assert control_http.DEFAULT_BUDGET_SECONDS == SERVER_BUDGET_SECONDS
    assert control_client.DEFAULT_TIMEOUT_SECONDS == CLIENT_TIMEOUT_SECONDS
    # Room for a slow success or the server's coded 504 to arrive, but bounded.
    assert SERVER_BUDGET_SECONDS < CLIENT_TIMEOUT_SECONDS <= SERVER_BUDGET_SECONDS + 10


# -- restart health: a transient inspect failure -----------------------------

class _Api:
    def __init__(self, failures):
        self.failures, self.gets = failures, 0

    def request(self, method, path, body=None, raw=False):
        if method == "GET" and path.endswith("/json") and path.startswith("/containers/"):
            self.gets += 1
            if self.gets <= self.failures:
                raise RuntimeError("Docker HTTP 500")
            return {"State": {"Running": True}}
        if path.endswith("/exec"):
            return {"Id": "e1"}
        if path.endswith("/start"):
            body = json.dumps({"status": "ok"}).encode()
            return b"\x01\x00\x00\x00" + len(body).to_bytes(4, "big") + body
        return {"ExitCode": 0}


_SPEC = {"health_timeout_seconds": 30, "health_url": "http://127.0.0.1/health", "health_field": "status",
         "health_value": "ok", "action": "restart"}


class _Clock:
    def __init__(self):
        self.t = 0.0
        self.sleeps = 0

    def __call__(self):
        return self.t

    def sleep(self, seconds):
        self.sleeps += 1
        self.t += seconds


def test_a_transient_inspect_failure_is_retried_until_healthy():
    api, clock = _Api(failures=2), _Clock()
    assert wait_healthy(api, "c1", _SPEC, clock=clock, sleep=clock.sleep) == "health status matched"
    assert api.gets == 3 and clock.sleeps == 2


def test_inspect_failing_past_the_deadline_is_a_known_unhealthy_restart():
    api, clock = _Api(failures=10_000), _Clock()
    with pytest.raises(RestartUnhealthy) as caught:
        wait_healthy(api, "c1", _SPEC, clock=clock, sleep=clock.sleep)
    assert caught.value.outcome_known is True
    assert "container inspect failed" in str(caught.value)
    assert api.gets > 1


# -- sequence: a success checkpoint that fails -------------------------------

def _snapshot(n=2):
    return {"sequence": {"steps": [
        {"snapshot": {"operation": {"slug": f"op-{i}"}, "execution": {"start_delay_seconds": 0}},
         "display_args": {}, "target": {"name": f"t{i}"}} for i in range(1, n + 1)]}}


def test_a_failed_success_checkpoint_keeps_the_step_result_and_stops():
    calls, executed = [], []

    def checkpoint(ledger):
        calls.append(json.loads(json.dumps(ledger)))
        if len(calls) == 3:  # initial, step-1 intent, step-1 success <- fails
            raise OSError(errno.ENOSPC, "no space")

    def execute(item):
        executed.append(item["target"]["name"])
        return "done"

    ledger = sequence.run(_snapshot(), api=SimpleNamespace(side_effect_started=True), preflight=lambda _i: None,
                          execute=execute, checkpoint=checkpoint, sleep=lambda _s: None)
    first, second = ledger["steps"]
    assert executed == ["t1"]  # never replayed, never continued
    assert (first["state"], first["exit"], first["health"]) == ("succeeded", 0, {"state": "healthy"})
    assert "checkpoint failed" in first["reason"] and "OSError" in first["reason"]
    assert second["state"] == "skipped"
    assert ledger["state"] == "failed"
    assert calls[-1] == ledger and len(calls) == 4
    sequence.read_ledger(ledger)  # still a valid ledger


def test_a_failed_step_is_still_reported_failed():
    def execute(item):
        raise RestartUnhealthy("not healthy")

    ledger = sequence.run(_snapshot(1), api=SimpleNamespace(side_effect_started=True), preflight=lambda _i: None,
                          execute=execute, checkpoint=lambda _l: None, sleep=lambda _s: None)
    step = ledger["steps"][0]
    assert (step["state"], step["exit"], step["health"]) == ("failed", 1, {"state": "failed"})


# -- process launch: diagnostics that cannot be written ----------------------

def test_an_exited_child_is_finalized_even_when_its_output_cannot_be_written(tmp_path, monkeypatch):
    popen = subprocess.Popen([sys.executable, "-c", "print('x')"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    child = pl._Child(popen=popen, attempt=1)
    launcher = pl.ProcessLauncher(client=lambda: None, boot_id=lambda: "boot")
    launcher._children["job"] = child

    def full_disk(*_a, **_k):
        raise OSError(errno.ENOSPC, "no space")
    monkeypatch.setattr(pl, "atomic_json", full_disk)
    with pytest.raises(OSError):
        launcher._supervise("job", child, tmp_path, "digest", 30, 1024)
    assert child.done.is_set()
    assert launcher.child_stats() == (0, None)


# -- catalog drift: whitespace before stored JSON ----------------------------

def _drift(seed_script, live_script):
    seed = {"slug": "x", "enabled": True, "command_script": seed_script}
    live = {**seed, "command_script": live_script, "updated_by": "someone", "version": 1}
    return next(e for e in catalog_drift.compare([seed], [live])["operations"] if e["slug"] == "x")


def test_leading_whitespace_before_a_json_script_is_not_drift():
    script = json.dumps({"action": "restart", "health": {"timeout": 5}})
    assert _drift(script, "\n  " + script)["status"] == "clean"


def test_a_seeded_nested_key_missing_behind_whitespace_is_reported_by_path():
    seed = json.dumps({"action": "restart", "health": {"timeout": 5, "field": "status"}})
    live = " \n" + json.dumps({"action": "restart", "health": {"timeout": 5}})
    entry = _drift(seed, live)
    assert entry["missing_paths"] == ["command_script.health.field"]
