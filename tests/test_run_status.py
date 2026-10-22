"""TASK-1232: the ``run.status`` reply contract and the queue-derived state."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent_bridge.abort_contract import process_boot_id
from agent_bridge.run_status import (
    RunState,
    RunStatusReply,
    RunStatusReplyError,
    answer_run_status,
    queue_run_state,
)
from agent_bridge.session_queue import RequestPhase

PARAMS = {"sessionKey": "bot:user", "requestId": "r1"}


def test_answer_is_the_v1_shape_for_this_process():
    reply = answer_run_status("codex", PARAMS, lambda sk, rid: RunState.RUNNING)
    assert reply == {"backend": "codex", "boot_id": process_boot_id(), "request_id": "r1",
                     "state": "running", "detail": ""}
    assert RunStatusReply.from_wire(reply, backend="codex").state is RunState.RUNNING


@pytest.mark.parametrize("params", [{}, {"sessionKey": "bot:user"}, {"requestId": " "}, "nope"])
def test_bad_params_answer_error_not_absence(params):
    reply = answer_run_status("codex", params, lambda sk, rid: RunState.NOT_FOUND)
    assert reply["state"] == "error" and "ValueError" in reply["detail"]


def test_a_raising_state_lookup_answers_error_for_the_named_request():
    def boom(sk, rid):
        raise RuntimeError("queue gone")
    reply = answer_run_status("openclaw", PARAMS, boom)
    assert reply["state"] == "error" and reply["request_id"] == "r1"
    assert reply["detail"] == "RuntimeError: queue gone"


@pytest.mark.parametrize("bad", ["running", None, RunState.UNSUPPORTED])
def test_a_lookup_returning_no_sendable_state_answers_error(bad):
    reply = answer_run_status("codex", PARAMS, lambda sk, rid: bad)
    assert reply["state"] == "error" and reply["request_id"] == "r1"
    assert reply["detail"].startswith("RunStatusReplyError:")


def test_only_not_found_proves_absence():
    proving = {s for s in RunState if s is not RunState.UNSUPPORTED
               and RunStatusReply("codex", "b", "r", s).proves_absent}
    assert proving == {RunState.NOT_FOUND}


def test_wrong_backend_reply_is_rejected():
    reply = answer_run_status("codex", PARAMS, lambda sk, rid: RunState.NOT_FOUND)
    with pytest.raises(RunStatusReplyError, match="addressed 'claude-code'"):
        RunStatusReply.from_wire(reply, backend="claude-code")


@pytest.mark.parametrize("mutate", [
    lambda d: d.pop("detail"),
    lambda d: d.update(extra=1),
    lambda d: d.update(state="gone"),
    lambda d: d.update(state="unsupported"),
    lambda d: d.update(boot_id=""),
])
def test_malformed_v1_reply_raises(mutate):
    reply = answer_run_status("codex", PARAMS, lambda sk, rid: RunState.NOT_FOUND)
    mutate(reply)
    with pytest.raises(RunStatusReplyError):
        RunStatusReply.from_wire(reply, backend="codex")


def test_pre_contract_bridge_is_unsupported_never_absent():
    reply = RunStatusReply.from_wire({"ok": False, "error": "unknown method: run.status"},
                                     backend="codex")
    assert reply.state is RunState.UNSUPPORTED and not reply.proves_absent
    assert reply.detail == "unknown method: run.status"


def _queue(phase=None, stopping=False, tombstoned=False):
    return SimpleNamespace(
        request_phase=lambda sk, rid: phase,
        is_stopping=lambda sk, rid: stopping,
        is_tombstoned=lambda sk, rid: tombstoned,
    )


@pytest.mark.parametrize("phase", list(RequestPhase))
def test_registered_request_reports_its_phase(phase):
    assert queue_run_state(_queue(phase), "bot:user", "r1").value == phase.value


def test_registered_and_stopping_wins_over_phase_and_tombstone():
    q = _queue(RequestPhase.RUNNING, stopping=True, tombstoned=True)
    assert queue_run_state(q, "bot:user", "r1") is RunState.STOPPING


def test_registered_tombstoned_request_still_reports_its_phase():
    # OpenClaw tombstones before the gateway confirms; the run is still live.
    q = _queue(RequestPhase.RUNNING, tombstoned=True)
    assert queue_run_state(q, "bot:user", "r1") is RunState.RUNNING


def test_unregistered_request_is_tombstoned_or_not_found():
    assert queue_run_state(_queue(tombstoned=True), "bot:user", "r1") is RunState.TOMBSTONED
    assert queue_run_state(_queue(), "bot:user", "r1") is RunState.NOT_FOUND
