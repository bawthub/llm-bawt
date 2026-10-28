"""TASK-1232: the ``run.status`` reply contract and the queue-derived state."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent_bridge.abort_contract import process_boot_id, process_is_pid1
from agent_bridge.run_contract import RunAccepted
from agent_bridge.run_status import (
    LIVE_STATES,
    RESTART_PROOF_BACKENDS,
    RunSighting,
    RunState,
    RunStatusReply,
    RunStatusReplyError,
    answer_run_status,
    proves_restart,
    queue_run_state,
)
from agent_bridge.session_queue import RequestPhase

PARAMS = {"sessionKey": "bot:user", "requestId": "r1"}


def test_answer_is_the_v1_shape_for_this_process():
    reply = answer_run_status("codex", PARAMS, lambda sk, rid: RunState.RUNNING)
    assert reply == {"backend": "codex", "boot_id": process_boot_id(), "request_id": "r1",
                     "state": "running", "detail": "", "pid1": process_is_pid1()}
    assert RunStatusReply.from_wire(reply, backend="codex").state is RunState.RUNNING


@pytest.mark.parametrize("params", [{}, {"sessionKey": "bot:user"}, {"requestId": " "}, "nope"])
def test_bad_params_answer_error_not_absence(params):
    reply = answer_run_status("codex", params, lambda sk, rid: RunState.NOT_FOUND)
    assert reply["state"] == "error" and "ValueError" in reply["detail"]


def test_a_raising_state_lookup_answers_error_for_the_named_request():
    def boom(sk, rid):
        raise RuntimeError("queue gone")
    reply = answer_run_status("local", PARAMS, boom)
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
    # A bridge may tombstone before its run confirms the stop; the run is still live.
    q = _queue(RequestPhase.RUNNING, tombstoned=True)
    assert queue_run_state(q, "bot:user", "r1") is RunState.RUNNING


def test_unregistered_request_is_tombstoned_or_not_found():
    assert queue_run_state(_queue(tombstoned=True), "bot:user", "r1") is RunState.TOMBSTONED
    assert queue_run_state(_queue(), "bot:user", "r1") is RunState.NOT_FOUND


# -- TASK-1233 Vex S2: pid1 and proves_restart ---------------------------------

def test_error_reply_also_reports_pid1():
    reply = answer_run_status("codex", {}, lambda sk, rid: RunState.NOT_FOUND)
    assert reply["state"] == "error" and reply["pid1"] is process_is_pid1()


def test_pid1_is_optional_on_the_wire():
    legacy = {"backend": "codex", "boot_id": "b", "request_id": "r1", "state": "not_found",
              "detail": ""}
    assert RunStatusReply.from_wire(legacy, backend="codex").pid1 is None
    assert RunStatusReply.from_wire({**legacy, "pid1": True}, backend="codex").pid1 is True
    with pytest.raises(RunStatusReplyError, match="pid1"):
        RunStatusReply.from_wire({**legacy, "pid1": "yes"}, backend="codex")


SIGHTED = RunSighting("r1", "codex", "boot-a", pid1=True)


def _reply(**over):
    fields = dict(backend="codex", boot_id="boot-b", request_id="r1", state=RunState.NOT_FOUND,
                  pid1=True)
    fields.update(over)
    return RunStatusReply(**fields)


@pytest.mark.parametrize("backend", ["claude-code", "codex"])
def test_a_new_pid1_boot_without_the_request_proves_restart(backend):
    sighted = RunSighting("r1", backend, "boot-a", pid1=True)
    assert proves_restart(sighted, _reply(backend=backend))
    assert proves_restart(sighted, _reply(backend=backend, state=RunState.TOMBSTONED))


def test_only_backends_whose_runs_die_with_the_bridge_prove_restart():
    """Snark step 34: a new boot proves a restart only for backends whose runs are bridge children."""
    assert RESTART_PROOF_BACKENDS == {"claude-code", "codex"}
    for backend in ("local", "local-model", "made-up"):
        sighted = RunSighting("r1", backend, "boot-a", pid1=True)
        assert not proves_restart(sighted, _reply(backend=backend))


@pytest.mark.parametrize("sighting,reply", [
    (SIGHTED, _reply(boot_id="boot-a")),                      # same process
    (SIGHTED, _reply(backend="claude-code")),                 # another backend
    (SIGHTED, _reply(request_id="r2")),                       # another request
    (SIGHTED, _reply(state=RunState.RUNNING)),                # still there
    (SIGHTED, _reply(state=RunState.STOPPING)),               # stopping there
    (SIGHTED, _reply(state=RunState.ERROR)),                  # no answer
    (SIGHTED, _reply(pid1=False)),                            # answerer not PID 1
    (SIGHTED, _reply(pid1=None)),                             # answerer unknown
    (RunSighting("r1", "codex", "boot-a", pid1=False), _reply()),  # sighted not PID 1
    (RunSighting("r1", "codex", "boot-a"), _reply()),              # sighted unknown
])
def test_anything_less_never_proves_restart(sighting, reply):
    assert not proves_restart(sighting, reply)


def test_a_sighting_comes_from_acceptance_or_a_live_reply_only():
    accepted = RunAccepted("r1", "codex", "boot-a", pid1=True)
    assert RunSighting.from_accepted(accepted) == SIGHTED
    for state in LIVE_STATES:
        assert RunSighting.from_reply(_reply(state=state, boot_id="boot-a")) == SIGHTED
    for state in set(RunState) - set(LIVE_STATES) - {RunState.UNSUPPORTED}:
        with pytest.raises(RunStatusReplyError, match="does not sight"):
            RunSighting.from_reply(_reply(state=state))
    with pytest.raises(RunStatusReplyError, match="pid1"):
        RunSighting("r1", "codex", "boot-a", pid1="yes")
    with pytest.raises(RunStatusReplyError, match="boot_id"):
        RunSighting("r1", "codex", " ")
