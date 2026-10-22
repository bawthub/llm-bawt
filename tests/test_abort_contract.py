"""TASK-1231B / TASK-1232 P3: the shared bridge abort contract."""

from __future__ import annotations

import asyncio

import pytest

from agent_bridge.abort_contract import (
    ABORT_CAUSE_KEY,
    ABORT_TEARDOWN_BOUND_S,
    HOST_SHUTDOWN,
    TEARDOWN_DISCONNECT_BACKSTOP_S,
    TEARDOWN_KILL_WAIT_S,
    AbortReply,
    AbortReplyError,
    AbortState,
    abort_cause_of,
    aborted_terminal_raw,
    process_boot_id,
)
from agent_bridge.session_queue import SessionQueue


# -- terminal raw --------------------------------------------------------


def test_no_cause_raw_is_the_historical_wire_shape():
    # The app finalizes a turn as aborted only on end_reason == "aborted";
    # the user-Stop shape must stay byte-for-byte unchanged.
    assert aborted_terminal_raw() == {"end_reason": "aborted", "status": "cancelled"}


def test_host_shutdown_is_additive_and_keeps_aborted_end_reason():
    raw = aborted_terminal_raw(HOST_SHUTDOWN)
    assert raw == {"end_reason": "aborted", "status": "cancelled", "abort_cause": "host_shutdown"}
    assert abort_cause_of(raw) == HOST_SHUTDOWN


def test_unknown_cause_is_refused_at_the_producer():
    with pytest.raises(ValueError):
        aborted_terminal_raw("restart")


@pytest.mark.parametrize("raw", [
    None,
    {},
    {"end_reason": "aborted", "status": "cancelled"},
    {"end_reason": "completed", ABORT_CAUSE_KEY: HOST_SHUTDOWN},
    {"end_reason": "aborted", ABORT_CAUSE_KEY: "bogus"},
    "aborted",
])
def test_abort_cause_of_only_trusts_known_cause_on_aborted_terminal(raw):
    assert abort_cause_of(raw) is None


def test_boot_id_is_stable_for_the_process():
    assert process_boot_id() == process_boot_id()
    assert len(process_boot_id()) == 32


# -- normalized reply ----------------------------------------------------


def _reply(**over):
    data = {"backend": "claude-code", "boot_id": "b" * 32, "request_id": "req",
            "cancelled": True, "state": "cancelled", "detail": "task_cancelled"}
    data.update(over)
    return data


@pytest.mark.parametrize("state", list(AbortState))
def test_build_derives_cancelled_and_round_trips(state):
    reply = AbortReply.build(backend="codex", boot_id="boot", request_id="r", state=state)
    assert reply.cancelled is (state is AbortState.CANCELLED)
    assert AbortReply.from_dict(reply.to_dict()) == reply
    assert reply.to_dict()["state"] == state.value


@pytest.mark.parametrize("over", [
    {"cancelled": False},                       # contradicts state
    {"state": "not_found", "cancelled": True},  # contradicts state
    {"cancelled": 1},                           # not a bool
    {"state": "gone"},                          # unknown state
    {"backend": ""},
    {"boot_id": "  "},
    {"request_id": None},
    {"detail": None},
])
def test_from_dict_rejects_invalid_values(over):
    with pytest.raises(AbortReplyError):
        AbortReply.from_dict(_reply(**over))


def test_from_dict_rejects_missing_and_extra_keys():
    missing = _reply()
    del missing["boot_id"]
    with pytest.raises(AbortReplyError, match="boot_id"):
        AbortReply.from_dict(missing)
    with pytest.raises(AbortReplyError, match="aborted"):
        AbortReply.from_dict(_reply(aborted="claude-code:x"))
    with pytest.raises(AbortReplyError):
        AbortReply.from_dict(["not", "a", "mapping"])


# -- from_wire (Snark decision, TASK-1232 P3) -----------------------------------


def test_from_wire_parses_v1_strictly_and_marks_contract():
    reply = AbortReply.from_wire(_reply(), backend="claude-code")
    assert reply.state is AbortState.CANCELLED and reply.cancelled
    assert reply.contract == "v1"
    assert "contract" not in reply.to_dict()
    assert AbortReply.from_dict(reply.to_dict()) == reply


@pytest.mark.parametrize("data", [
    _reply(cancelled=False),                     # contradicts state
    _reply(state="gone"),
    {**_reply(), "ok": True},                    # extra key on a v1 reply
    {"state": "cancelled", "ok": True},          # v1 marker, otherwise legacy
    {"boot_id": "b" * 32, "aborted": True},      # v1 marker, otherwise legacy
])
def test_from_wire_never_falls_back_to_legacy_for_a_malformed_v1_reply(data):
    with pytest.raises(AbortReplyError):
        AbortReply.from_wire(data, backend="claude-code")


def test_from_wire_rejects_a_v1_reply_from_another_backend():
    with pytest.raises(AbortReplyError, match="codex"):
        AbortReply.from_wire(_reply(backend="codex"), backend="claude-code")


def test_from_wire_rejects_a_non_mapping():
    with pytest.raises(AbortReplyError):
        AbortReply.from_wire(["cancelled"], backend="codex")


@pytest.mark.parametrize("data, state, request_id", [
    # claude-code legacy: `aborted` is the session key, `cancelled` the bool.
    ({"ok": True, "aborted": "bot:user", "cancelled": True, "request_id": "r"}, "cancelled", "r"),
    ({"ok": True, "aborted": "bot:user", "cancelled": False, "request_id": "r"}, "not_found", "r"),
    # codex / local / openclaw legacy.
    ({"ok": True, "aborted": True, "requestId": "r", "detail": "task_cancelled"}, "cancelled", "r"),
    ({"ok": True, "aborted": False, "requestId": "r"}, "not_found", "r"),
    # Errors and anything unrecognized never confirm anything.
    ({"ok": False, "error": "chat.abort requires sessionKey and requestId"}, "error", "unknown"),
    ({"ok": True, "aborted": True, "error": "late"}, "error", "unknown"),
    ({"ok": True}, "error", "unknown"),
    ({"ok": True, "aborted": "bot:user"}, "error", "unknown"),
    # A detail is prose, not proof: only a bool flag counts.
    ({"ok": True, "aborted": "bot:user", "detail": "no_active_task"}, "error", "unknown"),
    ({"ok": True, "aborted": True, "detail": "no_active_task"}, "cancelled", "unknown"),
    ({"ok": True, "cancelled": 1}, "error", "unknown"),
])
def test_from_wire_legacy_mapping_claims_no_more_than_the_reply_proves(data, state, request_id):
    reply = AbortReply.from_wire(data, backend="claude-code")
    assert reply.state is AbortState(state)
    assert reply.cancelled is (state == "cancelled")
    assert (reply.backend, reply.boot_id, reply.contract) == ("claude-code", "legacy", "legacy")
    assert reply.request_id == request_id


def test_legacy_mapping_can_never_produce_tombstoned():
    shapes = [{"ok": ok, key: value} for ok in (True, False)
              for key in ("cancelled", "aborted", "detail", "error")
              for value in (True, False, "tombstoned", None)]
    assert all(AbortReply.from_wire(s, backend="codex").state is not AbortState.TOMBSTONED
               for s in shapes)


# -- teardown bound (TASK-1232 S3) ----------------------------------------------


def test_teardown_waits_fit_inside_the_published_bound():
    # The app's abort RPC timeout is this bound plus a margin, so the bridge's
    # teardown waits may never add up past it.
    assert TEARDOWN_DISCONNECT_BACKSTOP_S + TEARDOWN_KILL_WAIT_S < ABORT_TEARDOWN_BOUND_S
    # claude-agent-sdk close(): 5s write lock + 5s EOF + 5s SIGTERM + 5s SIGKILL.
    assert TEARDOWN_DISCONNECT_BACKSTOP_S > 4 * 5


# -- SessionQueue cause attribution -----------------------------------------


def test_cancel_all_records_cause_for_running_and_queued_sends():
    async def run():
        queue = SessionQueue()
        running_in = asyncio.Event()
        seen: dict[str, str | None] = {}

        async def send_reading_cause(request_id):
            # on_cancel runs where a send publishes its terminal.
            async def on_cancel():
                seen[request_id] = queue.cancel_cause("s", request_id)
            async with queue.active("s", request_id=request_id, on_cancel=on_cancel):
                running_in.set()
                await asyncio.Event().wait()

        running = asyncio.create_task(send_reading_cause("running"))
        await running_in.wait()
        queued = asyncio.create_task(send_reading_cause("queued"))
        await asyncio.sleep(0)
        cancelled = queue.cancel_all(cause=HOST_SHUTDOWN)
        assert set(cancelled) == {running, queued}
        # The running send also gets the cooperative flag (SDK may absorb cancel).
        assert queue.cancel_event("s").is_set()
        await asyncio.gather(running, queued, return_exceptions=True)
        assert seen == {"running": HOST_SHUTDOWN, "queued": HOST_SHUTDOWN}
        # Attribution is dropped with the registration.
        assert queue.cancel_cause("s", "running") is None
        assert queue.cancel_cause("s", "queued") is None
    asyncio.run(asyncio.wait_for(run(), timeout=5))


def test_cancel_all_keeps_existing_abort_attribution_and_does_not_recancel():
    async def run():
        queue = SessionQueue()
        entered = asyncio.Event()
        release_cleanup = asyncio.Event()
        cause_at_cleanup: list[str | None] = []

        async def on_cancel():
            # Terminal publication in progress; a second cancel would cut it off.
            await release_cleanup.wait()
            cause_at_cleanup.append(queue.cancel_cause("s", "r"))

        async def send():
            async with queue.active("s", request_id="r", on_cancel=on_cancel):
                entered.set()
                await asyncio.Event().wait()

        task = asyncio.create_task(send())
        await entered.wait()
        assert queue.cancel_request("s", "r")  # user Stop
        await asyncio.sleep(0)
        assert queue.cancel_all(cause=HOST_SHUTDOWN) == []
        release_cleanup.set()
        await asyncio.gather(task, return_exceptions=True)
        assert task.cancelled()
        assert cause_at_cleanup == [None]
    asyncio.run(asyncio.wait_for(run(), timeout=5))


def test_cancel_all_with_nothing_registered_is_a_noop():
    assert SessionQueue().cancel_all(cause=HOST_SHUTDOWN) == []
