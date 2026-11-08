"""TASK-1385: an app-side stream error aborts the bridge run before going terminal."""

import asyncio
import threading
from types import SimpleNamespace

from llm_bawt.service import turn_abort


def _loop_thread():
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    return loop, thread


def _service(turn):
    return SimpleNamespace(_turn_log_store=SimpleNamespace(get_turn=lambda _id: turn))


def test_aborts_this_turns_exact_run_and_reports_state(monkeypatch) -> None:
    turn = SimpleNamespace(id="t1", agent_session_key="caid:nick", agent_request_id="req-1")
    seen = []

    async def fake_abort(service, row, execution):
        seen.append((row.agent_session_key, row.agent_request_id, execution))
        return {"bridge_state": "aborted"}

    monkeypatch.setattr(turn_abort, "_abort_bridge_run", fake_abort)
    loop, thread = _loop_thread()
    try:
        state = turn_abort.abort_bridge_run_before_app_error(_service(turn), "t1", "exec", loop)
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(2)

    assert state == "aborted"
    assert seen == [("caid:nick", "req-1", "exec")]


def test_abort_failure_never_raises(monkeypatch) -> None:
    turn = SimpleNamespace(id="t1", agent_session_key="caid:nick", agent_request_id="req-1")

    async def broken(service, row, execution):
        raise RuntimeError("redis down")

    monkeypatch.setattr(turn_abort, "_abort_bridge_run", broken)
    loop, thread = _loop_thread()
    try:
        state = turn_abort.abort_bridge_run_before_app_error(_service(turn), "t1", None, loop)
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(2)

    assert state == "abort_failed:RuntimeError"


def test_hung_bridge_abort_times_out_and_is_recorded(monkeypatch) -> None:
    """A dead bridge must not hold the row non-terminal: the wait is bounded."""
    import agent_bridge.abort_contract as contract

    turn = SimpleNamespace(id="t1", agent_session_key="caid:nick", agent_request_id="req-1")

    async def hang(service, row, execution):
        await asyncio.sleep(60)

    monkeypatch.setattr(turn_abort, "_abort_bridge_run", hang)
    monkeypatch.setattr(contract, "ABORT_RPC_TIMEOUT_S", -4.8)  # wait = timeout + 5 = 0.2s
    loop, thread = _loop_thread()
    try:
        state = turn_abort.abort_bridge_run_before_app_error(_service(turn), "t1", None, loop)
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(2)

    assert state == "abort_failed:TimeoutError"


def test_missing_row_skips_abort() -> None:
    assert turn_abort.abort_bridge_run_before_app_error(_service(None), "t1", None, None) == "no_turn_row"
