"""Codex app-server JSON-RPC client, run and event adapter (TASK-1243 P2).

The client tests run the fake app-server (``tests/codex_app_server_fake``) as
a real child process; the messages both sides wrote are checked against the
pinned 0.157.1 schema.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from agent_bridge.steer_contract import STEER_RUN_FINISHING
from codex_bridge.app_server import (
    APP_SERVER_VERSION,
    INTERRUPT_NO_TURN,
    METHOD_NOT_FOUND,
    STEERED_UNCONFIRMED,
    TURN_ENDED,
    TURN_INTERRUPTED,
    AppServerClosed,
    AppServerVersionMismatch,
    CodexAppServerRun,
    JsonRpcStdioClient,
    server_version,
    spawn_app_server,
)
from codex_bridge.app_server_events import ABORTED, AppServerEventAdapter, app_server_input
from codex_bridge.process_ledger import ProcessLedger, bind_ledger
from tests import codex_app_server_fake as fake

# Hang guard for a broken client: only a failure ever waits this long.
HANG_GUARD_S = 30.0

REAL = pytest.mark.real_process(reason="runs the fake codex app-server as a child process")


async def _spawn(tmp_path, **behaviour):
    return await spawn_app_server(
        str(fake.write_fake(tmp_path)), overrides=(), env=fake.fake_env(tmp_path, **behaviour),
        cwd=str(tmp_path), timeout=HANG_GUARD_S,
    )


async def _start_turn(client, cwd):
    opened = await client.request("thread/start", {"model": "m", "approvalPolicy": "never",
                                                   "sandbox": "danger-full-access", "cwd": cwd})
    started = await client.request("turn/start", {"threadId": opened["thread"]["id"],
                                                  "input": app_server_input("hello")})
    return opened["thread"]["id"], started["turn"]["id"]


async def _notes_until(client, predicate):
    seen = []
    while True:
        note = await client.next_notification(HANG_GUARD_S)
        assert note is not None, f"app-server exited; saw {[n['method'] for n in seen]}"
        seen.append(note)
        if predicate(note):
            return seen


# ---- client ---------------------------------------------------------------


@REAL
def test_handshake_turn_and_teardown_follow_the_schema(tmp_path):
    async def run():
        argv = tmp_path / "argv.json"
        client = await spawn_app_server(
            str(fake.write_fake(tmp_path)), overrides=("model_reasoning_effort=high", 'plugins."a".enabled=true'),
            env=fake.fake_env(tmp_path, argv=str(argv)), cwd=str(tmp_path), timeout=HANG_GUARD_S,
        )
        await _start_turn(client, str(tmp_path))
        notes = await _notes_until(client, lambda n: n["method"] == "turn/completed")
        detail = await client.close()
        return json.loads(argv.read_text()), notes, detail

    argv, notes, detail = asyncio.run(run())
    # --config pairs are global options: before the subcommand.
    assert argv == ["--config", "model_reasoning_effort=high",
                    "--config", 'plugins."a".enabled=true', "app-server"]
    assert [n["method"] for n in notes][-2:] == ["thread/tokenUsage/updated", "turn/completed"]
    # app-server exits on stdin EOF, so the first teardown step reaps it.
    assert detail == "teardown:close=stdin;exited=true"
    received = fake.read_log(tmp_path / "received.jsonl")
    assert [m.get("method") for m in received][:2] == ["initialize", "initialized"]
    fake.assert_matches_schema(tmp_path)


@REAL
def test_server_requests_are_refused_never_approved(tmp_path):
    async def run():
        client = await _spawn(tmp_path, mode="complete")
        await _start_turn(client, str(tmp_path))
        # The fake completes the turn only after it reads the client's answer.
        await _notes_until(client, lambda n: n["method"] == "turn/completed")
        await client.close()

    asyncio.run(run())
    replies = [m for m in fake.read_log(tmp_path / "received.jsonl") if m.get("id") == "srv-1"]
    assert len(replies) == 1 and "result" not in replies[0]
    assert replies[0]["error"]["code"] == METHOD_NOT_FOUND


@REAL
@pytest.mark.parametrize("version", ["0.158.0", "unknown"])
def test_another_version_is_refused_and_reaped(tmp_path, version):
    ledger = ProcessLedger()

    async def run():
        bind_ledger(ledger)
        with pytest.raises(AppServerVersionMismatch, match=APP_SERVER_VERSION):
            await _spawn(tmp_path, version=version)

    asyncio.run(run())
    (proc,) = ledger.procs
    assert proc.returncode is not None
    assert [m.get("method") for m in fake.read_log(tmp_path / "received.jsonl")] == ["initialize"]


@REAL
def test_a_cancel_during_the_refused_handshake_teardown_waits_for_the_reap(tmp_path, monkeypatch):
    """A cancel landing while spawn_app_server tears a refused child down is
    held until the child is reaped, then re-raised."""
    ledger = ProcessLedger()
    entered, release = None, None
    teardown = JsonRpcStdioClient._teardown

    async def gated_teardown(self):
        entered.set()
        await release.wait()
        return await teardown(self)

    monkeypatch.setattr(JsonRpcStdioClient, "_teardown", gated_teardown)

    async def run():
        nonlocal entered, release
        entered, release = asyncio.Event(), asyncio.Event()
        bind_ledger(ledger)
        task = asyncio.create_task(_spawn(tmp_path, version="0.158.0"))
        await asyncio.wait_for(entered.wait(), HANG_GUARD_S)
        task.cancel()
        for _ in range(20):
            await asyncio.sleep(0)
        assert not task.done(), "spawn_app_server returned while its child was still being torn down"
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())
    (proc,) = ledger.procs
    assert proc.returncode is not None


def test_server_version_reads_the_user_agent():
    assert server_version({"userAgent": "x/0.157.1 (Linux 6; x86_64)"}) == "0.157.1"
    assert server_version({"userAgent": "no-version"}) is None
    assert server_version(None) is None


@REAL
def test_a_closed_client_refuses_requests_as_never_sent(tmp_path):
    async def run():
        client = await _spawn(tmp_path, mode="hold")
        await client.close()
        with pytest.raises(AppServerClosed) as raised:
            await client.request("turn/interrupt", {"threadId": "t", "turnId": "u"})
        return raised.value, await client.next_notification(HANG_GUARD_S)

    error, note = asyncio.run(run())
    assert error.sent is False
    assert note is None  # closed: every later read says so at once


# ---- run: steer and interrupt against the fake ------------------------------


@REAL
def test_steer_carries_the_client_id_and_interrupt_ends_the_turn(tmp_path):
    async def run():
        client = await _spawn(tmp_path, mode="hold")
        thread_id, turn_id = await _start_turn(client, str(tmp_path))
        run_ = CodexAppServerRun(client, request_id="r1", thread_id=thread_id)
        run_.set_turn(turn_id)
        await run_.steer("more please", message_id="m-2", origin="user", steer_rpc_id="rpc-2")
        pending = run_.pending_steers["m-2"]
        notes = await _notes_until(client, lambda n: n["method"] == "item/started"
                                   and n["params"]["item"]["type"] == "userMessage")
        reason = await run_.stop(HANG_GUARD_S)
        detail = await client.close()
        return pending, notes[-1], reason, run_.turn_status, detail

    pending, user_note, reason, status, detail = asyncio.run(run())
    assert pending.accepted and pending.resolved.is_set() and not pending.started
    assert user_note["params"]["item"]["clientId"] == "m-2"
    assert (reason, status) == (TURN_INTERRUPTED, "interrupted")
    assert detail == "teardown:close=stdin;exited=true"
    steer = next(m for m in fake.read_log(tmp_path / "received.jsonl") if m.get("method") == "turn/steer")
    assert steer["params"] == {"threadId": "thr-1", "expectedTurnId": "turn-1", "clientUserMessageId": "m-2",
                               "input": [{"type": "text", "text": "more please", "text_elements": []}]}
    fake.assert_matches_schema(tmp_path)


@REAL
def test_a_steer_app_server_refuses_is_run_finishing(tmp_path):
    async def run():
        client = await _spawn(tmp_path, mode="hold")
        thread_id, _ = await _start_turn(client, str(tmp_path))
        run_ = CodexAppServerRun(client, request_id="r1", thread_id=thread_id)
        run_.set_turn("turn-stale")  # app-server answers INVALID_REQUEST
        with pytest.raises(RuntimeError, match=STEER_RUN_FINISHING):
            await run_.steer("late", message_id="m-3", origin="user", steer_rpc_id=None)
        await client.close()
        return run_

    run_ = asyncio.run(run())
    assert run_.pending_steers == {}


def _bare_run(**client):
    stub = type("Client", (), {"add_observer": lambda self, fn: None, **client})()
    return CodexAppServerRun(stub, request_id="r1", thread_id="thr-1")


@pytest.mark.parametrize("state", ["stopping", "done", "no_turn"])
def test_steer_without_a_live_turn_is_refused_before_any_rpc(state):
    sent = []

    async def request(self, *args, **kwargs):
        sent.append(args)

    run_ = _bare_run(request=request)
    if state != "no_turn":
        run_.set_turn("turn-1")
    if state == "stopping":
        run_.stopping = True
    elif state == "done":
        run_.mark_finished("completed")

    with pytest.raises(RuntimeError, match=STEER_RUN_FINISHING):
        asyncio.run(run_.steer("x", message_id="m", origin="user", steer_rpc_id=None))
    assert sent == [] and run_.pending_steers == {}


def _closing_run(sent):
    async def request(self, *args, **kwargs):
        raise AppServerClosed("gone", sent=sent)

    run_ = _bare_run(request=request)
    run_.set_turn("turn-1")
    return run_


def test_a_closed_pipe_before_delivery_is_run_finishing():
    run_ = _closing_run(sent=False)
    with pytest.raises(RuntimeError, match=STEER_RUN_FINISHING):
        asyncio.run(run_.steer("x", message_id="m", origin="user", steer_rpc_id=None))
    assert run_.pending_steers == {}


def test_a_reply_lost_after_the_write_is_unconfirmed_and_kept_pending():
    run_ = _closing_run(sent=True)
    detail = asyncio.run(run_.steer("x", message_id="m", origin="user", steer_rpc_id=None))
    assert detail == STEERED_UNCONFIRMED
    pending = run_.pending_steers["m"]
    assert pending.accepted and pending.resolved.is_set() and not pending.started


def test_interrupt_without_a_turn_or_after_the_end_sends_nothing():
    async def request(self, *args, **kwargs):
        raise AssertionError("no RPC expected")

    run_ = _bare_run(request=request)
    assert asyncio.run(run_.interrupt()) == INTERRUPT_NO_TURN
    run_.set_turn("turn-1")
    run_.mark_finished("completed")
    assert asyncio.run(run_.interrupt()) == TURN_ENDED


def test_interrupt_when_app_server_is_gone_ends_the_turn():
    async def request(self, *args, **kwargs):
        raise AppServerClosed("gone", sent=False)

    run_ = _bare_run(request=request)
    run_.set_turn("turn-1")
    assert asyncio.run(run_.stop(HANG_GUARD_S)) == TURN_ENDED
    assert run_.turn_done.is_set()


def test_a_turn_completed_read_before_set_turn_still_counts():
    observers = []
    stub = type("Client", (), {"add_observer": lambda self, fn: observers.append(fn)})()
    run_ = CodexAppServerRun(stub, request_id="r1", thread_id="thr-1")
    observers[0]({"method": "turn/completed", "params": {"turn": {"id": "turn-1", "status": "interrupted"}}})
    assert not run_.turn_done.is_set()
    run_.set_turn("turn-1")
    assert run_.turn_done.is_set() and run_.turn_status == "interrupted"


# ---- event adapter -----------------------------------------------------------


def _item(method, **item):
    return {"method": method, "params": {"threadId": "t", "turnId": "u", "item": item}}


def test_items_become_exec_events_with_snake_case_fields():
    adapter = AppServerEventAdapter()
    event = adapter.adapt(_item("item/completed", type="commandExecution", id="c1", command="ls",
                                aggregatedOutput="a\n", exitCode=0, durationMs=5, status="completed"))
    assert (event.type, event.item.type, event.item.id) == ("item.completed", "command_execution", "c1")
    assert (event.item.aggregated_output, event.item.exit_code, event.item.duration_ms) == ("a\n", 0, 5)

    change = adapter.adapt(_item("item/started", type="fileChange", id="f1", status="inProgress",
                                 changes=[{"path": "a.py", "kind": {"type": "add"}, "diff": ""}]))
    assert (change.item.type, change.item.changes[0]["kind"]) == ("file_change", "add")

    tool = adapter.adapt(_item("item/started", type="mcpToolCall", id="t1", server="s", tool="x",
                               arguments={"q": 1}, status="inProgress"))
    assert tool.item.type == "mcp_tool_call" and tool.item.arguments == {"q": 1}


def test_agent_message_deltas_accumulate_per_item():
    adapter = AppServerEventAdapter()
    started = adapter.adapt(_item("item/started", type="agentMessage", id="m1", text=""))
    assert (started.type, started.item.type, started.item.text) == ("item.started", "agent_message", "")
    texts = [adapter.adapt({"method": "item/agentMessage/delta",
                            "params": {"itemId": "m1", "delta": d, "threadId": "t", "turnId": "u"}}).item.text
             for d in ("Hel", "lo")]
    assert texts == ["Hel", "Hello"]
    done = adapter.adapt(_item("item/completed", type="agentMessage", id="m1", text="Hello"))
    assert (done.type, done.item.text) == ("item.completed", "Hello")
    again = adapter.adapt({"method": "item/agentMessage/delta", "params": {"itemId": "m1", "delta": "!"}})
    assert again.item.text == "!"  # a completed item's text is not carried over


def test_turn_completed_carries_the_latest_thread_total():
    adapter = AppServerEventAdapter()
    for total in (5, 13):
        assert adapter.adapt({"method": "thread/tokenUsage/updated", "params": {"tokenUsage": {
            "last": {"inputTokens": 1}, "total": {"inputTokens": total, "cachedInputTokens": 2,
                                                  "outputTokens": 3, "reasoningOutputTokens": 0,
                                                  "totalTokens": total + 3}}}}) is None
    event = adapter.adapt({"method": "turn/completed", "params": {"turn": {"id": "u", "status": "completed"}}})
    assert event.type == "turn.completed"
    assert event.usage == {"input_tokens": 13, "cached_input_tokens": 2, "output_tokens": 3,
                           "reasoning_output_tokens": 0, "total_tokens": 16}


@pytest.mark.parametrize(("turn", "kind", "message"), [
    ({"status": "interrupted"}, ABORTED, None),
    ({"status": "failed", "error": {"message": "usage limit"}}, "turn.failed", "usage limit"),
    ({"status": "failed"}, "turn.failed", "Codex turn failed"),
])
def test_turn_end_statuses(turn, kind, message):
    event = AppServerEventAdapter().adapt({"method": "turn/completed", "params": {"turn": {"id": "u", **turn}}})
    assert event.type == kind
    if message is not None:
        assert event.error == {"message": message}


def test_retry_errors_and_unknown_notes_are_skipped():
    adapter = AppServerEventAdapter()
    assert adapter.adapt({"method": "error", "params": {"error": {"message": "x"}, "willRetry": True}}) is None
    assert adapter.adapt({"method": "thread/status/changed", "params": {}}) is None
    assert adapter.adapt({"method": "item/started", "params": {"item": "bad"}}) is None


def test_prompt_input_maps_text_and_images():
    assert app_server_input("hi") == [{"type": "text", "text": "hi", "text_elements": []}]
    parts = app_server_input([{"type": "text", "text": "see"}, {"type": "local_image", "path": "/tmp/a.png"}])
    assert parts == [{"type": "text", "text": "see", "text_elements": []},
                     {"type": "localImage", "path": "/tmp/a.png"}]
    with pytest.raises(ValueError, match="unsupported"):
        app_server_input([{"type": "audio"}])
