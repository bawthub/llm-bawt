"""The codex send on app-server: its steers and its stop (TASK-1243 P2/P3).

Sends go through the real ``_handle_send`` / ``_handle_steer`` /
``_handle_rpc`` handlers. Most run the fake app-server as a child process
(``tests/codex_app_server_fake``); the kill-escalation case runs a hung fake
process on the virtual clock, so its 10s interrupt bound and 2s teardown steps
take no real time and are asserted exactly.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_bridge.abort_contract import ABORT_TEARDOWN_BOUND_S
from agent_bridge.events import AgentEventKind
from agent_bridge.steer_contract import (
    LIFECYCLE_UNCERTAIN,
    STEER_PRIORITY_UNSUPPORTED,
    STEER_RUN_FINISHING,
    STEER_UNFRAMED,
    STEER_UNSUPPORTED_TRANSPORT,
)
from codex_bridge import app_server_send
from codex_bridge.app_server import CLOSE_STEP_GRACE_S, CodexAppServerRun, JsonRpcStdioClient
from codex_bridge.app_server_send import ABORT_INTERRUPT_BOUND_S
from codex_bridge.command_ops import CodexCommandMixin
from codex_bridge.event_ops import CodexEventMixin
from codex_bridge.process_ledger import register_process
from tests import codex_app_server_fake as fake
from tests import virtual_clock
from tests.test_other_bridge_abort_targeting import abort, make_bridge, send

SESSION = "bot:user"
HANG_GUARD_S = 30.0
REAL = pytest.mark.real_process(reason="runs the fake codex app-server as a child process")


class Bridge(CodexCommandMixin, CodexEventMixin):
    pass


def _bridge(tmp_path, codex):
    bridge = make_bridge("codex")
    bridge.__class__ = Bridge
    bridge._cwd = str(tmp_path)
    bridge._request_timeout = HANG_GUARD_S
    bridge._ensure_codex = lambda: codex
    bridge._build_prompt_input = lambda message, attachments, system_prompt=None: (message, [])
    bridge._rollout_usage = SimpleNamespace(resolve=AsyncMock(return_value=None))
    return bridge


def _fake_codex(tmp_path, **behaviour):
    env = fake.fake_env(tmp_path, **behaviour)
    return SimpleNamespace(_exec=SimpleNamespace(
        executable_path=str(fake.write_fake(tmp_path)), _build_env=lambda args: dict(env)))


def _send(bridge, redis, request_id="active", **extra):
    return send(bridge, request_id, redis, codex_transport="app_server", **extra)


def _published(bridge, kind):
    return [call.kwargs for call in bridge._publish_event.call_args_list if call.kwargs.get("kind") == kind]


def _lifecycle(bridge):
    return [(e["extra_raw"]["message_id"], e["extra_raw"]["state"])
            for e in _published(bridge, AgentEventKind.STEER_LIFECYCLE)]


def _rpc_result(bridge, rpc_id):
    return next(call.args[1] for call in bridge._publisher.publish_rpc_result.call_args_list
                if call.args[0] == rpc_id)


async def until(predicate):
    async with asyncio.timeout(HANG_GUARD_S):
        while not predicate():
            await asyncio.sleep(0.005)


def _live_run(bridge):
    run = bridge._session_queue.get_active_client(SESSION)
    return run if isinstance(run, CodexAppServerRun) else None


def _steer_fields(**fields):
    return {"request_id": "steer-rpc", "session_key": SESSION, "target_request_id": "active",
            "priority": "next", "origin": "user", "framed_message": "framed text",
            "message_id": "m-2", **fields}


# ---- a whole turn --------------------------------------------------------------


@REAL
def test_app_server_turn_streams_and_completes_like_exec(tmp_path):
    bridge = _bridge(tmp_path, _fake_codex(tmp_path, mode="complete"))
    redis = SimpleNamespace(xack=AsyncMock())
    asyncio.run(_send(bridge, redis))

    deltas = [e["text"] for e in _published(bridge, AgentEventKind.ASSISTANT_DELTA)]
    (done,) = _published(bridge, AgentEventKind.ASSISTANT_DONE)
    assert deltas == ["Hel", "lo"] and done["text"] == "Hello"
    assert not _published(bridge, AgentEventKind.ERROR)
    bridge._publisher.publish_run_done.assert_called_once()
    assert bridge._publisher.publish_run_done.call_args.kwargs["teardown"].exited is True
    assert bridge._session_queue.get_active_client(SESSION) is None
    received = fake.read_log(tmp_path / "received.jsonl")
    start = next(m for m in received if m.get("method") == "thread/start")
    assert start["params"] == {"model": "test-model", "approvalPolicy": "never",
                               "sandbox": "danger-full-access", "cwd": str(tmp_path)}
    fake.assert_matches_schema(tmp_path)


@REAL
def test_a_resumed_thread_uses_thread_resume(tmp_path):
    bridge = _bridge(tmp_path, _fake_codex(tmp_path, mode="complete"))
    asyncio.run(_send(bridge, SimpleNamespace(xack=AsyncMock()), thread_resume_id="thr-1"))
    resume = next(m for m in fake.read_log(tmp_path / "received.jsonl") if m.get("method") == "thread/resume")
    assert resume["params"]["threadId"] == "thr-1" and resume["params"]["excludeTurns"] is True
    assert len(_published(bridge, AgentEventKind.ASSISTANT_DONE)) == 1
    fake.assert_matches_schema(tmp_path)


@REAL
def test_a_failed_turn_publishes_its_error_once(tmp_path):
    bridge = _bridge(tmp_path, _fake_codex(tmp_path, mode="fail", error="model overloaded"))
    asyncio.run(_send(bridge, SimpleNamespace(xack=AsyncMock())))
    assert [e["text"] for e in _published(bridge, AgentEventKind.ERROR)] == ["model overloaded"]
    assert not _published(bridge, AgentEventKind.ASSISTANT_DONE)


@REAL
def test_another_app_server_version_falls_back_to_exec_and_is_remembered(tmp_path):
    codex = _fake_codex(tmp_path, version="0.158.0")
    exec_threads = []

    async def run_streamed(prompt, options):
        return "streamed"

    def start_thread(options):
        exec_threads.append(options)
        return SimpleNamespace(id="exec-thread", run_streamed=run_streamed)

    codex.start_thread = start_thread
    bridge = _bridge(tmp_path, codex)

    async def events(streamed):
        yield SimpleNamespace(type="turn.completed", usage=None)

    bridge._iter_events = events
    redis = SimpleNamespace(xack=AsyncMock())

    async def run():
        await _send(bridge, redis, request_id="one")
        await _send(bridge, redis, request_id="two")

    asyncio.run(run())
    assert len(exec_threads) == 2
    assert len(_published(bridge, AgentEventKind.ASSISTANT_DONE)) == 2
    assert "0.158.0" in bridge._app_server_unsupported
    # Spawned once: the second send went straight to exec.
    assert [m.get("method") for m in fake.read_log(tmp_path / "received.jsonl")] == ["initialize"]


# ---- steer -----------------------------------------------------------------


@REAL
def test_steer_reaches_the_live_turn_and_abort_interrupts_without_a_kill(tmp_path):
    """B2(a): the interrupt ends the turn, so nothing is cancelled or killed."""
    bridge = _bridge(tmp_path, _fake_codex(tmp_path, mode="hold"))
    redis = SimpleNamespace(xack=AsyncMock())

    async def run():
        task = asyncio.create_task(_send(bridge, redis))
        await until(lambda: _live_run(bridge) is not None)
        await bridge._handle_steer(_steer_fields(), "steer-msg", redis)
        await until(lambda: ("m-2", "started") in _lifecycle(bridge))
        reply = await abort(bridge, "active", redis)
        await asyncio.wait_for(task, HANG_GUARD_S)
        return reply, task

    reply, task = asyncio.run(run())
    assert _rpc_result(bridge, "steer-rpc") == {"ok": True, "detail": "steered", "active_request_id": "active",
                                                "can_escalate": False, "can_cancel": False}
    (lifecycle,) = _published(bridge, AgentEventKind.STEER_LIFECYCLE)
    assert lifecycle["extra_raw"] == {"message_id": "m-2", "state": "started", "origin": "user",
                                      "steer_rpc_id": "steer-rpc", "run_request_id": "active"}
    assert reply["state"] == "cancelled" and reply["cancelled"] is True
    assert reply["detail"] == "turn_interrupted,teardown:close=stdin;exited=true"
    assert not task.cancelled()
    assert not _published(bridge, AgentEventKind.ASSISTANT_DONE)
    assert bridge._publisher.publish_run_done.call_args.kwargs["teardown"].exited is True
    methods = [m.get("method") for m in fake.read_log(tmp_path / "received.jsonl")]
    assert methods[-2:] == ["turn/steer", "turn/interrupt"]
    steer = next(m for m in fake.read_log(tmp_path / "received.jsonl") if m.get("method") == "turn/steer")
    assert steer["params"]["input"][0]["text"] == "framed text"
    assert steer["params"]["clientUserMessageId"] == "m-2"
    fake.assert_matches_schema(tmp_path)


@REAL
def test_a_steer_accepted_but_never_shown_is_uncertain(tmp_path):
    bridge = _bridge(tmp_path, _fake_codex(tmp_path, mode="hold", steer="lose"))
    redis = SimpleNamespace(xack=AsyncMock())

    async def run():
        task = asyncio.create_task(_send(bridge, redis))
        await until(lambda: _live_run(bridge) is not None)
        await bridge._handle_steer(_steer_fields(), "steer-msg", redis)
        await asyncio.wait_for(task, HANG_GUARD_S)

    asyncio.run(run())
    assert _rpc_result(bridge, "steer-rpc")["ok"] is True
    assert _lifecycle(bridge) == [("m-2", LIFECYCLE_UNCERTAIN)]
    assert len(_published(bridge, AgentEventKind.ASSISTANT_DONE)) == 1


def _stub_run(**state):
    stub = type("Client", (), {"add_observer": lambda self, fn: None, "request": AsyncMock()})()
    run = CodexAppServerRun(stub, request_id="active", thread_id="thr-1")
    run.set_turn("turn-1")
    for name, value in state.items():
        setattr(run, name, value)
    return run


@pytest.mark.parametrize(("active", "fields", "error"), [
    (None, {}, "no_active_run"),
    ("exec", {}, STEER_UNSUPPORTED_TRANSPORT),
    ("run", {"target_request_id": "other"}, "active_run_mismatch"),
    ("run", {"priority": "now"}, STEER_PRIORITY_UNSUPPORTED),
    ("run", {"steer_action": "escalate"}, STEER_PRIORITY_UNSUPPORTED),
    ("run", {"escalate": "1"}, STEER_PRIORITY_UNSUPPORTED),
    ("run", {"framed_message": ""}, STEER_UNFRAMED),
    ("stopping", {}, STEER_RUN_FINISHING),
    ("run", {"session_key": ""}, "chat.steer requires session_key"),
])
def test_steer_refusals(active, fields, error):
    bridge = make_bridge("codex")
    redis = SimpleNamespace(xack=AsyncMock())
    run = None
    if active == "exec":
        bridge._session_queue.set_active_client(SESSION, SimpleNamespace(request_id="active", abort=MagicMock()))
    elif active is not None:
        run = _stub_run(stopping=active == "stopping")
        bridge._session_queue.set_active_client(SESSION, run)

    asyncio.run(bridge._handle_steer(_steer_fields(**fields), "steer-msg", redis))

    assert _rpc_result(bridge, "steer-rpc") == {"ok": False, "error": error}
    redis.xack.assert_awaited_once()
    assert redis.xack.call_args.args[-1] == "steer-msg"
    if run is not None:
        run.client.request.assert_not_awaited()


def test_a_system_steer_may_be_unframed():
    bridge = make_bridge("codex")
    run = _stub_run()
    bridge._session_queue.set_active_client(SESSION, run)
    fields = _steer_fields(origin="system", message="  note  ", framed_message="")
    fields.pop("framed_message")
    asyncio.run(bridge._handle_steer(fields, "steer-msg", SimpleNamespace(xack=AsyncMock())))
    assert _rpc_result(bridge, "steer-rpc")["ok"] is True
    params = run.client.request.call_args.args[1]
    assert params["input"][0]["text"] == "note" and params["expectedTurnId"] == "turn-1"


def test_the_listener_routes_codex_steers_only(monkeypatch):
    bridge = make_bridge("codex")
    steered = []

    async def handle(fields, msg_id, redis):
        steered.append(msg_id)

    bridge._handle_steer = handle
    batches = [[("stream", [
        ("1-0", {"action": "chat.steer", "backend": "codex", "session_key": SESSION}),
        ("2-0", {"action": "chat.steer", "backend": "claude-code", "session_key": SESSION}),
    ])]]

    async def xreadgroup(*args, **kwargs):
        if batches:
            return batches.pop()
        raise asyncio.CancelledError

    redis = SimpleNamespace(xack=AsyncMock(), ping=AsyncMock(), xgroup_create=AsyncMock(),
                            aclose=AsyncMock(), xreadgroup=xreadgroup)
    monkeypatch.setattr("redis.asyncio.Redis", lambda **kwargs: redis)

    async def run():
        with pytest.raises(asyncio.CancelledError):
            await bridge._command_listener()
        await asyncio.sleep(0)

    asyncio.run(run())
    assert steered == ["1-0"]
    assert [call.args[-1] for call in redis.xack.call_args_list] == ["2-0"]


# ---- abort: a hung app-server is killed in bound (B2(b)) ---------------------


class _HungStdin:
    def __init__(self, proc):
        self._proc = proc
        self._closing = False

    def write(self, data):
        for line in data.decode().splitlines():
            self._proc.on_message(json.loads(line))

    async def drain(self):
        pass

    def is_closing(self):
        return self._closing

    def close(self):
        self._closing = True  # a hung server does not exit on EOF


class HungAppServer:
    """Answers the handshake and the turn, then ignores ``turn/interrupt``,
    stdin EOF and SIGTERM; only SIGKILL ends it."""

    pid = 4242

    def __init__(self):
        self.stdout = asyncio.StreamReader()
        self.stderr = None
        self.returncode = None
        self.stdin = _HungStdin(self)
        self.signals = []
        self.received = []
        self._exited = asyncio.Event()

    def _reply(self, message):
        self.stdout.feed_data((json.dumps(message) + "\n").encode())

    def on_message(self, message):
        self.received.append(message.get("method"))
        method, rid = message.get("method"), message.get("id")
        if method in ("thread/start", "thread/resume"):
            self._reply({"id": rid, "result": {"thread": {"id": "thr-1"}}})
        elif method == "turn/start":
            self._reply({"id": rid, "result": {"turn": {"id": "turn-1", "items": [], "status": "inProgress"}}})

    def terminate(self):
        self.signals.append("SIGTERM")

    def kill(self):
        self.signals.append("SIGKILL")
        self.returncode = -9
        self.stdout.feed_eof()
        self._exited.set()

    async def wait(self):
        await self._exited.wait()
        return self.returncode


def test_abort_kills_an_app_server_that_ignores_the_interrupt(tmp_path, monkeypatch):
    procs = []

    async def spawn(executable, **kwargs):
        procs.append(HungAppServer())  # in the loop: it owns a StreamReader
        register_process(procs[-1])
        return JsonRpcStdioClient(procs[-1])

    monkeypatch.setattr(app_server_send, "spawn_app_server", spawn)
    bridge = _bridge(tmp_path, SimpleNamespace(_exec=SimpleNamespace(
        executable_path="codex", _build_env=lambda args: {})))
    redis = SimpleNamespace(xack=AsyncMock())

    async def run():
        loop = asyncio.get_running_loop()
        task = asyncio.create_task(_send(bridge, redis))
        while _live_run(bridge) is None:
            await asyncio.sleep(0)
        started = loop.time()
        reply = await abort(bridge, "active", redis)
        spent = loop.time() - started
        await asyncio.gather(task, return_exceptions=True)
        return reply, spent

    reply, spent = virtual_clock.run(run())
    (proc,) = procs
    assert reply["state"] == "cancelled" and reply["cancelled"] is True
    assert reply["detail"] == "interrupt_timeout,task_cancelled,teardown:close=sigkill;exited=true"
    assert proc.signals == ["SIGTERM", "SIGKILL"]
    assert "turn/interrupt" in proc.received
    # The interrupt's bound, then stdin and SIGTERM each get their grace.
    assert spent == pytest.approx(ABORT_INTERRUPT_BOUND_S + 2 * CLOSE_STEP_GRACE_S)
    assert spent <= ABORT_TEARDOWN_BOUND_S  # timing-ok: virtual clock, exact
    assert bridge._publisher.publish_run_done.call_args.kwargs["teardown"].exited is True
    assert not _published(bridge, AgentEventKind.ASSISTANT_DONE)
    assert bridge._session_queue.get_active_client(SESSION) is None


def test_abort_of_another_send_takes_the_cancel_path(monkeypatch):
    """B2(c) in miniature: an exec client, or an app-server run for another
    request, is cancelled as before (tests/test_other_bridge_abort_targeting.py
    covers that path in full)."""
    bridge = make_bridge("codex")
    seen = []

    def handler(queue):
        async def cancel(session_key, request_id):
            seen.append((session_key, request_id))
            return "cancelled", "task_cancelled"
        return cancel

    monkeypatch.setattr(app_server_send, "cancel_request_handler", handler)
    run = _stub_run()

    async def scenario():
        bridge._session_queue.set_active_client(SESSION, SimpleNamespace(request_id="active"))
        assert await bridge._abort_request(SESSION, "active") == ("cancelled", "task_cancelled")
        bridge._session_queue.set_active_client(SESSION, run)
        assert (await bridge._abort_request(SESSION, "other"))[1] == "task_cancelled"

    asyncio.run(scenario())
    assert seen == [(SESSION, "active"), (SESSION, "other")]
    assert run.stopping is False
    run.client.request.assert_not_awaited()
