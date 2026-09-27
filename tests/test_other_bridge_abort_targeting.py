from __future__ import annotations

import asyncio
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_bridge.events import AgentEventKind
from agent_bridge.session_queue import SessionQueue
from codex_bridge.command_ops import CodexCommandMixin
from local_model_bridge.bridge import LocalModelBridge
from openclaw_bridge.bridge import SessionBridge
from openclaw_bridge.ingest import EventIngestPipeline


def make_bridge(kind, ws=None):
    if kind == "codex":
        bridge = CodexCommandMixin()
        bridge._default_model = "test-model"
        bridge._cleanup_tmp_files = MagicMock()
        bridge._discard_changed_file_request = MagicMock()
        bridge._publish_event = MagicMock()
    elif kind == "local":
        bridge = LocalModelBridge.__new__(LocalModelBridge)
        bridge._publish_event = MagicMock()
    else:
        bridge = SessionBridge(ws or MagicMock(), EventIngestPipeline(), MagicMock(), MagicMock())
    bridge._session_queue = SessionQueue()
    bridge._publisher = MagicMock()
    bridge._trigger_message_ids = {}
    bridge._backend_name = kind
    return bridge


def send(bridge, request_id, redis, **extra):
    handler = getattr(bridge, "_handle_send", None) or bridge._handle_send_command
    return handler({"request_id": request_id, "session_key": "bot:user", "message": "hello",
                    "model": "test-model", "trigger_message_id": "trigger", **extra}, request_id, redis)


async def abort(bridge, target, redis, session="bot:user"):
    handler = getattr(bridge, "_handle_rpc", None) or bridge._handle_rpc_command
    await handler({"request_id": "rpc", "backend": bridge._backend_name, "method": "chat.abort",
                   "params": {"sessionKey": session, "requestId": target}}, "rpc", redis)
    return bridge._publisher.publish_rpc_result.call_args.args[1]


async def until(predicate):
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(0.001)


@pytest.mark.parametrize("kind", ["codex", "local", "openclaw"])
def test_queued_abort_is_exact_and_publishes_terminal(kind):
    async def run():
        bridge = make_bridge(kind)
        redis = SimpleNamespace(xack=AsyncMock())
        blocker = asyncio.Event()

        async def hold():
            async with bridge._session_queue.active("bot:user", request_id="active"):
                await blocker.wait()

        active = asyncio.create_task(hold())
        await until(lambda: bridge._session_queue.is_busy("bot:user"))
        queued = asyncio.create_task(send(bridge, "queued", redis))
        await until(lambda: ("bot:user", "queued") in bridge._session_queue._request_tasks)
        sibling = asyncio.create_task(send(bridge, "sibling", redis))
        await until(lambda: ("bot:user", "sibling") in bridge._session_queue._request_tasks)
        for target, session in [("missing", "bot:user"), ("queued", "other:user"), ("", "bot:user")]:
            result = await abort(bridge, target, redis, session)
            assert not result.get("aborted")
            assert not active.cancelling() and not queued.cancelling() and not sibling.cancelling()
        assert (await abort(bridge, "queued", redis))["aborted"] is True
        await abort(bridge, "queued", redis)  # repeated cancellation cannot cut off cleanup
        await asyncio.gather(queued, return_exceptions=True)
        bridge._publisher.publish_run_done.assert_called_once_with("queued")
        assert [call.args[-1] for call in redis.xack.call_args_list].count("queued") == 1
        assert not active.cancelling() and not sibling.cancelling()
        assert not bridge._session_queue.cancel_event("bot:user").is_set()
        sibling.cancel()
        blocker.set()
        await asyncio.gather(active, sibling, return_exceptions=True)
    asyncio.run(run())


def test_codex_abort_interrupts_only_target_controller(monkeypatch):
    async def run():
        bridge = make_bridge("codex")
        bridge._cwd = "/tmp"
        redis = SimpleNamespace(xack=AsyncMock())
        controller = SimpleNamespace(signal=object(), abort=MagicMock())
        monkeypatch.setitem(sys.modules, "openai_codex_sdk", SimpleNamespace(
            AbortController=lambda: controller, AbortError=type("AbortError", (Exception,), {})))
        started = asyncio.Event()

        async def stream(*args):
            started.set()
            await asyncio.Event().wait()

        bridge._ensure_codex = lambda: SimpleNamespace(start_thread=lambda _: SimpleNamespace(run_streamed=stream))
        bridge._build_prompt_input = lambda *args, **kwargs: ("hello", [])
        task = asyncio.create_task(send(bridge, "active", redis))
        await started.wait()
        assert not (await abort(bridge, "stale", redis))["aborted"]
        controller.abort.assert_not_called()
        assert (await abort(bridge, "active", redis))["aborted"]
        await asyncio.gather(task, return_exceptions=True)
        controller.abort.assert_called_once_with("task_cancelled")
        bridge._publisher.publish_run_done.assert_called_once_with("active")
        assert not any(call.kwargs.get("kind") == AgentEventKind.ASSISTANT_DONE
                       for call in bridge._publish_event.call_args_list)
    asyncio.run(run())


def test_codex_cancel_during_setup_also_publishes_terminal():
    async def run():
        bridge = make_bridge("codex")
        redis = SimpleNamespace(xack=AsyncMock())
        started = asyncio.Event()

        async def context(bot):
            started.set()
            await asyncio.Event().wait()

        bridge._get_mcp_tool_context = context
        task = asyncio.create_task(send(bridge, "active", redis, system_prompt="system"))
        await started.wait()
        await abort(bridge, "active", redis)
        await asyncio.gather(task, return_exceptions=True)
        bridge._publisher.publish_run_done.assert_called_once_with("active")
        bridge._publish_event.assert_not_called()
        assert [call.args[-1] for call in redis.xack.call_args_list].count("active") == 1
    asyncio.run(run())


def test_local_abort_waits_for_worker_and_does_not_complete_successfully():
    async def run():
        bridge = make_bridge("local")
        redis = SimpleNamespace(xack=AsyncMock())
        started, release, stopped = threading.Event(), threading.Event(), threading.Event()

        class Client:
            def stream_raw(self, messages):
                try:
                    started.set()
                    release.wait(3)
                    yield "late token"
                    raise AssertionError("cancelled producer must not generate another token")
                finally:
                    stopped.set()

        bridge._loader = SimpleNamespace(get_client=lambda *args, **kwargs: Client())
        with ThreadPoolExecutor(max_workers=1) as executor:
            bridge._executor = executor
            task = asyncio.create_task(send(bridge, "active", redis))
            await until(started.is_set)
            await abort(bridge, "active", redis)
            await asyncio.sleep(0)
            await abort(bridge, "active", redis)
            assert bridge._session_queue.is_busy("bot:user")
            bridge._publisher.publish_run_done.assert_not_called()
            release.set()
            await asyncio.gather(task, return_exceptions=True)
        assert stopped.is_set()
        bridge._publisher.publish_run_done.assert_called_once_with("active")
        bridge._publish_event.assert_not_called()
    asyncio.run(run())


class Gateway:
    def __init__(self):
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.abort_started = asyncio.Event()
        self.abort_release = asyncio.Event()
        self.abort_calls = []
        self.rejected = False
        self.no_events = False
        self.send_ack = asyncio.Event()
        self.send_ack.set()

    def clear_session_cancel(self, session):
        pass

    async def send_and_stream(self, session, text, **kwargs):
        self.started.set()
        await self.send_ack.wait()
        kwargs["on_run_started"]("gateway-run")
        if not self.no_events:
            yield {"type": "event", "event": "agent", "payload": {
                "runId": "gateway-run", "sessionKey": session,
                "stream": "assistant", "data": {"delta": "partial"}, "seq": 1}}
        await self.release.wait()

    async def _request(self, method, params):
        self.abort_calls.append((method, params))
        self.abort_started.set()
        await self.abort_release.wait()
        if self.rejected:
            raise RuntimeError("gateway unavailable")
        return {"ok": True, "payload": {"aborted": True}}


def test_openclaw_remote_abort_is_run_scoped_and_late_response_spares_sibling():
    async def run():
        ws = Gateway()
        bridge = make_bridge("openclaw", ws)
        redis = SimpleNamespace(xack=AsyncMock())
        task = asyncio.create_task(send(bridge, "active", redis))
        await until(lambda: bridge._request_runs.get(("bot:user", "active"), {}).get("run_id"))
        rpc = asyncio.create_task(abort(bridge, "active", redis))
        await ws.abort_started.wait()
        duplicate_rpc = asyncio.create_task(abort(bridge, "active", redis))
        await asyncio.sleep(0)
        assert ws.abort_calls == [("chat.abort", {"sessionKey": "bot:user", "runId": "gateway-run"})]
        ws.release.set()
        await task
        sibling_ready = asyncio.Event()
        sibling_release = asyncio.Event()

        async def sibling():
            async with bridge._session_queue.active("bot:user", request_id="sibling"):
                sibling_ready.set()
                await sibling_release.wait()

        sibling_task = asyncio.create_task(sibling())
        await sibling_ready.wait()
        ws.abort_release.set()
        await asyncio.gather(rpc, duplicate_rpc)
        assert len(ws.abort_calls) == 1
        assert not sibling_task.cancelling()
        assert not bridge._session_queue.cancel_event("bot:user").is_set()
        sibling_release.set()
        await sibling_task
    asyncio.run(run())


@pytest.mark.parametrize("unavailable_run_id", [False, True])
def test_openclaw_failed_abort_keeps_remote_stream_running(unavailable_run_id):
    async def run():
        ws = Gateway()
        ws.no_events = unavailable_run_id
        if unavailable_run_id:
            ws.send_ack.clear()
        ws.rejected = True
        ws.abort_release.set()
        bridge = make_bridge("openclaw", ws)
        bridge._abort_run_id_timeout_s = 0.01
        bridge._wait_for_history_reply = AsyncMock(return_value=None)
        redis = SimpleNamespace(xack=AsyncMock())
        task = asyncio.create_task(send(bridge, "active", redis))
        await ws.started.wait()
        result = await abort(bridge, "active", redis)
        assert result["ok"] is False
        assert not task.done() and not task.cancelling()
        bridge._publisher.publish_run_done.assert_not_called()
        if unavailable_run_id:
            assert not ws.abort_calls
        ws.send_ack.set()
        ws.release.set()
        await task
    asyncio.run(run())


def test_openclaw_ack_callback_exposes_run_before_first_event():
    from openclaw_bridge.ws_client import OpenClawWsClient

    async def run():
        ws = OpenClawWsClient.__new__(OpenClawWsClient)
        ws._run_queues = {}
        ws._session_cancel_events = {}
        ws.send_user_message = AsyncMock(return_value="acknowledged-run")
        ready = asyncio.Event()
        runs = []

        def on_run_started(run_id):
            runs.append(run_id)
            ready.set()

        stream = ws.send_and_stream("bot:user", "hello", on_run_started=on_run_started)
        next_event = asyncio.create_task(anext(stream))
        await asyncio.wait_for(ready.wait(), timeout=1)
        assert runs == ["acknowledged-run"]
        assert not next_event.done()
        assert "acknowledged-run" in ws._run_queues
        next_event.cancel()
        await asyncio.gather(next_event, return_exceptions=True)
        assert not ws._run_queues
    asyncio.run(run())


def test_openclaw_abort_waits_for_ack_but_not_first_event():
    async def run():
        ws = Gateway()
        ws.no_events = True
        ws.send_ack.clear()
        ws.abort_release.set()
        bridge = make_bridge("openclaw", ws)
        redis = SimpleNamespace(xack=AsyncMock())
        task = asyncio.create_task(send(bridge, "active", redis))
        await ws.started.wait()
        rpc = asyncio.create_task(abort(bridge, "active", redis))
        await asyncio.sleep(0)
        assert not ws.abort_calls
        assert not task.cancelling()
        ws.send_ack.set()
        result = await asyncio.wait_for(rpc, timeout=1)
        assert result["ok"] and result["aborted"]
        assert result["requestId"] == "active"
        assert result["detail"] == "run_aborted"
        assert ws.abort_calls == [("chat.abort", {"sessionKey": "bot:user", "runId": "gateway-run"})]
        await asyncio.gather(task, return_exceptions=True)
        bridge._publisher.publish_run_done.assert_called_once_with("active")
        bridge._publisher.publish_run_event.assert_not_called()
    asyncio.run(run())


def test_openclaw_successful_abort_finishes_without_assistant_done():
    async def run():
        ws = Gateway()
        ws.abort_release.set()
        bridge = make_bridge("openclaw", ws)
        redis = SimpleNamespace(xack=AsyncMock())
        task = asyncio.create_task(send(bridge, "active", redis))
        await until(lambda: bridge._request_runs.get(("bot:user", "active"), {}).get("run_id"))
        assert (await abort(bridge, "active", redis))["aborted"]
        await asyncio.gather(task, return_exceptions=True)
        bridge._publisher.publish_run_done.assert_called_once_with("active")
        assert not any(call.args[1].kind == AgentEventKind.ASSISTANT_DONE
                       for call in bridge._publisher.publish_run_event.call_args_list)
    asyncio.run(run())
