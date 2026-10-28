"""Isolated regressions for Snark's queued-run and buffered-result failures."""
import asyncio
import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
import threading
import time
import uuid
from typing import Any, Iterator

import pytest
from claude_agent_sdk.types import AssistantMessage, ToolResultBlock, ToolUseBlock, UserMessage

from agent_bridge.session_queue import SessionQueue
from claude_code_bridge.turn_watchdog import TurnWatchdog, TurnWatchdogTimeout


def test_buffered_result_after_long_sibling_does_not_false_timeout():
    async def run():
        now = [0.0]
        watchdog = TurnWatchdog(300, clock=lambda: now[0])
        watchdog.observe(AssistantMessage(content=[
            ToolUseBlock(id="agent", name="Agent", input={}),
            ToolUseBlock(id="bash", name="Bash", input={"timeout": 120000}),
        ], model="test"))
        now[0] = 690.0
        watchdog.observe(UserMessage(content=[ToolResultBlock(tool_use_id="agent", content="done")]))

        async def results():
            yield UserMessage(content=[ToolResultBlock(tool_use_id="bash", content="cancelled", is_error=True)])

        await watchdog.receive_next(results(), request_id="request", session_key="snark:nick")
        assert watchdog.active_tools == ()
    asyncio.run(run())


def test_overdue_tool_drain_window_is_bounded():
    async def run():
        watchdog = TurnWatchdog(0.01, tool_grace=0.02, health_interval=0.005)
        watchdog.observe(AssistantMessage(content=[
            ToolUseBlock(id="bash", name="Bash", input={"timeout": 1}),
        ], model="test"))
        await asyncio.sleep(0.03)

        async def silent():
            await asyncio.sleep(10)
            yield UserMessage(content="late")

        with pytest.raises(TurnWatchdogTimeout):
            await asyncio.wait_for(watchdog.receive_next(
                silent(), request_id="request", session_key="snark:nick",
            ), timeout=0.2)
    asyncio.run(run())


def test_cancel_queued_request_preserves_active_and_next_waiter():
    async def run():
        queue = SessionQueue()
        started = asyncio.Event()
        release = asyncio.Event()
        entered = []

        async def active():
            async with queue.active("snark:nick", request_id="active"):
                started.set()
                await release.wait()

        async def waiting(request):
            async with queue.active("snark:nick", request_id=request):
                entered.append(request)

        first = asyncio.create_task(active())
        await started.wait()
        doomed = asyncio.create_task(waiting("doomed"))
        next_task = asyncio.create_task(waiting("next"))
        await asyncio.sleep(0)
        assert not queue.cancel_request("other:nick", "doomed")
        assert queue.cancel_request("snark:nick", "doomed")
        with pytest.raises(asyncio.CancelledError):
            await doomed
        assert not first.done()
        assert not queue.cancel_request("snark:nick", "missing")
        release.set()
        await asyncio.gather(first, next_task)
        assert entered == ["next"]
        assert queue._request_tasks == {}
        assert not queue.is_busy("snark:nick")
    asyncio.run(run())


def test_request_cancel_rpc_never_uses_session_wide_abort():
    # Extract the actual RPC method without importing bridge bootstrap/network.
    source = Path(__file__).parents[1] / "src/claude_code_bridge/command_ops.py"
    tree = ast.parse(source.read_text())
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef) and n.name == "_handle_rpc")
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    scope = {"asyncio": asyncio, "logger": Mock(), "COMMANDS_STREAM": "agent:commands"}
    exec(compile(module, str(source), "exec"), scope)
    owner = SimpleNamespace(_session_queue=Mock(), _publisher=Mock())
    # TASK-1232 P2: chat.cancel stops exactly this request (tombstone + cancel).
    owner._session_queue.stop_request.return_value = True
    redis = SimpleNamespace(xack=AsyncMock())
    asyncio.run(scope["_handle_rpc"](owner, {
        "method": "chat.cancel", "request_id": "rpc-1",
        "params": {"sessionKey": "snark:nick", "requestId": "doomed"},
    }, "entry", redis))
    owner._session_queue.stop_request.assert_called_once_with("snark:nick", "doomed")
    owner._session_queue.cancel_active.assert_not_called()
    owner._session_queue.signal_cancel.assert_not_called()
    owner._publisher.publish_rpc_result.assert_called_once_with("rpc-1", {
        "ok": True, "cancelled": True, "request_id": "doomed",
    })


@pytest.mark.parametrize("backend", ["claude-code", "codex"])
def test_app_timeout_cancels_only_supported_request_and_preserves_error(monkeypatch, backend):
    from agent_bridge import subscriber as transport

    source = Path(__file__).parents[1] / "src/llm_bawt/agent_backends/agent_bridge.py"
    tree = ast.parse(source.read_text())
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "stream_raw")
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    subscriber = SimpleNamespace(_redis=SimpleNamespace(connection_pool=SimpleNamespace(
        connection_kwargs={"url": "redis://test.invalid"},
    )))
    scope = {
        "asyncio": asyncio, "logger": Mock(), "time": time, "uuid": uuid,
        "Iterator": Iterator, "Any": Any, "AgentResult": lambda **kw: SimpleNamespace(**kw),
        "get_agent_subscriber": lambda: subscriber,
        "resolve_seed_framing": lambda messages, _bot_id: None if not messages else pytest.fail("seeded"),
    }
    exec(compile(module, str(source), "exec"), scope)
    local = SimpleNamespace(connect=AsyncMock(), close=AsyncMock(), send_command=AsyncMock(),
                            send_rpc=AsyncMock(return_value={"ok": True, "cancelled": True}))

    async def timeout(*args, **kwargs):
        raise TimeoutError("original inactivity error")
        yield  # async iterator contract

    local.subscribe_run = timeout
    monkeypatch.setattr(transport, "RedisSubscriber", lambda url: local)
    owner = SimpleNamespace(name=backend, _resolve_session_key=lambda config: "snark:nick",
                            _thread_local=threading.local())
    with pytest.raises(TimeoutError, match="original inactivity error"):
        list(scope["stream_raw"](owner, "test", {"request_id": "doomed"}))
    if backend == "claude-code":
        local.send_rpc.assert_awaited_once_with(
            "chat.cancel", {"sessionKey": "snark:nick", "requestId": "doomed"},
            request_id="cancel_doomed", backend="claude-code",
        )
    else:
        local.send_rpc.assert_not_awaited()
    local.close.assert_awaited_once()
