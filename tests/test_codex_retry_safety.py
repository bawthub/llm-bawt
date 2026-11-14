"""No native Codex prompt replay after tools may have run (TASK-2277).

Actual send/event paths; all SDK/JSON-RPC I/O is in-memory, with no subprocess.
"""
from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_bridge.events import AgentEventKind
from codex_bridge.command_ops import CodexCommandMixin
from codex_bridge.event_ops import CodexEventMixin
from tests.test_other_bridge_abort_targeting import make_bridge, send


class Bridge(CodexCommandMixin, CodexEventMixin):
    pass


def setup_bridge(monkeypatch):
    bridge = make_bridge("codex")
    bridge.__class__ = Bridge
    bridge._cwd = "/unused"
    bridge._request_timeout = 30
    bridge._build_prompt_input = lambda *args, **kwargs: ("mutate once", [])
    bridge._rollout_usage = SimpleNamespace(resolve=AsyncMock(return_value=None))
    monkeypatch.setitem(sys.modules, "openai_codex_sdk", SimpleNamespace(
        AbortController=lambda: SimpleNamespace(signal=object(), abort=MagicMock()),
        AbortError=type("AbortError", (Exception,), {}),
    ))
    return bridge


class FakeClient:
    def __init__(self, state, *, fail_at, item_type, error):
        self.state = state
        self.fail_at = fail_at
        self.item_type = item_type
        self.error = error
        self.noted = False
        self._close_task = None

    async def request(self, method, params, **kwargs):
        if method in ("thread/start", "thread/resume"):
            self.state["opens"] += 1
            if self.fail_at == "open" and self.state["opens"] == 1:
                raise self.error
            return {"thread": {"id": "thread-1"}}
        assert method == "turn/start"
        self.state["submissions"] += 1
        if self.fail_at == "submit":
            self.state["effects"] += 1  # request can execute before the response is lost
            raise self.error
        return {"turn": {"id": "turn-1"}}

    async def next_notification(self, timeout):
        if self.fail_at == "open":
            return {"method": "turn/completed", "params": {"turn": {"id": "turn-1", "status": "completed"}}}
        if not self.noted:
            self.noted = True
            self.state["effects"] += 1
            return {"method": "item/completed", "params": {"item": {
                "id": "tool-1", "type": self.item_type, "status": "completed",
                "command": "fake mutation", "server": "fake", "tool": "mutate",
                "arguments": {}, "result": {},
            }}}
        raise self.error

    def add_observer(self, observer):
        pass

    def close(self):
        if self._close_task is None:
            self._close_task = asyncio.create_task(asyncio.sleep(0, result="no process"))
        return self._close_task


async def exercise(monkeypatch, transport, *, fail_at="tool", item_type="command_execution", error=None):
    bridge = setup_bridge(monkeypatch)
    state = {"opens": 0, "submissions": 0, "effects": 0}
    error = error or TimeoutError("stream stalled")

    async def run_streamed(*args):
        state["submissions"] += 1

        async def events():
            if fail_at == "silent":
                state["effects"] += 1  # lost events are not proof of no side effects
                raise error
            state["effects"] += 1
            yield SimpleNamespace(type="item.completed", item={
                "id": "tool-1", "type": item_type, "status": "completed",
                "command": "fake mutation", "server": "fake", "tool": "mutate",
                "arguments": {}, "result": {},
            })
            raise error
        return SimpleNamespace(events=events())

    codex = SimpleNamespace(
        start_thread=lambda *args: SimpleNamespace(id="thread-1", run_streamed=run_streamed),
        resume_thread=lambda *args: SimpleNamespace(id="thread-1", run_streamed=run_streamed),
    )
    bridge._ensure_codex = lambda: codex

    async def open_client(*args):
        return FakeClient(state, fail_at=fail_at, item_type=item_type, error=error)
    bridge._open_app_server = open_client
    await send(bridge, "request-1", SimpleNamespace(xack=AsyncMock()),
               codex_transport=transport, thread_resume_id="old-thread")
    return bridge, state


@pytest.mark.parametrize("transport,item_type", [
    ("exec", "command_execution"), ("exec", "mcp_tool_call"),
    ("app_server", "commandExecution"), ("app_server", "mcpToolCall"),
])
def test_completed_tool_then_timeout_never_replays(monkeypatch, transport, item_type):
    bridge, state = asyncio.run(exercise(monkeypatch, transport, item_type=item_type))
    assert state["effects"] == 1
    assert state["submissions"] == 1
    assert any(c.kwargs.get("kind") == AgentEventKind.ERROR for c in bridge._publish_event.call_args_list)
    bridge._publisher.publish_run_done.assert_called_once()


def test_exec_timeout_with_no_events_is_not_safe_to_replay(monkeypatch):
    _, state = asyncio.run(exercise(monkeypatch, "exec", fail_at="silent"))
    assert state["effects"] == state["submissions"] == 1


def test_app_server_turn_start_timeout_is_not_safe_to_replay(monkeypatch):
    _, state = asyncio.run(exercise(monkeypatch, "app_server", fail_at="submit"))
    assert state["effects"] == state["submissions"] == 1


@pytest.mark.parametrize("error", [TimeoutError("resume stalled"), RuntimeError("thread not found")])
def test_app_server_pre_input_resume_failure_still_retries_once(monkeypatch, error):
    bridge, state = asyncio.run(exercise(monkeypatch, "app_server", fail_at="open", error=error))
    assert state == {"opens": 2, "submissions": 1, "effects": 0}
    assert not any(c.kwargs.get("kind") == AgentEventKind.ERROR for c in bridge._publish_event.call_args_list)


def test_exec_known_session_rejection_before_activity_still_retries_once(monkeypatch):
    bridge = setup_bridge(monkeypatch)
    submissions = []

    async def run_streamed(prompt, options):
        submissions.append(prompt)
        if len(submissions) == 1:
            raise RuntimeError("thread not found")

        async def events():
            yield SimpleNamespace(type="turn.completed", usage=None)
        return SimpleNamespace(events=events())

    thread = SimpleNamespace(id="thread-1", run_streamed=run_streamed)
    codex = SimpleNamespace(start_thread=MagicMock(return_value=thread),
                            resume_thread=MagicMock(return_value=thread))
    bridge._ensure_codex = lambda: codex
    asyncio.run(send(bridge, "request-1", SimpleNamespace(xack=AsyncMock()),
                     thread_resume_id="missing-thread"))
    assert len(submissions) == 2
    codex.resume_thread.assert_called_once()
    codex.start_thread.assert_called_once()
    assert not any(c.kwargs.get("kind") == AgentEventKind.ERROR for c in bridge._publish_event.call_args_list)


@pytest.mark.parametrize("transport", ["exec", "app_server"])
def test_session_error_after_tool_is_not_safe_to_replay(monkeypatch, transport):
    _, state = asyncio.run(exercise(monkeypatch, transport, error=RuntimeError("encrypted_content invalid")))
    assert state["effects"] == state["submissions"] == 1
