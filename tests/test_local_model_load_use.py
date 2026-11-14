"""TASK-2278: a local model stays loaded from a send's load to its generation.

Load and generation are separate jobs on the single inference worker. Without
a lifetime lock, session B's load can queue between A's load and A's
generation, unload A's model, and fail A's turn.
"""
from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import AsyncMock, Mock

from agent_bridge.events import AgentEventKind
from agent_bridge.session_queue import SessionQueue
from local_model_bridge.bridge import LocalModelBridge


class _Client:
    def __init__(self, alias):
        self.alias = alias
        self.loaded = True

    def stream_raw(self, _messages):
        if not self.loaded:
            raise RuntimeError(f"model {self.alias} already unloaded")
        yield f"hello from {self.alias}"


class _Loader:
    """One model at a time, like LocalModelLoader; load of A waits on a gate."""

    def __init__(self):
        self.current: _Client | None = None
        self.gate_a = threading.Event()
        self.loading_a = threading.Event()
        self.unloads: list[str] = []

    def get_client(self, alias, *, fallback_definition=None):
        if self.current is not None and self.current.alias == alias:
            return self.current
        if alias == "A":
            self.loading_a.set()
            assert self.gate_a.wait(5), "test gate never released"
        self.unload()
        self.current = _Client(alias)
        return self.current

    def unload(self):
        if self.current is None:
            return False
        self.current.loaded = False
        self.unloads.append(self.current.alias)
        self.current = None
        return True


def _bridge(loader):
    bridge = LocalModelBridge.__new__(LocalModelBridge)
    bridge._publisher = Mock()
    bridge._backend_name = "local"
    bridge._request_timeout = 30
    bridge._command_task = None
    bridge._session_queue = SessionQueue()
    bridge._loader = loader
    bridge._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="test-infer")
    bridge._trigger_message_ids = {}
    bridge._run_workers = {}
    bridge._model_lock = asyncio.Lock()
    bridge._stop_task = None
    return bridge


def _send(bridge, request_id, session_key, alias):
    redis = Mock(xack=AsyncMock())
    fields = {"request_id": request_id, "session_key": session_key,
              "message": "hi", "model": alias}
    return asyncio.create_task(bridge._run_send(fields, f"msg-{request_id}", redis))


def _events(bridge, request_id):
    return [call.args[1] for call in bridge._publisher.publish_run_event.call_args_list
            if call.args[0] == request_id]


def test_other_session_load_cannot_unload_a_model_before_its_generation():
    async def run():
        loader = _Loader()
        bridge = _bridge(loader)
        a = _send(bridge, "ra", "bot-a:nick", "A")
        await asyncio.to_thread(loader.loading_a.wait, 5)
        b = _send(bridge, "rb", "bot-b:nick", "B")
        for _ in range(20):  # let B reach its load (or the model lock)
            await asyncio.sleep(0)
        loader.gate_a.set()
        await asyncio.wait_for(asyncio.gather(a, b), 10)
        bridge._executor.shutdown(wait=True)
        return bridge, loader

    bridge, loader = asyncio.run(run())
    for request_id, alias in (("ra", "A"), ("rb", "B")):
        kinds = [e.kind for e in _events(bridge, request_id)]
        assert AgentEventKind.ERROR not in kinds, (request_id, _events(bridge, request_id))
        done = [e for e in _events(bridge, request_id) if e.kind == AgentEventKind.ASSISTANT_DONE]
        assert [e.text for e in done] == [f"hello from {alias}"]
    assert loader.unloads == ["A"]  # switched to B only after A finished


def test_stop_does_not_unload_under_a_running_generation():
    async def run():
        loader = _Loader()
        bridge = _bridge(loader)
        bridge.SHUTDOWN_UNLOAD_WAIT = 0.05
        a = _send(bridge, "ra", "bot-a:nick", "A")
        await asyncio.to_thread(loader.loading_a.wait, 5)
        await bridge.stop()  # A holds the model: no unload
        unloads_at_stop = list(loader.unloads)
        loader.gate_a.set()
        # The executor is shut down, so A ends with an error, but never one
        # from generating against an unloaded model.
        await asyncio.wait_for(a, 10)
        return bridge, loader, unloads_at_stop

    bridge, loader, unloads_at_stop = asyncio.run(run())
    assert unloads_at_stop == [] and loader.unloads == []
    assert not any("unloaded" in (e.text or "") for e in _events(bridge, "ra"))


def test_stop_unloads_on_the_worker_when_idle():
    async def run():
        loader = _Loader()
        loader.current = _Client("B")
        bridge = _bridge(loader)
        await bridge.stop()
        return loader, bridge

    loader, bridge = asyncio.run(run())
    assert loader.unloads == ["B"]
    assert bridge._model_lock.locked()  # no send can load after stop


def test_repeated_and_concurrent_stop_share_one_shutdown(caplog):
    async def run():
        loader = _Loader()
        loader.current = _Client("B")
        bridge = _bridge(loader)
        # Short wait: a caller contending for the retained lock would time out
        # and log the running-generation warning instead of sharing shutdown.
        bridge.SHUTDOWN_UNLOAD_WAIT = 0.05
        await asyncio.gather(bridge.stop(), bridge.stop())  # run_forever finally + __main__
        await bridge.stop()  # a later caller
        return loader, bridge

    with caplog.at_level("WARNING"):
        loader, bridge = asyncio.run(run())
    assert loader.unloads == ["B"]
    assert "Generation still running" not in caplog.text
    assert bridge._model_lock.locked()
    assert bridge._publisher.close.call_count == 1
