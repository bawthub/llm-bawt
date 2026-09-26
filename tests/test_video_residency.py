"""Resident worker supervisor serializes commands and fails closed on ambiguity."""
import asyncio
import json

import pytest

from local_model_bridge.video_residency import ResidentVideoWorker, WorkerOutcomeUnknown


class FakeStream:
    def __init__(self):
        self.commands = []

    def write(self, data):
        self.commands.append(json.loads(data))

    async def drain(self):
        pass


class FakeReader:
    def __init__(self, process):
        self.process = process

    async def readline(self):
        action = self.process.stdin.commands[-1]["action"]
        if action == "unload":
            return b'{"ok":true,"resident":false}\n'
        with open(self.process.stdin.commands[-1]["output"], "wb") as output:
            output.write(b"video")
        return b'{"ok":true,"resident":true,"metadata":{"width":832}}\n'


class FakeProcess:
    def __init__(self):
        self.stdin = FakeStream()
        self.stdout = FakeReader(self)
        self.returncode = None

    async def wait(self):
        self.returncode = 0
        return 0


def test_two_renders_share_child_and_unload_exits(tmp_path):
    processes = []

    async def startup(*args, **kwargs):
        processes.append(FakeProcess())
        return processes[-1]

    async def exercise():
        worker = ResidentVideoWorker(startup=startup)
        results = await asyncio.gather(
            worker.render(tmp_path / "one.json", tmp_path / "one.mp4"),
            worker.render(tmp_path / "two.json", tmp_path / "two.mp4"),
        )
        assert results == [{"width": 832}] * 2
        assert worker.resident
        assert len(processes) == 1
        assert [cmd["action"] for cmd in processes[0].stdin.commands] == ["render", "render"]
        await worker.unload()
        assert not worker.resident and not worker.running
        assert [cmd["action"] for cmd in processes[0].stdin.commands] == ["render", "render", "unload"]

    asyncio.run(exercise())


def test_missing_reply_requires_reconciliation(tmp_path):
    class BrokenReader:
        async def readline(self):
            return b""

    async def startup(*args, **kwargs):
        process = FakeProcess()
        process.stdout = BrokenReader()
        return process

    async def exercise():
        worker = ResidentVideoWorker(startup=startup)
        with pytest.raises(RuntimeError, match="bounded response"):
            await worker.render(tmp_path / "one.json", tmp_path / "one.mp4")
        assert worker.uncertain
        with pytest.raises(WorkerOutcomeUnknown):
            await worker.render(tmp_path / "two.json", tmp_path / "two.mp4")
        with pytest.raises(WorkerOutcomeUnknown):
            await worker.unload()

    asyncio.run(exercise())


def test_unexpected_idle_exit_is_not_restarted(tmp_path):
    processes = []

    async def startup(*args, **kwargs):
        processes.append(FakeProcess())
        return processes[-1]

    async def exercise():
        worker = ResidentVideoWorker(startup=startup)
        await worker.render(tmp_path / "one.json", tmp_path / "one.mp4")
        processes[0].returncode = 1
        assert worker.uncertain
        with pytest.raises(WorkerOutcomeUnknown):
            await worker.render(tmp_path / "two.json", tmp_path / "two.mp4")
        assert len(processes) == 1

    asyncio.run(exercise())


def test_real_child_status_and_unload_without_loading_gpu():
    async def exercise():
        worker = ResidentVideoWorker(timeout=5)
        # Status launches the child but does not import CUDA or load weights.
        status = await worker._command("status")
        assert status == {"ok": True, "resident": False}
        assert worker.running and not worker.resident
        await worker.unload()
        assert not worker.running and not worker.uncertain

    asyncio.run(exercise())


def test_cancelled_spawn_is_not_retried(tmp_path):
    calls = []

    async def cancelled(*args, **kwargs):
        calls.append(True)
        raise asyncio.CancelledError()

    async def exercise():
        worker = ResidentVideoWorker(startup=cancelled)
        with pytest.raises(asyncio.CancelledError):
            await worker.render(tmp_path / "one.json", tmp_path / "one.mp4")
        assert worker.uncertain
        with pytest.raises(WorkerOutcomeUnknown):
            await worker.render(tmp_path / "two.json", tmp_path / "two.mp4")
        assert len(calls) == 1

    asyncio.run(exercise())
