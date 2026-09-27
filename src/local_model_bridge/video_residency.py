"""Private resident Wan subprocess supervisor; does not authorize GPU ownership.

The caller must hold the durable video claim and ensure the handoff is complete
before invoking a render. This class only serializes the child protocol and
keeps one worker alive across successful requests.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path


class WorkerOutcomeUnknown(RuntimeError):
    """The child may still own GPU allocations or have completed its work."""


class WorkerBusy(RuntimeError):
    """A render command is in flight; recovery must wait for its outcome."""


class ResidentVideoWorker:
    def __init__(self, *, startup=None, timeout: float = 3600):
        self._startup = startup or asyncio.create_subprocess_exec
        self._timeout = timeout
        self._lock = asyncio.Lock()
        self._process = None
        self._resident = False
        self._uncertain = False

    @property
    def resident(self) -> bool:
        return self._resident and self._process is not None and self._process.returncode is None

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.returncode is None

    @property
    def uncertain(self) -> bool:
        return self._uncertain or (self._process is not None and self._process.returncode is not None)

    async def _ensure_process(self):
        if self.uncertain:
            self._uncertain = True
            self._resident = False
            raise WorkerOutcomeUnknown("Wan worker outcome requires reconciliation")
        if self._process is None:
            # stderr never pipes into the protocol stream or blocks a full pipe;
            # a bounded worker error is returned on stdout instead.
            try:
                self._process = await self._startup(
                    sys.executable, "-m", "local_model_bridge.video_worker", "--resident",
                    stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL,
                )
            except BaseException:
                # A cancelled spawn could have created a GPU-owning child.
                self._uncertain = True
                raise
        return self._process

    async def _command(self, action: str, **payload) -> dict:
        process = await self._ensure_process()
        try:
            if process.stdin is None or process.stdout is None:
                raise RuntimeError("Wan worker has no control pipes")
            process.stdin.write((json.dumps({"action": action, **payload}) + "\n").encode())
            await asyncio.wait_for(process.stdin.drain(), timeout=5)
            reply = await asyncio.wait_for(process.stdout.readline(), timeout=self._timeout)
            if not reply or len(reply) > 8192:
                raise RuntimeError("Wan worker returned no bounded response")
            result = json.loads(reply)
            if not isinstance(result, dict) or type(result.get("ok")) is not bool or type(result.get("resident")) is not bool:
                raise RuntimeError("Wan worker returned an invalid response")
            self._resident = result["resident"]
            return result
        except BaseException:
            # A timeout, cancellation, broken pipe or malformed reply does not
            # prove the child has stopped or that output is not still being written.
            self._uncertain = True
            self._resident = False
            raise

    async def render(self, job: Path, output: Path) -> dict:
        async with self._lock:
            result = await self._command("render", job=str(job), output=str(output))
            if not result["ok"]:
                # The worker exits after any failed command. Never reuse it.
                self._uncertain = True
                detail = result.get("error") if isinstance(result.get("error"), str) else "no worker detail"
                raise WorkerOutcomeUnknown(f"Wan render failed: {detail[-600:]}")
            if (not self.resident or not isinstance(result.get("metadata"), dict)
                    or not output.is_file() or output.stat().st_size == 0):
                self._uncertain = True
                raise WorkerOutcomeUnknown("Wan worker did not confirm a resident render and output")
            return result["metadata"]

    async def reset(self) -> None:
        """Operator recovery: end the child (and every CUDA allocation it holds).

        Refuses while a command is in flight. Terminating the process is the
        only reliable release after an uncertain outcome; weights stay on disk.
        """
        if self._lock.locked():
            raise WorkerBusy("A Wan render is in progress")
        async with self._lock:
            process = self._process
            if process is not None and process.returncode is None:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=30)
                except TimeoutError:
                    process.kill()
                    await asyncio.wait_for(process.wait(), timeout=10)
            self._process = None
            self._resident = False
            self._uncertain = False

    async def unload(self) -> None:
        """Release the child after all queued work drains; never delete weights."""
        async with self._lock:
            if self.uncertain:
                raise WorkerOutcomeUnknown("Wan worker needs reconciliation before unload")
            if self._process is None:
                return
            process = self._process
            if process.returncode is not None:
                self._process = None
                self._resident = False
                return
            result = await self._command("unload")
            if not result["ok"] or result["resident"]:
                self._uncertain = True
                raise WorkerOutcomeUnknown("Wan worker did not confirm unload")
            try:
                await asyncio.wait_for(process.wait(), timeout=30)
            except BaseException:
                self._uncertain = True
                raise
            if process.returncode != 0:
                self._uncertain = True
                raise WorkerOutcomeUnknown("Wan worker exited abnormally after unload")
            self._process = None
            self._resident = False
