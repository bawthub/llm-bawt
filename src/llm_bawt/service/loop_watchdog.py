"""Observe the serving loop without scheduling stack capture on that loop."""

from __future__ import annotations

import asyncio
import json
import math
import sys
import threading
import time
from collections import deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from os import write as _write_fd
from pathlib import PurePath
from types import FrameType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import FastAPI

HEARTBEAT_SECONDS = 0.05
STALL_SECONDS = 0.25
WINDOW_SECONDS = 60.0
LOG_INTERVAL_SECONDS = 30.0
LONG_STALL_SECONDS = 1.0
MAX_STACK_FRAMES = 24
MAX_PENDING_LONG_STACKS = 32


def _stack_frames(frame: FrameType | None) -> list[dict[str, str | int]]:
    """Innermost first; never read source files, locals, or request values."""
    frames: list[dict[str, str | int]] = []
    while frame is not None and len(frames) < MAX_STACK_FRAMES:
        code = frame.f_code
        frames.append({
            "module": str(frame.f_globals.get("__name__", "")),
            "file": PurePath(code.co_filename).name,
            "line": frame.f_lineno,
            "function": code.co_name,
        })
        frame = frame.f_back
    return frames


def _emit_warning(lag: float, frames: list[dict[str, str | int]], episode: int, *, long: bool) -> None:
    # One write within Linux PIPE_BUF, without StreamHandler or TextIO locks.
    # The OS sink can still block this daemon, but it cannot hold a logging
    # lock that the serving thread needs. Heartbeat accounting is independent.
    payload = {"stall_episode": episode, "lag_ms_at_capture": round(lag * 1000, 3), "frames": frames}
    if long:
        payload["long_stall"] = True
    while True:
        line = ("WARNING event_loop_stall " + json.dumps(payload, separators=(",", ":")) + "\n").encode()
        if len(line) <= 4096:
            break
        frames.pop()
    try:
        _write_fd(2, line)
    except OSError:
        # No retry and no fallback to the shared logger if stderr is unavailable.
        pass


class LoopStallWatchdog:
    """One monitor per serving loop, with bounded process-local state."""

    def __init__(self, *, heartbeat: float = HEARTBEAT_SECONDS, threshold: float = STALL_SECONDS):
        self._heartbeat_interval = heartbeat
        self._threshold = threshold
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._deadline = 0.0
        self._episode_counted = False
        self._long_captured = False
        self._last_log = -math.inf
        self._output_ready = threading.Condition()
        self._pending_long = deque(maxlen=MAX_PENDING_LONG_STACKS)
        self._pending_short = None
        self._output_closed = False
        self._stall_count = 0
        self._samples: deque[tuple[float, float]] = deque(maxlen=math.ceil(WINDOW_SECONDS / heartbeat) + 2)
        self._loop_thread = 0
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._watcher = threading.Thread(target=self._watch, name="event-loop-watchdog", daemon=True)
        self._writer = threading.Thread(target=self._write_warnings, name="event-loop-watchdog-output", daemon=True)

    def start(self) -> None:
        """Call once, from the serving loop."""
        loop = asyncio.get_running_loop()
        self._loop_thread = threading.get_ident()
        self._deadline = time.monotonic() + self._heartbeat_interval
        self._writer.start()
        self._watcher.start()
        self._heartbeat_task = loop.create_task(self._heartbeat())

    async def stop(self) -> None:
        self._stop.set()
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            await asyncio.gather(self._heartbeat_task, return_exceptions=True)
            await asyncio.to_thread(self._watcher.join, 1.0)
            self._close_output()
            # A log sink can be slow; never wait for it on the serving loop.
            await asyncio.to_thread(self._writer.join, 1.0)

    def _close_output(self) -> None:
        with self._output_ready:
            self._output_closed = True
            self._output_ready.notify()

    def _record_stall(self, lag: float) -> None:
        if lag >= self._threshold and not self._episode_counted:
            self._episode_counted = True
            self._stall_count += 1

    def _beat(self) -> float:
        with self._lock:
            now = time.monotonic()
            lag = max(0, now - self._deadline)
            self._record_stall(lag)
            self._samples.append((now, lag))
            self._episode_counted = False
            self._long_captured = False
            self._deadline = now + self._heartbeat_interval
        return lag

    async def _heartbeat(self) -> None:
        while not self._stop.is_set():
            await asyncio.sleep(max(0, self._deadline - time.monotonic()))
            self._beat()

    def _watch(self) -> None:
        while not self._stop.wait(self._heartbeat_interval):
            self._sample_stall()

    def _sample_stall(self) -> None:
        with self._lock:
            now = time.monotonic()
            lag = now - self._deadline
            if lag < self._threshold:
                return
            first = not self._episode_counted
            self._record_stall(lag)
            long = lag >= LONG_STALL_SECONDS
            if long:
                if self._long_captured:
                    return
                self._long_captured = True
            elif first and now - self._last_log >= LOG_INTERVAL_SECONDS:
                self._last_log = now
            else:
                return
            frames = _stack_frames(sys._current_frames().get(self._loop_thread))
            warning = (lag, frames, self._stall_count)

        # A blocked sink cannot stop sampling. Keep the newest severe captures
        # under sustained output backpressure and prefer them to a short warning.
        with self._output_ready:
            if self._output_closed:
                return
            if long:
                self._pending_long.append(warning)
            else:
                self._pending_short = warning
            self._output_ready.notify()

    def _write_warnings(self) -> None:
        while True:
            with self._output_ready:
                self._output_ready.wait_for(
                    lambda: bool(self._pending_long) or self._pending_short is not None or self._output_closed,
                )
                long = bool(self._pending_long)
                if long:
                    warning = self._pending_long.popleft()
                elif self._pending_short is not None:
                    warning, self._pending_short = self._pending_short, None
                else:
                    return
            # No sink access while holding either monitor lock.
            _emit_warning(*warning, long=long)

    def snapshot(self) -> dict[str, float | int]:
        """Non-destructive rolling window; independent scrape clients cannot erase peaks."""
        with self._lock:
            now = time.monotonic()
            samples = tuple(self._samples)
            count = self._stall_count
            active = max(0, now - self._deadline)
        lags = sorted(lag for at, lag in samples if now - at <= WINDOW_SECONDS)
        p99 = lags[math.ceil(len(lags) * 0.99) - 1] if lags else 0
        return {
            "event_loop_lag_max_ms": round(max(active, lags[-1] if lags else 0) * 1000, 3),
            "event_loop_lag_p99_ms": round(p99 * 1000, 3),
            "event_loop_stalls_total": count,
            "event_loop_window_seconds": int(WINDOW_SECONDS),
        }


@asynccontextmanager
async def monitor_serving_loop(
    app: FastAPI, *, watchdog: LoopStallWatchdog | None = None,
) -> AsyncIterator[None]:
    monitor = watchdog if watchdog is not None else LoopStallWatchdog()
    monitor.start()
    app.state.loop_watchdog = monitor
    try:
        yield
    finally:
        del app.state.loop_watchdog
        await monitor.stop()
