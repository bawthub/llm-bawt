"""Temporary TASK-1940 one-call evidence capture; inert without a private arm file.

This observes decoded Responses argument strings, NOT raw HTTP/SSE bytes.
No request bodies, auth, general logging, DB settings or argument repair belong here.
See docs/RELAY_ARGUMENT_TRACE.md for the separately authorized activation boundary.
"""
from __future__ import annotations

import asyncio
import json
import math
import os
from pathlib import Path
import stat
import time

from .request_context import ProxyRequestContext

TOOL_NAME = "mcp__bawthub__bots_send_message"
MAX_BYTES = 65_536
MAX_EVENTS = 256
MAX_TTL = 300
_FOOTER_RESERVE = 512


def _directory() -> Path:
    # Deliberately no env/config switch and no directory creation on the hot path.
    return Path.home() / ".llm-bawt-relay-trace"


def _private(info: os.stat_result, mode: int, *, directory: bool = False) -> bool:
    kind = stat.S_ISDIR if directory else stat.S_ISREG
    return kind(info.st_mode) and info.st_uid == os.getuid() and stat.S_IMODE(info.st_mode) == mode


class RelayTrace:
    """Bounded private JSONL, one selected item; failures never affect translation."""

    def __init__(self, fd: int, item_id: str, expires_at: float) -> None:
        self.fd: int | None = fd
        self.item_id = item_id
        self.expires_at = expires_at
        self.bytes_written = 0
        self.events_written = 0
        self.timer = asyncio.get_running_loop().call_later(
            max(0, expires_at - time.time()), self.finish, "expired",
        )

    def _write(self, data: bytes) -> None:
        while data:
            count = os.write(self.fd, data)
            if count <= 0:
                raise OSError("short diagnostic write")
            self.bytes_written += count
            data = data[count:]

    def record(self, stage: str, **fields: object) -> None:
        if self.fd is None:
            return
        try:
            if time.time() >= self.expires_at:
                self.finish("expired")
                return
            # Only known scalar fields: never recursively serialize upstream objects.
            if any(value is not None and not isinstance(value, (str, int, float))
                   for value in fields.values()):
                self.finish("invalid_value")
                return
            # Reject an oversized string before making an escaped serialization.
            if any(isinstance(value, str) and len(value) > MAX_BYTES for value in fields.values()):
                self.finish("byte_limit")
                return
            data = (json.dumps({"stage": stage, **fields}, ensure_ascii=True) + "\n").encode("ascii")
            if self.events_written >= MAX_EVENTS - 1:
                self.finish("event_limit")
            elif self.bytes_written + len(data) > MAX_BYTES - _FOOTER_RESERVE:
                self.finish("byte_limit")
            else:
                self._write(data)
                self.events_written += 1
        except (OSError, TypeError, ValueError):
            self.finish("write_error")

    def finish(self, status: str) -> None:
        if self.fd is None:
            return
        self.timer.cancel()
        try:
            footer = (json.dumps({"stage": "end", "status": status}) + "\n").encode("ascii")
            if self.bytes_written + len(footer) <= MAX_BYTES:
                self._write(footer)
        except OSError:
            pass  # Missing/invalid footer is explicitly NOT successful evidence.
        finally:
            fd, self.fd = self.fd, None
            try:
                os.close(fd)
            except OSError:
                pass


def claim_relay_trace(
    context: ProxyRequestContext | None, block: dict, message_id: str,
) -> RelayTrace | None:
    """Atomically claim the sole artifact, after exact bot AND tool selection.

    Fixed O_EXCL artifact is also the cross-request/process/restart spent latch.
    Only an operator can deliberately remove/archive it to allow another capture.
    """
    if context is None or not context.bot_id or block.get("name") != TOOL_NAME:
        return None
    directory_fd = arm_fd = output_fd = None
    trace = None
    try:
        directory_fd = os.open(_directory(), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        if not _private(os.fstat(directory_fd), 0o700, directory=True):
            return None
        arm_fd = os.open("arm.json", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory_fd)
        info = os.fstat(arm_fd)
        if not _private(info, 0o600) or info.st_size > 1024 or info.st_nlink != 1:
            return None
        arm = json.loads(os.read(arm_fd, 1025))
        if not isinstance(arm, dict) or arm.get("bot_id") != context.bot_id:
            return None
        armed_at, expires_at = arm.get("armed_at"), arm.get("expires_at")
        now = time.time()
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(value) for value in (armed_at, expires_at)):
            return None
        if not armed_at <= now < expires_at <= armed_at + MAX_TTL:
            return None
        correlation = {
            "bot_id": context.bot_id, "request_id": context.request_id,
            "attempt": context.attempt, "message_id": message_id,
            "item_id": block.get("item_id"), "call_id": block.get("call_id"),
            "block_index": block.get("index"),
        }
        if any(not isinstance(correlation[key], str) or not 0 < len(correlation[key]) <= 256
               for key in ("bot_id", "request_id", "message_id", "item_id", "call_id")):
            return None
        output_fd = os.open(
            "capture.jsonl", os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600, dir_fd=directory_fd,
        )
        # No claim can win after O_EXCL, even if unlink or subsequent I/O fails.
        os.unlink("arm.json", dir_fd=directory_fd)
        trace = RelayTrace(output_fd, block["item_id"], expires_at)
        output_fd = None  # ownership transferred to trace
        trace.record("start", version=1, expires_at=expires_at, **correlation)
        return trace
    except (OSError, ValueError, TypeError, RuntimeError, OverflowError):
        if trace is not None:
            trace.finish("setup_error")
        return None
    finally:
        for fd in (output_fd, arm_fd, directory_fd):
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
