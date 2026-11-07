"""What a house's preview is serving, published for the Campus view (TASK-2131).

``house-preview <worktree>/frontend`` runs Vite on port 5173 inside the house
container. The house bridge shares that container, so each heartbeat it looks
for the Vite process in ``/proc``, takes its working directory as the checkout,
reads the branch and commit with git and checks that the port answers. Nothing
is launched or written by the preview itself, so this needs no image change.

The result goes to ``agent:house:<H>:preview`` as JSON, TTL :data:`TTL_S`. The
key is written only when the value changes and its TTL extended otherwise, so
the app's keyspace watcher sees one ``set`` per change; a stopped preview
deletes it and a dead house lets it expire.

Value: ``{"up", "checkout", "repo", "branch", "commit", "task"}``. ``up`` is
false while Vite is starting or not answering. ``task`` is the ``TASK-N`` named
by the branch or checkout path, else null.
"""

from __future__ import annotations

import json
import logging
import os
import re
import socket
import subprocess
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PREVIEW_PORT = 5173
TTL_S = 30
TASK_RE = re.compile(r"TASK-\d+", re.IGNORECASE)
_GIT_TIMEOUT_S = 2.0


def preview_key(house: str) -> str:
    return f"agent:house:{house}:preview"


def _is_vite(argv: list[str]) -> bool:
    """The preview's Vite server, not the shell or ``concurrently`` around it."""
    if not argv or argv[0].endswith(("sh", "concurrently")):
        return False
    joined = " ".join(argv)
    return ("vite" in joined and str(PREVIEW_PORT) in joined
            and "concurrently" not in joined)


def find_preview_cwd(proc: Path = Path("/proc")) -> Path | None:
    """The working directory of the preview's Vite process, if one is running."""
    try:
        entries = list(proc.iterdir())
    except OSError:
        return None
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        argv = [part.decode("utf-8", "replace") for part in raw.split(b"\0") if part]
        if not _is_vite(argv):
            continue
        try:
            return Path(os.readlink(entry / "cwd"))
        except OSError:
            continue
    return None


def _port_open(port: int = PREVIEW_PORT) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
            return True
    except OSError:
        return False


def _git(cwd: Path, *args: str) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(cwd), *args], capture_output=True,
                             text=True, timeout=_GIT_TIMEOUT_S, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    value = out.stdout.strip()
    return value if out.returncode == 0 and value else None


def _repo_name(cwd: Path) -> str | None:
    """The repository a (worktree) checkout belongs to, from its common git dir."""
    common = _git(cwd, "rev-parse", "--path-format=absolute", "--git-common-dir")
    if not common:
        return None
    path = Path(common)
    return (path.parent if path.name == ".git" else path).name or None


def describe(cwd: Path, up: bool) -> dict:
    """The published value for a preview served from ``cwd``."""
    checkout = _git(cwd, "rev-parse", "--show-toplevel") or str(cwd)
    branch = _git(cwd, "rev-parse", "--abbrev-ref", "HEAD")
    if branch == "HEAD":  # detached
        branch = None
    match = TASK_RE.search(branch or "") or TASK_RE.search(checkout)
    return {
        "up": up, "checkout": checkout, "repo": _repo_name(cwd), "branch": branch,
        "commit": _git(cwd, "rev-parse", "--short", "HEAD"),
        "task": match.group(0).upper() if match else None,
    }


def probe(proc: Path = Path("/proc")) -> dict | None:
    """This container's preview now, or None when no preview is running."""
    cwd = find_preview_cwd(proc)
    if cwd is None:
        return None
    return describe(cwd, _port_open())


class PreviewPublisher:
    """Keeps ``agent:house:<H>:preview`` in step with :func:`probe`."""

    def __init__(self, house: str, probe_fn=probe) -> None:
        self.house = house
        self._probe = probe_fn
        self._last: str | None = None

    async def publish(self, redis: Any) -> None:
        import asyncio

        value = await asyncio.to_thread(self._probe)
        key = preview_key(self.house)
        if value is None:
            if self._last is not None:
                await redis.delete(key)
                self._last = None
            return
        rendered = json.dumps(value, sort_keys=True)
        # Rewrite on change, or when the key lapsed (e.g. Redis restarted).
        if rendered != self._last or not await redis.expire(key, TTL_S):
            await redis.set(key, rendered, ex=TTL_S)
            if rendered != self._last:
                logger.info("house %s preview: %s %s@%s", self.house,
                            "up" if value["up"] else "starting",
                            value.get("branch"), value.get("commit"))
            self._last = rendered

    async def clear(self, redis: Any) -> None:
        if self._last is not None:
            await redis.delete(preview_key(self.house))
            self._last = None


__all__ = ["PREVIEW_PORT", "TTL_S", "PreviewPublisher", "describe", "find_preview_cwd",
           "preview_key", "probe"]
