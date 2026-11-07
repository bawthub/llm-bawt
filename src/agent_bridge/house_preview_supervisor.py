"""The house runs its occupant's preview itself (TASK-2135).

A house bridge keeps port 5173 answering at all times:

* House empty, or no checkout to serve: a small built-in page says so (who is
  in the house and how a preview starts), instead of the proxy's Bad Gateway.
* Occupied: it installs the checkout's frontend dependencies from its lockfile
  into the shared pnpm store, generates the Prisma client, then runs the
  normal Vite + Hono pair there. The page stays up while that installs and
  comes back if the preview exits; a failed start is retried with backoff.

Because the bridge does this from its own startup, a preview comes back by
itself after any house or container restart.

Which checkout: the one the occupant chose with ``use``, else the occupant's
most recently used ``<dev>/.worktrees/.wt-bawthub-<bot>-TASK-N`` worktree. A
choice is kept in Redis at ``agent:house:<H>:preview_target`` as
``{"bot", "path"}`` and only counts while that bot's profile still names this
house, checked on every tick: a new occupant never inherits it, even when the
old one left and the new one arrived between two ticks. An empty house clears
it. Occupancy is read from the app's ``/v1/campus``; if the app can't be read
the current state is kept.

Each preview step runs in its own process group. Stopping it ends the whole
group (TERM, then KILL after a grace), even when the launcher already exited,
so no leftover server holds the preview ports.

Nothing here runs installs into shared checkouts, migrations, schema pushes or
a per-house backend: Hono is part of the frontend, everything else is shared.

CLI, inside a house::

    python -m agent_bridge.house_preview_supervisor use <checkout>/frontend
    python -m agent_bridge.house_preview_supervisor clear
"""

from __future__ import annotations

import argparse
import asyncio
import html
import json
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .house_preview import PREVIEW_PORT

logger = logging.getLogger(__name__)

DEV_ROOT = Path(os.environ.get("BAWT_HOUSE_DEV_ROOT", "/home/bridge/dev"))
PNPM_STORE = DEV_ROOT / ".pnpm-store"
LOG_PATH = Path("/tmp/house-preview.log")
CHECK_S = 10.0
RETRY_S = (15.0, 60.0, 300.0)
STOP_GRACE_S = 8.0
INSTALL_TIMEOUT_S = 900.0
#: The normal frontend pair, as ``docker/house-preview`` runs it.
PAIR = ["pnpm", "exec", "concurrently", "--kill-others",
        f"vite --host 0.0.0.0 --port {PREVIEW_PORT} --strictPort",
        "tsx watch server/index.ts"]


def target_key(house: str) -> str:
    return f"agent:house:{house}:preview_target"


def is_frontend(path: Path) -> bool:
    return path.is_absolute() and (path / "package.json").is_file()


def _index_mtime(worktree: Path) -> float:
    """When the worktree was last used through git (its index), else its own mtime."""
    try:
        text = (worktree / ".git").read_text().strip()
        if text.startswith("gitdir:"):
            return (Path(text[7:].strip()) / "index").stat().st_mtime
    except OSError:
        pass
    try:
        return worktree.stat().st_mtime
    except OSError:
        return 0.0


def newest_worktree(dev_root: Path, bots: list[str]) -> Path | None:
    """The occupant's most recently used BawtHub task worktree's frontend."""
    best: tuple[float, Path] | None = None
    for bot in bots:
        for wt in (dev_root / ".worktrees").glob(f".wt-bawthub-{bot}-TASK-*"):
            front = wt / "frontend"
            if not is_frontend(front):
                continue
            when = _index_mtime(wt)
            if best is None or when > best[0]:
                best = (when, front)
    return best[1] if best else None


@dataclass(frozen=True)
class Occupancy:
    tenants: tuple[str, ...]
    hidden: int

    @property
    def occupied(self) -> bool:
        return bool(self.tenants) or self.hidden > 0


def occupancy_from_campus(snapshot: dict, house: str) -> Occupancy | None:
    for row in snapshot.get("houses") or []:
        if row.get("id") == house:
            return Occupancy(tuple(row.get("tenants") or ()),
                             int(row.get("hidden_tenants") or 0))
    return Occupancy((), 0) if isinstance(snapshot.get("houses"), list) else None


def parse_choice(raw: Any) -> tuple[str, str] | None:
    """``(bot, path)`` from the stored value; anything else is no choice."""
    if isinstance(raw, bytes):
        raw = raw.decode(errors="replace")
    try:
        value = json.loads(raw) if raw else None
    except ValueError:
        return None
    if (isinstance(value, dict) and isinstance(value.get("bot"), str)
            and isinstance(value.get("path"), str) and value["bot"]):
        return value["bot"], value["path"]
    return None


def worktree_bot(front: Path) -> str | None:
    """The bot named by a ``.wt-bawthub-<bot>-TASK-N/frontend`` path."""
    import re

    match = re.fullmatch(r"\.wt-bawthub-([a-z0-9][a-z0-9_-]*?)-TASK-\d+", front.parent.name)
    return match.group(1) if match else None


def choose_target(occ: Occupancy, declared: str | None, dev_root: Path) -> Path | None:
    if not occ.occupied:
        return None
    if declared and is_frontend(Path(declared)):
        return Path(declared)
    return newest_worktree(dev_root, list(occ.tenants))


# ------------------------------------------------------------- fallback page


def render_page(house: str, occ: Occupancy | None, status: str, detail: str,
                campus_url: str | None) -> str:
    esc = html.escape
    if occ is None:
        who = "Occupancy unknown."
    elif not occ.occupied:
        who = "Nobody lives here right now."
    else:
        names = list(occ.tenants) + (["a private bot"] * occ.hidden)
        who = "Occupant: " + ", ".join(esc(n) for n in names) + "."
    campus = (f'<p><a href="{esc(campus_url)}/tools/campus">Back to Campus</a></p>'
              if campus_url else "")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="15">
<title>House {esc(house)}: {esc(status)}</title>
<style>
body{{margin:0;min-height:100vh;display:grid;place-items:center;background:#0b0f17;
color:#e5e7eb;font:16px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}}
main{{max-width:34rem;padding:2rem 1.5rem}}
h1{{font-size:1.4rem;margin:0 0 .5rem}} .muted{{color:#9ca3af}}
code{{background:#1f2937;padding:.1rem .35rem;border-radius:.3rem;font-size:.85em;
word-break:break-all}} pre{{white-space:pre-wrap;background:#111827;padding:.75rem;
border-radius:.5rem;font-size:.75rem;max-height:14rem;overflow:auto}}
a{{color:#22d3ee}}
</style></head><body><main>
<h1>House {esc(house)}: {esc(status)}</h1>
<p>{who}</p>
{detail}
<p class="muted">A bot that moves into this house gets its preview started
automatically from its latest BawtHub task worktree. To serve a different
checkout, the occupant runs
<code>python -m agent_bridge.house_preview_supervisor use &lt;worktree&gt;/frontend</code>.
This page refreshes itself.</p>
{campus}
</main></body></html>"""


class FallbackServer:
    """Answers every request on the preview port with the current page."""

    def __init__(self, port: int = PREVIEW_PORT) -> None:
        self.port = port
        self.page = ""
        self._server: asyncio.base_events.Server | None = None

    @property
    def serving(self) -> bool:
        return self._server is not None

    async def start(self) -> None:
        if self._server is None:
            self._server = await asyncio.start_server(
                self._handle, "0.0.0.0", self.port, reuse_address=True)

    async def stop(self) -> None:
        server, self._server = self._server, None
        if server is not None:
            server.close()
            await server.wait_closed()

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=5)
            body = self.page.encode()
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\n"
                b"Cache-Control: no-store\r\nConnection: close\r\n"
                + f"Content-Length: {len(body)}\r\n\r\n".encode() + body)
            await writer.drain()
        except (asyncio.TimeoutError, asyncio.IncompleteReadError, asyncio.LimitOverrunError,
                ConnectionError):
            pass
        finally:
            writer.close()


# ------------------------------------------------------------- supervisor


def _child_env() -> dict[str, str]:
    env = dict(os.environ)
    # The house root is read-only and ~/.config is an empty tmpfs: point pnpm
    # at the shared store explicitly (same filesystem, so packages are linked).
    env["npm_config_store_dir"] = str(PNPM_STORE)
    env["CI"] = "true"  # never prompt
    return env


class PreviewSupervisor:
    """Keeps the preview port serving either the occupant's preview or the page."""

    def __init__(self, house: str, redis: Any, *,
                 fetch_campus: Callable[[], Any] | None = None,
                 fetch_house_of: Callable[[str], Any] | None = None,
                 dev_root: Path = DEV_ROOT, fallback: FallbackServer | None = None) -> None:
        self.house = house
        self._redis = redis
        self._fetch = fetch_campus or self._fetch_campus
        self._house_of = fetch_house_of or fetch_house_of_bot
        self.dev_root = dev_root
        self.fallback = fallback or FallbackServer()
        self.occ: Occupancy | None = None
        self.campus_url: str | None = None
        self.target: Path | None = None
        self._run: asyncio.Task | None = None
        self._proc: asyncio.subprocess.Process | None = None
        # Process groups this supervisor started and has not yet ended.
        self._groups: set[int] = set()
        self._failures = 0
        self._retry_at = 0.0
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._loop())

    async def _loop(self) -> None:
        while True:
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("house %s preview supervisor tick failed", self.house,
                               exc_info=True)
            await asyncio.sleep(CHECK_S)

    async def _fetch_campus(self) -> dict | None:
        import httpx

        base = os.environ.get("LLM_BAWT_API_URL") or "http://app:8642"
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{base}/v1/campus")
                resp.raise_for_status()
                return resp.json()
        except Exception as exc:  # unreachable app: keep the current state
            logger.info("house %s: campus unreadable (%s)", self.house, exc)
            return None

    async def tick(self) -> None:
        snapshot = await self._fetch()
        if snapshot is not None:
            occ = occupancy_from_campus(snapshot, self.house)
            if occ is not None:
                self.occ = occ
            mansion = (snapshot.get("mansion") or {}).get("preview_host")
            self.campus_url = mansion or self.campus_url
        if self.occ is None:
            await self._show("starting up", "<p>Waiting for the app.</p>")
            return
        key = target_key(self.house)
        if not self.occ.occupied:
            await self._redis.delete(key)
            declared = None
        else:
            declared = await self._valid_choice(key)
        target = choose_target(self.occ, declared, self.dev_root)
        if target != self.target:
            await self._stop_run()
            self.target, self._failures, self._retry_at = target, 0, 0.0
            if target is not None:
                logger.info("house %s preview target: %s", self.house, target)
        if self.target is None:
            if self.occ.occupied:
                await self._show("no preview running", "<p>The occupant has no BawtHub "
                                 "task worktree to serve yet.</p>")
            else:
                await self._show("empty", "<p>No preview is running.</p>")
            return
        if self._run is None or self._run.done():
            if time.monotonic() >= self._retry_at:
                self._run = asyncio.create_task(self._serve(self.target))

    async def _valid_choice(self, key: str) -> str | None:
        """The stored path if its bot still lives here; a stale choice is deleted."""
        choice = parse_choice(await self._redis.get(key))
        if choice is None:
            await self._redis.delete(key)
            return None
        bot, path = choice
        house = await self._house_of(bot)
        if house is UNKNOWN:
            # App unreadable: keep serving what we serve, choose nothing new.
            return str(self.target) if self.target and str(self.target) == path else None
        if house != self.house:
            logger.info("house %s: dropping %s's preview choice (now in %s)",
                        self.house, bot, house or "the mansion")
            await self._redis.delete(key)
            return None
        return path

    async def _show(self, status: str, detail: str) -> None:
        self.fallback.page = render_page(self.house, self.occ, status, detail, self.campus_url)
        if not self.fallback.serving and self._proc is None:
            try:
                await self.fallback.start()
            except OSError as exc:  # something else holds the port, e.g. a manual run
                logger.info("house %s: preview port busy (%s)", self.house, exc)

    async def _serve(self, front: Path) -> None:
        esc = html.escape
        where = f"<p>Checkout: <code>{esc(str(front))}</code></p>"
        await self._show("preview starting", where + "<p>Installing dependencies from the "
                         "lockfile. This can take a minute on first use.</p>")
        LOG_PATH.write_text("")
        ok = await self._step(front, ["pnpm", "install", "--frozen-lockfile", "--prefer-offline"])
        ok = ok and await self._step(front, ["pnpm", "exec", "prisma", "generate"])
        if not ok:
            return await self._failed(front, "dependency install failed")
        await self.fallback.stop()
        with LOG_PATH.open("ab") as log:
            self._proc = await asyncio.create_subprocess_exec(
                *PAIR, cwd=front, env=_child_env(), stdout=log,
                stderr=asyncio.subprocess.STDOUT, start_new_session=True)
        self._groups.add(self._proc.pid)
        logger.info("house %s preview started: %s pid %s", self.house, front, self._proc.pid)
        code = await self._proc.wait()
        # The launcher is gone; end anything it left behind before the page
        # (or a new attempt) takes the port.
        await self._end_groups()
        self._proc = None
        if self.target == front:
            await self._failed(front, f"preview exited (code {code})")

    async def _step(self, front: Path, argv: list[str]) -> bool:
        with LOG_PATH.open("ab") as log:
            log.write(f"$ {' '.join(argv)}\n".encode())
            log.flush()
            proc = await asyncio.create_subprocess_exec(
                *argv, cwd=front, env=_child_env(), stdout=log,
                stderr=asyncio.subprocess.STDOUT, start_new_session=True)
            self._groups.add(proc.pid)
            try:
                code = await asyncio.wait_for(proc.wait(), INSTALL_TIMEOUT_S)
            except asyncio.TimeoutError:
                code = None
            # On cancel, _stop_run ends the group after awaiting this run.
            await self._end_groups()
            return code == 0

    async def _failed(self, front: Path, why: str) -> None:
        self._failures += 1
        delay = RETRY_S[min(self._failures - 1, len(RETRY_S) - 1)]
        self._retry_at = time.monotonic() + delay
        logger.warning("house %s preview %s; retry in %ss", self.house, why, int(delay))
        try:
            tail = LOG_PATH.read_text(errors="replace")[-3000:]
        except OSError:
            tail = ""
        await self._show("preview not running", (
            f"<p>Checkout: <code>{html.escape(str(front))}</code></p>"
            f"<p>{html.escape(why)}; retrying in {int(delay)}s.</p>"
            f"<pre>{html.escape(tail)}</pre>"))

    async def _stop_run(self) -> None:
        """Cancel the run first (so it can't report a failure), then end every group."""
        if self._run is not None and not self._run.done():
            self._run.cancel()
            try:
                await self._run
            except (asyncio.CancelledError, Exception):
                pass
        self._run = None
        await self._end_groups()
        self._proc = None

    async def _end_groups(self) -> None:
        """TERM each owned group, KILL what outlives the grace, wait until empty.

        Independent of the leader: a launcher that already exited can leave a
        server in its group holding the port.
        """
        for pgid in list(self._groups):
            if _group_alive(pgid):
                _killpg(pgid, signal.SIGTERM)
                if not await _wait_group_gone(pgid, STOP_GRACE_S):
                    _killpg(pgid, signal.SIGKILL)
                    if not await _wait_group_gone(pgid, 5.0):
                        logger.warning("house %s: process group %s survived KILL",
                                       self.house, pgid)
                        continue
            self._groups.discard(pgid)


def _killpg(pid: int, sig: int) -> None:
    try:
        os.killpg(pid, sig)
    except (ProcessLookupError, PermissionError):
        pass


def _group_alive(pgid: int) -> bool:
    """Any process still in the group (a reaped leader doesn't count)."""
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


async def _wait_group_gone(pgid: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while _group_alive(pgid):
        if time.monotonic() >= deadline:
            return False
        await asyncio.sleep(0.1)
    return True


#: The profile lookup failed; distinct from "not in any house" (None).
UNKNOWN = object()


async def fetch_house_of_bot(bot: str) -> Any:
    """The house a bot's profile names (None = the mansion), or :data:`UNKNOWN`."""
    import httpx

    base = os.environ.get("LLM_BAWT_API_URL") or "http://app:8642"
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{base}/v1/bots/{bot}/profile")
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return (resp.json().get("house") or None)
    except Exception:
        return UNKNOWN


# ------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Choose what this house's preview serves.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    use = sub.add_parser("use", help="serve this checkout's frontend")
    use.add_argument("frontend", type=Path)
    use.add_argument("--bot", help="the occupant making the choice (default: from the "
                     "worktree name)")
    sub.add_parser("clear", help="go back to the occupant's latest task worktree")
    args = parser.parse_args(argv)
    house = (os.environ.get("BAWT_HOUSE") or "").strip().lower()
    if not house:
        parser.exit(2, "not in a house (BAWT_HOUSE is unset)\n")
    import redis

    client = redis.Redis.from_url(os.environ.get("LLM_BAWT_REDIS_URL")
                                  or "redis://redis:6379/0")
    if args.cmd == "clear":
        client.delete(target_key(house))
        print(f"house {house}: preview follows the occupant's latest task worktree")
        return 0
    front = args.frontend.resolve()
    if not is_frontend(front):
        parser.exit(2, f"{front} is not an absolute frontend directory with package.json\n")
    bot = args.bot or worktree_bot(front)
    if not bot:
        parser.exit(2, "can't tell which bot this choice is for: pass --bot <your id>\n")
    lives = asyncio.run(fetch_house_of_bot(bot))
    if lives is UNKNOWN:
        parser.exit(1, "the app can't be read right now; try again\n")
    if lives != house:
        parser.exit(2, f"{bot} doesn't live in house {house}; move in first\n")
    client.set(target_key(house), json.dumps({"bot": bot, "path": str(front)}))
    print(f"house {house}: preview will serve {front} within {int(CHECK_S)}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["FallbackServer", "Occupancy", "PreviewSupervisor", "choose_target",
           "newest_worktree", "occupancy_from_campus", "render_page", "target_key"]
