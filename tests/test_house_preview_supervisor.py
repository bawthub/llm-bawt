"""A house serves its occupant's preview, or a page saying why not (TASK-2135)."""

from __future__ import annotations

import asyncio
import os
import socket
import sys

import pytest

from agent_bridge import house_preview_supervisor as sup
from agent_bridge.house_preview_supervisor import Occupancy


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _worktree(dev, name, mtime):
    """A task worktree whose git index was last touched at ``mtime``."""
    wt = dev / ".worktrees" / name
    (wt / "frontend").mkdir(parents=True)
    (wt / "frontend" / "package.json").write_text("{}")
    gitdir = dev / "bawthub" / ".git" / "worktrees" / name
    gitdir.mkdir(parents=True)
    (gitdir / "index").write_text("")
    os.utime(gitdir / "index", (mtime, mtime))
    (wt / ".git").write_text(f"gitdir: {gitdir}\n")
    return wt / "frontend"


class FakeRedis:
    def __init__(self, data=None):
        self.data = dict(data or {})

    async def get(self, key):
        return self.data.get(key)

    async def delete(self, key):
        self.data.pop(key, None)


def _campus(house="h1", tenants=(), hidden=0):
    return {"houses": [{"id": house, "tenants": list(tenants), "hidden_tenants": hidden}],
            "mansion": {"preview_host": "https://dev.example.test"}}


# --- choosing the checkout --------------------------------------------------

def test_empty_house_serves_nothing_even_if_a_choice_lingers(tmp_path):
    front = _worktree(tmp_path, ".wt-bawthub-nova-TASK-1", 100)
    assert sup.choose_target(Occupancy((), 0), str(front), tmp_path) is None


def test_occupant_choice_wins_over_its_latest_worktree(tmp_path):
    chosen = _worktree(tmp_path, ".wt-bawthub-nova-TASK-1", 100)
    _worktree(tmp_path, ".wt-bawthub-nova-TASK-2", 200)
    assert sup.choose_target(Occupancy(("nova",), 0), str(chosen), tmp_path) == chosen


def test_latest_used_worktree_of_the_occupant_only(tmp_path):
    _worktree(tmp_path, ".wt-bawthub-nova-TASK-1", 100)
    newest = _worktree(tmp_path, ".wt-bawthub-nova-TASK-2", 300)
    _worktree(tmp_path, ".wt-bawthub-vex-TASK-3", 900)  # someone else's
    _worktree(tmp_path, ".wt-llm-bawt-nova-TASK-4", 900).parent  # other repo
    occ = Occupancy(("nova",), 0)
    assert sup.choose_target(occ, None, tmp_path) == newest
    # A stale or bogus choice falls back to the latest worktree.
    assert sup.choose_target(occ, str(tmp_path / "gone"), tmp_path) == newest
    assert sup.choose_target(occ, "relative/frontend", tmp_path) == newest


def test_occupant_without_a_worktree_has_no_target(tmp_path):
    assert sup.choose_target(Occupancy(("nova",), 0), None, tmp_path) is None
    assert sup.choose_target(Occupancy((), 1), None, tmp_path) is None


def test_occupancy_from_campus():
    assert sup.occupancy_from_campus(_campus(tenants=["nova"]), "h1") == Occupancy(("nova",), 0)
    assert sup.occupancy_from_campus(_campus(hidden=1), "h1").occupied
    assert sup.occupancy_from_campus(_campus(house="h2"), "h1") == Occupancy((), 0)
    assert sup.occupancy_from_campus({}, "h1") is None


def test_page_escapes_and_names_the_occupant():
    page = sup.render_page("h1", Occupancy(("<nova>",), 1), "empty", "<p>x</p>",
                           "https://dev.example.test")
    assert "&lt;nova&gt;" in page and "a private bot" in page
    assert "https://dev.example.test/tools/campus" in page
    assert "house_preview_supervisor use" in page


# --- the fallback page and supervision --------------------------------------

async def _http_get(port: int) -> str:
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    writer.write(b"GET / HTTP/1.1\r\nHost: dev1.example.test\r\n\r\n")
    await writer.drain()
    data = await reader.read()
    writer.close()
    return data.decode()


def _supervisor(tmp_path, monkeypatch, campus, redis=None):
    port = _free_port()
    monkeypatch.setattr(sup, "LOG_PATH", tmp_path / "preview.log")

    async def fetch():
        return campus["value"]

    s = sup.PreviewSupervisor("h1", redis or FakeRedis(), fetch_campus=fetch,
                              dev_root=tmp_path, fallback=sup.FallbackServer(port))
    return s, port


@pytest.mark.asyncio
async def test_empty_house_answers_with_the_page_and_forgets_the_choice(tmp_path, monkeypatch):
    redis = FakeRedis({sup.target_key("h1"): "/old/frontend"})
    s, port = _supervisor(tmp_path, monkeypatch, {"value": _campus()}, redis)
    try:
        await s.tick()
        body = await _http_get(port)
        assert body.startswith("HTTP/1.1 200 OK")
        assert "House h1: empty" in body and "Nobody lives here" in body
        assert sup.target_key("h1") not in redis.data
    finally:
        await s._stop_run()
        await s.fallback.stop()


@pytest.mark.asyncio
async def test_occupied_house_installs_then_hands_the_port_to_the_preview(tmp_path, monkeypatch):
    front = _worktree(tmp_path, ".wt-bawthub-nova-TASK-7", 100)
    campus = {"value": _campus(tenants=["nova"])}
    s, port = _supervisor(tmp_path, monkeypatch, campus)
    steps: list[list[str]] = []
    release = asyncio.Event()

    async def step(self, cwd, argv):
        steps.append(argv)
        await release.wait()
        return True

    monkeypatch.setattr(sup.PreviewSupervisor, "_step", step)
    # Stand-in for the Vite pair: serve on the same port, like strictPort.
    monkeypatch.setattr(sup, "PAIR", [sys.executable, "-m", "http.server", str(port),
                                      "--bind", "127.0.0.1"])
    try:
        await s.tick()
        await asyncio.sleep(0.05)
        assert s.target == front
        assert "preview starting" in await _http_get(port)  # page while installing
        release.set()
        for _ in range(100):
            await asyncio.sleep(0.05)
            if s._proc is not None:
                try:
                    if "Directory listing" in await _http_get(port):
                        break
                except OSError:
                    pass
        else:
            pytest.fail("preview never took over the port")
        assert steps == [["pnpm", "install", "--frozen-lockfile", "--prefer-offline"],
                         ["pnpm", "exec", "prisma", "generate"]]
        assert not s.fallback.serving
        # The occupant leaves: the preview stops and the page comes back.
        campus["value"] = _campus()
        await s.tick()
        assert s._proc is None and s.target is None
        assert "House h1: empty" in await _http_get(port)
    finally:
        await s._stop_run()
        await s.fallback.stop()


@pytest.mark.asyncio
async def test_failed_install_shows_why_and_retries_later(tmp_path, monkeypatch):
    _worktree(tmp_path, ".wt-bawthub-nova-TASK-7", 100)
    s, port = _supervisor(tmp_path, monkeypatch, {"value": _campus(tenants=["nova"])})
    calls = []

    async def step(self, cwd, argv):
        calls.append(argv)
        sup.LOG_PATH.write_text("ERR_PNPM_OUTDATED_LOCKFILE <boom>")
        return False

    monkeypatch.setattr(sup.PreviewSupervisor, "_step", step)
    try:
        await s.tick()
        await s._run
        body = await _http_get(port)
        assert "dependency install failed; retrying in 15s" in body
        assert "ERR_PNPM_OUTDATED_LOCKFILE &lt;boom&gt;" in body
        await s.tick()  # inside the backoff: no new attempt
        assert len(calls) == 1
    finally:
        await s._stop_run()
        await s.fallback.stop()


@pytest.mark.asyncio
async def test_unreadable_campus_keeps_the_current_state(tmp_path, monkeypatch):
    campus = {"value": None}
    s, port = _supervisor(tmp_path, monkeypatch, campus)
    try:
        await s.tick()
        assert "starting up" in await _http_get(port)
        campus["value"] = _campus()
        await s.tick()
        campus["value"] = None
        await s.tick()
        assert "Nobody lives here" in await _http_get(port)
    finally:
        await s.fallback.stop()
