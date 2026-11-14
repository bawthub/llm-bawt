"""A house serves its occupant's preview, or a page saying why not (TASK-2135)."""

from __future__ import annotations

import asyncio
import json
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

    async def eval(self, script, numkeys, key, expected):  # compare-and-delete
        if self.data.get(key) == expected:
            del self.data[key]
            return 1
        return 0


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


def _supervisor(tmp_path, monkeypatch, campus, redis=None, homes=None):
    port = _free_port()
    monkeypatch.setattr(sup, "LOG_PATH", tmp_path / "preview.log")
    homes = homes if homes is not None else {}

    async def fetch():
        return campus["value"]

    async def house_of(bot):
        return homes.get(bot, "h1")

    s = sup.PreviewSupervisor("h1", redis or FakeRedis(), fetch_campus=fetch,
                              fetch_house_of=house_of, dev_root=tmp_path,
                              fallback=sup.FallbackServer(port))
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


# --- review round 1: choices belong to a bot; groups end completely ----------

def _choice(bot, path):
    return json.dumps({"bot": bot, "path": str(path)})


def test_choice_parsing_and_worktree_bot(tmp_path):
    assert sup.parse_choice(_choice("nova", "/x")) == ("nova", "/x")
    assert sup.parse_choice(b'{"bot": "nova", "path": "/x"}') == ("nova", "/x")
    for raw in (None, "", "/plain/path", "{", '{"path": "/x"}', '{"bot": "", "path": "/x"}'):
        assert sup.parse_choice(raw) is None
    front = tmp_path / ".wt-bawthub-nova-TASK-12" / "frontend"
    assert sup.worktree_bot(front) == "nova"
    assert sup.worktree_bot(tmp_path / "bawthub" / "frontend") is None


@pytest.mark.asyncio
async def test_new_occupant_never_inherits_the_last_ones_choice(tmp_path, monkeypatch):
    chosen = _worktree(tmp_path, ".wt-bawthub-first-TASK-1", 100)
    second = _worktree(tmp_path, ".wt-bawthub-second-TASK-2", 50)
    redis = FakeRedis({sup.target_key("h1"): _choice("first", chosen)})
    campus = {"value": _campus(tenants=["first"])}
    homes = {"first": "h1"}
    s, _ = _supervisor(tmp_path, monkeypatch, campus, redis, homes)

    async def never(self, front):
        return None

    monkeypatch.setattr(sup.PreviewSupervisor, "_serve", never)
    try:
        await s.tick()
        assert s.target == chosen
        # first leaves and second arrives between two ticks: no empty sample.
        homes["first"] = None
        campus["value"] = _campus(tenants=["second"])
        await s.tick()
        assert s.target == second
        assert sup.target_key("h1") not in redis.data
    finally:
        await s._stop_run()
        await s.fallback.stop()


@pytest.mark.asyncio
async def test_private_occupant_swap_also_drops_the_choice(tmp_path, monkeypatch):
    chosen = _worktree(tmp_path, ".wt-bawthub-ghost-TASK-1", 100)
    redis = FakeRedis({sup.target_key("h1"): _choice("ghost", chosen)})
    homes = {"ghost": "h1"}
    s, _ = _supervisor(tmp_path, monkeypatch, {"value": _campus(hidden=1)}, redis, homes)

    async def never(self, front):
        return None

    monkeypatch.setattr(sup.PreviewSupervisor, "_serve", never)
    try:
        await s.tick()
        assert s.target == chosen
        homes["ghost"] = "h2"  # another private bot now holds h1
        await s.tick()
        assert s.target is None and sup.target_key("h1") not in redis.data
    finally:
        await s.fallback.stop()


@pytest.mark.asyncio
async def test_legacy_or_unreadable_choice_is_ignored(tmp_path, monkeypatch):
    newest = _worktree(tmp_path, ".wt-bawthub-nova-TASK-2", 300)
    other = _worktree(tmp_path, ".wt-bawthub-nova-TASK-1", 100)
    redis = FakeRedis({sup.target_key("h1"): str(other)})
    s, _ = _supervisor(tmp_path, monkeypatch, {"value": _campus(tenants=["nova"])}, redis)

    async def never(self, front):
        return None

    monkeypatch.setattr(sup.PreviewSupervisor, "_serve", never)
    try:
        await s.tick()
        assert s.target == newest and sup.target_key("h1") not in redis.data
    finally:
        await s.fallback.stop()


# A launcher that leaves a TERM-ignoring server in its process group.
_SERVER = (
    "import signal,socket,sys,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);"
    "s=socket.socket();s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);"
    "s.bind(('127.0.0.1',int(sys.argv[1])));s.listen();"
    "open(sys.argv[2],'w').write('up');time.sleep(600)"
)


def _stubborn_pair(port, ready, leader_exits):
    # Exit only once the server is listening, so it is the one left behind.
    tail = (f"while [ ! -f {ready} ]; do sleep 0.05; done; exit 0" if leader_exits
            else "wait")
    return ["sh", "-c", f'"{sys.executable}" -c "$0" {port} {ready} & {tail}', _SERVER]


async def _until(cond, timeout=10.0):
    for _ in range(int(timeout / 0.05)):
        if cond():
            return True
        await asyncio.sleep(0.05)
    return False


@pytest.mark.asyncio
@pytest.mark.parametrize("leader_exits", [False, True])
async def test_stopping_ends_the_whole_group(tmp_path, monkeypatch, leader_exits):
    _worktree(tmp_path, ".wt-bawthub-nova-TASK-7", 100)
    campus = {"value": _campus(tenants=["nova"])}
    s, port = _supervisor(tmp_path, monkeypatch, campus)
    ready = tmp_path / "ready"
    monkeypatch.setattr(sup, "STOP_GRACE_S", 0.5)
    monkeypatch.setattr(sup, "RETRY_S", (600.0,))

    async def step(self, cwd, argv):
        return True

    monkeypatch.setattr(sup.PreviewSupervisor, "_step", step)
    monkeypatch.setattr(sup, "PAIR", _stubborn_pair(port, ready, leader_exits))
    try:
        await s.tick()
        assert await _until(ready.exists)
        if leader_exits:
            # The launcher exited on its own; its server must not keep the port.
            assert await _until(lambda: s._run.done())
            assert "preview exited (code 0)" in await _http_get(port)
        else:
            campus["value"] = _campus()  # occupant leaves: stop the preview
            await s.tick()
            assert "House h1: empty" in await _http_get(port)
        assert s._groups == set()
    finally:
        await s._stop_run()
        await s.fallback.stop()
