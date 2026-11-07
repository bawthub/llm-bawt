"""House preview status for the Campus view (TASK-2131)."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
from types import SimpleNamespace

import pytest

from agent_bridge import house_preview as hp
from agent_bridge.house_pool import validate_pool
from llm_bawt import house_watcher as hw
from llm_bawt.service.routes.campus import house_rows

# --- process discovery -------------------------------------------------------

VITE = ["node", "/w/frontend/node_modules/.bin/../vite/bin/vite.js",
        "--host", "0.0.0.0", "--port", "5173", "--strictPort"]


def _proc(tmp_path, pid: int, argv: list[str], cwd) -> None:
    entry = tmp_path / "proc" / str(pid)
    entry.mkdir(parents=True)
    (entry / "cmdline").write_bytes(b"\0".join(a.encode() for a in argv) + b"\0")
    os.symlink(cwd, entry / "cwd")


def test_finds_the_vite_process_not_its_wrappers(tmp_path):
    front = tmp_path / "wt" / "frontend"
    front.mkdir(parents=True)
    _proc(tmp_path, 10, ["sh", "-c", "vite --host 0.0.0.0 --port 5173"], tmp_path)
    _proc(tmp_path, 11, ["node", "/x/pnpm", "exec", "concurrently",
                         "vite --port 5173", "tsx watch server/index.ts"], tmp_path)
    _proc(tmp_path, 12, ["node", "/x/tsx", "watch", "server/index.ts"], tmp_path)
    _proc(tmp_path, 13, VITE, front)
    (tmp_path / "proc" / "self").mkdir()
    assert hp.find_preview_cwd(tmp_path / "proc") == front


def test_no_preview_process_is_none(tmp_path):
    _proc(tmp_path, 10, ["python", "-m", "claude_code_bridge"], tmp_path)
    assert hp.find_preview_cwd(tmp_path / "proc") is None
    assert hp.find_preview_cwd(tmp_path / "missing") is None


# --- checkout description ----------------------------------------------------

def _git(cwd, *args):
    subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True)


@pytest.fixture
def worktree(tmp_path):
    repo = tmp_path / "bawthub"
    (repo / "frontend").mkdir(parents=True)
    (repo / "frontend" / "package.json").write_text("{}")
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "add", ".")
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init")
    wt = tmp_path / ".worktrees" / ".wt-bawthub-nova-TASK-77"
    _git(repo, "worktree", "add", "-q", "-b", "nova/task-77-thing", str(wt))
    return repo, wt


def test_describe_worktree_names_repo_branch_commit_and_task(worktree):
    repo, wt = worktree
    out = hp.describe(wt / "frontend", up=True)
    assert out["up"] is True and out["repo"] == "bawthub"
    assert out["checkout"] == str(wt) and out["branch"] == "nova/task-77-thing"
    assert out["task"] == "TASK-77" and len(out["commit"]) >= 7


def test_describe_main_checkout_has_no_task(worktree):
    repo, _ = worktree
    out = hp.describe(repo / "frontend", up=False)
    assert out["branch"] == "main" and out["task"] is None and out["up"] is False


def test_detached_head_falls_back_to_the_path_for_task(worktree):
    _, wt = worktree
    _git(wt, "checkout", "-q", "--detach")
    out = hp.describe(wt / "frontend", up=True)
    assert out["branch"] is None and out["task"] == "TASK-77"


# --- publisher ---------------------------------------------------------------

class FakeRedis:
    def __init__(self):
        self.data: dict[str, str] = {}
        self.ops: list[tuple] = []

    async def set(self, key, value, ex=None):
        self.data[key] = value
        self.ops.append(("set", key))

    async def expire(self, key, ttl):
        self.ops.append(("expire", key))
        return key in self.data

    async def delete(self, key):
        self.data.pop(key, None)
        self.ops.append(("del", key))


def test_publisher_sets_on_change_extends_otherwise_and_deletes_on_stop():
    values = iter([None, {"up": False}, {"up": True}, {"up": True}, None, None])
    pub = hp.PreviewPublisher("h1", probe_fn=lambda: next(values))
    redis = FakeRedis()

    async def run():
        for _ in range(6):
            await pub.publish(redis)

    asyncio.run(run())
    key = hp.preview_key("h1")
    # none -> starting (set) -> up (set: changed) -> up (extend) -> stopped (del) -> none
    assert redis.ops == [("set", key), ("set", key), ("expire", key), ("del", key)]
    assert key not in redis.data


def test_publisher_rewrites_a_lapsed_key():
    pub = hp.PreviewPublisher("h1", probe_fn=lambda: {"up": True})
    redis = FakeRedis()

    async def run():
        await pub.publish(redis)
        redis.data.clear()  # Redis restarted
        await pub.publish(redis)

    asyncio.run(run())
    assert redis.data[hp.preview_key("h1")] == json.dumps({"up": True})


def test_heartbeat_publishes_preview_and_survives_probe_failure():
    from agent_bridge.houses import HouseSlot, lease_key

    slot = HouseSlot("h1", "claude-code", SimpleNamespace(request_phase=lambda *a: None))
    slot.epoch = "3"
    redis = FakeRedis()
    redis.data[lease_key("h1", "claude-code")] = "3"

    async def get(key):
        return redis.data.get(key)

    redis.get = get
    slot._redis = redis
    slot._preview = hp.PreviewPublisher("h1", probe_fn=lambda: {"up": True})
    asyncio.run(slot.refresh())
    assert hp.preview_key("h1") in redis.data

    def boom():
        raise OSError("proc gone")

    slot._preview = hp.PreviewPublisher("h1", probe_fn=boom)
    asyncio.run(slot.refresh())  # lease renewal is unaffected
    assert redis.data[lease_key("h1", "claude-code")] == "3"


# --- app side ----------------------------------------------------------------

def test_watcher_turns_preview_keyspace_events_into_frames(monkeypatch):
    watcher = hw.HouseWatcher(SimpleNamespace(REDIS_URL="redis://x"))
    sent: list = []

    async def emit(changes):
        sent.extend(hw.preview_frames(changes))

    monkeypatch.setattr(watcher, "_emit_previews", emit)
    value = {"up": True, "branch": "b"}

    class Client:
        async def get(self, key):
            return json.dumps(value)

    channel = "__keyspace@0__:agent:house:h2:preview"

    async def run():
        await watcher.handle(Client(), channel, "set")
        await watcher.handle(Client(), channel, "set")  # unchanged: no frame
        await watcher.handle(Client(), channel, "expire")  # TTL extension
        await watcher.handle(Client(), channel, "expired")

    asyncio.run(run())
    assert sent == [
        {"_type": "house_preview_changed", "house_id": "h2", "preview": value},
        {"_type": "house_preview_changed", "house_id": "h2", "preview": None},
    ]


def test_reseed_reports_previews_changed_while_disconnected():
    watcher = hw.HouseWatcher(SimpleNamespace())
    watcher.previews = {"h1": {"up": True}, "h2": {"up": True}}
    assert watcher.reseed_previews({"h2": {"up": True}, "h3": {"up": False}}) == [
        ("h1", None), ("h3", {"up": False})]


def test_failed_preview_read_on_reconnect_does_not_swallow_lease_transitions(monkeypatch):
    """A lease vanishes while disconnected; the first resubscribe's preview read
    fails. The next successful resubscribe must still report the house down."""
    import redis.asyncio as aioredis

    class PubSub:
        def __init__(self, last):
            self.last = last

        async def psubscribe(self, *patterns):
            pass

        async def listen(self):
            yield {"type": "psubscribe"}
            if self.last:
                raise asyncio.CancelledError()
            raise OSError("disconnect")

        async def aclose(self):
            pass

    class Client:
        connection_pool = SimpleNamespace(connection_kwargs={"db": 0})

        def __init__(self, last):
            self.last = last

        def pubsub(self):
            return PubSub(self.last)

        async def aclose(self):
            pass

    watcher = hw.HouseWatcher(SimpleNamespace(REDIS_URL="redis://offline"))
    clients = iter([Client(False), Client(False), Client(True)])
    leases = iter([{("h1", "claude-code"): "1"}, {}, {}])
    reads = {"n": 0}
    sent: list = []

    async def nothing(*_):
        return None

    async def read_leases(_):
        return next(leases)

    async def read_previews(_):
        reads["n"] += 1
        if reads["n"] == 2:
            raise TimeoutError("preview scan timed out")
        return {}

    async def emit(transitions):
        sent.extend(transitions)

    monkeypatch.setattr(watcher, "_ensure_flags", nothing)
    monkeypatch.setattr(watcher, "_read_leases", read_leases)
    monkeypatch.setattr(watcher, "_emit", emit)
    monkeypatch.setattr(aioredis, "from_url", lambda *a, **k: next(clients))
    monkeypatch.setattr(hw, "read_previews", read_previews)
    monkeypatch.setattr(hw.asyncio, "sleep", nothing)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(watcher.run())
    assert ("h1", "up", "down") in sent


def test_malformed_preview_value_is_ignored():
    assert hw.parse_preview("not json") is None and hw.parse_preview("[1]") is None


def test_snapshot_rows_carry_preview_or_unknown():
    pool = [{"id": "h1", "harness": "claude-code", "preview_host": "https://dev1.example.test"}]
    rows = house_rows(pool, {}, {}, {}, previews={"h1": {"up": True}})
    assert rows[0]["preview"] == {"up": True}
    assert house_rows(pool, {}, {}, {}, previews={})[0]["preview"] is None
    assert house_rows(pool, None, {}, {}, previews=None)[0]["preview"] == "unknown"


def test_pool_accepts_an_optional_mansion_origin(tmp_path, monkeypatch):
    from llm_bawt.house_pool_config import POOL_FILE_ENV, mansion_origin

    assert "mansion_origin" not in validate_pool({"houses": []})
    assert validate_pool({"houses": [], "mansion_origin": "https://dev.example.test/"})[
        "mansion_origin"] == "https://dev.example.test"
    # A tenant serves the shared site on a plain-http published port (TASK-2129).
    assert validate_pool({"houses": [], "mansion_origin": "http://host.example.test:3101"})[
        "mansion_origin"] == "http://host.example.test:3101"
    with pytest.raises(ValueError):
        validate_pool({"houses": [], "mansion_origin": "https://dev.example.test/app"})
    pool = tmp_path / "houses.json"
    pool.write_text(json.dumps({"houses": [], "mansion_origin": "https://dev.example.test"}))
    monkeypatch.setenv(POOL_FILE_ENV, str(pool))
    assert mansion_origin() == "https://dev.example.test"
    monkeypatch.delenv(POOL_FILE_ENV)
    assert mansion_origin() is None
