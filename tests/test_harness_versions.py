"""TASK-2470: running harness versions per house/mansion, and the latest check."""

from __future__ import annotations

import asyncio
import fnmatch
import json
from datetime import UTC, datetime
from types import SimpleNamespace

from fastapi.testclient import TestClient

from agent_bridge import versions
from agent_bridge.health_http import health_app
from agent_bridge.houses import HouseSlot, lease_key
from llm_bawt import harness_versions as hv
from llm_bawt import house_watcher as hw

VERSIONS = {"schema": 1, "harness": {"claude-agent-sdk": "0.2.180"}, "claude_cli": "2.1.317",
            "image": None, "started_at": "2026-01-01T00:00:00+00:00", "instance": "h:1"}


class FakeRedis:
    def __init__(self, data=None):
        self.data = dict(data or {})
        self.ttl: dict[str, int] = {}

    async def get(self, key):
        return self.data.get(key)

    async def set(self, key, value, ex=None, nx=False):
        self.data[key] = value
        if ex:
            self.ttl[key] = ex
        return True

    async def incr(self, key):
        self.data[key] = str(int(self.data.get(key) or 0) + 1)
        return int(self.data[key])

    async def delete(self, key):
        self.data.pop(key, None)

    async def expire(self, key, ttl):
        return key in self.data

    async def scan_iter(self, match, count=None):
        for key in list(self.data):
            if fnmatch.fnmatchcase(key, match):
                yield key


def _cached(monkeypatch):
    monkeypatch.setattr(versions, "_cache", dict(VERSIONS))


def _house(epoch, versions_=VERSIONS):
    return json.dumps({"epoch": epoch, "versions": versions_})


# --- bridge side --------------------------------------------------------------

def test_collect_reads_installed_packages_and_never_runs_per_request(monkeypatch):
    calls = {"n": 0}

    def collect():
        calls["n"] += 1
        return dict(VERSIONS)

    monkeypatch.setattr(versions, "_cache", None)
    monkeypatch.setattr(versions, "collect", collect)
    client = TestClient(health_app(SimpleNamespace(connected=True)))
    assert client.get("/versions").json() == VERSIONS
    assert client.get("/versions").json() == VERSIONS
    assert calls["n"] == 1


def test_missing_package_and_failed_cli_are_null(monkeypatch, tmp_path):
    assert versions.package_version("no-such-package-xyz") is None
    assert versions.cli_version(None) is None
    bad = tmp_path / "claude"
    bad.write_text("#!/bin/sh\nexit 3\n")
    bad.chmod(0o755)
    assert versions.cli_version(bad) is None
    good = tmp_path / "ok"
    good.write_text("#!/bin/sh\necho '2.1.3 (Claude Code)'\n")
    good.chmod(0o755)
    assert versions.cli_version(good) == "2.1.3"


def test_house_binds_versions_to_its_lease_epoch_and_renews(monkeypatch):
    _cached(monkeypatch)
    redis = FakeRedis()
    slot = HouseSlot("h1", "claude-code", SimpleNamespace(request_phase=lambda *a: None))

    async def run():
        await slot.start(redis)
        slot._task.cancel()
        key = versions.house_versions_key("h1", "claude-code")
        first = json.loads(redis.data[key])
        del redis.data[key]
        await slot.refresh()
        return first, json.loads(redis.data[key]), redis.ttl[key]

    first, renewed, ttl = asyncio.run(run())
    assert first == {"epoch": "1", "versions": VERSIONS} == renewed
    assert ttl == 30


def test_mansion_publishes_last_reported_on_every_start(monkeypatch):
    _cached(monkeypatch)
    redis = FakeRedis()
    slot = HouseSlot("", "codex", None)

    async def run():
        await slot.start(redis)
        redis.data.clear()  # Redis restarted
        await slot.start(redis)  # listener restart after reconnect

    asyncio.run(run())
    assert json.loads(redis.data["agent:mansion:versions:codex"]) == {"versions": VERSIONS}


def test_publish_failure_does_not_break_the_lease(monkeypatch):
    async def boom():
        raise RuntimeError("no metadata")

    monkeypatch.setattr(versions, "current", boom)
    redis = FakeRedis()
    slot = HouseSlot("h1", "claude-code", None)

    async def run():
        await slot.start(redis)
        slot._task.cancel()

    asyncio.run(run())
    assert redis.data[lease_key("h1", "claude-code")] == "1"


# --- classification ------------------------------------------------------------

def test_house_is_running_only_under_its_live_epoch():
    assert hv.classify("h1", "cc", _house("4"), "4")["status"] == "running"
    for raw, epoch in ((_house("4"), "5"),  # replaced: new epoch, no record yet
                       (_house("4"), None),  # lease expired or released
                       (None, "4"),  # lease, no record
                       ("not json", "4"), (json.dumps({"epoch": "4"}), "4")):
        row = hv.classify("h1", "cc", raw, epoch)
        assert row["status"] == "unknown" and row["versions"] is None


def test_mansion_is_last_reported_never_running():
    row = hv.classify("mansion", "cc", json.dumps({"versions": VERSIONS}), None)
    assert row == {"slot": "mansion", "backend": "cc", "status": "last_reported",
                   "versions": VERSIONS, "started_at": VERSIONS["started_at"]}


def test_read_all_includes_leased_houses_without_records():
    redis = FakeRedis({
        lease_key("h1", "cc"): "2", "agent:house:h1:versions:cc": _house("2"),
        lease_key("h2", "codex"): "7",
        "agent:house:h3:versions:cc": _house("1"),
        "agent:mansion:versions:cc": json.dumps({"versions": VERSIONS}),
    })
    rows = asyncio.run(hv.read_all(redis))
    assert {k: r["status"] for k, r in rows.items()} == {
        ("h1", "cc"): "running", ("h2", "codex"): "unknown", ("h3", "cc"): "unknown",
        ("mansion", "cc"): "last_reported"}


# --- watcher ------------------------------------------------------------------

def _watcher(monkeypatch):
    watcher = hw.HouseWatcher(SimpleNamespace(REDIS_URL="redis://x"))
    sent: list = []

    async def emit(rows):
        sent.extend(hv.row_frame(r) for r in rows)

    async def nothing(*_):
        return None

    monkeypatch.setattr(watcher, "_emit_harness", emit)
    monkeypatch.setattr(watcher, "_emit", nothing)
    return watcher, sent


def test_watcher_follows_epoch_restart_expiry_and_matches_the_snapshot(monkeypatch):
    watcher, sent = _watcher(monkeypatch)
    redis = FakeRedis()
    lease, rec = lease_key("h1", "cc"), "agent:house:h1:versions:cc"
    ch = "__keyspace@0__:"

    async def step(key, op, **changes):
        for k, v in changes.items():
            if v is None:
                redis.data.pop(k, None)
            else:
                redis.data[k] = v
        await watcher.handle(redis, ch + key, op)
        snapshot = await hv.read_all(redis)
        if watcher.harness.get(("h1", "cc")) is not None and ("h1", "cc") in snapshot:
            assert snapshot[("h1", "cc")] == watcher.harness[("h1", "cc")]

    async def run():
        await step(lease, "set", **{lease: "1"})
        await step(rec, "set", **{rec: _house("1")})
        await step(rec, "set", **{rec: _house("1")})  # heartbeat: no frame
        await step(lease, "set", **{lease: "2"})  # restarted, metadata not yet
        await step(rec, "set", **{rec: _house("2")})
        await step(lease, "expired", **{lease: None})

    asyncio.run(run())
    assert [f["status"] for f in sent] == ["unknown", "running", "unknown", "running", "unknown"]
    assert all(f["_type"] == "harness_versions_changed" for f in sent)
    assert sent[-1]["versions"] is None


def test_backend_switch_shows_each_backend_separately(monkeypatch):
    watcher, sent = _watcher(monkeypatch)
    redis = FakeRedis({lease_key("h1", "codex"): "3",
                       "agent:house:h1:versions:codex": _house("3")})

    async def run():
        await watcher.handle(redis, "__keyspace@0__:agent:house:h1:versions:codex", "set")
        redis.data.pop(lease_key("h1", "codex"))
        await watcher.handle(redis, "__keyspace@0__:" + lease_key("h1", "codex"), "del")
        redis.data[lease_key("h1", "claude-code")] = "4"
        redis.data["agent:house:h1:versions:claude-code"] = _house("4")
        await watcher.handle(redis, "__keyspace@0__:agent:house:h1:versions:claude-code", "set")

    asyncio.run(run())
    assert [(f["backend"], f["status"]) for f in sent] == [
        ("codex", "running"), ("codex", "unknown"), ("claude-code", "running")]


def test_reseed_reports_slots_gone_and_changed_while_disconnected(monkeypatch):
    watcher, _ = _watcher(monkeypatch)
    running = hv.classify("h1", "cc", _house("1"), "1")
    watcher.harness = {("h1", "cc"): running, ("h2", "cc"): running | {"slot": "h2"}}
    out = watcher.reseed_harness({("h1", "cc"): hv.classify("h1", "cc", _house("1"), "2")})
    assert [(r["slot"], r["status"]) for r in out] == [("h2", "unknown"), ("h1", "unknown")]


# --- latest -------------------------------------------------------------------

def test_latest_merges_per_package_and_survives_bad_runs():
    t1, t2, t3 = (datetime(2026, 1, d, tzinfo=UTC) for d in (3, 2, 1))
    merged = hv.merge_latest([
        (t1, json.dumps({"packages": [{"name": "openai-codex", "latest": "0.169.0"}]})),  # subset
        (t2, "{broken"),
        (t2, json.dumps({"packages": [{"name": "claude-agent-sdk", "latest": None}]})),
        (t3, json.dumps({"packages": [
            {"name": "claude-agent-sdk", "pinned": "0.1", "latest": "0.2.183",
             "latest_cli": "2.1.320", "behind": True},
            {"name": "openai-codex", "latest": "0.167.0"}]})),
    ])
    assert merged == {
        "packages": {"openai-codex": {"version": "0.169.0", "observed_at": t1.isoformat()},
                     "claude-agent-sdk": {"version": "0.2.183", "observed_at": t3.isoformat()}},
        "claude_cli": {"version": "2.1.320", "observed_at": t3.isoformat()},
    }
    assert hv.merge_latest([]) == {"packages": {}, "claude_cli": None}


# --- reconnect through the real listener (review cmwm92lc) ---------------------

class _Reader:
    """First read: Redis restarted (data gone) and the read fails; then idle."""

    def __init__(self, redis, stop_after):
        self.redis, self.reads, self.stop_after = redis, 0, stop_after
        self.recreations = 0

    async def ensure(self, strict=False):
        return None

    async def read(self, count=1, block=0):
        self.reads += 1
        if self.reads == 1:
            self.redis.data.clear()
            raise ConnectionError("redis restarted")
        if self.reads >= self.stop_after:
            raise asyncio.CancelledError()
        return []


def _listener(monkeypatch, module, backend):
    import importlib

    mod = importlib.import_module(module)
    redis = FakeRedis()
    redis.ping = _noop
    redis.aclose = _noop
    reader = _Reader(redis, stop_after=3)
    monkeypatch.setattr(mod, "GroupReader", lambda *a, **k: reader)
    monkeypatch.setattr(mod.asyncio, "sleep", lambda *_: _noop())
    monkeypatch.setenv("BAWT_HOUSE", "")
    if hasattr(mod, "CommandAckOwner"):
        monkeypatch.setattr(mod, "CommandAckOwner",
                            lambda *a, **k: SimpleNamespace(start=lambda: None))
    return mod, redis


async def _noop(*_):
    return None


def test_mansion_republishes_after_redis_restart_through_the_claude_listener(monkeypatch):
    _cached(monkeypatch)
    mod, redis = _listener(monkeypatch, "claude_code_bridge.command_ops", "claude-code")
    bridge = mod.ClaudeCommandMixin.__new__(mod.ClaudeCommandMixin)
    bridge._backend_name = "claude-code"
    bridge._session_queue = None
    bridge._approval_reload_task = SimpleNamespace(done=lambda: False)

    try:
        asyncio.run(bridge._consume_commands(redis, "redis://x"))
    except asyncio.CancelledError:
        pass
    assert json.loads(redis.data["agent:mansion:versions:claude-code"]) == {"versions": VERSIONS}


class _NogroupRedis(FakeRedis):
    """Retained connection, Redis restarted: the group (and the mansion key)
    are gone, so the first XREADGROUP answers NOGROUP; later reads idle."""

    def __init__(self, stop_after):
        super().__init__()
        self.reads, self.stop_after = 0, stop_after
        self.ping = self.aclose = _noop

    async def xgroup_create(self, stream, group, id, mkstream=False):
        return True

    async def xinfo_groups(self, stream):
        return []

    async def xreadgroup(self, group, consumer, streams, count=None, block=None):
        from redis.exceptions import ResponseError

        self.reads += 1
        if self.reads == 1:
            self.data.clear()
            raise ResponseError("NOGROUP No such key or consumer group")
        if self.reads >= self.stop_after:
            raise asyncio.CancelledError()
        return []


def test_mansion_republishes_after_nogroup_recovery_through_the_real_reader(monkeypatch):
    import claude_code_bridge.command_ops as mod

    _cached(monkeypatch)
    redis = _NogroupRedis(stop_after=3)
    monkeypatch.setattr(mod.asyncio, "sleep", lambda *_: _noop())
    monkeypatch.setenv("BAWT_HOUSE", "")
    if hasattr(mod, "CommandAckOwner"):
        monkeypatch.setattr(mod, "CommandAckOwner",
                            lambda *a, **k: SimpleNamespace(start=lambda: None))
    bridge = mod.ClaudeCommandMixin.__new__(mod.ClaudeCommandMixin)
    bridge._backend_name = "claude-code"
    bridge._session_queue = None
    bridge._approval_reload_task = SimpleNamespace(done=lambda: False)

    try:
        asyncio.run(bridge._consume_commands(redis, "redis://x"))
    except asyncio.CancelledError:
        pass
    assert redis.reads >= 2
    assert json.loads(redis.data["agent:mansion:versions:claude-code"]) == {"versions": VERSIONS}
