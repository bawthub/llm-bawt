"""TASK-2379: codex and local-model command listeners keep the Redis URL.

Rebuilding the listener client from the pool's host/port/db drops the
password, so Redis AUTH would break them while the Claude bridge kept working.
"""

import asyncio
from types import SimpleNamespace

import pytest

from codex_bridge.command_ops import CodexCommandMixin as CodexCommandOps
from local_model_bridge.bridge import LocalModelBridge

URLS = [
    "redis://:fixture-pass@redis.invalid:6380/4",
    "rediss://fixture-user:fixture-pass@redis.invalid:6380/4?ssl_cert_reqs=required",
]


class _Stop(Exception):
    pass


class _FakeRedis:
    async def ping(self):
        raise _Stop


def _publisher(url):
    pool = SimpleNamespace(connection_kwargs={"host": "redis.invalid", "port": 6380, "db": 4})
    return SimpleNamespace(redis_url=url, _redis=SimpleNamespace(connection_pool=pool))


def _run_listener(monkeypatch, listener_owner, url):
    seen = {}

    def from_url(u, **kwargs):
        seen["url"], seen["kwargs"] = u, kwargs
        return _FakeRedis()

    def no_rebuild(**kwargs):
        raise AssertionError("host/port rebuild drops the Redis password")

    monkeypatch.setattr("redis.asyncio.from_url", from_url)
    monkeypatch.setattr("redis.asyncio.Redis", no_rebuild)
    listener_owner._publisher = _publisher(url)
    with pytest.raises(_Stop):
        asyncio.run(listener_owner._command_listener())
    return seen


@pytest.mark.parametrize("url", URLS)
def test_codex_listener_reconnects_with_original_url(monkeypatch, url):
    owner = object.__new__(type("CodexOwner", (CodexCommandOps,), {}))
    seen = _run_listener(monkeypatch, owner, url)
    assert seen["url"] == url
    assert seen["kwargs"]["socket_timeout"] is None


@pytest.mark.parametrize("url", URLS)
def test_local_model_listener_reconnects_with_original_url(monkeypatch, url):
    owner = object.__new__(LocalModelBridge)
    seen = _run_listener(monkeypatch, owner, url)
    assert seen["url"] == url
    assert seen["kwargs"]["socket_timeout"] is None


def test_url_less_publisher_falls_back_to_host_port(monkeypatch):
    built = {}

    def fake_redis(**kwargs):
        built.update(kwargs)
        return _FakeRedis()

    monkeypatch.setattr("redis.asyncio.Redis", fake_redis)
    owner = object.__new__(LocalModelBridge)
    owner._publisher = _publisher(None)
    with pytest.raises(_Stop):
        asyncio.run(owner._command_listener())
    assert (built["host"], built["port"], built["db"]) == ("redis.invalid", 6380, 4)
