"""The unified /v1/ws stream must emit a NAMED heartbeat event.

EventSource never surfaces SSE ``: comment`` lines, so a comment keepalive gave
the bawthub client watchdog no liveness signal: a proxy hop that kept the
browser socket open after llm-bawt restarted left the tab silently deaf until
a hard refresh. A named ``heartbeat`` event is what the client watchdog counts.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from llm_bawt.service.routes import openclaw_ws


class _FakeSubscriber:
    groups_include_global = False
    subscription_include_global = False
    groups_scope = None
    subscription_scope = None

    def __init__(self, _url: str) -> None:
        self.closed = False

    async def connect(self) -> None:
        return None

    async def ensure_groups(self, *args, **kwargs) -> None:
        type(self).groups_scope = args[:2]
        type(self).groups_include_global = kwargs.get("include_global", False)

    async def subscribe_group(self, *args, **kwargs):
        type(self).subscription_scope = args[:2]
        type(self).subscription_include_global = kwargs.get("include_global", False)
        yield None  # idle keepalive tick
        if not args[0]:
            yield {"_type": "bot_profile_changed", "bot_id": "al"}
        else:
            yield {"_type": "turn_start", "turn_id": "t1", "_replayed": True}
        yield None

    async def close(self) -> None:
        self.closed = True


def _frames(monkeypatch, interval: float) -> list[tuple[str, dict]]:
    monkeypatch.setattr(openclaw_ws, "RedisSubscriber", _FakeSubscriber)
    monkeypatch.setattr(openclaw_ws, "HEARTBEAT_INTERVAL_S", interval)

    async def collect() -> list[str]:
        return [
            chunk
            async for chunk in openclaw_ws._unified_event_stream(
                "redis://unused", ["caid"], "nick", "window-x"
            )
        ]

    out = []
    for chunk in asyncio.run(collect()):
        assert not chunk.startswith(":"), "comment keepalives are invisible to EventSource"
        event_line, data_line = chunk.strip().split("\n")
        out.append((event_line.removeprefix("event: "), json.loads(data_line.removeprefix("data: "))))
    return out


def test_idle_ticks_emit_named_heartbeat(monkeypatch):
    frames = _frames(monkeypatch, interval=0.0)
    names = [name for name, _ in frames]
    assert names[0] == "hello"
    assert "heartbeat" in names
    assert all("ts" in data for name, data in frames if name == "heartbeat")


def test_events_pass_through_and_heartbeat_is_throttled(monkeypatch):
    frames = _frames(monkeypatch, interval=3600.0)
    assert [name for name, _ in frames] == ["hello", "event"]
    assert frames[1][1] == {"_type": "turn_start", "turn_id": "t1", "replayed": True}
    assert _FakeSubscriber.groups_include_global is True
    assert _FakeSubscriber.subscription_include_global is True


def test_config_only_sse_receives_global_events_without_a_user_or_bot(monkeypatch):
    monkeypatch.setattr(openclaw_ws, "RedisSubscriber", _FakeSubscriber)
    monkeypatch.setattr(openclaw_ws, "get_service", lambda: SimpleNamespace(
        _redis_subscriber=SimpleNamespace(_redis_url="redis://unused"),
    ))
    app = FastAPI()
    app.include_router(openclaw_ws.router)
    client = TestClient(app)

    response = client.get("/v1/ws?config_only=true&consumer_id=window-guest")
    assert response.status_code == 200
    assert '"mode": "config-only"' in response.text
    assert '"_type": "bot_profile_changed"' in response.text
    assert _FakeSubscriber.groups_scope == ([], "")
    assert _FakeSubscriber.subscription_scope == ([], "")
    assert _FakeSubscriber.groups_include_global is True

    for invalid in (
        "/v1/ws?config_only=true",
        "/v1/ws?config_only=true&consumer_id=x&user_id=nick",
        "/v1/ws?config_only=true&consumer_id=x&bot_id=al",
    ):
        assert client.get(invalid).status_code == 400
