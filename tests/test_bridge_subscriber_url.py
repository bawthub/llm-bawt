"""Per-turn subscribers preserve the original Redis connection URL offline."""
from unittest.mock import Mock

import pytest
from redis.asyncio import Redis
from redis.asyncio.connection import SSLConnection

from agent_bridge.events import AgentEvent, AgentEventKind
from agent_bridge.subscriber import RedisSubscriber
from llm_bawt.agent_backends import agent_bridge


@pytest.mark.parametrize("url", [
    "redis://fixture-user:fixture-pass@redis.invalid:6380/4?socket_timeout=7&health_check_interval=13",
    "rediss://fixture-user:fixture-pass@redis.invalid:6380/4?socket_timeout=7&ssl_cert_reqs=required",
])
def test_turn_subscriber_keeps_auth_tls_and_url_options(monkeypatch, url):
    root = RedisSubscriber(url)
    # redis-py parses URLs without retaining a 'url' entry. Constructing pools
    # is offline: all connection/publication methods below are replaced.
    root._redis = Redis.from_url(url)
    source_pool = root._redis.connection_pool
    assert "url" not in source_pool.connection_kwargs
    monkeypatch.setattr(agent_bridge, "get_agent_subscriber", lambda: root)
    created = []
    sent = Mock()

    class LocalSubscriber(RedisSubscriber):
        def __init__(self, original_url):
            super().__init__(original_url)
            self._redis = Redis.from_url(original_url)
            created.append(self)

        async def connect(self):
            pass

        async def close(self):
            await self._redis.aclose()

        async def send_command(self, **kwargs):
            sent(**kwargs)

        async def subscribe_run(self, request_id, **kwargs):
            yield AgentEvent(kind=AgentEventKind.ASSISTANT_DONE, run_id=request_id,
                             session_key="fixture:user", seq=1, text="done",
                             event_id="fixture-event", origin="test")

    monkeypatch.setattr("agent_bridge.subscriber.RedisSubscriber", LocalSubscriber)
    assert list(agent_bridge.AgentBridgeBackend().stream_raw("hello", {
        "session_key": "fixture:user", "request_id": "fixture-request",
    })) == ["done"]
    assert len(created) == 1
    assert created[0]._redis_url == url
    local_pool = created[0]._redis.connection_pool
    assert local_pool is not source_pool
    assert local_pool.connection_class is source_pool.connection_class
    # Redis also injects per-pool handler objects, which must NOT be shared.
    for key in ("host", "port", "db", "username", "password", "socket_timeout",
                "health_check_interval", "ssl_cert_reqs"):
        assert local_pool.connection_kwargs.get(key) == source_pool.connection_kwargs.get(key)
    assert local_pool.connection_kwargs["username"] == "fixture-user"
    assert local_pool.connection_kwargs["password"] == "fixture-pass"
    assert local_pool.connection_kwargs["db"] == 4
    assert local_pool.connection_kwargs["socket_timeout"] == 7
    if url.startswith("rediss:"):
        assert local_pool.connection_class is SSLConnection
    sent.assert_called_once()
