"""Bot profile writes fan out over the existing unified SSE transport."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agent_bridge.publisher import GLOBAL_CONFIG_STREAM
from agent_bridge.subscriber import RedisSubscriber
from llm_bawt.service.routes import settings


def test_global_config_stream_is_subscribed_by_each_window_without_cross_user_turns():
    async def exercise():
        sub = object.__new__(RedisSubscriber)
        sub._redis = SimpleNamespace(xgroup_create=AsyncMock())
        nick = await sub.ensure_groups(["al"], "nick", "window-nick", include_global=True)
        guest = await sub.ensure_groups(["al"], "guest", "window-guest", include_global=True)
        regular = await sub.ensure_groups(["al"], "nick", "worker")
        assert GLOBAL_CONFIG_STREAM in nick and GLOBAL_CONFIG_STREAM in guest
        assert "events:al:guest" not in nick
        assert "events:al:nick" not in guest
        assert GLOBAL_CONFIG_STREAM not in regular
        created = sub._redis.xgroup_create.await_args_list
        assert any(c.args[:2] == (GLOBAL_CONFIG_STREAM, "ui:window-nick") for c in created)
        assert any(c.args[:2] == (GLOBAL_CONFIG_STREAM, "ui:window-guest") for c in created)

        sub._pub_redis = SimpleNamespace(xadd=AsyncMock(return_value="123-0"))
        assert await sub.publish_global_config_event({"_type": "bot_profile_changed", "bot_id": "al"}) == "123-0"
        assert sub._pub_redis.xadd.await_args.args[0] == GLOBAL_CONFIG_STREAM

    asyncio.run(exercise())


def test_profile_change_publishes_only_after_commit_and_cache_reload(monkeypatch):
    lifecycle = []
    profile = SimpleNamespace(slug="al", agent_backend="claude-code")
    store = SimpleNamespace(engine=object(), get=lambda slug: None,
                            upsert=lambda payload: (lifecycle.append("commit"), profile)[1])
    subscriber = SimpleNamespace(publish_global_config_event=AsyncMock(
        side_effect=lambda event: (lifecycle.append(("event", event)), "123-0")[1]
    ))
    service = SimpleNamespace(config=object(), _redis_subscriber=subscriber)
    monkeypatch.setattr(settings, "get_service", lambda: service)
    monkeypatch.setattr(settings, "get_bot_profile_store", lambda _config: store)
    monkeypatch.setattr(settings, "_validate_profile_payload", lambda _payload: None)
    monkeypatch.setattr(settings, "_reload_bot_registry", lambda: lifecycle.append("reload"))
    monkeypatch.setattr(settings, "_reload_service_model_catalog", lambda _service: None)
    monkeypatch.setattr(settings, "_invalidate_bot_instance_cache", lambda *_args: None)
    monkeypatch.setattr(settings, "_clear_session_model_overrides", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(settings, "_resolve_effective_bot_prompt", lambda *_args: "")
    monkeypatch.setattr(settings, "_effective_bot_settings", lambda *_args: {})
    monkeypatch.setattr(settings, "_to_profile_response", lambda profile, settings: profile)

    result = asyncio.run(settings._persist_bot_profile({"slug": "al", "endpoint_id": 42}, create_only=False))
    assert result is profile
    assert lifecycle == ["commit", "reload", ("event", {"_type": "bot_profile_changed", "bot_id": "al"})]

    def reject(_payload):
        raise ValueError("invalid model")

    monkeypatch.setattr(settings, "_validate_profile_payload", reject)
    with pytest.raises(ValueError, match="invalid model"):
        asyncio.run(settings._persist_bot_profile({"slug": "al", "endpoint_id": -1}, create_only=False))
    assert subscriber.publish_global_config_event.await_count == 1


@pytest.mark.parametrize("failure", [None, RuntimeError("redis down")])
def test_committed_profile_write_does_not_fail_when_event_transport_does(failure):
    sub = None if failure is None else SimpleNamespace(
        publish_global_config_event=AsyncMock(side_effect=failure)
    )
    asyncio.run(settings._publish_bot_profile_changed(SimpleNamespace(_redis_subscriber=sub), "al"))
