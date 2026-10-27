"""``agent_backend_config.codex_transport`` from the profile to the bridge (TASK-1243).

The profile write validates it, the codex backend derives the run's steer
capabilities from it, and the app passes it on the ``chat.send`` command,
where the codex bridge picks app-server or exec.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from agent_bridge.events import AgentEvent, AgentEventKind
from agent_bridge.subscriber import RedisSubscriber
from llm_bawt.agent_backends import agent_bridge
from llm_bawt.agent_backends.codex import CodexBackend
from llm_bawt.service.routes import bot_profile_validation
from llm_bawt.steer_capabilities import NEXT, NO_STEER_CAPS, bot_steer_caps


@pytest.mark.parametrize(("config", "caps"), [
    ({"codex_transport": "app_server"}, frozenset({NEXT})),
    ({"codex_transport": " app_server "}, frozenset({NEXT})),
    ({"codex_transport": "exec"}, NO_STEER_CAPS),
    ({}, NO_STEER_CAPS),
])
def test_only_app_server_codex_runs_can_steer(config, caps):
    assert CodexBackend().steer_capabilities(config) == caps
    bot = SimpleNamespace(agent_backend="codex", agent_backend_config=config)
    assert bot_steer_caps(bot) == caps


def _payload(backend, transport):
    return {"slug": "bot", "bot_type": "agent", "agent_backend": backend,
            "agent_backend_config": {"codex_transport": transport}}


@pytest.mark.parametrize(("backend", "transport", "detail"), [
    ("claude-code", "app_server", "codex_transport requires the Codex backend"),
    ("codex", "server", "codex_transport must be one of: exec, app_server"),
    ("codex", "", "codex_transport must be one of: exec, app_server"),
])
def test_profile_refuses_a_bad_codex_transport(monkeypatch, backend, transport, detail):
    monkeypatch.setattr(bot_profile_validation, "_validate_model_endpoint", lambda payload: None)
    with pytest.raises(HTTPException) as refused:
        bot_profile_validation._validate_profile_payload(_payload(backend, transport))
    assert (refused.value.status_code, refused.value.detail) == (400, detail)


@pytest.mark.parametrize("transport", ["exec", "app_server"])
def test_profile_accepts_a_codex_transport(monkeypatch, transport):
    monkeypatch.setattr(bot_profile_validation, "_validate_model_endpoint", lambda payload: None)
    payload = _payload("codex", transport)
    bot_profile_validation._validate_profile_payload(payload)
    assert payload["agent_backend_config"] == {"codex_transport": transport}


def _published(**kwargs):
    subscriber = RedisSubscriber("redis://localhost:6379/0")
    fake = MagicMock()
    commands = []

    async def eval_script(_script, _numkeys, _dedupe, _stream, _maxlen, _request, *flat):
        commands.append(dict(zip(flat[::2], flat[1::2])))
        return [1, "1-0"]

    fake.eval = eval_script
    subscriber._pub_redis = fake
    asyncio.run(subscriber.send_command(session_key="bot:user", message="hi", request_id="req", **kwargs))
    return commands[0]


def test_the_command_carries_codex_transport_only_when_set():
    assert _published(codex_transport="app_server")["codex_transport"] == "app_server"
    assert "codex_transport" not in _published()


def test_the_backend_passes_the_bot_codex_transport(monkeypatch):
    sent = []
    root = SimpleNamespace(_redis=SimpleNamespace(connection_pool=SimpleNamespace(connection_kwargs={})))
    monkeypatch.setattr(agent_bridge, "get_agent_subscriber", lambda: root)

    class Subscriber:
        def __init__(self, url):
            pass

        async def connect(self):
            pass

        async def close(self):
            pass

        async def send_command(self, **kwargs):
            sent.append(kwargs)

        async def subscribe_run(self, request_id, **kwargs):
            yield AgentEvent(kind=AgentEventKind.ASSISTANT_DONE, run_id=request_id, session_key="bot:user",
                             seq=1, text="done", event_id="evt-1", origin="test")

    monkeypatch.setattr("agent_bridge.subscriber.RedisSubscriber", Subscriber)
    for config, expected in (({"codex_transport": " app_server "}, "app_server"), ({}, None)):
        list(agent_bridge.AgentBridgeBackend().stream_raw("hello", {
            "session_key": "bot:user", "request_id": "req", **config,
        }))
        assert sent.pop()["codex_transport"] == expected
