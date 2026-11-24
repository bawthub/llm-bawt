"""TASK-2590: strict-reader tools are off at the bridge, and unsupported paths refuse."""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import HTTPException

from claude_code_bridge.send_request import SendRequest
from claude_code_bridge.send_stream import ClaudeStreamMixin, deny_all_tools_hook
from llm_bawt.bots import Bot
from llm_bawt.service import strict_reader_execution as strict

FIELDS = {"request_id": "req-1", "session_key": "sample:operator", "bot_id": "sample",
          "message": "hello", "model": "test-model"}


def _options(tools_disabled: bool):
    bridge = SimpleNamespace(
        _mcp_servers={"bawthub": {"type": "http", "url": "http://test/mcp"}},
        _cwd="/tmp", _permission_mode="default", _add_dirs=[],
        _make_external_failure_hook=lambda: None,
    )
    return ClaudeStreamMixin._build_agent_options(
        bridge, model="test-model", system_prompt="p", disallowed_tools=[], resume_id=None,
        sdk_env={}, settings_path=None, bot_effort=None, bot_max_turns=None,
        can_use_tool_cb=None, pre_tool_use_cb=None, post_tool_use_cb=None, stderr=None,
        tools_disabled=tools_disabled,
    )


def test_bridge_request_parses_tools_disabled_only_when_set():
    assert SendRequest.from_fields({**FIELDS, "tools_disabled": "1"}).tools_disabled is True
    assert SendRequest.from_fields(FIELDS).tools_disabled is False
    assert SendRequest.from_fields({**FIELDS, "tools_disabled": "0"}).tools_disabled is False


def test_tools_disabled_options_have_no_tools_no_mcp_and_a_deny_hook():
    off = _options(True)
    assert off.tools == [] and off.strict_mcp_config is True
    hooks = [h for m in off.hooks["PreToolUse"] for h in m.hooks]
    assert hooks == [deny_all_tools_hook]
    out = asyncio.run(deny_all_tools_hook({"tool_name": "Bash"}, "t1", None))
    assert out["hookSpecificOutput"]["permissionDecision"] == "deny"


def _cli_args(options) -> list[str]:
    from claude_agent_sdk._internal.transport.subprocess_cli import SubprocessCLITransport

    transport = SubprocessCLITransport(prompt="hi", options=options)
    transport._cli_path = "claude"
    return transport._build_command()


def test_tools_disabled_serializes_explicit_empty_strict_mcp_config():
    args = _cli_args(_options(True))
    i = args.index("--mcp-config")
    assert json.loads(args[i + 1]) == {"mcpServers": {}}
    assert "--strict-mcp-config" in args
    assert "bawthub" not in " ".join(args)
    t = args.index("--tools")
    assert args[t + 1] == ""


def test_ordinary_options_keep_tools_and_mcp():
    on = _options(False)
    assert on.tools is None and "bawthub" in on.mcp_servers
    args = _cli_args(on)
    assert "--strict-mcp-config" not in args
    assert "bawthub" in args[args.index("--mcp-config") + 1]


def test_non_streaming_strict_request_refused_before_any_lookup():
    from llm_bawt.service.routes import chat

    req = SimpleNamespace(stream=False, bot_id="sample", user="operator", model="m",
                          messages=[], session_id=None)
    http = SimpleNamespace(headers={})
    with _no_registry(), patch.object(chat, "get_service", return_value=_service()), \
            patch.object(chat, "_internal_inter_bot_sender", return_value=None), \
            patch("llm_bawt.service.chat_response_adapter.completion_response",
                  side_effect=AssertionError("ordinary path")), \
            pytest.raises(HTTPException) as exc:
        asyncio.run(chat.chat_completions(req, http))
    assert exc.value.status_code == 400


def test_publisher_sends_tools_disabled_only_when_true():
    from agent_bridge.subscriber import RedisSubscriber

    def send(**kwargs):
        sub = RedisSubscriber("redis://localhost:6379/0")
        captured: dict = {}

        async def fake_xadd(_stream, fields, **_kwargs):
            captured.update(fields)
            return b"1"

        sub._pub_redis = SimpleNamespace(xadd=fake_xadd)
        asyncio.run(sub.send_command(session_key="s", message="m", request_id="r", **kwargs))
        return captured

    assert send(tools_disabled=True)["tools_disabled"] == "1"
    assert "tools_disabled" not in send()


def test_strict_instance_seals_its_client():
    instance = strict.StrictReaderLLMBawt.__new__(strict.StrictReaderLLMBawt)
    instance.client = SimpleNamespace()
    instance._seal_tools()
    assert instance.client.tools_disabled is True


def _service():
    return SimpleNamespace(strict_reader_bot="sample", config=SimpleNamespace())


def _req(**over):
    base = dict(bot_id="sample", user="operator", model="test-model",
                inter_bot_delivery_id=None, session_id=None)
    return SimpleNamespace(**{**base, **over})


@pytest.mark.parametrize("over", [{"inter_bot_delivery_id": "d-1"}, {"session_id": "s-1"}])
def test_strict_refuses_delivery_and_explicit_thread_before_reading(over):
    def read(*_a):
        raise AssertionError("profile read")
    with pytest.raises(strict.StrictReaderRefused, match="direct ordinary turn"):
        strict.create_strict_instance(_service(), _req(**over), read_bot=read)


def test_strict_refuses_backend_without_tools_off_enforcement():
    bot = Bot(slug="sample", name="Sample", description="d", system_prompt="p", bot_type="agent", agent_backend="codex")
    with patch.object(strict, "validate_bound_endpoint", side_effect=AssertionError):
        with pytest.raises(strict.StrictReaderRefused, match="claude-code"):
            strict.create_strict_instance(_service(), _req(), read_bot=lambda *_a: bot)


def _no_registry():
    return patch("llm_bawt.bots.BotManager.__init__", side_effect=AssertionError("registry read"))


def test_steer_refused_for_strict_bot_without_registry_read():
    from llm_bawt.service.routes.chat import _run_steer_caps

    with _no_registry(), pytest.raises(HTTPException) as exc:
        _run_steer_caps(_service(), SimpleNamespace(bot_id="sample", agent_backend=None))
    assert exc.value.status_code == 400


def test_session_reset_refused_for_strict_bot_without_registry_read():
    from llm_bawt.service.routes import chat

    with _no_registry(), patch.object(chat, "get_service", return_value=_service()), \
            pytest.raises(HTTPException) as exc:
        asyncio.run(chat.session_reset(SimpleNamespace(bot_id="sample")))
    assert exc.value.status_code == 400


def test_abort_backend_for_strict_bot_needs_no_registry_read():
    from llm_bawt.service.turn_abort import _bot_backend

    with _no_registry():
        assert _bot_backend(_service(), "sample") == "claude-code"
