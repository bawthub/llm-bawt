"""Regression tests for the data-driven Claude SDK disallowed-tools policy."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from claude_code_bridge.tool_policy import (
    CLAUDE_CODE_DISALLOWED_TOOLS_KEY,
    DEFAULT_DISALLOWED_TOOLS,
    effective_disallowed_tools,
)


class TestToolPolicy:
    def test_default_disables_plan_and_worktree_workflows(self):
        assert effective_disallowed_tools(None, use_proxy=False) == list(
            DEFAULT_DISALLOWED_TOOLS
        )

    def test_configured_policy_replaces_default_and_deduplicates(self):
        raw = json.dumps(["Bash", "Read", "Bash"])
        assert effective_disallowed_tools(raw, use_proxy=False) == ["Bash", "Read"]

    def test_explicit_empty_list_is_respected(self):
        assert effective_disallowed_tools("[]", use_proxy=False) == []

    def test_proxy_additions_are_mandatory_and_deduplicated(self):
        raw = json.dumps(["EnterPlanMode", "WebSearch"])
        assert effective_disallowed_tools(raw, use_proxy=True) == [
            "EnterPlanMode",
            "WebSearch",
            "WebFetch",
        ]

    @pytest.mark.parametrize("raw", ["not-json", json.dumps("Bash"), json.dumps([1])])
    def test_malformed_transport_falls_back_to_default(self, raw):
        assert effective_disallowed_tools(raw, use_proxy=False) == list(
            DEFAULT_DISALLOWED_TOOLS
        )


class TestSettingDefinition:
    def test_setting_is_registered_as_string_list(self):
        from llm_bawt.setting_definitions import (
            SETTING_DEFINITIONS,
            STORAGE_RUNTIME_SETTING,
        )

        definition = SETTING_DEFINITIONS[CLAUDE_CODE_DISALLOWED_TOOLS_KEY]
        assert definition.type == "string_list"
        assert definition.storage == STORAGE_RUNTIME_SETTING
        assert definition.default == list(DEFAULT_DISALLOWED_TOOLS)


class TestGlobalResolution:
    def test_resolver_ignores_bot_scope(self, monkeypatch):
        from llm_bawt import runtime_settings
        from llm_bawt.runtime_setting_resolution import resolve_global_runtime_setting
        from llm_bawt.utils.config import Config

        captured: dict = {}

        class FakeResolver:
            def __init__(self, config, bot=None):
                captured["bot"] = bot

            def resolve(self, key, fallback=None):
                captured["key"] = key
                captured["fallback"] = fallback
                return fallback

        monkeypatch.setattr(runtime_settings, "RuntimeSettingsResolver", FakeResolver)
        result = resolve_global_runtime_setting(
            Config(), CLAUDE_CODE_DISALLOWED_TOOLS_KEY
        )

        assert captured["bot"] is None
        assert result == list(DEFAULT_DISALLOWED_TOOLS)


class TestTransport:
    def test_subscriber_json_encodes_explicit_empty_list(self):
        from agent_bridge.subscriber import RedisSubscriber

        subscriber = RedisSubscriber("redis://localhost:6379/0")
        captured: dict = {}

        async def fake_xadd(_stream, fields, **_kwargs):
            captured.update(fields)
            return b"1"

        subscriber._pub_redis = SimpleNamespace(xadd=fake_xadd)
        asyncio.run(subscriber.send_command(
            session_key="test-session",
            message="hello",
            request_id="req-test",
            disallowed_tools=[],
        ))

        assert captured["disallowed_tools"] == "[]"

    def test_subscriber_forwards_opaque_task_turn_capability(self):
        from agent_bridge.subscriber import RedisSubscriber

        subscriber = RedisSubscriber("redis://localhost:6379/0")
        captured: dict = {}

        async def fake_xadd(_stream, fields, **_kwargs):
            captured.update(fields)
            return b"1"

        subscriber._pub_redis = SimpleNamespace(xadd=fake_xadd)
        asyncio.run(subscriber.send_command(
            session_key="test-session",
            message="hello",
            request_id="req-test",
            task_turn_capability="opaque-capability",
        ))

        assert captured["task_turn_capability"] == "opaque-capability"

    def test_agent_bridge_forwards_configured_list(self, monkeypatch):
        from llm_bawt.agent_backends import agent_bridge as module

        captured: dict = {}

        class FakeRedisSubscriber:
            def __init__(self, _url):
                self._redis = SimpleNamespace(
                    connection_pool=SimpleNamespace(
                        connection_kwargs={"url": "redis://localhost:6379/0"}
                    )
                )

            async def connect(self):
                return None

            async def send_command(self, **kwargs):
                captured.update(kwargs)

            async def subscribe_run(self, *_args, **_kwargs):
                if False:
                    yield None

            async def close(self):
                return None

        root_subscriber = SimpleNamespace(
            _redis_url="redis://fixture.invalid/0", _redis=SimpleNamespace(
                connection_pool=SimpleNamespace(
                    connection_kwargs={"url": "redis://localhost:6379/0"}
                )
            )
        )
        monkeypatch.setattr(module, "get_agent_subscriber", lambda: root_subscriber)
        monkeypatch.setattr(
            "agent_bridge.subscriber.RedisSubscriber", FakeRedisSubscriber
        )

        backend = module.AgentBridgeBackend()
        list(
            backend.stream_raw(
                "hello",
                {
                    "bot_id": "test",
                    "disallowed_tools": ["EnterPlanMode"],
                    "timeout_seconds": 1,
                },
            )
        )
        assert captured["disallowed_tools"] == ["EnterPlanMode"]

    def test_agent_bridge_forwards_live_tool_callbacks(self, monkeypatch):
        from agent_bridge.events import AgentEvent, AgentEventKind
        from llm_bawt.agent_backends import agent_bridge as module

        class FakeRedisSubscriber:
            def __init__(self, _url):
                self._redis = SimpleNamespace(
                    connection_pool=SimpleNamespace(
                        connection_kwargs={"url": "redis://localhost:6379/0"}
                    )
                )

            async def connect(self):
                return None

            async def send_command(self, **_kwargs):
                return None

            async def subscribe_run(self, *_args, **_kwargs):
                yield AgentEvent(
                    event_id="start", session_key="s", run_id="run",
                    kind=AgentEventKind.TOOL_START, origin="assistant",
                    tool_name="Read", tool_arguments={"file_path": "x.py"},
                    tool_use_id="tool-1",
                )
                yield AgentEvent(
                    event_id="end", session_key="s", run_id="run",
                    kind=AgentEventKind.TOOL_END, origin="tool",
                    tool_name="Read", tool_arguments={"file_path": "x.py"},
                    tool_result="contents", tool_use_id="tool-1",
                )
                yield AgentEvent(
                    event_id="done", session_key="s", run_id="run",
                    kind=AgentEventKind.ASSISTANT_DONE, origin="assistant",
                    text="ok",
                )

            async def close(self):
                return None

        root_subscriber = SimpleNamespace(
            _redis_url="redis://fixture.invalid/0", _redis=SimpleNamespace(
                connection_pool=SimpleNamespace(
                    connection_kwargs={"url": "redis://localhost:6379/0"}
                )
            )
        )
        monkeypatch.setattr(module, "get_agent_subscriber", lambda: root_subscriber)
        monkeypatch.setattr(
            "agent_bridge.subscriber.RedisSubscriber", FakeRedisSubscriber
        )

        callbacks = []
        backend = module.AgentBridgeBackend()
        list(backend.stream_raw(
            "hello",
            {
                "bot_id": "test",
                "timeout_seconds": 1,
                "event_callback": callbacks.append,
            },
        ))

        assert [item["event"] for item in callbacks] == ["tool_call", "tool_result"]
        assert callbacks[0]["tool_use_id"] == "tool-1"
        assert callbacks[1]["result"] == "contents"
        assert len(backend.get_last_stream_result().tool_calls) == 1

    def test_agent_bridge_retains_terminal_usage_in_result(self, monkeypatch):
        from agent_bridge.events import AgentEvent, AgentEventKind
        from llm_bawt.agent_backends import agent_bridge as module

        usage = {
            "input_tokens": 10170,
            "cache_read_tokens": 9728,
            "resident_tokens": 16475,
            "resident_source": "claude_sdk_context",
            "context_window": 372000,
        }

        class FakeRedisSubscriber:
            def __init__(self, _url):
                self._redis = SimpleNamespace(
                    connection_pool=SimpleNamespace(
                        connection_kwargs={"url": "redis://localhost:6379/0"}
                    )
                )

            async def connect(self):
                return None

            async def send_command(self, **_kwargs):
                return None

            async def subscribe_run(self, *_args, **_kwargs):
                yield AgentEvent(
                    event_id="done", session_key="s", run_id="run",
                    kind=AgentEventKind.ASSISTANT_DONE, origin="assistant",
                    text="ok", token_usage=usage,
                )

            async def close(self):
                return None

        root_subscriber = SimpleNamespace(
            _redis_url="redis://fixture.invalid/0", _redis=SimpleNamespace(
                connection_pool=SimpleNamespace(
                    connection_kwargs={"url": "redis://localhost:6379/0"}
                )
            )
        )
        monkeypatch.setattr(module, "get_agent_subscriber", lambda: root_subscriber)
        monkeypatch.setattr(
            "agent_bridge.subscriber.RedisSubscriber", FakeRedisSubscriber
        )

        backend = module.AgentBridgeBackend()
        streamed = list(backend.stream_raw(
            "hello", {"bot_id": "test", "timeout_seconds": 1}
        ))

        assert {"event": "token_usage", "token_usage": usage} in streamed
        assert backend.get_last_stream_result().usage == usage

    def test_agent_bridge_preserves_usage_from_terminal_error(self, monkeypatch):
        from agent_bridge.events import AgentEvent, AgentEventKind
        from llm_bawt.agent_backends import agent_bridge as module

        usage = {
            "input_tokens": 10,
            "cache_read_tokens": 90,
            "output_tokens": 4,
            "usage_status": "partial",
        }

        class FakeRedisSubscriber:
            def __init__(self, _url):
                self._redis = SimpleNamespace(
                    connection_pool=SimpleNamespace(
                        connection_kwargs={"url": "redis://localhost:6379/0"}
                    )
                )

            async def connect(self):
                return None

            async def send_command(self, **_kwargs):
                return None

            async def subscribe_run(self, *_args, **_kwargs):
                yield AgentEvent(
                    event_id="err", session_key="s", run_id="run",
                    kind=AgentEventKind.ERROR, origin="system", text="boom",
                    token_usage=usage,
                )

            async def close(self):
                return None

        root_subscriber = SimpleNamespace(
            _redis_url="redis://fixture.invalid/0", _redis=SimpleNamespace(
                connection_pool=SimpleNamespace(
                    connection_kwargs={"url": "redis://localhost:6379/0"}
                )
            )
        )
        monkeypatch.setattr(module, "get_agent_subscriber", lambda: root_subscriber)
        monkeypatch.setattr(
            "agent_bridge.subscriber.RedisSubscriber", FakeRedisSubscriber
        )

        backend = module.AgentBridgeBackend()
        iterator = iter(backend.stream_raw(
            "hello", {"bot_id": "test", "timeout_seconds": 1}
        ))
        assert next(iterator) == {"event": "token_usage", "token_usage": usage}
        with pytest.raises(RuntimeError, match="boom"):
            next(iterator)
        assert backend.get_last_stream_result().usage == usage

    def test_agent_bridge_drains_file_changed_after_error(self, monkeypatch):
        from agent_bridge.events import AgentEvent, AgentEventKind
        from llm_bawt.agent_backends import agent_bridge as module

        class FakeRedisSubscriber:
            def __init__(self, _url):
                self._redis = SimpleNamespace(
                    connection_pool=SimpleNamespace(
                        connection_kwargs={"url": "redis://localhost:6379/0"}
                    )
                )

            async def connect(self):
                return None

            async def send_command(self, **_kwargs):
                return None

            async def subscribe_run(self, *_args, **_kwargs):
                yield AgentEvent(
                    event_id="err", session_key="s", run_id="run",
                    kind=AgentEventKind.ERROR, origin="system", text="boom",
                )
                yield AgentEvent(
                    event_id="file", session_key="s", run_id="run",
                    kind=AgentEventKind.FILE_CHANGED, origin="system",
                    tool_name="Edit", tool_arguments={"file_path": "/r/a.py"},
                    tool_use_id="tool-1",
                    raw={"file": {"repo_key": "r", "path": "a.py"}},
                )

            async def close(self):
                return None

        root_subscriber = SimpleNamespace(
            _redis_url="redis://fixture.invalid/0", _redis=SimpleNamespace(
                connection_pool=SimpleNamespace(
                    connection_kwargs={"url": "redis://localhost:6379/0"}
                )
            )
        )
        monkeypatch.setattr(module, "get_agent_subscriber", lambda: root_subscriber)
        monkeypatch.setattr(
            "agent_bridge.subscriber.RedisSubscriber", FakeRedisSubscriber
        )

        backend = module.AgentBridgeBackend()
        iterator = iter(backend.stream_raw(
            "hello", {"bot_id": "test", "timeout_seconds": 1}
        ))
        changed = next(iterator)
        assert changed["event"] == "file_changed"
        assert changed["file"]["path"] == "a.py"
        with pytest.raises(RuntimeError, match="boom"):
            next(iterator)


class TestClaudeBackendResolution:
    def test_claude_backend_adds_resolved_policy_before_dispatch(self):
        from llm_bawt.agent_backends.claude_code import ClaudeCodeBackend

        backend = ClaudeCodeBackend()
        configured = ["EnterPlanMode", "ExitWorktree"]

        with (
            patch(
                "llm_bawt.runtime_setting_resolution.resolve_global_runtime_setting",
                return_value=configured,
            ),
            patch(
                "llm_bawt.agent_backends.agent_bridge.AgentBridgeBackend.stream_raw",
                return_value=iter(()),
            ) as parent_stream,
        ):
            list(
                backend.stream_raw(
                    "hello",
                    {"model": "claude-opus-4-7", "bot_id": "loopy"},
                )
            )

        dispatched = parent_stream.call_args.args[1]
        assert dispatched["disallowed_tools"] == configured

    def test_provider_system_prompt_follows_bot_prompt_and_is_byte_stable(self):
        from llm_bawt.agent_backends.claude_code import ClaudeCodeBackend

        backend = ClaudeCodeBackend()
        config = {
            "model": "openai_chatgpt/gpt-5.6-sol",
            "bot_id": "loopy",
            "system_prompt": "Bot instructions.",
            "provider_system_prompt": "OpenAI harness instructions.",
        }

        with (
            patch(
                "llm_bawt.runtime_setting_resolution.resolve_global_runtime_setting",
                return_value=[],
            ),
            patch(
                "llm_bawt.agent_backends.agent_bridge.AgentBridgeBackend.stream_raw",
                return_value=iter(()),
            ) as parent_stream,
        ):
            list(backend.stream_raw("first", config))
            first = parent_stream.call_args.args[1]["system_prompt"]
            list(backend.stream_raw("second", config))
            second = parent_stream.call_args.args[1]["system_prompt"]

        assert first == second
        assert first.endswith(
            "Bot instructions.\n\nOpenAI harness instructions."
        )
        assert first.index("<runtime-context>") < first.index("Bot instructions.")
        assert first.index("Bot instructions.") < first.index(
            "OpenAI harness instructions."
        )

    def test_provider_system_prompt_refreshes_from_catalog_each_dispatch(self):
        from llm_bawt.agent_backends.claude_code import ClaudeCodeBackend

        backend = ClaudeCodeBackend()
        current = {"provider_system_prompt": "First provider instructions."}
        backend._config = SimpleNamespace(resolve_model=lambda *args, **kwargs: current)
        config = {
            "model": "openai_chatgpt/gpt-5.6-sol",
            "bot_id": "loopy",
            "endpoint_id": 20,
            "harness": "claude-proxy",
            "system_prompt": "Bot instructions.",
        }

        with (
            patch(
                "llm_bawt.runtime_setting_resolution.resolve_global_runtime_setting",
                return_value=[],
            ),
            patch(
                "llm_bawt.agent_backends.agent_bridge.AgentBridgeBackend.stream_raw",
                return_value=iter(()),
            ) as parent_stream,
        ):
            list(backend.stream_raw("first", config))
            first = parent_stream.call_args.args[1]["system_prompt"]
            current["provider_system_prompt"] = "Second provider instructions."
            list(backend.stream_raw("second", config))
            second = parent_stream.call_args.args[1]["system_prompt"]

        assert first.endswith("First provider instructions.")
        assert second.endswith("Second provider instructions.")

    def test_blank_provider_system_prompt_adds_no_separator(self):
        from llm_bawt.agent_backends.claude_code import ClaudeCodeBackend

        backend = ClaudeCodeBackend()
        with (
            patch(
                "llm_bawt.runtime_setting_resolution.resolve_global_runtime_setting",
                return_value=[],
            ),
            patch(
                "llm_bawt.agent_backends.agent_bridge.AgentBridgeBackend.stream_raw",
                return_value=iter(()),
            ) as parent_stream,
        ):
            list(
                backend.stream_raw(
                    "hello",
                    {
                        "model": "openai_chatgpt/gpt-5.6-sol",
                        "bot_id": "loopy",
                        "system_prompt": "Bot instructions.",
                        "provider_system_prompt": " \n ",
                    },
                )
            )

        assert parent_stream.call_args.args[1]["system_prompt"].endswith(
            "Bot instructions."
        )
