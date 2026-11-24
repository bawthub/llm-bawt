"""Agents learn their installation's UI address from the composed prompt (TASK-2523)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from llm_bawt.agent_backends.installation_context import (
    installation_context_block,
    with_installation_context,
)

ORIGIN = "http://hub.example.com:13000"


class _Resolved:
    def __init__(self, body: str) -> None:
        self.body = body


class _FakeResolver:
    def __init__(self, bodies: dict[str, str] | None = None, fail: bool = False) -> None:
        self.bodies = bodies or {}
        self.fail = fail

    def resolve(self, key: str, *args, **kwargs):
        if self.fail:
            raise RuntimeError("registry down")
        body = self.bodies.get(key)
        return _Resolved(body) if body is not None else None


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("LLM_BAWT_PUBLIC_ORIGIN", raising=False)
    monkeypatch.delenv("BAWTHUB_INSPECTION_KEY_EXPORT_FILE", raising=False)
    # Product defaults unless a test installs registry overrides.
    monkeypatch.setattr("llm_bawt.prompt_registry.get_prompt_resolver", lambda: _FakeResolver())


@pytest.mark.parametrize(
    "value",
    [
        None,
        "",
        "   ",
        "hub.example.com",
        "ftp://hub.example.com",
        "http://",
        "http://[::1",
        "http://[not-an-ip]:13000",
        "http://hub.example.com:notaport",
        "http://user:secret@hub.example.com",
        "http://hub.example.com/app",
        "http://hub.example.com/?q=1",
        "http://hub.example.com/#frag",
    ],
)
def test_no_usable_origin_adds_nothing(monkeypatch, value):
    if value is not None:
        monkeypatch.setenv("LLM_BAWT_PUBLIC_ORIGIN", value)
    assert installation_context_block() == ""
    assert with_installation_context("Bot.") == "Bot."


def test_origin_is_named_and_passwords_are_off_limits(monkeypatch):
    monkeypatch.setenv("LLM_BAWT_PUBLIC_ORIGIN", ORIGIN + "/")
    block = installation_context_block()
    assert block.startswith("## This installation\n")
    assert f"web UI is at {ORIGIN}," in block
    assert "never ask anyone for a password" in block
    # Inspection is only mentioned where the tool is registered.
    assert "inspect_bawthub_tasks" not in block


def test_inspection_hint_follows_the_tool_switch(monkeypatch):
    monkeypatch.setenv("LLM_BAWT_PUBLIC_ORIGIN", ORIGIN)
    monkeypatch.setenv("BAWTHUB_INSPECTION_KEY_EXPORT_FILE", "/run/inspection/key.pem")
    block = installation_context_block()
    assert "inspect_bawthub_tasks" in block and "Bot inspection" in block


def test_block_is_appended_and_stable(monkeypatch):
    monkeypatch.setenv("LLM_BAWT_PUBLIC_ORIGIN", ORIGIN)
    first = with_installation_context("Bot.\n")
    assert first == "Bot.\n\n" + installation_context_block()
    assert with_installation_context("Bot.\n") == first
    assert with_installation_context("") == installation_context_block()


def test_default_backend_composer_appends_block(monkeypatch):
    from llm_bawt.agent_backends.codex import CodexBackend

    monkeypatch.setenv("LLM_BAWT_PUBLIC_ORIGIN", ORIGIN)
    composed = CodexBackend().compose_system_prompt("Bot.", {})
    assert composed.startswith("Bot.\n\n## This installation")


def test_claude_composer_places_block_before_provider_prompt(monkeypatch):
    from llm_bawt.agent_backends.claude_code import ClaudeCodeBackend

    monkeypatch.setenv("LLM_BAWT_PUBLIC_ORIGIN", ORIGIN)
    base = {"model": "claude-test", "bot_id": "b", "provider_system_prompt": "Provider."}
    with patch("llm_bawt.runtime_setting_resolution.resolve_global_runtime_setting", return_value=[]):
        composed = ClaudeCodeBackend().compose_system_prompt("Bot.", base)
    assert (
        composed.index("<runtime-context>")
        < composed.index("Bot.")
        < composed.index("## This installation")
        < composed.index("Provider.")
    )


def test_ipv6_origin_keeps_brackets(monkeypatch):
    monkeypatch.setenv("LLM_BAWT_PUBLIC_ORIGIN", "http://[fd00::5]:13000")
    assert "web UI is at http://[fd00::5]:13000," in installation_context_block()


def test_wording_comes_from_the_prompt_registry(monkeypatch):
    monkeypatch.setenv("LLM_BAWT_PUBLIC_ORIGIN", ORIGIN)
    monkeypatch.setenv("BAWTHUB_INSPECTION_KEY_EXPORT_FILE", "/run/inspection/key.pem")
    resolver = _FakeResolver(
        {
            "agents.installation_context_template": "## Here\nOpen {origin}.",
            "agents.inspection_hint": "Inspect with the tool.",
        }
    )
    monkeypatch.setattr("llm_bawt.prompt_registry.get_prompt_resolver", lambda: resolver)
    assert installation_context_block() == f"## Here\nOpen {ORIGIN}.\nInspect with the tool."


@pytest.mark.parametrize(
    "resolver",
    [_FakeResolver(fail=True), _FakeResolver({"agents.installation_context_template": "Bad {nope}"})],
)
def test_registry_failure_or_bad_template_uses_the_default(monkeypatch, resolver):
    from llm_bawt.chat_prompt_defaults import INSTALLATION_CONTEXT_TEMPLATE

    monkeypatch.setenv("LLM_BAWT_PUBLIC_ORIGIN", ORIGIN)
    monkeypatch.setattr("llm_bawt.prompt_registry.get_prompt_resolver", lambda: resolver)
    assert installation_context_block() == INSTALLATION_CONTEXT_TEMPLATE.format(origin=ORIGIN)


def test_installation_prompts_are_registered():
    from llm_bawt.prompt_definitions import DEFAULT_PROMPT_DEFINITIONS

    template = DEFAULT_PROMPT_DEFINITIONS["agents.installation_context_template"]
    assert template.required_vars == ("origin",)
    assert "{origin}" in template.loader()
    assert "inspect_bawthub_tasks" in DEFAULT_PROMPT_DEFINITIONS["agents.inspection_hint"].loader()


def _codex_dispatch(monkeypatch, config):
    from llm_bawt.agent_backends.agent_bridge import AgentBridgeBackend
    from llm_bawt.agent_backends.codex import CodexBackend

    sent: list[dict] = []

    def fake_stream(self, prompt, config, attachments=None, trigger_message_id=None):
        sent.append(config)
        return iter(())

    monkeypatch.setattr(AgentBridgeBackend, "stream_raw", fake_stream)
    list(CodexBackend().stream_raw("hi", config))
    return sent[0]


def test_unpinned_codex_dispatch_composes_the_block(monkeypatch):
    monkeypatch.setenv("LLM_BAWT_PUBLIC_ORIGIN", ORIGIN)
    sent = _codex_dispatch(monkeypatch, {"system_prompt": "Bot."})
    assert sent["system_prompt"] == with_installation_context("Bot.")
    assert sent["system_prompt"].count("## This installation") == 1


def test_pinned_codex_dispatch_sends_pinned_bytes(monkeypatch):
    monkeypatch.setenv("LLM_BAWT_PUBLIC_ORIGIN", ORIGIN)
    pinned = {"system_prompt": "Pinned exact bytes.", "system_prompt_pinned": True}
    assert _codex_dispatch(monkeypatch, pinned)["system_prompt"] == "Pinned exact bytes."
