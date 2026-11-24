"""Agents learn their installation's UI address from the composed prompt (TASK-2523)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from llm_bawt.agent_backends.installation_context import (
    installation_context_block,
    with_installation_context,
)

ORIGIN = "http://hub.example.com:13000"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("LLM_BAWT_PUBLIC_ORIGIN", raising=False)
    monkeypatch.delenv("BAWTHUB_INSPECTION_KEY_EXPORT_FILE", raising=False)


@pytest.mark.parametrize("value", [None, "", "   ", "hub.example.com", "ftp://hub.example.com", "http://"])
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
