"""Proxy-routed turns tell the Claude CLI the real context window.

Background: the CLI assumes 200k for any model name it doesn't recognise and
never auto-compacts a 64k local model — the upstream (Ollama) silently drops
the oldest messages instead. Verified live on qwen3.8:27b-64k: prompt in
78,332 tokens, prompt out 51,516, no error, confabulated recall of the
dropped tool result.
"""

from __future__ import annotations

import pytest

from claude_code_bridge.context_env import (
    AUTO_COMPACT_WINDOW_ENV,
    MAX_CONTEXT_TOKENS_ENV,
    MAX_OUTPUT_TOKENS_ENV,
    compact_window_env,
    output_reserve_for_window,
    proxy_context_window_env,
)
from claude_code_bridge.send_request import SendRequest
from claude_code_bridge.send_stream import ClaudeStreamMixin


class _Harness(ClaudeStreamMixin):
    _proxy_base_url = "http://127.0.0.1:12345"


def _env(
    *, use_proxy: bool, context_window: int | None, compact_threshold: int | None = None
) -> dict:
    return _Harness()._build_sdk_env(
        use_proxy=use_proxy,
        model="local/qwen3.8:27b-64k",
        subagent_model=None,
        force_refresh=False,
        bot_id="qlocal",
        session_key="qlocal:nick",
        thread_session_id="thread-1",
        request_id="request-1",
        context_window=context_window,
        compact_threshold=compact_threshold,
    )


# --- pure helper -----------------------------------------------------------

def test_small_window_sets_both_hints() -> None:
    env = proxy_context_window_env(65_536)
    assert env == {
        MAX_CONTEXT_TOKENS_ENV: "65536",
        MAX_OUTPUT_TOKENS_ENV: "8192",
    }


@pytest.mark.parametrize("window", [None, 0, -1])
def test_unresolved_window_defers_to_cli(window) -> None:
    assert proxy_context_window_env(window) == {}


@pytest.mark.parametrize("window", [200_000, 372_000, 1_000_000])
def test_default_or_larger_window_is_propagated(window) -> None:
    """Unknown proxy models must not inherit Claude Code's 200k fallback."""
    assert proxy_context_window_env(window) == {
        MAX_CONTEXT_TOKENS_ENV: str(window),
        MAX_OUTPUT_TOKENS_ENV: "16384",
    }


@pytest.mark.parametrize(
    ("window", "reserve"),
    [
        (8_192, 4_096),    # floor
        (32_768, 4_096),   # exactly the floor
        (65_536, 8_192),
        (131_072, 16_384),  # ceiling
        (199_999, 16_384),
    ],
)
def test_output_reserve_is_proportional_and_clamped(window, reserve) -> None:
    assert output_reserve_for_window(window) == reserve


def test_reserve_leaves_usable_compact_threshold() -> None:
    """Mirror the CLI's formula: min(0.8*eff, eff-13000), eff = window-reserve.

    With the CLI's own 32k default reserve a 64k window compacts at ~20k —
    every turn. The proportional reserve must land the threshold well above
    a typical 12-15k tool-prefix baseline and below the window.
    """
    window = 65_536
    eff = window - output_reserve_for_window(window)
    threshold = min(int(eff * 0.8), eff - 13_000)
    assert 40_000 <= threshold < window


# --- _build_sdk_env integration -------------------------------------------

def test_proxy_turn_injects_hints_for_small_window() -> None:
    env = _env(use_proxy=True, context_window=65_536)
    assert env[MAX_CONTEXT_TOKENS_ENV] == "65536"
    assert env[MAX_OUTPUT_TOKENS_ENV] == "8192"
    # Existing proxy env is untouched.
    assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:12345"
    assert env["CLAUDE_CODE_SUBAGENT_MODEL"] == "local/qwen3.8:27b-64k"


def test_proxy_turn_injects_hints_for_gpt_5_6_window() -> None:
    env = _env(use_proxy=True, context_window=372_000)
    assert env[MAX_CONTEXT_TOKENS_ENV] == "372000"
    assert env[MAX_OUTPUT_TOKENS_ENV] == "16384"


def test_proxy_turn_routes_every_explicit_subagent_tier_to_proxy_model() -> None:
    env = _Harness()._build_sdk_env(
        use_proxy=True,
        model="openai_chatgpt/gpt-6-astra",
        subagent_model=None,
        force_refresh=False,
        bot_id="snark",
        session_key="snark:nick",
        thread_session_id="thread-1",
        request_id="request-1",
        context_window=272_000,
    )

    expected = "openai_chatgpt/gpt-6-astra"
    assert env["CLAUDE_CODE_SUBAGENT_MODEL"] == expected
    assert env["ANTHROPIC_SMALL_FAST_MODEL"] == expected
    for tier in ("HAIKU", "FABLE", "SONNET", "OPUS"):
        assert env[f"ANTHROPIC_DEFAULT_{tier}_MODEL"] == expected


def test_proxy_turn_routes_tiers_to_configured_subagent_override() -> None:
    env = _Harness()._build_sdk_env(
        use_proxy=True,
        model="openai_chatgpt/gpt-6-astra",
        subagent_model="openai_chatgpt/gpt-5.6-luna",
        force_refresh=False,
        bot_id="snark",
        session_key="snark:nick",
        thread_session_id="thread-1",
        request_id="request-1",
        context_window=272_000,
    )

    expected = "openai_chatgpt/gpt-5.6-luna"
    assert env["CLAUDE_CODE_SUBAGENT_MODEL"] == expected
    for tier in ("HAIKU", "FABLE", "SONNET", "OPUS"):
        assert env[f"ANTHROPIC_DEFAULT_{tier}_MODEL"] == expected


def test_proxy_turn_without_window_sets_no_hints() -> None:
    env = _env(use_proxy=True, context_window=None)
    assert MAX_CONTEXT_TOKENS_ENV not in env
    assert MAX_OUTPUT_TOKENS_ENV not in env


def test_direct_turn_never_sets_hints(monkeypatch) -> None:
    """Anthropic-direct models are recognised by the CLI; don't second-guess it."""
    monkeypatch.setattr(
        "claude_code_bridge.send_stream._get_fresh_oauth_token", lambda **_: None
    )
    env = _env(use_proxy=False, context_window=65_536)
    assert MAX_CONTEXT_TOKENS_ENV not in env
    assert MAX_OUTPUT_TOKENS_ENV not in env
    assert "CLAUDE_CODE_SUBAGENT_MODEL" not in env
    for tier in ("HAIKU", "FABLE", "SONNET", "OPUS"):
        assert f"ANTHROPIC_DEFAULT_{tier}_MODEL" not in env


# --- compact threshold (CLAUDE_CODE_AUTO_COMPACT_WINDOW) --------------------

def test_compact_window_env_passes_threshold_through() -> None:
    assert compact_window_env(200_000) == {AUTO_COMPACT_WINDOW_ENV: "200000"}


@pytest.mark.parametrize("threshold", [None, 0, -5])
def test_unset_compact_threshold_defers_to_cli(threshold) -> None:
    assert compact_window_env(threshold) == {}


@pytest.mark.parametrize("use_proxy", [True, False])
def test_compact_threshold_applies_on_both_paths(monkeypatch, use_proxy) -> None:
    monkeypatch.setattr(
        "claude_code_bridge.send_stream._get_fresh_oauth_token", lambda **_: None
    )
    env = _env(use_proxy=use_proxy, context_window=1_000_000, compact_threshold=400_000)
    assert env[AUTO_COMPACT_WINDOW_ENV] == "400000"


def test_no_compact_threshold_sets_no_window(monkeypatch) -> None:
    env = _env(use_proxy=True, context_window=65_536)
    assert AUTO_COMPACT_WINDOW_ENV not in env


def _fields(**extra) -> dict:
    return {
        "request_id": "r1", "session_key": "caid:nick", "bot_id": "caid", "message": "hi",
        "model": "claude-opus-5-5", **extra,
    }


def test_send_request_parses_compact_threshold() -> None:
    assert SendRequest.from_fields(_fields(compact_threshold="200000")).compact_threshold == 200_000


@pytest.mark.parametrize("raw", ["", "abc", "0", "-1"])
def test_send_request_ignores_invalid_compact_threshold(raw) -> None:
    assert SendRequest.from_fields(_fields(compact_threshold=raw)).compact_threshold is None
