import pytest

from llm_bawt.memory.model_catalog_migration import (
    _CREATE_SCHEMA_SQL,
    _REMOVED_HARNESS,
    _derive_harness,
    _refuse_removed_harness,
)


def test_access_path_schema_adds_harness_prompt_mapping_idempotently():
    assert "system_prompt_overrides JSONB NOT NULL DEFAULT '{}'::jsonb" in _CREATE_SCHEMA_SQL
    assert "ADD COLUMN IF NOT EXISTS system_prompt_overrides" in _CREATE_SCHEMA_SQL


def test_derive_harness_maps_backend_to_native_harness():
    assert _derive_harness("codex", "openai", "responses") == "codex"


def test_derive_harness_claude_code_is_direct_only_for_native_anthropic():
    # Native Anthropic messages → the direct claude-code harness.
    assert (
        _derive_harness("claude-code", "anthropic", "anthropic-messages")
        == "claude-code"
    )
    # A non-Anthropic vendor (or non-messages protocol) must ride the proxy.
    assert _derive_harness("claude-code", "zai", "anthropic-messages") == "claude-proxy"
    assert _derive_harness("claude-code", "openai", "responses") == "claude-proxy"
    assert _derive_harness("claude-code", "xai", "chat-completions") == "claude-proxy"


def test_derive_harness_defaults_to_chat_for_direct_llm_backends():
    assert _derive_harness(None, "openai", "chat-completions") == "chat"
    assert _derive_harness("", "xai", "chat-completions") == "chat"
    assert _derive_harness("grok", "xai", "chat-completions") == "chat"


class _Rows:
    def __init__(self, slugs):
        self._slugs = slugs

    def scalars(self):
        return self

    def all(self):
        return list(self._slugs)


class _Conn:
    def __init__(self, slugs):
        self.slugs = slugs
        self.params = None

    def execute(self, _sql, params=None):
        self.params = params
        return _Rows(self.slugs)


def test_migration_refuses_rows_on_the_removed_harness():
    conn = _Conn(["claw-a", "claw-b"])
    with pytest.raises(RuntimeError, match="claw-a, claw-b"):
        _refuse_removed_harness(conn)
    assert conn.params == {"removed": _REMOVED_HARNESS}


def test_migration_passes_when_no_row_uses_the_removed_harness():
    _refuse_removed_harness(_Conn([]))
