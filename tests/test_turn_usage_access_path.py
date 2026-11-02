"""Turn usage freezes the serving access path at write time (TASK-1056)."""

import json
from unittest.mock import patch

import pytest

from llm_bawt.model_catalog import AccessPath, ModelCatalog, ModelEndpoint, ModelIdentity
from llm_bawt.service.turn_usage import access_path_snapshot, encode_token_usage


def _catalog(access_key="openai-oauth", auth_mechanism="oauth") -> ModelCatalog:
    return ModelCatalog(
        [
            ModelEndpoint(
                id=20,
                model=ModelIdentity(20, "gpt-6-sol", "openai", "GPT-6 Sol"),
                access_path=AccessPath(
                    11, access_key, "openai", "responses",
                    "https://chatgpt.com/backend-api/codex", auth_mechanism,
                ),
                upstream_model_id="gpt-6-sol",
            ),
            ModelEndpoint(
                id=21,
                model=ModelIdentity(21, "qwen-local", "local", "Qwen"),
                access_path=AccessPath(
                    12, "local-vllm", "local", "chat-completions",
                    "http://local:8683", "none", engine_kind="vllm",
                ),
                upstream_model_id="qwen",
            ),
        ]
    )


USAGE = {"input_tokens": 10, "cache_read_tokens": 90, "output_tokens": 5, "context_window": 272000}


def test_endpoint_by_ref_is_exact_and_never_guesses_bare_model_keys() -> None:
    catalog = _catalog()
    assert catalog.endpoint_by_ref("gpt-6-sol@openai-oauth").id == 20
    assert catalog.endpoint_by_ref(" gpt-6-sol@openai-oauth ").id == 20
    assert catalog.endpoint_by_ref("gpt-6-sol") is None
    assert catalog.endpoint_by_ref(None) is None


def test_snapshot_carries_access_path_facts_not_labels() -> None:
    assert access_path_snapshot(_catalog(), "qwen-local@local-vllm") == {
        "key": "local-vllm",
        "vendor": "local",
        "protocol": "chat-completions",
        "auth_mechanism": "none",
        "engine_kind": "vllm",
        "billing_mode": "unknown",
    }


def test_encode_stamps_access_path_and_keeps_usage() -> None:
    encoded = json.loads(encode_token_usage(USAGE, model_ref="gpt-6-sol@openai-oauth", catalog=_catalog()))
    assert {key: encoded[key] for key in USAGE} == USAGE
    assert encoded["access_path"]["key"] == "openai-oauth"
    assert encoded["access_path"]["auth_mechanism"] == "oauth"


@pytest.mark.parametrize(("key", "auth", "billing_mode"), [
    ("openai-oauth", "oauth", "subscription"),
    ("openai-api", "api-key", "metered"),
    ("kimi-coding-chat", "api-key", "subscription"),
    ("zai-anthropic", "api-key", "unknown"),
    ("future-api", "api-key", "unknown"),
])
def test_encode_freezes_owner_billing_without_auth_inference(key, auth, billing_mode) -> None:
    catalog = _catalog(key, auth)
    encoded = json.loads(encode_token_usage(USAGE, model_ref=f"gpt-6-sol@{key}", catalog=catalog))
    assert encoded["access_path"]["billing_mode"] == billing_mode
    assert encoded["access_path"]["auth_mechanism"] == auth
    assert "access_path" not in USAGE


@pytest.mark.parametrize("snapshot", [
    {"key": "openai-oauth", "vendor": "openai", "protocol": "responses"},  # Legacy, never backfilled.
    {"key": "openai-oauth", "billing_mode": "metered"},  # Frozen metadata wins over today's owner.
    {"key": "openai-oauth", "billing_mode": "unknown"},
    {},
    None,
])
def test_encode_keeps_a_caller_supplied_snapshot(snapshot) -> None:
    frozen = {**USAGE, "access_path": snapshot}
    with patch(
        "llm_bawt.service.turn_usage.access_path_snapshot",
        side_effect=AssertionError("replaced caller snapshot"),
    ):
        encoded = json.loads(encode_token_usage(frozen, model_ref="qwen-local@local-vllm", catalog=_catalog()))
    assert encoded == frozen


def test_encode_degrades_without_catalog_or_known_ref() -> None:
    assert "access_path" not in json.loads(encode_token_usage(USAGE, model_ref="gpt-6-sol@openai-oauth", catalog=None))
    assert "access_path" not in json.loads(encode_token_usage(USAGE, model_ref="unknown@nowhere", catalog=_catalog()))
    assert encode_token_usage(None, model_ref="gpt-6-sol@openai-oauth", catalog=_catalog()) is None
