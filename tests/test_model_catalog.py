import pytest

from llm_bawt.model_catalog import (
    AccessPath,
    AmbiguousModelError,
    IncompatibleModelError,
    ModelCatalog,
    ModelEndpoint,
    ModelIdentity,
    ProtocolCompatibility,
)


def _endpoint(
    endpoint_id: int,
    model_key: str,
    access_key: str,
    vendor: str,
    protocol: str,
    upstream: str,
    *,
    engine_kind: str | None = None,
    serving_config=None,
    system_prompt_overrides=None,
):
    return ModelEndpoint(
        id=endpoint_id,
        model=ModelIdentity(endpoint_id, model_key, vendor, model_key),
        access_path=AccessPath(
            endpoint_id,
            access_key,
            vendor,
            protocol,
            None,
            "test",
            engine_kind=engine_kind,
            system_prompt_overrides=system_prompt_overrides or {},
        ),
        upstream_model_id=upstream,
        serving_config=serving_config or {},
    )


def test_shared_openai_oauth_endpoint_resolves_differently_by_harness():
    endpoint = _endpoint(
        20,
        "gpt-5.6-sol",
        "openai-oauth",
        "openai",
        "responses",
        "gpt-5.6-sol",
    )
    catalog = ModelCatalog([endpoint])

    codex = catalog.resolve("gpt-5.6-sol", harness="codex")
    proxy = catalog.resolve("gpt-5.6-sol", harness="claude-proxy")

    assert codex["type"] == "agent_backend"
    assert codex["backend"] == "codex"
    assert codex["model_id"] == "gpt-5.6-sol"
    assert proxy["type"] == "claude-code"
    assert proxy["backend"] == "claude-code"
    assert proxy["model_id"] == "openai_chatgpt/gpt-5.6-sol"
    assert codex["endpoint_id"] == proxy["endpoint_id"] == 20


def test_openai_platform_responses_routes_through_proxy_not_native_codex():
    endpoint = _endpoint(
        21,
        "gpt-6-astra",
        "openai-responses",
        "openai",
        "responses",
        "gpt-6-astra",
    )
    catalog = ModelCatalog([endpoint])

    proxy = catalog.resolve("gpt-6-astra", harness="claude-proxy")

    assert proxy["type"] == "claude-code"
    assert proxy["backend"] == "claude-code"
    assert proxy["model_id"] == "openai/gpt-6-astra"
    assert not ProtocolCompatibility.is_compatible("codex", endpoint.access_path)


def test_openai_chat_completions_path_is_not_advertised_to_agent_proxy():
    endpoint = _endpoint(
        22,
        "gpt-chat",
        "openai-api",
        "openai",
        "chat-completions",
        "gpt-chat",
    )

    assert not ProtocolCompatibility.is_compatible(
        "claude-proxy", endpoint.access_path
    )


def test_provider_system_prompt_resolves_only_for_active_harness():
    instruction = "Use OpenAI-compatible tool semantics."
    endpoint = _endpoint(
        20,
        "gpt-5.6-sol",
        "openai-oauth",
        "openai",
        "responses",
        "gpt-5.6-sol",
        system_prompt_overrides={"claude-proxy": f"  {instruction}  "},
    )
    catalog = ModelCatalog([endpoint])

    proxy = catalog.resolve("gpt-5.6-sol", harness="claude-proxy")
    codex = catalog.resolve("gpt-5.6-sol", harness="codex")

    assert proxy["provider_system_prompt"] == instruction
    assert "provider_system_prompt" not in codex


def test_blank_provider_system_prompt_is_absent():
    endpoint = _endpoint(
        20,
        "gpt-5.6-sol",
        "openai-oauth",
        "openai",
        "responses",
        "gpt-5.6-sol",
        system_prompt_overrides={"claude-proxy": " \n "},
    )

    resolved = ModelCatalog([endpoint]).resolve(
        "gpt-5.6-sol", harness="claude-proxy"
    )

    assert "provider_system_prompt" not in resolved


def test_bare_unique_proxy_ref_preserves_legacy_provider_prefix():
    endpoint = _endpoint(
        2,
        "grok-4.5",
        "xai-responses",
        "xai",
        "responses",
        "grok-4.5",
    )

    # A bare (no-harness) ref can't know the endpoint is proxy-only, so type is
    # derived from the vendor; the provider prefix on model_id is still preserved.
    resolved = ModelCatalog([endpoint]).resolve("grok-4.5")
    assert resolved["type"] == "grok"
    assert resolved["model_id"] == "xai/grok-4.5"


def test_kimi_coding_proxy_gets_provider_namespace():
    endpoint = _endpoint(
        4,
        "kimi-k3",
        "kimi-coding-chat",
        "kimi",
        "chat-completions",
        "k3",
    )

    resolved = ModelCatalog([endpoint]).resolve(
        "kimi-k3", harness="claude-proxy"
    )
    assert resolved["type"] == "claude-code"
    assert resolved["model_id"] == "kimi_coding/k3"


def test_protocol_compatibility_is_single_filter_rule():
    anthropic = _endpoint(
        1, "claude-opus", "anthropic-oauth", "anthropic", "anthropic-messages", "opus"
    )
    xai = _endpoint(2, "grok", "xai-responses", "xai", "responses", "grok-4")
    zai = _endpoint(
        3, "glm", "zai-anthropic", "zai", "anthropic-messages", "glm-5.2"
    )

    assert ProtocolCompatibility.is_compatible("claude-code", anthropic.access_path)
    assert not ProtocolCompatibility.is_compatible("claude-code", zai.access_path)
    assert not ProtocolCompatibility.is_compatible("codex", anthropic.access_path)
    assert not ProtocolCompatibility.is_compatible("codex", xai.access_path)
    assert ProtocolCompatibility.is_compatible("claude-proxy", xai.access_path)
    assert ProtocolCompatibility.is_compatible("claude-proxy", zai.access_path)
    assert not ProtocolCompatibility.is_compatible(
        "claude-proxy", anthropic.access_path
    )


def test_bare_collision_requires_harness_or_endpoint_ref():
    chat = _endpoint(1, "same", "openai-api", "openai", "chat-completions", "same")
    responses = _endpoint(2, "same", "openai-oauth", "openai", "responses", "same")
    catalog = ModelCatalog([chat, responses])

    with pytest.raises(AmbiguousModelError):
        catalog.resolve("same")
    assert catalog.resolve("same", harness="chat")["endpoint_id"] == 1
    assert catalog.resolve("same", harness="codex")["endpoint_id"] == 2
    assert catalog.resolve("same@openai-oauth")["endpoint_id"] == 2


def test_incompatible_explicit_endpoint_is_rejected():
    endpoint = _endpoint(
        1, "claude-opus", "anthropic-oauth", "anthropic", "anthropic-messages", "opus"
    )
    catalog = ModelCatalog([endpoint])

    with pytest.raises(IncompatibleModelError):
        catalog.resolve(1, harness="chat")


def test_endpoint_reasoning_effort_is_flattened_for_bridge_dispatch():
    endpoint = _endpoint(
        21,
        "astra",
        "openai-oauth",
        "openai",
        "responses",
        "gpt-6-astra",
        serving_config={
            "reasoning_effort": "high",
            "responses_transport": "sse",
        },
    )

    resolved = ModelCatalog([endpoint]).resolve("astra", harness="claude-proxy")

    assert resolved["reasoning_effort"] == "high"
    assert resolved["responses_transport"] == "sse"


def test_local_serving_config_is_flattened_for_existing_consumers():
    endpoint = _endpoint(
        3,
        "dolphin",
        "local-llamacpp",
        "local",
        "chat-completions",
        "dolphin",
        engine_kind="llama-cpp",
        serving_config={
            "repo_id": "org/repo",
            "filename": "model.gguf",
            "n_gpu_layers": 33,
        },
    )
    catalog = ModelCatalog([endpoint])

    resolved = catalog.resolve("dolphin", harness="chat")
    assert resolved["type"] == "gguf"
    assert resolved["repo_id"] == "org/repo"
    assert resolved["filename"] == "model.gguf"
    assert resolved["n_gpu_layers"] == 33
    assert resolved["engine_kind"] == "llama-cpp"


def test_local_anthropic_access_path_gets_local_namespace():
    """Ollama / llama-server native Anthropic surface → LocalAdapter ("local/")."""
    endpoint = _endpoint(
        5,
        "qwen3.8-27b-local",
        "local-anthropic",
        "local",
        "anthropic-messages",
        "qwen3.8:27b",
    )
    resolved = ModelCatalog([endpoint]).resolve(
        "qwen3.8-27b-local", harness="claude-proxy"
    )
    assert resolved["type"] == "claude-code"
    assert resolved["model_id"] == "local/qwen3.8:27b"


def test_local_chat_completions_path_keeps_no_namespace():
    """The app-served ollama chat path must NOT be routed to the proxy."""
    endpoint = _endpoint(
        6,
        "qwen3.8-27b-chat",
        "ollama",
        "local",
        "chat-completions",
        "qwen3.8:27b",
    )
    resolved = ModelCatalog([endpoint]).resolve("qwen3.8-27b-chat")
    assert resolved["model_id"] == "qwen3.8:27b"


def test_model_compact_threshold_reaches_dispatch_config_only_when_set():
    from dataclasses import replace

    endpoint = _endpoint(31, "opus", "anthropic-oauth", "anthropic", "anthropic-messages", "claude-opus-5-5")
    assert "compact_threshold" not in ModelCatalog([endpoint]).resolve("opus", harness="claude-code")

    tuned = replace(endpoint, model=replace(endpoint.model, compact_threshold=400_000))
    assert ModelCatalog([tuned]).resolve("opus", harness="claude-code")["compact_threshold"] == 400_000
