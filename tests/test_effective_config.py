from types import SimpleNamespace

from llm_bawt.core.effective_config import describe_effective_config


class _Resolved:
    def __init__(self, value, source="test"):
        self.value = value
        self.source = source


class _Builder:
    enabled_sections = []

    @staticmethod
    def build():
        return "bot prompt"


class _Resolver:
    @staticmethod
    def resolve_scalar(key, fallback=None):
        return _Resolved(fallback)

    @staticmethod
    def resolve_config_setting(key):
        return _Resolved(False)


def test_effective_config_reports_provider_prompt_provenance_and_order():
    model_definition = {
        "type": "claude-code",
        "access_path": "openai-oauth",
        "harness": "claude-proxy",
        "provider_system_prompt": "Provider instructions.",
    }
    owner = SimpleNamespace(
        model_definition=model_definition,
        bot=SimpleNamespace(endpoint_id=20, harness="claude-proxy"),
        config=SimpleNamespace(
            TEMPERATURE=0.7,
            TOP_P=1.0,
            MEMORY_MIN_RELEVANCE=0.3,
            resolve_model=lambda *args, **kwargs: model_definition,
        ),
        config_resolver=_Resolver(),
        bot_id="loopy",
        user_id="operator",
        resolved_model_alias="gpt-5.6-sol",
        _include_summaries=True,
        _tts_mode=False,
        _assemble_system_builder=lambda prompt: _Builder(),
    )

    result = describe_effective_config(owner)
    provider = next(
        item
        for item in result["downstream_augmentation"]
        if item["name"] == "provider_system_prompt"
    )

    assert provider["source"] == "access_path:openai-oauth"
    assert provider["harness"] == "claude-proxy"
    assert provider["char_len"] == len("Provider instructions.")
    assert provider["order"] == 3
