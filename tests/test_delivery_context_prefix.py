"""TASK-2502: the fleet-delivery lineage marker rides the outbound message only."""

from types import SimpleNamespace

from llm_bawt.core.base import BaseLLMBawt


def _agent(monkeypatch):
    monkeypatch.setattr(
        "llm_bawt.prompt_registry.get_prompt_resolver",
        lambda _config: SimpleNamespace(resolve=lambda _key: None),
    )
    return SimpleNamespace(
        _assemble_system_builder=lambda _prompt: SimpleNamespace(
            build=lambda: "system", enabled_sections=[],
        ),
        config_resolver=None, config=None, debug=False,
        model_definition={"type": "agent_backend"},
        _inject_user_prefix=False, _tts_mode=False,
    )


def test_prefix_precedes_user_text_and_suffix_on_outbound_copy(monkeypatch):
    agent = _agent(monkeypatch)
    messages = BaseLLMBawt._build_context_messages(
        agent, "do the task", context_suffix="manifest", context_prefix='{"source":"fleet-delivery"}',
    )
    assert messages[-1].content == '{"source":"fleet-delivery"}\n\ndo the task\n\nmanifest'


def test_no_prefix_leaves_outbound_unchanged(monkeypatch):
    agent = _agent(monkeypatch)
    messages = BaseLLMBawt._build_context_messages(agent, "do the task")
    assert messages[-1].content == "do the task"


def test_style_keywords_and_system_prompt_see_clean_body(monkeypatch):
    seen = []
    agent = _agent(monkeypatch)
    agent._assemble_system_builder = lambda prompt: (seen.append(prompt), SimpleNamespace(
        build=lambda: "system", enabled_sections=[],
    ))[1]
    BaseLLMBawt._build_context_messages(agent, "body", context_prefix="marker")
    assert seen == ["body"]
