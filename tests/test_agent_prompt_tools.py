"""Regression coverage for native-agent tool prompt ownership (TASK-633)."""

from types import SimpleNamespace

import pytest

from llm_bawt.core.base import BaseLLMBawt
from llm_bawt.core.prompt_builder import PromptBuilder, SectionPosition
from llm_bawt.utils.config import Config


LEGACY_PROMPT_MARKERS = (
    "Use this EXACT format for tool calls",
    "<tool_call>",
    "Call ONE tool at a time",
    "### Tool Selection:",
    "### Self-Development",
)


class _PromptProbe:
    """Minimal object that exercises BaseLLMBawt's real manifest walker."""

    _prompt_bot_type = BaseLLMBawt._prompt_bot_type
    _walk_manifest = BaseLLMBawt._walk_manifest
    _assemble_system_builder = BaseLLMBawt._assemble_system_builder
    _sec_working_memory = BaseLLMBawt._sec_working_memory
    _working_memory_section = None

    def __init__(self, *, model_type: str, harness: str):
        self.model_definition = {"type": model_type, "harness": harness}
        self.bot = SimpleNamespace(harness=harness)
        self._prompt_builder = PromptBuilder().add_section(
            "base_prompt",
            "native harness regression probe",
            position=SectionPosition.BASE_PROMPT,
        )

    def _sec_tools(self, builder: PromptBuilder, prompt: str) -> None:
        # Deliberately use every deleted marker: if the manifest ever routes this
        # chat-only renderer into an agent prompt, the regression is obvious.
        builder.add_section(
            "tools",
            "\n".join(LEGACY_PROMPT_MARKERS),
            position=SectionPosition.TOOLS,
        )

    def _sec_client_context(self, builder: PromptBuilder, prompt: str) -> None:
        return None

    def _sec_cold_start_memory(self, builder: PromptBuilder, prompt: str) -> None:
        return None

    def _sec_tts_output(self, builder: PromptBuilder, prompt: str) -> None:
        return None

    def _sec_agent_global_prompt(self, builder: PromptBuilder, prompt: str) -> None:
        return None

    def _sec_tag_context(self, builder: PromptBuilder, prompt: str) -> None:
        return None


@pytest.mark.parametrize(
    ("model_type", "harness"),
    (
        ("claude-code", "claude-code"),
        ("agent_backend", "codex"),
        ("agent_backend", "openclaw"),
    ),
)
def test_native_agent_harness_prompt_omits_app_tool_protocol(model_type, harness):
    probe = _PromptProbe(model_type=model_type, harness=harness)

    builder = probe._assemble_system_builder("run tools")
    prompt = builder.build()

    assert not builder.has_section("tools")
    for marker in LEGACY_PROMPT_MARKERS:
        assert marker not in prompt


def test_chat_prompt_still_receives_its_model_format_instructions():
    probe = _PromptProbe(model_type="gguf", harness="chat")

    builder = probe._assemble_system_builder("run tools")

    assert builder.has_section("tools")


@pytest.mark.parametrize(
    "model_definition",
    (
        {"tool_support": "xml"},
        {"tool_format": "xml"},
        {"type": "unknown-provider"},
    ),
)
def test_retired_xml_configuration_degrades_to_react(model_definition):
    assert Config.get_tool_format(object(), model_def=model_definition) == "react"
