"""Core LLM processing components.

This package provides the main orchestration for llm-bawt:
- PromptBuilder: Template-based system prompt assembly
- BaseLLMBawt: Shared logic for the service orchestrator
- ModelLifecycleManager: Singleton for model loading/unloading
"""

from .prompt_builder import PromptBuilder, PromptSection, SectionPosition
from .base import BaseLLMBawt
from .model_lifecycle import ModelLifecycleManager, get_model_lifecycle

__all__ = [
    "PromptBuilder",
    "PromptSection",
    "SectionPosition",
    "BaseLLMBawt",
    "ModelLifecycleManager",
    "get_model_lifecycle",
]
