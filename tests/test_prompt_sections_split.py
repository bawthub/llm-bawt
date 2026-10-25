"""TASK-1249 A1: prompt section renderers live in core/prompt_sections.py.

Every PROMPT_MANIFEST entry must resolve to a renderer that BaseLLMBawt
inherits from PromptSectionsMixin, and base.py must not grow its own
``_sec_*`` copies back (the monolith gate this split exists for).
"""
import ast
from pathlib import Path

from llm_bawt.core.base import BaseLLMBawt
from llm_bawt.core.prompt_manifest import PROMPT_MANIFEST
from llm_bawt.core.prompt_sections import PromptSectionsMixin

BASE_PY = Path(__file__).resolve().parents[1] / "src" / "llm_bawt" / "core" / "base.py"


def test_every_manifest_method_comes_from_the_sections_mixin():
    assert PROMPT_MANIFEST
    for spec in PROMPT_MANIFEST:
        method = getattr(BaseLLMBawt, spec.method)
        assert method is getattr(PromptSectionsMixin, spec.method), spec.method
        assert spec.method in vars(PromptSectionsMixin), spec.method


def test_base_py_defines_no_section_renderers():
    tree = ast.parse(BASE_PY.read_text())
    defined = [
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_sec_")
    ]
    assert defined == []
