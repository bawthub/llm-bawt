"""TASK-1721: background-job system instructions come from the prompt registry."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from llm_bawt.prompt_definitions import DEFAULT_PROMPT_DEFINITIONS

SYSTEM_KEYS = {
    "history.summarization.system": "summarizes conversations concisely",
    "memory.extraction.fact.system": "Extract facts from conversations",
    "memory.extraction.summary.system": "from conversation summaries",
    "memory.extraction.update.system": "(ADD, UPDATE, DELETE, NONE)",
}
SRC = Path(__file__).resolve().parents[1] / "src" / "llm_bawt" / "memory"


@pytest.mark.parametrize("key, fragment", SYSTEM_KEYS.items())
def test_system_keys_registered_with_unchanged_defaults(key, fragment):
    definition = DEFAULT_PROMPT_DEFINITIONS[key]
    assert fragment in definition.default_body()
    assert definition.required_vars == ()


@pytest.mark.parametrize("path, needle", [
    ("summarization.py", '"system": "You are'),
    ("summarization.py", 'content=SUMMARY_SYSTEM_PROMPT'),
    ("extraction/service.py", 'content="You are'),
])
def test_call_sites_have_no_inline_system_instruction(path, needle):
    assert needle not in (SRC / path).read_text()


def test_summarize_with_client_resolves_system_at_call_time(monkeypatch):
    from llm_bawt.memory.summarization import Session, summarize_session_with_client

    resolver = MagicMock()
    resolver.render.side_effect = lambda key, variables: f"<{key}>"
    monkeypatch.setattr("llm_bawt.prompt_registry.PromptResolver", lambda cfg: resolver)
    client = MagicMock()
    client.query.return_value = "Summary: ok"
    session = Session(session_id="s1", start_timestamp=0, end_timestamp=1,
                      messages=[{"role": "user", "content": "hi", "timestamp": 0}], message_ids=["1"])
    summarize_session_with_client(session, client, config=None)
    system = client.query.call_args.kwargs["messages"][0]
    assert system.role == "system" and system.content == "<history.summarization.system>"


def test_extraction_system_prompt_uses_registry(monkeypatch):
    from llm_bawt.memory.extraction import prompts

    resolver = MagicMock()
    resolver.render.return_value = "OVERRIDE"
    monkeypatch.setattr(prompts, "get_prompt_resolver", lambda: resolver)
    assert prompts.get_extraction_system_prompt("memory.extraction.fact.system") == "OVERRIDE"
    resolver.render.assert_called_once_with(key="memory.extraction.fact.system", variables={})
