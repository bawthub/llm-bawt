"""Tests for prompt registry defaults and validation."""

from pathlib import Path

from sqlmodel import SQLModel, create_engine

from llm_bawt.agent_backends.prompts import AGENTS_DOCS_PROMPT
from llm_bawt.prompt_registry import (
    AGENT_GLOBAL_PROMPT,
    PromptResolver,
    PromptTemplate,
    PromptTemplateStore,
    PromptTemplateVersion,
    extract_placeholders,
)


def test_extract_placeholders_ignores_escaped_braces() -> None:
    body = "JSON example: {{\"ok\": true}} {messages}"
    assert extract_placeholders(body) == ["messages"]


def test_prompt_resolver_renders_default_summarization_prompt() -> None:
    resolver = PromptResolver()
    rendered = resolver.render(
        key="history.summarization.single",
        variables={"messages": "User: hi\nAssistant: hello"},
    )
    assert "User: hi" in rendered
    assert "Output format" in rendered


def test_prompt_validation_flags_missing_required_placeholders() -> None:
    resolver = PromptResolver()
    result = resolver.validate(
        key="history.summarization.single",
        body="Summarize this\nConversation:\n{wrong_name}",
    )
    assert result["valid"] is False
    assert "messages" in result["missing_required"]


def test_prompt_resolver_renders_memory_maintenance_prompt() -> None:
    resolver = PromptResolver()
    rendered = resolver.render(
        key="memory.maintenance.intent_content_only",
        variables={"fact": "User prefers compact summaries"},
    )
    assert "User prefers compact summaries" in rendered
    assert "intent phrase" in rendered


def test_scoped_commit_prompt_is_catalog_owned_and_renders_runtime_scope() -> None:
    resolver = PromptResolver()
    definition = resolver.definition_for("chat.scoped_commit")

    assert definition is not None
    assert definition.required_vars == ("scope",)
    rendered = resolver.render(
        key="chat.scoped_commit",
        variables={"scope": "Repository: bawthub\n- frontend/src/app/chat/ChangedFilesRow.tsx"},
    )
    assert "Repository: bawthub" in rendered
    assert "selected conversation scope" in rendered
    assert "stage only your hunks" in rendered
    assert "Do not amend or push" in rendered


def test_agent_global_prompt_explains_playwright_artifact_workflow() -> None:
    assert "browser_take_screenshot` without `filename`" in AGENT_GLOBAL_PROMPT
    assert "returned `original`, `preview`, or `thumb` URL" in AGENT_GLOBAL_PROMPT
    assert "suppresses the inline-image/Garage attachment path" in AGENT_GLOBAL_PROMPT


def test_agent_global_prompt_requires_durable_manager_callbacks() -> None:
    assert "default asynchronous bots_send_message mode; it is durable" in AGENT_GLOBAL_PROMPT
    assert "redirects that SAME turn" in AGENT_GLOBAL_PROMPT
    assert "delivery='when_idle' only when steering is intentionally undesired" in AGENT_GLOBAL_PROMPT
    assert "force=true never authorizes concurrent agent turns" in AGENT_GLOBAL_PROMPT
    assert "Never poll as the primary scheduler" in AGENT_GLOBAL_PROMPT


def test_agent_global_prompt_makes_canonical_branch_the_explicit_exception() -> None:
    assert "If on the default branch, branch first" in AGENT_GLOBAL_PROMPT
    assert "concrete exception to the generic branch-first default" in AGENT_GLOBAL_PROMPT
    assert "Stay on that branch and commit there" in AGENT_GLOBAL_PROMPT
    assert "request a commit handoff" in AGENT_GLOBAL_PROMPT


def test_inter_bot_docs_do_not_restore_queue_only_contract() -> None:
    repo = Path(__file__).resolve().parents[1]
    authoritative = [
        repo / "docs" / "INTER_BOT_COMMUNICATION.md",
        repo / "docs" / "MCP_SERVER.md",
    ]
    combined = "\n".join(path.read_text() for path in authoritative)

    assert "steers an active Claude Code turn in place" in combined
    assert "force=true` never authorizes concurrent agent turns" in combined
    assert "default immediate async" not in combined
    assert "may overlap the manager" not in combined
    assert "wakes one fresh manager turn after idle" not in combined


def test_agents_docs_seed_keeps_visual_review_distinct_from_coordinator_closure() -> None:
    assert "Visual UI/UX Nick can see stays in `REVIEW` for Nick" in AGENTS_DOCS_PROMPT
    assert "put the dev URL and\n  exact steps to check in `response`" in AGENTS_DOCS_PROMPT
    assert "a coordinator (Snark/Rook)\n  closes it to `COMPLETED`" in AGENTS_DOCS_PROMPT
    assert "a recorded review, a pushed commit,\n  and verification evidence" in AGENTS_DOCS_PROMPT
    assert "Only humans transition tasks to `COMPLETED`" not in AGENTS_DOCS_PROMPT


def test_prompt_store_seed_defaults_is_idempotent() -> None:
    store = PromptTemplateStore.__new__(PromptTemplateStore)
    store.config = None
    store.connection_url = "sqlite://test"
    store.engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(
        store.engine,
        tables=[PromptTemplate.__table__, PromptTemplateVersion.__table__],
    )

    first = store.seed_defaults()
    second = store.seed_defaults()

    assert first["created"] > 0
    assert second["created"] == 0
    assert second["skipped"] == first["total"]
