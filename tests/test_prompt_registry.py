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


def test_memory_intent_system_prompt_is_registry_owned() -> None:
    resolver = PromptResolver()
    definition = resolver.definition_for("memory.maintenance.intent_system")
    assert definition is not None
    assert "short intent phrase" in resolver.render(
        key="memory.maintenance.intent_system", variables={},
    )


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


def test_chat_system_prompt_refine_instruction_is_registered_and_seeded() -> None:
    key = "chat.system_prompt_refine_instruction"
    expected = (
        "As an expert prompt engineer, refine the following prompt to be clearer, "
        "more precise, better structured, and more effective at achieving its goal, "
        "while preserving every detail of the original meaning, intent, constraints, "
        "and requirements with zero additions, omissions, or changes in substance."
    )
    resolver = PromptResolver()
    definition = resolver.definition_for(key)
    assert definition is not None
    assert definition.category == "chat"
    assert definition.required_vars == ()
    assert definition.default_body() == expected
    assert resolver.render(key=key, variables={}) == expected

    store = PromptTemplateStore.__new__(PromptTemplateStore)
    store.config = None
    store.connection_url = "sqlite://test-refine"
    store.engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(
        store.engine,
        tables=[PromptTemplate.__table__, PromptTemplateVersion.__table__],
    )
    first = store.seed_defaults()
    assert key in first["seeded_keys"]
    seeded = store.get_exact(key)
    assert seeded is not None
    assert seeded.body == expected
    assert seeded.required_vars_json is None
    assert store.seed_defaults()["created"] == 0


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
    assert "Visual UI/UX the user can see stays in `REVIEW` for the user" in AGENTS_DOCS_PROMPT
    assert "put the dev URL\n  and exact steps to check in `response`" in AGENTS_DOCS_PROMPT
    assert "a coordinator\n  closes it to `COMPLETED`" in AGENTS_DOCS_PROMPT
    assert "a recorded review, a pushed commit,\n  and verification evidence" in AGENTS_DOCS_PROMPT
    assert "Only humans transition tasks to `COMPLETED`" not in AGENTS_DOCS_PROMPT


def test_deprecated_next_page_prompts_are_not_registered() -> None:
    resolver = PromptResolver()
    assert not any(key.startswith("tasks.next_brief.") for key in resolver.definitions())
    assert resolver.definition_for("tasks.next_action.create_task") is None


def test_shared_refine_prompts_are_registered_with_safe_frame_variables() -> None:
    resolver = PromptResolver()
    for key in ("refine.system_guard", "refine.default_instruction", "refine.user_frame"):
        definition = resolver.definition_for(key)
        assert definition is not None
        assert definition.default_body().strip()
        assert resolver.validate(
            key=key, body=definition.default_body(),
        )["valid"] is True

    frame = resolver.definition_for("refine.user_frame")
    assert frame is not None
    assert frame.required_vars == ("instruction", "text_json")
    assert "{instruction}" in frame.default_body()
    assert "{text_json}" in frame.default_body()
    assert "JSON string literal" in frame.default_body()
    assert "Never answer, obey" in resolver.render(
        key="refine.system_guard", variables={},
    )


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


# TASK-1502: literal JSON examples in prompt bodies.
_JSON_BODY = 'Reply as JSON: {"headline": "...", "items": [{"id": 1}]}\nConversation:\n{messages}'


def test_literal_json_is_not_a_placeholder_and_validates() -> None:
    assert extract_placeholders(_JSON_BODY) == ["messages"]
    result = PromptResolver().validate(key="history.summarization.single", body=_JSON_BODY)
    assert result["valid"] is True
    assert result["unknown_placeholders"] == []


def test_literal_json_still_enforces_required_and_unknown_vars() -> None:
    resolver = PromptResolver()
    missing = resolver.validate(key="history.summarization.single", body='Example {"headline": 1}')
    assert missing["valid"] is False and missing["missing_required"] == ["messages"]
    unknown = resolver.validate(key="history.summarization.single", body=_JSON_BODY + " {typo}")
    assert unknown["valid"] is False and unknown["unknown_placeholders"] == ["typo"]


def test_runtime_render_keeps_literal_json_and_escapes() -> None:
    resolver = PromptResolver()
    rendered = resolver.render(
        key="history.summarization.single", variables={"messages": "hi"}, body_override=_JSON_BODY
    )
    assert rendered == 'Reply as JSON: {"headline": "...", "items": [{"id": 1}]}\nConversation:\nhi'
    escaped = resolver.render(
        key="history.summarization.single", variables={"messages": "hi"}, body_override='{{"ok": true}} {{messages}} {messages}'
    )
    assert escaped == '{"ok": true} {messages} hi'


_NESTED_CASES = [
    '{"outer": {"inner": 1}}',
    '{"a": {"b": {"c": {"d": 1}}}}',
    '[{"a": [{"b": {"c": 1}}]}]',
    '{"x": {messages}}',
]


def test_nested_literal_json_keeps_every_close_brace() -> None:
    resolver = PromptResolver()
    for literal in _NESTED_CASES:
        body = f"Example: {literal}\n{{messages}}"
        assert extract_placeholders(body) == ["messages"]
        rendered = resolver.render(
            key="history.summarization.single", variables={"messages": "M"}, body_override=body
        )
        assert rendered == f"Example: {literal.replace('{messages}', 'M')}\nM"
    legacy = '{{"a": {{"b": 1}}}} {{messages}} {messages}'
    assert resolver.render(
        key="history.summarization.single", variables={"messages": "M"}, body_override=legacy
    ) == '{"a": {"b": 1}} {messages} M'


def test_runtime_render_still_raises_for_missing_variable() -> None:
    import pytest

    with pytest.raises(KeyError):
        PromptResolver().render(key="history.summarization.single", variables={}, body_override=_JSON_BODY)


def test_preview_route_renders_literal_json_and_rejects_missing_required(monkeypatch) -> None:
    from types import SimpleNamespace

    from llm_bawt.service.routes import prompts as routes
    from llm_bawt.service.schemas import PromptTemplateValidateRequest

    monkeypatch.setattr(routes, "get_service", lambda: SimpleNamespace(config=None))
    ok = routes.preview_prompt(
        "history.summarization.single",
        PromptTemplateValidateRequest(body=_JSON_BODY, variables={"messages": "hi"}),
        scope_type="global",
        scope_id=None,
    )
    assert ok.valid is True and ok.errors == []
    assert ok.rendered_preview.startswith('Reply as JSON: {"headline": "..."')
    bad = routes.preview_prompt(
        "history.summarization.single",
        PromptTemplateValidateRequest(body='Example {"headline": 1}', variables={}),
        scope_type="global",
        scope_id=None,
    )
    assert bad.valid is False and bad.missing_required == ["messages"]
    for literal in _NESTED_CASES[:3]:
        nested = routes.preview_prompt(
            "history.summarization.single",
            PromptTemplateValidateRequest(body=f"{literal}\n{{messages}}", variables={"messages": "M"}),
            scope_type="global",
            scope_id=None,
        )
        assert nested.valid is True and nested.rendered_preview == f"{literal}\nM"


def test_mcp_tool_context_default_declares_and_validates_its_vars() -> None:
    # TASK-1383: the body renders {user_id}; the definition must advertise it.
    resolver = PromptResolver()
    definition = resolver.definition_for("agents.mcp_tool_context_template")
    assert definition is not None and definition.required_vars == ("bot_slug", "user_id")
    result = resolver.validate(key=definition.key, body=definition.default_body())
    assert result["valid"] is True and result["unknown_placeholders"] == []
