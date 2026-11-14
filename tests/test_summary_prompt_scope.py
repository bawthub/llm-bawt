"""TASK-1622: scope prompts to the summarized bot, not the global job client.

All rows and calls here are disposable, in-memory fixtures, never live bots.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from sqlmodel import SQLModel, create_engine

from llm_bawt.memory.summarization import (
    Session,
    compress_structured_summary_text,
    is_summary_low_quality,
    summarize_session_with_client,
    summarize_session_with_llm,
)
from llm_bawt.prompt_registry import (
    PromptResolver, PromptTemplate, PromptTemplateStore, PromptTemplateVersion,
)
from llm_bawt.service.routes import history_summaries, prompts
from llm_bawt.service.schemas import PromptTemplateUpsertRequest
from llm_bawt.utils.history import HistoryManager


@pytest.fixture
def resolver(monkeypatch):
    store = PromptTemplateStore.__new__(PromptTemplateStore)
    store.engine = create_engine("sqlite://")
    store.config = None
    SQLModel.metadata.create_all(
        store.engine, tables=[PromptTemplate.__table__, PromptTemplateVersion.__table__],
    )
    result = PromptResolver.__new__(PromptResolver)
    result.store = store
    result.config = None
    monkeypatch.setattr("llm_bawt.prompt_registry.PromptResolver", lambda cfg: result)
    monkeypatch.setattr(prompts, "PromptResolver", lambda cfg: result)
    monkeypatch.setattr(prompts, "get_service", lambda: SimpleNamespace(config=None))
    yield result
    store.engine.dispose()


def session():
    return Session(
        session_id="fixture-thread", start_timestamp=1, end_timestamp=2,
        messages=[{"role": "user", "content": "Keep API config and TASK-1234.", "timestamp": 1}],
        message_ids=["fixture-message"],
    )


def seed(resolver):
    for key, suffix in (
        ("history.summarization.single", "{messages}"),
        ("history.summarization.batch", "{sessions_blob}"),
        ("history.summarization.system", ""),
        ("history.summarization.batch.system", ""),
        ("history.summarization.revision.system", ""),
    ):
        for scope_type, scope_id, label in (("global", "*", "GLOBAL"), ("bot", "scoped-bot", "BOT")):
            resolver.store.upsert(
                key=key, body=f"{label}:{key} {suffix}", scope_type=scope_type,
                scope_id=scope_id, updated_by="fixture-author",
            )


@pytest.mark.parametrize("bot_id,label", [("scoped-bot", "BOT"), ("other-bot", "GLOBAL"), (None, "GLOBAL")])
@pytest.mark.parametrize("path", ["client", "http"])
def test_single_summary_and_system_use_bot_scope_with_global_fallback(resolver, monkeypatch, bot_id, label, path):
    seed(resolver)
    if path == "client":
        client = MagicMock()
        client.query.return_value = "Summary: retained API config."
        assert summarize_session_with_client(session(), client, bot_id=bot_id)
        sent = client.query.call_args.kwargs["messages"]
        system, user = sent[0].content, sent[1].content
    else:
        post = MagicMock()
        post.return_value.json.return_value = {"content": "Summary: retained API config."}
        monkeypatch.setattr("llm_bawt.memory.summarization.httpx.post", post)
        assert summarize_session_with_llm(session(), bot_id=bot_id)
        sent = post.call_args.kwargs["json"]
        system, user = sent["system"], sent["prompt"]
    assert system.startswith(f"{label}:history.summarization.system")
    assert user.startswith(f"{label}:history.summarization.single")
    assert "Keep API config and TASK-1234." in user


@pytest.mark.parametrize("path", ["client", "http", "route"])
@pytest.mark.parametrize("content,accepted", [
    ("Summary: Fixed KeyError: missing session after a configuration change.", True),
    ("Summary: Diagnosed CUDA error: out of memory and reduced the batch size.", True),
    ("Summary: The exception occurred during startup; the patch fixed it.", True),
    ("Summary: Checked whether tokens exceed the configured budget.", True),
    ("Summary: Requests can exceed context window limits; split the input.", True),
    ("Error: model unavailable", False),
    ("  CUDA error: out of memory  ", False),
    ("Exception occurred while loading the model", False),
    ("Tokens exceed the configured context", False),
    ("Exceed context window", False),
    ("Out of memory", False),
])
def test_summary_error_mentions_are_not_provider_failures(resolver, monkeypatch, path, content, accepted):
    seed(resolver)
    if path == "client":
        client = MagicMock()
        client.query.return_value = content
        result = summarize_session_with_client(session(), client)
    elif path == "http":
        post = MagicMock()
        post.return_value.json.return_value = {"content": content}
        monkeypatch.setattr("llm_bawt.memory.summarization.httpx.post", post)
        result = summarize_session_with_llm(session())
    else:
        client = MagicMock()
        client.query.return_value = content
        config = SimpleNamespace(
            SUMMARIZATION_QUALITY_RETRY=False,
            resolve_summarization_job=lambda: {"model": "fixture-model"},
            get_model_context_window=lambda alias: 32000,
            get_model_max_tokens=lambda alias: 4096,
        )
        service = SimpleNamespace(config=config, _get_background_client=lambda **kw: (client, "fixture-model"))
        summarize, _batch, _chunk = history_summaries._build_summary_callable(service, "scoped-bot")
        result = summarize(session())
    expected = content.strip() if accepted else None
    if path == "route" and accepted:
        expected = compress_structured_summary_text(expected, source_session=session())
    assert result == expected


@pytest.mark.parametrize("bot_id,label", [("scoped-bot", "BOT"), ("other-bot", "GLOBAL")])
def test_route_single_batch_and_revision_share_scope(resolver, monkeypatch, bot_id, label):
    seed(resolver)
    client = MagicMock()
    config = SimpleNamespace(
        SUMMARIZATION_QUALITY_RETRY=True,
        resolve_summarization_job=lambda: {"model": "fixture-model"},
        get_model_context_window=lambda alias: 32000,
        get_model_max_tokens=lambda alias: 4096,
    )
    service = SimpleNamespace(config=config, _get_background_client=lambda **kw: (client, "fixture-model"))
    client.query.side_effect = ["Summary: first.", "Summary: revised."]
    # The callable imports quality checking locally; drive its retry deterministically.
    quality = MagicMock(side_effect=[True, False])
    monkeypatch.setattr("llm_bawt.memory.summarization.is_summary_low_quality", quality)
    # Rebuild to pick up the quality checker at its import boundary.
    single, batch, _ = history_summaries._build_summary_callable(service, bot_id)
    assert single(session()) == "Summary: revised."
    first, revision = client.query.call_args_list
    assert first.kwargs["messages"][0].content.startswith(f"{label}:history.summarization.system")
    assert first.kwargs["messages"][1].content.startswith(f"{label}:history.summarization.single")
    assert revision.kwargs["messages"][0].content.startswith(f"{label}:history.summarization.revision.system")
    client.reset_mock(side_effect=True)
    client.query.return_value = '{"summaries":[{"session_index":1,"summary":"Retain API config for TASK-1234."}]}'
    quality.side_effect = None
    quality.return_value = False
    assert batch([session()]) == {1: "Summary: Retain API config for TASK-1234."}
    sent = client.query.call_args.kwargs["messages"]
    assert sent[0].content.startswith(f"{label}:history.summarization.batch.system")
    assert sent[1].content.startswith(f"{label}:history.summarization.batch")
    assert "Keep API config and TASK-1234." in sent[1].content


def test_scoped_listing_author_versions_and_reset_leave_global_untouched(resolver):
    key = "history.summarization.single"
    seed(resolver)
    global_before = resolver.store.get_exact(key).body
    assert prompts.list_prompt_overrides("no-such-key").prompts == []
    overrides = prompts.list_prompt_overrides(key)
    assert [(p.key, p.scope_id, p.updated_by) for p in overrides.prompts] == [
        (key, "scoped-bot", "fixture-author"),
    ]
    assert "body" not in overrides.prompts[0].model_dump()
    index = prompts.list_prompts_index(
        category=None, scope_type="bot", scope_id="scoped-bot", include_defaults=False,
    )
    assert len(index.prompts) == 5
    assert all(p.scope_type == "bot" and p.scope_id == "scoped-bot" for p in index.prompts)
    saved = prompts.upsert_prompt(key, PromptTemplateUpsertRequest(
        body="Edited {messages}", scope_type="bot", scope_id="scoped-bot", updated_by="second-author",
    ))
    assert saved.updated_by == "second-author"
    assert len(resolver.store.list_versions(key, "bot", "scoped-bot")) == 2
    reset = prompts.reset_prompt(key, scope_type="bot", scope_id="scoped-bot")
    assert reset.scope_type == "global" and reset.body == global_before
    assert prompts.list_prompt_overrides(key).prompts == []
    assert resolver.store.get_exact(key).body == global_before
    assert len(resolver.store.list_versions(key)) == 1
    assert len(resolver.store.list_versions(key, "bot", "scoped-bot")) == 2


@pytest.mark.parametrize("key,expected", [
    ("history.summarization.system", "You are a helpful assistant that summarizes conversations concisely."),
    ("history.summarization.batch.system", "You are a helpful assistant that summarizes conversations with strict JSON output."),
    ("history.summarization.revision.system", "You are a precise summarizer. Avoid generic phrasing and preserve concrete facts."),
])
def test_registered_system_defaults_preserve_previous_bytes(resolver, key, expected):
    assert resolver.render(key=key, variables={}, scope_type="bot", scope_id="unconfigured-bot") == expected


def compact(text, enabled=True):
    manager = HistoryManager.__new__(HistoryManager)
    manager._setting = lambda key, default: enabled
    return manager._compact_summary_content(text)


@pytest.mark.parametrize("marker", ["N/A", "None", "None stated", "not applicable", "Not stated.", " - N/A"])
@pytest.mark.parametrize("multiline", [False, True])
def test_context_compaction_drops_only_optional_placeholder_sections(marker, multiline):
    sep = "\n" if multiline else " "
    text = "Summary: Keep the API config.\n" + "\n".join(
        f"{label}:{sep}{marker}" for label in ("Key Details", "Intent", "Tone", "Open Loops")
    )
    assert compact(text) == "Keep the API config."
    assert compact(text, enabled=False) == text


def test_compaction_keeps_concrete_none_statements_and_metadata():
    text = (
        "[HISTORICAL SUMMARY]\nOn 2026-10-28\nSummary: Updated the API config.\n"
        "Key Details: None of the 12 tests failed.\nIntent: Keep the old default.\n"
        "Open Loops: None of the clients have deployed the new config."
    )
    result = compact(text)
    assert "On 2026-10-28" in result
    assert "Details: None of the 12 tests failed." in result
    assert "Intent: Keep the old default." in result
    assert "Open: None of the clients have deployed the new config." in result


def test_quality_check_does_not_require_optional_filler_sections():
    text = (
        "Summary: TASK-1234 API config was updated and reviewed; the default model remains unchanged. "
        "The next session should preserve the scoped prompt and wait for the deployment callback.\n"
        "Key Details: Owner approved commit abc1234 after 42 tests passed. The config path 'service/config.py' "
        "now accepts per-bot prompt scope, while clients without overrides retain the global default body."
    )
    assert not is_summary_low_quality(text)
