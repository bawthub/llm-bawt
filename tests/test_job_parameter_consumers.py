"""TASK-1198: local controls reach real worker decisions, not just stored JSON."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from llm_bawt.memory.extraction.service import ExtractedFact, MemoryExtractionService
from llm_bawt.service.background_tasks import BackgroundTasksMixin


@pytest.mark.parametrize("job_type", ["memory_consolidation", "memory_decay"])
def test_maintenance_identity_is_documented_routing_not_a_memory_owner_filter(job_type):
    import json

    from llm_bawt.service.job_parameters import JobParameterRegistry
    from llm_bawt.service.scheduler import JobScheduler, JobType, ScheduledJob

    schema = JobParameterRegistry.describe(job_type, None)["parameter_schema"]["properties"]["entity_id"]
    assert schema["title"] == "Task routing identity"
    assert "not a memory-owner filter" in schema["description"]
    assert "all memories for the target bot" in schema["description"]
    service = BackgroundTasksMixin()
    client = SimpleNamespace(run_maintenance=Mock(return_value={"dry_run": True}))
    service.get_memory_client = Mock(return_value=client)
    scheduler = JobScheduler(None, None)
    for owner in ("owner-a", "owner-b"):
        job = ScheduledJob(
            job_type=JobType(job_type), bot_id="fixture",
            config_json=json.dumps({"entity_id": owner, "dry_run": True}),
        )
        task = scheduler._create_task_for_job(job)
        assert task.user_id == owner  # Historical routing is preserved.
        assert service._process_maintenance(task) == {"dry_run": True}
    assert [call.args for call in service.get_memory_client.call_args_list] == [("fixture",), ("fixture",)]
    calls = client.run_maintenance.call_args_list
    assert calls[0] == calls[1]  # Both local and remote clients receive bot-wide operations.
    assert calls[0].kwargs["dry_run"] is True
    assert calls[0].kwargs["run_consolidation"] is (job_type == "memory_consolidation")
    assert calls[0].kwargs["run_decay_pruning"] is (job_type == "memory_decay")
    assert "entity_id" not in calls[0].kwargs and "user_id" not in calls[0].kwargs


@pytest.mark.parametrize("job_type", ["media_gc", "tool_result_gc", "harness_version_check", "playground_expiry"])
def test_system_jobs_do_not_advertise_routing_identity_as_a_scope_filter(job_type):
    from llm_bawt.service.job_parameters import JobParameterRegistry

    schema = JobParameterRegistry.describe(job_type, None)["parameter_schema"]["properties"]["user_id"]
    assert schema["title"] == "Task routing identity"
    assert "does not restrict" in schema["description"]


@pytest.mark.parametrize("source", ["messages", "summaries"])
@pytest.mark.parametrize("overrides, expected", [
    ({}, 0),
    ({"min_importance": .4, "profile_enabled": True}, 1),
    ({"min_importance": .4, "profile_enabled": False}, 1),
])
def test_extraction_consumes_job_threshold_and_profile_control(monkeypatch, source, overrides, expected):
    service = BackgroundTasksMixin()
    service.config = SimpleNamespace(
        MEMORY_EXTRACTION_MIN_IMPORTANCE=.9, MEMORY_PROFILE_ATTRIBUTE_ENABLED=False,
        MEMORY_EXTRACTION_ENABLED=False,
    )
    service._get_background_client = lambda **kw: (object(), "inherited-model")
    service._mark_summary_extracted = Mock()
    backend = SimpleNamespace(
        get_unprocessed_messages=lambda **kw: [{
            "id": "message", "role": "user", "content": "I own a dog.",
            "author_entity_type": "user", "author_entity_id": "owner",
        }],
        mark_messages_processed=Mock(),
    )
    store = SimpleNamespace(list_memories=lambda **kw: [], add_memory=Mock(return_value="stored"))
    service.get_memory_client = lambda *args: store
    monkeypatch.setattr("llm_bawt.memory.postgresql.PostgreSQLMemoryBackend", lambda *a, **kw: backend)
    monkeypatch.setattr("llm_bawt.memory.extraction.storage.ExtractionMemoryStore", lambda *a: store)
    fact = ExtractedFact(content="I own a dog.", importance=.6, tags=["pets"],
                         source_message_ids=["message"])
    monkeypatch.setattr(MemoryExtractionService, "extract_from_conversation", lambda *a, **kw: [fact])
    monkeypatch.setattr(MemoryExtractionService, "extract_from_summary", lambda *a, **kw: [fact])
    monkeypatch.setattr("llm_bawt.memory.summary_extraction_policy.summary_allows_extraction", lambda *a: True)
    monkeypatch.setattr("llm_bawt.memory.summary_extraction_policy.summary_has_human_source", lambda *a: True)
    profile = Mock()
    monkeypatch.setattr("llm_bawt.mcp_server.extraction.extract_profile_attributes_from_fact", profile)
    if source == "messages":
        task = SimpleNamespace(bot_id="fixture", user_id="owner", payload=overrides)
        result = service._extract_owned_memory_batch(task, backend, "fixture", "owner", 50)
    else:
        result = service._extract_from_summaries(
            [{"created": True, "summary_id": "summary", "summary_text": "User owns a dog."}],
            "fixture", "owner", parameters=overrides,
        )
    assert result["facts_stored"] == expected
    assert store.add_memory.call_count == expected
    assert profile.call_count == (1 if overrides.get("profile_enabled") else 0)
    if profile.called:
        assert profile.call_args.kwargs["enabled"] is True
    assert service.config.MEMORY_EXTRACTION_MIN_IMPORTANCE == .9
    assert service.config.MEMORY_PROFILE_ATTRIBUTE_ENABLED is False


def test_raw_extraction_can_be_disabled_locally_without_building_a_backend():
    service = BackgroundTasksMixin()
    service.config = SimpleNamespace(MEMORY_EXTRACTION_ENABLED=True)
    task = SimpleNamespace(bot_id="fixture", user_id="owner", payload={"extraction_enabled": False})
    assert service._process_memory_extraction(task)["skipped"] is True
    assert service.config.MEMORY_EXTRACTION_ENABLED is True


def test_summarizer_local_controls_leave_manual_and_reset_defaults_unchanged(monkeypatch):
    from llm_bawt.memory.summarization import HistorySummarizer

    shared = {"session_gap_seconds": 3600, "min_messages_per_session": 2, "protected_recent_turns": 3}
    config = SimpleNamespace(resolve_summarization_job=lambda: shared)
    monkeypatch.setattr("llm_bawt.memory.postgresql.PostgreSQLMemoryBackend", lambda *a, **kw: object())
    local = HistorySummarizer(config, bot_id="fixture", max_context_tokens=100,
                              job_parameters={"session_gap_seconds": 20, "protected_recent_turns": 0})
    manual = HistorySummarizer(config, bot_id="fixture", max_context_tokens=100)
    assert local.session_gap_seconds == 20 and local.protected_recent_turns == 0
    assert local.min_messages_per_session == 2
    assert manual.session_gap_seconds == 3600 and manual.protected_recent_turns == 3
    assert shared == {"session_gap_seconds": 3600, "min_messages_per_session": 2, "protected_recent_turns": 3}


@pytest.mark.parametrize("model, allow_fallback", [("", True), ("explicit-model", False)])
def test_history_worker_forwards_controls_and_keeps_model_provenance(monkeypatch, model, allow_fallback):
    service = BackgroundTasksMixin()
    service.config = SimpleNamespace(
        resolve_summarization_job=lambda: {
            "model": "shared-model", "trigger_tokens": 12000, "session_gap_seconds": 3600,
            "min_messages_per_session": 2, "protected_recent_turns": 3,
        },
        resolve_context_budget=lambda alias: (0, 0, 7000),
    )
    service._get_background_client = Mock(return_value=(object(), "served-model"))
    service._extract_from_summaries = Mock(return_value={})
    service._llm_bawt_cache = {}
    monkeypatch.setattr("llm_bawt.memory.summarization_limits.resolve_summarization_limits",
                        lambda *a: (8000, 4000))
    monkeypatch.setattr("llm_bawt.bots.BotManager", lambda *a: SimpleNamespace(
        get_bot=lambda bot: None, get_default_bot=lambda: None,
    ))
    monkeypatch.setattr("llm_bawt.runtime_settings.RuntimeSettingsResolver",
                        lambda **kw: SimpleNamespace(resolve=lambda key, fallback: fallback))
    captured = {}

    class Summarizer:
        def __init__(self, config, **kwargs):
            captured.update(kwargs)

        def summarize_eligible_sessions(self, **kwargs):
            captured["run"] = kwargs
            return {"results": []}

    monkeypatch.setattr("llm_bawt.memory.summarization.HistorySummarizer", Summarizer)
    payload = {"model": model, "trigger_tokens": 0, "protected_recent_turns": 0,
               "session_gap_seconds": 20, "max_tokens_per_chunk": 0, "profile_enabled": False}
    service._process_history_summarization(SimpleNamespace(bot_id="fixture", user_id="owner", payload=payload))
    assert service._get_background_client.call_args.kwargs["allow_fallback"] is allow_fallback
    assert captured["job_parameters"]["protected_recent_turns"] == 0
    assert captured["job_parameters"]["session_gap_seconds"] == 20
    assert captured["max_context_tokens"] == 7000
    assert captured["run"]["max_tokens_per_chunk"] == 0
    assert service._extract_from_summaries.call_args.kwargs["parameters"] == payload


def test_inner_profile_gate_uses_override_without_changing_manual_defaults(monkeypatch):
    from llm_bawt.mcp_server.extraction import extract_profile_attributes_from_fact

    config = SimpleNamespace(MEMORY_PROFILE_ATTRIBUTE_ENABLED=False)
    candidate = SimpleNamespace(created_at=1, updated_at=1)
    manager = SimpleNamespace(add_candidate=Mock(return_value=candidate))
    monkeypatch.setattr("llm_bawt.mcp_server.server._get_profile_manager", lambda: manager)
    fact = {"content": "User's name is Example.", "importance": .9,
            "profile_attribute": {"category": "identity", "key": "name"}}
    assert extract_profile_attributes_from_fact(fact, "fixture-owner", config) is False
    assert extract_profile_attributes_from_fact(fact, "fixture-owner", config, enabled=True) is True
    assert manager.add_candidate.call_count == 1
    assert extract_profile_attributes_from_fact(fact, "fixture-owner", config, enabled=False) is False
    assert config.MEMORY_PROFILE_ATTRIBUTE_ENABLED is False


def test_history_activity_probe_uses_the_same_local_controls_as_worker(monkeypatch):
    from llm_bawt.service.scheduler import JobScheduler

    config = SimpleNamespace(resolve_summarization_job=lambda: {
        "model": "shared-model", "session_gap_seconds": 3600, "min_messages_per_session": 2,
        "protected_recent_turns": 3, "trigger_tokens": 12000,
    })
    processor = SimpleNamespace(config=config, _resolve_request_model=lambda *a, **kw: ("resolved-model", None))
    monkeypatch.setattr("llm_bawt.bots.BotManager", lambda *a: SimpleNamespace(
        get_bot=lambda bot: None, get_default_bot=lambda: None,
    ))
    monkeypatch.setattr("llm_bawt.service.scheduler.RuntimeSettingsResolver",
                        lambda **kw: SimpleNamespace(resolve=lambda key, fallback: fallback))
    captured = {}

    class Summarizer:
        def __init__(self, config, **kwargs):
            captured.update(kwargs)

        def preview_summarizable_sessions(self):
            return ["eligible"]

    monkeypatch.setattr("llm_bawt.memory.summarization.HistorySummarizer", Summarizer)
    controls = {"session_gap_seconds": 40, "min_messages_per_session": 6,
                "protected_recent_turns": 0, "trigger_tokens": 400}
    assert JobScheduler(None, processor)._has_pending_history_summaries("fixture", controls) == (True, 1)
    assert captured["max_context_tokens"] == 400
    assert {key: captured["job_parameters"][key] for key in controls} == controls
