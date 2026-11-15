"""TASK-1198: maintenance controls use one effective policy without rewriting rows."""
import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from llm_bawt.service.job_parameters import JobParameterRegistry as Registry
from llm_bawt.service.scheduler import JobScheduler, JobType, ScheduledJob


def config(**overrides):
    return SimpleNamespace(
        MEMORY_EXTRACTION_MIN_IMPORTANCE=.8,
        MEMORY_PROFILE_ATTRIBUTE_ENABLED=False,
        MEMORY_EXTRACTION_ENABLED=False,
        resolve_summarization_job=lambda: {
            "model": "inherited-summary-model", "session_gap_seconds": 7200,
            "min_messages_per_session": 4, "protected_recent_turns": 6,
            "trigger_tokens": 18000,
        },
        **overrides,
    )


def test_all_maintenance_types_have_editable_flat_schemas():
    assert set(Registry.models) == {job.value for job in JobType if job != JobType.SEND_PROMPT}
    for job in Registry.models:
        described = Registry.describe(job, None, config=config())
        assert described["parameters"] is not None
        assert described["configuration_error"] is None
        for field in described["parameter_schema"]["properties"].values():
            assert field["type"] in ("string", "integer", "number", "boolean", "array")
            assert "anyOf" not in field  # Existing editor needs no new UI.
            if field["type"] == "array":
                assert field["items"]["type"] == "string"


@pytest.mark.parametrize("job", ["history_summarization", "memory_extraction"])
def test_shared_defaults_and_saved_overrides_keep_their_precedence(job):
    inherited = Registry.resolve(job, None, config=config())
    assert inherited.min_importance == .8 and inherited.profile_enabled is False
    explicit = Registry.resolve(job, '{"min_importance":0,"profile_enabled":true}', config=config())
    assert explicit.min_importance == 0 and explicit.profile_enabled is True
    assert explicit.model == ""  # Inherited model keeps its legacy fallback policy.
    saved = Registry.merge(job, '{"min_importance":0.6}', {"user_id": "owner"}, config=config())
    assert json.loads(saved) == {"min_importance": .6, "user_id": "owner"}
    changed = config()
    changed.MEMORY_PROFILE_ATTRIBUTE_ENABLED = True
    assert Registry.resolve(job, saved, config=changed).profile_enabled is True


@pytest.mark.parametrize(("job", "values"), [
    ("memory_extraction", {"batch_size": 0}),
    ("memory_extraction", {"min_importance": 1.1}),
    ("memory_extraction", {"profile_enabled": "true"}),
    ("memory_extraction", {"batch_size": True}),
    ("history_summarization", {"trigger_tokens": -1}),
    ("media_gc", {"grace_days": -1}),
    ("tool_result_gc", {"dry_run": "false"}),
    ("profile_maintenance", {"entity_type": "unknown"}),
    ("harness_version_check", {"packages": [""]}),
])
def test_invalid_parameters_are_rejected(job, values):
    with pytest.raises(ValidationError):
        Registry.merge(job, None, values, config=config())


@pytest.mark.parametrize(("job_type", "stored", "expected"), [
    (JobType.PROFILE_MAINTENANCE, {"entity_id": "owner", "model": "chosen", "dry_run": True},
     {"entity_id": "owner", "model": "chosen", "dry_run": True}),
    (JobType.MEMORY_CONSOLIDATION, {"entity_id": "owner", "dry_run": True},
     {"run_consolidation": True, "run_decay_pruning": False, "dry_run": True}),
    (JobType.MEMORY_DECAY, {}, {"run_consolidation": False, "run_decay_pruning": True}),
    (JobType.MEDIA_GC, {"grace_days": 0}, {"grace_days": 0, "dry_run": False}),
    (JobType.TOOL_RESULT_GC, {"retention_days": 21}, {"retention_days": 21}),
    (JobType.MEMORY_EXTRACTION, {"batch_size": 15, "profile_enabled": True},
     {"batch_size": 15, "min_importance": .8, "profile_enabled": True, "extraction_enabled": False}),
    (JobType.HISTORY_SUMMARIZATION, {"trigger_tokens": 0, "max_tokens_per_chunk": 0},
     {"trigger_tokens": 0, "max_tokens_per_chunk": 0, "session_gap_seconds": 7200}),
    (JobType.HARNESS_VERSION_CHECK, {"packages": ["example"], "agent_bot_id": None},
     {"packages": ["example"], "agent_bot_id": ""}),
    (JobType.PLAYGROUND_EXPIRY, {}, {"user_id": "system"}),
])
def test_scheduled_task_consumes_the_effective_policy(job_type, stored, expected):
    scheduler = JobScheduler(None, SimpleNamespace(config=config()))
    job = ScheduledJob(job_type=job_type, bot_id="fixture", config_json=json.dumps(stored))
    before = job.config_json
    task = scheduler._create_task_for_job(job)
    assert {key: task.payload[key] for key in expected} == expected
    assert task.bot_id == "fixture"
    assert job.config_json == before
    snapshot = json.loads(Registry.run_snapshot(job_type, before, 37, config=config()))
    assert snapshot["interval_minutes"] == 37
    assert snapshot["parameters"] == Registry.describe(job_type, before, config=config())["parameters"]


def test_snapshot_policy_is_used_even_after_shared_configuration_changes():
    shared = config()
    job = ScheduledJob(job_type=JobType.MEMORY_EXTRACTION, bot_id="fixture")
    snapshot = Registry.resolve(job.job_type, None, config=shared).model_dump(mode="json")
    shared.MEMORY_EXTRACTION_MIN_IMPORTANCE = .1
    policy_job = job.model_copy(update={"config_json": json.dumps(snapshot)})
    task = JobScheduler(None, SimpleNamespace(config=shared))._create_task_for_job(policy_job)
    assert task.payload["min_importance"] == .8


def test_legacy_null_model_and_harness_optional_values_keep_inheritance():
    assert Registry.resolve("memory_extraction", '{"model":null}').model == ""
    harness = Registry.resolve("harness_version_check", '{"packages":null,"constraints_path":null}')
    assert harness.packages == () and harness.constraints_path == ""
