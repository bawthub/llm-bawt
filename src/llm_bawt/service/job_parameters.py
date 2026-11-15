"""Opt-in, schema-backed scheduler parameters; legacy jobs retain their resolvers."""
from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator

from ..setting_definitions import HISTORY_SCOPES

# Scheduled-reset continuity: the bot's own ``history_scope`` or one explicit
# canonical scope for this job's resets only. Titles ride the JSON schema
# (``x-enum-titles``, parallel to ``enum``) so the generic editor can label them.
_SCOPE_TITLES = {
    "inline+summaries": "Messages + summaries",
    "inline": "Messages only",
    "summaries": "Summaries only",
    "none": "Nothing (clean start)",
}
if tuple(_SCOPE_TITLES) != HISTORY_SCOPES:
    raise RuntimeError("continuity titles must cover every history scope, in order")
RESET_CONTINUITY_TITLES = {"configured": "Use bot setting", **_SCOPE_TITLES}
ResetContinuity = Literal[tuple(RESET_CONTINUITY_TITLES)]  # type: ignore[valid-type]


class JobParameters(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    @field_validator("model", "constraints_path", "agent_bot_id", mode="before", check_fields=False)
    @classmethod
    def blank_inherits(cls, value):
        return "" if value is None else value


class ProfileMaintenanceParameters(JobParameters):
    entity_id: str = Field(default="user", min_length=1, title="Profile owner")
    entity_type: Literal["user", "bot"] = Field(default="user", title="Profile type")
    dry_run: StrictBool = Field(default=False, title="Dry run")
    model: str = Field(default="", title="Model",
                       description="Blank inherits the shared profile/maintenance model.")


class MemoryMaintenanceParameters(JobParameters):
    entity_id: str = Field(
        default="system", min_length=1, title="Task routing identity",
        description=("Legacy task routing identity only, not a memory-owner filter. "
                     "Maintenance processes all memories for the target bot."),
    )
    dry_run: StrictBool = Field(default=False, title="Dry run")


class ExtractionControls(JobParameters):
    min_importance: float = Field(default=0.5, strict=True, ge=0, le=1,
                                  title="Minimum fact importance")
    profile_enabled: StrictBool = Field(default=True, title="Extract profile attributes")


class MemoryExtractionParameters(ExtractionControls):
    user_id: str = Field(default="system", min_length=1, title="User")
    batch_size: int = Field(default=50, strict=True, ge=1, le=1000, title="Batch size")
    model: str = Field(default="", title="Model",
                       description="Blank inherits the shared extraction/maintenance model.")
    extraction_enabled: StrictBool = Field(default=True, title="Extract memories")


class HistorySummarizationParameters(ExtractionControls):
    user_id: str = Field(default="system", min_length=1, title="User")
    model: str = Field(default="", title="Model",
                       description="Blank inherits the shared summarization/maintenance model.")
    use_heuristic_fallback: StrictBool = Field(default=True, title="Heuristic fallback")
    max_tokens_per_chunk: int = Field(default=4000, strict=True, title="Chunk token limit",
                                      description="Zero or negative disables splitting.")
    session_gap_seconds: int = Field(default=3600, strict=True, ge=0, title="Session gap (seconds)")
    min_messages_per_session: int = Field(default=2, strict=True, ge=1, title="Minimum messages")
    protected_recent_turns: int = Field(default=3, strict=True, ge=0, title="Protected recent turns")
    trigger_tokens: int = Field(default=12000, strict=True, ge=0, title="Summarization trigger",
                                description="Zero uses the model context budget.")


class IdleContextResetParameters(JobParameters):
    resident_tokens_gt: int = Field(
        default=100_000, strict=True, ge=1, le=2_000_000,
        title="Context threshold", description="Reset only above this resident context size (tokens).",
    )
    idle_minutes_gt: int = Field(
        default=15, strict=True, ge=1, le=10_080,
        title="Idle threshold", description="Reset only after more than this many idle minutes.",
    )
    protected_bot_ids: tuple[str, ...] = Field(
        default=(), title="Additional protected bots",
        description="Additional bot IDs to exclude. Bots tagged private or inactive are always excluded.",
    )
    continuity: ResetContinuity = Field(
        default="configured", title="Continuity",
        description=("What the new session carries over. 'Use bot setting' follows each bot's "
                     "history scope; the other choices apply to this job's resets only and never "
                     "change the bot's setting."),
        json_schema_extra={"x-enum-titles": list(RESET_CONTINUITY_TITLES.values())},
    )
    reset_when_prompt_stale: bool = Field(
        default=False, strict=True, title="Reset stale prompts",
        description=("Also reset an idle session whose pinned system prompt is older than the "
                     "current one, at any context size."),
    )
    prompt_stale_settle_minutes: int = Field(
        default=10, strict=True, ge=0, le=1_440, title="Stale settle time",
        description=("Wait this many minutes after the prompt last changed, so a burst of edits "
                     "costs one reset."),
    )

    @property
    def history_scope_override(self) -> str | None:
        """The scope a scheduled reset must use instead of the bot's, if any."""
        return None if self.continuity == "configured" else self.continuity

    @field_validator("protected_bot_ids", mode="before")
    @classmethod
    def validate_protected(cls, value):
        if not isinstance(value, (list, tuple)) or any(
            not isinstance(item, str) or not item.strip() for item in value
        ):
            raise ValueError("protected_bot_ids must be a list of nonempty bot IDs")
        return tuple(dict.fromkeys(item.strip().lower() for item in value))


class SystemJobParameters(JobParameters):
    user_id: str = Field(
        default="system", min_length=1, title="Task routing identity",
        description="Task routing identity only; it does not restrict this system-wide job's data scope.",
    )


class MediaGcParameters(SystemJobParameters):
    grace_days: int = Field(default=7, strict=True, ge=0, title="Grace period (days)")
    dry_run: StrictBool = Field(default=False, title="Dry run")


class ToolResultGcParameters(SystemJobParameters):
    retention_days: int = Field(default=14, strict=True, ge=0, title="Retention (days)")
    dry_run: StrictBool = Field(default=False, title="Dry run")


class HarnessVersionParameters(SystemJobParameters):
    constraints_path: str = Field(default="", title="Constraints file",
                                  description="Blank uses the installed harness pins.")
    packages: tuple[str, ...] = Field(default=(), title="Packages",
                                      description="An empty list uses the installed harness package list.")
    agent_bot_id: str = Field(default="", title="Task assignee")
    priority: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"] = Field(default="MEDIUM", title="Task priority")

    @field_validator("packages", mode="before")
    @classmethod
    def validate_packages(cls, value):
        if value is None:
            return ()
        if not isinstance(value, (list, tuple)) or any(
            not isinstance(item, str) or not item.strip() for item in value
        ):
            raise ValueError("packages must be a list of nonempty names")
        return value


class JobParameterRegistry:
    """Saved values override current shared defaults; reads never rewrite jobs."""

    models: dict[str, type[JobParameters]] = {
        "idle_context_reset": IdleContextResetParameters,
        "profile_maintenance": ProfileMaintenanceParameters,
        "memory_consolidation": MemoryMaintenanceParameters,
        "memory_decay": MemoryMaintenanceParameters,
        "history_summarization": HistorySummarizationParameters,
        "memory_extraction": MemoryExtractionParameters,
        "media_gc": MediaGcParameters,
        "tool_result_gc": ToolResultGcParameters,
        "harness_version_check": HarnessVersionParameters,
        "playground_expiry": SystemJobParameters,
    }

    @classmethod
    def _defaults(cls, job_type: str, config) -> dict:
        defaults = {}
        if config is not None and job_type in ("history_summarization", "memory_extraction"):
            defaults.update(
                min_importance=getattr(config, "MEMORY_EXTRACTION_MIN_IMPORTANCE", 0.5),
                profile_enabled=getattr(config, "MEMORY_PROFILE_ATTRIBUTE_ENABLED", True),
            )
            if job_type == "memory_extraction":
                defaults["extraction_enabled"] = getattr(config, "MEMORY_EXTRACTION_ENABLED", True)
            else:
                shared = config.resolve_summarization_job()
                defaults.update({key: shared[key] for key in (
                    "session_gap_seconds", "min_messages_per_session",
                    "protected_recent_turns", "trigger_tokens",
                )})
        return defaults

    @classmethod
    def resolve(cls, job_type: str, config_json: str | None, *, config=None) -> JobParameters:
        model = cls.models[job_type]
        values = json.loads(config_json) if config_json else {}
        if not isinstance(values, dict):
            raise ValueError("Job parameters must be a JSON object")
        return model.model_validate({**cls._defaults(job_type, config), **values})

    @classmethod
    def describe(cls, job_type: str, config_json: str | None, *, config=None) -> dict:
        model = cls.models.get(job_type)
        if model is None:
            return {"parameter_schema": None, "parameters": None, "configuration_error": None}
        result = {"parameter_schema": model.model_json_schema(), "parameters": None,
                  "configuration_error": None, "stored_parameters": None}
        try:
            result["parameters"] = cls.resolve(job_type, config_json, config=config).model_dump(mode="json")
        except ValueError as exc:
            result["configuration_error"] = str(exc)
            try:
                stored = json.loads(config_json) if config_json else {}
            except ValueError:
                stored = None
            if isinstance(stored, dict):
                # Unvalidated values are a separate repair baseline, never effective policy.
                result["stored_parameters"] = {
                    key: value for key, value in stored.items() if key in model.model_fields
                }
        return result

    @classmethod
    def run_snapshot(cls, job_type: str, config_json: str | None, interval_minutes: int, *,
                     config=None) -> str | None:
        """Persist policy before execution; invalid configuration is evidence too."""
        if job_type not in cls.models:
            return None
        snapshot = {"interval_minutes": interval_minutes, "parameters": None}
        try:
            snapshot["parameters"] = cls.resolve(job_type, config_json, config=config).model_dump(mode="json")
        except ValueError as exc:
            snapshot["configuration_error"] = str(exc)
        return json.dumps(snapshot)

    @classmethod
    def merge(cls, job_type: str, config_json: str | None, patch: dict, *, config=None) -> str:
        if job_type not in cls.models:
            raise ValueError("This job still uses legacy settings; per-job parameters are not registered")
        values = json.loads(config_json) if config_json else {}
        if not isinstance(values, dict):
            raise ValueError("Stored job parameters must be a JSON object")
        merged = {**values, **patch}
        resolved = cls.resolve(job_type, json.dumps(merged), config=config)
        # Do not pin inherited controls as a side effect of an unrelated edit.
        # Idle reset retains its original fully materialized repair contract.
        return resolved.model_dump_json(
            include=None if job_type == "idle_context_reset" else set(merged),
        )


class JobConfigurationPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: StrictBool | None = None
    interval_minutes: int | None = Field(default=None, strict=True, ge=1, le=525_600)
    parameters: dict[str, Any] | None = None

    @field_validator("enabled", "interval_minutes", "parameters", mode="before")
    @classmethod
    def reject_null(cls, value):
        if value is None:
            raise ValueError("Omit unchanged fields; explicit null is not supported")
        return value
