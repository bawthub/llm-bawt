"""Opt-in, schema-backed scheduler parameters; legacy jobs retain their resolvers."""
from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator

class JobParameters(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


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
        description="Additional bot IDs to exclude. Built-in protected bots are always excluded.",
    )
    @field_validator("protected_bot_ids", mode="before")
    @classmethod
    def validate_protected(cls, value):
        if not isinstance(value, (list, tuple)) or any(
            not isinstance(item, str) or not item.strip() for item in value
        ):
            raise ValueError("protected_bot_ids must be a list of nonempty bot IDs")
        return tuple(dict.fromkeys(item.strip().lower() for item in value))


class JobParameterRegistry:
    """Only registered jobs opt into new resolution; no legacy config is rewritten."""

    models: dict[str, type[JobParameters]] = {
        "idle_context_reset": IdleContextResetParameters,
    }

    @classmethod
    def resolve(cls, job_type: str, config_json: str | None) -> JobParameters:
        model = cls.models[job_type]
        values = json.loads(config_json) if config_json else {}
        if not isinstance(values, dict):
            raise ValueError("Job parameters must be a JSON object")
        return model.model_validate(values)

    @classmethod
    def describe(cls, job_type: str, config_json: str | None) -> dict:
        model = cls.models.get(job_type)
        if model is None:
            return {"parameter_schema": None, "parameters": None, "configuration_error": None}
        result = {"parameter_schema": model.model_json_schema(), "parameters": None,
                  "configuration_error": None, "stored_parameters": None}
        try:
            result["parameters"] = cls.resolve(job_type, config_json).model_dump(mode="json")
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
    def run_snapshot(cls, job_type: str, config_json: str | None, interval_minutes: int) -> str | None:
        """Persist policy before execution; invalid configuration is evidence too."""
        if job_type not in cls.models:
            return None
        snapshot = {"interval_minutes": interval_minutes, "parameters": None}
        try:
            snapshot["parameters"] = cls.resolve(job_type, config_json).model_dump(mode="json")
        except ValueError as exc:
            snapshot["configuration_error"] = str(exc)
        return json.dumps(snapshot)

    @classmethod
    def merge(cls, job_type: str, config_json: str | None, patch: dict) -> str:
        if job_type not in cls.models:
            raise ValueError("This job still uses legacy settings; per-job parameters are not registered")
        values = json.loads(config_json) if config_json else {}
        if not isinstance(values, dict):
            raise ValueError("Stored job parameters must be a JSON object")
        return cls.models[job_type].model_validate({**values, **patch}).model_dump_json()


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
