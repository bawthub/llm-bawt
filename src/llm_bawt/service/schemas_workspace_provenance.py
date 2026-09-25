"""Pydantic contracts for the bounded workspace provenance API (TASK-729)."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class WorkspaceLineChangeInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["+", "-"]
    text: str = Field(max_length=32_768)
    hunk: int = Field(ge=1, le=10_000)
    line: int = Field(ge=1, le=10_000_000)


class WorkspaceFileInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repo_id: str = Field(min_length=1, max_length=128)
    repo_key: str = Field(min_length=1, max_length=255)
    repo_aliases: list[str] = Field(default_factory=list, max_length=12)
    path: str = Field(min_length=1, max_length=4096)
    staged: bool = False
    status: str = Field(min_length=1, max_length=8)
    old_text: str = Field(default="", max_length=1_048_576)
    new_text: str = Field(default="", max_length=1_048_576)
    binary: bool = False
    truncated: bool = False
    changes: list[WorkspaceLineChangeInput] = Field(default_factory=list, max_length=20_000)

    @field_validator("repo_aliases")
    @classmethod
    def validate_aliases(cls, aliases: list[str]) -> list[str]:
        return [alias.strip() for alias in aliases if alias.strip()]


class WorkspaceProvenanceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(min_length=1, max_length=128)
    status_fingerprint: str = Field(min_length=1, max_length=128)
    lookback_hours: int = Field(default=48, ge=1, le=168)
    files: list[WorkspaceFileInput] = Field(min_length=1, max_length=200)


class WorkspaceCandidateResponse(BaseModel):
    turn_id: str
    trigger_message_id: str | None
    bot_id: str | None
    session_id: str | None = None
    created_at: datetime
    confidence: Literal["exact", "strong", "partial", "probable"]
    matched_changes: int
    dirty_changes: int
    candidate_changes: int
    dirty_coverage: float
    candidate_coverage: float
    dirty_hunks: list[int]
    exact_transition: bool
    final_snapshot: bool
    source_tool_call_ids: list[str]
    prompt: str | None
    truncated: bool


class WorkspaceFileAttributionResponse(BaseModel):
    repo_id: str
    repo_key: str
    path: str
    staged: bool
    ownership: Literal["owned", "shared", "unattributed"]
    candidates: list[WorkspaceCandidateResponse]
    unmatched_changes: int


class WorkspaceTurnResponse(BaseModel):
    turn_id: str
    trigger_message_id: str | None
    bot_id: str | None
    session_id: str | None = None
    created_at: datetime
    prompt: str | None


class WorkspaceProvenanceResponse(BaseModel):
    generated_at: datetime
    status_fingerprint: str
    lookback_hours: int
    files: list[WorkspaceFileAttributionResponse]
    turns: list[WorkspaceTurnResponse]
    warnings: list[str] = Field(default_factory=list)
    truncated: bool = False
