"""Recall's typed public contracts and non-negotiable resource ceilings."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Source = Literal["messages", "memories", "tasks", "tool_evidence"]
MAX_RECORDS = 24
MAX_SOURCE_RECORDS = 6
# Message ranking/expansion policy (TASK-1210), kept together on purpose.
# R1: hits order by query coverage first: the share of the query's distinct
# lexemes a message contains, rounded UP into this many buckets so near-equal
# coverage ties (7/10 and 6/10 share the top-but-one bucket). A message that
# matches one or two generic words never outranks one that matches most of the
# question, however old. Within a bucket, one recency term divides the
# full-text rank (rank halves at this age), so the newer of two equally
# relevant statements wins.
COVERAGE_BUCKETS = 4
RECENCY_HALF_WEIGHT_DAYS = 14.0
# R2: a hit may carry its next turns in the same caller session (a correction
# can share no query terms). Follow-ons are admitted only after every source's
# direct hits and still count against MAX_RECORDS / MAX_INPUT_TOKENS.
MAX_FOLLOW_ONS_PER_HIT = 2
FOLLOW_ON_WINDOW_SECONDS = 1800
MAX_RECORD_CHARS = 2400
# Returned quote caps; an over-long grounded pick is shortened to a prefix, not dropped.
MAX_QUOTE_CHARS = 400
MAX_QUOTE_TOKENS = 100
MAX_INPUT_TOKENS = 6000
MAX_QUERY_TOKENS = 256
MAX_MODEL_BYTES = 16_384
MAX_SECONDS = 30.0
TASK_REF = re.compile(r"(?:TASK-[1-9][0-9]{0,9}|c[a-z0-9]{20,31})\Z")
CUID = re.compile(r"c[a-z0-9]{20,31}\Z")


class RecallError(Exception):
    """Only a fixed code, never exception/provider/source text, reaches callers."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


class RecallRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    query: str = Field(min_length=1, max_length=1000)
    sources: tuple[Source, ...] = ("messages", "memories")
    task_ids: tuple[str, ...] = Field(default=(), max_length=3)
    handle: str | None = Field(default=None, max_length=64)
    output_tokens: int = Field(default=1000, ge=400, le=1600, strict=True)

    @field_validator("query")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be blank")
        return value.strip()

    @field_validator("sources")
    @classmethod
    def valid_sources(cls, value: tuple[Source, ...]) -> tuple[Source, ...]:
        if not value or len(value) > 4 or len(set(value)) != len(value):
            raise ValueError("select one to four distinct sources")
        return value

    @field_validator("task_ids")
    @classmethod
    def valid_tasks(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not TASK_REF.fullmatch(ref) for ref in value):
            raise ValueError("invalid task reference")
        return tuple(dict.fromkeys(value))

    @model_validator(mode="after")
    def task_scope(self):
        if self.task_ids and "tasks" not in self.sources:
            raise ValueError("task_ids require tasks source")
        return self


@dataclass(frozen=True)
class RecallScope:
    bot_id: str
    user_id: str
    session_id: str
    excluded_bot_ids: frozenset[str] = frozenset()
    # Read from the current DB policy, not caller/model supplied.
    allowed_bot_ids: frozenset[str] = frozenset()
    capability_expires_at: float | None = None

    @property
    def owner(self) -> tuple[str, str, str]:
        return self.bot_id, self.user_id, self.session_id


@dataclass(frozen=True)
class Evidence:
    source: Source
    id: str
    bot_id: str
    user_id: str
    text: str
    start_char: int = 0
    truncated: bool = False
    window_kind: str = "matched"
    # Server-owned provenance so the model can order and supersede statements.
    # `at` is UTC ISO-8601 (`...Z`); `speaker` is the stored role, never model text.
    at: str | None = None
    speaker: str | None = None
    # Id of the direct hit this record continues; None for a direct hit.
    follows: str | None = None
    # These identify the authorized source row, not the recipient/owner above.
    # Never derive a review reference from model output: it must occur in the
    # original, authorized message excerpt.
    source_bot: str | None = None
    session_id: str | None = None
    review_ref: str | None = None


class EvidenceBatch(list[Evidence]):
    """List-compatible source result with bounded, non-identifying coverage."""

    def __init__(self, values=(), *, withheld: int = 0, search_mode: str = "lexical"):
        super().__init__(values)
        self.withheld = withheld
        self.search_mode = search_mode


class SourceProvider(ABC):
    source: Source
    # Rows a source may return: direct hits are always capped at
    # MAX_SOURCE_RECORDS; only follow-ons may take a source past it.
    max_records: int = MAX_SOURCE_RECORDS

    @abstractmethod
    async def search(self, query: str, scope: RecallScope, task_ids: tuple[str, ...]) -> list[Evidence]:
        """Return at most six provenance-checked hits (plus declared follow-ons)."""


class Synthesizer(ABC):
    @abstractmethod
    def system_prompt(self, bot_id: str) -> str:
        """Resolve the live instructions once per recall (prompt registry; never hardcoded).

        The service budgets evidence against exactly this text and passes it
        back to ``synthesize``, so what is counted is what is sent.
        """

    @abstractmethod
    async def synthesize(self, system: str, query: str, evidence: list[Evidence],
                         output_tokens: int) -> str:
        """Return JSON {findings:[{source,id,quote}]}; no tools or side effects."""
