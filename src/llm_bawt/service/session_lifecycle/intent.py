"""One application-owned new-session intent, independent of caller transport.

Adapters may parse a chat command or public reset policy, but only the lifecycle
service may execute an accepted intent. The owner of a conversation is not its
message author: a scheduled command is authored by BOT:system while targeting
an existing bot/user session.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from ...message_authorship import AuthorReference
from ..job_parameters import IdleContextResetParameters


class Continuity(StrEnum):
    CONFIGURED = "configured"
    RETAIN = "retain"
    CLEAN = "clean"


class Admission(StrEnum):
    WAIT = "wait"  # user or bot delivery waits for the target
    SKIP = "skip"  # unattended maintenance must not queue stale work


class Origin(StrEnum):
    CHAT_COMMAND = "chat_command"
    BOT_DELIVERY = "bot_delivery"
    SCHEDULER = "scheduler"
    CONTEXT_API = "context_api"
    SESSION_API = "session_api"
    OVERFLOW_RECOVERY = "overflow_recovery"


class Stage(StrEnum):
    ADMISSION = "admission"
    ROTATION = "rotation"
    SUMMARY = "summary"
    SEED = "seed"
    PROVIDER_BIND = "provider_bind"
    RECEIPT = "receipt"
    FOLLOW_UP = "follow_up"


class Outcome(StrEnum):
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True, kw_only=True)
class NewSessionIntent:
    """A durable instruction; ``follow_up=None`` means reset only.

    ``expected_session_id`` fences an unattended reset against a changed target;
    ``expected_usage_turn_id`` also fences the observed resident-context gauge.
    These are not inferred from the sender or from the visible command text.
    """

    bot_id: str
    user_id: str
    author: AuthorReference
    continuity: Continuity
    admission: Admission
    origin: Origin
    idempotency_key: str
    follow_up: str | None = None
    expected_session_id: str | None = None
    expected_usage_turn_id: str | None = None
    job_id: str | None = None
    run_id: str | None = None
    reason: str | None = None
    scheduler_parameters: IdleContextResetParameters | None = None

    def __post_init__(self) -> None:
        # Durable delivery JSON turns enum values back into plain strings.
        # Coercion here keeps fences effective for both in-process and replayed
        # requests; identity comparisons below must never see raw strings.
        for name, enum in (("continuity", Continuity), ("admission", Admission),
                           ("origin", Origin)):
            object.__setattr__(self, name, enum(getattr(self, name)))
        if self.scheduler_parameters is not None:
            if self.origin is not Origin.SCHEDULER:
                raise ValueError("scheduler parameters require scheduler origin")
            object.__setattr__(self, "scheduler_parameters", IdleContextResetParameters.model_validate(
                self.scheduler_parameters,
            ))
        for name in ("bot_id", "user_id", "idempotency_key"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"new-session {name} must be nonempty")
        if self.author.entity_type not in {"user", "bot"} or not self.author.entity_id:
            raise ValueError("new-session author must be a named user or bot")
        if (self.admission is Admission.SKIP and not self.expected_session_id
                and self.origin not in {Origin.CONTEXT_API, Origin.SESSION_API}):
            raise ValueError("skip-if-stale requires expected_session_id")
        if self.expected_usage_turn_id and self.admission is not Admission.SKIP:
            raise ValueError("usage fence requires skip-if-stale admission")
        if self.origin is Origin.SCHEDULER and (
            self.admission is not Admission.SKIP
            or not self.expected_usage_turn_id or not self.job_id or not self.run_id
            or self.author != AuthorReference.bot("system")
            or self.follow_up is not None or self.continuity is not Continuity.CONFIGURED
        ):
            raise ValueError("scheduled /new requires System, configured continuity, and job/run/session/usage fences")
        if self.follow_up is not None and not self.follow_up.strip():
            raise ValueError("follow-up message cannot be blank")


@dataclass(frozen=True, kw_only=True)
class NewSessionResult:
    """The operation's observed phase, never a prediction from enqueue/rotation."""

    outcome: Outcome
    stage: Stage
    old_session_id: str | None = None
    new_session_id: str | None = None
    delivery_id: str | None = None
    turn_id: str | None = None
    receipt_message_id: str | None = None
    detail: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "outcome", Outcome(self.outcome))
        object.__setattr__(self, "stage", Stage(self.stage))
        if self.outcome is Outcome.COMPLETED and not self.new_session_id:
            raise ValueError("completed reset requires a new session")
        if self.outcome is Outcome.COMPLETED and (
            self.stage is not Stage.RECEIPT or not self.receipt_message_id
        ):
            raise ValueError("completed reset requires a persisted receipt")
