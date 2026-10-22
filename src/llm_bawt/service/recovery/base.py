"""The recovery step contract (TASK-1232 Phase 2).

A step finds candidates (``detect``), decides what it would do to each and why
(``plan``), and only outside a dry run does it (``apply``). ``apply`` re-checks
its preconditions inside its own fenced transaction, so a candidate that
changed since ``detect`` is a reported no-op, never a blind write.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, ClassVar

from .audit import RunStamp


@dataclass(frozen=True)
class SweepScope:
    """One run. ``stamp`` goes on every row the run writes (``audit``)."""

    engine: Any
    now: datetime
    dry_run: bool
    bots: frozenset[str] | None = None
    stamp: RunStamp = field(default_factory=lambda: RunStamp.new("recovery_sweep"))

    def includes(self, bot_id: str | None) -> bool:
        return self.bots is None or bot_id in self.bots


@dataclass(frozen=True)
class Plan:
    """What ``apply`` would do (``action``), or why the candidate is left (``why``).

    ``detail`` goes into the report; ``carry`` is handed to ``apply`` only.
    """

    action: str | None
    why: str = ""
    detail: dict[str, Any] = field(default_factory=dict)
    carry: Any = field(default=None, compare=False)

    @classmethod
    def leave(cls, why: str, **detail: Any) -> "Plan":
        return cls(None, why, detail)


@dataclass
class StepReport:
    step: str
    found: list[dict[str, Any]] = field(default_factory=list)
    planned: list[dict[str, Any]] = field(default_factory=list)
    done: list[dict[str, Any]] = field(default_factory=list)
    ambiguous: list[dict[str, Any]] = field(default_factory=list)
    raced: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"step": self.step, "found": self.found, "planned": self.planned,
                "done": self.done, "ambiguous": self.ambiguous, "raced": self.raced}


class RecoveryStep(ABC):
    name: ClassVar[str]

    @abstractmethod
    async def detect(self, scope: SweepScope) -> list[dict[str, Any]]:
        """Candidates, as plain dicts (they go into the report as-is)."""

    @abstractmethod
    async def plan(self, scope: SweepScope, candidate: dict[str, Any]) -> Plan: ...

    async def apply(self, scope: SweepScope, candidate: dict[str, Any], plan: Plan) -> dict[str, Any] | None:
        """Do ``plan``; return what was done, or None when the candidate raced away."""
        raise NotImplementedError(f"{self.name} plans no actions")

    async def run(self, scope: SweepScope) -> StepReport:
        report = StepReport(self.name)
        for candidate in await self.detect(scope):
            report.found.append(candidate)
            plan = await self.plan(scope, candidate)
            entry = {**candidate, **({"action": plan.action} if plan.action else {}), **plan.detail}
            if plan.action is None:
                if plan.why:
                    report.ambiguous.append({**entry, "why": plan.why})
                continue
            if scope.dry_run:
                report.planned.append(entry)
                continue
            outcome = await self.apply(scope, candidate, plan)
            (report.done if outcome is not None else report.raced).append(
                {**entry, **(outcome or {})})
        return report
