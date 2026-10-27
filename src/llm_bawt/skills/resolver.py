"""Effective skills for one bot, from assignment rows and a skill source (TASK-1309).

Pure: no database, no I/O beyond the source's cached reads. The store, the
API and the prompt section all resolve through :func:`resolve`, so the rules
below exist once:

- an unmanaged bot resolves to nothing (it keeps the harness's ambient skills);
- a private bot gets nothing, whatever its rows say;
- only agent bots on a supported harness apply skills;
- a bot row beats tag rows; among tag rows the strongest mode wins
  (injected > available); ``off`` exists only at bot scope and excludes;
- a bundle bot's skills come from its bundle, not this source;
- injected bodies are admitted in (position, ref) order under a per-skill and
  a total token cap; the rest are skipped ``over_cap``.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass, field

from agent_bridge.skill_sources import SkillSource

from .harness import capabilities_for

OFF, AVAILABLE, INJECTED = "off", "available", "injected"
MODES = (OFF, AVAILABLE, INJECTED)
_STRENGTH = {AVAILABLE: 1, INJECTED: 2}

# Why an assigned skill does not apply.
PROTECTED_BOT = "protected_bot"
UNSUPPORTED_HARNESS = "unsupported_harness"
NOT_IN_BUNDLE = "not_in_bundle"
SOURCE_MISSING = "source_missing"
OVER_CAP = "over_cap"


@dataclass(frozen=True)
class BotFacts:
    slug: str
    bot_type: str
    harness: str | None  # Bot.agent_backend
    bundle: str | None
    tags: frozenset[str]
    private: bool
    managed: bool


@dataclass(frozen=True)
class Assignment:
    scope_type: str  # "bot" | "tag"
    scope_id: str
    skill_ref: str
    mode: str
    position: int = 0
    updated_by: str | None = None


@dataclass(frozen=True)
class Caps:
    per_skill: int
    total: int


@dataclass(frozen=True)
class ResolvedSkill:
    ref: str
    mode: str
    origin: str  # "bot" or "tag:<slug>"
    position: int
    source: str
    applied: bool
    reason: str | None = None
    tokens: int | None = None
    content_sha: str | None = None
    # Injected skills only: the body admitted under the cap and the directory
    # its relative paths resolve against. Not part of the public view.
    body_text: str | None = field(default=None, repr=False, compare=False)
    base_dir: str | None = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict:
        return {"ref": self.ref, "mode": self.mode, "origin": self.origin,
                "position": self.position, "source": self.source, "applied": self.applied,
                "reason": self.reason, "tokens": self.tokens, "content_sha": self.content_sha}


@dataclass(frozen=True)
class SkillResolution:
    bot: str
    managed: bool
    harness: str | None
    entries: tuple[ResolvedSkill, ...]
    caps: Caps
    #: Why a managed bot's rows apply nothing (private, harness, bundle).
    skip_reason: str | None = None

    @property
    def applied(self) -> tuple[ResolvedSkill, ...]:
        return tuple(e for e in self.entries if e.applied and e.mode != OFF)

    @property
    def injected(self) -> tuple[ResolvedSkill, ...]:
        """Applied injected skills in prompt order."""
        return tuple(e for e in self.applied if e.mode == INJECTED)

    @property
    def injected_tokens(self) -> int:
        return sum(e.tokens or 0 for e in self.injected)

    @property
    def over_cap(self) -> frozenset[str]:
        return frozenset(e.ref for e in self.entries if e.reason == OVER_CAP)

    def manifest(self) -> dict | None:
        """What the bridge applies for this bot; None when nothing is managed.

        ``manifest_sha`` covers every other field, so a pinned session can
        tell whether its skills changed. A bot whose rows a blanket reason
        skips also gets None: it keeps its own (ambient or bundle) skills.
        """
        if not self.managed or self.skip_reason is not None:
            return None
        body = {
            "harness": self.harness,
            "entries": [{"ref": e.ref, "mode": e.mode, "source": e.source} for e in self.applied],
            "injected": [{"ref": e.ref, "content_sha": e.content_sha} for e in self.injected],
        }
        canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
        return {**body, "manifest_sha": hashlib.sha256(canonical.encode()).hexdigest()}


def _effective_rows(facts: BotFacts, rows: Iterable[Assignment]) -> dict[str, Assignment]:
    """One winning row per ref for this bot."""
    bot_rows: dict[str, Assignment] = {}
    tag_rows: dict[str, Assignment] = {}
    for row in rows:
        if row.scope_type == "bot" and row.scope_id == facts.slug:
            bot_rows[row.skill_ref] = row
        elif row.scope_type == "tag" and row.scope_id in facts.tags and row.mode in _STRENGTH:
            held = tag_rows.get(row.skill_ref)
            key = (-_STRENGTH[row.mode], row.position, row.scope_id)
            if held is None or key < (-_STRENGTH[held.mode], held.position, held.scope_id):
                tag_rows[row.skill_ref] = row
    return {**tag_rows, **bot_rows}


def _skip_reason(facts: BotFacts) -> str | None:
    if facts.private:
        return PROTECTED_BOT
    if facts.bot_type != "agent" or not capabilities_for(facts.harness).supported:
        return UNSUPPORTED_HARNESS
    if facts.bundle:
        return NOT_IN_BUNDLE
    return None


def resolve(facts: BotFacts, rows: Iterable[Assignment], source: SkillSource,
            caps: Caps) -> SkillResolution:
    if not facts.managed:
        return SkillResolution(facts.slug, False, facts.harness, (), caps)
    winners = _effective_rows(facts, rows)
    blanket = _skip_reason(facts)
    order = sorted(winners.values(), key=lambda r: (r.position, r.skill_ref))
    entries: list[ResolvedSkill] = []
    total = 0
    for row in order:
        origin = "bot" if row.scope_type == "bot" else f"tag:{row.scope_id}"
        base = dict(ref=row.skill_ref, mode=row.mode, origin=origin,
                    position=row.position, source=source.kind)
        if row.mode == OFF:
            entries.append(ResolvedSkill(**base, applied=False))
            continue
        if blanket is not None:
            entries.append(ResolvedSkill(**base, applied=False, reason=blanket))
            continue
        if row.mode == AVAILABLE:
            present = source.get(row.skill_ref) is not None
            entries.append(ResolvedSkill(**base, applied=present,
                                         reason=None if present else SOURCE_MISSING))
            continue
        skill = source.get(row.skill_ref)
        body = source.read_body(row.skill_ref) if skill is not None else None
        if body is None:
            entries.append(ResolvedSkill(**base, applied=False, reason=SOURCE_MISSING))
            continue
        fits = body.tokens <= caps.per_skill and total + body.tokens <= caps.total
        if fits:
            total += body.tokens
        entries.append(ResolvedSkill(
            **base, applied=fits, reason=None if fits else OVER_CAP,
            tokens=body.tokens, content_sha=body.content_sha,
            body_text=body.text if fits else None, base_dir=str(skill.root) if fits else None))
    return SkillResolution(facts.slug, True, facts.harness, tuple(entries), caps, blanket)


def newly_over_cap(before: SkillResolution, after: SkillResolution) -> frozenset[str]:
    """Refs a write would push over a cap that were not over it already."""
    return after.over_cap - before.over_cap
