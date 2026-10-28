"""Effective skills for one bot, from exception rows and a skill source (TASK-1309).

Pure: no database, no I/O beyond the source's cached reads. The store, the
API and the prompt section all resolve through :func:`resolve`, so the rules
below exist once:

- every skill in the catalog is enabled by default; rows are exceptions;
- a bot row beats tag rows, which beat the global ("all bots") row; among
  the tag rows a bot carries, ``disable`` > ``inject`` > ``enable``;
- a private bot gets nothing, whatever its rows say;
- only agent bots on a supported harness apply skills;
- a bundle bot's skills come from its bundle, not this source;
- the harness gets an allow-list (catalog minus disabled) only when at least
  one disable applies; otherwise it keeps its own discovery, so skills added
  later reach the bot without a row;
- an unreadable (empty) catalog applies no disable: hiding every skill is the
  worse failure, and the listing filter is hygiene, not a sandbox;
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

ENABLE, DISABLE, INJECT = "enable", "disable", "inject"
MODES = (ENABLE, DISABLE, INJECT)
BOT, TAG, GLOBAL = "bot", "tag", "global"
SCOPES = (BOT, TAG, GLOBAL)
#: The one scope_id a global row carries.
GLOBAL_ID = "*"
#: Scope precedence: a lower number wins.
_SCOPE_RANK = {BOT: 0, TAG: 1, GLOBAL: 2}
#: The adopt-era vocabulary, still accepted on read for one release so rows an
#: older process writes during a rolling restart resolve the same way.
LEGACY_MODES = {"available": ENABLE, "off": DISABLE, "injected": INJECT}
#: Among tag rows the lowest rank wins: a disable sticks across tags.
_TAG_RANK = {DISABLE: 0, INJECT: 1, ENABLE: 2}

# Why a row does not apply (or applies without changing anything).
PROTECTED_BOT = "protected_bot"
UNSUPPORTED_HARNESS = "unsupported_harness"
NOT_IN_BUNDLE = "not_in_bundle"
SOURCE_MISSING = "source_missing"
OVER_CAP = "over_cap"
NO_EFFECT = "no_effect"  # an enable with no disable under it
CATALOG_UNREADABLE = "catalog_unreadable"


def canonical_mode(mode: str) -> str | None:
    """The current name for ``mode`` (legacy names map), or None if unknown."""
    if mode in MODES:
        return mode
    return LEGACY_MODES.get(mode)


@dataclass(frozen=True)
class BotFacts:
    slug: str
    bot_type: str
    harness: str | None  # Bot.agent_backend
    bundle: str | None
    tags: frozenset[str]
    private: bool


@dataclass(frozen=True)
class Assignment:
    scope_type: str  # "bot" | "tag" | "global" (scope_id "*")
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
    origin: str  # "bot", "tag:<slug>" or "global"
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
    harness: str | None
    entries: tuple[ResolvedSkill, ...]
    caps: Caps
    #: Why this bot's rows apply nothing (private, harness, bundle).
    skip_reason: str | None = None
    #: Catalog refs, read only when a disable needs them.
    catalog: tuple[str, ...] = ()
    source_kind: str = ""

    @property
    def applied(self) -> tuple[ResolvedSkill, ...]:
        """Applied enable and inject rows (disables are in :attr:`disabled`)."""
        return tuple(e for e in self.entries if e.applied and e.mode != DISABLE)

    @property
    def disabled(self) -> frozenset[str]:
        return frozenset(e.ref for e in self.entries if e.applied and e.mode == DISABLE)

    @property
    def catalog_unreadable(self) -> bool:
        return any(e.reason == CATALOG_UNREADABLE for e in self.entries)

    @property
    def injected(self) -> tuple[ResolvedSkill, ...]:
        """Applied injected skills in prompt order."""
        return tuple(e for e in self.applied if e.mode == INJECT)

    @property
    def injected_tokens(self) -> int:
        return sum(e.tokens or 0 for e in self.injected)

    @property
    def over_cap(self) -> frozenset[str]:
        return frozenset(e.ref for e in self.entries if e.reason == OVER_CAP)

    def manifest(self) -> dict | None:
        """The skill allow-list the bridge applies; None keeps harness discovery.

        Only a bot with at least one applied disable gets one: the catalog
        minus its disables. ``manifest_sha`` covers every other field, so a
        pinned session can tell whether its skills changed.
        """
        disabled = self.disabled
        if self.skip_reason is not None or not disabled:
            return None
        injected = {e.ref for e in self.injected}
        body = {
            "harness": self.harness,
            "entries": [{"ref": ref, "mode": INJECT if ref in injected else ENABLE,
                         "source": self.source_kind}
                        for ref in self.catalog if ref not in disabled],
            "injected": [{"ref": e.ref, "content_sha": e.content_sha} for e in self.injected],
        }
        canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
        return {**body, "manifest_sha": hashlib.sha256(canonical.encode()).hexdigest()}


def _effective_rows(facts: BotFacts, rows: Iterable[Assignment]
                    ) -> tuple[dict[str, Assignment], dict[str, int]]:
    """One winning row per ref for this bot (bot > tag > global), and for each
    ref a lower layer disables, the rank of the lowest such layer."""
    layers: dict[str, dict[str, Assignment]] = {BOT: {}, TAG: {}, GLOBAL: {}}
    for raw in rows:
        mode = canonical_mode(raw.mode)
        if mode is None:
            continue
        row = raw if mode == raw.mode else Assignment(
            raw.scope_type, raw.scope_id, raw.skill_ref, mode, raw.position, raw.updated_by)
        if row.scope_type == BOT and row.scope_id == facts.slug:
            layers[BOT][row.skill_ref] = row
        elif row.scope_type == GLOBAL and row.scope_id == GLOBAL_ID:
            layers[GLOBAL][row.skill_ref] = row
        elif row.scope_type == TAG and row.scope_id in facts.tags:
            held = layers[TAG].get(row.skill_ref)
            key = (_TAG_RANK[row.mode], row.position, row.scope_id)
            if held is None or key < (_TAG_RANK[held.mode], held.position, held.scope_id):
                layers[TAG][row.skill_ref] = row
    disabled_rank: dict[str, int] = {}
    for scope in (BOT, TAG, GLOBAL):  # ascending rank, so the last write is the lowest layer
        for ref, row in layers[scope].items():
            if row.mode == DISABLE:
                disabled_rank[ref] = _SCOPE_RANK[scope]
    return {**layers[GLOBAL], **layers[TAG], **layers[BOT]}, disabled_rank


def origin_of(row: Assignment) -> str:
    if row.scope_type == TAG:
        return f"tag:{row.scope_id}"
    return row.scope_type


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
    winners, disabled_rank = _effective_rows(facts, rows)
    blanket = _skip_reason(facts)
    order = sorted(winners.values(), key=lambda r: (r.position, r.skill_ref))
    needs_catalog = blanket is None and any(r.mode == DISABLE for r in order)
    # A partial listing (one root unreadable) would hide that root's skills,
    # so it counts as unreadable too: no disables apply.
    catalog = (tuple(e.ref for e in source.list())
               if needs_catalog and source.complete() else ())
    entries: list[ResolvedSkill] = []
    total = 0
    for row in order:
        base = dict(ref=row.skill_ref, mode=row.mode, origin=origin_of(row),
                    position=row.position, source=source.kind)
        if blanket is not None:
            entries.append(ResolvedSkill(**base, applied=False, reason=blanket))
            continue
        if row.mode == DISABLE:
            if not catalog:
                entries.append(ResolvedSkill(**base, applied=False, reason=CATALOG_UNREADABLE))
            else:
                present = row.skill_ref in catalog
                entries.append(ResolvedSkill(**base, applied=present,
                                             reason=None if present else SOURCE_MISSING))
            continue
        if row.mode == ENABLE:
            present = source.get(row.skill_ref) is not None
            # Applies only over a disable from a lower-precedence layer.
            overrides = disabled_rank.get(row.skill_ref, -1) > _SCOPE_RANK[row.scope_type]
            reason = SOURCE_MISSING if not present else (None if overrides else NO_EFFECT)
            entries.append(ResolvedSkill(**base, applied=present, reason=reason))
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
    return SkillResolution(facts.slug, facts.harness, tuple(entries), caps, blanket,
                           catalog, source.kind)


def legacy_rows(rows: Iterable[Assignment], managed: frozenset[str],
                bots: Iterable[BotFacts], catalog: Iterable[str] | None) -> list[Assignment]:
    """Exception rows that resolve, for every bot, as the adopt-era rows did.

    The single conversion rule: the store's one-time migration persists it and,
    until that migration has run, the service resolves through it, so a
    deferred migration cannot change any bot's skills. Adopt-era semantics:
    an unmanaged bot ignored every row (ambient discovery); a managed bot got
    exactly the refs with a winning ``available``/``injected`` row (bot row
    over tag rows; ``off`` was bot-only).

    - Modes are renamed (``off``/``available``/``injected`` to
      ``disable``/``enable``/``inject``).
    - A bot ``available`` row is kept as ``enable`` only where it overrides a
      tag ``injected`` row the bot carries; elsewhere it changes nothing and
      is dropped.
    - An unmanaged bot's own rows are dropped (they never applied), and each
      tag ``injected`` row it carries gets a bot ``enable`` so it stays ambient.
    - A managed bot gets a bot ``disable`` for every catalog ref no row listed.
      Without a ``catalog`` (unreadable) that step is skipped, which fails open
      like any other disable on an unreadable catalog.
    - Private bots and rows for unknown bots are only renamed: neither model
      applies them.
    """
    renamed = [Assignment(r.scope_type, r.scope_id, r.skill_ref,
                          canonical_mode(r.mode) or r.mode, r.position, r.updated_by)
               for r in rows]
    tag_rows = [r for r in renamed if r.scope_type == "tag"]
    facts = {f.slug: f for f in bots}
    # Global rows postdate the adopt era (its CHECK refused them); kept as-is.
    out = list(tag_rows) + [r for r in renamed if r.scope_type == GLOBAL]
    for r in renamed:
        bot = facts.get(r.scope_id) if r.scope_type == "bot" else None
        if r.scope_type != "bot" or bot is None or bot.private:
            if r.scope_type == "bot":
                out.append(r)
            continue
        if r.scope_id not in managed:
            continue
        if r.mode == ENABLE and not any(t.mode == INJECT and t.skill_ref == r.skill_ref
                                        and t.scope_id in bot.tags for t in tag_rows):
            continue
        out.append(r)
    refs = tuple(catalog) if catalog is not None else ()
    for bot in sorted(facts.values(), key=lambda f: f.slug):
        if bot.private:
            continue
        carried = [t for t in tag_rows if t.scope_id in bot.tags]
        if bot.slug not in managed:
            for ref in sorted({t.skill_ref for t in carried if t.mode == INJECT}):
                out.append(Assignment("bot", bot.slug, ref, ENABLE))
            continue
        listed = {r.skill_ref for r in renamed
                  if r.scope_type == "bot" and r.scope_id == bot.slug}
        listed |= {t.skill_ref for t in carried}
        out.extend(Assignment("bot", bot.slug, ref, DISABLE)
                   for ref in refs if ref not in listed)
    return out


def newly_over_cap(before: SkillResolution, after: SkillResolution) -> frozenset[str]:
    """Refs a write would push over a cap that were not over it already."""
    return after.over_cap - before.over_cap
