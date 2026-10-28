"""Skill catalog reads, effective views and checked assignment writes (TASK-1309).

The API routes and the prompt section both go through :class:`SkillService`,
so bot facts, the tag snapshot and the cap check are assembled one way.

Writes are refused when they would:
- touch a private bot (403) or run with the tag snapshot unreadable (503);
- name an unknown skill or tag, or use a bad mode or position (422);
- add a bot row to a bundle bot (409);
- push any affected bot's injected skills over a cap that were not over it
  before (409, listing each bot, its refs, total and cap).
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from agent_bridge.skill_sources import AmbientSource, SkillSource, SkillSourceError

from ..bot_tags.facade import BotTags, TagSnapshot
from ..bot_tags.vocabulary import BehaviorTag
from .harness import capabilities_for
from .resolver import (
    MODES,
    Assignment,
    BotFacts,
    Caps,
    SkillResolution,
    canonical_mode,
    newly_over_cap,
    resolve,
)
from .store import SkillAssignmentStore

PER_SKILL_CAP_KEY = "skills_skill_token_cap"
TOTAL_CAP_KEY = "skills_total_token_cap"
MAX_POSITION = 10_000


class SkillServiceError(Exception):
    """A request that cannot be served; ``detail`` is the HTTP body detail."""

    def __init__(self, status: int, detail: Any):
        super().__init__(str(detail))
        self.status = status
        self.detail = detail


@dataclass(frozen=True)
class _World:
    """Everything one resolution or write decision reads, from one connection."""

    rows: list[Assignment]
    snapshot: TagSnapshot
    bots: dict[str, Any]

    def facts(self, slug: str) -> BotFacts:
        bot = self.bots[slug]
        return BotFacts(
            slug=slug,
            bot_type=bot.bot_type,
            harness=bot.agent_backend,
            bundle=(bot.agent_backend_config or {}).get("skill_bundle"),
            tags=self.snapshot.tags.get(slug, frozenset()) if self.snapshot.available else frozenset(),
            private=self.snapshot.has(slug, BehaviorTag.PRIVATE),
        )


def caps_from_settings(config) -> Caps:
    from ..runtime_setting_resolution import resolve_global_runtime_setting

    return Caps(per_skill=int(resolve_global_runtime_setting(config, PER_SKILL_CAP_KEY)),
                total=int(resolve_global_runtime_setting(config, TOTAL_CAP_KEY)))


class SkillService:
    def __init__(self, engine, *, list_bots: Callable[[], Iterable[Any]],
                 caps: Callable[[], Caps], source: SkillSource | None = None,
                 tags: BotTags | None = None):
        self.store = SkillAssignmentStore(engine)
        self.source = source or AmbientSource()
        self._list_bots = list_bots
        self._caps = caps
        self._tags = tags or BotTags(engine)

    @classmethod
    def from_config(cls, engine, config) -> SkillService:
        from ..bots import BotManager

        return cls(engine, list_bots=lambda: BotManager(config).list_bots(),
                   caps=lambda: caps_from_settings(config))

    # -- shared reads ------------------------------------------------------

    def _world(self, conn) -> _World:
        return _World(rows=self.store.rows(conn), snapshot=self._tags.snapshot(conn),
                      bots={b.slug: b for b in self._list_bots()})

    def _resolve(self, world: _World, slug: str, caps: Caps) -> SkillResolution:
        return resolve(world.facts(slug), world.rows, self.source, caps)

    def resolve_bot(self, slug: str) -> SkillResolution | None:
        """This bot's resolution, or None when it is unknown."""
        with self.store.read() as conn:
            world = self._world(conn)
        if slug not in world.bots:
            return None
        return self._resolve(world, slug, self._caps())

    # -- catalog -----------------------------------------------------------

    def catalog(self) -> list[dict]:
        with self.store.read() as conn:
            rows = self.store.rows(conn)
        counts: dict[str, dict[str, int]] = {}
        for row in rows:
            bucket = counts.setdefault(row.skill_ref, {"bot": 0, "tag": 0})
            bucket[row.scope_type] += 1
        out = []
        for entry in self.source.list():
            body = self.source.read_body(entry.ref)
            out.append({
                "ref": entry.ref, "name": entry.name, "description": entry.description,
                "source": entry.source, "token_estimate": body.tokens if body else None,
                "content_sha": body.content_sha if body else None,
                "file_count": len(self.source.tree(entry.ref)),
                "assignments": counts.get(entry.ref, {"bot": 0, "tag": 0}),
            })
        return out

    def tree(self, ref: str) -> dict:
        with _source_errors():
            return {"ref": ref, "provenance": self.source.provenance(ref),
                    "files": [{"path": n.path, "size": n.size} for n in self.source.tree(ref)]}

    def file(self, ref: str, path: str) -> dict:
        with _source_errors():
            found = self.source.read_file(ref, path)
        return {"ref": ref, "path": found.path, "size": found.size, "text": found.text}

    def assignments(self, scope_type: str | None = None, scope_id: str | None = None) -> list[dict]:
        with self.store.read() as conn:
            rows = self.store.rows(conn)
        return [vars(r) for r in rows
                if (scope_type is None or r.scope_type == scope_type)
                and (scope_id is None or r.scope_id == scope_id)]

    def harnesses(self) -> list[dict]:
        from .harness import HARNESS_CAPABILITIES

        return [c.to_dict() for c in HARNESS_CAPABILITIES.values()]

    # -- per-bot views -----------------------------------------------------

    def bots_summary(self) -> dict:
        caps = self._caps()
        with self.store.read() as conn:
            world = self._world(conn)
        bots = []
        for slug in sorted(world.bots):
            facts = world.facts(slug)
            if facts.private:
                bots.append({"slug": slug, "locked": True})
                continue
            resolution = self._resolve(world, slug, caps)
            bots.append({
                "slug": slug, "locked": False,
                "bot_type": facts.bot_type, "harness": facts.harness,
                "supported": facts.bot_type == "agent" and capabilities_for(facts.harness).supported,
                "bundle": facts.bundle,
                "row_count": len(resolution.entries),
                "disabled_count": len(resolution.disabled),
                "catalog_unreadable": resolution.catalog_unreadable,
                "injected_count": len(resolution.injected),
                "injected_tokens": resolution.injected_tokens,
                "over_cap": sorted(resolution.over_cap),
            })
        return {"caps": vars(caps), "tags_available": world.snapshot.available, "bots": bots}

    def effective(self, slug: str) -> dict:
        caps = self._caps()
        with self.store.read() as conn:
            world = self._world(conn)
        if slug not in world.bots:
            raise SkillServiceError(404, f"unknown bot {slug!r}")
        facts = world.facts(slug)
        view = {"slug": slug, "locked": facts.private,
                "harness": facts.harness, "bundle": facts.bundle,
                "capabilities": capabilities_for(facts.harness).to_dict(), "caps": vars(caps)}
        if facts.private:
            return {**view, "tags": [], "entries": [], "manifest": None,
                    "catalog_unreadable": False}
        resolution = self._resolve(world, slug, caps)
        return {**view, "tags": sorted(facts.tags),
                "entries": [e.to_dict() for e in resolution.entries],
                "injected_tokens": resolution.injected_tokens,
                # Disables a missing catalog left unapplied; the card says so.
                "catalog_unreadable": resolution.catalog_unreadable,
                "manifest": resolution.manifest()}

    # -- writes ------------------------------------------------------------

    def put(self, scope_type: str, scope_id: str, ref: str, mode: str, position: int,
            actor: str | None) -> dict:
        # Legacy names are accepted for one release (an older UI may still send them).
        mode = canonical_mode(mode) or mode
        row = Assignment(scope_type, scope_id, ref, mode, position, actor)

        def validate(world: _World) -> list[str]:
            if mode not in MODES:
                raise SkillServiceError(422, f"mode must be one of {', '.join(MODES)}")
            if type(position) is not int or not 0 <= position <= MAX_POSITION:
                raise SkillServiceError(422, f"position must be an integer 0..{MAX_POSITION}")
            if self.source.get(ref) is None:
                raise SkillServiceError(422, f"unknown skill {ref!r}")
            return self._affected(world, scope_type, scope_id)

        self._write(validate, lambda conn: self.store.upsert(conn, row))
        return vars(row)

    def remove(self, scope_type: str, scope_id: str, ref: str) -> None:
        removed = []

        def apply(conn) -> None:
            removed.append(self.store.delete(conn, scope_type, scope_id, ref))

        self._write(lambda world: self._affected(world, scope_type, scope_id), apply)
        if not removed[0]:
            raise SkillServiceError(404, "no such assignment")

    def _unprotected_bot(self, world: _World, slug: str) -> BotFacts:
        if slug not in world.bots:
            raise SkillServiceError(404, f"unknown bot {slug!r}")
        facts = world.facts(slug)
        if facts.private:
            raise SkillServiceError(403, f"{slug} is a protected bot")
        return facts

    def _writable_bot(self, world: _World, slug: str) -> BotFacts:
        facts = self._unprotected_bot(world, slug)
        if facts.bundle:
            raise SkillServiceError(409, f"{slug} takes its skills from bundle {facts.bundle!r}")
        return facts

    def _affected(self, world: _World, scope_type: str, scope_id: str) -> list[str]:
        """Bots a write at this scope can change; validates the scope itself."""
        if scope_type == "bot":
            self._writable_bot(world, scope_id)
            return [scope_id]
        if scope_type != "tag":
            raise SkillServiceError(422, "scope_type must be 'bot' or 'tag'")
        if scope_id not in world.snapshot.catalog:
            raise SkillServiceError(422, f"unknown tag {scope_id!r}")
        return sorted(s for s in world.bots if scope_id in world.snapshot.tags.get(s, ()))

    def _write(self, validate: Callable[[_World], list[str]], apply: Callable[[Any], None]) -> None:
        caps = self._caps()
        with self.store.write() as conn:
            before_world = self._world(conn)
            if not before_world.snapshot.available:
                raise SkillServiceError(503, "bot tags are unavailable; skills can't be changed")
            affected = validate(before_world)
            before = {s: self._resolve(before_world, s, caps) for s in affected}
            apply(conn)
            after_world = _World(self.store.rows(conn), before_world.snapshot, before_world.bots)
            offenders = []
            for slug in affected:
                after = self._resolve(after_world, slug, caps)
                refs = newly_over_cap(before[slug], after)
                if refs:
                    offenders.append({"bot": slug, "refs": sorted(refs),
                                      "injected_tokens": after.injected_tokens,
                                      "per_skill_cap": caps.per_skill, "total_cap": caps.total})
            if offenders:
                # Raising inside the transaction rolls the write back.
                raise SkillServiceError(409, {"code": "over_cap", "bots": offenders,
                                              "message": "injected skills would exceed the cap"})


class _source_errors:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if isinstance(exc, SkillSourceError):
            raise SkillServiceError(exc.status, str(exc)) from exc
        return False
