"""Fan-out target selectors and their expansion (TASK-1431 design r3 §1.1).

A selector names exactly one of an explicit bot list, every agent bot, agent
bots on named backends, or agent bots carrying a role tag. Expansion reads the
bot catalog and one tag snapshot once, at submit:

* ``private`` and ``inactive`` bots never come out of a selector;
* an explicit list naming one gets a per-target refusal instead. For a bot
  caller a private target reads exactly like an unknown one, so its existence
  is not revealed;
* the sender's own bot is left out unless ``include_self``;
* only bots that can take a durable delivery are sent to.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from .bot_types import durable_agent_backend

MAX_FANOUT_TARGETS = 64

UNKNOWN_TARGET = "unknown_target"
TARGET_PRIVATE = "target_private"
TARGET_INACTIVE = "target_inactive"
TARGET_NOT_DURABLE = "target_not_durable"


class SelectorError(ValueError):
    """The selector is malformed or names an unknown tag (HTTP 422)."""


@dataclass(frozen=True)
class TargetRefusal:
    """One explicit target that gets no row, with the HTTP shape it maps to."""

    target: str
    status_code: int
    error: str
    detail: str

    def to_api(self) -> dict[str, Any]:
        return {"target": self.target, "error": self.error, "detail": self.detail}


@dataclass(frozen=True)
class Expansion:
    targets: tuple[str, ...]
    refused: tuple[TargetRefusal, ...]


def _slugs(values: Any, field: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise SelectorError(f"{field} must be a list of strings")
    out: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise SelectorError(f"{field} must be a list of non-empty strings")
        slug = value.strip().lower()
        if slug not in out:
            out.append(slug)
    return tuple(out)


@dataclass(frozen=True)
class TargetSelector:
    """Exactly one of ``bots``, ``all_agents``, ``backends`` or ``tags``."""

    bots: tuple[str, ...] = ()
    all_agents: bool = False
    backends: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        named = sum((bool(self.bots), bool(self.all_agents), bool(self.backends), bool(self.tags)))
        if named != 1:
            raise SelectorError("targets names exactly one of bots, all_agents, backends or tags")

    @classmethod
    def parse(cls, raw: Any) -> TargetSelector:
        """A selector object, or a bare list as shorthand for ``{"bots": [...]}``."""
        if isinstance(raw, list):
            return cls(bots=_slugs(raw, "targets"))
        if not isinstance(raw, Mapping):
            raise SelectorError("targets must be a selector object or a list of bot ids")
        unknown = set(raw) - {"bots", "all_agents", "backends", "tags"}
        if unknown:
            raise SelectorError(f"unknown targets selector field(s): {', '.join(sorted(unknown))}")
        all_agents = raw.get("all_agents", False)
        if type(all_agents) is not bool:
            raise SelectorError("targets.all_agents must be true or false")
        return cls(
            bots=_slugs(raw.get("bots") or (), "targets.bots"),
            all_agents=all_agents,
            backends=_slugs(raw.get("backends") or (), "targets.backends"),
            tags=_slugs(raw.get("tags") or (), "targets.tags"),
        )

    @property
    def explicit(self) -> bool:
        return bool(self.bots)

    def to_json(self) -> dict[str, Any]:
        if self.bots:
            return {"bots": list(self.bots)}
        if self.all_agents:
            return {"all_agents": True}
        if self.backends:
            return {"backends": list(self.backends)}
        return {"tags": list(self.tags)}


class TargetExpander:
    """Resolve a selector against one bot catalog read and one tag snapshot."""

    def __init__(self, bots: Iterable[Any], snapshot):
        self._bots = {str(bot.slug).strip().lower(): bot for bot in bots}
        self._snapshot = snapshot

    def expand(
        self,
        selector: TargetSelector,
        *,
        sender: str,
        exclude: Iterable[str] = (),
        include_self: bool = False,
        bot_caller: bool,
    ) -> Expansion:
        skipped = set(_slugs(exclude, "exclude"))
        if not include_self:
            skipped.add(sender)
        if selector.explicit:
            return self._explicit(selector.bots, skipped, bot_caller=bot_caller)
        return Expansion(tuple(sorted(
            slug for slug, bot in self._bots.items()
            if slug not in skipped and self._selected(selector, slug, bot)
            and self._snapshot.excluded_reason(slug) is None
            and durable_agent_backend(bot)
        )), ())

    def _selected(self, selector: TargetSelector, slug: str, bot: Any) -> bool:
        from .service.turn_gates import GateScope

        kind = getattr(bot, "bot_type", None)
        backend = getattr(bot, "agent_backend", None)
        if selector.backends:
            return GateScope.on_backends(*selector.backends).covers(
                slug, bot_type=kind, agent_backend=backend)
        if not GateScope.every_agent().covers(slug, bot_type=kind, agent_backend=backend):
            return False
        if selector.all_agents:
            return True
        return any(self._snapshot.has(slug, tag) for tag in selector.tags)

    def validate_tags(self, selector: TargetSelector) -> None:
        """An unknown tag is a caller error, never an empty fan-out."""
        for tag in selector.tags:
            try:
                self._snapshot.has("", tag)
            except ValueError as exc:
                raise SelectorError(str(exc)) from exc

    def _explicit(self, names: tuple[str, ...], skipped: set[str], *, bot_caller: bool) -> Expansion:
        targets: list[str] = []
        refused: list[TargetRefusal] = []
        for name in names:
            if name in skipped:
                continue
            refusal = self._refusal(name, bot_caller=bot_caller)
            if refusal is None:
                targets.append(name)
            else:
                refused.append(refusal)
        return Expansion(tuple(sorted(targets)), tuple(refused))

    def _refusal(self, name: str, *, bot_caller: bool) -> TargetRefusal | None:
        from .bot_tags import BehaviorTag

        unknown = TargetRefusal(name, 404, UNKNOWN_TARGET, f"Unknown target bot {name!r}")
        bot = self._bots.get(name)
        if bot is None:
            return unknown
        if self._snapshot.has(name, BehaviorTag.PRIVATE):
            # Same answer as an unknown bot for a bot caller (r3 §1.1).
            return unknown if bot_caller else TargetRefusal(
                name, 403, TARGET_PRIVATE, f"Target bot {name!r} is private")
        if self._snapshot.has(name, BehaviorTag.INACTIVE):
            return TargetRefusal(
                name, 422, TARGET_INACTIVE,
                f"'{name}' is inactive; it must be reactivated before it can receive messages.")
        if not durable_agent_backend(bot):
            return TargetRefusal(
                name, 422, TARGET_NOT_DURABLE,
                f"Target bot {name!r} does not support durable agent delivery")
        return None
