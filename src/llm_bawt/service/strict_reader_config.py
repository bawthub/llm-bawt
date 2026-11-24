"""Startup snapshot of the operator's strict reader selection (TASK-2590).

The global-only ``strict_reader`` runtime setting is read once when the
service starts and copied onto the four attributes the strict path reads
(``strict_reader_bot``, ``strict_reader_house``,
``strict_reader_skill_generation``, ``strict_reader_skill_layers``). They do
not change for the life of the process; a new selection needs a restart. The
settings API refuses this key, so only the operator's database access writes it.

Off only when proven off: an app with no database configured, an absent row,
or an object with ``enabled: false``. Otherwise nothing degrades to ordinary
shared routing:

- enabled with a trustworthy bot slug but anything else invalid: the bot stays
  fenced as the strict bot with house and generation cleared, so every strict
  ingress refuses its turns;
- unreadable, undecodable or non-object rows, or no trustworthy slug: startup
  fails with a redacted diagnostic (:class:`StrictReaderConfigError`).

Rollback: the reader inverse keeps the same bot and house with
``skill_generation: "legacy"`` and no layers (configured roots, never the
``SKILL_SOURCES_CURRENT`` link); ``enabled: false`` is only the dormant
pre-admission rollback.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Callable

from .strict_reader_execution import _GENERATION_NAME, LEGACY_GENERATION

logger = logging.getLogger(__name__)

SETTING_KEY = "strict_reader"
_FIELDS = {"enabled", "bot", "house", "skill_generation", "skill_layers"}
_SLUG = re.compile(r"^[a-z0-9][a-z0-9_-]{0,127}$")


class StrictReaderConfigError(RuntimeError):
    """The stored selection cannot be trusted; the app must not start."""


@dataclass(frozen=True)
class StrictReaderSnapshot:
    bot: str | None = None
    house: str | None = None
    skill_generation: str | None = None
    skill_layers: dict[str, str] | None = None
    error: str | None = None

    def apply(self, service: Any) -> None:
        service.strict_reader_bot = self.bot
        service.strict_reader_house = self.house
        service.strict_reader_skill_generation = self.skill_generation
        service.strict_reader_skill_layers = self.skill_layers
        service.strict_reader_config_error = self.error


OFF = StrictReaderSnapshot()


def _slug(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    slug = value.strip().lower()
    return slug if _SLUG.fullmatch(slug) else None


def parse(raw: Any) -> StrictReaderSnapshot:
    """Validate a decoded value. Raises :class:`StrictReaderConfigError` when
    no bot can be fenced; returns a fenced snapshot for other invalid values."""
    if not isinstance(raw, dict):
        raise StrictReaderConfigError("strict_reader setting is not an object")
    if raw.get("enabled") is False:
        return OFF
    bot = _slug(raw.get("bot"))
    if bot is None:
        raise StrictReaderConfigError("strict_reader.bot is missing or invalid")

    def fenced(reason: str) -> StrictReaderSnapshot:
        return StrictReaderSnapshot(bot=bot, error=reason)

    if raw.get("enabled") is not True:
        return fenced("strict_reader.enabled must be true or false")
    unknown = set(raw) - _FIELDS
    if unknown:
        return fenced(f"strict_reader has unknown fields: {sorted(unknown)}")
    from agent_bridge.houses import normalize_house

    try:
        house = normalize_house(raw.get("house")) if isinstance(raw.get("house"), str) else ""
    except ValueError:
        house = ""
    if not house:  # the mansion is shared; strict needs a named house
        return fenced("strict_reader.house is missing or invalid")
    generation = raw.get("skill_generation")
    layers = raw.get("skill_layers")
    if generation == LEGACY_GENERATION:
        if layers not in (None, {}):
            return fenced("strict_reader.skill_layers must be empty for legacy")
        return StrictReaderSnapshot(bot=bot, house=house, skill_generation=LEGACY_GENERATION)
    if not isinstance(generation, str) or not _GENERATION_NAME.fullmatch(generation):
        return fenced("strict_reader.skill_generation must be 'legacy' or gen-<name>")
    if (not isinstance(layers, dict) or not layers
            or not all(isinstance(k, str) and k and isinstance(v, str) and v
                       for k, v in layers.items())):
        return fenced("strict_reader.skill_layers must be a non-empty {source: revision} map")
    return StrictReaderSnapshot(bot=bot, house=house, skill_generation=generation,
                                skill_layers=dict(layers))


def load_snapshot(config: Any, store_factory: Callable[[], Any]) -> StrictReaderSnapshot:
    """Read the setting once at startup; raise rather than guess."""
    from ..utils.config import has_database_credentials

    if not has_database_credentials(config):
        return OFF  # DB-less stock install: nothing can configure a reader
    try:
        text = store_factory().get_raw_value_json("global", "*", SETTING_KEY)
    except Exception as exc:  # noqa: BLE001
        logger.error("strict_reader: settings read failed (%s); refusing to start",
                     type(exc).__name__)
        raise StrictReaderConfigError("strict_reader settings could not be read") from None
    if text is None:
        return OFF
    try:
        raw = json.loads(text)
    except ValueError:
        logger.error("strict_reader: stored value is not valid JSON; refusing to start")
        raise StrictReaderConfigError("strict_reader setting is not valid JSON") from None
    try:
        snap = parse(raw)
    except StrictReaderConfigError as exc:
        logger.error("strict_reader: %s; refusing to start", exc)
        raise
    if snap.error:
        logger.error("strict_reader: %s; bot %s is fenced and its turns are refused",
                     snap.error, snap.bot)
    elif snap.bot:
        logger.info("strict_reader: enabled for %s on house %s, skills %s",
                    snap.bot, snap.house, snap.skill_generation)
    return snap
