"""Strict named-reader execution: one admitted bot, never the shared registry.

A service may name exactly one eligible bot (``service.strict_reader_bot``).
A streaming request for that bot is routed here before ordinary model
resolution. It reads only that bot's own profile and tags, validates the
explicit request endpoint against them, and constructs a request-owned
instance from that profile. Nothing on this path consults, fills or
invalidates the shared bot registry, selects a default bot, or falls back to
another bot or model: a missing, disabled or mismatched profile, or a missing
or invalid endpoint, fails the request.

Unset (the default), every request keeps the ordinary path unchanged.
"""
from __future__ import annotations

import json
import re

from ..bot_types import durable_agent_backend
from ..bots import Bot, load_named_bot
from .request_model_execution import BoundAgentLLMBawt, build_bound_instance, validate_bound_endpoint


STRICT_BACKEND = "claude-code"


class StrictReaderRefused(ValueError):
    """The strict path cannot serve this request; there is no fallback."""


def strict_reader_slug(service) -> str | None:
    slug = (getattr(service, "strict_reader_bot", None) or "").strip().lower()
    return slug or None


LEGACY_GENERATION = "legacy"
_GENERATION_NAME = re.compile(r"^gen-[A-Za-z0-9_]{1,64}$")


def strict_skill_source(service):
    """The operator-selected skill tree for the strict reader; refuse otherwise.

    ``service.strict_reader_skill_generation`` is either ``"legacy"`` (the
    ordinary ambient roots, chosen explicitly) or one immutable generation
    directory name under ``<SKILL_SOURCES_ROOT>/generations``, never the
    mutable ``current`` link. A generation also needs
    ``service.strict_reader_skill_layers``, the exact ``{source: revision}``
    map it must have been built from. Unset, missing, escaping or mismatched
    fails the turn; nothing falls back to another tree.
    """
    from agent_bridge.skill_sources import AmbientSource

    from ..skills.source_config import store_from_env
    from ..skills.source_store import LAYERS_FILE

    name = (getattr(service, "strict_reader_skill_generation", None) or "").strip()
    if name == LEGACY_GENERATION:
        return AmbientSource()
    if not _GENERATION_NAME.fullmatch(name):
        raise StrictReaderRefused("Strict reader needs an explicit skill generation")
    expected = getattr(service, "strict_reader_skill_layers", None)
    if not isinstance(expected, dict) or not expected:
        raise StrictReaderRefused("Strict reader skill generation has no pinned layers")
    store = store_from_env()
    if not store.enabled:
        raise StrictReaderRefused("Skill sources root is not configured")
    generations = store.root / "generations"
    candidate = generations / name
    try:
        if candidate.is_symlink() or not candidate.is_dir():
            raise StrictReaderRefused("Strict reader skill generation is missing")
        resolved = candidate.resolve(strict=True)
        if resolved.parent != generations.resolve(strict=True):
            raise StrictReaderRefused("Strict reader skill generation escapes the store")
        layers = json.loads((resolved / LAYERS_FILE).read_text())
    except (OSError, ValueError) as exc:
        raise StrictReaderRefused("Strict reader skill generation is unreadable") from exc
    if layers != {str(k): str(v) for k, v in expected.items()}:
        raise StrictReaderRefused("Strict reader skill generation does not match its pin")
    return AmbientSource(roots=(resolved,))


def strict_house(service, bot) -> str:
    """The operator-selected house; the admitted profile must name the same one."""
    house = (getattr(service, "strict_reader_house", None) or "").strip().lower()
    if not house:
        raise StrictReaderRefused("Strict reader needs an explicit house route")
    if (getattr(bot, "house", None) or "").strip().lower() != house:
        raise StrictReaderRefused("Strict reader bot is not on its selected house")
    return house


def is_strict_request(service, bot_id: str | None) -> bool:
    slug = strict_reader_slug(service)
    return bool(slug) and (bot_id or "").strip().lower() == slug


def effective_bot_id(service, bot_id: str | None) -> str | None:
    """The bot a request will actually reach: its own ID, else the service default.

    Reads only configuration (never the bot registry), so ingress guards can
    refuse the strict bot even when it is reached as the default.
    """
    explicit = (bot_id or "").strip()
    if explicit:
        return explicit
    override = getattr(service, "_default_bot_override", None)
    if override:
        return override
    from ..utils.bot_identity import optional_default_bot_id
    return optional_default_bot_id(getattr(service, "config", None))


class StrictReaderLLMBawt(BoundAgentLLMBawt):
    """A request-owned instance built from one pre-read profile."""

    #: Turn-level extras that need tools or fleet-wide reads (working memory,
    #: assigned cards) are omitted for this instance.
    turn_context_sealed = True

    def __init__(self, *args, admitted_bot: Bot, skill_source=None, house: str | None = None, **kwargs):
        self._admitted_bot = admitted_bot
        self._skill_source = skill_source
        self._house = house
        super().__init__(*args, **kwargs)
        self._seal_tools()

    def _seal_tools(self) -> None:
        # Tools are off at the bridge boundary, not just hidden from the prompt:
        # the turn command carries tools_disabled, and the bridge then runs it
        # with no MCP server, no built-in tools and a deny-all tool hook.
        self.client.tools_disabled = True
        # The publisher routes a tools-off turn only to this house, no fallback.
        self.client.strict_house = self._house

    def _lookup_bot(self, config, slug):
        if (slug or "").strip().lower() != self._admitted_bot.slug:
            raise StrictReaderRefused("Strict instance asked for a bot it was not admitted for")
        return self._admitted_bot

    def _fallback_bot(self, config):
        raise StrictReaderRefused("Strict instance has no default-bot fallback")

    def _tag_context_enabled(self) -> bool:
        # The family roster lists every bot; a strict turn omits it.
        return False

    def _skills_fail_closed(self) -> bool:
        return True

    def _resolve_skills(self, service_cls, engine):
        from ..skills.service import caps_from_settings

        def no_roster():
            raise StrictReaderRefused("Strict skill resolution may not list bots")

        if self._skill_source is None:
            raise StrictReaderRefused("Strict instance has no selected skill tree")
        service = service_cls(engine, list_bots=no_roster, source=self._skill_source,
                              caps=lambda: caps_from_settings(self.config))
        return service.resolve_admitted(self._admitted_bot)


def create_strict_instance(service, request, *, read_bot=load_named_bot,
                           instance_type=StrictReaderLLMBawt):
    slug = strict_reader_slug(service)
    requested = (request.bot_id or "").strip().lower()
    if not slug or requested != slug:
        raise StrictReaderRefused("Request is not for the strict reader's bot")
    if not request.user or not request.model:
        raise StrictReaderRefused("Strict execution needs an explicit user and model")
    # Only one ordinary human turn is supported: no inter-bot, scheduled or
    # routed delivery, and no explicit thread selection. A durable claim is
    # accepted only when the claim validator marked it as that same ordinary
    # human start; a request field alone never qualifies.
    delivery_id = getattr(request, "inter_bot_delivery_id", None)
    if delivery_id and getattr(request, "_ordinary_start_claim", None) != delivery_id:
        raise StrictReaderRefused("Strict execution supports only a direct ordinary turn")
    if getattr(request, "session_id", None):
        raise StrictReaderRefused("Strict execution supports only a direct ordinary turn")
    bot = read_bot(service.config, slug)
    if bot is None or bot.slug != slug:
        raise StrictReaderRefused("Strict reader bot profile is missing or mismatched")
    # Only the Claude Code bridge enforces the tools-off turn field.
    if durable_agent_backend(bot) != STRICT_BACKEND:
        raise StrictReaderRefused("Strict execution requires the claude-code backend")
    house = strict_house(service, bot)
    skill_source = strict_skill_source(service)
    model = validate_bound_endpoint(service, bot, request.model)

    def build(**kwargs):
        return instance_type(admitted_bot=bot, skill_source=skill_source, house=house, **kwargs)

    return build_bound_instance(service, request, model, instance_type=build)
