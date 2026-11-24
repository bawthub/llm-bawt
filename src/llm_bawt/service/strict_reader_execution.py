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

from ..bot_types import durable_agent_backend
from ..bots import Bot, load_named_bot
from .request_model_execution import BoundAgentLLMBawt, build_bound_instance, validate_bound_endpoint


STRICT_BACKEND = "claude-code"


class StrictReaderRefused(ValueError):
    """The strict path cannot serve this request; there is no fallback."""


def strict_reader_slug(service) -> str | None:
    slug = (getattr(service, "strict_reader_bot", None) or "").strip().lower()
    return slug or None


def is_strict_request(service, bot_id: str | None) -> bool:
    slug = strict_reader_slug(service)
    return bool(slug) and (bot_id or "").strip().lower() == slug


class StrictReaderLLMBawt(BoundAgentLLMBawt):
    """A request-owned instance built from one pre-read profile."""

    def __init__(self, *args, admitted_bot: Bot, **kwargs):
        self._admitted_bot = admitted_bot
        super().__init__(*args, **kwargs)
        self._seal_tools()

    def _seal_tools(self) -> None:
        # Tools are off at the bridge boundary, not just hidden from the prompt:
        # the turn command carries tools_disabled, and the bridge then runs it
        # with no MCP server, no built-in tools and a deny-all tool hook.
        self.client.tools_disabled = True

    def _lookup_bot(self, config, slug):
        if (slug or "").strip().lower() != self._admitted_bot.slug:
            raise StrictReaderRefused("Strict instance asked for a bot it was not admitted for")
        return self._admitted_bot

    def _fallback_bot(self, config):
        raise StrictReaderRefused("Strict instance has no default-bot fallback")

    def _tag_context_enabled(self) -> bool:
        # The family roster lists every bot; a strict turn omits it.
        return False

    def _resolve_skills(self, service_cls, engine):
        from ..skills.service import caps_from_settings

        def no_roster():
            raise StrictReaderRefused("Strict skill resolution may not list bots")

        service = service_cls(engine, list_bots=no_roster,
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
    # Only one direct ordinary turn is supported: no inter-bot, scheduled or
    # routed delivery, and no explicit thread selection.
    if getattr(request, "inter_bot_delivery_id", None) or getattr(request, "session_id", None):
        raise StrictReaderRefused("Strict execution supports only a direct ordinary turn")
    bot = read_bot(service.config, slug)
    if bot is None or bot.slug != slug:
        raise StrictReaderRefused("Strict reader bot profile is missing or mismatched")
    # Only the Claude Code bridge enforces the tools-off turn field.
    if durable_agent_backend(bot) != STRICT_BACKEND:
        raise StrictReaderRefused("Strict execution requires the claude-code backend")
    model = validate_bound_endpoint(service, bot, request.model)

    def build(**kwargs):
        return instance_type(admitted_bot=bot, **kwargs)

    return build_bound_instance(service, request, model, instance_type=build)
