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

from ..bots import Bot, load_named_bot
from .request_model_execution import BoundAgentLLMBawt, build_bound_instance, validate_bound_endpoint


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

    def _lookup_bot(self, config, slug):
        if (slug or "").strip().lower() != self._admitted_bot.slug:
            raise StrictReaderRefused("Strict instance asked for a bot it was not admitted for")
        return self._admitted_bot

    def _fallback_bot(self, config):
        raise StrictReaderRefused("Strict instance has no default-bot fallback")


def create_strict_instance(service, request, *, read_bot=load_named_bot,
                           instance_type=StrictReaderLLMBawt):
    slug = strict_reader_slug(service)
    requested = (request.bot_id or "").strip().lower()
    if not slug or requested != slug:
        raise StrictReaderRefused("Request is not for the strict reader's bot")
    if not request.user or not request.model:
        raise StrictReaderRefused("Strict execution needs an explicit user and model")
    bot = read_bot(service.config, slug)
    if bot is None or bot.slug != slug:
        raise StrictReaderRefused("Strict reader bot profile is missing or mismatched")
    model = validate_bound_endpoint(service, bot, request.model)

    def build(**kwargs):
        return instance_type(admitted_bot=bot, **kwargs)

    return build_bound_instance(service, request, model, instance_type=build)
