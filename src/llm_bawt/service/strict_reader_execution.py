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

import hashlib
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
        # Explicit legacy is the configured roots, even while a store's
        # mutable SKILL_SOURCES_CURRENT link is set.
        from agent_bridge.skill_selection import configured_claude_skill_roots

        return AmbientSource(roots=configured_claude_skill_roots())
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


def strict_selection_fingerprint(service) -> dict:
    """The operator's strict selection as data: generation, pinned layers, house.

    Reads configuration only. A queued start stores it at receipt and its
    drain must see the same value; a session records it and resumes only
    under the same one.
    """
    name = (getattr(service, "strict_reader_skill_generation", None) or "").strip()
    layers = getattr(service, "strict_reader_skill_layers", None)
    digest = None
    if name != LEGACY_GENERATION and isinstance(layers, dict) and layers:
        canonical = json.dumps({str(k): str(v) for k, v in layers.items()},
                               sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(canonical.encode()).hexdigest()
    return {
        "bot": strict_reader_slug(service),
        "generation": name,
        "layers_sha256": digest,
        "house": (getattr(service, "strict_reader_house", None) or "").strip().lower(),
    }


#: ``sessions.session_metadata`` key: the strict selection a session's SDK
#: transcript runs under, and the one SDK key confirmed under it.
SESSION_SELECTION_KEY = "strict_reader_selection"


def bind_session_selection(engine, binding: dict | None, fingerprint: dict) -> dict | None:
    """Resume only the exact SDK key confirmed under ``fingerprint``; else start fresh.

    The pinned prompt carries the selected generation's skill text, so a
    resumed SDK session started under another selection, or written by an
    ordinary turn, would carry other skills. The thread's marker names the
    selection and the single SDK key a strict turn under it bound
    (``confirm_session_key``). Any other key, a missing marker or another
    selection drops the resume id (a cold start re-pins the current render)
    and resets the marker with no confirmed key.

    Every strict turn also mints a fresh run token, records it in the marker
    and carries it on its bridge command (``binding["strict_run"]``). Only a
    writeback presenting the current token is confirmed, so a late PUT from
    an ordinary run, an earlier strict run or another selection never
    inherits strict provenance.
    """
    import uuid

    from sqlalchemy import text

    binding = dict(binding or {})
    binding.pop("strict_run", None)
    session_id = str(binding.get("thread_session_id") or "").strip()
    if not session_id:
        return binding
    resume_id = str(binding.get("thread_resume_id") or "").strip() or None
    if engine is None:
        binding["thread_resume_id"] = None  # nothing proves the session's selection
        return binding
    run = uuid.uuid4().hex
    with engine.begin() as conn:
        metadata = conn.execute(
            text("SELECT session_metadata FROM sessions WHERE id=:sid FOR UPDATE"),
            {"sid": session_id},
        ).scalar_one_or_none()
        recorded = (metadata or {}).get(SESSION_SELECTION_KEY) if isinstance(metadata, dict) else None
        resumes = bool(resume_id and isinstance(recorded, dict) and recorded.get("selection") == fingerprint
                       and recorded.get("sdk_key") == resume_id)
        value = {"selection": fingerprint, "sdk_key": resume_id if resumes else None, "run": run}
        conn.execute(
            text("UPDATE sessions SET session_metadata = COALESCE(session_metadata, '{}'::jsonb)"
                 " || jsonb_build_object(CAST(:key AS text), CAST(:value AS jsonb)) WHERE id=:sid"),
            {"key": SESSION_SELECTION_KEY, "value": json.dumps(value), "sid": session_id},
        )
    if not resumes:
        binding["thread_resume_id"] = None
    binding["strict_run"] = run
    return binding


def confirm_session_key(service, bot_id: str | None, metadata: dict | None, session_key: str,
                        strict_run: str | None = None) -> dict:
    """The marker patch for an SDK key written back onto a thread.

    The new key becomes the confirmed strict key only when the writeback
    carries the marker's current run token (minted by the strict turn that
    produced it) while the thread's bot is still the strict reader under the
    marker's exact selection. Any other writeback (an ordinary or earlier run,
    strict disabled, retargeted or another selection) clears the marker so the
    key never inherits strict provenance. Returns ``{}`` when the thread has
    no marker.
    """
    recorded = (metadata or {}).get(SESSION_SELECTION_KEY) if isinstance(metadata, dict) else None
    if not isinstance(recorded, dict):
        return {}
    run = recorded.get("run")
    if (isinstance(run, str) and run and strict_run == run
            and is_strict_request(service, bot_id)
            and recorded.get("selection") == strict_selection_fingerprint(service)):
        return {SESSION_SELECTION_KEY: {"selection": recorded["selection"], "sdk_key": session_key,
                                        "run": run}}
    return {SESSION_SELECTION_KEY: None}


def clear_session_selection(engine, binding: dict | None) -> None:
    """An ordinary turn ran on this thread: its SDK key loses strict provenance."""
    from sqlalchemy import text

    session_id = str((binding or {}).get("thread_session_id") or "").strip()
    if not session_id or engine is None:
        return
    with engine.begin() as conn:
        conn.execute(
            text("UPDATE sessions SET session_metadata = session_metadata - CAST(:key AS text)"
                 " WHERE id=:sid AND session_metadata ? CAST(:key AS text)"),
            {"key": SESSION_SELECTION_KEY, "sid": session_id},
        )


def refuse_stamped_ordinary(service, request, bot_id: str | None) -> None:
    """A row admitted under strict mode never runs on the ordinary path.

    Its receipt-time stamp is a durable execution constraint: if the strict
    selection was disabled or retargeted before drain, the turn is refused
    before any ordinary model resolution or publish.
    """
    if isinstance(getattr(request, "_strict_selection_stamp", None), dict) and not is_strict_request(service, bot_id):
        raise StrictReaderRefused("Strict reader selection changed since this turn was received")


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
    fingerprint = strict_selection_fingerprint(service)
    if delivery_id and getattr(request, "_strict_selection_stamp", None) != fingerprint:
        # The selection changed between receipt and drain (or the row was
        # admitted before strict mode): never run it under another tree.
        raise StrictReaderRefused("Strict reader selection changed since this turn was received")
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
        instance = instance_type(admitted_bot=bot, skill_source=skill_source, house=house, **kwargs)
        instance.strict_selection = fingerprint
        return instance

    return build_bound_instance(service, request, model, instance_type=build)
