"""Bot system for llm-bawt.

Bots are AI personalities with their own system prompts and isolated memory.
Bot definitions are loaded exclusively from the database (bot_profiles table).
"""

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from .bot_types import normalize_bot_type
from .utils.bot_identity import MissingBotIdentity, configured_default_bot_id

logger = logging.getLogger(__name__)


@dataclass
class Bot:
    """A bot personality with its own system prompt and capabilities."""

    slug: str  # Unique identifier (e.g., "helper", "reviewer")
    name: str  # Display name (e.g., "Helper", "Reviewer")
    description: str  # Short description for --list-bots
    system_prompt: str  # The system message sent to the LLM
    prompt_override_id: int | None = None  # Active persona override (prompt_templates.id); None => use system_prompt
    requires_memory: bool = True  # Whether this bot needs database/memory persistence
    voice_optimized: bool = False  # Whether output is optimized for TTS
    tts_mode: bool = False  # Default for tts_mode request flag (TTS formatting instructions)
    include_summaries: bool = True  # Default for include_summaries request flag
    include_in_global_search: bool = True  # Legacy mirror of "not private" (TASK-1329 §4.1)
    tags: list[str] | None = None  # Assigned bot tags, sorted; None = could not be read
    default_voice: str | None = None  # Optional default voice identifier for TTS/voice clients
    default_model: str | None = None  # Default model alias for this bot
    harness: str | None = None  # Turn execution harness
    endpoint_id: int | None = None  # Canonical normalized model endpoint
    house: str | None = None  # TASK-2077 command slot; None = mansion
    house_pinned: bool = False  # TASK-2080: keep the house through /new and the janitor
    uses_tools: bool = False  # Whether this bot can use tools (memory search, etc.)
    uses_search: bool = False  # Whether this bot can search the web
    uses_home_assistant: bool = False  # Whether this bot can control Home Assistant via MCP
    color: str | None = None  # UI color, normally a #rrggbb hex
    cli_color: str | None = None  # Optional Rich color name for the CLI
    avatar: str | None = None  # Optional emoji or image URL for bot avatar
    avatar_render: str | None = None  # Self-hosted data: URL render of avatar (no CDN at runtime)
    nextcloud: dict | None = None  # Nextcloud integration config (bot_id, secret, etc.)
    bot_type: str = "chat"  # High-level bot type (chat or agent)
    agent_backend: str | None = None  # External agent backend slug (e.g. "claude-code")
    agent_backend_config: dict[str, Any] = field(default_factory=dict)  # Backend-specific config

    def __post_init__(self):
        self.slug = self.slug.lower().strip()
        self.bot_type = normalize_bot_type(self.bot_type, self.agent_backend)
        if not self.harness:
            self.harness = self.agent_backend or "chat"


@dataclass
class ModelSelection:
    """Resolved model selection with source metadata."""

    alias: str | None
    source: str  # explicit | bot_default | config_default | none


# Global bot registry — populated from DB on first access
BUILTIN_BOTS: dict[str, Bot] = {}
_SYSTEM_PROMPTS: dict[str, str] = {}
_RAW_BOT_DATA: dict[str, dict] = {}
_FORCE_RELOAD: bool = False
_LOADED: bool = False
# Why the last load failed, or None. A failed load is never an empty registry:
# the previous registry is kept and the load retried after a short backoff
# (TASK-1819). An explicit invalidate retries at once.
_LOAD_ERROR: str | None = None
_RETRY_AFTER: float = 0.0
_LOAD_RETRY_SECONDS = 15.0


class BotRegistryUnavailable(RuntimeError):
    """The bot profile database could not be read; distinct from zero bots."""


# ── Named system prompts (code constants, not runtime config) ──────────
_BUILTIN_SYSTEM_PROMPTS: dict[str, str] = {
    "refine": """\
You are a Prompt-Refinement Tool that transforms raw user messages into natural, human-like conversational prompts.

You MUST:
1. Take the user's raw input and convert it into a prompt that will generate natural, casual human responses
2. Ensure prompts encourage conversational, intimate, and authentic replies - not assistant-like or service-oriented responses
3. Preserve the core intent of the user's input (e.g., a question asking for an opinion should result in a prompt asking for an opinion, not a command to act)
4. Clearly reference the core topic or question from the user's raw input within the refined instructions part of the prompt.
5. Format the output using the exact template below, including the labels:
   What the user asked: <user's raw input>
   Refined prompt: <refined instructions>

You MUST NOT:
1. Create prompts that lead to formal, assistant-like, or overly professional responses
2. Include phrases like "assist them," "help the user," or any service-oriented language within the refined instructions
3. Change the fundamental nature of the request (e.g., don't turn a question about preference into a command to perform an action)
4. Add any text before or after the required output template
5. Include meta-commentary about the prompt or process
6. Include example questions or specific suggestions within the refined instructions (e.g., avoid phrases like "such as '...'")

Example:
Raw user input: "hi"
INCORRECT response: "Greet the user in a friendly and professional tone, and ask how you can assist them today."
CORRECT response:
What the user asked: hi
Refined prompt: Respond to this greeting as a companion would, with warmth and authenticity. Keep it brief and natural.

Example:
Raw user input: "tell me about dogs"
INCORRECT response: "Provide information about dogs in a helpful and informative manner."
CORRECT response:
What the user asked: tell me about dogs
Refined prompt: The user wants to talk about dogs. Share some interesting thoughts about dogs as if chatting with an intimate companion who loves pets. Be conversational and authentic, not like you're giving a formal presentation.

Your entire response MUST follow the specified template: provide the user's raw input and then the refined instructions on how the final LLM should respond.
""",
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge override into base, with override taking priority.

    For nested dicts, merges recursively. For other types, override wins.
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def invalidate_bots_cache() -> None:
    """Force the next access to reload bots from DB."""
    global _FORCE_RELOAD, _RETRY_AFTER
    _FORCE_RELOAD = True
    _RETRY_AFTER = 0.0


def _check_reload() -> bool:
    """Reload bots from DB if invalidated or not yet loaded.

    Returns True if reload was performed.
    """
    global _FORCE_RELOAD, _LOADED

    if (_FORCE_RELOAD or not _LOADED) and time.monotonic() >= _RETRY_AFTER:
        _FORCE_RELOAD = False
        _load_bots_config()
        return True
    return False


def bot_registry_error() -> str | None:
    """Why the bot registry could not be loaded from the DB, or None when it was.

    Callers that report the registry (``/v1/bots``) must not present an
    unreadable DB as "no bots configured".
    """
    _check_reload()
    return _LOAD_ERROR


def _load_db_bot_profiles() -> dict[str, dict[str, Any]]:
    """Load all bot profiles from DB, keyed by slug.

    Returns an empty dict only when this process has no database configured
    (a remote CLI). A configured database that cannot be read raises
    :class:`BotRegistryUnavailable`; it is never reported as zero bots.
    """
    store = None
    try:
        from llm_bawt.runtime_settings import BotProfileStore
        from llm_bawt.utils.config import Config, has_database_credentials

        config = Config()

        if not has_database_credentials(config):
            # Expected for remote/API clients (e.g. the `llm` CLI on a laptop):
            # bot profiles are resolved server-side via the service API, so the
            # absence of local DB creds is normal, not a warning condition.
            logger.debug("No local database credentials — bot profiles will come from the service API if used")
            return {}

        store = BotProfileStore(config)
        if store.engine is None:
            raise BotRegistryUnavailable(store.init_error or "bot profiles DB engine unavailable")

        rows = store.list_all()
        tags_by_bot = _load_bot_tags(store.engine)
        profiles: dict[str, dict[str, Any]] = {}
        for row in rows:
            slug = (row.slug or "").strip().lower()
            if not slug:
                continue
            profiles[slug] = _profile_entry(
                row, None if tags_by_bot is None else tags_by_bot.get(slug, []),
            )
        return profiles
    except BotRegistryUnavailable:
        raise
    except Exception as e:
        raise BotRegistryUnavailable(f"could not read bot profiles: {e}") from e


def _profile_entry(row, tags: list[str] | None) -> dict[str, Any]:
    """One registry entry for a bot_profiles row (shared by the registry and the named reader)."""
    entry: dict[str, Any] = {
        "name": row.name,
        "description": row.description,
        "system_prompt": row.system_prompt,
        "prompt_override_id": getattr(row, "prompt_override_id", None),
        "requires_memory": row.requires_memory,
        "voice_optimized": row.voice_optimized,
        "tts_mode": row.tts_mode,
        "include_summaries": row.include_summaries,
        "include_in_global_search": row.include_in_global_search,
        "tags": tags,
        "uses_tools": row.uses_tools,
        "uses_search": row.uses_search,
        "uses_home_assistant": row.uses_home_assistant,
    }
    if row.default_model is not None:
        entry["default_model"] = row.default_model
    entry["harness"] = getattr(row, "harness", None)
    entry["endpoint_id"] = getattr(row, "endpoint_id", None)
    if getattr(row, "house", None):
        entry["house"] = row.house
    if getattr(row, "house_pinned", False):
        entry["house_pinned"] = True
    if row.color is not None:
        entry["color"] = row.color
    if getattr(row, "cli_color", None):
        entry["cli_color"] = row.cli_color
    if row.avatar is not None:
        entry["avatar"] = row.avatar
    if getattr(row, "avatar_render", None):
        entry["avatar_render"] = row.avatar_render
    if row.default_voice is not None:
        entry["default_voice"] = row.default_voice
    if row.nextcloud_config is not None:
        entry["nextcloud"] = row.nextcloud_config
    entry["bot_type"] = normalize_bot_type(
        getattr(row, "bot_type", None), row.agent_backend,
    )
    if row.agent_backend is not None:
        entry["agent_backend"] = row.agent_backend
    if row.agent_backend_config is not None:
        entry["agent_backend_config"] = row.agent_backend_config
    return entry


def _load_bot_tags(engine) -> dict[str, list[str]] | None:
    """Assigned tags per bot, or None (unavailable) so a tag read failure never drops a bot."""
    try:
        from llm_bawt.bot_tags.store import BotTagStore
        return BotTagStore(engine).tags_by_bot()
    except Exception as e:
        logger.warning("Could not load bot tags; bots report them unavailable: %s", e)
        return None


class _ReadOnlyProfileReader:
    """Slug-filtered profile read over the existing schema.

    Unlike BotProfileStore it never bootstraps (no DDL, column migration or the
    fleet-wide tag migration); a schema that is not ready refuses instead.
    """

    def __init__(self, engine):
        self.engine = engine

    @classmethod
    def open(cls, config):
        from llm_bawt.utils.config import has_database_credentials
        from llm_bawt.utils.db import get_shared_engine
        if not has_database_credentials(config):
            raise BotRegistryUnavailable("strict reader needs a configured database")
        engine = get_shared_engine(config)
        if engine is None:
            raise BotRegistryUnavailable("bot profiles DB engine unavailable")
        from sqlalchemy import inspect
        names = set(inspect(engine).get_table_names())
        if not {"bot_profiles", "bot_profile_tags"} <= names:
            raise BotRegistryUnavailable("bot profile schema not bootstrapped; strict reader refuses")
        return cls(engine)

    def get(self, slug: str):
        from sqlmodel import Session, select
        from llm_bawt.runtime_settings import BotProfile
        with Session(self.engine) as session:
            return session.exec(select(BotProfile).where(BotProfile.slug == slug)).first()


def load_named_bot(config, slug: str, *, profile_store=None, tag_store=None) -> Bot | None:
    """Read exactly one bot by slug, bypassing the shared registry.

    Strict callers use this so no other bot's profile or tags are read: one
    slug-filtered profile row and that bot's own tag rows. It never consults,
    fills or invalidates the registry cache and never falls back to a default
    bot. Returns None when the profile is absent; read errors propagate.
    """
    normalized = (slug or "").strip().lower()
    if not normalized:
        return None
    if profile_store is None:
        profile_store = _ReadOnlyProfileReader.open(config)
    row = profile_store.get(normalized)
    if row is None or (row.slug or "").strip().lower() != normalized:
        return None
    if tag_store is None:
        from llm_bawt.bot_tags.store import BotTagStore
        tag_store = BotTagStore(profile_store.engine)
    entry = _profile_entry(row, tag_store.tags_for(normalized))
    return _bot_from_entry(normalized, entry, entry["bot_type"])


def _bot_from_entry(slug: str, data: dict[str, Any], resolved_bot_type: str) -> Bot:
    return Bot(
        slug=slug,
        name=data.get("name", slug.title()),
        description=data.get("description", ""),
        system_prompt=data.get("system_prompt", "You are a helpful assistant."),
        prompt_override_id=data.get("prompt_override_id"),
        requires_memory=data.get("requires_memory", True),
        voice_optimized=data.get("voice_optimized", False),
        tts_mode=data.get("tts_mode", False),
        include_summaries=data.get("include_summaries", True),
        include_in_global_search=data.get("include_in_global_search", True),
        tags=data.get("tags"),
        default_voice=data.get("default_voice"),
        default_model=data.get("default_model"),
        harness=data.get("harness"),
        endpoint_id=data.get("endpoint_id"),
        house=data.get("house") or None,
        house_pinned=bool(data.get("house_pinned", False)),
        uses_tools=data.get("uses_tools", False),
        uses_search=data.get("uses_search", False),
        uses_home_assistant=data.get("uses_home_assistant", False),
        color=data.get("color"),
        cli_color=data.get("cli_color"),
        avatar=data.get("avatar"),
        avatar_render=data.get("avatar_render"),
        nextcloud=data.get("nextcloud"),
        bot_type=resolved_bot_type,
        agent_backend=data.get("agent_backend"),
        agent_backend_config=data.get("agent_backend_config") or {},
    )


def _load_bots_config() -> None:
    """Load bot definitions from the database.

    This is the sole source of bot data — no YAML fallback.
    """
    global BUILTIN_BOTS, _SYSTEM_PROMPTS, _RAW_BOT_DATA, _LOADED
    global _LOAD_ERROR, _RETRY_AFTER, _FORCE_RELOAD

    # Named prompts are code constants and never depend on the DB read.
    _SYSTEM_PROMPTS.clear()
    _SYSTEM_PROMPTS.update(_BUILTIN_SYSTEM_PROMPTS)

    try:
        db_profiles = _load_db_bot_profiles()
    except BotRegistryUnavailable as exc:
        _LOAD_ERROR = str(exc)
        _RETRY_AFTER = time.monotonic() + _LOAD_RETRY_SECONDS
        _FORCE_RELOAD = True
        logger.error(
            "Bot registry load failed; keeping %d previously loaded bots, retrying in %.0fs: %s",
            len(BUILTIN_BOTS), _LOAD_RETRY_SECONDS, exc,
        )
        return

    bots: dict[str, Bot] = {}
    raw_data: dict[str, dict] = {}

    for slug, data in db_profiles.items():
        resolved_bot_type = normalize_bot_type(
            data.get("bot_type"),
            data.get("agent_backend"),
        )
        raw_entry = dict(data)
        raw_entry["bot_type"] = resolved_bot_type
        raw_data[slug] = raw_entry

        bots[slug] = _bot_from_entry(slug, data, resolved_bot_type)

    BUILTIN_BOTS.clear()
    BUILTIN_BOTS.update(bots)
    _RAW_BOT_DATA.clear()
    _RAW_BOT_DATA.update(raw_data)

    _LOADED = True
    _LOAD_ERROR = None
    _RETRY_AFTER = 0.0

    logger.debug("Loaded %d bots from database", len(bots))


def get_raw_bot_data(slug: str) -> dict | None:
    """Get raw bot config data by slug (for integrations).

    Returns the DB-sourced data dict for a bot, including integration-specific
    sections like 'nextcloud'.
    """
    _check_reload()
    return _RAW_BOT_DATA.get(slug.lower().strip())


def get_all_raw_bot_data() -> dict[str, dict]:
    """Get all raw bot config data (for integrations)."""
    _check_reload()
    return _RAW_BOT_DATA.copy()


def get_bot_settings_template() -> dict[str, Any]:
    """Get default bot settings.

    Returns RuntimeTunables class-level defaults. Per-bot and global overrides
    are resolved at runtime by RuntimeSettingsResolver.
    """
    from .utils.config import RuntimeTunables

    defaults: dict[str, Any] = {}
    for fname in RuntimeTunables.model_fields:
        field_info = RuntimeTunables.model_fields[fname]
        if field_info.default is not None:
            defaults[fname.lower()] = field_info.default
    return defaults


def save_user_bot_config(slug: str, section: str, data: dict) -> None:
    """Save bot-specific integration config to DB-backed bot profile."""
    normalized_slug = (slug or "").strip().lower()
    if not normalized_slug:
        raise ValueError("Bot slug is required")

    if section != "nextcloud":
        raise ValueError(f"Unsupported bot config section: {section}")

    from llm_bawt.utils.config import Config, has_database_credentials

    config = Config()
    if not has_database_credentials(config):
        raise RuntimeError("Database credentials required to persist bot profile config")

    from llm_bawt.runtime_settings import BotProfileStore

    profile_store = BotProfileStore(config)
    if profile_store.engine is None:
        raise RuntimeError("Bot profiles DB unavailable")

    if profile_store.get(normalized_slug) is None:
        raise ValueError(f"Bot profile not found: {normalized_slug}")
    # A section update is never an alternate bot creation path; recheck under
    # the store's write lock if the profile is removed after the read above.
    profile_store.upsert({"slug": normalized_slug, "nextcloud_config": data}, require_existing=True)
    _load_bots_config()


def remove_user_bot_section(slug: str, section: str) -> bool:
    """Remove a section from DB-backed bot profile."""
    normalized_slug = (slug or "").strip().lower()
    if not normalized_slug:
        return False

    if section != "nextcloud":
        return False

    from llm_bawt.utils.config import Config, has_database_credentials

    config = Config()
    if not has_database_credentials(config):
        return False

    from llm_bawt.runtime_settings import BotProfileStore

    profile_store = BotProfileStore(config)
    if profile_store.engine is None:
        return False

    profile = profile_store.get(normalized_slug)
    if profile is None or profile.nextcloud_config is None:
        return False

    from sqlalchemy import text

    with profile_store.engine.begin() as conn:
        cleared = conn.execute(text("""
            UPDATE bot_profiles SET nextcloud_config=NULL, updated_at=CURRENT_TIMESTAMP
            WHERE slug=:slug AND nextcloud_config IS NOT NULL
        """), {"slug": normalized_slug}).rowcount > 0
    invalidate_bots_cache()
    return cleared


def get_system_prompt(name: str) -> str | None:
    """Get a named system prompt (e.g., 'refine')."""
    _check_reload()
    return _SYSTEM_PROMPTS.get(name)


class BotManager:
    """Manages bot loading and retrieval."""

    def __init__(self, config: Any = None, local_only: bool = False):
        self.config = config
        # local_only is deprecated — DB is the sole source.
        # Kept as a parameter for API compatibility but ignored.
        if local_only:
            logger.debug("BotManager local_only=True is deprecated; loading from DB")
        logger.debug("BotManager initialized with %d bots", len(self._bots))

    @property
    def _bots(self) -> dict[str, Bot]:
        """Get bots from global registry, checking for reload."""
        _check_reload()
        return BUILTIN_BOTS

    def get_bot(self, slug: str) -> Bot | None:
        """Get a bot by slug.

        Args:
            slug: The bot identifier (case-insensitive)

        Returns:
            The Bot instance, or None if not found
        """
        return self._bots.get(slug.lower().strip())

    def get_default_bot(self, local_mode: bool = False) -> Bot:
        """Get the default bot based on mode.

        Args:
            local_mode: Deprecated, ignored.

        Returns:
            The default Bot instance
        """
        # TASK-1373: the configured DEFAULT_BOT is the only default; a missing
        # or unknown one fails loudly instead of picking some other bot.
        default_slug = configured_default_bot_id(self.config)
        bot = self._bots.get(default_slug)
        if not bot:
            raise MissingBotIdentity(f"configured DEFAULT_BOT {default_slug!r} is not a known bot")
        return bot

    def _is_model_alias(self, alias: str) -> bool:
        """Check if an alias exists in model definitions."""
        if not self.config or not hasattr(self.config, "defined_models"):
            return False
        models = getattr(self.config, "defined_models", {}).get("models", {})
        return alias in models

    def select_model(
        self,
        requested_model: str | None,
        bot_slug: str | None = None,
        local_mode: bool = False,
    ) -> ModelSelection:
        """Resolve the effective model alias for a request.

        Priority:
        0) Agent-backend bots ALWAYS use their default_model (bound to backend)
        1) Explicit request model (unless it matches a bot slug and isn't a model alias)
        2) Bot default_model
        3) Config DEFAULT_MODEL_ALIAS
        """
        # Agent-backend bots are bound to their backend — the client-provided
        # model must be ignored so requests always route through the backend.
        bot = self.get_bot(bot_slug) if bot_slug else None
        if bot and bot.agent_backend and bot.default_model:
            from .model_catalog import bot_model_ref

            model_ref = bot_model_ref(self.config, bot) if self.config else bot.default_model
            if self.config:
                self.config.resolve_model(model_ref, harness=bot.harness)
            return ModelSelection(alias=model_ref, source="bot_default")

        model_alias = requested_model.strip() if requested_model else None
        if model_alias == "":
            model_alias = None

        # If the provided model matches a bot slug (and isn't a model alias),
        # treat it as no model specified.
        if model_alias:
            bot_match = self.get_bot(model_alias)
            if bot_match and not self._is_model_alias(model_alias):
                model_alias = None

        if model_alias:
            return ModelSelection(alias=model_alias, source="explicit")

        if not bot:
            bot = self.get_default_bot(local_mode=local_mode)

        if bot and bot.default_model:
            return ModelSelection(alias=bot.default_model, source="bot_default")

        config_default = getattr(self.config, "DEFAULT_MODEL_ALIAS", None) if self.config else None
        if config_default:
            return ModelSelection(alias=config_default, source="config_default")

        return ModelSelection(alias=None, source="none")

    def list_bots(self) -> list[Bot]:
        """List all available bots.

        Returns:
            List of all Bot instances, sorted by slug
        """
        return sorted(self._bots.values(), key=lambda b: b.slug)

def get_bot(slug: str) -> Bot | None:
    """Convenience function to get a bot by slug.

    Args:
        slug: The bot identifier (case-insensitive)

    Returns:
        The Bot instance, or None if not found
    """
    _check_reload()
    return BUILTIN_BOTS.get(slug.lower().strip())


def strip_emotes(text: str) -> str:
    """Strip roleplay emotes/actions from text for TTS output.

    RP-tuned models often include *action* text despite prompt instructions.
    This function removes them for clean TTS output.

    Patterns removed:
    - *action text* (asterisk-wrapped actions)
    - ::action:: (colon-wrapped actions)
    - (action) when on its own line or at sentence boundaries
    - Multiple consecutive whitespace normalized

    Args:
        text: The raw LLM response text

    Returns:
        Clean text suitable for TTS
    """
    if not text:
        return text
    for stage in _EMOTE_STAGES:
        text = stage.pattern.sub('', text)
    return _normalize_emote_whitespace(text)


def _normalize_emote_whitespace(text: str) -> str:
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# ---------------------------------------------------------------------------
# Emote patterns. ``strip_emotes`` (complete text) and ``StreamingEmoteFilter``
# (incremental text) both apply exactly these stages in this order, so the
# streamed result always concatenates to ``strip_emotes`` of the full text.
#
# Each stage pairs its regex with a ``pending`` function that returns the
# earliest index whose match decision still depends on text that has not
# arrived yet (``len(text)`` when everything seen so far is decided).
# ---------------------------------------------------------------------------

# *smiles warmly*, *pauses* — single-line, not part of ** runs.
_EMOTE_STAR_RE = re.compile(r'(?<!\*)\*(?=\S)[^*\r\n]*\S\*(?!\*)')
# ::waves:: — single-line, opener at start or after whitespace (spares std::vector).
_EMOTE_COLON_RE = re.compile(r'(?<!\S)::(?=\S)[^:\r\n]*\S::')
# A line holding only "(action)". ``\s`` and ``[^)]`` can cross newlines.
_EMOTE_PAREN_LINE_RE = re.compile(r'^\s*\([^)]+\)\s*$', re.MULTILINE)

_STAR_STOP_RE = re.compile(r'[*\r\n]')
_COLON_STOP_RE = re.compile(r'[:\r\n]')
_NON_SPACE_RE = re.compile(r'\S')


def _pending_star(text: str) -> int:
    k = 0
    while (i := text.find('*', k)) != -1:
        # A match from i ends at the next '*' and then looks one char past it;
        # a line break or another char settles it sooner.
        stop = _STAR_STOP_RE.search(text, i + 1)
        if stop is None or (text[stop.start()] == '*' and stop.start() + 1 >= len(text)):
            return i
        m = _EMOTE_STAR_RE.match(text, i)
        k = m.end() if m else i + 1
    return len(text)


def _pending_colon(text: str) -> int:
    k = 0
    while (i := text.find('::', k)) != -1:
        if i > 0 and not text[i - 1].isspace():
            k = i + 1  # (?<!\S) fails whatever follows
            continue
        # A match from i reads to the first ':' after the opener plus the two
        # chars after it (``\S::`` may end on that colon or just before it).
        stop = _COLON_STOP_RE.search(text, i + 2)
        if stop is None or (text[stop.start()] == ':' and stop.start() + 2 >= len(text)):
            return i
        m = _EMOTE_COLON_RE.match(text, i)
        k = m.end() if m else i + 1
    # A trailing ':' that could still become an opener.
    n = len(text)
    if n > k and text.endswith(':') and (n == 1 or text[-2].isspace()):
        return n - 1
    return n


def _next_line_start(text: str, pos: int) -> int:
    if pos == 0 or text[pos - 1] == '\n':
        return pos
    nl = text.find('\n', pos)
    return -1 if nl == -1 else nl + 1


def _pending_paren(text: str) -> int:
    n = len(text)
    p = 0
    while 0 <= p < n:
        q = _NON_SPACE_RE.search(text, p)
        if q is None:
            return p  # only whitespace so far; a '(' may follow
        if text[q.start()] == '(':
            close = text.find(')', q.start() + 1)
            if close == -1:
                return p
            # ``\s*$`` after ')' is settled by the next non-space char.
            if _NON_SPACE_RE.search(text, close + 1) is None:
                return p
            m = _EMOTE_PAREN_LINE_RE.match(text, p)
            if m:
                p = _next_line_start(text, m.end())
                continue
        p = _next_line_start(text, p + 1)
    return n


@dataclass(frozen=True)
class _EmoteStage:
    pattern: re.Pattern
    pending: Any  # Callable[[str], int]


_EMOTE_STAGES = (
    _EmoteStage(_EMOTE_STAR_RE, _pending_star),
    _EmoteStage(_EMOTE_COLON_RE, _pending_colon),
    _EmoteStage(_EMOTE_PAREN_LINE_RE, _pending_paren),
)


def _strip_emotes_settled(text: str) -> str:
    """``strip_emotes`` of the part of ``text`` that later text cannot change.

    Each stage strips only up to its first undecided position (with trailing
    whitespace held back, since ``\\s*$`` and normalization depend on what
    follows), and the next stage sees that settled output as its own
    still-growing input. The result is always a prefix of ``strip_emotes`` of
    any continuation of ``text``.
    """
    for stage in _EMOTE_STAGES:
        settled = text[: stage.pending(text)].rstrip()
        text = stage.pattern.sub('', settled)
    return _normalize_emote_whitespace(text)


def scrub_for_tts(text: str) -> str:
    """Scrub a COMPLETE assistant response into clean, speakable text for TTS.

    This is the single source of truth for TTS-safe text. It removes Markdown
    structure that reads terribly aloud (headers, emphasis markers, code fences,
    link/image URLs, list bullets, blockquote/table punctuation) and then applies
    ``strip_emotes`` for the roleplay-emote + whitespace-normalization pass.

    Markdown cannot be scrubbed reliably mid-stream (fences/links/tables span
    token boundaries), so this MUST run on the finalized full response — never
    per-chunk.

    Args:
        text: The full, finalized LLM response text.

    Returns:
        Plain text suitable for speech synthesis.
    """
    if not text:
        return text

    # NOTE: leading-indent anchors use ``[ \t]`` (never ``\s``) — ``\s`` matches
    # newlines and would swallow the blank lines that separate blocks.

    # Fenced code blocks: drop the ``` fence lines (and any language tag), keep
    # the inner content so meaning isn't lost — reading fence markers aloud is
    # noise, deleting the code entirely could drop the answer.
    text = re.sub(r'^[ \t]*```[^\n]*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^[ \t]*~~~[^\n]*$', '', text, flags=re.MULTILINE)

    # Images: ![alt](url) -> alt text (URL is unspeakable).
    text = re.sub(r'!\[([^\]]*)\]\([^)]*\)', r'\1', text)
    # Links: [text](url) -> text.
    text = re.sub(r'\[([^\]]*)\]\([^)]*\)', r'\1', text)

    # Inline code: strip the backticks, keep the token.
    text = re.sub(r'`+([^`]*)`+', r'\1', text)

    # ATX headers: strip leading #'s (and trailing closing #'s).
    text = re.sub(r'^[ \t]{0,3}#{1,6}[ \t]*', '', text, flags=re.MULTILINE)
    text = re.sub(r'[ \t]+#+[ \t]*$', '', text, flags=re.MULTILINE)

    # Blockquote markers.
    text = re.sub(r'^[ \t]{0,3}>+[ \t]?', '', text, flags=re.MULTILINE)

    # Horizontal rules (---, ***, ___ on their own line). Must run before the
    # unordered-list rule so a `---` rule isn't mistaken for a `-` bullet.
    text = re.sub(r'^[ \t]{0,3}([-*_])(?:[ \t]*\1){2,}[ \t]*$', '', text, flags=re.MULTILINE)

    # Table separator rows: | --- | :--: | etc.
    text = re.sub(r'^[ \t]{0,3}\|?[ \t:]*-{3,}[ \t:|-]*$', '', text, flags=re.MULTILINE)
    # Remaining table cell pipes -> spaces so cells read as a phrase.
    text = re.sub(r'\|', ' ', text)

    # List markers (unordered then ordered), keep the item text.
    text = re.sub(r'^[ \t]*[-*+][ \t]+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^[ \t]*\d+[.)][ \t]+', '', text, flags=re.MULTILINE)

    # Multi-char emphasis markers (**bold**, __under__, ~~strike~~). Single `*`
    # pairs are handled by strip_emotes below (which deletes *action* emotes).
    text = re.sub(r'(\*\*|__|~~)', '', text)

    # Emote + whitespace normalization pass (single source for that logic).
    return strip_emotes(text)


def should_scrub_for_tts(bot, request=None) -> bool:
    """Single source of truth: does THIS turn's output get scrubbed for TTS?

    Consolidates every "is this a voice/TTS-scrubbed turn" decision into one
    place so the answer is computed once, shared, and tracked on the turn.
    Any chat turn can be spoken: the UI selects a bot voice, a global voice,
    or a per-user override independently of the chat request. ``voice_optimized``
    controls prompt/style and visual emote filtering, not audio hygiene.

    Args:
        bot: The resolved Bot for this turn (may be None).
        request: The chat request, for future per-request overrides.

    Returns:
        True if this turn's output should be scrubbed for TTS.
    """
    return True


class StreamingTTSScrubber:
    """Block-boundary streaming scrubber for TTS.

    Markdown can't be scrubbed a *token* at a time — a link, fence or table
    spans token boundaries. But it CAN be scrubbed a *block* at a time: once a
    blank-line boundary arrives (and we are not inside an open ``` code fence),
    everything before it is a complete block that ``scrub_for_tts`` can clean
    safely. This lets voice-optimized TTS keep streaming at paragraph
    granularity instead of waiting for the entire turn.

    Usage:
        sc = StreamingTTSScrubber()
        for raw_chunk in stream:
            out = sc.feed(raw_chunk)   # scrubbed completed block(s), or ""
            if out:
                speak(out)
        tail = sc.flush()              # scrub + emit the final partial block
        if tail:
            speak(tail)
    """

    def __init__(self) -> None:
        self.buf = ""
        self.offset = 0
        self._parts: list[str] = []

    @property
    def text(self) -> str:
        """Exact emitted speech, including separators, for terminal recovery."""
        return "".join(self._parts)

    def _record(self, text: str) -> str:
        if text:
            self._parts.append(text)
            self.offset += len(text)
        return text

    def feed(self, chunk: str) -> str:
        """Accumulate raw text; return scrubbed text for any COMPLETED blocks."""
        if not chunk:
            return ""
        self.buf += chunk
        out: list[str] = []
        while True:
            # Find the earliest blank-line boundary whose preceding text has an
            # EVEN number of ``` fences (i.e. we are NOT inside an open fence).
            search = 0
            idx = -1
            while True:
                cand = self.buf.find("\n\n", search)
                if cand == -1:
                    break
                if self.buf[:cand].count("```") % 2 == 0:
                    idx = cand
                    break
                search = cand + 2
            if idx == -1:
                break
            block = self.buf[:idx]
            self.buf = self.buf[idx + 2:]
            scrubbed = scrub_for_tts(block)
            if scrubbed:
                out.append(scrubbed)
        # Trailing newline separates consecutive blocks fed to the TTS engine
        # across successive tts_delta events so words don't run together.
        return self._record(("\n".join(out) + "\n") if out else "")

    def flush(self) -> str:
        """Scrub and return the final buffered (partial) block at turn end."""
        scrubbed = scrub_for_tts(self.buf)
        self.buf = ""
        return self._record(scrubbed)

    def end_block(self) -> str:
        """Close the pending block at a mid-turn text-block end (TASK-2613).

        A tool call ends the model's text block, so its text is complete and
        can be spoken now instead of waiting for the next block's separator.
        Held while inside an open ``` fence; separated like ``feed`` output.
        """
        if not self.buf or self.buf.count("```") % 2:
            return ""
        scrubbed = scrub_for_tts(self.buf)
        self.buf = ""
        return self._record(scrubbed + "\n" if scrubbed else "")


class StreamingEmoteFilter:
    """Incremental ``strip_emotes`` for streaming text.

    The concatenation of every ``process`` result plus ``flush`` equals
    ``strip_emotes`` of the full text, for any chunking. Text is released once
    no emote rule can still change it; a possible emote (an unclosed ``*``,
    a ``::`` opener, a line starting with ``(``) and trailing whitespace are
    held until they resolve or the stream ends.

    Usage:
        filter = StreamingEmoteFilter()
        for chunk in stream:
            filtered = filter.process(chunk)
            if filtered:
                yield filtered
        # Flush any remaining buffered content
        final = filter.flush()
        if final:
            yield final
    """

    def __init__(self):
        self._raw = ""
        self._emitted = ""
        self._stalled = False

    def _release(self, clean: str) -> str:
        if not clean.startswith(self._emitted):
            # Unreachable by construction; never emit text a later chunk
            # would contradict.
            logger.warning("StreamingEmoteFilter: settled text diverged; holding until flush")
            self._stalled = True
            return ""
        delta = clean[len(self._emitted):]
        self._emitted = clean
        return delta

    def process(self, chunk: str) -> str:
        """Add a chunk; return newly settled clean text (possibly empty)."""
        if not chunk:
            return ""
        self._raw += chunk
        if self._stalled:
            return ""
        return self._release(_strip_emotes_settled(self._raw))

    def flush(self) -> str:
        """End of stream: return the rest of ``strip_emotes(full_text)``."""
        clean = strip_emotes(self._raw)
        self._stalled = False
        return self._release(clean)
