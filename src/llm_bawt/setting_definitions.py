"""Typed setting definitions with bot-type applicability (TASK-491/492).

The counterpart to ``DEFAULT_PROMPT_DEFINITIONS`` (bodies) for scalar/flag
settings. Every tunable declares its type, default, which bot_type(s) it applies
to, and where it is stored. This metadata is the single source of truth the
effective-config inspector and the frontend consume so they can render/mark only
the settings that actually apply to a given bot — killing the class of confusion
where an inert flag (e.g. ``include_summaries`` on an agent bot) is shown as if
it did something.

This registry does NOT itself resolve values — resolution stays in
``ConfigResolver``. It only declares the shape and applicability.
"""

from dataclasses import dataclass, field


# Storage backends a setting can live in.
STORAGE_RUNTIME_SETTING = "runtime_setting"        # runtime_settings table (scoped)
STORAGE_BOT_COLUMN = "bot_profiles_column"          # a first-class bot_profiles column
STORAGE_AGENT_BACKEND_CONFIG = "agent_backend_config"  # legacy JSON blob (being retired)
STORAGE_REQUEST_FLAG = "request_flag"               # per-turn request-only flag

BOT_TYPES_ALL = ("chat", "agent")

# Tier-1 offline-summarization job parameters (TASK-602/610). ONE global-only
# dict; never bot-aware, never model-catalog policy. These are the canonical
# CODE defaults (fresh install / no DB row). TASK-614 later inserts the
# operator-intent dict from the migrated DB rows (which honors min=4, the value
# a key-mismatch bug had been silently ignoring); the resolver merges any stored
# dict OVER these per-key so a partial row can't drop a field.
SUMMARIZATION_JOB_DEFAULTS = {
    "session_gap_seconds": 3600,        # gap that splits history into sessions
    "min_messages_per_session": 2,      # min msgs for a session to be summarizable
    "protected_recent_turns": 3,        # recent turns the JOB never folds into a summary
    "trigger_tokens": 12000,            # recent raw kept verbatim before older folds
    "model": None,                      # None = inherit maintenance_model
}


@dataclass(frozen=True)
class SettingDefinition:
    """One typed setting."""

    key: str
    type: str  # "bool" | "int" | "float" | "str" | "string_list"
    default: object
    applies_to: tuple[str, ...]  # bot types this setting affects
    storage: str
    label: str = ""
    help: str = ""
    # If this setting supersedes/absorbs a legacy key, name it here so the
    # migration and compat shims know the provenance.
    legacy_keys: tuple[str, ...] = field(default_factory=tuple)
    # Optional frontend render hint. Blank = render by ``type``; "model" tells
    # the UI to render a model-alias picker (dropdown of /v1/models) instead of
    # a free-text box. Surfaced verbatim in /v1/config/schema (TASK-522).
    ui_widget: str = ""


# Canonical typed settings. Keyed by setting key.
SETTING_DEFINITIONS: dict[str, SettingDefinition] = {
    # --- session-memory continuity (TASK-492: unifies the two flags) ---------
    "session_memory_continuity": SettingDefinition(
        key="session_memory_continuity",
        type="bool",
        default=True,
        applies_to=BOT_TYPES_ALL,
        storage=STORAGE_RUNTIME_SETTING,
        label="Carry prior conversation into context",
        help=(
            "Coarse master gate — a MIRROR of history_scope (TASK-518): true iff "
            "history_scope != 'none' (i.e. at least one of inline history / "
            "summaries is on). Kept as its own key so legacy readers (bridge seed "
            "gate, migrations) need no change. UI writes it derived; history_scope "
            "is the real source of truth for what gets carried."
        ),
        legacy_keys=("include_summaries", "seed_summary_on_new_session"),
    ),
    # --- history scope: the two independent carry-bits (TASK-493/518) ---------
    # NOTE: modelled as a constrained str until the registry grows a real "enum"
    # type. It encodes TWO INDEPENDENT bits, substring-tested: include_history =
    # "inline" in scope, include_summaries = "summaries" in scope. The four
    # canonical values are the full cross-product; there is no coupling between
    # the two axes (see utils.history.scope_flags).
    "history_scope": SettingDefinition(
        key="history_scope",
        type="str",
        default="inline+summaries",
        applies_to=BOT_TYPES_ALL,
        storage=STORAGE_RUNTIME_SETTING,
        label="History scope",
        help=(
            "What prior context is carried, as two independent bits: "
            "'inline+summaries' (recent messages + rolling summaries), "
            "'inline' (recent messages only), "
            "'summaries' (dense summary-only, no raw messages), or "
            "'none' (carry nothing). Absorbs the legacy include_summaries flag."
        ),
        legacy_keys=("include_summaries",),
    ),
    # --- Tier-3 context-generation policy (TASK-602/611) ----------------------
    # Bot-aware sizing of the two independent history_scope buckets. Global +
    # per-bot override (lean agents vs. aware chat bots). These decide HOW BIG
    # each bucket is; history_scope decides WHICH buckets. Distinct from Tier-1
    # job params (summarization_job, global-only) and Tier-2 physical limits
    # (resolve_context_budget, model-aware). The allocation ladder that consumes
    # history_tokens is a separate task; this registers + resolves the settings.
    "history_tokens": SettingDefinition(
        key="history_tokens",
        type="int",
        default=12000,
        applies_to=BOT_TYPES_ALL,
        storage=STORAGE_RUNTIME_SETTING,
        label="Raw history token budget",
        help=(
            "Tokens of recent raw conversation to carry (the inline-history "
            "bucket). Bounded on purpose — NEVER 0=fill-the-window (the footgun "
            "TASK-602 exists to kill). A lean agent can set this small (raw-only "
            "recent window, summaries off via history_scope) while a chat bot "
            "carries more. Absorbs the positive-valued legacy max_context_tokens "
            "bot overrides; the old 0=auto total-budget meaning is gone (Tier-2 "
            "resolve_context_budget owns the total budget now)."
        ),
        legacy_keys=("max_context_tokens",),
    ),
    "history_max_age_hours": SettingDefinition(
        key="history_max_age_hours",
        type="int",
        default=0,
        applies_to=BOT_TYPES_ALL,
        storage=STORAGE_RUNTIME_SETTING,
        label="Raw history max age (hours)",
        help=(
            "Age bound on the raw inline-history bucket (TASK-647): messages "
            "older than this many hours are excluded from context regardless of "
            "the history_tokens budget. 0 = no age limit (default). Bounds raw "
            "history ONLY — summaries are unaffected (they carry their own "
            "temporal framing and are capped by summary_count). A stale-enough "
            "conversation therefore seeds summary-first instead of replaying "
            "day-old raw messages."
        ),
    ),
    "summary_count": SettingDefinition(
        key="summary_count",
        type="int",
        default=5,
        applies_to=BOT_TYPES_ALL,
        storage=STORAGE_RUNTIME_SETTING,
        label="Summaries in context",
        help=(
            "Max rolling summaries injected into the prompt. 0 = carry no "
            "summaries (a raw-only bot, same effect as history_scope without "
            "'summaries'). Absorbs the legacy summarization_max_in_context."
        ),
        legacy_keys=("summarization_max_in_context",),
    ),
    "seed_message_max_chars": SettingDefinition(
        key="seed_message_max_chars",
        type="int",
        default=3000,
        applies_to=BOT_TYPES_ALL,
        storage=STORAGE_RUNTIME_SETTING,
        label="Seed per-message char cap",
        help=(
            "Max characters any single message may occupy in a fresh-session "
            "continuity seed (TASK-785). Oversized messages (typically inter-bot "
            "task-report deliveries, 12-16k chars each) are truncated with a "
            "marker pointing at full chat history instead of being seeded "
            "verbatim. Applies to the SEED delivery only — inline chat context "
            "and rolling summaries are untouched. 0 = no cap (pre-785 behavior)."
        ),
    ),
    "compact_context": SettingDefinition(
        key="compact_context",
        type="bool",
        default=True,
        applies_to=BOT_TYPES_ALL,
        storage=STORAGE_RUNTIME_SETTING,
        label="Compact summary form",
        help=(
            "Use the shorter/compressed summary representation in the prompt to "
            "save tokens. Absorbs the legacy summarization_compact_context."
        ),
        legacy_keys=("summarization_compact_context",),
    ),
    # --- promoted from agent_backend_config JSON blob (TASK-491) --------------
    # (TASK-615) The standalone seed_summary_on_new_session setting is retired —
    # session_memory_continuity (above) is the canonical seed gate, and it
    # already carries seed_summary_on_new_session as a legacy_key for migration.
    "agent_global_prompt_enabled": SettingDefinition(
        key="agent_global_prompt_enabled",
        type="bool",
        default=False,
        applies_to=("agent",),
        storage=STORAGE_RUNTIME_SETTING,
        label="Inject shared agent global prompt",
        help=(
            "Agent bots only: inject the shared 'agents.global_prompt' block into "
            "the cacheable system-prompt prefix. Steers planning to the BawtHub "
            "task system (observable) instead of the harness plan mode. Resolves "
            "global-then-bot, so a global=true baseline applies to every agent bot "
            "unless a bot overrides it. Chat bots never see it (manifest-gated)."
        ),
    ),
    "claude_code_disallowed_tools": SettingDefinition(
        key="claude_code_disallowed_tools",
        type="string_list",
        default=[
            "EnterPlanMode",
            "ExitPlanMode",
            "EnterWorktree",
            "ExitWorktree",
        ],
        applies_to=("agent",),
        storage=STORAGE_RUNTIME_SETTING,
        label="Claude Code disabled SDK tools",
        help=(
            "Global Claude Code harness policy: tool names withheld from every "
            "Claude Agent SDK turn. Changes are delivered with the next command; "
            "proxy turns additionally disable WebSearch and WebFetch."
        ),
    ),
    "agent_context_warning_percent": SettingDefinition(
        key="agent_context_warning_percent",
        type="int",
        default=75,
        applies_to=("agent",),
        storage=STORAGE_RUNTIME_SETTING,
        label="Agent context warning (%)",
        help="Resident prompt percentage that marks context health as warning.",
    ),
    "agent_context_critical_percent": SettingDefinition(
        key="agent_context_critical_percent",
        type="int",
        default=90,
        applies_to=("agent",),
        storage=STORAGE_RUNTIME_SETTING,
        label="Agent context critical (%)",
        help="Resident prompt percentage that triggers preventive maintenance before autonomous delivery.",
    ),
    "agent_context_critical_policy": SettingDefinition(
        key="agent_context_critical_policy",
        type="str",
        default="reset_retain_history",
        applies_to=("agent",),
        storage=STORAGE_RUNTIME_SETTING,
        label="Agent critical-context action",
        help=(
            "Preventive action for a critical agent context before durable delivery. "
            "Capability-gated per backend; unsupported retained reset falls back to "
            "reset_without_history only when that backend supports it."
        ),
    ),
    "bot_send_wait_seconds": SettingDefinition(
        key="bot_send_wait_seconds",
        type="int",
        default=300,
        applies_to=(),
        storage=STORAGE_RUNTIME_SETTING,
        label="Synchronous bot reply wait (s)",
        help=(
            "Global maximum wait for bots_send_message(wait_for_reply=true). "
            "Claude MCP calls receive 30 seconds of extra client-side headroom."
        ),
    ),
    "timeout_seconds": SettingDefinition(
        key="timeout_seconds",
        type="int",
        default=None,
        applies_to=("agent",),
        storage=STORAGE_RUNTIME_SETTING,
        label="Agent turn timeout (s)",
        help="Agent bots only: max seconds for a single agent turn before timeout.",
        legacy_keys=("timeout_seconds",),
    ),
    "session_model": SettingDefinition(
        key="session_model",
        type="str",
        default=None,
        applies_to=("agent",),
        storage=STORAGE_RUNTIME_SETTING,
        label="Session model (raw SDK id)",
        help="Agent bots only: the raw backend/SDK model id the session runs.",
        legacy_keys=("session_model",),
    ),
    # --- identity: keep in the blob/column, but typed so the UI knows it ------
    "session_key": SettingDefinition(
        key="session_key",
        type="str",
        default=None,
        applies_to=("agent",),
        storage=STORAGE_AGENT_BACKEND_CONFIG,
        label="Session key",
        help="Agent bots only: stable session identity. Not a tunable — do not edit casually.",
        legacy_keys=("session_key",),
    ),
    # --- chat-only legacy flag (kept for compat; superseded by continuity) ----
    "include_summaries": SettingDefinition(
        key="include_summaries",
        type="bool",
        default=True,
        applies_to=("chat",),
        storage=STORAGE_REQUEST_FLAG,
        label="Include summaries",
        help="Chat bots only: include summary rows when assembling history. "
             "Superseded by session_memory_continuity.",
        legacy_keys=("include_summaries",),
    ),
    # --- background-job models (TASK-522): global-only, data-driven ----------
    # System-wide model selection for the 3 LLM-using background jobs
    # (summarization, extraction, profile maintenance). Resolved at GLOBAL scope
    # only — no per-bot override — via resolve_job_model(). Cascade at every call
    # site is JOB-SPECIFIC FIRST, then the shared maintenance_model fallback:
    #     <job>_model -> maintenance_model
    # so each job's own picker actually controls that job, and maintenance_model
    # is the single "default for all jobs" knob. The terminal default lives on
    # maintenance_model (dolphin-qwen-3b), so an all-unset instance still runs.
    # These retire the env/config vars SUMMARIZATION_MODEL / EXTRACTION_MODEL /
    # MAINTENANCE_MODEL / PROFILE_MAINTENANCE_MODEL (option (a): keep the shared
    # maintenance knob).
    "maintenance_model": SettingDefinition(
        key="maintenance_model",
        type="str",
        default="dolphin-qwen-3b",
        applies_to=BOT_TYPES_ALL,
        storage=STORAGE_RUNTIME_SETTING,
        label="Default job model (all jobs)",
        help="Default model for every background job. Each job (summarization, "
             "extraction, profile maintenance) inherits this unless it has its "
             "own model set below. This is the last resort, so it always has a "
             "value — an all-unset instance runs on it.",
        legacy_keys=("MAINTENANCE_MODEL",),
        ui_widget="model",
    ),
    # summarization_model RETIRED (TASK-610): absorbed into summarization_job.model
    # (global Tier-1 dict). All consumers rerouted; had 0 DB rows. The generic
    # resolve_job_model() helper stays for extraction/maintenance/profile models.
    "extraction_model": SettingDefinition(
        key="extraction_model",
        type="str",
        default=None,
        applies_to=BOT_TYPES_ALL,
        storage=STORAGE_RUNTIME_SETTING,
        label="Memory extraction model",
        help="Model the memory-extraction job runs. Unset = inherit the default "
             "job model (maintenance_model).",
        legacy_keys=("EXTRACTION_MODEL",),
        ui_widget="model",
    ),
    "profile_maintenance_model": SettingDefinition(
        key="profile_maintenance_model",
        type="str",
        default=None,
        applies_to=BOT_TYPES_ALL,
        storage=STORAGE_RUNTIME_SETTING,
        label="Profile maintenance model",
        help="Model the profile-maintenance job runs. Unset = inherit the "
             "default job model (maintenance_model).",
        legacy_keys=("PROFILE_MAINTENANCE_MODEL",),
        ui_widget="model",
    ),
    # --- subagent/small-fast model (TASK-546): proxy-compatible override -------
    # When the claude-code bridge routes through the Anthropic-compat proxy
    # (non-Anthropic providers like openai_chatgpt/zai/xai), Claude Code's
    # internal background Haiku calls and subagent model resolution send bare
    # Anthropic model IDs that the proxy rejects (400). This setting overrides
    # BOTH ANTHROPIC_SMALL_FAST_MODEL and CLAUDE_CODE_SUBAGENT_MODEL env vars
    # for proxy-routed turns, so background tasks and subagents use a
    # provider-qualified model the proxy accepts. Unset = inherit the bot's
    # main model (default_model) on proxy turns; ignored on Anthropic-direct
    # turns where the built-in Haiku path works natively.
    "subagent_model": SettingDefinition(
        key="subagent_model",
        type="str",
        default=None,
        applies_to=("agent",),
        storage=STORAGE_AGENT_BACKEND_CONFIG,
        label="Subagent / background model",
        help=(
            "Agent bots only: model used for Claude Code's internal subagents "
            "and background tasks (title generation, tool-use summaries, etc.) "
            "when running through the proxy. Unset = inherit the bot's main "
            "model. Must be a provider-qualified model the proxy accepts "
            "(e.g. openai_chatgpt/gpt-5.4-mini)."
        ),
        legacy_keys=("subagent_model",),
        ui_widget="model",
    ),
    # --- context window / output budget (TASK-602: retire env fallbacks) -------
    # These are resolved at GLOBAL scope via resolve_global_runtime_setting()
    # (global runtime_settings row -> this declared default), NEVER from a Config
    # BaseSettings env field. Same pattern resolve_job_model() used to retire the
    # SUMMARIZATION_MODEL / EXTRACTION_MODEL env vars.
    "model_context_window_default": SettingDefinition(
        key="model_context_window_default",
        type="int",
        default=128000,
        applies_to=BOT_TYPES_ALL,
        storage=STORAGE_RUNTIME_SETTING,
        label="Default model context window",
        help=(
            "Fallback context window (tokens) used ONLY when a model has no "
            "per-model window in the catalog (models.default_context_window / "
            "model_endpoints.context_window_override). The catalog is the source "
            "of truth for per-model windows; this global default just catches "
            "NULLs. Not a per-model or per-bot value — set the window on the "
            "model in the catalog instead."
        ),
    ),
    "agent_compact_threshold": SettingDefinition(
        key="agent_compact_threshold",
        type="int",
        default=200000,
        applies_to=(),  # GLOBAL-ONLY: per-model overrides live in the catalog
                        # (models.compact_threshold), not on bots.
        storage=STORAGE_RUNTIME_SETTING,
        label="Agent compact threshold",
        help=(
            "Claude Code auto-compact window (tokens) for agent bots, used when "
            "the model has no compact threshold in the catalog. Sent as "
            "CLAUDE_CODE_AUTO_COMPACT_WINDOW; the CLI floors it at 100k, caps it "
            "at the model's context window, and compacts a little before it "
            "(~33k early on Claude models, for the output reserve and buffer)."
        ),
    ),
    "max_output_tokens": SettingDefinition(
        key="max_output_tokens",
        type="int",
        default=4096,
        applies_to=BOT_TYPES_ALL,
        storage=STORAGE_RUNTIME_SETTING,
        label="Max output tokens",
        help=(
            "Model output CAPABILITY — the max tokens the model may generate per "
            "reply, sent as the API request's output cap. Catalog per-model "
            "max_tokens overrides this global default. NOTE (TASK-609): this is "
            "no longer what the prompt budget subtracts — that is the separate "
            "'request_output_reserve' policy knob. Retires LLM_BAWT_MAX_OUTPUT_TOKENS."
        ),
        legacy_keys=("max_output_tokens",),
    ),
    "request_output_reserve": SettingDefinition(
        key="request_output_reserve",
        type="int",
        default=4096,
        applies_to=(),  # GLOBAL-ONLY (TASK-602 Tier 2): rendered in NO per-bot UI;
                        # bot overrides impossible. Resolved via
                        # resolve_global_runtime_setting(), which ignores bot rows.
        storage=STORAGE_RUNTIME_SETTING,
        label="Request output reserve",
        help=(
            "Tokens held back from the context window for the model's reply. "
            "prompt_budget = context_window - min(reserve, model max-output capability). "
            "Global-only infrastructure policy (Tier 2 is not bot-aware); distinct "
            "from max_output_tokens, which is the model's per-request output cap. "
            "Default 4096 preserves prior behavior (old max_output_tokens double-duty)."
        ),
    ),
    "summarization_job": SettingDefinition(
        key="summarization_job",
        type="json",
        default=SUMMARIZATION_JOB_DEFAULTS,
        applies_to=(),  # GLOBAL-ONLY (TASK-602 Tier 1): no per-bot override.
                        # Resolved via resolve_global_runtime_setting(); bot rows ignored.
        storage=STORAGE_RUNTIME_SETTING,
        label="Summarization job parameters",
        help=(
            "Tier-1 offline-summarization job settings as ONE global dict: "
            "session_gap_seconds, min_messages_per_session, protected_recent_turns, "
            "trigger_tokens, and model (None = inherit maintenance_model). "
            "Global-only job scheduling policy — NOT bot-aware and NOT model-catalog "
            "physical limits (those are Tier-2 resolve_context_budget). Absorbs the "
            "legacy summarization_session_gap_seconds / summarization_min_messages / "
            "memory_protected_recent_turns / summarization_trigger_tokens / "
            "summarization_model keys."
        ),
        legacy_keys=(
            "summarization_session_gap_seconds",
            "summarization_min_messages_per_session",
            "summarization_min_messages",
            "memory_protected_recent_turns",
            "summarization_trigger_tokens",
            "summarization_model",
        ),
    ),
}


def definitions_for_bot_type(bot_type: str) -> list[SettingDefinition]:
    """Settings that apply to a given bot_type."""
    return [d for d in SETTING_DEFINITIONS.values() if bot_type in d.applies_to]


def applies_to_bot_type(key: str, bot_type: str) -> bool:
    """Whether a setting applies to a bot_type. Unknown keys default True (be
    permissive — an undeclared setting is not something we should hide)."""
    d = SETTING_DEFINITIONS.get(key)
    return True if d is None else (bot_type in d.applies_to)


def setting_default(key: str, fallback: object = None) -> object:
    """The registry-declared default for a setting key (TASK-611).

    The single source of truth for a setting's code-default, so consumers stop
    reading retired ``config.*`` env attributes as fallbacks. Returns ``fallback``
    only for an unregistered key.
    """
    d = SETTING_DEFINITIONS.get(key)
    return d.default if d is not None else fallback
