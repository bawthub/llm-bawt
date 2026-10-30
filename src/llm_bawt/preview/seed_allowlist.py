"""Seed A: the deny-by-default allowlist for the preview database (TASK-1802).

Every store-owned table is classified here, and every column of a copied table
has an explicit treatment. Anything not named is an error, never a default:

* a table the full-store bootstrap creates but this module doesn't classify
  fails ``tests/test_preview_seed_allowlist_pg.py``;
* a column added to a copied table fails the same test until it is classified;
* a production table or runtime-settings key the exporter doesn't recognise
  stops the export.

Seed A carries configuration only: the model catalog, prompt registry, tag
catalog, non-private bot profiles, tuning settings and approval policies. No
credentials, connection records, people, history, memories or tokens. Rows of
private-tagged bots are excluded from every copied table.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from llm_bawt.bot_tags.vocabulary import BehaviorTag

SEED_ACTOR = "preview-seed"
"""Replaces every who-did-this column (created_by, assigned_by, ...)."""


@dataclass(frozen=True)
class Keep:
    """Copy the value unchanged."""


@dataclass(frozen=True)
class Null:
    """Never copy the value; load NULL."""


@dataclass(frozen=True)
class Actor:
    """Identity column: replace with ``SEED_ACTOR``."""


@dataclass(frozen=True)
class JsonKeys:
    """JSON object: keep only these top-level keys."""

    keys: tuple[str, ...]


Treatment = Keep | Null | Actor | JsonKeys
KEEP, NULL, ACTOR = Keep(), Null(), Actor()

# Same rule as bot_tags.facade: private = ``private`` tag OR
# include_in_global_search = false.
PRIVATE_BOTS_SQL = (
    "SELECT bot_slug FROM public.bot_profile_tags "
    f"WHERE tag_slug = '{BehaviorTag.PRIVATE.value}' "
    "UNION SELECT slug FROM public.bot_profiles WHERE NOT include_in_global_search"
)


@dataclass(frozen=True)
class Copy:
    """Copy rows of this table, column by column, minus private-bot rows."""

    columns: dict[str, Treatment]
    bot_column: str | None = None
    """Column naming a bot slug; rows of private-tagged bots are excluded."""
    where: str | None = None
    """Extra row filter (SQL over the table's own columns)."""
    order_by: str = "1"


# Global runtime settings that are pure tuning. Everything else is denied,
# and a key in neither set stops the export until it is classified.
GLOBAL_SETTING_KEYS: frozenset[str] = frozenset({
    "agent_compact_threshold", "agent_global_prompt_enabled", "bot_send_wait_seconds",
    "default_voice", "history_bridge_messages", "history_duration_seconds",
    "history_reload_ttl_seconds", "history_summarization_interval_minutes",
    "inactive_usage_providers", "maintenance_model", "maintenance_model_fallback",
    "maintenance_reasoning_effort", "max_output_tokens", "memory_dedup_similarity",
    "memory_min_relevance", "memory_n_results", "profile_maintenance_interval_minutes",
    "prompt_cache_ttl", "question_answer_wait_seconds", "summarization_job",
    "summary_count", "temperature", "top_p",
})
# Denied on purpose: credentials, connection/account records, a person's
# devices and navigation, deployment URLs, prod-only enforcement switches.
DENIED_GLOBAL_SETTING_KEYS: frozenset[str] = frozenset({
    "agent.dispatch_url_override", "bridge_gate_enforced:broker", "google_home_probe",
    "nav_favorites", "notify",
})
DENIED_GLOBAL_SETTING_PREFIXES: tuple[str, ...] = ("provider_connection:", "integration_connection:")

BOT_SETTING_KEYS: frozenset[str] = frozenset({
    "default_voice", "history_bridge_messages", "history_duration_seconds", "history_scope",
    "history_tokens", "memory_dedup_similarity", "memory_min_relevance", "memory_n_results",
    "prompt_cache_ttl", "session_memory_continuity", "session_model", "summary_count",
    "timeout_seconds", "ui_color", "working_memory_enabled",
})


def _in(keys: frozenset[str]) -> str:
    return ", ".join(f"'{key}'" for key in sorted(keys))


# Load order is FK order: parents before children.
COPY: dict[str, Copy] = {
    "models": Copy({
        "id": KEEP, "key": KEEP, "vendor": KEEP, "display_name": KEEP, "description": KEEP,
        "default_context_window": KEEP, "default_tool_support": KEEP, "compact_threshold": KEEP,
        "created_at": KEEP, "updated_at": KEEP,
    }),
    "access_paths": Copy({
        "id": KEEP, "key": KEEP, "vendor": KEEP, "protocol": KEEP, "base_url": KEEP,
        "auth_mechanism": KEEP, "engine_kind": KEEP, "system_prompt_overrides": KEEP,
        "created_at": KEEP, "updated_at": KEEP,
    }),
    "model_endpoints": Copy({
        "id": KEEP, "model_id": KEEP, "access_path_id": KEEP, "upstream_model_id": KEEP,
        "serving_config": JsonKeys(("responses_transport", "reasoning_effort", "compat_extra")),
        "context_window_override": KEEP, "tool_support_override": KEEP, "pricing": KEEP,
        "created_at": KEEP, "updated_at": KEEP,
    }),
    "prompt_templates": Copy({
        "id": KEEP, "key": KEEP, "scope_type": KEEP, "scope_id": KEEP, "title": KEEP,
        "category": KEEP, "format": KEEP, "body": KEEP, "required_vars_json": KEEP,
        "metadata_json": KEEP, "updated_by": ACTOR, "created_at": KEEP, "updated_at": KEEP,
    }, where="scope_type = 'global'"),
    "prompt_template_versions": Copy({
        "id": KEEP, "key": KEEP, "scope_type": KEEP, "scope_id": KEEP, "version": KEEP,
        "body": KEEP, "change_note": KEEP, "created_by": ACTOR, "created_at": KEEP,
    }, where=f"scope_type = 'global' OR (scope_type = 'bot' AND scope_id NOT IN ({PRIVATE_BOTS_SQL}))"),
    "bot_tag_catalog": Copy({
        "slug": KEEP, "name": KEEP, "description": KEEP, "kind": KEEP, "prompt_key": KEEP,
        "capabilities": KEEP, "system": KEEP, "created_at": KEEP, "updated_at": KEEP,
    }),
    "bot_profiles": Copy({
        "id": KEEP, "slug": KEEP, "name": KEEP, "description": KEEP, "system_prompt": KEEP,
        "requires_memory": KEEP, "voice_optimized": KEEP, "uses_tools": KEEP,
        "uses_search": KEEP, "uses_home_assistant": KEEP, "default_model": KEEP,
        "nextcloud_config": NULL, "color": KEEP, "agent_backend": KEEP,
        "agent_backend_config": JsonKeys(("effort", "session_model", "timeout_seconds", "codex_transport")),
        "tts_mode": KEEP, "include_summaries": KEEP, "default_voice": KEEP, "avatar": KEEP,
        "bot_type": KEEP, "include_in_global_search": KEEP, "avatar_render": KEEP,
        "prompt_override_id": KEEP, "harness": KEEP, "endpoint_id": KEEP,
        "last_session_id": NULL, "created_at": KEEP, "updated_at": KEEP,
    }, bot_column="slug"),
    "bot_profile_tags": Copy({
        "bot_slug": KEEP, "tag_slug": KEEP, "assigned_at": KEEP, "assigned_by": ACTOR,
    }, bot_column="bot_slug", order_by="1, 2"),
    "runtime_settings": Copy({
        "id": KEEP, "scope_type": KEEP, "scope_id": KEEP, "key": KEEP, "value_json": KEEP,
        "updated_at": KEEP,
    }, where=(
        f"(scope_type = 'global' AND key IN ({_in(GLOBAL_SETTING_KEYS)})) OR "
        f"(scope_type = 'bot' AND key IN ({_in(BOT_SETTING_KEYS)}) "
        f"AND scope_id NOT IN ({PRIVATE_BOTS_SQL}))"
    )),
    "tool_approval_policies": Copy({
        "id": KEEP, "enabled": KEEP, "backend_scope": KEEP, "tool_name": KEEP,
        "matcher_type": KEEP, "pattern": KEEP, "field": KEEP, "action": KEEP, "severity": KEEP,
        "category": KEEP, "approval_prompt": KEEP, "order_index": KEEP, "version": KEEP,
        "created_by": ACTOR, "updated_by": ACTOR, "created_at": KEEP, "updated_at": KEEP,
    }),
}

# Created empty in the preview: history, memory, people, credentials, tokens,
# audits, queues, jobs and every other runtime record.
SCHEMA_ONLY: frozenset[str] = frozenset({
    "agent_events", "agent_runs", "agent_session_state", "api_balance_ledger", "api_balances",
    "approval_approver_fanout", "bawthub_auth_secrets", "bot_profile_changes",
    "bridge_run_accepts", "chat_pending_questions", "entity_profile_attributes",
    "entity_profiles", "forgotten_messages", "gate_escalations", "gate_members",
    "google_home_probe_captures", "google_home_probe_links", "google_home_probe_tokens",
    "gpu_handoff_plans", "gpu_handoff_resolved_video_jobs", "gpu_handoff_state",
    "gpu_handoff_video_jobs", "gpu_video_calibrations", "home_audio_jobs", "home_audio_lane",
    "inter_bot_deliveries", "job_runs", "media_assets", "media_generations", "memories",
    "message_author_backfill_audit", "message_route_eligibility", "message_routes", "messages",
    "new_session_event_outbox", "notify_log", "ops_jobs", "ops_operation_revisions",
    "ops_operations", "ops_release_events", "ops_release_runs", "ops_resume_notices",
    "prompt_occurrences",
    "prompt_schedules", "quiesce_run_sightings", "recall_evidence_ledgers", "scheduled_jobs",
    "sessions", "skill_assignments", "skill_managed_bots", "system_events", "system_gates",
    "talk_reply_subscriptions", "tool_approval_decisions", "tool_approval_policy_revisions",
    "tool_approval_requests", "tool_call_records", "tool_call_result_payloads",
    "turn_changed_files", "turn_diff_payloads", "turn_logs", "working_memory_changes",
    "working_memory_items", "working_memory_namespaces", "working_memory_task_purge_audit",
    "working_memory_task_suppressions",
})

PARTITION_MARKER = "_p_"
"""Per-bot partitions (``messages_p_<bot>``) are runtime data, never seeded."""


@dataclass
class Classification:
    unclassified_tables: list[str] = field(default_factory=list)
    column_drift: dict[str, tuple[list[str], list[str]]] = field(default_factory=dict)
    """table -> (columns with no treatment, treatments with no column)."""

    @property
    def ok(self) -> bool:
        return not self.unclassified_tables and not self.column_drift


def classify(tables: dict[str, list[str]]) -> Classification:
    """Check ``{table: [columns]}`` (non-partition tables) against the allowlist."""
    result = Classification()
    for table, columns in sorted(tables.items()):
        if table in COPY:
            declared = COPY[table].columns
            missing = sorted(set(columns) - set(declared))
            extra = sorted(set(declared) - set(columns))
            if missing or extra:
                result.column_drift[table] = (missing, extra)
        elif table not in SCHEMA_ONLY:
            result.unclassified_tables.append(table)
    assert not set(COPY) & SCHEMA_ONLY, "a table is both copied and schema-only"
    return result


def unclassified_global_keys(keys: list[str]) -> list[str]:
    """Global runtime-settings keys that are neither allowed nor denied."""
    return sorted(
        key for key in keys
        if key not in GLOBAL_SETTING_KEYS
        and key not in DENIED_GLOBAL_SETTING_KEYS
        and not key.startswith(DENIED_GLOBAL_SETTING_PREFIXES)
    )


def unclassified_bot_keys(keys: list[str]) -> list[str]:
    return sorted(set(keys) - BOT_SETTING_KEYS)


def _select_expr(column: str, treatment: Treatment) -> str:
    quoted = f'"{column}"'
    if isinstance(treatment, Keep):
        return quoted
    if isinstance(treatment, Null):
        return f"NULL AS {quoted}"
    if isinstance(treatment, Actor):
        return f"CASE WHEN {quoted} IS NULL THEN NULL ELSE '{SEED_ACTOR}' END AS {quoted}"
    keys = ", ".join(f"'{key}'" for key in treatment.keys)
    return (
        f"CASE WHEN {quoted} IS NULL THEN NULL "
        f"WHEN jsonb_typeof({quoted}::jsonb) <> 'object' THEN '{{}}'::jsonb ELSE COALESCE("
        f"(SELECT jsonb_object_agg(k, v) FROM jsonb_each({quoted}::jsonb) AS e(k, v) "
        f"WHERE k IN ({keys})), '{{}}'::jsonb) END AS {quoted}"
    )


def export_query(table: str) -> str:
    """The read-only SELECT that produces ``table``'s seed rows."""
    rule = COPY[table]
    columns = ", ".join(_select_expr(c, t) for c, t in rule.columns.items())
    filters = []
    if rule.bot_column:
        filters.append(f'"{rule.bot_column}" NOT IN ({PRIVATE_BOTS_SQL})')
    if rule.where:
        filters.append(f"({rule.where})")
    where = f" WHERE {' AND '.join(filters)}" if filters else ""
    return f"SELECT {columns} FROM public.{table}{where} ORDER BY {rule.order_by}"


def column_list(table: str) -> str:
    return ", ".join(f'"{column}"' for column in COPY[table].columns)
