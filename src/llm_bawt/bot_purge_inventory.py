"""Versioned inventory of the existing standalone purge, not an execution plan.

Update the scope version when predicates or retention semantics change. Counts
are row counts (including overlapping retained subsets), never a summable total.
"""
from __future__ import annotations

from dataclasses import dataclass

SCOPE_VERSION = "bot-purge/4"


@dataclass(frozen=True)
class InventorySpec:
    key: str
    table: str
    columns: tuple[str, ...]
    predicate: str
    disposition: str
    reason: str
    dependencies: tuple[tuple[str, tuple[str, ...]], ...] = ()


def owned(table, column="bot_id", *, disposition="deleted", reason=None):
    return InventorySpec(table, table, (column,), f'{column}=:slug', disposition,
                         reason or "Selected by the existing standalone purge if its guards and transaction succeed.")


def scoped(table, *, disposition="deleted", reason=None):
    return InventorySpec(table, table, ("scope_type", "scope_id"),
                         "scope_type='bot' AND scope_id=:slug", disposition,
                         reason or "Only the exact bot scope is selected; user/global scopes are retained.")


_JOB_DEP = (("scheduled_jobs", ("id", "bot_id")),)
_JOB_PREDICATE = "job_id IN (SELECT id FROM scheduled_jobs WHERE bot_id=:slug)"

DELETED = (
    *(InventorySpec(t, t, ("bot_id",), "bot_id=:storage", "deleted",
                    "Storage-key selection, conditional on observed raw-identity attribution.")
      for t in ("messages", "memories", "forgotten_messages")),
    owned("sessions"), scoped("runtime_settings"),
    *(InventorySpec(t, t, ("entity_type", "entity_id"),
                    "CAST(entity_type AS TEXT) IN ('BOT','bot') AND entity_id=:slug", "deleted",
                    "Bot-typed entity rows only; same-ID USER/user rows are retained.")
      for t in ("entity_profile_attributes", "entity_profiles")),
    *(InventorySpec(t, t, ("job_id",), _JOB_PREDICATE, "deleted",
                    "Ownership follows the exact bot's scheduled job, not the child's bot_id or status.", _JOB_DEP)
      for t in ("prompt_occurrences", "prompt_schedules", "job_runs")),
    owned("scheduled_jobs"), scoped("prompt_template_versions"), scoped("prompt_templates"),
    InventorySpec("tool_call_result_payloads", "tool_call_result_payloads", ("tool_call_record_id",),
                  "tool_call_record_id IN (SELECT id FROM tool_call_records WHERE bot_id=:slug)", "deleted",
                  "Legacy payloads selected through owned tool records, not shared content-addressed blobs.",
                  (("tool_call_records", ("id", "bot_id")),)),
    owned("tool_call_records"), owned("llm_response_records"), owned("worker_lifecycle_records"), owned("turn_logs"),
    owned("session_resumption_receipts"),
)

_RETAINED_REASON = "Not deleted by the existing standalone purge; operational records can survive."
RETAINED = (
    InventorySpec("memory_ownership_activations", "memory_ownership_activations", ("bot_id",),
                  "bot_id=:storage", "retained",
                  "Ownership cutover/receipt survives purge and profile deletion; identity reuse must not disable isolation."),
    owned("bot_profiles", "slug", disposition="retained", reason="Standalone purge retains the definition; session FK pointers may be nulled."),
    owned("bot_profile_tags", "bot_slug", disposition="retained", reason="Assignments outlive profiles, including protection/reuse metadata."),
    owned("bot_profile_changes", "bot_slug", disposition="retained", reason="Historical profile-change audit is retained."),
    InventorySpec("shared_job_runs", "job_runs", ("job_id", "bot_id"),
                  "bot_id=:slug AND NOT EXISTS (SELECT 1 FROM scheduled_jobs j WHERE j.id=job_runs.job_id AND j.bot_id=:slug)",
                  "retained", "Shared/other-job provenance is retained, including RUNNING markers; not cancellation proof.", _JOB_DEP),
    InventorySpec("inter_bot_deliveries", "inter_bot_deliveries", ("sender_bot_id", "target_bot_id"),
                  "sender_bot_id=:slug OR target_bot_id=:slug", "retained",
                  "Cross-bot receipts/replay/provenance are retained; no transport cancellation is performed."),
    InventorySpec("nonterminal_deliveries", "inter_bot_deliveries", ("sender_bot_id", "target_bot_id", "status"),
                  "(sender_bot_id=:slug OR target_bot_id=:slug) AND (status IS NULL OR status NOT IN ('DELIVERED','FAILED','CANCELLED'))",
                  "retained", "Overlapping operational subset, not a drain or producer-fence check."),
    owned("new_session_event_outbox", disposition="retained", reason="Published AND pending session-event records are untouched."),
    InventorySpec("pending_session_events", "new_session_event_outbox", ("bot_id", "published_at"),
                  "bot_id=:slug AND published_at IS NULL", "retained",
                  "Overlapping operational subset; existing purge does not acknowledge or drain it."),
    *(owned(t, disposition="retained", reason=_RETAINED_REASON) for t in (
        "turn_changed_files", "tool_approval_requests", "chat_pending_questions",
        "working_memory_namespaces", "working_memory_items", "working_memory_changes",
        "working_memory_task_suppressions", "message_route_eligibility", "talk_reply_subscriptions",
        "recall_evidence_ledgers", "notify_log", "ops_resume_notices", "system_events",
    )),
    scoped("skill_assignments", disposition="retained", reason=_RETAINED_REASON),
    owned("message_routes", "selected_bot_id", disposition="retained", reason="Routing receipts are retained; other actor references are not exhaustively counted."),
    owned("ops_jobs", "caller_bot_id", disposition="retained", reason="Operational jobs/provenance are retained, not cancelled."),
    InventorySpec("profile_session_pointer_effects", "bot_profiles", ("last_session_id",),
                  "last_session_id IN (SELECT id FROM sessions WHERE bot_id=:slug)", "retained",
                  "Retained profiles (possibly another bot) can have last_session_id SET NULL by installed FKs.",
                  (("sessions", ("id", "bot_id")),)),
    InventorySpec("tool_turn_pointer_effects", "tool_call_records", ("bot_id", "turn_id"),
                  "(bot_id IS NULL OR bot_id!=:slug) AND turn_id IN (SELECT id FROM turn_logs WHERE bot_id=:slug)", "retained",
                  "Retained tool records can have turn_id SET NULL by installed FKs.",
                  (("turn_logs", ("id", "bot_id")),)),
)

# The combined delete removes the definition and every tag assignment (privacy
# tags included: the data they protected is deleted in the same transaction).
PROFILE_DELETED = {
    "bot_profiles": "The profile itself is deleted in the same transaction as its data.",
    "bot_profile_tags": "All tag assignments, including private/no-global-search, are deleted with the data they protected.",
}

# These stores are untouched, but this connection cannot establish exhaustive
# ownership/exclusive bytes or cross-store counts. Unknown never means zero.
UNKNOWN = {
    "bawthub_associations": "External database task/session/turn associations are untouched and not inspected.",
    "object_storage_and_media": "Shared/user-owned media, attachments and content-addressed blobs are untouched; exclusive ownership/physical bytes unknown.",
    "redis_and_bridge_state": "Redis, agent session/run keys, providers and bridge-local files are not inspected or drained.",
    "backups": "Backup copies and historical exports are untouched and not inspected.",
    "indirect_and_unmapped_references": "Indirect approval/gate/audit/working-memory references and future stores are not exhaustively inventoried; no complete erasure claim.",
    "writer_and_fk_readiness": "No writer fence, active-work drain, FK-catalog proof or future execution/atomicity guarantee is provided.",
}

# Raw identities, including profile-less owners, not partition bot_id values.
OWNER_SOURCES = (
    ("bot_profiles", ("slug",), "slug", "1=1"),
    ("bot_profile_tags", ("bot_slug",), "bot_slug", "1=1"),
    ("runtime_settings", ("scope_type", "scope_id"), "scope_id", "scope_type='bot'"),
    ("sessions", ("bot_id",), "bot_id", "1=1"),
    ("entity_profiles", ("entity_type", "entity_id"), "entity_id", "CAST(entity_type AS TEXT) IN ('BOT','bot')"),
)
