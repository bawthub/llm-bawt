"""Explicit protected-data tool classifications; catalog tests enforce coverage."""

BOT_TARGET_TOOLS = frozenset({
    "memory_store", "memory_search", "memory_update", "memory_delete", "memory_clear",
    "memory_supersede", "memory_list_recent", "memory_list_high_importance",
    "memory_delete_by_source_messages", "memory_regenerate_embeddings", "memory_consolidate",
    "memory_update_meaning", "messages_add", "messages_get", "messages_get_by_id",
    "messages_get_for_summary", "messages_preview_recent", "messages_preview_since_minutes",
    "messages_preview_ignored", "messages_ignore_recent", "messages_ignore_since_minutes",
    "messages_ignore_by_id", "messages_restore_ignored", "messages_mark_recalled",
    "messages_clear", "messages_remove_last_partial", "facts_extract", "context_get_recent",
    "system_stats", "system_run_maintenance", "sessions_get", "sessions_close",
    "sessions_get_active", "sessions_get_or_create_active", "sessions_rotate",
    "self_tail", "self_recap", "self_system_prompt", "self_model", "agent_context_health",
})
AGGREGATE_TOOLS = frozenset({
    "memory_list_sources", "memory_search_all", "messages_search_all", "sessions_list",
})
SPECIAL_TOOLS = frozenset({"memory_search_source", "self_fwd", "profile"})
# Existing self-namespace capabilities/backend/protected gates remain authoritative.
SELF_NAMESPACE_TOOLS = frozenset({
    "working_memory_list", "working_memory_create", "working_memory_update", "context_recall",
})
# These are explicitly outside this history boundary (not inferred by a prefix).
NON_HISTORY_TOOLS = frozenset({
    "tasks_list", "tasks_search_semantic", "tasks_get", "tasks_reviews_list",
    "tasks_reviews_add", "tasks_associate_current", "tasks_update", "tasks_delete",
    "tasks_add_dependency", "tasks_remove_dependency", "tasks_promote", "tasks_regenerate",
    "tasks_create", "tasks_get_context", "steps_update", "steps_delete", "steps_add", "steps_set",
    "projects_list", "projects_get", "projects_create", "projects_update", "projects_get_context",
    "projects_delete", "activity_get", "initiative_state", "ops_list_operations", "ops_run",
    "ops_job_status", "ops_quiesce_ack", "media_lookup", "media_add", "media_queue", "media_history",
    "media_pipeline_status", "media_library_stats", "generate_image", "speech_generate",
    "home_audio_devices", "home_audio_enqueue", "home_audio_status", "home_audio_cancel",
    "x_search", "x_counts", "web_search", "bots_list_available",
})
PARTICIPANT_TOOLS = frozenset({"bots_delivery_get", "bots_deliveries_list"})
# Nick's 2026-10-23 ruling: do not add private-target delivery/reset mutation
# locks. Discovery filtering for availability/router candidates is a follow-up.
UNSCOPED_MUTATION_TOOLS = frozenset({
    "bots_send_message", "bots_route_message", "agent_context_reset", "agent_context_compact",
    "bots_delivery_cancel",
})
# TASK-1323: resolves BotAccess per call and narrows its own scope. Outside
# the guard on purpose: an excluded bot filter must not answer access_denied.
SCOPED_RETRIEVAL_TOOLS = frozenset({"messages_find"})
GUARDED_TOOLS = BOT_TARGET_TOOLS | AGGREGATE_TOOLS | SPECIAL_TOOLS | PARTICIPANT_TOOLS
CLASSIFIED_TOOLS = (GUARDED_TOOLS | SELF_NAMESPACE_TOOLS | NON_HISTORY_TOOLS
                    | UNSCOPED_MUTATION_TOOLS | SCOPED_RETRIEVAL_TOOLS)
