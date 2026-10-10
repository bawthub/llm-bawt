"""Model-facing selection guidance; workflows live in the bawthub-mcp skill.

Keep names and parameters stable: approval policies and saved calls address them.
Each description must retain side effects and call-critical constraints. Do not
copy argument types/defaults or tutorials here. See docs/MCP_TOOL_DESIGN.md.
"""

TOOL_SUMMARIES: dict[str, str] = {
    "memory_store": "Store a fact in bot_id's memory; importance is 0–1. Pass your own bot slug.",
    "memory_search": "Semantic search within bot_id's memories; relevance is 0–1. Pass your own bot slug.",
    "memory_list_sources": "List searchable memory namespaces and counts; does not authorize access to protected bots.",
    "memory_search_source": "Read-only semantic search of an allowed source bot's memories; relevance is 0–1.",
    "memory_search_all": "Semantic search across bots, or restrict with bot_id; since/until are Unix seconds. Respect protected-bot boundaries.",
    "memory_update": "Patch a memory in bot_id's namespace; omitted fields stay unchanged.",
    "memory_delete": "Delete one memory in bot_id's namespace.",
    "memory_clear": "Delete ALL distilled memories for bot_id. Destructive; requires explicit scope authorization.",
    "memory_supersede": "Mark old_memory_id superseded by new_memory_id, or use DELETED as the replacement marker.",
    "memory_list_recent": "Read recent memories for bot_id in backend-native format.",
    "memory_list_high_importance": "Read memories ranked by importance; min_importance is 0–1.",
    "memory_delete_by_source_messages": "Delete memories derived from the specified source message IDs in bot_id's namespace.",
    "memory_regenerate_embeddings": "Rebuild stored memory embeddings for bot_id in batches; writes state.",
    "memory_consolidate": "Merge similar memories for bot_id; dry_run previews without applying merges.",
    "memory_update_meaning": "Update a memory's intent, stakes, emotional charge, recurrence keywords, or tags.",
    "messages_search_all": "Full-text search across bots; bot_id restricts scope. since/until accept ISO dates or Unix seconds; sort_by: relevance|recent. Respect protected bots.",
    "messages_get": "Read history for bot_id; session_id scopes one thread. summaries_only selects summaries; exclude_summarized avoids duplicate raw context.",
    "messages_get_by_id": "Read a message by UUID or unique prefix (minimum 8 chars), optionally with before/after neighbors.",
    "messages_add": "Write a history message, not a bot turn. Use session_id or user_id to resolve the intended thread; role: user|assistant|system.",
    "messages_clear": "Delete ALL messages for bot_id. Destructive; requires explicit scope authorization.",
    "messages_preview_recent": "Preview bot_id's recent messages without changing history.",
    "messages_preview_since_minutes": "Preview bot_id's messages within the last minutes without changing history.",
    "messages_preview_ignored": "Read bot_id's ignored messages without restoring them.",
    "messages_ignore_recent": "Move the latest count messages out of active history into ignored storage; preview first.",
    "messages_ignore_since_minutes": "Ignore messages from the last minutes; preview the affected window first.",
    "messages_ignore_by_id": "Soft-delete one message into ignored storage; use messages_restore_ignored to restore ignored history.",
    "messages_restore_ignored": "Restore ALL ignored messages for bot_id into history.",
    "messages_get_for_summary": "Read original messages referenced by a stored summary_id.",
    "messages_mark_recalled": "Mark specified messages as recalled from summary expansion; writes metadata.",
    "messages_remove_last_partial": "Delete bot_id's latest message if role matches. Despite the name, it does NOT verify partial status; use only for an explicitly authorized cleanup.",
    "context_get_recent": "Read recent messages plus relevant memories for bot_id; query optionally guides memory retrieval.",
    "facts_extract": "Extract facts from supplied role/content messages; store=true persists them. bot_id and user_id are required scopes.",
    "system_stats": "Read bot_id's memory and message statistics.",
    "system_run_maintenance": "Run selected memory maintenance operations for bot_id; dry_run=false applies changes.",
    "profile": "Structured profile attributes, not semantic memory. action: summary|list|get|set|delete; entity_type: user|bot. set/delete mutate.",
    "sessions_close": "Archive one session without deleting its messages.",
    "sessions_get": "Read one durable session by ID and bot_id.",
    "sessions_list": "List bot_id's sessions newest first; since accepts ISO time or Unix seconds.",
    "sessions_get_active": "Read the active session for bot_id/user_id without creating one.",
    "sessions_get_or_create_active": "Return the active session for bot_id/user_id; creates one if absent.",
    "sessions_rotate": "Archive the active conversation and create its successor. Changes the user's active thread; not a provider-context reset.",
    "tasks_list": "List compact tasks by status/project/literal q; project_id='none' means unassigned. Use tasks_get for full details.",
    "tasks_search_semantic": "Find related tasks by meaning; mode reports semantic or keyword fallback. Use tasks_list for exact filters.",
    "tasks_get": "Read full task, steps, dependencies, and response by UUID or TASK-N.",
    "tasks_get_context": "Read task and project briefing by UUID or TASK-N before starting work.",
    "tasks_associate_current": "Link this trusted turn when actually starting/resuming a task, never from a mention alone. Supply UUID or TASK-N; unsupported caller context fails closed.",
    "tasks_create": "Create a task with optional steps. Pass your bot_id; associate_current_turn=true links work started now using trusted server context.",
    "tasks_update": "Patch provided task fields. Submit your own work to REVIEW; if Nick directly requests task review and closure, close verified, complete REVIEW tasks as COMPLETED. Agents cannot leave BUG. Pass own bot_id (owner if none); associate_current_turn links actual work.",
    "initiative_state": "Read, acquire an expiring lease, or CAS-checkpoint recurring task state; trusted turn identity owns writes and stale owners conflict.",
    "tasks_delete": "Permanently delete task and steps. Prefer tasks_update status=CANCELLED to retain history.",
    "tasks_add_dependency": "Make task_id wait for depends_on_id (UUID or TASK-N). Self-dependencies and cycles are rejected.",
    "tasks_remove_dependency": "Remove depends_on_id as a prerequisite of task_id; UUID or TASK-N accepted.",
    "tasks_promote": "Create a project from a task and move that task into it.",
    "tasks_regenerate": "Use server-side inference to replace task title and steps. Prefer writing your plan with tasks_update and steps_set.",
    "steps_update": "Update one step's status/output, or batch updates (each needs step_id); nonempty updates overrides scalar fields. Full batch fields: skill reference.",
    "steps_delete": "Permanently delete step_id or nonempty step_ids (batch wins). Prefer SKIPPED via steps_update to preserve the audit trail.",
    "steps_add": "Append task steps; each needs title, defaults to type=PLAN/status=PENDING. Does not replace existing steps.",
    "steps_set": "Atomically replace the ENTIRE checklist, deleting existing steps. Each needs title; [] clears it. Use steps_add to append.",
    "projects_list": "List compact projects with task counts; projects_get returns full context and tasks.",
    "projects_get": "Read a project, context prompt, and tasks by project UUID.",
    "projects_get_context": "Read only the project's name and instruction briefing, without its task list.",
    "projects_create": "Create a project; context_prompt carries shared instructions and agent_bot_id is its assigned bot.",
    "projects_update": "Patch provided project fields by UUID; omitted fields remain unchanged.",
    "projects_delete": "Permanently delete a project; its tasks remain but become unassigned.",
    "activity_get": "Read recent task/project activity; limit is capped at 100.",
    "bots_list_available": "List possible inter-bot message targets; availability does not override protected-bot boundaries.",
    "bots_send_message": "Durable async message: steer a busy capable bot or start one idle turn. delivery=when_idle skips steering; wait_for_reply blocks. force never permits concurrent turns.",
    "bots_delivery_get": "Read durable delivery status/error by delivery_id; accepted or queued is not completed.",
    "bots_deliveries_list": "Read durable deliveries filtered by sender, target, or status.",
    "bots_delivery_cancel": "Cancel a durable delivery only while QUEUED; cannot abort an active turn.",
    "agent_context_health": "Read context headroom, estimates, and supported reset/compaction capabilities for a bot.",
    "agent_context_reset": "Reset an idle provider session, preserving durable history. session_policy: reset_retain_history|reset_without_history; inspect capabilities first.",
    "agent_context_compact": "Queue one Claude /compact turn for idle execution; use a stable idempotency_key and your sender_bot_id.",
    "self_recap": "Generate a continuation briefing from your bot_id's history using inference. Window is hours+days*24; store=true writes a summary.",
    "self_tail": "Read your bot_id's last count raw conversation bubbles for context restoration. Absorb them; do not reprint the transcript unless asked.",
    "self_fwd": "Forward YOUR recent bubbles to target_bot_id; sender_bot_id must be your explicit slug. Async by default; force never authorizes concurrency.",
    "self_system_prompt": "Read your own durable persona (action=view) or fully replace it (edit, new_prompt). Use your bot_id; edits apply next turn, not to the harness wrapper.",
    "self_model": "View or persistently switch bot_id's model endpoint; applies next turn, not this one. continue_prompt queues one when_idle follow-up turn on the new model.",
    "ops_list_operations": "Discover enabled ops and schemas before ops_run; include_disabled grants no execution permission.",
    "ops_run": "Run a discovered op with validated args and idempotency_key. Approval/queued is not success; verify ops_job_status and target health. Never bypass a denial.",
    "ops_job_status": "Read/reconcile an ops job's terminal state and exit code; output_tail_bytes optionally includes bounded command output.",
    "media_lookup": "Search movie/series titles before media_add; returns exact external IDs and existing-library status. kind: auto|movie|series.",
    "media_add": "Add movie|series and optionally start downloading (search_now defaults true). Prefer external_id from media_lookup; query otherwise picks first match. Advanced options: skill reference.",
    "media_queue": "Read combined Sonarr/Radarr/SABnzbd download queues, bounded per service.",
    "media_history": "Read recent downloads/imports/failures. kind: all|movie|series; event: grabbed|imported|failed or omit for all.",
    "media_pipeline_status": "Trace one title through library, downloader, and recent events. kind: auto|movie|series.",
    "media_library_stats": "Read compact library counts and downloader health/queue summary.",
    "home_audio_devices": "List allowed Home Assistant speakers/displays, state, and voice catalog URL. Read-only.",
    "speech_generate": "Generate a stored WAV; no playback. Uses voice or bot default, never substitutes voices. Returns asset_id for home_audio_enqueue.",
    "home_audio_enqueue": "Audible: queue text or a generated WAV asset_id to an explicit target. Shared FIFO, waits for idle, expires after five minutes. Reuse idempotency_key; queued is not played.",
    "home_audio_status": "Read announcement state and errors. completed means observed playback completion; interrupted/failed may have played partially and are never auto-replayed.",
    "home_audio_cancel": "Cancel only a queued announcement. Does not stop preparing/playing jobs; inspect returned status.",
    "generate_image": "Generate or edit an image; provider: grok (default)|openai. reference_asset_id reuses an image. Returns inline image and asset_id/URLs for iteration.",
    "web_search": "Search current web sources; provider: brave|reddit|tavily|duckduckgo, or omit for configured fan-out. max_results is per provider. Use crawl4ai for page content.",
    "x_search": "Paid X search, last 7 days, 10–100 posts/page. sort_order=relevancy for summaries; min_likes:/min_reposts: (not min_faves). include_authors adds handles at extra cost.",
    "x_counts": "Paid X post counts per minute|hour|day, last 7 days. Find peaks, then x_search them.",
}

# A compact pointer on every tool works even when discovery loads one schema only.
# Group references are loaded on demand, not injected with the whole catalog.
REFERENCE_GROUPS: dict[str, tuple[str, ...]] = {
    "tasks": ("tasks_", "steps_", "projects_", "activity_", "initiative_"),
    "memory-history": ("memory_", "messages_", "context_", "facts_", "system_", "profile"),
    "sessions-delivery": ("sessions_", "bots_", "agent_", "self_"),
    "ops-media": ("ops_", "media_", "generate_", "web_", "x_", "speech_", "home_audio_"),
}


def tool_reference(name: str) -> str:
    """Return a skill-relative reference for a known tool family."""
    for group, prefixes in REFERENCE_GROUPS.items():
        if name.startswith(prefixes):
            return f"bawthub-mcp/references/{group}.md"
    raise ValueError(f"No skill reference for MCP tool {name!r}")


def tool_description(name: str) -> str:
    return f"{TOOL_SUMMARIES[name]} Details: {tool_reference(name)}."
