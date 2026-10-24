"""Model-facing selection guidance; workflows live in the bawthub-mcp skill.

Keep names and parameters stable: approval policies and saved calls address them.
Each description must retain side effects and call-critical constraints. Do not
copy argument types/defaults or tutorials here. See docs/MCP_TOOL_DESIGN.md.
"""

from .task_step_types import STEP_STATUSES, TASK_STATUSES

_TASK_STATUS_GUIDE = "|".join(TASK_STATUSES)
_STEP_STATUS_GUIDE = "|".join(STEP_STATUSES) + "; not IN_PROGRESS"
# TASK-1368: compact form of task_step_types.BLOCKED_DEFINITION (full text in
# the tool docstrings and the bawthub-mcp reference).
_BLOCKED_GUIDE = (
    "BLOCKED needs blocked_reason (what Nick must do/decide): only his decisions, "
    "ship/restart/credentials/external deps, or his intent. Never nits or bot disputes."
)
# tasks_update is near the 100-token description cap; it points at the rules.
_BLOCKED_POINTER = "BLOCKED (rare) needs blocked_reason: what Nick must do/decide; rules in tasks_create."

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
    "messages_find": "Locate messages across bots you may read: compact session cards (best + latest), bot/age/speaker facets. Identifiers match exactly. No query browses recent; handle alone pages. ok|no_match|unavailable.",
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
    "context_recall": "Read-only isolated recall; trusted caller required. Returns bounded cited quotes, not transcripts. tasks opts into shared-task search; tool evidence is metadata only. Expiring handles refresh evidence; repeat sources/task_ids. Uses configured utility model.",
    "working_memory_list": "Read your own working-memory items for the current user from trusted turn identity; open only unless include_archived.",
    "working_memory_create": "Add an item. idempotency_key: a fresh UUID per create, reused only to retry it. Open-item cap 20. initiative_link points at a task; no next_action/blockers.",
    "working_memory_update": "CAS edit: stale expected_revision returns the current item, writes nothing. Omitted fields stay; clear nulls listed fields. done|cancelled archives.",
    "facts_extract": "Extract facts from supplied role/content messages; store=true persists them. bot_id and user_id are required scopes.",
    "system_stats": "Read bot_id's memory and message statistics.",
    "system_run_maintenance": "Run selected memory maintenance operations for bot_id; dry_run=false applies changes.",
    "profile": "Structured profile attributes, not semantic memory. action: summary|list|get|set|delete; entity_type: user|bot. set/delete mutate.",
    "sessions_close": "Archive session in required bot_id; retain messages.",
    "sessions_get": "Read session by ID in required bot_id.",
    "sessions_list": "List required bot_id's sessions newest first; since: ISO time or Unix seconds.",
    "sessions_get_active": "Read required bot_id/user_id's active session; never creates.",
    "sessions_get_or_create_active": "Get/create active session: required bot_id/user_id.",
    "sessions_rotate": "Rotate required bot_id's active thread: archive/create successor, not provider-context reset.",
    "tasks_list": f"List compact tasks; literal q, project_id='none' unassigned. One status: {_TASK_STATUS_GUIDE}. Omitted excludes DELETED; tasks_get for detail.",
    "tasks_search_semantic": "Semantic task search; mode reports keyword fallback. Exact filters: tasks_list.",
    "tasks_get": "Read task, steps, dependencies, response by cuid/TASK-N.",
    "tasks_get_context": "Read task/project briefing by cuid/TASK-N before work.",
    "tasks_reviews_list": "Task reviews, newest first; cursor paginates. Not response.",
    "tasks_reviews_add": "Append review, not response/status. Author: own bot_id; retry: stable idempotency_key.",
    "tasks_associate_current": "Link trusted turn when actually working a task, never from a mention. Supply cuid or TASK-N; unsupported context fails closed.",
    "tasks_create": f"Create task/steps. Own bot_id; associate_current_turn links new work via trusted turn. {_BLOCKED_GUIDE}",
    "tasks_update": f"Patch; UI/UX Nick can see → REVIEW with dev URL/steps. Other work: Snark/Rook close COMPLETED after review (Tier 0: self-check), attributable push and test/live receipt. BUG is human-locked. {_BLOCKED_POINTER} Own bot_id fills missing owner; associate_current_turn links actual work.",
    "initiative_state": "Read, acquire an expiring lease, or CAS-checkpoint recurring task state; trusted turn identity owns writes and stale owners conflict.",
    "tasks_delete": "Permanently delete task/steps; prefer tasks_update CANCELLED to retain history.",
    "tasks_add_dependency": "Add task_id prerequisite depends_on_id; cuid/TASK-N. Rejects self/cycles.",
    "tasks_remove_dependency": "Remove prerequisite depends_on_id from task_id; cuid/TASK-N.",
    "tasks_promote": "Promote task into new project.",
    "tasks_regenerate": "Inference replaces task title/steps; prefer tasks_update/steps_set.",
    "steps_update": f"Patch step or batch (step_id required); nonempty updates overrides scalars. Status: {_STEP_STATUS_GUIDE}.",
    "steps_delete": "Permanently delete step_id or nonempty step_ids (batch wins). Prefer SKIPPED via steps_update to preserve the audit trail.",
    "steps_add": f"Append titled steps (default PLAN/PENDING). Status: {_STEP_STATUS_GUIDE}.",
    "steps_set": f"Atomically replace ENTIRE checklist, deleting old steps; title required, [] clears. Status: {_STEP_STATUS_GUIDE}.",
    "projects_list": "List compact projects with task counts; projects_get returns full context and tasks.",
    "projects_get": "Read project context and tasks by project ID (cuid).",
    "projects_get_context": "Read only the project's name and instruction briefing, without its task list.",
    "projects_create": "Create a project; context_prompt carries shared instructions and agent_bot_id is its assigned bot.",
    "projects_update": "Patch provided project fields by project ID (cuid); omitted fields remain unchanged.",
    "projects_delete": "Permanently delete a project; its tasks remain but become unassigned.",
    "activity_get": "Read recent task/project activity; limit is capped at 100.",
    "bots_list_available": "List inter-bot message targets and their tags; private and inactive bots are hidden.",
    "bots_send_message": "Durable async message: steer a busy capable bot or start one idle turn. delivery=when_idle skips steering; wait_for_reply blocks. force never permits concurrent turns.",
    "bots_route_message": "Queue a separate routed bot turn using trusted caller identity; never steer or broaden opt-in. Reuse idempotency_key for retries.",
    "bots_delivery_get": "Read durable delivery status/error by delivery_id; accepted or queued is not completed.",
    "bots_deliveries_list": "Compact delivery rows, newest first; bots_delivery_get reads one full receipt. bot_id matches either side; active_only=in flight; status takes a comma list; chat/self rows hidden unless include_chat. Limit ≤50.",
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
    "ops_quiesce_ack": "Answer the quiesce holding this turn: stopping|cannot_stop, optional note.",
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
    "memory-history": ("memory_", "messages_", "context_", "facts_", "system_", "profile", "working_memory_"),
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
