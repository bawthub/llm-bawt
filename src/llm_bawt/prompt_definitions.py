"""Built-in prompt-registry definitions: every default key, title and seed loader.

Split out of ``prompt_registry`` (monolith gate) so the registry module keeps the
store/resolver logic and this module owns the catalog of built-in keys. Bodies
are seed data only; callers resolve live text through ``get_prompt_resolver()``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .chat_prompt_defaults import (
    AGENT_GLOBAL_PROMPT,
    AGENT_USER_PREFIX,
    COMPACT_SUMMARY_DEFAULT,
    AGENT_VOICE_PREFIX,
    MCP_TOOL_CONTEXT_TEMPLATE,
    RESPONSE_STYLE_DEEP_DIVE,
    RESPONSE_STYLE_ELI5,
    RESPONSE_STYLE_TLDR,
    RUNTIME_CONTEXT_TEMPLATE,
    SCOPED_COMMIT_PROMPT,
    SESSION_SEED_ACK,
    TURN_CUTOFF_NOTICE,
    SESSION_SEED_BRIEFING,
    TTS_OUTPUT_INSTRUCTIONS,
)
from .prompt_default_loaders import (
    load_agents_docs,
    load_approvals_approver_request,
    load_approvals_approver_resolved,
    load_approvals_mcp_result_guidance,
    load_approvals_continuation_reset_guidance,
    load_approvals_approved_guidance,
    load_approvals_denied_guidance,
    load_approvals_responded_guidance,
    load_approvals_no_response_guidance,
    load_agents_project_plan,
    load_agents_review,
    load_agents_task_execution,
    load_agents_task_spec,
    load_chat_steer_interrupt_replacement,
    load_chat_steer_next_aside,
    load_deliveries_cancel_notice,
    load_deliveries_stopped_after_start_notice,
    load_global_recall_guidance,
    load_inter_bot_reply_channel_async,
    load_inter_bot_restart_retry_continue,
    load_inter_bot_reply_channel_waited,
    load_inter_bot_reply_channel_waited_no_return,
    load_history_summarization_batch,
    load_history_summarization_single,
    load_history_summarization_system,
    load_injected_skills_entry,
    load_injected_skills_preamble,
    load_media_prompt_expansion_video,
    load_memory_extraction_fact,
    load_memory_extraction_fact_system,
    load_memory_extraction_summary,
    load_memory_extraction_summary_system,
    load_memory_extraction_update,
    load_memory_extraction_update_system,
    load_memory_maintenance_intent_content_only,
    load_memory_maintenance_intent_with_context,
    load_ops_quiesce_end_turn_new_turn,
    load_ops_quiesce_end_turn_no_receipt,
    load_ops_quiesce_receipt,
    load_ops_quiesce_interrupt_frame,
    load_ops_quiesce_request,
    load_ops_quiesce_resume,
    load_ops_quiesce_resume_blocked,
    load_profile_consolidation,
    load_recall_synthesis_system,
    load_self_recap_system,
    load_setup_first_agent_prompt,
    load_working_memory_notice,
    load_working_memory_scheduled_note,
    load_working_memory_section,
    load_working_memory_section_empty,
    load_working_memory_section_unavailable,
)

COMPACT_SUMMARY_KEY = "agents.compaction_summary"


@dataclass(frozen=True)
class PromptDefinition:
    """One built-in prompt definition."""

    key: str
    title: str
    category: str
    required_vars: tuple[str, ...] = ()
    format: str = "plain_text"
    loader: Callable[[], str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def default_body(self) -> str:
        if self.loader is None:
            return ""
        return self.loader()


DEFAULT_PROMPT_DEFINITIONS: dict[str, PromptDefinition] = {
    "history.summarization.single": PromptDefinition(
        key="history.summarization.single",
        title="History Summarization",
        category="summarization",
        required_vars=("messages",),
        loader=load_history_summarization_single,
    ),
    "history.summarization.batch": PromptDefinition(
        key="history.summarization.batch",
        title="History Summarization Batch",
        category="summarization",
        required_vars=("sessions_blob",),
        loader=load_history_summarization_batch,
    ),
    "memory.extraction.fact": PromptDefinition(
        key="memory.extraction.fact",
        title="Fact Extraction",
        category="memory_extraction",
        required_vars=("conversation",),
        loader=load_memory_extraction_fact,
    ),
    "memory.extraction.update": PromptDefinition(
        key="memory.extraction.update",
        title="Memory Update Actions",
        category="memory_extraction",
        required_vars=("existing_memories", "new_facts"),
        loader=load_memory_extraction_update,
    ),
    "memory.extraction.summary": PromptDefinition(
        key="memory.extraction.summary",
        title="Summary Fact Extraction",
        category="memory_extraction",
        required_vars=("summary_text", "start_date", "end_date"),
        loader=load_memory_extraction_summary,
    ),
    # System instructions for the background jobs (TASK-1721).
    "history.summarization.system": PromptDefinition(
        key="history.summarization.system",
        title="History Summarization System",
        category="summarization",
        loader=load_history_summarization_system,
    ),
    "memory.extraction.fact.system": PromptDefinition(
        key="memory.extraction.fact.system",
        title="Fact Extraction System",
        category="memory_extraction",
        loader=load_memory_extraction_fact_system,
    ),
    "memory.extraction.summary.system": PromptDefinition(
        key="memory.extraction.summary.system",
        title="Summary Fact Extraction System",
        category="memory_extraction",
        loader=load_memory_extraction_summary_system,
    ),
    "memory.extraction.update.system": PromptDefinition(
        key="memory.extraction.update.system",
        title="Memory Update Actions System",
        category="memory_extraction",
        loader=load_memory_extraction_update_system,
    ),
    "profile.consolidation": PromptDefinition(
        key="profile.consolidation",
        title="Profile Consolidation",
        category="profile",
        required_vars=("attributes",),
        loader=load_profile_consolidation,
        format="json_instruction",
    ),
    "memory.maintenance.intent_with_context": PromptDefinition(
        key="memory.maintenance.intent_with_context",
        title="Memory Intent Inference With Context",
        category="memory_maintenance",
        required_vars=("conversation", "fact"),
        loader=load_memory_maintenance_intent_with_context,
    ),
    "memory.maintenance.intent_content_only": PromptDefinition(
        key="memory.maintenance.intent_content_only",
        title="Memory Intent Inference Content Only",
        category="memory_maintenance",
        required_vars=("fact",),
        loader=load_memory_maintenance_intent_content_only,
    ),
    "chat.tts_output_instructions": PromptDefinition(
        key="chat.tts_output_instructions",
        title="TTS Output Instructions",
        category="chat",
        required_vars=(),
        loader=lambda: TTS_OUTPUT_INSTRUCTIONS,
    ),
    "chat.scoped_commit": PromptDefinition(
        key="chat.scoped_commit",
        title="Scoped Commit Request",
        category="chat",
        required_vars=("scope",),
        loader=lambda: SCOPED_COMMIT_PROMPT,
        metadata={
            "notes": (
                "Sent when the Commit action is used on a completed changed-files "
                "summary. BawtHub supplies {scope} from that turn's repository and "
                "file summary; llm-bawt validates and renders the final request."
            ),
        },
    ),
    "chat.global_recall_guidance": PromptDefinition(
        key="chat.global_recall_guidance",
        title="Global Recall / Cross-Bot Memory Guidance",
        category="chat",
        required_vars=(),
        loader=load_global_recall_guidance,
        metadata={
            "notes": (
                "Injected into the system prompt of every memory-enabled bot "
                "(gate: requires_memory). Default is the GLOBAL_SYSTEM_PROMPT "
                "constant; override per bot to change recall/cross-bot guidance. "
                "TASK-490: migrated out of a hardcoded Python constant."
            ),
        },
    ),
    "chat.response_style.tldr": PromptDefinition(
        key="chat.response_style.tldr",
        title="Response Style — TL;DR",
        category="chat",
        required_vars=(),
        loader=lambda: RESPONSE_STYLE_TLDR,
        metadata={"notes": "Stamped on the user message when 'tldr' keyword fires. TASK-490."},
    ),
    "chat.response_style.eli5": PromptDefinition(
        key="chat.response_style.eli5",
        title="Response Style — ELI5",
        category="chat",
        required_vars=(),
        loader=lambda: RESPONSE_STYLE_ELI5,
        metadata={"notes": "Stamped on the user message when 'eli5' keyword fires. TASK-490."},
    ),
    "chat.response_style.deep_dive": PromptDefinition(
        key="chat.response_style.deep_dive",
        title="Response Style — Deep Dive",
        category="chat",
        required_vars=(),
        loader=lambda: RESPONSE_STYLE_DEEP_DIVE,
        metadata={"notes": "Stamped on the user message when 'deep dive' keyword fires. TASK-490."},
    ),
    "chat.agent_voice_prefix": PromptDefinition(
        key="chat.agent_voice_prefix",
        title="Agent Voice-Mode User Message Prefix",
        category="chat",
        required_vars=(),
        loader=lambda: AGENT_VOICE_PREFIX,
        metadata={
            "notes": (
                "Prepended to user messages sent to agent backends "
                "(Claude Code, Codex) ONLY when tts_mode=true. "
                "Text-mode turns are passed through untouched. Exists because "
                "agent SDKs lock the system prompt at session start and ignore "
                "it on resume — so the existing chat.tts_output_instructions "
                "that ride along in the system prompt never reach the agent "
                "if the session started in text mode. This per-turn prefix is "
                "the reliable signal. Leave body empty to disable."
            ),
        },
    ),
    "chat.agent_user_prefix": PromptDefinition(
        key="chat.agent_user_prefix",
        title="Agent Per-Turn User Message Prefix",
        category="chat",
        required_vars=(),
        loader=lambda: AGENT_USER_PREFIX,
        metadata={
            "notes": (
                "Prepended to EVERY user message sent to agent backends "
                "(Claude Code, Codex) when the request flag "
                "inject_user_prefix=true. Independent of tts_mode and of "
                "chat.agent_voice_prefix — if both are active, they stack. "
                "Exists for the same reason agent_voice_prefix does (agent "
                "SDKs lock the system prompt at session start and ignore "
                "later edits) but isn't gated on voice mode, so it's the "
                "right place for any always-on per-turn directive (voice "
                "guidance, identity reminder, output-format constraints "
                "that must survive session resume). The toggle lives in "
                "the chat composer's unified popup. Leave body empty to "
                "disable even when the flag is on."
            ),
        },
    ),
    COMPACT_SUMMARY_KEY: PromptDefinition(
        key=COMPACT_SUMMARY_KEY,
        title="Agent Compaction Summary Guidance",
        category="agent_execution",
        required_vars=(),
        loader=lambda: COMPACT_SUMMARY_DEFAULT,
        metadata={"notes": (
            "Default guidance for bare /compact and automatic Claude CLI compaction. "
            "Explicit user instructions always bypass it. Changes take effect on the "
            "next compaction without resetting the session; this is a soft target, "
            "not a hard output cap. The bridge fetches this global prompt at each use."
        )},
    ),
    "agents.mcp_tool_context_template": PromptDefinition(
        key="agents.mcp_tool_context_template",
        title="Agent MCP Tool Context Block",
        category="agent_execution",
        required_vars=("bot_slug",),
        loader=lambda: MCP_TOOL_CONTEXT_TEMPLATE,
        metadata={
            "notes": (
                "Appended to the system prompt by the claude-code BRIDGE process "
                "(gated on MCP servers being configured). Tells the agent its "
                "bot_id for bawthub MCP calls. Requires {bot_slug}. The bridge "
                "cannot import llm_bawt, so it fetches this via GET /v1/prompts/"
                "{key} with a byte-identical local fallback. TASK-490."
            ),
        },
    ),
    "agents.runtime_context_template": PromptDefinition(
        key="agents.runtime_context_template",
        title="Agent Runtime-Context Model Block",
        category="agent_execution",
        required_vars=("model",),
        loader=lambda: RUNTIME_CONTEXT_TEMPLATE,
        metadata={
            "notes": (
                "Injected app-side by the claude-code agent backend (prepended "
                "to the system prompt) so the agent has a ground-truth model id. "
                "Requires {model}. TASK-490: migrated out of a hardcoded literal."
            ),
        },
    ),
    "agents.global_prompt": PromptDefinition(
        key="agents.global_prompt",
        title="Agent Global System Prompt",
        category="agent_execution",
        required_vars=(),
        loader=lambda: AGENT_GLOBAL_PROMPT,
        metadata={
            "notes": (
                "Shared system-prompt block for AGENT backends (Claude Code, "
                "Codex). Injected ONLY when the per-bot runtime setting "
                "`agent_global_prompt_enabled` is true. Added to the cacheable "
                "system-prompt prefix (agent-only; chat bots never see it). "
                "Default body steers planning away from harness plan mode and "
                "toward the BawtHub task system for observability. Supports "
                "per-bot overrides via scope_type=bot; leave body empty to "
                "inject nothing even when the flag is on."
            ),
        },
    ),
    "agents.session_seed_briefing": PromptDefinition(
        key="agents.session_seed_briefing",
        title="Agent Session Seed Briefing",
        category="agent_execution",
        required_vars=(),
        loader=lambda: SESSION_SEED_BRIEFING,
        metadata={
            "notes": (
                "Header of the briefing turn that opens a fresh agent session "
                "(/new, reset, backend switch, cold start) with app-selected "
                "history. The bridge renders the seed transcript beneath it. "
                "Claude Code writes it into a synthetic transcript; Codex "
                "injects it as a thread item. Resolved per bot at dispatch. "
                "TASK-1579."
            ),
        },
    ),
    "agents.turn_cutoff_notice": PromptDefinition(
        key="agents.turn_cutoff_notice",
        title="Agent Turn Cut-off Notice",
        category="agent_execution",
        required_vars=("cause", "ended_at"),
        loader=lambda: TURN_CUTOFF_NOTICE,
        metadata={
            "notes": (
                "Prefixed to the first agent turn after the system cut off the "
                "previous turn in the same session: a bridge restart (host "
                "shutdown) or the app's inactivity watchdog. A killed tool call "
                "reaches the agent as the SDK's user-rejection text, so this must "
                "say nobody rejected anything and the bot should resume. {cause} "
                "is the cut-off cause, {ended_at} its UTC time. A body that is "
                "blank or no longer renders falls back to this default. Resolved "
                "per bot at dispatch. TASK-1738."
            ),
        },
    ),
    "agents.session_seed_ack": PromptDefinition(
        key="agents.session_seed_ack",
        title="Agent Session Seed Acknowledgement",
        category="agent_execution",
        required_vars=(),
        loader=lambda: SESSION_SEED_ACK,
        metadata={
            "notes": (
                "Synthetic assistant reply that follows the session seed "
                "briefing, so the seeded history ends on an assistant turn. "
                "Resolved per bot at dispatch. TASK-1579."
            ),
        },
    ),
    "agents.tag_context": PromptDefinition(
        key="agents.tag_context",
        title="Agent Bot Tags and Roster",
        category="agent_execution",
        required_vars=(),
        loader=None,
        metadata={
            "notes": (
                "Bot-tag section for AGENT backends (TASK-1249). Optional variables: "
                "{own_tag_instructions} (instructions of this bot's role tags, or "
                "their descriptions), {bot_roster} (visible agent bots: tags, "
                "endpoint ref, provider) and {tag_catalog} (role tags: name, "
                "description, capabilities). Literal braces must be doubled. The "
                "default body is empty, which injects nothing. Supports per-bot "
                "overrides via scope_type=bot."
            ),
        },
    ),
    "agents.task_spec": PromptDefinition(
        key="agents.task_spec",
        title="Agent Task Spec (Planning Mode)",
        category="agent_execution",
        required_vars=("task_context", "task_url", "extra_api_lines"),
        loader=load_agents_task_spec,
    ),
    "agents.task_execution": PromptDefinition(
        key="agents.task_execution",
        title="Agent Task Execution",
        category="agent_execution",
        required_vars=("task_context", "task_url", "finish_instruction"),
        loader=load_agents_task_execution,
    ),
    "agents.project_plan": PromptDefinition(
        key="agents.project_plan",
        title="Agent Project Plan",
        category="agent_execution",
        required_vars=(
            "project_name",
            "project_details",
            "project_context",
            "tasks_url",
            "project_url",
            "project_id",
        ),
        loader=load_agents_project_plan,
    ),
    "agents.review": PromptDefinition(
        key="agents.review",
        title="Agent Review Dispatch",
        category="agent_execution",
        required_vars=(
            "task_context",
            "review_comment",
            "previous_response",
            "task_url",
            "finish_instruction",
        ),
        loader=load_agents_review,
    ),
    "agents.docs": PromptDefinition(
        key="agents.docs",
        title="Agent Task System Reference Doc",
        category="agent_execution",
        required_vars=("origin", "task_section"),
        loader=load_agents_docs,
    ),
    "self_recap.system": PromptDefinition(
        key="self_recap.system",
        title="Self-Recap Handoff Analyst",
        category="agent_execution",
        required_vars=(),
        loader=load_self_recap_system,
        metadata={
            "notes": (
                "System prompt for the self_recap MCP tool's HANDOFF ANALYST. "
                "Sent to Grok with the raw conversation transcript to produce a "
                "cold-start continuation briefing (itemized TOPIC LOG + pending "
                "work). No placeholders — the transcript is passed as a separate "
                "user message, so the body is used verbatim (never .format()-ed). "
                "Bot-scoped overrides supported via scope_type=bot."
            ),
        },
    ),
    "recall.synthesis.system": PromptDefinition(
        key="recall.synthesis.system",
        title="Context Recall Evidence Selector",
        category="agent_execution",
        loader=load_recall_synthesis_system,
        metadata={"notes": (
            "System prompt for the context_recall MCP tool's isolated synthesis call "
            "(global maintenance_model). The question + evidence JSON is the separate "
            "user prompt; body is used verbatim (never .format()-ed). Must keep the "
            "JSON findings contract — quotes are re-grounded against source bytes "
            "and ungrounded ones dropped. Bot-scoped overrides supported."
        )},
    ),
    "inter_bot.reply_channel.waited": PromptDefinition(
        key="inter_bot.reply_channel.waited",
        title="Inter-Bot Reply Channel: Waited",
        category="agent_execution",
        loader=load_inter_bot_reply_channel_waited,
        metadata={"notes": (
            "Appended to a bot-authored delivery sent with wait_for_reply=true. Tells "
            "the recipient its final response is returned to the sender's tool call, "
            "so it should not send a redundant callback. Resolved per delivery; "
            "target-bot overrides supported."
        )},
    ),
    "inter_bot.reply_channel.async": PromptDefinition(
        key="inter_bot.reply_channel.async",
        title="Inter-Bot Reply Channel: Async",
        category="agent_execution",
        required_vars=("sender_bot_id",),
        loader=load_inter_bot_reply_channel_async,
        metadata={"notes": (
            "Appended to a bot-authored async (default, fire-and-forget) delivery. "
            "Tells the recipient its final response is NOT returned to the sender, so "
            "any requested answer must be sent back with bots_send_message. "
            "{sender_bot_id} is substituted by plain replacement (never .format()). "
            "Resolved per delivery; target-bot overrides supported."
        )},
    ),
    "inter_bot.reply_channel.waited_no_return": PromptDefinition(
        key="inter_bot.reply_channel.waited_no_return",
        title="Inter-Bot Reply Channel: Waited, Sender Unreachable",
        category="agent_execution",
        loader=load_inter_bot_reply_channel_waited_no_return,
        metadata={"notes": (
            "Replaces the waited note when the sender is an inactive bot (such as "
            "the external outside-agent bot, TASK-1362), which cannot receive bot "
            "messages. Same contract minus the separate follow-up. An async "
            "delivery from an inactive sender gets no note. Resolved per "
            "delivery; target-bot overrides supported."
        )},
    ),
    "inter_bot.restart_retry.continue": PromptDefinition(
        key="inter_bot.restart_retry.continue",
        title="Inter-Bot Restart Retry: Continue",
        category="agent_execution",
        loader=load_inter_bot_restart_retry_continue,
        metadata={"notes": (
            "Body of the system-created restart-retry delivery (TASK-1231B) that "
            "resumes a turn killed by its bridge host shutting down. It is prefixed "
            "to the original request, so it must tell the bot to check what it "
            "already completed and not repeat side effects. Used verbatim; a blank "
            "body falls back to the seed. Target-bot overrides supported."
        )},
    ),
    "approvals.approved_guidance": PromptDefinition(
        key="approvals.approved_guidance",
        title="Approved Tool Continuation",
        category="agent_execution",
        required_vars=("tool_name", "shown"),
        loader=load_approvals_approved_guidance,
        metadata={"notes": "Re-issues a user-approved tool call with its complete arguments; bot overrides supported."},
    ),
    "approvals.denied_guidance": PromptDefinition(
        key="approvals.denied_guidance",
        title="Denied Tool Continuation",
        category="agent_execution",
        required_vars=("tool_name", "shown"),
        loader=load_approvals_denied_guidance,
        metadata={"notes": "Continues without a user-denied tool call; bot overrides supported."},
    ),
    "approvals.responded_guidance": PromptDefinition(
        key="approvals.responded_guidance",
        title="Responded Tool Continuation",
        category="agent_execution",
        required_vars=("tool_name", "shown", "message"),
        loader=load_approvals_responded_guidance,
        metadata={"notes": "Delivers a user's guidance instead of running the tool; bot overrides supported."},
    ),
    "approvals.no_response_guidance": PromptDefinition(
        key="approvals.no_response_guidance",
        title="Reviewed Tool Continuation (No Response)",
        category="agent_execution",
        required_vars=("tool_name",),
        loader=load_approvals_no_response_guidance,
        metadata={"notes": "Continues without the tool when a review supplies no further guidance; bot overrides supported."},
    ),
    "approvals.mcp_result_guidance": PromptDefinition(
        key="approvals.mcp_result_guidance",
        title="MCP Result Continuation Guidance",
        category="agent_execution",
        loader=load_approvals_mcp_result_guidance,
        metadata={"notes": (
            "Instructions after a server-executed MCP tool result. The app frames the "
            "result as protocol data, so editing this text cannot change tool-result "
            "classification. Resolved at each continuation; bot overrides supported."
        )},
    ),
    "approvals.continuation_reset_guidance": PromptDefinition(
        key="approvals.continuation_reset_guidance",
        title="Approval Continuation After /new",
        category="agent_execution",
        loader=load_approvals_continuation_reset_guidance,
        metadata={"notes": (
            "Instructions when an approval continuation follows the user's /new "
            "reset. Versioned MCP-result frames retain classification after edits. "
            "Resolved at each redirected continuation; bot overrides supported."
        )},
    ),
    "approvals.approver_request": PromptDefinition(
        key="approvals.approver_request",
        title="Approver Notice: Approval Requested",
        category="agent_execution",
        required_vars=("request_id", "requested_by", "operation", "title", "risk_level",
                       "severity", "args"),
        loader=load_approvals_approver_request,
        metadata={"notes": (
            "Body of the system delivery (TASK-1661) that tells a bot holding the "
            "approve_ops grant that another bot's ops_run approval is pending. "
            "{args} are the operation's arguments with x-sensitive values masked. "
            "It must name ops_approvals_get and ops_approval_resolve and say "
            "someone else may decide first. Approver-bot overrides supported."
        )},
    ),
    "approvals.approver_resolved": PromptDefinition(
        key="approvals.approver_resolved",
        title="Approver Notice: Approval Decided",
        category="agent_execution",
        required_vars=("request_id", "requested_by", "operation", "status", "resolved_by"),
        loader=load_approvals_approver_resolved,
        metadata={"notes": (
            "Steer-only follow-up (TASK-1661) telling an approver bot that a request "
            "it was notified about has been decided by someone else. {resolved_by} "
            "is a bot slug or \"a person\"; the resolver's reason is never included. "
            "Approver-bot overrides supported."
        )},
    ),
    "deliveries.cancel_notice": PromptDefinition(
        key="deliveries.cancel_notice",
        title="Cancelled Delivery Notice",
        category="agent_execution",
        required_vars=("delivery_id", "target_bot_id", "task_id", "reason", "excerpt"),
        loader=load_deliveries_cancel_notice,
        metadata={"notes": (
            "Body of the system delivery (TASK-1327) that tells an agent bot a "
            "message it sent to another bot was cancelled before it ran: by the "
            "user's Stop, the queue hold's Cancel all, the cancel API or recovery. "
            "{reason} is the recorded cancel reason and {excerpt} the message's "
            "first 200 characters. It must say the message will not be delivered "
            "and that a user-stopped message is not to be resent automatically. "
            "A blank body or one that no longer renders leaves the delivery "
            "unstamped for retry. Sender-bot overrides supported."
        )},
    ),
    "deliveries.stopped_after_start_notice": PromptDefinition(
        key="deliveries.stopped_after_start_notice",
        title="Stopped Started Delivery Notice",
        category="agent_execution",
        required_vars=("delivery_id", "target_bot_id", "task_id", "reason", "excerpt"),
        loader=load_deliveries_stopped_after_start_notice,
        metadata={"notes": (
            "Notifies a sender that the user's Stop ended a delivery after its "
            "target actually began running. Never say no work was done or that "
            "the message was cancelled before delivery; partial effects are possible. "
            "Do not resend automatically. Sender-bot overrides supported."
        )},
    ),
    "ops.quiesce_receipt": PromptDefinition(
        key="ops.quiesce_receipt",
        title="Ops Quiesce Receipt",
        category="agent_execution",
        loader=load_ops_quiesce_receipt,
        metadata={"notes": (
            "Prefix of the system delivery (TASK-1233 e) that brings a direct "
            "quiesce ops job's terminal receipt to the caller it held: that caller "
            "was told to end its turn, so this new turn is how it learns the "
            "outcome. The receipt JSON is appended. It must say the operation "
            "already ran or failed and is not to be re-run. Used verbatim; a blank "
            "body falls back to the seed. Target-bot overrides supported."
        )},
    ),
    "ops.quiesce_resume": PromptDefinition(
        key="ops.quiesce_resume",
        title="Ops Quiesce Resume",
        category="agent_execution",
        loader=load_ops_quiesce_resume,
        metadata={"notes": (
            "TASK-1798: prefix of the system delivery sent once per bot whose turn a "
            "quiesce ops job held, after the job ended and its target is up (it "
            "succeeded, or never ran). JSON with the job id, operation, state and the "
            "bot's own ops_quiesce_ack note is appended. It must tell the bot to pick "
            "its work up again and not to re-run the operation. Used verbatim; a blank "
            "body falls back to the seed. Target-bot overrides supported."
        )},
    ),
    "ops.quiesce_resume_blocked": PromptDefinition(
        key="ops.quiesce_resume_blocked",
        title="Ops Quiesce Resume Blocked (coordinator notice)",
        category="agent_execution",
        loader=load_ops_quiesce_resume_blocked,
        metadata={"notes": (
            "TASK-1798: prefix of the system delivery sent to each coordinator bot "
            "(runtime setting ops_resume_coordinator_capability) instead of resuming "
            "held bots, when a quiesce ops job ran and did not succeed. JSON with the "
            "job, its error and the held bots with their notes is appended. It must "
            "ask for a health check and a manual resume, never a re-run. Used "
            "verbatim; a blank body falls back to the seed. Target-bot overrides supported."
        )},
    ),
    "ops.quiesce_request": PromptDefinition(
        key="ops.quiesce_request",
        title="Ops Quiesce Request (control steer)",
        category="agent_execution",
        loader=load_ops_quiesce_request,
        metadata={"notes": (
            "The control steer (TASK-1233 B3) the app's quiesce control pump sends into "
            "each running turn a quiesce ops job holds, at the next tool boundary "
            "(priority next: running tools are never killed). It must ask the bot to "
            "reach a safe stopping point and end its turn, not start new work or poll, "
            "and may invite an ops_quiesce_ack. Sent once per held run, from system. "
            "Used verbatim; a blank body falls back to the seed. Target-bot overrides supported."
        )},
    ),
    "ops.quiesce_interrupt_frame": PromptDefinition(
        key="ops.quiesce_interrupt_frame",
        title="Ops Quiesce Interrupt Frame (escalated control steer)",
        category="agent_execution",
        loader=load_ops_quiesce_interrupt_frame,
        required_vars=("messages",),
        metadata={"notes": (
            "TASK-1748: the replacement query when a quiesce request sat queued behind a "
            "long-running tool and the app interrupted that tool. {messages} is replaced by "
            "the ops.quiesce_request text. It must tell the bot the interruption came from "
            "the system (not a user rejection) and to end its turn now, never to continue "
            "the task. A body without {messages} falls back to the seed. Target-bot "
            "overrides supported."
        )},
    ),
    "ops.quiesce_end_turn.new_turn": PromptDefinition(
        key="ops.quiesce_end_turn.new_turn",
        title="Ops Quiesce End Turn (receipt as a new turn)",
        category="agent_execution",
        loader=load_ops_quiesce_end_turn_new_turn,
        metadata={"notes": (
            "caller_instruction.message on ops_run / ops_job_status (TASK-1233) when a "
            "quiesce job holds its own caller and the terminal receipt will arrive as a "
            "new turn (approved job, or a direct job with a receipt callback). It must "
            "tell the bot to end its turn and not poll. Used verbatim; a blank body "
            "falls back to the seed. Target-bot overrides supported."
        )},
    ),
    "ops.quiesce_end_turn.no_receipt": PromptDefinition(
        key="ops.quiesce_end_turn.no_receipt",
        title="Ops Quiesce End Turn (no receipt pushed)",
        category="agent_execution",
        loader=load_ops_quiesce_end_turn_no_receipt,
        metadata={"notes": (
            "caller_instruction.message (TASK-1233) when a direct quiesce job holds a "
            "caller that had no session to bind a receipt callback to. It must tell the "
            "bot to end its turn and must not promise a receipt, callback or new turn; "
            "the bot checks ops_job_status in a later turn. Used verbatim; a blank body "
            "falls back to the seed. Target-bot overrides supported."
        )},
    ),
    "chat.steer.next_aside": PromptDefinition(
        key="chat.steer.next_aside",
        title="Mid-turn Message (next tool boundary)",
        category="agent_execution",
        required_vars=("message",),
        loader=load_chat_steer_next_aside,
        metadata={"notes": (
            "Frames a priority=next steer (TASK-1300): a user or bot message delivered "
            "into a running claude-code turn at its next tool boundary, without "
            "interrupting it. {message} is the message text. It must say the running "
            "work was not interrupted and ask the bot to continue the task. A body "
            "that is blank or no longer renders {message} falls back to this prompt's "
            "default body; if that can't render either, the steer is refused. Not used for system control steers. Target-bot overrides supported."
        )},
    ),
    "chat.steer.interrupt_replacement": PromptDefinition(
        key="chat.steer.interrupt_replacement",
        title="Mid-turn Interruption (replacement query)",
        category="agent_execution",
        required_vars=("messages",),
        loader=load_chat_steer_interrupt_replacement,
        metadata={"notes": (
            "The query that replaces an interrupted claude-code turn (TASK-1300): a "
            "priority=now steer, or 'Interrupt now' on held messages. {messages} is "
            "every interrupting message in send order, each attributed to its sender "
            "and separated by a blank line. It must ask the bot to continue the "
            "interrupted task unless a message says to stop. A body that is blank or "
            "no longer renders {messages} falls back to this prompt's default body; if "
            "that can't render either, the interrupt is refused. Target-bot overrides "
            "supported."
        )},
    ),
    "agents.working_memory.section": PromptDefinition(
        key="agents.working_memory.section",
        title="Working Memory: Section",
        category="agent_execution",
        required_vars=("items",),
        loader=load_working_memory_section,
        metadata={"notes": (
            "System-prompt section listing a bot's open working-memory items "
            "(TASK-1281, docs/WORKING_MEMORY.md 8.1). Rendered only for opted-in "
            "claude-code bots in a pinned session, and frozen into the session's pinned "
            "prompt until its next boundary. {items} is one JSON line per item, filled by "
            "plain replacement; a body without {items} falls back to this default. "
            "Bot overrides supported."
        )},
    ),
    "agents.working_memory.section_empty": PromptDefinition(
        key="agents.working_memory.section_empty",
        title="Working Memory: Empty Section",
        category="agent_execution",
        loader=load_working_memory_section_empty,
        metadata={"notes": (
            "Working-memory section used when the bot has no open items (TASK-1281, 8.1). "
            "Bot overrides supported."
        )},
    ),
    "agents.working_memory.section_unavailable": PromptDefinition(
        key="agents.working_memory.section_unavailable",
        title="Working Memory: Unavailable Section",
        category="agent_execution",
        loader=load_working_memory_section_unavailable,
        metadata={"notes": (
            "Working-memory section used when storage failed at render time (TASK-1281, "
            "8.1). It must say the list could not be loaded, never that it is empty. "
            "Bot overrides supported."
        )},
    ),
    "agents.working_memory.notice": PromptDefinition(
        key="agents.working_memory.notice",
        title="Working Memory: Change Notice",
        category="agent_execution",
        loader=load_working_memory_notice,
        metadata={"notes": (
            "One line appended to the outbound user message of the next started turn "
            "after working memory changed outside this session (TASK-1281, 9). Never "
            "stored in chat history. No item content. Bot overrides supported."
        )},
    ),
    "agents.working_memory.scheduled_note": PromptDefinition(
        key="agents.working_memory.scheduled_note",
        title="Working Memory: Scheduled Turn Note",
        category="agent_execution",
        loader=load_working_memory_scheduled_note,
        metadata={"notes": (
            "Added to the outbound message of a scheduler turn for an opted-in bot "
            "(TASK-1281, OI-R5): scheduled automation should not act on working memory. "
            "An instruction, not an execution ban. Bot overrides supported."
        )},
    ),
    "agents.injected_skills.preamble": PromptDefinition(
        key="agents.injected_skills.preamble",
        title="Injected Skills: Preamble",
        category="agent_execution",
        loader=load_injected_skills_preamble,
        metadata={"notes": (
            "Heading of the injected-skills section (TASK-1309), shown above the "
            "skill bodies assigned to an agent bot in 'inject' mode. The section "
            "is omitted when the bot has no injected skills. Bot overrides supported."
        )},
    ),
    "agents.injected_skills.entry": PromptDefinition(
        key="agents.injected_skills.entry",
        title="Injected Skills: Entry",
        category="agent_execution",
        required_vars=("body",),
        loader=load_injected_skills_entry,
        metadata={"notes": (
            "One injected skill (TASK-1309). Variables: {ref} (skill name), "
            "{base_dir} (directory its relative paths resolve against) and {body} "
            "(SKILL.md without frontmatter; required, or the section is dropped). "
            "Filled by plain substitution, so other braces are literal. Bot "
            "overrides supported."
        )},
    ),
    "setup.first_agent_prompt": PromptDefinition(
        key="setup.first_agent_prompt",
        title="Setup: First Agent Prompt",
        category="setup",
        loader=load_setup_first_agent_prompt,
        metadata={"notes": (
            "Starting system prompt the setup wizard offers for the tenant's first "
            "bot, a claude-code agent (TASK-1293/1297). Served by GET /v1/setup/first-bot "
            "as default_system_prompt; the operator can edit it before creating the bot, "
            "and the bot then owns its own copy. Keep it owner-neutral."
        )},
    ),
    "media.prompt_expansion.video": PromptDefinition(
        key="media.prompt_expansion.video",
        title="Video Prompt Expansion",
        category="media",
        required_vars=("prompt", "duration", "aspect_ratio"),
        loader=load_media_prompt_expansion_video,
        metadata={
            "notes": (
                "Studio's magic button (TASK-958) sends this to the global "
                "maintenance_model to turn a short idea into a detailed "
                "text-to-video prompt. {prompt} is the user's text, {duration} "
                "the clip seconds, {aspect_ratio} e.g. 16:9."
            ),
        },
    ),
}
