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
    TTS_OUTPUT_INSTRUCTIONS,
)
from .prompt_default_loaders import (
    load_agents_docs,
    load_agents_project_plan,
    load_agents_review,
    load_agents_task_execution,
    load_agents_task_spec,
    load_global_recall_guidance,
    load_history_summarization_batch,
    load_history_summarization_single,
    load_media_prompt_expansion_video,
    load_memory_extraction_fact,
    load_memory_extraction_summary,
    load_memory_extraction_update,
    load_memory_maintenance_intent_content_only,
    load_memory_maintenance_intent_with_context,
    load_profile_consolidation,
    load_self_recap_system,
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
                "(Claude Code, Codex, OpenClaw) ONLY when tts_mode=true. "
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
                "(Claude Code, Codex, OpenClaw) when the request flag "
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
                "Codex, OpenClaw). Injected ONLY when the per-bot runtime setting "
                "`agent_global_prompt_enabled` is true. Added to the cacheable "
                "system-prompt prefix (agent-only; chat bots never see it). "
                "Default body steers planning away from harness plan mode and "
                "toward the BawtHub task system for observability. Supports "
                "per-bot overrides via scope_type=bot; leave body empty to "
                "inject nothing even when the flag is on."
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
