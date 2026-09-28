"""DB-backed prompt templates with code-default fallback."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from string import Formatter
from typing import Any, Callable
from urllib.parse import quote_plus

from sqlalchemy import Column, DateTime, String, Text, UniqueConstraint
from sqlmodel import Field, SQLModel, Session, create_engine, select

from .chat_prompt_defaults import (
    AGENT_GLOBAL_PROMPT,
    AGENT_USER_PREFIX,
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
from .utils.config import Config, has_database_credentials
from .utils.schema import SchemaBootstrapGuard

logger = logging.getLogger(__name__)

#: Category used to group switchable persona prompts in the registry, keeping
#: them separate from summarization/memory/agent/chat prompts. See TASK-477.
PERSONA_CATEGORY = "persona"
#: Key prefix for persona rows (e.g. ``persona.grumpy-editor``).
PERSONA_KEY_PREFIX = "persona."


def _slugify(text: str) -> str:
    """Lowercase, hyphenated slug from arbitrary text (ASCII word chars only)."""
    import re

    slug = re.sub(r"[^a-z0-9]+", "-", (text or "").strip().lower()).strip("-")
    return slug or "persona"


class PromptTemplate(SQLModel, table=True):
    """Stored active prompt template."""

    __tablename__ = "prompt_templates"
    __table_args__ = (
        UniqueConstraint("key", "scope_type", "scope_id", name="uq_prompt_templates_scope_key"),
    )

    id: int | None = Field(default=None, primary_key=True)
    key: str = Field(sa_column=Column(String(255), nullable=False, index=True))
    scope_type: str = Field(sa_column=Column(String(16), nullable=False, index=True))
    scope_id: str = Field(sa_column=Column(String(128), nullable=False, index=True))
    title: str | None = Field(default=None, sa_column=Column(String(255), nullable=True))
    category: str | None = Field(default=None, sa_column=Column(String(64), nullable=True, index=True))
    format: str = Field(default="plain_text", sa_column=Column(String(32), nullable=False))
    body: str = Field(sa_column=Column(Text, nullable=False))
    required_vars_json: str | None = Field(default=None, sa_column=Column(Text, nullable=True))
    metadata_json: str | None = Field(default=None, sa_column=Column(Text, nullable=True))
    updated_by: str | None = Field(default=None, sa_column=Column(String(255), nullable=True))
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )


class PromptTemplateVersion(SQLModel, table=True):
    """Append-only prompt version history."""

    __tablename__ = "prompt_template_versions"

    id: int | None = Field(default=None, primary_key=True)
    key: str = Field(sa_column=Column(String(255), nullable=False, index=True))
    scope_type: str = Field(sa_column=Column(String(16), nullable=False, index=True))
    scope_id: str = Field(sa_column=Column(String(128), nullable=False, index=True))
    version: int = Field(default=1)
    body: str = Field(sa_column=Column(Text, nullable=False))
    change_note: str | None = Field(default=None, sa_column=Column(Text, nullable=True))
    created_by: str | None = Field(default=None, sa_column=Column(String(255), nullable=True))
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )


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


@dataclass(frozen=True)
class ResolvedPrompt:
    """Resolved prompt template payload."""

    key: str
    title: str
    category: str
    format: str
    body: str
    required_vars: list[str]
    metadata: dict[str, Any]
    scope_type: str
    scope_id: str
    source: str
    updated_at: datetime | None = None


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


def _normalize_scope(scope_type: str = "global", scope_id: str | None = None) -> tuple[str, str]:
    normalized_type = (scope_type or "global").strip().lower() or "global"
    if normalized_type not in {"global", "bot"}:
        raise ValueError("scope_type must be 'global' or 'bot'")
    if normalized_type == "global":
        return ("global", "*")
    normalized_id = (scope_id or "").strip().lower()
    if not normalized_id:
        raise ValueError("scope_id is required for bot-scoped prompts")
    return ("bot", normalized_id)


def _parse_json_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    return [str(item) for item in data if str(item).strip()]


def _parse_json_dict(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def extract_placeholders(body: str) -> list[str]:
    """Extract format placeholders from a prompt template."""
    names: list[str] = []
    for _, field_name, _, _ in Formatter().parse(body or ""):
        if not field_name:
            continue
        base_name = field_name.split("!", 1)[0].split(":", 1)[0]
        if base_name and base_name not in names:
            names.append(base_name)
    return names


class PromptTemplateStore:
    """Database access for prompt templates."""

    _seeded_urls: set[str] = set()
    _schema_guard = SchemaBootstrapGuard()

    def __init__(self, config: Config):
        self.config = config
        self.engine = None
        self.connection_url: str | None = None
        if not has_database_credentials(config):
            return
        try:
            from .utils.db import build_postgres_url, get_shared_engine
            self.connection_url = build_postgres_url(config)
            # Process-wide shared engine (TASK-202).
            self.engine = get_shared_engine(config)
            if self.engine is None:
                return
            self._ensure_tables_exist()
        except Exception as e:
            self.engine = None
            logger.warning("Prompt template DB unavailable: %s", e)
            return

        try:
            self._seed_defaults_once()
        except Exception as e:
            logger.warning("Prompt template default seeding skipped: %s", e)

    def _ensure_tables_exist(self) -> None:
        if self.engine is None:
            return
        def bootstrap(conn) -> None:
            SQLModel.metadata.create_all(
                bind=conn,
                tables=[
                    PromptTemplate.__table__,
                    PromptTemplateVersion.__table__,
                ],
            )

        self._schema_guard.run(self.engine, "prompt-template-store", bootstrap)

    def _seed_defaults_once(self) -> None:
        if self.engine is None or not self.connection_url:
            return
        if self.connection_url in self._seeded_urls:
            return
        result = self.seed_defaults()
        self._seeded_urls.add(self.connection_url)
        if result["created"] > 0:
            logger.info("Seeded %s default prompt templates into DB", result["created"])

    def seed_defaults(self) -> dict[str, Any]:
        """Insert built-in prompt defaults into DB when missing."""
        if self.engine is None:
            raise RuntimeError("Prompt template DB unavailable")

        definitions = DEFAULT_PROMPT_DEFINITIONS
        global_scope_type = "global"
        global_scope_id = "*"
        existing_keys: set[str] = set()

        with Session(self.engine) as session:
            statement = select(PromptTemplate.key).where(
                PromptTemplate.scope_type == global_scope_type,
                PromptTemplate.scope_id == global_scope_id,
            )
            existing_keys = {str(key) for key in session.exec(statement).all()}

            now = datetime.now(timezone.utc)
            created = 0
            skipped = 0
            seeded_keys: list[str] = []

            for key, definition in definitions.items():
                if key in existing_keys:
                    skipped += 1
                    continue

                body = definition.default_body()
                metadata = {"seeded_from": "code_default"}

                session.add(
                    PromptTemplate(
                        key=key,
                        scope_type=global_scope_type,
                        scope_id=global_scope_id,
                        title=definition.title,
                        category=definition.category,
                        format=definition.format,
                        body=body,
                        required_vars_json=json.dumps(list(definition.required_vars))
                        if definition.required_vars
                        else None,
                        metadata_json=json.dumps(metadata),
                        updated_by="system:seed",
                        created_at=now,
                        updated_at=now,
                    )
                )
                session.add(
                    PromptTemplateVersion(
                        key=key,
                        scope_type=global_scope_type,
                        scope_id=global_scope_id,
                        version=1,
                        body=body,
                        change_note="Seeded from code default",
                        created_by="system:seed",
                        created_at=now,
                    )
                )
                created += 1
                seeded_keys.append(key)

            if created > 0:
                session.commit()

        total = len(definitions)
        return {
            "created": created,
            "skipped": skipped,
            "total": total,
            "seeded_keys": seeded_keys,
        }

    def get_exact(self, key: str, scope_type: str = "global", scope_id: str | None = None) -> PromptTemplate | None:
        if self.engine is None:
            return None
        normalized_type, normalized_id = _normalize_scope(scope_type, scope_id)
        with Session(self.engine) as session:
            return session.exec(
                select(PromptTemplate).where(
                    PromptTemplate.key == key,
                    PromptTemplate.scope_type == normalized_type,
                    PromptTemplate.scope_id == normalized_id,
                )
            ).first()

    def list_rows(
        self,
        *,
        category: str | None = None,
        scope_type: str | None = None,
        scope_id: str | None = None,
    ) -> list[PromptTemplate]:
        if self.engine is None:
            return []
        statement = select(PromptTemplate)
        if category:
            statement = statement.where(PromptTemplate.category == category.strip().lower())
        if scope_type:
            normalized_type, normalized_id = _normalize_scope(scope_type, scope_id)
            statement = statement.where(
                PromptTemplate.scope_type == normalized_type,
                PromptTemplate.scope_id == normalized_id,
            )
        statement = statement.order_by(PromptTemplate.category, PromptTemplate.key, PromptTemplate.scope_type, PromptTemplate.scope_id)
        with Session(self.engine) as session:
            return list(session.exec(statement).all())

    def upsert(
        self,
        *,
        key: str,
        body: str,
        scope_type: str = "global",
        scope_id: str | None = None,
        title: str | None = None,
        category: str | None = None,
        format: str = "plain_text",
        required_vars: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        updated_by: str | None = None,
        change_note: str | None = None,
    ) -> PromptTemplate:
        if self.engine is None:
            raise RuntimeError("Prompt template DB unavailable")
        normalized_type, normalized_id = _normalize_scope(scope_type, scope_id)
        required_vars = [str(item) for item in (required_vars or []) if str(item).strip()]
        metadata = metadata or {}
        now = datetime.now(timezone.utc)
        with Session(self.engine) as session:
            row = session.exec(
                select(PromptTemplate).where(
                    PromptTemplate.key == key,
                    PromptTemplate.scope_type == normalized_type,
                    PromptTemplate.scope_id == normalized_id,
                )
            ).first()
            version_statement = select(PromptTemplateVersion).where(
                PromptTemplateVersion.key == key,
                PromptTemplateVersion.scope_type == normalized_type,
                PromptTemplateVersion.scope_id == normalized_id,
            )
            existing_versions = list(session.exec(version_statement).all())
            next_version = (max((v.version for v in existing_versions), default=0) + 1)

            if row is None:
                row = PromptTemplate(
                    key=key,
                    scope_type=normalized_type,
                    scope_id=normalized_id,
                    title=title,
                    category=category,
                    format=format,
                    body=body,
                    required_vars_json=json.dumps(required_vars) if required_vars else None,
                    metadata_json=json.dumps(metadata) if metadata else None,
                    updated_by=updated_by,
                    created_at=now,
                    updated_at=now,
                )
            else:
                row.title = title
                row.category = category
                row.format = format
                row.body = body
                row.required_vars_json = json.dumps(required_vars) if required_vars else None
                row.metadata_json = json.dumps(metadata) if metadata else None
                row.updated_by = updated_by
                row.updated_at = now

            session.add(row)
            session.add(
                PromptTemplateVersion(
                    key=key,
                    scope_type=normalized_type,
                    scope_id=normalized_id,
                    version=next_version,
                    body=body,
                    change_note=change_note,
                    created_by=updated_by,
                    created_at=now,
                )
            )
            session.commit()
            session.refresh(row)
            return row

    def reset(self, key: str, scope_type: str = "global", scope_id: str | None = None) -> bool:
        if self.engine is None:
            raise RuntimeError("Prompt template DB unavailable")
        normalized_type, normalized_id = _normalize_scope(scope_type, scope_id)
        with Session(self.engine) as session:
            row = session.exec(
                select(PromptTemplate).where(
                    PromptTemplate.key == key,
                    PromptTemplate.scope_type == normalized_type,
                    PromptTemplate.scope_id == normalized_id,
                )
            ).first()
            if row is None:
                return False
            session.delete(row)
            session.commit()
            return True

    def list_versions(self, key: str, scope_type: str = "global", scope_id: str | None = None) -> list[PromptTemplateVersion]:
        if self.engine is None:
            return []
        normalized_type, normalized_id = _normalize_scope(scope_type, scope_id)
        with Session(self.engine) as session:
            statement = (
                select(PromptTemplateVersion)
                .where(
                    PromptTemplateVersion.key == key,
                    PromptTemplateVersion.scope_type == normalized_type,
                    PromptTemplateVersion.scope_id == normalized_id,
                )
                .order_by(PromptTemplateVersion.version.desc())
            )
            return list(session.exec(statement).all())

    # ── Persona helpers (TASK-477) ─────────────────────────────────────
    # Personas are ordinary prompt_templates rows with category='persona'
    # and global scope, so they inherit versioning + resolution for free.
    # They are shareable across bots; a bot points at one via
    # bot_profiles.prompt_override_id (== PromptTemplate.id).

    def get_by_id(self, template_id: int) -> PromptTemplate | None:
        if self.engine is None or template_id is None:
            return None
        with Session(self.engine) as session:
            return session.get(PromptTemplate, template_id)

    def list_personas(self) -> list[PromptTemplate]:
        """All global persona prompts, ordered by title/key."""
        rows = self.list_rows(category=PERSONA_CATEGORY, scope_type="global")
        return sorted(rows, key=lambda r: ((r.title or r.key or "").lower(), r.id or 0))

    def create_persona(
        self,
        *,
        title: str,
        body: str,
        updated_by: str | None = None,
    ) -> PromptTemplate:
        """Create a new global persona, generating a unique persona.<slug> key."""
        if self.engine is None:
            raise RuntimeError("Prompt template DB unavailable")
        base_slug = _slugify(title)
        existing_keys = {r.key for r in self.list_rows(category=PERSONA_CATEGORY, scope_type="global")}
        key = f"{PERSONA_KEY_PREFIX}{base_slug}"
        suffix = 2
        while key in existing_keys:
            key = f"{PERSONA_KEY_PREFIX}{base_slug}-{suffix}"
            suffix += 1
        return self.upsert(
            key=key,
            body=body,
            scope_type="global",
            title=title.strip() or key,
            category=PERSONA_CATEGORY,
            updated_by=updated_by,
            change_note="Persona created",
        )

    def update_persona(
        self,
        template_id: int,
        *,
        body: str | None = None,
        title: str | None = None,
        updated_by: str | None = None,
    ) -> PromptTemplate | None:
        """Edit an existing persona by id; bumps a new version row."""
        if self.engine is None:
            raise RuntimeError("Prompt template DB unavailable")
        row = self.get_by_id(template_id)
        if row is None or row.category != PERSONA_CATEGORY:
            return None
        return self.upsert(
            key=row.key,
            body=body if body is not None else row.body,
            scope_type=row.scope_type,
            scope_id=row.scope_id,
            title=title.strip() if title is not None else row.title,
            category=PERSONA_CATEGORY,
            format=row.format,
            updated_by=updated_by,
            change_note="Persona edited",
        )

    def delete_persona(self, template_id: int) -> bool:
        """Delete a persona by id (its version history is retained)."""
        if self.engine is None:
            raise RuntimeError("Prompt template DB unavailable")
        row = self.get_by_id(template_id)
        if row is None or row.category != PERSONA_CATEGORY:
            return False
        return self.reset(row.key, row.scope_type, row.scope_id)


class PromptResolver:
    """Resolve prompt templates from DB with code-default fallback."""

    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self.store = PromptTemplateStore(self.config)

    @staticmethod
    def definitions() -> dict[str, PromptDefinition]:
        return DEFAULT_PROMPT_DEFINITIONS.copy()

    @staticmethod
    def definition_for(key: str) -> PromptDefinition | None:
        return DEFAULT_PROMPT_DEFINITIONS.get(key)

    def resolve(self, key: str, scope_type: str = "global", scope_id: str | None = None) -> ResolvedPrompt | None:
        normalized_type, normalized_id = _normalize_scope(scope_type, scope_id)

        if self.store.engine is not None:
            row = self.store.get_exact(key, normalized_type, normalized_id)
            if row is None and normalized_type != "global":
                row = self.store.get_exact(key, "global", "*")
            if row is not None:
                definition = self.definition_for(key)
                return ResolvedPrompt(
                    key=key,
                    title=row.title or (definition.title if definition else key),
                    category=row.category or (definition.category if definition else "custom"),
                    format=row.format or (definition.format if definition else "plain_text"),
                    body=row.body,
                    required_vars=_parse_json_list(row.required_vars_json) or list(definition.required_vars if definition else ()),
                    metadata=_parse_json_dict(row.metadata_json) or dict(definition.metadata if definition else {}),
                    scope_type=row.scope_type,
                    scope_id=row.scope_id,
                    source="db_override",
                    updated_at=row.updated_at,
                )

        definition = self.definition_for(key)
        if definition is None:
            return None
        return ResolvedPrompt(
            key=key,
            title=definition.title,
            category=definition.category,
            format=definition.format,
            body=definition.default_body(),
            required_vars=list(definition.required_vars),
            metadata=dict(definition.metadata),
            scope_type="global",
            scope_id="*",
            source="code_default",
            updated_at=None,
        )

    def validate(
        self,
        *,
        key: str,
        body: str,
        required_vars: list[str] | None = None,
    ) -> dict[str, Any]:
        definition = self.definition_for(key)
        normalized_required = [str(item) for item in (required_vars or []) if str(item).strip()]
        if not normalized_required and definition is not None:
            normalized_required = list(definition.required_vars)

        placeholders = extract_placeholders(body)
        missing_required = [name for name in normalized_required if name not in placeholders]
        unknown_placeholders = [name for name in placeholders if normalized_required and name not in normalized_required]

        errors: list[str] = []
        if not (body or "").strip():
            errors.append("Prompt body cannot be empty")
        if missing_required:
            errors.append(f"Missing required placeholders: {', '.join(missing_required)}")
        if unknown_placeholders and definition is not None:
            errors.append(f"Unknown placeholders for '{key}': {', '.join(unknown_placeholders)}")

        return {
            "valid": not errors,
            "placeholders": placeholders,
            "required_vars": normalized_required,
            "missing_required": missing_required,
            "unknown_placeholders": unknown_placeholders,
            "errors": errors,
        }

    def render(
        self,
        *,
        key: str,
        variables: dict[str, Any],
        scope_type: str = "global",
        scope_id: str | None = None,
        body_override: str | None = None,
    ) -> str:
        definition = self.definition_for(key)
        body = body_override
        source = "override"
        if body is None:
            resolved = self.resolve(key, scope_type, scope_id)
            if resolved is None:
                raise KeyError(f"Unknown prompt key '{key}'")
            body = resolved.body
            source = resolved.source

        try:
            return body.format(**variables)
        except KeyError:
            if source != "code_default" and definition is not None and body_override is None:
                logger.warning("Prompt '%s' DB override failed to render; falling back to code default", key)
                return definition.default_body().format(**variables)
            raise


_DEFAULT_PROMPT_RESOLVER: PromptResolver | None = None


def get_prompt_resolver(config: Config | None = None) -> PromptResolver:
    """Return a prompt resolver, caching the configless default path."""
    global _DEFAULT_PROMPT_RESOLVER
    if config is None:
        if _DEFAULT_PROMPT_RESOLVER is None:
            _DEFAULT_PROMPT_RESOLVER = PromptResolver()
        return _DEFAULT_PROMPT_RESOLVER
    return PromptResolver(config)
