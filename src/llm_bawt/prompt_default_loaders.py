"""Lazy loaders for built-in prompt bodies with dependency-heavy owners."""


def load_history_summarization_single() -> str:
    from .memory.summarization import SUMMARIZATION_PROMPT

    return SUMMARIZATION_PROMPT


def load_history_summarization_batch() -> str:
    from .memory.summarization import BATCH_SUMMARIZATION_PROMPT

    return BATCH_SUMMARIZATION_PROMPT


def load_memory_extraction_fact() -> str:
    from .memory.extraction.prompts import FACT_EXTRACTION_PROMPT_TEMPLATE

    return FACT_EXTRACTION_PROMPT_TEMPLATE


def load_memory_extraction_update() -> str:
    from .memory.extraction.prompts import MEMORY_UPDATE_PROMPT_TEMPLATE

    return MEMORY_UPDATE_PROMPT_TEMPLATE


def load_memory_extraction_summary() -> str:
    from .memory.extraction.prompts import SUMMARY_EXTRACTION_PROMPT_TEMPLATE

    return SUMMARY_EXTRACTION_PROMPT_TEMPLATE


def load_history_summarization_system() -> str:
    from .memory.summarization import SUMMARY_SYSTEM_PROMPT

    return SUMMARY_SYSTEM_PROMPT


def load_memory_extraction_fact_system() -> str:
    from .memory.extraction.prompts import FACT_EXTRACTION_SYSTEM_PROMPT

    return FACT_EXTRACTION_SYSTEM_PROMPT


def load_memory_extraction_summary_system() -> str:
    from .memory.extraction.prompts import SUMMARY_EXTRACTION_SYSTEM_PROMPT

    return SUMMARY_EXTRACTION_SYSTEM_PROMPT


def load_memory_extraction_update_system() -> str:
    from .memory.extraction.prompts import MEMORY_UPDATE_SYSTEM_PROMPT

    return MEMORY_UPDATE_SYSTEM_PROMPT


def load_profile_consolidation() -> str:
    from .memory.extraction.prompts import PROFILE_CONSOLIDATION_PROMPT

    return PROFILE_CONSOLIDATION_PROMPT


def load_memory_maintenance_intent_with_context() -> str:
    from .memory.maintenance import INTENT_PROMPT_WITH_CONTEXT

    return INTENT_PROMPT_WITH_CONTEXT


def load_memory_maintenance_intent_content_only() -> str:
    from .memory.maintenance import INTENT_PROMPT_CONTENT_ONLY

    return INTENT_PROMPT_CONTENT_ONLY


def load_agents_task_spec() -> str:
    from .agent_backends.prompts import TASK_SPEC_PROMPT

    return TASK_SPEC_PROMPT


def load_global_recall_guidance() -> str:
    from .core.prompt_builder import GLOBAL_SYSTEM_PROMPT

    return GLOBAL_SYSTEM_PROMPT


def load_agents_task_execution() -> str:
    from .agent_backends.prompts import TASK_EXECUTION_PROMPT

    return TASK_EXECUTION_PROMPT


def load_agents_project_plan() -> str:
    from .agent_backends.prompts import PROJECT_PLAN_PROMPT

    return PROJECT_PLAN_PROMPT


def load_agents_review() -> str:
    from .agent_backends.prompts import REVIEW_DISPATCH_PROMPT

    return REVIEW_DISPATCH_PROMPT


def load_agents_docs() -> str:
    from .agent_backends.prompts import AGENTS_DOCS_PROMPT

    return AGENTS_DOCS_PROMPT


def load_self_recap_system() -> str:
    from .mcp_server.recap_prompt import RECAP_SYSTEM_PROMPT

    return RECAP_SYSTEM_PROMPT


def load_recall_synthesis_system() -> str:
    from .recall.prompt import DEFAULT_RECALL_SYSTEM_PROMPT

    return DEFAULT_RECALL_SYSTEM_PROMPT


def load_inter_bot_reply_channel_waited() -> str:
    from .inter_bot_reply_channel import DEFAULT_REPLY_CHANNEL_WAITED

    return DEFAULT_REPLY_CHANNEL_WAITED


def load_inter_bot_reply_channel_async() -> str:
    from .inter_bot_reply_channel import DEFAULT_REPLY_CHANNEL_ASYNC

    return DEFAULT_REPLY_CHANNEL_ASYNC


def load_inter_bot_reply_channel_waited_no_return() -> str:
    from .inter_bot_reply_channel import DEFAULT_REPLY_CHANNEL_WAITED_NO_RETURN

    return DEFAULT_REPLY_CHANNEL_WAITED_NO_RETURN


def load_inter_bot_restart_retry_continue() -> str:
    from .inter_bot_delivery_retry import DEFAULT_RESTART_RETRY_CONTINUE

    return DEFAULT_RESTART_RETRY_CONTINUE


def load_deliveries_cancel_notice() -> str:
    from .service.delivery_cancel_notice_prompt import DEFAULT_CANCEL_NOTICE

    return DEFAULT_CANCEL_NOTICE


def load_approvals_approver_request() -> str:
    from .service.approval_fanout_prompt import DEFAULT_APPROVER_REQUEST

    return DEFAULT_APPROVER_REQUEST


def load_approvals_approver_resolved() -> str:
    from .service.approval_fanout_prompt import DEFAULT_APPROVER_RESOLVED

    return DEFAULT_APPROVER_RESOLVED


def load_ops_quiesce_receipt() -> str:
    from .ops.quiesce_prompts import DEFAULT_RECEIPT

    return DEFAULT_RECEIPT


def load_ops_quiesce_request() -> str:
    from .ops.quiesce_prompts import DEFAULT_QUIESCE_REQUEST

    return DEFAULT_QUIESCE_REQUEST


def load_ops_quiesce_interrupt_frame() -> str:
    from .ops.quiesce_prompts import DEFAULT_QUIESCE_INTERRUPT_FRAME

    return DEFAULT_QUIESCE_INTERRUPT_FRAME


def load_ops_quiesce_end_turn_new_turn() -> str:
    from .ops.quiesce_prompts import DEFAULT_END_TURN_NEW_TURN

    return DEFAULT_END_TURN_NEW_TURN


def load_ops_quiesce_end_turn_no_receipt() -> str:
    from .ops.quiesce_prompts import DEFAULT_END_TURN_NO_RECEIPT

    return DEFAULT_END_TURN_NO_RECEIPT


def load_chat_steer_next_aside() -> str:
    from .steer_prompts import DEFAULT_NEXT_ASIDE

    return DEFAULT_NEXT_ASIDE


def load_chat_steer_interrupt_replacement() -> str:
    from .steer_prompts import DEFAULT_INTERRUPT_REPLACEMENT

    return DEFAULT_INTERRUPT_REPLACEMENT


def load_media_prompt_expansion_video() -> str:
    from .media.prompt_expansion import VIDEO_PROMPT_EXPANSION_TEMPLATE

    return VIDEO_PROMPT_EXPANSION_TEMPLATE


def load_working_memory_section() -> str:
    from .working_memory.prompts import DEFAULT_SECTION

    return DEFAULT_SECTION


def load_working_memory_section_empty() -> str:
    from .working_memory.prompts import DEFAULT_SECTION_EMPTY

    return DEFAULT_SECTION_EMPTY


def load_working_memory_section_unavailable() -> str:
    from .working_memory.prompts import DEFAULT_SECTION_UNAVAILABLE

    return DEFAULT_SECTION_UNAVAILABLE


def load_working_memory_notice() -> str:
    from .working_memory.prompts import DEFAULT_NOTICE

    return DEFAULT_NOTICE


def load_working_memory_scheduled_note() -> str:
    from .working_memory.prompts import DEFAULT_SCHEDULED_NOTE

    return DEFAULT_SCHEDULED_NOTE


def load_injected_skills_preamble() -> str:
    from .skills.prompts import DEFAULT_PREAMBLE

    return DEFAULT_PREAMBLE


def load_injected_skills_entry() -> str:
    from .skills.prompts import DEFAULT_ENTRY

    return DEFAULT_ENTRY


def load_setup_first_agent_prompt() -> str:
    from .seeding.data import FIRST_AGENT_PROMPT_SEED

    return FIRST_AGENT_PROMPT_SEED
