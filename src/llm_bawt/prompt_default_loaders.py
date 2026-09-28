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


def load_media_prompt_expansion_video() -> str:
    from .media.prompt_expansion import VIDEO_PROMPT_EXPANSION_TEMPLATE

    return VIDEO_PROMPT_EXPANSION_TEMPLATE
