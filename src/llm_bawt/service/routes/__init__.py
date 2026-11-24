"""Route router collection for app registration."""

from importlib import import_module

from ...public_text_profile import capability_present
from .admin_gates import router as admin_gates_router
from .auth_observations import router as auth_observations_router
from .bawthub_auth_secrets import router as bawthub_auth_secrets_router
from .internal_connections import router as internal_connections_router
from .bot_grants import router as bot_grants_router
from .houses import router as houses_router
from .bot_tags import router as bot_tags_router
from .botchat import router as botchat_router
from .chat import router as chat_router
from .config import router as config_router
from .event_stream import router as event_stream_router
from .system_events import router as system_events_router
from .health import router as health_router
from .google_home_probe import router as google_home_probe_router
from .history import router as history_router
from .jobs import router as jobs_router
from .prompt_schedules import router as prompt_schedules_router
from .message_routing import router as message_routing_router
from .llm import router as llm_router
from .memory import router as memory_router
from .models import router as models_router
from .model_catalog import router as model_catalog_router
from .nextcloud import router as nextcloud_router
from .nextcloud_setup import router as nextcloud_setup_router
from .prompts import router as prompts_router
from .profiles import router as profiles_router
from .providers import router as providers_router
from .integrations import router as integrations_router
from .settings import router as settings_router
from .bot_purge_preview import router as bot_purge_preview_router
from .turn_tasks import router as turn_tasks_router
from .task_retros import router as task_retros_router
from .setup import router as setup_router
from .setup_presets import router as setup_presets_router
from .sessions import router as sessions_router
from .skills import router as skills_router
from .session_continuity import router as session_continuity_router
from .approval_policies import router as approval_policies_router
from .content_provenance import router as content_provenance_router
from .ops import router as ops_router
from .ops_approvals import router as ops_approvals_router
from .tasks import router as tasks_router
from .task_events import router as task_events_router
from .ha_weather import router as ha_weather_router
from .turn_logs import router as turn_logs_router
from .recall_evidence import router as recall_evidence_router
from .workspace_provenance import router as workspace_provenance_router
from .inter_bot_deliveries import router as inter_bot_deliveries_router
from .bridge_inbox import router as bridge_inbox_router
from .queue_hold import router as queue_hold_router
from .bot_availability import router as bot_availability_router
from .peers import router as peers_router
from .agent_context import router as agent_context_router
from .usage import router as usage_router
# TASK-214: avatar_animations CRUD moved to bawthub Prisma.
# /v1/avatar/animations routes were deleted; admin lives at
# bawthub's /api/avatar/animation-triggers now.
from .uploads import router as uploads_router
from .working_memory import router as working_memory_router
from .workspace import router as workspace_router
from .campus import router as campus_router
from .playground import router as playground_router
from .guidance import router as guidance_router


def _optional_routers(name: str) -> list:
    """The router of ``routes.<name>`` when its source ships (public text profile omits it)."""
    module = f"{__name__}.{name}"
    return [import_module(module).router] if capability_present(module) else []


all_routers = [
    health_router,
    google_home_probe_router,
    ha_weather_router,
    nextcloud_router,
    models_router,
    model_catalog_router,
    event_stream_router,
    system_events_router,
    botchat_router,
    chat_router,
    tasks_router,
    task_events_router,
    turn_logs_router,
    recall_evidence_router,
    workspace_provenance_router,
    inter_bot_deliveries_router,
    bridge_inbox_router,
    queue_hold_router,
    bot_availability_router,
    peers_router,
    agent_context_router,
    usage_router,
    jobs_router,
    prompt_schedules_router,
    message_routing_router,
    history_router,
    memory_router,
    prompts_router,
    providers_router,
    nextcloud_setup_router,  # Specific setup routes precede generic integration paths.
    integrations_router,
    settings_router,
    bot_purge_preview_router,
    turn_tasks_router,
    task_retros_router,
    bot_tags_router,
    bot_grants_router,
    houses_router,
    skills_router,
    setup_router,
    setup_presets_router,
    sessions_router,
    session_continuity_router,
    config_router,
    approval_policies_router,
    content_provenance_router,
    ops_router,
    ops_approvals_router,
    profiles_router,
    llm_router,
    *_optional_routers("media"),
    uploads_router,
    *_optional_routers("response_audio"),
    working_memory_router,
    workspace_router,
    campus_router,
    playground_router,
    guidance_router,
    bawthub_auth_secrets_router,
    internal_connections_router,
    admin_gates_router,
    auth_observations_router,
]

__all__ = ["all_routers"]
