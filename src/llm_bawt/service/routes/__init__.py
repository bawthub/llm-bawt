"""Route router collection for app registration."""

from .botchat import router as botchat_router
from .chat import router as chat_router
from .config import router as config_router
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
from .openclaw_ws import router as openclaw_ws_router
from .nextcloud import router as nextcloud_router
from .nextcloud_setup import router as nextcloud_setup_router
from .prompts import router as prompts_router
from .profiles import router as profiles_router
from .providers import router as providers_router
from .integrations import router as integrations_router
from .settings import router as settings_router
from .setup import router as setup_router
from .sessions import router as sessions_router
from .session_continuity import router as session_continuity_router
from .approval_policies import router as approval_policies_router
from .ops import router as ops_router
from .tasks import router as tasks_router
from .ha_weather import router as ha_weather_router
from .turn_logs import router as turn_logs_router
from .workspace_provenance import router as workspace_provenance_router
from .inter_bot_deliveries import router as inter_bot_deliveries_router
from .agent_context import router as agent_context_router
from .usage import router as usage_router
# TASK-214: avatar_animations CRUD moved to bawthub Prisma.
# /v1/avatar/animations routes were deleted; admin lives at
# bawthub's /api/avatar/animation-triggers now.
from .media import router as media_router
from .uploads import router as uploads_router

all_routers = [
    health_router,
    google_home_probe_router,
    ha_weather_router,
    nextcloud_router,
    models_router,
    model_catalog_router,
    openclaw_ws_router,
    botchat_router,
    chat_router,
    tasks_router,
    turn_logs_router,
    workspace_provenance_router,
    inter_bot_deliveries_router,
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
    setup_router,
    sessions_router,
    session_continuity_router,
    config_router,
    approval_policies_router,
    ops_router,
    profiles_router,
    llm_router,
    media_router,
    uploads_router,
]

__all__ = ["all_routers"]
