"""History route facade.

Implementation is split by responsibility so this module remains the stable
router and import surface used by the service and existing callers.
"""

from fastapi import APIRouter

from .history_pages import (
    _clear_history_direct,
    clear_history,
    delete_message,
    get_history,
    get_history_around,
    mutation_router,
    read_router,
)
from .history_search import (
    router as search_router,
    search_all_history,
    search_history,
    search_history_get,
)
from .history_timeline import (
    get_history_timeline,
    get_history_timeline_prompts,
    router as timeline_router,
)
from .history_seed import (
    build_context_seed,
    get_context_seed,
    maybe_build_session_seed,
    router as seed_router,
)
from .history_summaries import (
    delete_router as summary_delete_router,
    delete_summary,
    list_summaries,
    management_router as summary_management_router,
    preview_summarizable_sessions,
    rebuild_history_summaries,
    summarize_history,
)

router = APIRouter()

# Preserve the original method/path registration order. Besides keeping OpenAPI
# output stable, this makes static routes visibly precede future dynamic peers.
router.include_router(read_router)
router.include_router(timeline_router)
router.include_router(search_router)
router.include_router(mutation_router)
router.include_router(summary_management_router)
router.include_router(seed_router)
router.include_router(summary_delete_router)

__all__ = [
    "router",
    "build_context_seed",
    "maybe_build_session_seed",
    "get_context_seed",
    "get_history",
    "get_history_around",
    "get_history_timeline",
    "get_history_timeline_prompts",
    "search_history",
    "search_history_get",
    "search_all_history",
    "clear_history",
    "delete_message",
    "preview_summarizable_sessions",
    "summarize_history",
    "rebuild_history_summaries",
    "list_summaries",
    "delete_summary",
    "_clear_history_direct",
]
