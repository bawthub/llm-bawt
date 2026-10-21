"""Raw LLM utility completion routes."""

from fastapi import APIRouter

from ..schemas import RawCompletionRequest, RawCompletionResponse
# Preserve the existing import seam for in-process callers.
from ..utility_completion import complete_utility as complete_utility
from .recall_completion import router as recall_completion_router

router = APIRouter()
router.include_router(recall_completion_router)


@router.post("/v1/llm/complete", response_model=RawCompletionResponse, tags=["LLM"])
def raw_completion(request: RawCompletionRequest):
    """One-shot completion without bot identity, history, tools or agent execution."""
    return complete_utility(request)
