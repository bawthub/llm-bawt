"""Raw LLM utility completion route."""

from fastapi import APIRouter, HTTPException

from ..dependencies import get_service
from ..logging import get_service_logger
from ..schemas import RawCompletionRequest, RawCompletionResponse

router = APIRouter()
log = get_service_logger(__name__)

@router.post("/v1/llm/complete", response_model=RawCompletionResponse, tags=["LLM"])
def raw_completion(request: RawCompletionRequest):
    """One-shot model completion without bot identity, history, tools or agent execution.

    An explicit model selects a direct API client from the catalog. Without a
    model, the global maintenance_model setting selects the job's API client.
    """
    return complete_utility(request)


def complete_utility(request: RawCompletionRequest) -> RawCompletionResponse:
    """Run one botless utility completion; the single implementation behind
    ``/v1/llm/complete`` and in-process callers such as media prompt expansion.

    Raises HTTPException: 503 when no maintenance model is configured, 422 when
    the model has no direct API client, 500 when the model call fails.
    """
    import time
    service = get_service()

    from ...runtime_settings import resolve_job_model

    # The global maintenance_model is the source of truth for utility calls.
    # Never pick a cached model by insertion order or borrow a bot's harness.
    requested_model = request.model or resolve_job_model(service.config, "maintenance_model")
    if not requested_model:
        raise HTTPException(status_code=503, detail="No maintenance model configured")
    client, model_alias = service._get_background_client(model_override=requested_model)
    if client is None or model_alias is None:
        raise HTTPException(status_code=422, detail="Configured model has no direct API client")

    try:
        start = time.perf_counter()

        from ...models.message import Message
        messages = []
        if request.system:
            messages.append(Message(role="system", content=request.system))
        messages.append(Message(role="user", content=request.prompt))

        response = client.query(
            messages=messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            plaintext_output=True,
            stream=False,
        )
        if not isinstance(response, str) or not response.strip() or response.lstrip().startswith("ERROR:"):
            raise ValueError("Model returned no usable completion")

        elapsed_ms = (time.perf_counter() - start) * 1000
        tokens = len(response) // 4 if response else 0

        return RawCompletionResponse(
            content=response,
            model=model_alias,
            tokens=tokens,
            elapsed_ms=elapsed_ms,
        )

    except Exception as e:
        log.error(f"Raw completion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =========================================================================
# History Summarization Endpoints
# =========================================================================
