"""Stateless utility completions shared by HTTP routes and in-process callers."""
from __future__ import annotations

import time

from fastapi import HTTPException

from .dependencies import get_service
from .logging import get_service_logger
from .schemas import RawCompletionRequest, RawCompletionResponse

log = get_service_logger(__name__)


def validate_model_budget(request: RawCompletionRequest, config, model_alias: str) -> None:
    """Catalog-driven preflight with conservative provider-framing headroom.

    o200k is an estimate for non-OpenAI models, not their exact tokenizer.
    Provider rejection is terminal, never a retry with a larger budget.
    """
    from ..recall.budgets import tokens

    try:
        window = config.get_model_context_window(model_alias)
        output_cap = config.get_model_max_tokens(model_alias)
        if type(window) is not int or window <= 0 or type(output_cap) is not int or output_cap <= 0:
            raise ValueError("missing model budget")
        needed = tokens(request.system or "") + tokens(request.prompt) + request.max_tokens + 256
        if needed > window or request.max_tokens > output_cap:
            raise ValueError("model budget exceeded")
    except Exception:
        raise HTTPException(status_code=422, detail="recall_model_budget_unavailable") from None


def complete_utility(
    request: RawCompletionRequest, *, timeout_seconds: float | None = None,
) -> RawCompletionResponse:
    """Complete explicit input without a bot, transcript, tools or agent fallback.

    Ordinary utility behavior is preserved. The bounded contract additionally
    requires a transport-bounded client, disables retries, validates model budgets
    and sanitizes failures. Model selection is the global maintenance setting
    unless an ordinary utility caller explicitly supplies a catalog model.
    """
    from ..runtime_settings import resolve_job_model

    service = get_service()
    requested_model = request.model or resolve_job_model(service.config, "maintenance_model")
    if not requested_model:
        raise HTTPException(status_code=503, detail="No maintenance model configured")
    client, model_alias = service._get_background_client(
        model_override=requested_model,
        job="recall_completion" if timeout_seconds is not None else "utility_completion",
    )
    if client is None or model_alias is None:
        raise HTTPException(status_code=422, detail="Configured model has no direct API client")
    if timeout_seconds is not None:
        validate_model_budget(request, service.config, model_alias)

    try:
        start = time.perf_counter()
        from ..models.message import Message

        messages = []
        if request.system:
            messages.append(Message(role="system", content=request.system))
        image_parts = None
        if request.image_url:
            from ..media.prompt_expansion import validate_reference_image
            image_parts = [{"type": "image_url", "image_url": {
                "url": validate_reference_image(request.image_url), "detail": "low",
            }}]
        messages.append(Message(role="user", content=request.prompt, content_parts=image_parts))
        if timeout_seconds is None:
            response = client.query(
                messages=messages, max_tokens=request.max_tokens, temperature=request.temperature,
                plaintext_output=True, stream=False,
            )
        else:
            response = client.query_bounded(
                messages=messages, timeout_seconds=timeout_seconds,
                max_tokens=request.max_tokens, temperature=request.temperature,
            )
        if not isinstance(response, str) or not response.strip() or response.lstrip().startswith("ERROR:"):
            raise ValueError("Model returned no usable completion")
        return RawCompletionResponse(
            content=response, model=model_alias,
            tokens=len(response) // 4, elapsed_ms=(time.perf_counter() - start) * 1000,
        )
    except Exception as error:
        if timeout_seconds is not None:
            log.warning("Bounded utility completion failed (%s)", type(error).__name__)
            status = 422 if isinstance(error, NotImplementedError) else 502
            raise HTTPException(status_code=status, detail="Recall completion unavailable") from None
        if request.image_url:
            from ..media.prompt_expansion import PromptExpansionError
            if isinstance(error, PromptExpansionError):
                raise HTTPException(status_code=400, detail=str(error)) from error
            log.error("Image-aware utility completion failed: %s", error)
            raise HTTPException(
                status_code=422,
                detail="The configured maintenance model could not process the reference image; choose a vision-capable model in settings",
            ) from error
        log.error("Raw completion failed: %s", error)
        raise HTTPException(status_code=500, detail=str(error))
