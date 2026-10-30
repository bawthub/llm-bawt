"""Stateless utility completions shared by HTTP routes and in-process callers."""
from __future__ import annotations

import time

from fastapi import HTTPException

from .dependencies import get_service
from .logging import get_service_logger
from .schemas import RawCompletionRequest, RawCompletionResponse

log = get_service_logger(__name__)


class BudgetUnavailable(Exception):
    """The catalog can't give a usable context window/output cap for the model."""


class EmptyCompletion(ValueError):
    """The model answered with no text, e.g. its reasoning spent the output cap."""


def validate_model_budget(request: RawCompletionRequest, config, model_ref: str) -> None:
    """Catalog-driven preflight with conservative provider-framing headroom.

    ``model_ref`` is the ref the client was resolved from, never the bare model
    key: one model on two access paths makes the bare key ambiguous (TASK-1795).
    o200k is an estimate for non-OpenAI models, not their exact tokenizer.
    Provider rejection is terminal, never a retry with a larger budget.

    Raises :class:`BudgetUnavailable` when the budget can't be read and
    HTTP 413 when the request doesn't fit it.
    """
    from ..recall.budgets import tokens

    try:
        window = config.get_model_context_window(model_ref)
        output_cap = config.get_model_max_tokens(model_ref)
    except Exception as error:
        raise BudgetUnavailable(type(error).__name__) from None
    if type(window) is not int or window <= 0 or type(output_cap) is not int or output_cap <= 0:
        raise BudgetUnavailable("missing model budget")
    try:
        needed = tokens(request.system or "") + tokens(request.prompt) + request.max_tokens + 256
    except Exception:
        raise HTTPException(status_code=503, detail="recall_tokenizer_unavailable") from None
    if needed > window or request.max_tokens > output_cap:
        raise HTTPException(status_code=413, detail="recall_input_too_large")


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
    from .maintenance_fallback import served_model

    service = get_service()
    requested_model = request.model or resolve_job_model(service.config, "maintenance_model")
    job = "recall_completion" if timeout_seconds is not None else "utility_completion"
    if not requested_model:
        from .maintenance_health import record_unconfigured
        record_unconfigured(job, "maintenance_model")
        raise HTTPException(status_code=503, detail="No maintenance model configured")
    client, model_alias = service._get_background_client(
        model_override=requested_model, job=job, allow_fallback=not request.model,
    )
    if client is None or model_alias is None:
        raise HTTPException(status_code=422, detail="Configured model has no direct API client")
    if timeout_seconds is not None:
        try:
            # The endpoint ref the client was built from (TASK-1795), not the
            # request string re-resolved under another harness.
            validate_model_budget(request, service.config, model_alias)
        except BudgetUnavailable as error:
            # A configuration fault, not a bad request: fail visibly (TASK-1795).
            from .maintenance_health import maintenance_health
            maintenance_health().record_failure(
                job, "budget_unavailable", f"no model budget for {model_alias}: {error}",
                model=model_alias, provider=None, credential=None)
            raise HTTPException(status_code=503, detail="recall_model_budget_unavailable") from None

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
        if isinstance(response, str) and not response.strip():
            raise EmptyCompletion("Model returned an empty completion")
        if not isinstance(response, str) or response.lstrip().startswith("ERROR:"):
            raise ValueError("Model returned no usable completion")
        return RawCompletionResponse(
            content=response, model=served_model(client, model_alias),
            tokens=len(response) // 4, elapsed_ms=(time.perf_counter() - start) * 1000,
        )
    except Exception as error:
        if timeout_seconds is not None:
            log.warning("Bounded utility completion failed (%s)", type(error).__name__)
            if isinstance(error, EmptyCompletion):
                from ..recall.contracts import EMPTY_COMPLETION_DETAIL
                raise HTTPException(status_code=502, detail=EMPTY_COMPLETION_DETAIL) from None
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
