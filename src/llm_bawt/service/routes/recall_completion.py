"""Private-network, botless recall synthesis with bounded admission and transport.

A timed-out request retains its admission slot until its worker actually exits.
The SDK transport has a timeout and zero retries as well; a slow/trickling
upstream cannot create an unbounded queue of abandoned calls.
"""
from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ...recall.budgets import BoundedWorker, tokens
from ...recall.contracts import EMPTY_COMPLETION_DETAIL, RecallError
from ..schemas import RawCompletionRequest
from ..utility_completion import complete_utility

MAX_REQUEST_BYTES = 65_536
MAX_INPUT_TOKENS = 8_000
MAX_RESPONSE_BYTES = 16_384
COMPLETION_TIMEOUT_SECONDS = 20.0


class RecallCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    prompt: str = Field(min_length=1, max_length=32_000)
    system: str = Field(min_length=1, max_length=4_000)
    max_tokens: int = Field(default=1000, ge=1, le=1600)

    @field_validator("prompt", "system")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("blank input")
        return value


class RecallCompletionRoute(APIRoute):
    """Cap body bytes before parsing and don't echo source text in 422 details."""

    def get_route_handler(self):
        original = super().get_route_handler()

        async def bounded_handler(request: Request):
            body = bytearray()
            async for chunk in request.stream():
                if len(body) + len(chunk) > MAX_REQUEST_BYTES:
                    return JSONResponse({"detail": "recall_input_too_large"}, status_code=413)
                body.extend(chunk)
            # Fill Starlette's request-body cache for FastAPI's parsing handler.
            request._body = bytes(body)
            try:
                return await original(request)
            except RequestValidationError:
                return JSONResponse({"detail": "invalid_recall_request"}, status_code=422)

        return bounded_handler


router = APIRouter(route_class=RecallCompletionRoute)


class RecallCompletionRunner:
    """Bound running workers and reject excess work instead of queuing it."""

    def __init__(self, capacity: int = 2):
        self._worker = BoundedWorker(capacity, "recall-utility")

    def _complete(self, request: RecallCompletionRequest) -> dict[str, str]:
        response = complete_utility(
            RawCompletionRequest(prompt=request.prompt, system=request.system,
                                 max_tokens=request.max_tokens, temperature=0.0),
            timeout_seconds=COMPLETION_TIMEOUT_SECONDS,
        )
        packet = {"content": response.content}
        if (tokens(response.content) > request.max_tokens
                or len(json.dumps(packet, ensure_ascii=False).encode("utf-8")) > MAX_RESPONSE_BYTES):
            raise HTTPException(status_code=502, detail="recall_output_too_large")
        return packet

    async def run(self, request: RecallCompletionRequest) -> dict[str, str]:
        try:
            if tokens(request.system) + tokens(request.prompt) > MAX_INPUT_TOKENS:
                raise HTTPException(status_code=413, detail="recall_input_too_large")
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(status_code=503, detail="recall_tokenizer_unavailable") from None
        try:
            return await asyncio.wait_for(self._worker.run(self._complete, request),
                                          COMPLETION_TIMEOUT_SECONDS)
        except TimeoutError:
            raise HTTPException(status_code=504, detail="recall_completion_timeout") from None
        except RecallError as exc:
            status = 429 if exc.code == "busy" else 503
            raise HTTPException(status_code=status, detail="recall_completion_unavailable") from None
        except HTTPException as exc:
            # Only the caller's own input is the caller's fault (413). Any other
            # failure is the server's: a 4xx here would read as a bad request.
            if exc.status_code == 413:
                raise HTTPException(status_code=413, detail="recall_input_too_large") from None
            if exc.detail == EMPTY_COMPLETION_DETAIL:
                raise HTTPException(status_code=502, detail=EMPTY_COMPLETION_DETAIL) from None
            status = 503 if exc.status_code in {422, 503} else 502
            raise HTTPException(status_code=status, detail="recall_completion_unavailable") from None
        except Exception:
            raise HTTPException(status_code=502, detail="recall_completion_unavailable") from None


_runner = RecallCompletionRunner()


@router.post("/v1/llm/recall-complete", tags=["LLM"])
async def recall_completion(request: RecallCompletionRequest) -> dict[str, str]:
    """Use the DB maintenance model; no identity, history, tools or model override."""
    return await _runner.run(request)
