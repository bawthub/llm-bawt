"""The service-bearer dependency for BawtHub-only routes (TASK-1472).

Routes BawtHub's server calls with the exported service token, never a bot:
the bridges don't mount the token, so a bot can't present it. Use as a router
dependency so the bearer is checked before anything else, including body
parsing (parse bodies in the handler, not as declared parameters).
"""

from __future__ import annotations

from typing import TypeVar

from fastapi import Header, HTTPException, Request
from pydantic import BaseModel, ValidationError

from llm_bawt.bawthub_auth import TokenCheck, service_token_source

NO_STORE = {"Cache-Control": "no-store"}

Body = TypeVar("Body", bound=BaseModel)


def require_service_token(authorization: str | None = Header(default=None)) -> None:
    result = service_token_source().check(authorization)
    if result is TokenCheck.OK:
        return
    if result is TokenCheck.UNCONFIGURED:
        raise HTTPException(503, "service_token_unconfigured", headers=NO_STORE)
    raise HTTPException(401, "unauthorized", headers={**NO_STORE, "WWW-Authenticate": "Bearer"})


async def read_body(request: Request, model: type[Body]) -> Body:
    """The request body as ``model``, parsed after the caller's auth check.

    Parsed in the handler, not as a declared body parameter: FastAPI decodes a
    declared body before router dependencies run, so a malformed body would
    answer 422 ahead of the bearer check.
    """
    try:
        return model.model_validate_json(await request.body())
    except ValidationError as error:
        detail = error.errors(include_url=False, include_input=False, include_context=False)
        raise HTTPException(422, detail, headers=NO_STORE) from None


__all__ = ["NO_STORE", "read_body", "require_service_token"]
