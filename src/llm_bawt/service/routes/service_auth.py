"""The service-bearer dependency for BawtHub-only routes (TASK-1472).

Routes BawtHub's server calls with the exported service token, never a bot:
the bridges don't mount the token, so a bot can't present it. Use as a router
dependency so the bearer is checked before anything else, including body
parsing (parse bodies in the handler, not as declared parameters).
"""

from __future__ import annotations

from fastapi import Header, HTTPException

from llm_bawt.bawthub_auth import TokenCheck, service_token_source

NO_STORE = {"Cache-Control": "no-store"}


def require_service_token(authorization: str | None = Header(default=None)) -> None:
    result = service_token_source().check(authorization)
    if result is TokenCheck.OK:
        return
    if result is TokenCheck.UNCONFIGURED:
        raise HTTPException(503, "service_token_unconfigured", headers=NO_STORE)
    raise HTTPException(401, "unauthorized", headers={**NO_STORE, "WWW-Authenticate": "Bearer"})


__all__ = ["NO_STORE", "require_service_token"]
