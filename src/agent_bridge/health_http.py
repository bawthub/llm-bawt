"""Shared internal bridge HTTP surface (health, optional models, workspace)."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from .workspace import add_workspace_routes


def health_app(
    publisher, models: Callable[[], Awaitable[list[dict]]] | None = None,
    *, consumer_ready: Callable[[], bool] | None = None,
) -> FastAPI:
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    add_workspace_routes(app)

    @app.get("/")
    @app.get("/health")
    async def health():
        payload = {"redis": bool(publisher.connected)}
        if consumer_ready is not None:
            payload["command_consumer"] = bool(consumer_ready())
        return JSONResponse(payload, status_code=200 if all(payload.values()) else 503)

    if models is not None:
        @app.get("/models")
        async def model_list():
            try:
                return {"models": await models()}
            except Exception:
                return JSONResponse({"error": "Model discovery unavailable."}, status_code=503)

    return app


async def serve_health(publisher, port: int, models=None, *, consumer_ready=None) -> None:
    server = uvicorn.Server(uvicorn.Config(
        health_app(publisher, models, consumer_ready=consumer_ready), host="0.0.0.0", port=port,
        access_log=False, log_level="warning",
    ))
    # The owning bridge handles SIGINT/SIGTERM and cancels this task.
    import contextlib
    server.capture_signals = contextlib.nullcontext
    await server.serve()
