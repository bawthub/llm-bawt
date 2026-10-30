"""Opt-in, fail-closed entrypoint for the isolated backend preview lane.

The production FastAPI app registers routes that can mutate durable state even
without a scheduler. Preview must instead opt in to each route deliberately;
this entrypoint does not import the production app or run its lifespan.
"""

from contextlib import asynccontextmanager
import os

from fastapi import FastAPI
from fastapi.responses import JSONResponse


@asynccontextmanager
async def _preview_lifespan(_app: FastAPI):
    if os.environ.get("LLM_BAWT_PREVIEW_ENABLED") != "1":
        raise RuntimeError("explicit preview role required")
    yield


def create_preview_app() -> FastAPI:
    """Expose only explicitly preview-safe routes, never the production router."""
    app = FastAPI(
        title="llm-bawt preview",
        lifespan=_preview_lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    @app.get("/health")
    async def health() -> JSONResponse:
        # A process that starts safely is not yet a working chat preview.
        return JSONResponse(
            status_code=503,
            content={"service": "preview", "state": "not_configured"},
        )

    return app


app = create_preview_app()
