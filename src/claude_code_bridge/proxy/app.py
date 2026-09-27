"""FastAPI app factory + uvicorn server lifecycle for the proxy.

The proxy runs in-process as a sibling task to the bridge's Redis consumer
and health TCP listener. Bound to ``127.0.0.1`` with an ephemeral port so
only the Claude Agent SDK subprocess spawned by the bridge can reach it;
the bridge captures the chosen port and injects it as ``ANTHROPIC_BASE_URL``.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import FastAPI

from .adapters import close_all, start_all
from .claude_oauth import ClaudeOAuthGateway, router as claude_oauth_router
from .routes import router as messages_router
from .request_context import ProxyStatusCallback

logger = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    logger.info("Proxy app starting")
    await start_all()
    app.state.claude_oauth_gateway = ClaudeOAuthGateway()
    try:
        yield
    finally:
        await app.state.claude_oauth_gateway.close()
        await close_all()
        logger.info("Proxy app shutting down")


def create_app(status_callback: ProxyStatusCallback | None = None) -> FastAPI:
    app = FastAPI(
        title="Claude Code Bridge — Anthropic-compatible proxy",
        description=(
            "Translates Anthropic Messages API requests from the Claude "
            "Agent SDK to upstream OpenAI-compatible providers via "
            "ProviderAdapters. Listens on 127.0.0.1 only."
        ),
        version="0.1.0",
        lifespan=_lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.proxy_status_callback = status_callback
    app.include_router(messages_router)
    app.include_router(claude_oauth_router)

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"ok": True}

    return app


class ProxyServer:
    """Owns a uvicorn ``Server`` running our FastAPI app on an ephemeral
    127.0.0.1 port. ``start()`` returns when the listening socket is bound
    so callers can read ``base_url`` immediately.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 0,
        *,
        status_callback: ProxyStatusCallback | None = None,
    ) -> None:
        self._host = host
        self._requested_port = port
        self._app = create_app(status_callback)
        self._server: Optional[uvicorn.Server] = None
        self._serve_task: Optional[asyncio.Task[None]] = None
        self._actual_port: Optional[int] = None

    @property
    def base_url(self) -> str:
        if self._actual_port is None:
            raise RuntimeError("ProxyServer not started yet")
        return f"http://{self._host}:{self._actual_port}"

    @property
    def port(self) -> int:
        if self._actual_port is None:
            raise RuntimeError("ProxyServer not started yet")
        return self._actual_port

    async def start(self) -> None:
        config = uvicorn.Config(
            app=self._app,
            host=self._host,
            port=self._requested_port,
            log_level="warning",  # quiet — bridge has its own logger
            access_log=False,
            loop="asyncio",
            lifespan="on",
        )
        self._server = uvicorn.Server(config)
        # Disable uvicorn's signal-handler installation — the bridge owns
        # process signals and uvicorn fights it for SIGTERM otherwise.
        self._server.install_signal_handlers = lambda: None  # type: ignore[assignment]

        self._serve_task = asyncio.create_task(self._server.serve())

        # Wait for the listening socket to bind so callers can capture the
        # ephemeral port. uvicorn flips ``started`` to True after bind.
        while not self._server.started:
            if self._serve_task.done():
                # Surface startup errors immediately instead of hanging.
                self._serve_task.result()
                raise RuntimeError("Proxy server exited before binding")
            await asyncio.sleep(0.01)

        # Read the actual bound port off the server's first socket.
        servers = getattr(self._server, "servers", []) or []
        for srv in servers:
            for sock in srv.sockets:
                self._actual_port = sock.getsockname()[1]
                break
            if self._actual_port is not None:
                break
        if self._actual_port is None:
            # Fallback: requested port was explicit
            self._actual_port = self._requested_port or 0

        logger.info("Proxy listening on %s", self.base_url)

    async def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._serve_task is not None:
            try:
                await asyncio.wait_for(self._serve_task, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._serve_task.cancel()
        logger.info("Proxy stopped")
