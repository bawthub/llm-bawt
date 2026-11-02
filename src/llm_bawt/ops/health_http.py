"""Internal FastAPI health host. Handlers read cached state, never external I/O."""
from __future__ import annotations

import re
import socket
from threading import Event, Thread

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException
import uvicorn
from uvicorn.protocols.http.h11_impl import H11Protocol

from .service_identity import PROJECT_PATTERN, SERVICE_PATTERN

SERVICE_ID = re.compile(rf"{PROJECT_PATTERN}\.{SERVICE_PATTERN}")


def create_health_app(state, *, lifespan=None):
    """Keep the existing exact, GET-only wire contract and private listener."""
    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None,
                  openapi_url=None, redirect_slashes=False)

    @app.middleware("http")
    async def fixed_surface(request: Request, call_next):
        raw = request.scope.get("raw_path", request.url.path.encode())
        if request.method != "GET":
            response = JSONResponse({"error": "method not allowed"}, status_code=405)
        elif request.scope.get("query_string") or b"%" in raw:
            response = JSONResponse({"error": "not found"}, status_code=404)
        else:
            response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["Connection"] = "close"
        return response

    @app.exception_handler(HTTPException)
    async def not_found(_request, exc):
        return JSONResponse({"error": "not found"}, status_code=exc.status_code)

    def reply(result):
        status, body = result
        return JSONResponse(body, status_code=status)

    @app.get("/health/live")
    async def live():
        return {"status": "ok"}

    @app.get("/health/ready")
    async def ready():
        return reply(state.ready())

    @app.get("/v1/health/containers")
    async def containers():
        return reply(state.containers())

    @app.get("/v1/health/services")
    async def services():
        return reply(state.services())

    @app.get("/v1/health/services/{service_id}")
    async def service(service_id: str):
        if not SERVICE_ID.fullmatch(service_id):
            return JSONResponse({"error": "not found"}, status_code=404)
        return reply(state.services(service_id))

    return app


class _HealthProtocol(H11Protocol):
    """Bound even clients that never produce an ASGI request.

    Uvicorn's keepalive timer starts only after a response, and its concurrency
    check runs only after complete headers. An absolute initial-header deadline
    prevents silent/trickling sockets from holding all health capacity forever.
    Every health response closes the connection, so there is no next request's
    header budget to manage here.
    """

    HEADER_TIMEOUT_SECONDS = 1.0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._headers_timer = None

    def connection_made(self, transport):
        super().connection_made(transport)
        self._headers_timer = self.loop.call_later(self.HEADER_TIMEOUT_SECONDS, self._expire_headers)

    def _cancel_headers_timer(self):
        if self._headers_timer is not None:
            self._headers_timer.cancel()
            self._headers_timer = None

    def _expire_headers(self):
        self._cancel_headers_timer()
        if self.scope is None and not self.transport.is_closing():
            self.transport.close()

    def data_received(self, data):
        super().data_received(data)
        if self.scope is not None:
            self._cancel_headers_timer()

    def connection_lost(self, exc):
        self._cancel_headers_timer()
        super().connection_lost(exc)


class _HealthServer(uvicorn.Server):
    def __init__(self, config, ready):
        super().__init__(config)
        self.ready = ready

    async def startup(self, sockets=None):
        try:
            await super().startup(sockets=sockets)
        finally:
            self.ready.set()


class OpsHealthServer:
    """Run in the main thread for signal handling; start/close support local tests.

    One async HTTP loop, bounded concurrency/backlog, no access-path logging or
    automatic proxy trust. The separate bounded control listener is unchanged.
    """

    def __init__(self, state, address=("0.0.0.0", 8650), *, lifespan=None):
        self.app = create_health_app(state, lifespan=lifespan)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind(address)
            self.socket.setblocking(False)
        except BaseException:
            self.socket.close()
            raise
        self.address = self.socket.getsockname()
        self._ready = Event()
        self.server = _HealthServer(uvicorn.Config(
            self.app, loop="asyncio", http=_HealthProtocol, ws="none", lifespan="on",
            access_log=False, log_config=None, server_header=False,
            proxy_headers=False, limit_concurrency=32, backlog=32,
            timeout_keep_alive=1, timeout_graceful_shutdown=2,
        ), self._ready)
        self.thread = Thread(target=self.run, name="ops-health-http", daemon=True)
        self._started = False

    def run(self):
        try:
            self.server.run(sockets=[self.socket])
        finally:
            self.socket.close()
            self._ready.set()

    def start(self):
        self.thread.start()
        self._started = True
        if not self._ready.wait(10) or not self.server.started:
            self.close()
            raise RuntimeError("ops health HTTP startup failed")

    def close(self):
        self.server.should_exit = True
        if self._started:
            # Two seconds each for requests, control, and fixed producer joins;
            # a blocked external RPC lives only in a daemon producer thread.
            self.thread.join(timeout=7)
        self.socket.close()
