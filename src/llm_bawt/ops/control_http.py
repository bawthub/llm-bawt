"""The ops service's internal control listener (TASK-1169 S3a; FastAPI host).

The app asks this process to prepare an invocation's snapshot, since only
this process may read Docker. It is a FastAPI app on a private listener of
its own (``:8651``, never published), with its own Uvicorn loop thread: the
:8650 health host never shares a loop or a slot with it, so a slow Docker read
here never delays a health read.

Bounded so a stuck Docker read can't pile up work:

* at most ``LLM_BAWT_OPS_CONTROL_CONCURRENCY`` (default 2) requests run at
  once; one more is answered at once with 503 ``ops_busy``;
* a body over 64 KiB is refused unread (413). A client gets 5s to send its
  headers (an absolute deadline, so trickling can't renew it) and 5s more for
  its body;
* the work runs on its own thread under a wall budget (default 20s). Past it
  the caller gets 504 ``ops_timeout``. Prepare is read-only, so the abandoned
  call has no side effects; its slot frees only when it really returns, which
  keeps the bound real.

The wire contract (routes, statuses, codes, protocol field, headers) is the
one ``control_client`` already speaks; this host changed none of it.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
from threading import BoundedSemaphore, Event, Lock, Thread

from fastapi import FastAPI, Request
from fastapi.responses import Response
import uvicorn

from .control_protocol import (CONTROL_PORT, EXECUTORS_PATH, MAX_BODY_BYTES, PREPARE_PATH, PROTOCOL_FIELD,
                               PROTOCOL_VERSION, control_token, refusal_status, token_matches)
from .health_http import _HealthProtocol
from .service import OpsDispatchError

logger = logging.getLogger(__name__)

CONCURRENCY_ENV = "LLM_BAWT_OPS_CONTROL_CONCURRENCY"
DEFAULT_CONCURRENCY = 2
MAX_CONCURRENCY = 16
DEFAULT_BUDGET_SECONDS = 20.0
READ_TIMEOUT_SECONDS = 5.0
RETRY_AFTER_SECONDS = 2
CONNECTION_LIMIT = 32
ROUTES = {PREPARE_PATH: "POST", EXECUTORS_PATH: "GET"}


def control_concurrency() -> int:
    """A missing, non-integer or out-of-range value is the default."""
    try:
        value = int(os.environ.get(CONCURRENCY_ENV, ""))
    except ValueError:
        return DEFAULT_CONCURRENCY
    return value if 1 <= value <= MAX_CONCURRENCY else DEFAULT_CONCURRENCY


def reply(status, body, headers=()):
    data = json.dumps({PROTOCOL_FIELD: PROTOCOL_VERSION, **body}, separators=(",", ":"), allow_nan=False).encode()
    response = Response(data, status_code=status, media_type="application/json")
    response.headers["Cache-Control"] = "no-store"
    response.headers["Connection"] = "close"
    for name, value in headers:
        response.headers[name] = value
    return response


def fail(status, code, message, headers=()):
    return reply(status, {"code": code, "message": message}, headers)


def create_control_app(control: "OpsControlServer") -> FastAPI:
    """The two typed control routes; everything else is a protocol 404/405."""
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None, redirect_slashes=False)

    @app.middleware("http")
    async def fixed_surface(request: Request, call_next):
        raw = request.scope.get("raw_path") or request.url.path.encode()
        method = ROUTES.get(raw.decode("latin-1")) if not request.scope.get("query_string") else None
        if request.method not in ("GET", "POST"):
            return fail(405, "method_not_allowed", "method not allowed")
        if method != request.method:
            return fail(404, "not_found", "unknown control route")
        token_refusal = control.refuse_unauthorized(request.headers.get("Authorization"))
        if token_refusal is not None:
            return token_refusal
        return await call_next(request)

    @app.post(PREPARE_PATH)
    async def prepare(request: Request):
        body = await read_json(request)
        if isinstance(body, Response):
            return body
        operation, args = body.get("operation"), body.get("args")
        if not isinstance(operation, str) or not (args is None or isinstance(args, dict)):
            return fail(400, "bad_request", "operation must be a string and args an object")
        return await control.run_bounded(lambda: {
            "snapshot": control.service().prepare_invocation(operation, args)})

    @app.get(EXECUTORS_PATH)
    async def executors():
        return await control.run_bounded(lambda: {"executors": control.service().executor_readiness()})

    return app


async def read_json(request: Request):
    """The JSON object body, or the refusal to send. Over-limit bodies stay unread.

    The contract is Content-Length framing only. h11 frames by
    Transfer-Encoding when both are sent, so any Transfer-Encoding is refused
    before the body is touched, and the streamed read is still capped and held
    to one absolute deadline."""
    if "transfer-encoding" in request.headers:
        return fail(411, "length_required", "a Content-Length body is required; Transfer-Encoding is not accepted")
    try:
        length = int(request.headers.get("Content-Length", ""))
    except ValueError:
        return fail(411, "length_required", "a Content-Length is required")
    if length < 0 or length > MAX_BODY_BYTES:
        return fail(413, "body_too_large", f"the body is limited to {MAX_BODY_BYTES} bytes")
    try:
        raw = await asyncio.wait_for(_read_capped(request, length), READ_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        return fail(408, "request_timeout", f"the body did not arrive within {READ_TIMEOUT_SECONDS:g}s")
    if raw is None:
        return fail(413, "body_too_large", f"the body is limited to {MAX_BODY_BYTES} bytes")
    try:
        body = json.loads(raw or b"null")
    except ValueError:
        return fail(400, "bad_request", "the body is not JSON")
    if not isinstance(body, dict):
        return fail(400, "bad_request", "the body is not a JSON object")
    return body


async def _read_capped(request: Request, length: int):
    """At most ``length`` bytes; None if the stream delivers more."""
    chunks, total = [], 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > length:
            return None
        chunks.append(chunk)
    return b"".join(chunks)


class _ControlProtocol(_HealthProtocol):
    """Absolute initial-header deadline, as on the health host, at the old 5s read bound."""

    HEADER_TIMEOUT_SECONDS = READ_TIMEOUT_SECONDS


class _ControlServer(uvicorn.Server):
    def __init__(self, config, ready):
        super().__init__(config)
        self.ready = ready

    async def startup(self, sockets=None):
        try:
            await super().startup(sockets=sockets)
        finally:
            self.ready.set()


class OpsControlServer:
    """``service_factory()`` builds the full OpsService on first use (and again
    after a failure), so listening never waits on the database or Docker."""

    def __init__(self, service_factory, address=("0.0.0.0", CONTROL_PORT), *,
                 concurrency: int | None = None, budget_seconds: float = DEFAULT_BUDGET_SECONDS,
                 token_source=control_token, budget_wait=None):
        """``budget_wait(done, seconds) -> bool`` waits for the work within the
        budget (default ``done.wait``); tests replace it to end the budget
        without a wall-clock race."""
        self._service_factory = service_factory
        self._service = None
        self._service_lock = Lock()
        self.slots = BoundedSemaphore(concurrency or control_concurrency())
        self.budget_seconds = budget_seconds
        self._budget_wait = budget_wait or (lambda done, seconds: done.wait(seconds))
        self._token_source = token_source
        self.app = create_control_app(self)
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
        self.server = _ControlServer(uvicorn.Config(
            self.app, loop="asyncio", http=_ControlProtocol, ws="none", lifespan="off",
            access_log=False, log_config=None, server_header=False, date_header=False,
            proxy_headers=False, limit_concurrency=CONNECTION_LIMIT, backlog=32,
            timeout_keep_alive=1, timeout_graceful_shutdown=2,
        ), self._ready)
        self.thread = Thread(target=self._serve, name="ops-control-http", daemon=True)
        self._started = False

    def service(self):
        with self._service_lock:
            if self._service is None:
                self._service = self._service_factory()
            return self._service

    def refuse_unauthorized(self, header):
        try:
            token = self._token_source()
        except Exception:  # noqa: BLE001 - an unusable key refuses everyone
            return fail(503, "ops_control_unconfigured", "the ops control token is unavailable")
        if not token_matches(header, token):
            return fail(401, "unauthorized", "ops control requires its bearer token",
                        (("WWW-Authenticate", "Bearer"),))
        return None

    async def run_bounded(self, work):
        if not self.slots.acquire(blocking=False):
            return fail(503, "ops_busy", "the ops service is busy; retry shortly",
                        (("Retry-After", str(RETRY_AFTER_SECONDS)),))
        done = Event()
        outcome = {}

        def run():
            try:
                outcome["body"] = work()
            except OpsDispatchError as exc:
                outcome["refusal"] = exc
            except Exception:  # noqa: BLE001 - logged here, reported as unavailable
                logger.exception("ops control request failed")
            finally:
                self.slots.release()
                done.set()

        Thread(target=run, name="ops-control-work", daemon=True).start()
        if not await asyncio.to_thread(self._budget_wait, done, self.budget_seconds):
            return fail(504, "ops_timeout", f"the ops service did not answer within {self.budget_seconds:g}s")
        if "body" in outcome:
            return reply(200, outcome["body"])
        if "refusal" in outcome:
            refusal = outcome["refusal"]
            return reply(refusal_status(refusal.code), {"refusal": {"code": refusal.code, "message": str(refusal)}})
        return fail(500, "ops_internal", "the ops service failed this request; see its log")

    def _serve(self):
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
            raise RuntimeError("ops control HTTP startup failed")

    def close(self):
        self.server.should_exit = True
        if self._started:
            self.thread.join(timeout=2)
        self.socket.close()
