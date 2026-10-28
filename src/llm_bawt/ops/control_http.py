"""The ops service's internal control listener (TASK-1169 S3a).

The app asks this process to prepare an invocation's snapshot, since only
this process may read Docker. It is a listener of its own: the :8650 health
surface stays a single GET-only thread, so a slow Docker read here never
delays a health read.

Bounded so a stuck Docker read can't pile up work:

* at most ``LLM_BAWT_OPS_CONTROL_CONCURRENCY`` (default 2) requests run at
  once; one more is answered at once with 503 ``ops_busy``;
* a body over 64 KiB is refused unread (413), and every socket read times out
  after 5s;
* the work runs on its own thread under a wall budget (default 20s). Past it
  the caller gets 504 ``ops_timeout``. Prepare is read-only, so the abandoned
  call has no side effects; its slot frees only when it really returns, which
  keeps the bound real.
"""
from __future__ import annotations

import json
import logging
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import BoundedSemaphore, Event, Lock, Thread

from .control_protocol import (CONTROL_PORT, EXECUTORS_PATH, MAX_BODY_BYTES, PREPARE_PATH, PROTOCOL_FIELD,
                               PROTOCOL_VERSION, control_token, refusal_status, token_matches)
from .service import OpsDispatchError

logger = logging.getLogger(__name__)

CONCURRENCY_ENV = "LLM_BAWT_OPS_CONTROL_CONCURRENCY"
DEFAULT_CONCURRENCY = 2
MAX_CONCURRENCY = 16
DEFAULT_BUDGET_SECONDS = 20.0
READ_TIMEOUT_SECONDS = 5.0
RETRY_AFTER_SECONDS = 2


def control_concurrency() -> int:
    """A missing, non-integer or out-of-range value is the default."""
    try:
        value = int(os.environ.get(CONCURRENCY_ENV, ""))
    except ValueError:
        return DEFAULT_CONCURRENCY
    return value if 1 <= value <= MAX_CONCURRENCY else DEFAULT_CONCURRENCY


class _ControlServer(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False  # close never waits on a request thread
    request_queue_size = 32

    def get_request(self):
        connection, address = super().get_request()
        connection.settimeout(READ_TIMEOUT_SECONDS)
        return connection, address


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
        control = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "OpsControl"
            sys_version = ""

            def log_message(self, *_args):
                pass  # never log request paths or bodies

            def reply(self, status, body, headers=()):
                data = json.dumps({PROTOCOL_FIELD: PROTOCOL_VERSION, **body},
                                  separators=(",", ":"), allow_nan=False).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("Connection", "close")
                for name, value in headers:
                    self.send_header(name, value)
                self.end_headers()
                self.wfile.write(data)
                self.close_connection = True

            def fail(self, status, code, message, headers=()):
                self.reply(status, {"code": code, "message": message}, headers)

            def authorized(self) -> bool:
                try:
                    token = control._token_source()
                except Exception:  # noqa: BLE001 - an unusable key refuses everyone
                    self.fail(503, "ops_control_unconfigured", "the ops control token is unavailable")
                    return False
                if not token_matches(self.headers.get("Authorization"), token):
                    self.fail(401, "unauthorized", "ops control requires its bearer token",
                              (("WWW-Authenticate", "Bearer"),))
                    return False
                return True

            def read_json(self):
                try:
                    length = int(self.headers.get("Content-Length", ""))
                except ValueError:
                    self.fail(411, "length_required", "a Content-Length is required")
                    return None
                if length < 0 or length > MAX_BODY_BYTES:
                    self.fail(413, "body_too_large", f"the body is limited to {MAX_BODY_BYTES} bytes")
                    return None
                try:
                    body = json.loads(self.rfile.read(length) or b"null")
                except ValueError:
                    self.fail(400, "bad_request", "the body is not JSON")
                    return None
                if not isinstance(body, dict):
                    self.fail(400, "bad_request", "the body is not a JSON object")
                    return None
                return body

            def do_POST(self):
                if self.path != PREPARE_PATH:
                    return self.fail(404, "not_found", "unknown control route")
                if not self.authorized():
                    return
                body = self.read_json()
                if body is None:
                    return
                operation, args = body.get("operation"), body.get("args")
                if not isinstance(operation, str) or not (args is None or isinstance(args, dict)):
                    return self.fail(400, "bad_request", "operation must be a string and args an object")
                control.run_bounded(self, lambda: {
                    "snapshot": control.service().prepare_invocation(operation, args)})

            def do_GET(self):
                if self.path != EXECUTORS_PATH:
                    return self.fail(404, "not_found", "unknown control route")
                if not self.authorized():
                    return
                control.run_bounded(self, lambda: {"executors": control.service().executor_readiness()})

            def unsupported(self):
                self.fail(405, "method_not_allowed", "method not allowed")

            do_PUT = do_PATCH = do_DELETE = do_HEAD = do_OPTIONS = unsupported

        self.server = _ControlServer(address, Handler)
        self.server.timeout = 0.1
        self._stop = Event()
        self.thread = Thread(target=self._serve, name="ops-control-http", daemon=True)
        self._started = False

    def service(self):
        with self._service_lock:
            if self._service is None:
                self._service = self._service_factory()
            return self._service

    def run_bounded(self, handler, work):
        if not self.slots.acquire(blocking=False):
            return handler.fail(503, "ops_busy", "the ops service is busy; retry shortly",
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
        if not self._budget_wait(done, self.budget_seconds):
            return handler.fail(504, "ops_timeout", f"the ops service did not answer within {self.budget_seconds:g}s")
        if "body" in outcome:
            return handler.reply(200, outcome["body"])
        if "refusal" in outcome:
            refusal = outcome["refusal"]
            return handler.reply(refusal_status(refusal.code),
                                 {"refusal": {"code": refusal.code, "message": str(refusal)}})
        return handler.fail(500, "ops_internal", "the ops service failed this request; see its log")

    def _serve(self):
        while not self._stop.is_set():
            self.server.handle_request()

    @property
    def address(self):
        return self.server.server_address

    def start(self):
        self.thread.start()
        self._started = True

    def close(self):
        self._stop.set()
        if self._started:
            self.thread.join(timeout=2)
        self.server.server_close()
