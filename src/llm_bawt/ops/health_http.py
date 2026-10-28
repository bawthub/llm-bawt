"""Fixed internal GET-only health surface. No external I/O in handlers."""
from __future__ import annotations

import json
import re
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Event, Thread

from .service_identity import PROJECT_PATTERN, SERVICE_PATTERN

# Exact path only: no query string, no encoding, the id grammar of S1b.
SERVICE_PATH = re.compile(rf"/v1/health/services/({PROJECT_PATTERN}\.{SERVICE_PATTERN})")


class _HealthServer(HTTPServer):
    # One HTTP worker, bounded socket reads/writes and response size. A request
    # never creates a thread or occupies a Docker/DB pool slot.
    request_queue_size = 32

    def get_request(self):
        connection, address = super().get_request()
        connection.settimeout(1.0)
        return connection, address


class OpsHealthServer:
    def __init__(self, state, address=("0.0.0.0", 8650)):
        class Handler(BaseHTTPRequestHandler):
            server_version = "OpsHealth"
            sys_version = ""

            def log_message(self, *_args):
                pass  # do not log arbitrary request paths/query contents

            def reply(self, status, body):
                data = json.dumps(body, separators=(",", ":"), allow_nan=False).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(data)
                self.close_connection = True

            def do_GET(self):
                if self.path == "/health/live":
                    self.reply(200, {"status": "ok"})
                elif self.path == "/health/ready":
                    self.reply(*state.ready())
                elif self.path == "/v1/health/containers":
                    self.reply(*state.containers())
                elif self.path == "/v1/health/services":
                    self.reply(*state.services())
                elif match := SERVICE_PATH.fullmatch(self.path):
                    self.reply(*state.services(match.group(1)))
                else:
                    self.reply(404, {"error": "not found"})

            def unsupported(self):
                self.reply(405, {"error": "method not allowed"})

            do_POST = do_PUT = do_PATCH = do_DELETE = do_HEAD = do_OPTIONS = unsupported

        self.server = _HealthServer(address, Handler)
        self.server.timeout = 0.1
        self._stop = Event()
        self.thread = Thread(target=self._serve, name="ops-health-http", daemon=True)
        self._started = False

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
            # HTTPServer.shutdown() itself waits without a deadline. Avoid it:
            # a trickling request must not hold process shutdown indefinitely.
            self.thread.join(timeout=2)
        self.server.server_close()
